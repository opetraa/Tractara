# src/clara_ssot/parsing/metadata_extractor.py
"""
PDF 메타데이터 추출 파이프라인.

설계:
  1단계 — Front-Matter Isolation: 처음 4페이지 + 마지막 2페이지만 대상
  2단계 — 병렬 추출:
    Track A (결정론적): dc:title 스코어링, dc:identifier 정규식
    Track B (LLM/Gemini): dc:creator, dc:publisher, dc:date 등 의미론적 필드
  3단계 — 병합: Track A(title/identifier) 우선, 나머지 Track B
"""

import logging
import os
import re
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── 검증 상수 (스키마 enum과 동기화) ─────────────────────────────────────────
_VALID_ROLES = frozenset({"author", "contributor", "reviewer", "approver"})
_VALID_ORG_TYPES = frozenset({"utility", "vendor", "regulator", "research"})
_VALID_DOC_TYPES = frozenset(
    {
        "TechnicalReport",
        "RegulatoryDocument",
        "SafetyAnalysisReport",
        "PeriodicSafetyReview",
        "LicenseRenewalApplication",
        "Code",
        "Standard",
        "Procedure",
        "LicenseeEventReport",
        "RegulatoryAuditItem",
        "JournalArticle",
        "Patent",
        "Other",
    }
)
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# 문서 번호 패턴: 대문자로 시작 + 2자 이상 + 구분자(-/) + 영숫자 조합
_IDENTIFIER_RE = re.compile(r"^[A-Z][A-Z0-9]{1,}[-/][A-Z0-9][-A-Z0-9/.-]*$")


# ── Pydantic 모델 (Track B LLM 출력 구조) ─────────────────────────────────────
class CreatorItem(BaseModel):
    name: str
    role: str | None = Field(
        None, description="author | contributor | reviewer | approver 중 하나"
    )
    affiliation: str | None = None


class PublisherInfo(BaseModel):
    name: str
    organizationType: str | None = Field(
        None, description="utility | vendor | regulator | research 중 하나"
    )


class DateInfo(BaseModel):
    created: str | None = Field(None, description="작성 완료일 YYYY-MM-DD")
    issued: str | None = Field(None, description="공식 발행일 YYYY-MM-DD")
    modified: str | None = Field(None, description="최종 개정일 YYYY-MM-DD")
    valid: str | None = Field(None, description="유효 기간 시작일 YYYY-MM-DD")


class CoverageInfo(BaseModel):
    nuclearPlant: str | None = None
    system: str | None = None
    component: str | None = None
    material: str | None = None
    spatialCoverage: str | None = None
    temporalCoverage: str | None = None


class LLMMetadata(BaseModel):
    dc_title: str | None = Field(None, description="문서 공식 제목")
    dc_creator: list[CreatorItem] | None = Field(None, description="저자/관계자 목록")
    dc_publisher: PublisherInfo | None = Field(
        None, description="발행 기관 (단일 객체)")
    dc_date: DateInfo | None = Field(None, description="날짜 정보")
    dc_language: str | None = Field(
        None, description="ISO 639-1 언어 코드 (ko, en 등)")
    dc_type: str | None = Field(
        None,
        description=(
            "TechnicalReport | RegulatoryDocument | SafetyAnalysisReport | "
            "PeriodicSafetyReview | LicenseRenewalApplication | Code | Standard | "
            "Procedure | LicenseeEventReport | RegulatoryAuditItem | "
            "JournalArticle | Patent | Other 중 하나"
        ),
    )
    dc_subject: list[str] | None = Field(None, description="핵심 기술 키워드 3~7개")
    dc_coverage: CoverageInfo | None = Field(None, description="적용 범위")


# ── 내부 블록 표현 ────────────────────────────────────────────────────────────
@dataclass
class _FrontBlock:
    text: str
    font_size: float
    is_bold: bool
    bbox_x0: float
    bbox_y0: float
    bbox_x1: float
    bbox_y1: float
    page_width: float
    page_height: float
    page: int  # 1-based


# ── 결과 데이터 클래스 ────────────────────────────────────────────────────────
@dataclass
class ExtractedMetadata:
    dc_title: str | None = None
    dc_identifier: list[dict[str, str]] | None = None
    dc_creator: list[dict[str, Any]] | None = None
    dc_publisher: dict[str, Any] | None = None
    dc_date: dict[str, str] | None = None
    dc_language: str | None = None
    dc_type: str | None = None
    dc_subject: list[str] | None = None
    dc_coverage: dict[str, Any] | None = None


# ── Phase 1: Front-Matter Isolation ──────────────────────────────────────────
def _extract_frontmatter_blocks(
    pdf_path: Path,
) -> tuple[list[_FrontBlock], float]:
    """
    처음 4페이지 + 마지막 2페이지의 텍스트 블록을 수집한다.
    body_font_size는 전체 페이지 span의 폰트 크기 최빈값으로 추정한다.

    Returns: (front_matter_blocks, body_font_size)
    """
    doc = pymupdf.open(str(pdf_path))
    total_pages = len(doc)

    front_indices = set(range(min(4, total_pages)))
    back_indices = set(range(max(0, total_pages - 2), total_pages))
    target_indices = front_indices | back_indices

    blocks: list[_FrontBlock] = []
    font_sizes: list[float] = []

    for page_index, page in enumerate(doc):
        page_dict = page.get_text("dict")
        page_width = page.rect.width
        page_height = page.rect.height

        for b in page_dict.get("blocks", []):
            if b["type"] != 0:  # 0: text block
                continue

            text_parts: list[str] = []
            max_font_size = 0.0
            is_bold = False

            for line in b["lines"]:
                for span in line["spans"]:
                    raw = span["text"]
                    if not raw.strip():
                        continue
                    text_parts.append(raw)
                    size = round(span["size"], 1)
                    # body_font_size 추정용: 전체 페이지에서 수집
                    font_sizes.append(size)
                    if size > max_font_size:
                        max_font_size = size
                    if span["flags"] & 16:  # bit 4 = bold
                        is_bold = True

            clean_text = " ".join(text_parts).strip()
            if not clean_text:
                continue

            # front-matter 대상 페이지에서만 블록 수집
            if page_index in target_indices:
                x0, y0, x1, y1 = b["bbox"]
                blocks.append(
                    _FrontBlock(
                        text=clean_text,
                        font_size=max_font_size,
                        is_bold=is_bold,
                        bbox_x0=x0,
                        bbox_y0=y0,
                        bbox_x1=x1,
                        bbox_y1=y1,
                        page_width=page_width,
                        page_height=page_height,
                        page=page_index + 1,
                    )
                )

    doc.close()

    body_font_size = Counter(font_sizes).most_common(1)[
        0][0] if font_sizes else 10.0
    return blocks, body_font_size


def _detect_primary_language(text: str, threshold: int = 50) -> str:
    """
    텍스트를 한 번만 순회하며 주 언어를 추정한다. (Early Exit 적용)
    한글이 threshold 이상 발견되면 즉시 'ko'를 반환하여 리소스를 아낀다.
    """
    if not text:
        return "ko"

    h_count = 0
    e_count = 0
    
    # 너무 긴 텍스트일 경우를 대비해 앞부분 2000자만 샘플링
    for char in text[:2000]:
        cp = ord(char)
        if 0xAC00 <= cp <= 0xD7AF: # 한글 유니코드 범위
            h_count += 1
            if h_count >= threshold: # 조기 종료: 이 정도면 한국어 문서가 확실함
                return "ko"
        elif (65 <= cp <= 90) or (97 <= cp <= 122): # 영문 A-Z, a-z
            e_count += 1

    return "ko" if h_count > e_count else "en"


# ── Track A: 결정론적 규칙 엔진 ───────────────────────────────────────────────
def _score_title(blocks: list[_FrontBlock], body_font_size: float) -> str | None:
    """
    1페이지 블록에 다중 특징 스코어링을 적용해 제목 후보를 선택한다.

    점수 기준:
      +3: 폰트 크기 ≥ body_font_size × 1.4
      +2: Bold
      +2: 중앙 정렬 (좌우 여백 각 15% 이상)
      +1: 페이지 상단 50% 이내
      +1: 텍스트 길이 10~200자
    """
    page1_blocks = [b for b in blocks if b.page == 1]
    if not page1_blocks:
        return None

    best_text: str | None = None
    best_score = -1.0

    for b in page1_blocks:
        score = 0.0

        if body_font_size > 0 and b.font_size >= body_font_size * 1.4:
            score += 3
        if b.is_bold:
            score += 2
        if b.page_width > 0:
            left_margin = b.bbox_x0 / b.page_width
            right_margin = (b.page_width - b.bbox_x1) / b.page_width
            if left_margin > 0.15 and right_margin > 0.15:
                score += 2
        if b.page_height > 0 and b.bbox_y0 < b.page_height * 0.5:
            score += 1
        if 10 <= len(b.text) <= 200:
            score += 1

        if score > best_score:
            best_score = score
            best_text = b.text

    return best_text if best_score > 0 else None


def _extract_identifier(
    blocks: list[_FrontBlock],
) -> list[dict[str, str]] | None:
    """
    헤더/푸터 여백(상단 10%, 하단 10%)에 고립된 문서 번호 패턴을 추출한다.

    여백 판정:
      - 헤더: bbox_y0 < page_height × 0.10
      - 푸터: bbox_y1 > page_height × 0.90
    """
    results: list[dict[str, str]] = []
    seen: set[str] = set()

    for b in blocks:
        text = b.text.strip()
        if not _IDENTIFIER_RE.match(text) or text in seen:
            continue

        in_header = b.page_height > 0 and b.bbox_y0 < b.page_height * 0.10
        in_footer = b.page_height > 0 and b.bbox_y1 > b.page_height * 0.90

        if in_header or in_footer:
            results.append({"scheme": "DOCKET", "value": text})
            seen.add(text)

    return results or None


def _run_track_a(
    blocks: list[_FrontBlock],
    body_font_size: float,
) -> tuple[str | None, list[dict[str, str]] | None]:
    """Track A 진입점. (title, identifier_list) 반환."""
    title = _score_title(blocks, body_font_size)
    identifier = _extract_identifier(blocks)
    return title, identifier


# ── Track B: LLM 기반 의미론적 추출 ──────────────────────────────────────────
def _build_llm_prompt(frontmatter_text: str) -> str:
    return f"""당신은 기술 규제 문서의 메타데이터를 추출하는 전문가입니다.
아래 문서의 표지/서문 텍스트에서 메타데이터를 추출하여 정해진 JSON 구조로 반환하세요.

[추출 규칙]
- dc_title: 문서의 공식 제목
- dc_creator: 저자/관계자 목록. role은 반드시 아래 중 하나
    author (Prepared by / 작성) | contributor | reviewer (Reviewed by / 검토) | approver (Approved by / 승인)
- dc_publisher: 발행 기관 (단일 객체). organizationType은 반드시 아래 중 하나
    utility (원전 운영자) | vendor (설계/엔지니어링사) | regulator (규제기관) | research (연구기관)
- dc_language: 문서의 주된 서술 언어(Primary Language)를 ISO 639-1 코드로 선택 (한국어: ko, 영어: en). 
    기술 용어나 한자가 섞여 있어도 문장 구조를 이루는 주 언어를 선택하세요.
- dc_date: 날짜는 YYYY-MM-DD 형식. 용도별 매핑:
    작성 완료일/Draft → created | 공식 발행/Issue → issued | 개정/Revision → modified
- dc_language: ISO 639-1 (한국어: ko, 영어: en)
- dc_type: 반드시 아래 중 하나
    TechnicalReport | RegulatoryDocument | SafetyAnalysisReport | PeriodicSafetyReview |
    LicenseRenewalApplication | Code | Standard | Procedure | LicenseeEventReport |
    RegulatoryAuditItem | JournalArticle | Patent | Other
- dc_subject: 핵심 기술 키워드 3~7개
- dc_coverage: 해당 항목만 (nuclearPlant, system, component, material, spatialCoverage, temporalCoverage)

[중요] 확인할 수 없는 필드는 null로 반환하세요. 추측하지 마세요.

[문서 표지/서문 텍스트]
{frontmatter_text[:3000]}"""


def _run_track_b(blocks: list[_FrontBlock], api_key: str) -> LLMMetadata | None:
    """
    Track B 진입점. Gemini + instructor로 의미론적 메타데이터를 추출한다.
    실패 시 None 반환 (Track A 결과만으로 graceful fallback).
    """
    try:
        import instructor
        import google.generativeai as genai
    except ImportError:
        logger.warning("instructor 또는 google-generativeai 미설치. Track B 건너뜀.")
        return None

    frontmatter_text = "\n".join(b.text for b in blocks)
    if len(frontmatter_text.strip()) < 20:
        logger.warning("Front-matter 텍스트 부족. Track B 건너뜀.")
        return None

    model_name = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
    genai.configure(api_key=api_key)
    client = instructor.from_gemini(
        client=genai.GenerativeModel(model_name=model_name),
        mode=instructor.Mode.GEMINI_JSON,
    )

    try:
        result: LLMMetadata = client.chat.completions.create(
            messages=[
                {"role": "user", "content": _build_llm_prompt(
                    frontmatter_text)}
            ],
            response_model=LLMMetadata,
            max_retries=2,
        )
        logger.info(
            f"Track B 추출 성공: title={result.dc_title!r}, type={result.dc_type!r}")
        return result
    except Exception as e:
        logger.error(f"Track B LLM 호출 실패: {e}")
        return None


# ── Phase 3: 결과 병합 ────────────────────────────────────────────────────────
def _merge_results(
    track_a_title: str | None,
    track_a_identifier: list[dict[str, str]] | None,
    fallback_lang: str,
    track_b: LLMMetadata | None,
) -> ExtractedMetadata:
    """
    Track A + Track B 결과를 병합한다.
    - title/identifier: Track A 우선
    - creator/publisher/date/language/type/subject/coverage: Track B 전담
    - enum 범위 이탈 값, 날짜 형식 오류는 무효 처리
    """
    result = ExtractedMetadata()

    # title: Track A 우선, 없으면 Track B
    result.dc_title = track_a_title or (track_b.dc_title if track_b else None)

    # identifier: Track A 전담
    result.dc_identifier = track_a_identifier

    # language: Track B 우선, 실패 시 Track A(Heuristic)
    result.dc_language = (track_b.dc_language.lower() if track_b and track_b.dc_language and len(track_b.dc_language) == 2 else fallback_lang)

    if track_b is None:
        return result

    # creator
    if track_b.dc_creator:
        creators: list[dict[str, Any]] = []
        for c in track_b.dc_creator:
            item: dict[str, Any] = {"name": c.name}
            if c.role and c.role in _VALID_ROLES:
                item["role"] = c.role
            if c.affiliation:
                item["affiliation"] = c.affiliation
            creators.append(item)
        if creators:
            result.dc_creator = creators

    # publisher
    if track_b.dc_publisher and track_b.dc_publisher.name:
        pub: dict[str, Any] = {"name": track_b.dc_publisher.name}
        if (
            track_b.dc_publisher.organizationType
            and track_b.dc_publisher.organizationType in _VALID_ORG_TYPES
        ):
            pub["organizationType"] = track_b.dc_publisher.organizationType
        result.dc_publisher = pub

    # date (각 필드 형식 검증 후 유효한 것만 포함)
    if track_b.dc_date:
        date_obj: dict[str, str] = {}
        for date_field in ("created", "issued", "modified", "valid"):
            val = getattr(track_b.dc_date, date_field, None)
            if val and _DATE_RE.match(val):
                date_obj[date_field] = val
        if date_obj:
            result.dc_date = date_obj

    # type
    if track_b.dc_type and track_b.dc_type in _VALID_DOC_TYPES:
        result.dc_type = track_b.dc_type

    # subject
    if track_b.dc_subject:
        subjects = [s for s in track_b.dc_subject if isinstance(
            s, str) and s.strip()]
        if subjects:
            result.dc_subject = subjects

    # coverage
    if track_b.dc_coverage:
        cov: dict[str, Any] = {}
        for attr in (
            "nuclearPlant",
            "system",
            "component",
            "material",
            "spatialCoverage",
            "temporalCoverage",
        ):
            val = getattr(track_b.dc_coverage, attr, None)
            if val:
                cov[attr] = val
        if cov:
            result.dc_coverage = cov

    return result


# ── 공개 인터페이스 ───────────────────────────────────────────────────────────
def extract_metadata(
    pdf_path: Path,
    api_key: str | None = None,
) -> ExtractedMetadata:
    """
    PDF 메타데이터 추출 파이프라인.

    1단계: Front-Matter Isolation (처음 4페이지 + 마지막 2페이지)
    2단계: Track A (결정론적) + Track B (LLM) 병렬 실행
    3단계: 결과 병합 (Track A title/identifier 우선)
    """
    resolved_key = api_key or os.getenv(
        "GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    blocks, body_font_size = _extract_frontmatter_blocks(pdf_path)
    logger.info(
        f"Front-matter blocks: {len(blocks)}, body_font_size: {body_font_size}pt"
    )

    combined_text = " ".join(b.text for b in blocks[:10])
    fallback_lang = _detect_primary_language(combined_text)

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a: Future[tuple[str | None, list[dict[str, str]] | None]] = (
            executor.submit(_run_track_a, blocks, body_font_size)
        )
        future_b: Future[LLMMetadata | None] | None = (
            executor.submit(_run_track_b, blocks, resolved_key)
            if resolved_key
            else None
        )

        if not resolved_key:
            logger.warning("API 키 없음. Track B 건너뜀, Track A 결과만 사용.")

        a_title, a_identifier = future_a.result()
        b_result = future_b.result() if future_b is not None else None

    logger.info(f"Track A — title: {a_title!r}, identifier: {a_identifier}")
    logger.info(f"Track B — {'성공' if b_result else '없음/실패'}")

    return _merge_results(a_title, a_identifier, fallback_lang, b_result)
