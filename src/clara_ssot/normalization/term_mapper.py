# src/clara_ssot/normalization/term_mapper.py
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
import logging
import os

import instructor
import google.generativeai as genai
from pydantic import BaseModel, Field

from ..models.term_types import TermType
from ..parsing.pdf_parser import ParsedDocument
from ..tracing import get_trace_id

logger = logging.getLogger(__name__)


# Pydantic 모델로 TERM 구조 정의
class ExtractedTerm(BaseModel):
    """LLM이 추출할 TERM 구조"""
    term: str = Field(description="용어 (약어 또는 전체 명칭)")
    headword_en: str = Field(description="영문 정식 명칭")
    headword_ko: str = Field(description="한글 정식 명칭")
    definition_en: str = Field(description="영문 정의")
    definition_ko: str = Field(description="한글 정의")
    domain: List[str] = Field(description="도메인 태그", default=["nuclear"])
    context: str = Field(description="원문 맥락 (출처 문장)")
    term_type: TermType = Field(
        description="TERM 타입 (현재 단계: 일반 문서 추출은 모두 TERM-CLASS로 고정)",
        default=TermType.CLASS,
    )


class TermExtractionResult(BaseModel):
    """LLM 응답 전체 구조"""
    terms: List[ExtractedTerm] = Field(description="추출된 용어 목록")
    reasoning: str = Field(description="Chain of Thought 추론 과정")


@dataclass
class TermCandidate:
    term: str
    definition_en: str | None = None
    definition_ko: str | None = None
    headword_en: str | None = None
    headword_ko: str | None = None
    domain: List[str] | None = None
    context: str | None = None
    term_type: TermType = field(default=TermType.CLASS)


class LLMTermExtractor:
    """
    LLM 기반 TERM 추출기 (첫 번째 문서 전략: CoT + Pydantic)
    """

    def __init__(self, api_key: str):
        # Gemini 설정 (구형 SDK 사용 - 안정성 확보)
        # TODO: [Migration] instructor 라이브러리가 google-genai(신형 SDK)를 완벽히 지원하면 마이그레이션 필요.
        # 현재(2026.02) instructor 1.14.x 버전은 구형 SDK(google-generativeai)와 호환성이 더 좋음.
        # 참조: https://github.com/google-gemini/deprecated-generative-ai-python
        genai.configure(api_key=api_key)

        # 사용 가능한 모델 후보군 설정 (Fallback을 위해 리스트로 관리)
        self.model_candidates = self._get_model_candidates()
        self.current_model_idx = 0
        self.model_name = self.model_candidates[0]

        logger.info(f"🤖 Initializing Gemini with model: {self.model_name}")
        self._init_client()

    def _init_client(self):
        # Instructor 클라이언트 래핑 (표준화된 인터페이스 제공)
        # 구형 SDK의 GenerativeModel 객체를 생성하여 전달
        self.client = instructor.from_gemini(
            client=genai.GenerativeModel(model_name=self.model_name),
            mode=instructor.Mode.GEMINI_JSON,
        )

    def _get_model_candidates(self) -> List[str]:
        """API 키로 접근 가능한 모델 중 최적의 모델 후보 리스트 반환"""
        candidates = []
        target_model = os.getenv("GEMINI_MODEL")

        # 1. 환경변수 모델 최우선
        if target_model:
            candidates.append(target_model)

        # 2. 선호하는 모델 순서 (성능/비용/쿼터 고려)
        # 429 에러 발생 시 순차적으로 다음 모델을 시도함
        preferences = [
            "gemini-3-flash-preview",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b",
            "gemini-1.5-pro",
        ]

        try:
            # 3. 실제 사용 가능한 모델 목록 조회
            available_models = [m.name.replace(
                "models/", "") for m in genai.list_models()]
            logger.info(f"📋 Available Gemini models: {available_models}")

            for pref in preferences:
                if pref in available_models and pref not in candidates:
                    candidates.append(pref)

            # 4. 선호 모델이 없으면 목록의 첫 번째 모델 사용
            if not candidates and available_models:
                candidates.append(available_models[0])

        except Exception as e:
            logger.error(f"⚠️ Failed to list models: {e}")
            # API 호출 실패 시 기본 리스트 사용
            for pref in preferences:
                if pref not in candidates:
                    candidates.append(pref)

        if not candidates:
            candidates = ["gemini-1.5-flash"]

        return candidates

    def extract(self, text_chunks: List[str]) -> Tuple[List[TermCandidate], List[str]]:
        """
        여러 텍스트 청크에서 TERM 추출
        Returns: (candidates, error_messages)
        """
        all_candidates = []
        errors = []

        for i, chunk in enumerate(text_chunks):
            if len(chunk.strip()) < 20:  # 너무 짧은 텍스트는 스킵 (기준 완화: 50 -> 20)
                continue

            logger.info(
                f"Sending chunk {i+1}/{len(text_chunks)} to LLM (len={len(chunk)})...")
            try:
                result = self._extract_from_chunk(chunk)
                candidates = [
                    TermCandidate(
                        term=t.term,
                        definition_en=t.definition_en,
                        definition_ko=t.definition_ko,
                        headword_en=t.headword_en,
                        headword_ko=t.headword_ko,
                        domain=t.domain,
                        context=t.context,
                        term_type=t.term_type,
                    )
                    for t in result.terms
                ]
                all_candidates.extend(candidates)

                logger.info(f"Extracted {len(candidates)} terms from chunk")
                logger.debug(f"CoT reasoning: {result.reasoning}")

            except Exception as e:
                msg = f"Chunk {i+1} failed: {str(e)}"
                logger.error(f"❌ TERM extraction failed: {msg}", exc_info=True)

                # 🚨 API 키 만료 또는 권한 에러 발생 시 즉시 중단
                if "expired" in str(e).lower() or "400" in str(e) or "403" in str(e):
                    logger.critical(
                        "🛑 Critical API Error: API Key expired or invalid. Stopping.")
                    errors.append(
                        "Critical: API Key issue. Please check your .env file.")
                    break

                # 404 모델 에러인 경우 사용 가능한 모델 목록 출력 (디버깅용)
                if "404" in str(e) and "models/" in str(e):
                    try:
                        available_models = [
                            m.name for m in genai.list_models()]
                        logger.error(f"Available models: {available_models}")
                    except Exception as list_err:
                        logger.error(f"Failed to list models: {list_err}")

                errors.append(msg)

        return all_candidates, errors

    def _extract_from_chunk(self, text: str) -> TermExtractionResult:
        """
        단일 청크에서 TERM 추출 (Instructor + CoT)
        """
        # Few-shot 예제
        few_shot_example = """
예시 1:
텍스트: "경년열화 관리 프로그램(AMP)은 원자력 발전소의 장기 운전을 위해 필수적이다. 또한 1차 계통의 응력부식균열(SCC) 및 피로(Fatigue) 손상을 감시해야 한다."

추출:
[
  {
    "term": "AMP",
    "headword_en": "Aging Management Program",
    "headword_ko": "경년열화 관리 프로그램",
    "definition_en": "A program to manage aging effects in nuclear power plants.",
    "definition_ko": "원자력 발전소의 경년 열화 영향을 관리하는 프로그램.",
    "domain": ["nuclear", "LTO", "safety"],
    "context": "경년열화 관리 프로그램(AMP)은..."
  },
  {
    "term": "SCC",
    "headword_en": "Stress Corrosion Cracking",
    "headword_ko": "응력부식균열",
    "definition_en": "Cracking induced from the combined influence of tensile stress and a corrosive environment.",
    "definition_ko": "인장 응력과 부식성 환경의 복합적인 영향으로 발생하는 균열.",
    "domain": ["nuclear", "materials"],
    "context": "또한 1차 계통의 응력부식균열(SCC) 및..."
  },
  {
    "term": "Fatigue",
    "headword_en": "Fatigue",
    "headword_ko": "피로",
    "definition_en": "Weakening of a material caused by cyclic loading.",
    "definition_ko": "반복적인 하중으로 인해 재료가 약해지는 현상.",
    "domain": ["nuclear", "materials", "mechanics"],
    "context": "...및 피로(Fatigue) 손상을 감시해야 한다."
  }
]
"""

        prompt = f"""당신은 기술 문서(원자력, 엔지니어링, IT, 환경 등)에서 전문 용어를 추출하는 전문가입니다.
문서에 등장하는 '모든' 주요 기술 용어, 약어, 시스템 명칭, 고유한 개념을 추출하세요.
단순히 특정 단어만 찾는 것이 아니라, 문서 내의 다양한 전문 용어를 포괄적으로 찾아야 합니다.

다음 단계를 따르세요:
1. 텍스트를 정독하고 약어(예: LOCA, RPV), 한글 전문 용어(예: 냉각재상실사고), 영문 전문 용어(예: Fatigue Monitoring)를 모두 식별하세요.
2. 각 용어의 정식 명칭(Full Name)을 영문과 한글로 최대한 복원하세요.
3. 문맥(Context)을 바탕으로 해당 용어의 정의를 요약하세요.
4. 적절한 도메인 태그를 할당하세요 (nuclear, safety, LTO, PSR, FSAR 등).

{few_shot_example}

이제 다음 텍스트에서 용어를 추출하세요:

{text}

추론 과정을 reasoning 필드에 자세히 기록하고, terms 배열에 추출 결과를 담으세요.
"""

        # 모델 Fallback 루프
        while True:
            try:
                # Instructor를 통한 구조화된 출력 요청
                response = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    response_model=TermExtractionResult,
                    max_retries=2,  # 내부 재시도 (일시적 오류용)
                )
                return response
            except Exception as e:
                # 429 Quota Exceeded 에러 처리
                if "429" in str(e) or "Quota exceeded" in str(e) or "ResourceExhausted" in str(e):
                    logger.warning(
                        f"⚠️ Quota exceeded for model {self.model_name}.")

                    # 다음 모델로 전환
                    self.current_model_idx += 1
                    if self.current_model_idx < len(self.model_candidates):
                        self.model_name = self.model_candidates[self.current_model_idx]
                        logger.info(
                            f"🔄 Switching to fallback model: {self.model_name}")
                        self._init_client()
                        continue
                    else:
                        logger.error("❌ All fallback models exhausted.")
                        raise e
                raise e


def extract_term_candidates(
    parsed: ParsedDocument,
    llm_api_key: str = None
) -> Tuple[List[TermCandidate], List[str]]:
    """
    ParsedDocument에서 TERM 후보 추출

    첫 번째 문서 전략:
    - LLM 기반 추출 (CoT + Few-shot)
    - Instructor로 구조화된 출력 보장
    """
    # 인자로 키가 안 넘어왔으면 환경 변수에서 조회
    if not llm_api_key:
        llm_api_key = os.environ.get(
            "GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    if not llm_api_key:
        logger.warning("No LLM API key provided, using dummy extractor")
        return [
            TermCandidate(
                term="AMP",
                definition_en=None,
                definition_ko="경년열화 관리 프로그램",
            )
        ], ["No LLM API Key provided."]

    # 텍스트 청크 준비
    text_chunks = [
        block.text for block in parsed.blocks
        if block.text and len(block.text) > 20
    ]

    if not text_chunks:
        logger.warning(
            "⚠️ No text chunks > 20 chars found in document. Term extraction skipped.")
        return [], ["No text chunks found in document (OCR might be needed)."]

    # LLM 추출
    extractor = LLMTermExtractor(api_key=llm_api_key)
    # 더 많은 용어를 찾기 위해 청크 수 증가 (테스트용 1개)
    chunks_to_process = text_chunks[:1]
    logger.info(f"Sending {len(chunks_to_process)} text chunks to LLM...")
    candidates, errors = extractor.extract(chunks_to_process)

    logger.info(
        f"Extracted {len(candidates)} TERM candidates. Errors: {len(errors)}")
    return candidates, errors


def _normalize_term_id(headword_en: str | None, fallback: str, term_type: TermType) -> str:
    """
    강타입 URN 형식의 termId를 생성한다.

    우선순위: headword_en > fallback(term 필드)
    정규화 규칙:
    - 영문 소문자로 변환
    - 공백/특수문자 → 언더스코어
    - ASCII 영숫자 + 언더스코어만 허용 (한글 등 비ASCII 제거)
    - 빈 문자열이 되면 "unknown"으로 대체
    """
    prefix_map = {
        TermType.CLASS: "term:class:",
        TermType.REL: "term:rel:",
        TermType.RULE: "term:rule:",
    }
    prefix = prefix_map[term_type]

    source = (headword_en or "").strip() or fallback.strip()

    # ASCII 영숫자와 공백만 남기고 제거
    ascii_only = re.sub(r"[^a-zA-Z0-9 ]", " ", source)
    # 소문자 변환 + 공백 연속 → 단일 언더스코어
    normalized = re.sub(r"\s+", "_", ascii_only.lower().strip())
    # 선행/후행 언더스코어 제거
    normalized = normalized.strip("_")

    if not normalized:
        normalized = "unknown"

    return prefix + normalized


def build_term_baseline_candidates(
    doc_baseline_id: str,
    candidates: List[TermCandidate],
) -> List[Dict[str, Any]]:
    """
    TermCandidate → TERM Baseline JSON 변환.

    termId는 headword_en 기반 강타입 URN으로 생성한다:
    - TERM-CLASS: term:class:{normalized_headword_en}
    - TERM-REL:   term:rel:{normalized_headword_en}
    - TERM-RULE:  term:rule:{normalized_headword_en}
    """
    term_jsons: List[Dict[str, Any]] = []

    for c in candidates:
        term_id = _normalize_term_id(c.headword_en, c.term, c.term_type)
        term_json = {
            "type": "term_entry",
            "termId": term_id,
            "termType": c.term_type.value,
            "term": c.term,
            "lang": "bilingual",
            "headword_en": c.headword_en or c.term,
            "headword_ko": c.headword_ko or "",
            "definition_en": c.definition_en or "[PENDING_DEFINITION]",
            "definition_ko": c.definition_ko or "",
            "slots": {
                "context_ko": c.context or "",
            },
            "examples": [],
            "negatives": [],
            "domain": c.domain or ["nuclear"],
            "relatedTerms": [],
            "relations": [],
            "taxonomyBindings": [],
            "provenance": {
                "sources": [{"docId": doc_baseline_id}],
                "curationStatus": "candidate",
                "trace_id": get_trace_id(),
            },
            "status": "candidate",
        }
        term_jsons.append(term_json)

    return term_jsons
