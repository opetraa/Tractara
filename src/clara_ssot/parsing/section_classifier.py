# src/clara_ssot/parsing/section_classifier.py
# 튜닝 가능한 파라미터 (section_classifier.py):
# 임계치: score >= 70 라인
# 각 힌트 가중치: +50, +40, +30, +20, +10 숫자들
# 패널티: -50 (문장부호), -30 (긴 텍스트)
"""
섹션 분류기 (Section Classifier)

PDF 블록 하나를 받아 S/A/B급 힌트를 조합한 가중치 스코어링으로
section / subsection / paragraph 등을 판별하고 sectionLabel / sectionTitle을 추출한다.

분류 로직:
  [단계 1] S급 Fast-Track: PDF 북마크 텍스트와 100% 일치 → 즉시 확정
  [단계 2] S급 Fast-Track: ToC 엔트리와 100% 일치       → 즉시 확정
  [단계 3] A/B급 스코어링: 가중치 합산 후 임계치(70점) 판정
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# 섹션 번호 패턴 (우선순위 순)
# ---------------------------------------------------------------------------
_SECTION_LABEL_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("숫자형",   re.compile(r'^(\d+(?:\.\d+)*)\.?\s+(.+)', re.DOTALL)),
    ("한국어형", re.compile(r'^(제\d+[장절항목])\s+(.+)', re.DOTALL)),
    ("알파숫자", re.compile(r'^([A-Z]\.\d*)\s+(.+)', re.DOTALL)),
    ("영문부록", re.compile(
        r'^(Appendix\s+[A-Z])\s+(.+)', re.DOTALL | re.IGNORECASE)),
    ("한글목록", re.compile(r'^([가-힣]\.\s*)(.+)', re.DOTALL)),
]


def extract_section_label(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    텍스트에서 섹션 번호(label)와 제목(title)을 분리한다.

    반환: (section_label, section_title)
      예: "1.2.3 배경 설명" → ("1.2.3", "배경 설명")
      예: "Abstract"        → (None, None)
    """
    for _name, pattern in _SECTION_LABEL_PATTERNS:
        m = pattern.match(text)
        if m:
            return m.group(1).strip(), m.group(2).strip()
    return None, None


def _normalize(text: str) -> str:
    """비교용 텍스트 정규화: 소문자 + 공백 압축."""
    return " ".join(text.lower().split())


def _depth_to_type(depth: int) -> str:
    if depth == 0:
        return "title"
    elif depth == 1:
        return "section"
    else:
        return "subsection"


# ---------------------------------------------------------------------------
# 데이터 클래스
# ---------------------------------------------------------------------------

@dataclass
class SectionFeatures:
    """분류기 입력: PDF 블록 하나의 시각적/텍스트적 특징."""
    text: str
    max_font_size: float
    is_bold: bool
    font_name: str        # 블록 내 지배적 폰트명 (디버깅·미래 확장용)
    page_width: float     # 페이지 너비 (중앙 정렬 판단)
    bbox_x0: float
    bbox_x1: float


@dataclass
class ClassificationResult:
    """분류기 출력: 블록 한 개에 대한 분류 결과."""
    level: int                    # 0=제목, 1=section, 2+=subsection, 999=paragraph
    block_type: str               # "title"|"section"|"subsection"|"paragraph"
    confidence: float             # 0.0 ~ 1.0
    section_label: Optional[str]  # "1.2.3", "제2장" 등
    section_title: Optional[str]  # 번호 이후 제목 텍스트
    decision_path: str            # 어떤 규칙 경로로 결정됐는지 (디버깅용)


# ---------------------------------------------------------------------------
# 분류기 본체
# ---------------------------------------------------------------------------

class SectionClassifier:
    """
    문서 레벨 컨텍스트(북마크, ToC, 본문 폰트 크기)를 보유하고
    블록 단위로 classify()를 호출하는 분류기.
    """

    def __init__(
        self,
        body_font_size: float,
        pdf_bookmarks: list,
        toc_entries: list,
    ) -> None:
        """
        Args:
            body_font_size: 문서 전체 최빈 폰트 크기 (파서가 전달).
            pdf_bookmarks:  doc.get_toc() 결과.
                            형식: [(depth, title, page_no), ...]
            toc_entries:    ToC 페이지 파싱 결과.
                            형식: [{"label": "1.2", "title": "Background"}, ...]
        """
        self.body_size = max(body_font_size, 1.0)  # 0 나누기 방지

        # S급 인덱스: O(1) 조회를 위해 dict/set으로 변환
        self.bookmark_index = {
            _normalize(title): depth
            for (depth, title, _page) in pdf_bookmarks
        }
        self.toc_index = {
            _normalize(entry["title"]): entry.get("label")
            for entry in toc_entries
        }

    # -----------------------------------------------------------------------
    # 공개 메서드
    # -----------------------------------------------------------------------

    def classify(self, features: SectionFeatures) -> ClassificationResult:
        """블록 하나를 분류하고 ClassificationResult를 반환한다."""

        # --- 사전 계산 ---
        normalized = _normalize(features.text)
        font_ratio = features.max_font_size / self.body_size
        center_x = (features.bbox_x0 + features.bbox_x1) / 2
        is_centered = abs(center_x - features.page_width /
                          2) < features.page_width * 0.10
        ends_with_punct = features.text.rstrip().endswith((".", "?", "!"))
        word_count = len(features.text.split())
        label, title = extract_section_label(features.text)

        # ===================================================================
        # [단계 1] S급 Fast-Track: PDF 북마크 100% 일치
        # ===================================================================
        if normalized in self.bookmark_index:
            depth = self.bookmark_index[normalized]
            return ClassificationResult(
                level=depth,
                block_type=_depth_to_type(depth),
                confidence=1.0,
                section_label=label,
                section_title=title or features.text,
                decision_path="S급[북마크일치]",
            )

        # ===================================================================
        # [단계 2] S급 Fast-Track: ToC 목차 일치
        # ===================================================================
        if normalized in self.toc_index:
            toc_label = self.toc_index[normalized]
            depth = toc_label.count(".") + 1 if toc_label else 1
            return ClassificationResult(
                level=depth,
                block_type=_depth_to_type(depth),
                confidence=0.95,
                section_label=toc_label,
                section_title=features.text,
                decision_path="S급[목차일치]",
            )

        # ===================================================================
        # [단계 3] A급+B급 가중치 스코어링
        # ===================================================================
        score = 0
        log: List[str] = []

        # A급: 폰트 크기 (본문 대비 비율)
        if font_ratio >= 1.5:
            score += 50
            log.append("폰트1.5배(+50)")
        elif font_ratio >= 1.2:
            score += 40
            log.append("폰트1.2배(+40)")
        elif font_ratio >= 1.1:
            score += 20
            log.append("폰트1.1배(+20)")

        # B급: 넘버링 패턴 정규식 일치
        if label is not None:
            score += 30
            log.append("넘버링패턴(+30)")

        # B급: 볼드 처리
        if features.is_bold:
            score += 20
            log.append("볼드(+20)")

        # B급: 중앙 정렬
        if is_centered:
            score += 10
            log.append("중앙정렬(+10)")

        # B급: 전체 대문자 (ALL CAPS) — 표지/장 제목에서 빈번
        if features.text == features.text.upper() and len(features.text) > 5:
            score += 20
            log.append("전체대문자(+20)")

        # 패널티: 문장 부호로 끝남 → 본문일 가능성
        if ends_with_punct:
            score -= 50
            log.append("문장부호끝(-50)")

        # 패널티: 단어 수 30개 초과 → 제목치고 너무 긺
        if word_count > 30:
            score -= 30
            log.append("긴텍스트(-30)")

        score_summary = f"[{score}점]: {', '.join(log)}" if log else f"[{score}점]"

        # --- 임계치 판정 ---
        if score >= 70:
            if label is not None:
                depth = label.count(".") + 1
                sec_title = title
            else:
                depth = 1 if font_ratio >= 1.5 else 2
                sec_title = features.text
                label = None

            return ClassificationResult(
                level=depth,
                block_type=_depth_to_type(depth),
                confidence=min(score / 100.0, 0.95),
                section_label=label,
                section_title=sec_title,
                decision_path=f"스코어링{score_summary}",
            )

        return ClassificationResult(
            level=999,
            block_type="paragraph",
            confidence=max(0.0, 1.0 - max(score, 0) / 100.0),
            section_label=None,
            section_title=None,
            decision_path=f"본문{score_summary}",
        )
