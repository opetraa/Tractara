# src/clara_ssot/normalization/term_mapper.py
from dataclasses import dataclass
from typing import Any, Dict, List
import logging
import os
import uuid

import instructor
import google.generativeai as genai
from pydantic import BaseModel, Field

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


class LLMTermExtractor:
    """
    LLM 기반 TERM 추출기 (첫 번째 문서 전략: CoT + Pydantic)
    """

    def __init__(self, api_key: str):
        # Gemini 설정
        genai.configure(api_key=api_key)
        self.client = instructor.from_gemini(
            client=genai.GenerativeModel(model_name="gemini-1.5-flash"),
            mode=instructor.Mode.GEMINI_JSON,
        )

    def extract(self, text_chunks: List[str]) -> List[TermCandidate]:
        """
        여러 텍스트 청크에서 TERM 추출
        """
        all_candidates = []

        for chunk in text_chunks:
            if len(chunk.strip()) < 50:  # 너무 짧은 텍스트는 스킵
                continue

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
                        context=t.context
                    )
                    for t in result.terms
                ]
                all_candidates.extend(candidates)

                logger.info(f"Extracted {len(candidates)} terms from chunk")
                logger.debug(f"CoT reasoning: {result.reasoning}")

            except Exception as e:
                logger.error(f"TERM extraction failed: {e}")

        return all_candidates

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

        prompt = f"""당신은 원자력 기술 문서에서 전문 용어를 추출하는 전문가입니다.
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

        # Instructor로 구조화된 출력 강제
        response = self.client.messages.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            response_model=TermExtractionResult,
        )

        return response


def extract_term_candidates(
    parsed: ParsedDocument,
    llm_api_key: str = None
) -> List[TermCandidate]:
    """
    ParsedDocument에서 TERM 후보 추출

    첫 번째 문서 전략:
    - LLM 기반 추출 (CoT + Few-shot)
    - Instructor로 구조화된 출력 보장
    """
    # 인자로 키가 안 넘어왔으면 환경 변수에서 조회
    if not llm_api_key:
        llm_api_key = os.environ.get("GEMINI_API_KEY")

    if not llm_api_key:
        logger.warning("No LLM API key provided, using dummy extractor")
        return [
            TermCandidate(
                term="AMP",
                definition_en=None,
                definition_ko="경년열화 관리 프로그램",
            )
        ]

    # 텍스트 청크 준비
    text_chunks = [
        block.text for block in parsed.blocks
        if block.text and len(block.text) > 50
    ]

    # LLM 추출
    extractor = LLMTermExtractor(api_key=llm_api_key)
    # 더 많은 용어를 찾기 위해 청크 수 증가 (테스트용 10개)
    candidates = extractor.extract(text_chunks[:10])

    logger.info(f"Extracted {len(candidates)} TERM candidates")
    return candidates


def build_term_baseline_candidates(
    doc_baseline_id: str,
    candidates: List[TermCandidate],
) -> List[Dict[str, Any]]:
    """
    TermCandidate → TERM Baseline JSON 변환
    """
    term_jsons: List[Dict[str, Any]] = []

    for c in candidates:
        term_id = str(uuid.uuid4())
        term_json = {
            "type": "term_entry",
            "termId": term_id,
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
