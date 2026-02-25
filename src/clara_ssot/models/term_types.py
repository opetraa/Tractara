"""
TERM 타입 체계 정의 — 강타입 식별자(Strongly Typed Identifier) 기반 아키텍처.

TERM은 세 가지 타입으로 분리된다:
- TERM-CLASS: 개념(명사). 예) OperatingTemperature, AMP, SCC
- TERM-REL:   관계(동사/서술어). 예) exceeds_threshold, requires_inspection
- TERM-RULE:  조건-결과형 규칙. 예) amp_inspection_interval_rule

모든 외래 키는 URN prefix로 타입을 강제한다:
  term:class:<name>  →  TERM-CLASS 만 허용
  term:rel:<name>    →  TERM-REL 만 허용
  term:rule:<name>   →  TERM-RULE 만 허용

이 파일에는 TermType Enum과 참조용 Pydantic 모델만 포함한다.
FormulaVariable, FormulaConstraint 등 도메인 복합 모델은 formula_parser 단계에서 추가한다.
"""

from enum import Enum

from pydantic import BaseModel, Field


class TermType(str, Enum):
    """TERM 타입 분류. str 상속으로 JSON 직렬화 시 값(문자열)이 그대로 출력된다."""

    CLASS = "TERM-CLASS"  # 개념 (예: OperatingTemperature)
    REL = "TERM-REL"  # 관계 (예: exceeds_threshold)
    RULE = "TERM-RULE"  # 규칙 (조건-결과형 로직)


class TermClassReference(BaseModel):
    """TERM-CLASS를 가리키는 외래 키 참조. term:class: prefix를 강제한다."""

    term_id: str = Field(
        ...,
        pattern=r"^term:class:[a-zA-Z0-9][a-zA-Z0-9_.-]*$",
        description="Must point to a TERM-CLASS (prefix: term:class:)",
    )


class TermRelReference(BaseModel):
    """TERM-REL을 가리키는 외래 키 참조. term:rel: prefix를 강제한다."""

    term_id: str = Field(
        ...,
        pattern=r"^term:rel:[a-zA-Z0-9][a-zA-Z0-9_.-]*$",
        description="Must point to a TERM-REL (prefix: term:rel:)",
    )


class TermRuleReference(BaseModel):
    """TERM-RULE을 가리키는 외래 키 참조. term:rule: prefix를 강제한다."""

    term_id: str = Field(
        ...,
        pattern=r"^term:rule:[a-zA-Z0-9][a-zA-Z0-9_.-]*$",
        description="Must point to a TERM-RULE (prefix: term:rule:)",
    )
