# src/clara_ssot/problem_details.py
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class MachineReadableError(BaseModel):
    code: str = Field(
        ..., description="짧은 오류 코드 (예: 'schema_validation_failed')"
    )
    target: Optional[str] = Field(
        None, description="에러가 발생한 필드나 리소스 (예: 'TERM.definition_en')"
    )
    detail: Optional[str] = Field(None, description="사람이 읽을 수 있는 세부 설명")
    meta: Optional[dict] = Field(default_factory=dict)


class ProblemDetails(BaseModel):
    """
    RFC 7807 스타일 에러 응답 모델
    """

    type: str = Field(
        "about:blank",
        description="에러 타입 URI (문서화된 에러 타입이면 URL 사용 권장)",
    )
    title: str
    status: int
    detail: Optional[str] = None
    instance: Optional[str] = None

    errors: List[MachineReadableError] = Field(
        default_factory=list,
        description="머신이 읽기 좋은 에러 리스트",
    )

    trace_id: Optional[str] = Field(
        None,
        description="분산 추적용 trace_id (W3C Trace Context)",
    )
    span_id: Optional[str] = Field(
        None,
        description="현재 span ID",
    )

    error_class: Optional[Literal["user_input", "system_bug", "dependency"]] = Field(
        default=None,
        description="에러 원인 분류 (사용자 입력 / 시스템 버그 / 외부 의존성)",
    )
