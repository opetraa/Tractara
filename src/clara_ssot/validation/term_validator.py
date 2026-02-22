# src/clara_ssot/validation/term_validator.py
from typing import Any, Dict, List

from ..problem_details import MachineReadableError, ProblemDetails
from ..tracing import get_span_id, get_trace_id
from .json_schema_validator import SchemaValidationException, schema_registry

# SSoT로 승격되려면 반드시 있어야 하는 필드들
# TERM 스키마에서 definition_en은 required + minLength 10임
REQUIRED_FIELDS: List[str] = ["definition_en"]

# LLM이 "뭘 기대하는 필드냐"를 이해할 수 있도록 힌트 추가
REQUIRED_FIELD_HINTS: Dict[str, Dict[str, Any]] = {
    "definition_en": {
        "minLength": 10,
        "description": "영문 정의 (검색·요약·툴팁에 직접 사용되는 교과서적 정의)",
    },
}


def determine_term_status(term: Dict[str, Any]) -> str:
    """
    TERM의 신뢰도(x축)와 완성도(y축)를 기반으로 상태(Status)를 결정합니다.

    Returns:
        'candidate' | 'anchored' | 'mature' | 'rejected'
    """
    # 0. 이미 Rejected 상태라면 유지
    if term.get("status") == "rejected":
        return "rejected"

    # 1. 신뢰도 체크 (x축)
    # 전문가 검증이 되었거나, 신뢰할 수 있는 소스에서 온 경우
    quality = term.get("qualityMetrics", {})
    is_trusted = quality.get("expertValidated", False)

    # (확장 가능) provenance 정보를 통해 신뢰도 판단 로직 추가 가능
    # if term.get("provenance", {}).get("sourceType") == "glossary": is_trusted = True

    # 2. 완성도 체크 (y축)
    # 필수 필드가 모두 채워졌는지 확인
    missing_fields = []
    for field in REQUIRED_FIELDS:
        val = term.get(field)
        if not val or (isinstance(val, str) and val.startswith("[PENDING")):
            missing_fields.append(field)
    is_complete = len(missing_fields) == 0

    # 3. 상태 결정 매트릭스
    if is_trusted and is_complete:
        return "mature"  # M: 신뢰도 높음 + 완성도 높음
    elif is_trusted:
        return "anchored"  # A: 신뢰도 높음 + 완성도 낮음 (ID 확정, 참조 가능)
    else:
        return "candidate"  # C: 신뢰도 낮음 (LLM 추출 단계)


def filter_promotable_terms(
    merged_terms: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[ProblemDetails]]:
    """
    병합된 TERM 들 중에서:
    - 필수 필드(REQUIRED_FIELDS)가 다 채워져 있고
    - JSON Schema 검증도 통과하는 것만 SSoT로 승격시킨다.

    return:
        (promotable_terms, problems)
    """
    promotable: List[Dict[str, Any]] = []
    problems: List[ProblemDetails] = []

    for term in merged_terms:
        term_id = term.get("termId", "<unknown-term-id>")

        # 1) 필수 필드 채워졌는지 확인
        missing: List[str] = []
        for field_name in REQUIRED_FIELDS:
            value = term.get(field_name)

            # 값이 없거나, 빈 문자열이거나, "[PENDING..." 플레이스홀더면 "실제 없음"으로 간주
            if value is None or (
                isinstance(value, str)
                and (value.strip() == "" or value.startswith("[PENDING"))
            ):
                missing.append(field_name)

        if missing:
            error_items: List[MachineReadableError] = []
            for field_name in missing:
                hints = REQUIRED_FIELD_HINTS.get(field_name, {})

                error_items.append(
                    MachineReadableError(
                        # LLM이 패턴으로 쓰기 좋게 도메인+상황이 보이도록 코드 설계
                        code="TERM_REQUIRED_FIELD_MISSING",
                        target=field_name,
                        detail=f"Field '{field_name}' must be filled before promotion to SSoT.",
                        meta={
                            "termId": term_id,
                            "field": field_name,
                            "expected": {
                                "description": hints.get("description"),
                                "minLength": hints.get("minLength"),
                            },
                            # 실제 값이 뭔지 그대로 보여주기 (LLM이 diff 계산하기 쉬움)
                            "actual": value,
                            # LLM이 '이건 유저/데이터 쪽 문제구나'를 바로 알 수 있도록
                            "errorClass": "user_input",
                        },
                    )
                )

            problems.append(
                ProblemDetails(
                    type="https://clara-ssot.org/problems/term-missing-fields",
                    title="TERM candidate is still partial",
                    status=422,
                    detail=(
                        f"TERM '{term_id}' is missing required fields for SSoT promotion: {missing}."
                        "See errors[].meta.expected/actual for details."
                    ),
                    instance=term_id,
                    errors=error_items,
                    trace_id=get_trace_id(),
                    span_id=get_span_id(),
                )
            )
            # 이 TERM은 아직 승격 불가 → 다음 TERM으로
            continue

        # 2) JSON Schema 검증
        try:
            schema_registry.validate("term", term, instance_path=term_id)
        except SchemaValidationException as ex:
            # Schema 쪽 에러는 json_schema_validator에서 이미
            # errors[]에 path / validator / 메시지를 채워서 ProblemDetails를 만들어줌
            problems.append(ex.problem)
            continue

        # 3) 둘 다 통과하면 승격 대상
        promotable.append(term)

    return promotable, problems
