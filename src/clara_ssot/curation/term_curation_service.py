# src/clara_ssot/curation/term_curation_service.py
from collections import defaultdict
from typing import Any, Dict, List
import re


def merge_term_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    여러 문서에서 나온 TERM 후보들을 (term, termType) 복합 키 기준으로 병합.

    그룹핑 키를 (term, termType)으로 사용하는 이유:
    - 같은 단어가 CLASS(개념)와 REL(관계)로 다르게 분류될 수 있다.
    - 타입이 다른 두 엔트리를 병합하면 termId prefix 불일치가 발생한다.
    """
    grouped: dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for c in candidates:
        key = (c["term"], c.get("termType", "TERM-CLASS"))
        grouped[key].append(c)

    merged_terms: List[Dict[str, Any]] = []
    for _key, items in grouped.items():
        base = items[0].copy()
        for other in items[1:]:
            # definition_en/ko 가 비어있거나 [PENDING...]이면 다른 값으로 채워넣기
            if base.get("definition_en", "").startswith("[PENDING") and other.get(
                "definition_en"
            ):
                base["definition_en"] = other["definition_en"]
            if not base.get("definition_ko") and other.get("definition_ko"):
                base["definition_ko"] = other["definition_ko"]

            # slots.* merge (간단히 '없으면 채우기' 방식)
            slots = base.setdefault("slots", {})
            other_slots = other.get("slots", {})
            for k, v in other_slots.items():
                if not slots.get(k) and v:
                    slots[k] = v

        merged_terms.append(base)

    return merged_terms


def generate_term_filename(term: Dict[str, Any]) -> str:
    """
    TERM의 상태와 ID, 버전을 조합하여 파일명을 생성합니다.
    규칙: [상태]_[TermID]_v[버전].json
    예시: M_AMP_v1.1.json
    """
    status_map = {
        "candidate": "C",
        "anchored": "A",
        "mature": "M",
        "rejected": "X"
    }

    # 1. 상태 코드 (기본값 C)
    status_str = term.get("status", "candidate")
    status_code = status_map.get(status_str, "C")

    # 2. Term ID 정제
    # "term:class:operating_temperature" → 마지막 세그먼트만 추출 → "OPERATING_TEMPERATURE"
    raw_id = term.get("termId", "unknown")
    clean_id = raw_id.split(":")[-1]
    clean_id = re.sub(r"[^a-zA-Z0-9_-]", "", clean_id).upper()

    if not clean_id:
        clean_id = "UNKNOWN"

    # 3. 버전
    version = term.get("version", "1.0")

    # 4. 조합
    filename = f"{status_code}_{clean_id}_v{version}.json"

    return filename
