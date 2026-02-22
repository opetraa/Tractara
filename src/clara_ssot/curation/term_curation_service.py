# src/clara_ssot/curation/term_curation_service.py
from collections import defaultdict
from typing import Any, Dict, List
import re


def merge_term_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    여러 문서에서 나온 TERM 후보들을 term(표제어) 기준으로 병합.
    - definition_en / definition_ko / slots.* 와 같이 부분적으로만 채워진 필드를 합친다.
    """
    grouped: dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in candidates:
        grouped[c["term"]].append(c)

    merged_terms: List[Dict[str, Any]] = []
    for term_id, items in grouped.items():
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
    status_map = {"candidate": "C", "anchored": "A", "mature": "M", "rejected": "X"}

    # 1. 상태 코드 (기본값 C)
    status_str = term.get("status", "candidate")
    status_code = status_map.get(status_str, "C")

    # 2. Term ID 정제
    # "term:amp" -> "AMP", "term.stress-corrosion" -> "STRESS-CORROSION"
    raw_id = term.get("termId", "unknown")
    # term: 또는 term. 접두어 제거
    clean_id = re.sub(r"^term[:.]", "", raw_id)
    # 파일명에 부적합한 문자 제거 및 대문자화
    clean_id = re.sub(r"[^a-zA-Z0-9_-]", "", clean_id).upper()

    if not clean_id:
        clean_id = "UNKNOWN"

    # 3. 버전
    version = term.get("version", "1.0")

    # 4. 조합
    filename = f"{status_code}_{clean_id}_v{version}.json"

    return filename
