# src/clara_ssot/ssot/term_ssot_repository.py
import json
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SSOT_TERM_DIR = BASE_DIR / "data" / "ssot" / "terms"

# termType → 서브디렉토리 매핑
_TYPE_SUBDIR: Dict[str, str] = {
    "TERM-CLASS": "class",
    "TERM-REL": "rel",
    "TERM-RULE": "rule",
}


def _term_subdir(term: Dict[str, Any]) -> Path:
    """termType 필드를 읽어 해당 서브디렉토리 경로를 반환한다."""
    term_type = term.get("termType", "TERM-CLASS")
    subdir_name = _TYPE_SUBDIR.get(term_type, "class")
    subdir = SSOT_TERM_DIR / subdir_name
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir


def upsert_terms(terms: List[Dict[str, Any]]) -> None:
    """
    TERM SSoT 저장/갱신.
    termType별 서브디렉토리로 분리 저장:
      data/ssot/terms/class/   ← TERM-CLASS
      data/ssot/terms/rel/     ← TERM-REL
      data/ssot/terms/rule/    ← TERM-RULE

    파일명: {termId 마지막 세그먼트}.json
    예: term:class:operating_temperature → class/operating_temperature.json
    """
    for term in terms:
        term_id: str = term["termId"]
        # termId에서 파일명 세그먼트 추출 (term:class:xxx → xxx)
        filename_stem = term_id.split(":")[-1]
        subdir = _term_subdir(term)
        path = subdir / f"{filename_stem}.json"
        path.write_text(
            json.dumps(term, ensure_ascii=False, indent=2), encoding="utf-8"
        )
