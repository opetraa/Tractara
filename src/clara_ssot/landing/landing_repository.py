# src/clara_ssot/landing/landing_repository.py
import json
import re
from pathlib import Path
from typing import Any, Dict, List

# BASE_DIR = 프로젝트 루트 (clara-ssot) 까지 올라감
BASE_DIR = Path(__file__).resolve().parent.parent.parent
LANDING_DIR = BASE_DIR / "data" / "landing"
LANDING_DIR.mkdir(parents=True, exist_ok=True)


def save_doc_landing(doc: Dict[str, Any]) -> str:
    """
    DOC baseline JSON을 Landing Zone에 저장.
    파일명: {documentId}.json
    """
    doc_id = doc["documentId"]
    docs_dir = LANDING_DIR / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    path = docs_dir / f"{doc_id}.json"
    path.write_text(json.dumps(doc, ensure_ascii=False,
                    indent=2), encoding="utf-8")
    return doc_id


# termType → 서브디렉토리 매핑
_TYPE_SUBDIR: Dict[str, str] = {
    "TERM-CLASS": "class",
    "TERM-REL": "rel",
    "TERM-RULE": "rule",
}


def save_term_candidates_landing(doc_id: str, terms: List[Dict[str, Any]]) -> None:
    """
    TERM baseline candidates 리스트를 Landing Zone에 저장.
    termType별 서브디렉토리로 분리:
      data/landing/terms/class/  ← TERM-CLASS
      data/landing/terms/rel/    ← TERM-REL
      data/landing/terms/rule/   ← TERM-RULE

    파일명: C_{TERM}_{termId 마지막 세그먼트}.json
    예: term:class:aging_management_program → class/C_AGINGMANAGEMENTPROGRAM_aging_management_program.json
    """
    for term in terms:
        # termType 기반 서브디렉토리 결정
        term_type = term.get("termType", "TERM-CLASS")
        subdir_name = _TYPE_SUBDIR.get(term_type, "class")
        terms_dir = LANDING_DIR / "terms" / subdir_name
        terms_dir.mkdir(parents=True, exist_ok=True)

        # 파일명용 짧은 레이블: headword_en > term 순서로 영문 알파벳만 추출
        clean_term = "TEMP"
        name_candidates = [term.get("headword_en"), term.get("term")]
        for cand in name_candidates:
            if cand:
                extracted = re.sub(r"[^a-zA-Z0-9_-]", "", cand).upper()
                if extracted:
                    clean_term = extracted
                    break

        # termId의 마지막 세그먼트 사용 (콜론 포함 문자를 파일명에서 배제)
        term_id_val: str = term.get("termId", "unknown")
        filename_stem = term_id_val.split(":")[-1]
        filename = f"C_{clean_term}_{filename_stem}.json"
        path = terms_dir / filename

        path.write_text(json.dumps(term, ensure_ascii=False,
                        indent=2), encoding="utf-8")
