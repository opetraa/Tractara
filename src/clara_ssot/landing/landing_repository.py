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
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return doc_id


def save_term_candidates_landing(doc_id: str, terms: List[Dict[str, Any]]) -> None:
    """
    TERM baseline candidates 리스트를 Landing Zone에 저장.
    파일명: C_{TERM}_{TERM_ID}.json
    예시: C_AMP_550e8400-e29b-41d4-a716-446655440000.json
    경로: data/landing/terms/
    """
    # 문서별 폴더 대신 통합 폴더 사용 (TERM 중심 관리)
    terms_dir = LANDING_DIR / "terms"
    terms_dir.mkdir(parents=True, exist_ok=True)

    for term in terms:
        # 파일명 생성: C_{TERM}_{TERM_ID}.json
        # 한글 용어일 경우 파일명 호환성을 위해 영문 표제어(headword_en)를 우선 사용하고,
        # 영문이 전혀 없으면 "TEMP"로 설정하여 추후 확정되도록 함.
        clean_term = "TEMP"
        # 1순위: headword_en (영문 정식 명칭), 2순위: term (약어 등)
        name_candidates = [term.get("headword_en"), term.get("term")]

        for cand in name_candidates:
            if cand:
                extracted = re.sub(r"[^a-zA-Z0-9_-]", "", cand).upper()
                if extracted:
                    clean_term = extracted
                    break

        term_id_val = term.get("termId", "unknown")
        filename = f"C_{clean_term}_{term_id_val}.json"
        path = terms_dir / filename

        # 파일별로 분리되므로 병합 로직 없이 바로 저장
        path.write_text(
            json.dumps(term, ensure_ascii=False, indent=2), encoding="utf-8"
        )
