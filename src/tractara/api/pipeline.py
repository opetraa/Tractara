"""PDF 인제스트 파이프라인: 파싱→정규화→검증→저장."""
# src/tractara/api/pipeline.py
import logging
import os
from pathlib import Path
from typing import Any, Dict

from ..curation.term_curation_service import merge_term_candidates
from ..landing.landing_repository import save_doc_landing, save_term_candidates_landing
from ..normalization.doc_mapper import build_doc_baseline
from ..normalization.term_mapper import (
    build_term_baseline_candidates,
    extract_term_candidates,
)
from ..parsing.pdf_parser import parse_pdf
from ..parsing.xml_parser import parse_xml
from ..ssot.doc_ssot_repository import upsert_doc as upsert_doc_ssot
from ..ssot.term_ssot_repository import upsert_terms as upsert_term_ssot
from ..validation.json_schema_validator import schema_registry
from ..validation.term_validator import filter_promotable_terms

logger = logging.getLogger(__name__)


def ingest_single_document(file_path: Path) -> Dict[str, Any]:
    """
    PDF 1개에 대한 전체 파이프라인:
      1) 파싱
      2) DOC baseline 생성 + 스키마 검증
      3) Landing 저장 + DOC SSoT 저장
      4) TERM 후보 생성 + Landing 저장
      5) TERM 병합 + 승격 가능 TERM 필터링 + TERM SSoT 저장
      6) 요약 결과 반환
    """

    # LLM API 키 가져오기
    llm_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    warnings = []
    if not llm_api_key:
        msg = (
            "⚠️ No LLM API Key found. Term extraction will likely return empty results."
        )
        logger.warning(msg)
        warnings.append(msg)

    # 1) Parsing (Docling+PyMuPDF 사용 또는 XML 파싱)
    if file_path.suffix.lower() == ".xml":
        parsed = parse_xml(file_path)
    else:
        parsed = parse_pdf(file_path)

    # 2) DOC baseline + 스키마 검증
    doc_baseline = build_doc_baseline(parsed)
    schema_registry.validate(
        "doc", doc_baseline, instance_path=doc_baseline["documentId"]
    )

    # 3) Landing + DOC SSoT
    doc_id = save_doc_landing(doc_baseline)
    upsert_doc_ssot(doc_baseline)

    # 4) TERM 후보 생성 (이제 LLM 사용!)
    logger.info("🚀 Starting TERM extraction (after DOC creation)...")
    term_candidates, extraction_errors = extract_term_candidates(
        parsed, llm_api_key=llm_api_key
    )
    logger.info("🔍 Extracted %d term candidates.", len(term_candidates or []))

    if extraction_errors:
        warnings.extend(extraction_errors)

    if not term_candidates and llm_api_key:
        warnings.append("LLM API Key was present, but 0 terms were extracted.")

    term_baseline_candidates = build_term_baseline_candidates(doc_id, term_candidates)
    save_term_candidates_landing(term_baseline_candidates)

    # 5) TERM 병합 + 승격
    merged_terms = merge_term_candidates(term_baseline_candidates)
    promotable, term_problems = filter_promotable_terms(merged_terms)

    if promotable:
        upsert_term_ssot(promotable)

    # 6) 클라이언트에게 돌려줄 결과
    return {
        "documentId": doc_id,
        "promotedTermCount": len(promotable),
        "termValidationProblems": [p.dict() for p in term_problems],
        "warnings": warnings,
    }
