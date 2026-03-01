"""Doc SSoT 데이터 매핑 모듈."""
# src/tractara/normalization/doc_mapper.py
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from ..parsing.metadata_extractor import extract_metadata
from ..parsing.pdf_parser import ParsedBlock, ParsedDocument

# DOC 스키마 blockType enum에 허용된 값
_VALID_BLOCK_TYPES = {
    "title",
    "abstract",
    "section",
    "subsection",
    "paragraph",
    "list",
    "listItem",
    "table",
    "figure",
    "equation",
    "code",
    "procedure",
    "procedureStep",
    "reference",
    "appendix",
    "footnote",
    "caption",
}


def _blocks_to_content(blocks: List[ParsedBlock]) -> List[Dict[str, Any]]:
    """
    ParsedBlock 리스트 → DOC content 배열 변환.

    계층 구조 필드 전달 규칙:
      - blockId:      b.block_id (UUID) 우선, 없으면 순번 fallback
      - parentId:     b.parent_id (None이면 루트)
      - level:        b.level < 999인 경우만 출력 (paragraph는 생략)
      - contextPath:  항상 출력 (빈 [] 포함)
      - sectionLabel: 값이 있을 때만 출력
      - sectionTitle: 값이 있을 때만 출력
    """
    content: List[Dict[str, Any]] = []

    for i, b in enumerate(blocks, start=1):
        # blockType: 스키마 enum에 없는 값은 "paragraph"로 강제
        block_type = b.block_type if b.block_type in _VALID_BLOCK_TYPES else "paragraph"

        item: Dict[str, Any] = {
            "blockId": b.block_id if b.block_id else f"block-{i:04d}",
            "parentId": b.parent_id,
            "blockType": block_type,
            "text": b.text or "",
            "contextPath": b.context_path,  # 빈 [] 포함 항상 출력
        }

        # level: paragraph(999)는 생략
        if b.level is not None and b.level < 999:
            item["level"] = b.level

        # 섹션 메타데이터
        if b.section_label:
            item["sectionLabel"] = b.section_label
        if b.section_title:
            item["sectionTitle"] = b.section_title

        # bbox
        if b.bbox:
            item["bbox"] = b.bbox.to_dict()

        # 추출 신뢰도 (1.0 미만인 경우만)
        if b.confidence < 1.0:
            item["extractionConfidence"] = b.confidence

        # 표 데이터
        if b.block_type == "table" and b.table_data:
            item["tableData"] = b.table_data

        # 수식 데이터
        if b.block_type == "equation" and b.equation_data:
            item["equationData"] = b.equation_data

        content.append(item)

    return content


def build_doc_baseline(parsed: ParsedDocument) -> Dict[str, Any]:
    """
    ParsedDocument → DOC Baseline JSON.

    메타데이터는 metadata_extractor.extract_metadata()를 통해 추출한다.
    여기서 만든 JSON은 DOC_baseline_schema.json을 반드시 통과해야 한다.
    """
    now = datetime.utcnow().isoformat() + "Z"
    doc_id = "DOC_" + uuid.uuid4().hex

    extracted = extract_metadata(Path(parsed.source_path))

    # 스키마 필수 필드 fallback (required: dc:title, dc:type, dc:language)
    metadata: Dict[str, Any] = {
        "dc:title": extracted.dc_title or "Unknown title (from PDF)",
        "dc:type": extracted.dc_type or "TechnicalReport",
        "dc:language": extracted.dc_language or "ko",
    }

    if extracted.dc_creator:
        metadata["dc:creator"] = extracted.dc_creator
    if extracted.dc_contributor:
        metadata["dc:contributor"] = extracted.dc_contributor
    if extracted.dc_publisher:
        metadata["dc:publisher"] = extracted.dc_publisher
    if extracted.dc_date:
        metadata["dc:date"] = extracted.dc_date
    if extracted.dc_identifier:
        metadata["dc:identifier"] = extracted.dc_identifier
    if extracted.dc_subject:
        metadata["dc:subject"] = extracted.dc_subject
    if extracted.dc_coverage:
        metadata["dc:coverage"] = extracted.dc_coverage

    doc: Dict[str, Any] = {
        "documentId": doc_id,
        "$schema": "https://tractara.org/schemas/doc-baseline/v1.0.0",
        "version": "1.0.0",
        "lastUpdated": now,
        "metadata": metadata,
        "provenance": {
            "sourceFile": parsed.source_path,
            "extractionMetadata": {
                "parserVersion": parsed.metadata.get("version", "0.0.0")
                if parsed.metadata
                else "0.0.0",
                "extractionDate": now,
                "ocrApplied": False,
                "confidence": 0.5,
            },
            "validationStatus": "draft",
            "curationHistory": [],
        },
        "content": _blocks_to_content(parsed.blocks),
    }

    return doc
