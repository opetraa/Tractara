# src/clara_ssot/normalization/doc_mapper.py
import uuid
from datetime import datetime
from typing import Any, Dict, List

from ..parsing.pdf_parser import ParsedBlock, ParsedDocument


def _blocks_to_content(blocks: List[ParsedBlock]) -> List[Dict[str, Any]]:
    """
    ParsedBlock → DOC content 변환 (bbox 포함)
    """
    content: List[Dict[str, Any]] = []

    for i, b in enumerate(blocks, start=1):
        item: Dict[str, Any] = {
            "blockId": f"block-{i:04d}",
            "parentId": None,
            "blockType": (
                b.block_type
                if b.block_type in ["paragraph", "table", "section", "title"]
                else "paragraph"
            ),
            "text": b.text or "",
        }

        # ✨ bbox 추가 (첫 번째 문서 전략 핵심)
        if b.bbox:
            item["bbox"] = b.bbox.to_dict()

        # ✨ 신뢰도 추가
        if b.confidence < 1.0:
            item["extractionConfidence"] = b.confidence

        # 표 데이터
        if b.block_type == "table" and b.table_data:
            item["tableData"] = b.table_data

        content.append(item)

    return content


def build_doc_baseline(parsed: ParsedDocument) -> Dict[str, Any]:
    """
    ParsedDocument → DOC Baseline JSON.

    여기서 만든 JSON은 DOC_baseline_schema.json을 반드시 통과해야 한다.
    """
    now = datetime.utcnow().isoformat() + "Z"

    # 스키마 패턴: "^[a-zA-Z0-9_-]+$"
    # → 콜론, 점 등은 안 되므로 UUID 기반으로 만든다.
    doc_id = "DOC_" + uuid.uuid4().hex

    doc: Dict[str, Any] = {
        "documentId": doc_id,
        "$schema": "https://clara-ssot.org/schemas/doc-baseline/v1.0.0",
        "version": "1.0.0",
        "lastUpdated": now,
        "metadata": {
            # DOC 스키마 required: dc:title, dc:type, dc:language
            "dc:title": "Unknown title (from PDF)",  # TODO: 제목 추출 로직으로 교체
            "dc:type": "TechnicalReport",  # enum 중 하나
            "dc:language": "ko",  # 나중에 lang detection 붙이면 됨
        },
        "provenance": {
            "sourceFile": parsed.source_path,
            "extractionMetadata": {
                "parserVersion": "0.1.0",
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
