# src/clara_ssot/parsing/pdf_parser.py
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BoundingBox:
    """PDF 좌표 정보 (첫 번째 문서 전략: bbox 보존 필수)"""
    x0: float
    y0: float
    x1: float
    y1: float
    page: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
            "page": self.page
        }


@dataclass
class ParsedBlock:
    page: int
    block_type: str  # "text" | "table" | "image" | "ocr"
    text: Optional[str] = None
    bbox: Optional[BoundingBox] = None  # 추가!
    table_data: Optional[Dict] = None   # 표 데이터용
    confidence: float = 1.0             # 추출 신뢰도


@dataclass
class ParsedDocument:
    source_path: str
    blocks: List[ParsedBlock]
    metadata: Dict = None


def parse_pdf(path: Path) -> ParsedDocument:
    """
    진짜 PDF 파싱은 나중에,
    지금은 흐름만 확인하는 더미 구현.

    나중에 여기서 pdfplumber / pymupdf / OCR 등을 붙이면 됨.
    """
    dummy_block = ParsedBlock(
        page=1,
        block_type="text",
        text="Dummy text extracted from PDF",
    )
    return ParsedDocument(source_path=str(path), blocks=[dummy_block])
