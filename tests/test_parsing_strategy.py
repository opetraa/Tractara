# tests/test_parsing_strategy.py
# tests/test_parsing_strategy.py
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from clara_ssot.parsing.pdf_parser import (
    parse_pdf,
    ParsedDocument,
    ParsedBlock,
    BoundingBox,
)
from clara_ssot.normalization.term_mapper import extract_term_candidates, TermCandidate


@patch("clara_ssot.parsing.pdf_parser.DoclingParser")
@patch("clara_ssot.parsing.pdf_parser.pymupdf")
def test_docling_pymupdf_parsing(MockPyMuPDF, MockDocling):
    """Docling+PyMuPDF 멀티엔진 테스트 (Mocked)"""
    pdf_path = Path("data/test_sample.pdf")

    # Mock pymupdf.open for text density check in parse_pdf
    mock_doc = MagicMock()
    mock_page = MagicMock()
    mock_page.get_text.return_value = "A" * 200  # 충분한 텍스트 (스캔 문서가 아님)
    mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
    mock_doc.__len__ = MagicMock(return_value=1)
    mock_doc.close = MagicMock()
    MockPyMuPDF.open.return_value = mock_doc

    # Mock DoclingParser behavior
    mock_docling_instance = MockDocling.return_value
    mock_docling_instance.parse.return_value = ParsedDocument(
        source_path=str(pdf_path),
        blocks=[
            ParsedBlock(
                page=1,
                block_type="paragraph",
                text="Sample text",
                bbox=BoundingBox(10, 10, 100, 20, 1),
            ),
            ParsedBlock(
                page=1,
                block_type="table",
                text="| col1 | col2 |",
                table_data={"headers": ["col1"], "rows": [["val1"]]},
                bbox=BoundingBox(10, 30, 100, 50, 1),
            ),
        ],
        metadata={"parser": "docling"},
    )

    parsed = parse_pdf(pdf_path)

    assert len(parsed.blocks) > 0
    assert parsed.metadata["parser"] == "docling"

    # bbox 좌표 확인
    blocks_with_bbox = [b for b in parsed.blocks if b.bbox is not None]
    assert len(blocks_with_bbox) > 0

    # 표 추출 확인
    table_blocks = [b for b in parsed.blocks if b.block_type == "table"]
    if table_blocks:
        assert table_blocks[0].table_data is not None


@patch("clara_ssot.normalization.term_mapper.LLMTermExtractor")
def test_llm_term_extraction(MockLLMExtractor):
    """LLM TERM 추출 테스트 (Mocked)"""

    # Mock LLM behavior - extract()는 (candidates, errors) 튜플 반환
    mock_extractor_instance = MockLLMExtractor.return_value
    mock_extractor_instance.extract.return_value = (
        [
            TermCandidate(
                term="AMP",
                definition_en="Aging Management Program",
                definition_ko="경년열화 관리 프로그램",
                headword_en="Aging Management Program",
                headword_ko="경년열화 관리 프로그램",
                domain=["nuclear"],
                context="경년열화 관리 프로그램(AMP)은...",
            )
        ],
        [],  # errors
    )

    # 더미 문서
    dummy_doc = ParsedDocument(
        source_path="test.pdf",
        blocks=[
            ParsedBlock(
                page=1,
                block_type="paragraph",
                text="경년열화 관리 프로그램(AMP)은 원자력 발전소의 장기 운전을 위해 필수적이다.",
            )
        ],
    )

    api_key = "test_key"  # 실제 테스트에서는 환경변수 사용
    candidates, errors = extract_term_candidates(dummy_doc, llm_api_key=api_key)

    assert len(candidates) > 0
    assert any("AMP" in c.term for c in candidates)
