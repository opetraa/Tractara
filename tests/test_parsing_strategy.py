# tests/test_parsing_strategy.py
# tests/test_parsing_strategy.py
from pathlib import Path
from unittest.mock import MagicMock, patch

from tractara.normalization.term_mapper import TermCandidate, extract_term_candidates
from tractara.parsing.pdf_parser import (
    BoundingBox,
    ParsedBlock,
    ParsedDocument,
    parse_pdf,
)


@patch("tractara.parsing.pdf_parser.pymupdf")
@patch("tractara.parsing.pdf_parser.DoclingParser")
@patch("tractara.parsing.pdf_parser.PyMuPDFParser")
def test_docling_pymupdf_parsing(MockPyMuPDF, MockDocling, mock_pymupdf_mod):
    """Docling+PyMuPDF 멀티엔진 테스트 (Mocked)"""
    pdf_path = Path("data/test_sample.pdf")

    # Mock pymupdf.open so parse_pdf doesn't try to open a real file
    mock_doc = MagicMock()
    mock_doc.__len__ = lambda self: 5
    mock_page = MagicMock()

    # get_text("dict") 호출 시 올바른 딕셔너리 구조를 반환하도록 수정
    def mock_get_text(format_type="text", *args, **kwargs):
        if format_type == "dict":
            return {"blocks": []}
        return "x" * 200

    mock_page.get_text.side_effect = mock_get_text

    # 페이지의 rect.height 가 Int 등 숫자 자료형으로 반환되도록 Mocking
    mock_rect = MagicMock()
    mock_rect.height = 800
    mock_page.rect = mock_rect

    mock_doc.__iter__ = lambda self: iter([mock_page] * 5)
    mock_pymupdf_mod.open.return_value = mock_doc

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
            ParsedBlock(
                page=1,
                block_type="equation",
                text="E = mc^2",
                equation_data={"latex": "E = mc^2"},
                bbox=BoundingBox(10, 60, 100, 80, 1),
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

    # 수식 추출 확인
    equation_blocks = [b for b in parsed.blocks if b.block_type == "equation"]
    if equation_blocks:
        assert equation_blocks[0].equation_data is not None
        assert equation_blocks[0].equation_data["latex"] == "E = mc^2"


@patch("tractara.normalization.term_mapper.LLMTermExtractor")
def test_llm_term_extraction(MockLLMExtractor):
    """LLM TERM 추출 테스트 (Mocked)"""
    from tractara.models.term_types import TermType

    # Mock은 (List[TermCandidate], List[str]) 튜플을 반환해야 한다
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
                term_type=TermType.CLASS,
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
    assert all(c.term_type == TermType.CLASS for c in candidates)


def test_pymupdf_is_equation():
    """PyMuPDFParser의 수식 탐지 휴리스틱 테스트"""
    from tractara.parsing.pdf_parser import PyMuPDFParser

    parser = PyMuPDFParser()

    # 끝이 괄호 번호
    assert parser._is_equation("E = mc^2 (1.1)") is True
    # 수학 기호 비율 / 개수
    assert parser._is_equation("∑ x_i = 100") is True
    assert parser._is_equation("∫ f(x) dx = F(x) + C") is True
    assert parser._is_equation("α + β = γ") is True

    # 일반 문장
    assert parser._is_equation("이것은 일반적인 문장입니다.") is False
    assert parser._is_equation("Figure 1.2를 참조하십시오.") is False
    assert parser._is_equation("Section 3.1: Introduction") is False


def test_reclassify_equations():
    """문맥 기반 수식 탐지(_reclassify_equations) 테스트"""
    from tractara.parsing.pdf_parser import ParsedBlock, _reclassify_equations

    blocks = [
        ParsedBlock(
            page=1, block_type="paragraph", text="The fatigue life is given by"
        ),
        ParsedBlock(page=1, block_type="paragraph", text="N = C / S^n (3.1)"),
        ParsedBlock(
            page=1,
            block_type="paragraph",
            text="where S is the stress amplitude and C is a constant.",
        ),
    ]

    # 처리 전 확인
    assert blocks[1].block_type == "paragraph"

    # 처리 후 확인
    res_blocks = _reclassify_equations(blocks)
    assert res_blocks[1].block_type == "equation"
    assert res_blocks[1].equation_data is not None
    assert res_blocks[1].equation_data["equationNumber"] == "3.1"

    # 0, 2번 블록은 paragraph 유지
    assert res_blocks[0].block_type == "paragraph"
    assert res_blocks[2].block_type == "paragraph"
