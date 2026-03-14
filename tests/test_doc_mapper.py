import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from tractara.normalization.doc_mapper import _blocks_to_content, build_doc_baseline
from tractara.parsing.models import ParsedBlock, ParsedDocument


def test_blocks_to_content():
    blocks = [
        ParsedBlock(
            page=1,
            block_type="title",
            text="Test Title",
            level=0,
            block_id="b1",
            context_path=[],
        ),
        ParsedBlock(
            page=1,
            block_type="paragraph",
            text="P1",
            level=999,
            block_id="b2",
            parent_id="b1",
            context_path=["Test Title"],
        ),
        ParsedBlock(
            page=1,
            block_type="note",
            text="N1",
            level=1,
            block_id="b3",
            context_path=["Test Title"],
        ),
        ParsedBlock(
            page=1,
            block_type="unknown_type",
            text="U1",
            level=1,
            block_id="b4",
            context_path=[],
        ),
        ParsedBlock(
            page=1,
            block_type="table",
            text="Table1",
            level=1,
            table_data={"headers": ["A"], "rows": [["1"]]},
            context_path=[],
        ),
        ParsedBlock(
            page=1,
            block_type="equation",
            text="Eq1",
            level=1,
            equation_data={"latex": "E=mc^2"},
            context_path=[],
            confidence=0.8,
        ),
    ]
    blocks[5].structured_content = {"formula": "E=mc^2"}

    content = _blocks_to_content(blocks)
    assert len(content) == 6

    # Title block
    assert content[0]["blockType"] == "title"
    assert content[0]["blockId"] == "b1"
    assert content[0]["level"] == 0

    # Paragraph block
    assert content[1]["blockType"] == "paragraph"
    assert content[1]["parentId"] == "b1"
    assert "level" not in content[1]  # level 999 should be omitted

    # Note block (valid type)
    assert content[2]["blockType"] == "note"

    # Unknown block -> defaults to paragraph
    assert content[3]["blockType"] == "paragraph"

    # Table
    assert content[4]["blockType"] == "table"
    assert "tableData" in content[4]

    # Equation with low confidence and structured content
    assert content[5]["blockType"] == "equation"
    assert "equationData" in content[5]
    assert content[5]["extractionConfidence"] == 0.8
    assert content[5]["structuredContent"]["formula"] == "E=mc^2"


@patch("tractara.normalization.doc_mapper.extract_metadata")
def test_build_doc_baseline(mock_extract):
    mock_meta = MagicMock()
    mock_meta.dc_title = "Mock Title"
    mock_meta.dc_type = "Procedure"
    mock_meta.dc_language = "en-US"
    mock_meta.dc_creator = [{"name": "Test Creator"}]
    mock_meta.dc_contributor = [{"name": "Contrib"}]
    mock_meta.dc_publisher = [{"name": "Pub"}]
    mock_meta.dc_date = {"issued": "2023-01-01"}
    mock_meta.dc_identifier = [{"scheme": "URI", "value": "ID-123"}]
    mock_meta.dc_subject = ["Subj"]
    mock_meta.dc_coverage = {"component": "C1"}
    mock_meta.dc_rights = {"accessRights": "public"}
    mock_meta.dc_alternative_titles = ["Alt title"]
    mock_meta.doc_status = "new"

    mock_extract.return_value = mock_meta

    doc = ParsedDocument(
        source_path="/fake/path.xml",
        metadata={"version": "1.0.0"},
        blocks=[
            ParsedBlock(page=1, block_type="title", text="Mock Title", context_path=[])
        ],
        relations=[
            {
                "sourceBlockId": "b1",
                "relationType": "RELATED_TO",
                "target": "b2",
                "confidence": 1.0,
            }
        ],
    )

    baseline = build_doc_baseline(doc)

    assert baseline["documentId"].startswith("DOC_")
    assert baseline["metadata"]["dc:title"] == "Mock Title"
    assert baseline["metadata"]["dc:creator"] == [{"name": "Test Creator"}]
    assert baseline["metadata"]["dc:contributor"] == [{"name": "Contrib"}]
    assert baseline["provenance"]["validationStatus"] == "draft"
    assert len(baseline["content"]) == 1
    assert baseline["content"][0]["blockType"] == "title"
    assert len(baseline["relations"]) == 1

    # Test valid status mapping
    mock_meta.doc_status = "changed"
    baseline = build_doc_baseline(doc)
    assert baseline["provenance"]["validationStatus"] == "partial"

    mock_meta.doc_status = "deleted"
    baseline = build_doc_baseline(doc)
    assert baseline["provenance"]["validationStatus"] == "deprecated"

    mock_meta.doc_status = "unknown_status"
    baseline = build_doc_baseline(doc)
    assert baseline["provenance"]["validationStatus"] == "validated"
