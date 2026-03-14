from pathlib import Path

import pytest
from lxml import etree

from tractara.parsing.metadata_extractor import (
    ExtractedMetadata,
    _apply_catalog_metadata,
    _merge_xml_metadata,
)


def test_merge_xml_metadata():
    base = ExtractedMetadata(dc_title="Base Title", dc_language="en")
    override = ExtractedMetadata(dc_title="Override Title", dc_description="New Desc")

    merged = _merge_xml_metadata(base, override)

    assert merged.dc_title == "Override Title"
    assert merged.dc_language == "en"
    assert merged.dc_description == "New Desc"


def test_apply_catalog_metadata_basic():
    xml_content = """<root>
        <title>Doc Title</title>
        <creator>John Doe</creator>
        <meta><subject>Math</subject></meta>
        <desc>Line 1</desc>
        <desc>Line 2</desc>
    </root>"""
    root = etree.fromstring(xml_content)

    catalog = {
        "format_id": "test_cat",
        "metadata": {
            "dc_title": {"xpath": ".//title"},
            "dc_creator": {"xpath": ".//creator", "entity_type": "person"},
            "dc_subject": {"xpath": ".//subject"},
            "dc_description": [
                {"xpath": ".//desc[1]"},
                {"xpath": ".//desc[2]", "join_separator": " | "},
            ],
        },
    }

    meta = _apply_catalog_metadata(root, catalog)
    assert meta.dc_title == "Doc Title"
    assert meta.dc_creator == [{"name": "John Doe", "entityType": "person"}]
    assert meta.dc_subject == "Math"
    assert meta.dc_description == "Line 1 | Line 2"
