# tests/test_catalog_loader.py
import pytest
from lxml import etree

from tractara.catalogs import catalog_loader
from tractara.catalogs.transforms import TRANSFORM_REGISTRY


def test_base_catalog_loads():
    base = catalog_loader.get_base_catalog()
    assert base is not None
    assert base["format_id"] == "_base"
    assert "dc_namespaces" in base


def test_detect_s1000d():
    cat = catalog_loader.detect_catalog("dmodule")
    assert cat is not None
    assert cat["format_id"] == "s1000d"


def test_detect_jats():
    cat = catalog_loader.detect_catalog("article")
    assert cat is not None
    assert cat["format_id"] == "jats"


def test_detect_unknown():
    cat = catalog_loader.detect_catalog("unknown_random_tag")
    assert cat is None


def test_transform_dmc_assemble():
    xml = """<dmCode modelIdentCode="S1000DBIKE" systemDiffCode="AAA" systemCode="DA1" subSystemCode="20" subSubSystemCode="0" assyCode="00AA" disassyCode="720" disassyCodeVariant="A" infoCode="A" infoCodeVariant="" itemLocationCode="A" />"""
    el = etree.fromstring(xml)
    res = TRANSFORM_REGISTRY["dmc_assemble"](el)
    assert len(res) == 1
    assert res[0]["scheme"] == "URI"
    assert res[0]["value"] == "DMC-S1000DBIKE-AAA-DA1-200-00AA-720A-A-A"


def test_transform_date():
    xml = '<issueDate year="2023" month="5" day="9"/>'
    res = TRANSFORM_REGISTRY["date_from_attrs"](etree.fromstring(xml))
    assert res == "2023-05-09"


def test_jats_author_names():
    xml = """<root>
    <contrib contrib-type="author"><name><surname>Smith</surname><given-names>John</given-names></name></contrib>
    <contrib contrib-type="author"><name><surname>Doe</surname><given-names>Jane</given-names></name></contrib>
    </root>"""
    els = etree.fromstring(xml).findall(".//contrib")
    res = TRANSFORM_REGISTRY["jats_author_name"](els)
    assert len(res) == 2
    assert res[0]["name"] == "John Smith"
    assert res[1]["name"] == "Jane Doe"
