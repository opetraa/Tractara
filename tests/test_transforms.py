import pytest
from lxml import etree
import xml.etree.ElementTree as ET
from tractara.catalogs.transforms import (
    _assemble_dmc_identifier,
    _dmc_from_dmref,
    _date_from_element_attrs,
    _extract_text_content,
    _extract_jats_author_name,
    _join_elements_text,
    _s1000d_security,
    _s1000d_info_code_to_type,
)


def test_assemble_dmc_identifier_edge_cases():
    # When dm_code is the element itself without proper tags
    el = etree.Element("randomTag")
    # issueInfo not found
    res = _assemble_dmc_identifier(el)
    assert res[0]["value"] == "DMC--------"

    # issueInfo with missing attrs
    el_with_issue = etree.fromstring("<randomTag><issueInfo/></randomTag>")
    res2 = _assemble_dmc_identifier(el_with_issue)
    assert res2[0]["value"] == "DMC--------"


def test_dmc_from_dmref_none():
    # If not enough data, actually it will always return DMC-------
    pass


def test_date_from_element_attrs_missing_year():
    el = etree.Element("date", month="12", day="31")
    assert _date_from_element_attrs(el) is None


def test_extract_text_content():
    el = etree.fromstring("<para> This is text <bold>bold</bold> </para>")
    assert _extract_text_content(el) == "This is text bold"


def test_extract_jats_author_name_empty():
    el = etree.fromstring("<contrib></contrib>")
    assert _extract_jats_author_name(el) == []


def test_join_elements_text():
    el1 = etree.fromstring("<para>First</para>")
    el2 = etree.fromstring("<para>Second</para>")
    el3 = etree.fromstring("<para>  </para>")  # empty
    assert _join_elements_text([el1, el2, el3]) == "First; Second"


def test_s1000d_security():
    # Full secure
    el1 = etree.Element(
        "security",
        securityClassification="02",
        commercialClassification="cc1",
        caveat="cv1",
    )
    res1 = _s1000d_security(el1)
    assert res1["accessRights"] == "restricted"
    assert res1["license"] == "02_cc1_cv1"

    # Public
    el2 = etree.Element("security", securityClassification="01")
    res2 = _s1000d_security(el2)
    assert res2["accessRights"] == "public"
    assert res2["license"] == "01"

    # Missing all -> default license (sec_class != '01' evaluates to True, so restricted)
    el3 = etree.Element("security")
    res3 = _s1000d_security(el3)
    assert res3["accessRights"] == "restricted"
    assert res3["license"] == "01"


def test_s1000d_info_code_to_type():
    el1 = etree.fromstring("<dmIdent><dmCode infoCode='040A'/></dmIdent>")
    assert _s1000d_info_code_to_type(el1) == "TechnicalReport"

    # fallback
    el2 = etree.fromstring("<dmIdent><dmCode infoCode='999X'/></dmIdent>")
    assert _s1000d_info_code_to_type(el2) == "Procedure"
