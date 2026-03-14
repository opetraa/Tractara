"""XML 파서 단위 테스트."""
# tests/test_xml_parser.py
from pathlib import Path

import pytest
from lxml import etree

from tractara.parsing.metadata_extractor import extract_metadata
from tractara.parsing.models import ParsedDocument
from tractara.parsing.xml_parser import parse_xml


# 테스트용 더미 파일 생성 픽스처
@pytest.fixture
def jats_xml_file(tmp_path: Path) -> Path:
    content = """<?xml version="1.0" encoding="UTF-8"?>
    <article>
        <front>
            <article-meta>
                <title-group>
                    <article-title>A Study on Nuclear Safety</article-title>
                </title-group>
                <contrib-group>
                    <contrib contrib-type="author">
                        <name><surname>Doe</surname><given-names>John</given-names></name>
                    </contrib>
                </contrib-group>
            </article-meta>
        </front>
        <body>
            <sec>
                <title>1. Introduction</title>
                <p>Safety is paramount <xref ref-type="bibr" rid="ref1">[1]</xref>.</p>
                <sec>
                    <title>1.1 Background</title>
                    <p>Background details.</p>
                </sec>
            </sec>
            <sec>
                <title>2. Methods</title>
                <p>We used the following equation:</p>
                <disp-formula>E = mc^2</disp-formula>
            </sec>
        </body>
        <back>
            <ref-list>
                <ref id="ref1">DOE, 2024</ref>
            </ref-list>
        </back>
    </article>
    """
    file_path = tmp_path / "test_jats.xml"
    file_path.write_text(content)
    return file_path


@pytest.fixture
def s1000d_xml_file(tmp_path: Path) -> Path:
    content = """<?xml version="1.0" encoding="UTF-8"?>
    <dmodule>
        <identAndStatusSection>
            <dmAddress>
                <dmIdent>
                    <dmCode modelIdentCode="BIKE" systemDiffCode="A" systemCode="00" subSystemCode="0" subSubSystemCode="0" assyCode="00" disassyCode="00" disassyCodeVariant="A" infoCode="922" infoCodeVariant="A" itemLocationCode="D"/>
                </dmIdent>
                <dmTitle>
                    <techName>Bicycle</techName>
                    <infoName>Maintenance</infoName>
                </dmTitle>
            </dmAddress>
            <dmStatus issueType="new">
                <skillLevel skillLevelCode="sk01"/>
                <qualityAssurance>
                    <firstVerification verificationType="tabtop"/>
                </qualityAssurance>
                <applicCrossRefTableRef>
                    <dmRef><dmRefIdent><dmCode modelIdentCode="BIKE" systemDiffCode="A" systemCode="00" subSystemCode="0" subSubSystemCode="0" assyCode="00" disassyCode="00" disassyCodeVariant="A" infoCode="00W" infoCodeVariant="A" itemLocationCode="D"/></dmRefIdent></dmRef>
                </applicCrossRefTableRef>
                <brexDmRef>
                    <dmRef><dmRefIdent><dmCode modelIdentCode="BIKE" systemDiffCode="A" systemCode="00" subSystemCode="0" subSubSystemCode="0" assyCode="00" disassyCode="00" disassyCodeVariant="A" infoCode="022" infoCodeVariant="A" itemLocationCode="D"/></dmRefIdent></dmRef>
                </brexDmRef>
            </dmStatus>
        </identAndStatusSection>
        <content>
            <procedure>
                <preliminaryRqmts>
                    <reqCondGroup><noConds/></reqCondGroup>
                    <reqSafety><noSafety/></reqSafety>
                </preliminaryRqmts>
                <mainProcedure>
                    <levelledPara>
                        <title>Removal of Wheel</title>
                        <warning>Always wear safety goggles.</warning>
                        <note><notePara>Ensure the bike is on a stand.</notePara></note>
                        <proceduralStep>
                            <note><notePara>This is an empty step note.</notePara></note>
                        </proceduralStep>
                        <proceduralStep>
                            <reqCondNo id="rc1">Release pressure</reqCondNo>
                            <supportEquipDescr>Wrench</supportEquipDescr>
                            <para>Use wrench to loosen the bolt.</para>
                            <torque>
                                <torqueValue>50</torqueValue>
                                <torqueUnit>Nm</torqueUnit>
                            </torque>
                        </proceduralStep>
                    </levelledPara>
                </mainProcedure>
                <closeRqmts>
                    <reqCondGroup><noConds/></reqCondGroup>
                </closeRqmts>
            </procedure>
        </content>
    </dmodule>
    """
    file_path = tmp_path / "test_s1000d.xml"
    file_path.write_text(content)
    return file_path


def test_jats_parsing(jats_xml_file: Path):
    doc = parse_xml(jats_xml_file)

    assert doc.metadata["parser"] == "jats_xml"

    # 9개 블록 예상. 계층 구조가 잘 잡히는지 확인.
    blocks = doc.blocks

    title_block = blocks[0]
    assert title_block.block_type == "title"
    assert title_block.text == "A Study on Nuclear Safety"

    # relations 확인
    assert len(doc.relations) == 2
    # xref 추출
    assert doc.relations[0]["relationType"] == "CITES"
    assert doc.relations[0]["target"] == "#ref1"


def test_s1000d_parsing(s1000d_xml_file: Path):
    doc = parse_xml(s1000d_xml_file)

    assert doc.metadata["parser"] == "s1000d_xml"

    assert len(doc.relations) == 2
    assert doc.relations[0]["relationType"] == "custom:GOVERNED_BY_APPLIC"
    assert doc.relations[1]["relationType"] == "custom:COMPLIES_WITH"

    blocks = doc.blocks

    title_block = blocks[0]
    assert title_block.block_type == "title"
    assert title_block.text == "Bicycle - Maintenance"

    prelim_block = blocks[1]
    assert prelim_block.block_type == "section"
    assert prelim_block.text == "preliminaryRqmts"
    assert len(prelim_block.structured_content["conditions"]) == 2
    assert prelim_block.structured_content["conditions"][0]["type"] == "condition"
    assert prelim_block.structured_content["conditions"][1]["type"] == "safety"

    # blocks[0] = title
    # blocks[1] = preliminaryRqmts
    # blocks[2] = section (Removal of Wheel)
    # blocks[3] = warning
    # blocks[4] = note (Ensure...)
    # blocks[5] = note (empty step)
    # blocks[6] = procedureStep
    # blocks[7] = closeRqmts

    note = blocks[4]
    assert note.block_type == "note"
    assert "Ensure the bike is on a stand." in note.text
    assert "📝" not in note.text

    empty_step_note = blocks[5]
    assert empty_step_note.block_type == "note"
    assert "This is an empty step note." in empty_step_note.text
    # Empty step suppression: parent_id should be the section (blocks[2])
    assert empty_step_note.parent_id == blocks[2].block_id

    step = blocks[6]
    assert step.block_type == "procedureStep"

    # 구조화된 파라미터 확인 (user request 5 핵심)
    assert step.structured_content is not None
    conditions = step.structured_content["conditions"]
    assert len(conditions) == 2


def test_s1000d_metadata_extraction(s1000d_xml_file: Path):
    meta = extract_metadata(s1000d_xml_file)
    assert meta.dc_description is not None
    assert "Skill Level: sk01" in meta.dc_description
    assert "Verification: tabtop" in meta.dc_description


def test_generic_xml_parsing(tmp_path: Path):
    generic_xml = """<root>
        <some_node>Some text</some_node>
    </root>"""
    p = tmp_path / "generic.xml"
    p.write_text(generic_xml)

    doc = parse_xml(p)
    assert doc.metadata["parser"] == "generic_xml"
    assert len(doc.blocks) == 1
    assert "Some text" in doc.blocks[0].text


def test_parse_xml_exception(tmp_path: Path):
    # invalid xml
    p = tmp_path / "invalid.xml"
    p.write_text("<root>unclosed")
    with pytest.raises(Exception):
        parse_xml(p)
