"""XML 파싱 모듈 (JATS + S1000D)."""
# src/tractara/parsing/xml_parser.py
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from lxml import etree

from tractara.catalogs import catalog_loader

from .models import ParsedBlock, ParsedDocument
from .section_classifier import extract_section_label

logger = logging.getLogger(__name__)


def parse_xml(file_path: Path) -> ParsedDocument:
    """XML 파일을 파싱하여 ParsedDocument로 변환합니다."""
    try:
        tree = etree.parse(str(file_path))  # pylint: disable=c-extension-no-member
        root = tree.getroot()

        # XML 루트 태그로 카탈로그 감지
        tag = root.tag.lower()
        catalog = catalog_loader.detect_catalog(tag)

        if catalog:
            logger.info(
                "Detected XML format mapping: %s for %s",
                catalog.get("format_id"),
                file_path.name,
            )
            parser = CatalogDrivenStrategy(catalog)
        else:
            logger.info(
                "Detected Unknown XML format. Using generic fallback: %s",
                file_path.name,
            )
            parser = GenericXmlStrategy()

        return parser.parse(root, str(file_path))
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Failed to parse XML file %s: %s", file_path, e)
        raise


class CatalogDrivenStrategy:
    """YAML 카탈로그에 정의된 매핑 규칙에 따라 XML 구조를 파싱합니다."""

    def __init__(self, catalog: Dict[str, Any]):
        self.catalog = catalog
        self.format_id = catalog.get("format_id", "unknown")
        # Pre-process mappings for faster lookup Strategy logic
        self.content_rules = {}
        content_cfg = catalog.get("content", {})
        self.traverse_strategy = content_cfg.get("traverse_strategy", "recursive")

        for mapping in content_cfg.get("mappings", []):
            tags = mapping.get("tag")
            if isinstance(tags, str):
                self.content_rules[tags.lower()] = mapping
            elif isinstance(tags, list):
                for t in tags:
                    self.content_rules[t.lower()] = mapping

    def parse(self, root: Any, source_path: str) -> ParsedDocument:
        blocks: List[ParsedBlock] = []
        relations: List[Dict[str, Any]] = []

        # (Title is extracted during metadata parsing, but for block generation we might want it here if specified)
        # For backward compatibility with tests/existing logic, we leave title block generation as a special case or
        # let metadata extractor handle it. Both JATS and S1000D previously created a title block here.
        # Let's extract title using the metadata rules just for the title block if possible, or fallback.
        from tractara.catalogs.transforms import TRANSFORM_REGISTRY

        # 문서 제목 추출 (Block용)
        title_text = ""
        meta_cfg = self.catalog.get("metadata", {})
        title_cfg = meta_cfg.get("dc_title", {})
        if "xpath" in title_cfg:
            main_el = root.find(title_cfg["xpath"])
            text_val = (
                "".join(main_el.itertext()).strip() if main_el is not None else ""
            )
            if "combine_with" in title_cfg:
                other_el = root.find(title_cfg["combine_with"])
                other_val = (
                    "".join(other_el.itertext()).strip() if other_el is not None else ""
                )
                sep = title_cfg.get("separator", " ")
                if text_val and other_val:
                    title_text = f"{text_val}{sep}{other_val}"
                else:
                    title_text = text_val or other_val
            else:
                title_text = text_val

        if title_text:
            blocks.append(
                ParsedBlock(
                    page=1,
                    block_type="title",
                    text=title_text,
                    level=0,
                    block_id=str(uuid.uuid4()),
                )
            )

        # Extract relations from catalog
        for rule in self.catalog.get("relations", []):
            xpath = rule.get("xpath")
            if not xpath:
                continue
            elements = root.findall(xpath)
            if not elements:
                continue

            transform_name = rule.get("transform")
            relation_type = rule.get("relation_type", "RELATED_TO")

            for el in elements:
                val = None
                if transform_name and transform_name in TRANSFORM_REGISTRY:
                    val = TRANSFORM_REGISTRY[transform_name](el)
                else:
                    val = "".join(el.itertext()).strip()

                if val:
                    relations.append(
                        {
                            "sourceBlockId": None,
                            "relationType": relation_type,
                            "target": val,
                            "confidence": 1.0,
                        }
                    )

        # 본문 순회
        if self.traverse_strategy == "jats_body":
            body = root.find(".//body")
            if body is not None:
                self._traverse_node(
                    body, blocks, relations, parent_id=None, level=1, context_path=[]
                )

            # 참조 처리 (JATS 특화 부분 - 현재는 하드코딩 유지, 추후 카탈로그 확장 가능)
            ref_list = root.find(".//ref-list")
            if ref_list is not None:
                ref_block = ParsedBlock(
                    page=1,
                    block_type="section",
                    text="References",
                    level=1,
                    block_id=str(uuid.uuid4()),
                )
                blocks.append(ref_block)
                for ref in ref_list.findall(".//ref"):
                    ref_id = ref.get("id", "")
                    ref_text = "".join(ref.itertext()).strip()
                    relations.append(
                        {
                            "sourceBlockId": None,
                            "relationType": "CITES",
                            "target": f"#{ref_id}",
                            "citationText": ref_text,
                            "confidence": 1.0,
                        }
                    )
        else:  # default recursive (like S1000D)
            content = root.find(".//content")
            if content is not None:
                rqmts_tag = "preliminaryRqmts"
                rqmts_el = content.find(f".//procedure/{rqmts_tag}")
                if rqmts_el is not None:
                    structured = self._parse_rqmts(rqmts_el)
                    blocks.append(
                        ParsedBlock(
                            page=1,
                            block_type="section",
                            text=rqmts_tag,
                            level=1,
                            block_id=str(uuid.uuid4()),
                            structured_content=structured,
                        )
                    )
                self._traverse_node(
                    content, blocks, relations, parent_id=None, level=1, context_path=[]
                )
                rqmts_tag = "closeRqmts"
                rqmts_el = content.find(f".//procedure/{rqmts_tag}")
                if rqmts_el is not None:
                    structured = self._parse_rqmts(rqmts_el)
                    blocks.append(
                        ParsedBlock(
                            page=1,
                            block_type="section",
                            text=rqmts_tag,
                            level=1,
                            block_id=str(uuid.uuid4()),
                            structured_content=structured,
                        )
                    )

        return ParsedDocument(
            source_path=source_path,
            blocks=blocks,
            metadata={"parser": f"{self.format_id}_xml", "version": "1.0.0"},
            relations=relations,
        )

    def _traverse_node(
        self,
        element: Any,
        blocks: List[ParsedBlock],
        relations: List[Dict],
        parent_id: Optional[str],
        level: int,
        context_path: List[str],
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-branches,too-many-statements
        """요소 트리를 순회하며 블록을 생성합니다."""
        current_context = list(context_path)

        for child in element:
            if not isinstance(child.tag, str):
                continue
            tag = child.tag.lower()
            rule = self.content_rules.get(tag)

            if rule:
                block_type = (
                    tag
                    if rule.get("block_type_from_tag")
                    else rule.get("block_type", "paragraph")
                )

                if block_type == "section":
                    # Title extraction
                    title_child_tag = rule.get("title_child", "title")
                    title_el = child.find(title_child_tag) if title_child_tag else None
                    sec_title = (
                        "".join(title_el.itertext()).strip()
                        if title_el is not None
                        else "Untitled Section"
                    )

                    label, title = extract_section_label(sec_title)
                    block_id = str(uuid.uuid4())

                    sec_block = ParsedBlock(
                        page=1,
                        block_type="section",
                        text=sec_title,
                        level=level,
                        parent_id=parent_id,
                        context_path=current_context,
                        section_label=label,
                        section_title=title or sec_title,
                        block_id=block_id,
                    )
                    blocks.append(sec_block)

                    new_context = current_context + [sec_title]
                    self._traverse_node(
                        child,
                        blocks,
                        relations,
                        parent_id=block_id,
                        level=level + 1,
                        context_path=new_context,
                    )

                elif block_type in ("paragraph", "note", "warning", "caution"):
                    text = "".join(child.itertext()).strip()
                    if not text:
                        continue

                    prefix = rule.get("prefix", "")
                    if "prefix_template" in rule:
                        prefix = rule["prefix_template"].format(TAG=tag.upper()) + " "

                    block_id = str(uuid.uuid4())

                    blocks.append(
                        ParsedBlock(
                            page=1,
                            block_type=block_type,
                            text=f"{prefix}{text}",
                            parent_id=parent_id,
                            context_path=current_context,
                            block_id=block_id,
                        )
                    )

                    # JATS style xref handling
                    for xref in child.findall(".//xref"):
                        ref_id = xref.get("rid")
                        ref_type = xref.get("ref-type")
                        if ref_id and ref_type == "bibr":
                            relations.append(
                                {
                                    "sourceBlockId": block_id,
                                    "relationType": "CITES",
                                    "target": f"#{ref_id}",
                                    "citationText": xref.text or "",
                                    "confidence": 0.9,
                                }
                            )

                elif block_type == "procedureStep":
                    self._parse_procedural_step(
                        child, blocks, relations, parent_id, current_context, level
                    )

                elif block_type in ("table", "equation"):
                    text = "".join(child.itertext()).strip()
                    kwargs = (
                        {"equation_data": {"latex": text}}
                        if block_type == "equation"
                        else {}
                    )
                    blocks.append(
                        ParsedBlock(
                            page=1,
                            block_type=block_type,
                            text=text,
                            parent_id=parent_id,
                            context_path=current_context,
                            block_id=str(uuid.uuid4()),
                            **kwargs,
                        )
                    )
            else:
                # Rule not found, recursively traverse if not a leaf node with text (or depending on strategy)
                if len(child) > 0:
                    self._traverse_node(
                        child,
                        blocks,
                        relations,
                        parent_id=parent_id,
                        level=level,
                        context_path=current_context,
                    )

    def _parse_procedural_step(
        self,
        step_el: Any,
        list_of_blocks: List[ParsedBlock],
        relations: List[Dict],
        parent_id: Optional[str],
        context_path: List[str],
        level: int,
    ):
        # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-branches,too-many-statements
        """절차 단계 파싱 (S1000D 특화 로직 유지, 중첩 및 분리 파싱)"""
        text_parts = []
        structured_content: Dict[str, List] = {
            "conditions": [],
            "actions": [],
            "acceptanceCriteria": [],
        }

        step_id = str(uuid.uuid4())
        nested_steps = []
        notes = []

        for child in step_el:
            if not isinstance(child.tag, str):
                continue
            tag = child.tag.lower()

            if tag == "note":
                note_text = " ".join([t.strip() for t in child.itertext() if t.strip()])
                if note_text:
                    notes.append(
                        ParsedBlock(
                            page=1,
                            block_type="note",
                            text=note_text,
                            parent_id=step_id,
                            context_path=context_path,
                            block_id=str(uuid.uuid4()),
                        )
                    )
            elif tag in ("warning", "caution"):
                warn_text = " ".join([t.strip() for t in child.itertext() if t.strip()])
                if warn_text:
                    notes.append(
                        ParsedBlock(
                            page=1,
                            block_type=tag,
                            text=warn_text,
                            parent_id=step_id,
                            context_path=context_path,
                            block_id=str(uuid.uuid4()),
                        )
                    )
            elif tag == "proceduralstep":
                nested_steps.append(child)
            elif tag == "para":
                text_parts.append(
                    " ".join([t.strip() for t in child.itertext() if t.strip()])
                )
            elif tag == "reqcondno":
                text = " ".join([t.strip() for t in child.itertext() if t.strip()])
                structured_content["conditions"].append(
                    {
                        "type": "required_condition",
                        "identifier": child.get("id", ""),
                        "description": text,
                    }
                )
            elif tag == "supportequipdescr":
                text = " ".join([t.strip() for t in child.itertext() if t.strip()])
                structured_content["conditions"].append(
                    {"type": "support_equipment", "description": text}
                )
            elif tag == "torque":
                val = child.findtext("torqueValue")
                unit = child.findtext("torqueUnit")
                if val and unit:
                    structured_content["actions"].append(
                        {
                            "actionType": "tighten",
                            "target": "bolt/fastener",
                            "parameters": {
                                "torqueValue": float(val),
                                "torqueUnit": unit,
                            },
                        }
                    )
                    text_parts.append(f"Torque: {val} {unit}")
            else:
                text_parts.append(
                    " ".join([t.strip() for t in child.itertext() if t.strip()])
                )

        list_of_blocks.extend(notes)

        full_text = " ".join([p for p in text_parts if p]).strip()

        applic_ref = step_el.get("applicRefId")
        if applic_ref:
            structured_content["applicRefId"] = applic_ref

        has_substance = (
            bool(full_text)
            or any(
                v for k, v in structured_content.items() if isinstance(v, list) and v
            )
            or nested_steps
        )

        # Only create step block if it has substantial content or nested steps
        if has_substance:
            step_block = ParsedBlock(
                page=1,
                block_type="procedureStep",
                text=full_text,
                parent_id=parent_id,
                context_path=context_path,
                block_id=step_id,
                level=level,
            )
            step_block.structured_content = structured_content

            list_of_blocks.append(step_block)
        else:
            # Empty shell -> attach notes directly to parent
            for n in notes:
                n.parent_id = parent_id

        for ns in nested_steps:
            self._parse_procedural_step(
                ns, list_of_blocks, relations, step_id, context_path, level + 1
            )

    def _parse_rqmts(self, el: Any) -> Dict[str, List]:
        conditions = []
        for child in el.iter():
            if not isinstance(child.tag, str):
                continue
            tag = child.tag.lower()
            if tag == "noconds":
                conditions.append({"type": "condition", "status": "none"})
            elif tag == "nopersonnel":
                conditions.append({"type": "personnel", "status": "none"})
            elif tag == "nosupportequips":
                conditions.append({"type": "support_equipment", "status": "none"})
            elif tag == "nosupplies":
                conditions.append({"type": "supplies", "status": "none"})
            elif tag == "nospares":
                conditions.append({"type": "spares", "status": "none"})
            elif tag == "nosafety":
                conditions.append({"type": "safety", "status": "none"})
        return {"conditions": conditions, "actions": [], "acceptanceCriteria": []}


class GenericXmlStrategy:
    """루트 태그를 알 수 없는 범용 XML 파싱 전략 (모든 텍스트 노드 추출)"""

    def parse(self, root: Any, source_path: str) -> ParsedDocument:
        blocks: List[ParsedBlock] = []

        text = "".join(root.itertext()).strip()
        if text:
            # 단순히 전체 텍스트를 하나의 문단으로 처리
            blocks.append(
                ParsedBlock(
                    page=1,
                    block_type="paragraph",
                    text=text,
                    level=999,
                    block_id=str(uuid.uuid4()),
                )
            )

        return ParsedDocument(
            source_path=source_path,
            blocks=blocks,
            metadata={"parser": "generic_xml", "version": "1.0.0"},
        )
