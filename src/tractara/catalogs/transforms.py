"""XML 변환 지원 함수 레지스트리."""
# src/tractara/catalogs/transforms.py
from typing import Any, Dict, List, Optional


def _assemble_dmc_identifier(element: Any) -> List[Dict[str, str]]:
    """dmIdent, dmRef 또는 dmCode 속성들을 DMC URI로 조립합니다."""
    dm_code = element if element.tag.endswith("dmCode") else element.find(".//dmCode")
    if dm_code is None:
        dm_code = element

    model = dm_code.get("modelIdentCode", "")
    system = dm_code.get("systemDiffCode", "")
    sys_code = dm_code.get("systemCode", "")
    subsys = dm_code.get("subSystemCode", "")
    subsubsys = dm_code.get("subSubSystemCode", "")
    assy = dm_code.get("assyCode", "")
    disassy = dm_code.get("disassyCode", "")
    disassy_var = dm_code.get("disassyCodeVariant", "")
    info = dm_code.get("infoCode", "")
    info_var = dm_code.get("infoCodeVariant", "")
    item_loc = dm_code.get("itemLocationCode", "")

    dmc = (
        f"DMC-{model}-{system}-{sys_code}-{subsys}{subsubsys}-{assy}-{disassy}{disassy_var}"
        f"-{info}{info_var}-{item_loc}"
    )

    # issueInfo 추가
    issue_info = None
    if not element.tag.endswith("dmCode"):
        issue_info = element.find(".//issueInfo")
    if issue_info is None and element.getparent() is not None:
        issue_info = element.getparent().find(".//issueInfo")

    if issue_info is not None:
        iss_num = issue_info.get("issueNumber", "")
        in_work = issue_info.get("inWork", "")
        if iss_num and in_work:
            dmc += f"_{iss_num}-{in_work}"

    return [{"scheme": "URI", "value": dmc}]


def _dmc_from_dmref(element: Any) -> Optional[str]:
    """dmRef 요소에서 대상 DMC 문자열을 반환합니다."""
    res = _assemble_dmc_identifier(element)
    if res:
        return res[0]["value"]
    return None


def _date_from_element_attrs(element: Any) -> Optional[str]:
    """year, month, day 속성에서 YYYY-MM-DD 날짜를 추출합니다."""
    year = element.get("year", "")
    if not year:
        return None
    month = element.get("month", "01").zfill(2)
    day = element.get("day", "01").zfill(2)
    return f"{year}-{month}-{day}"


def _extract_text_content(element: Any) -> Optional[str]:
    """요소의 전체 텍스트를 추출합니다."""
    text = "".join(element.itertext()).strip()
    return text if text else None


def _extract_jats_author_name(element: Any) -> List[Dict[str, str]]:
    """JATS contrib 요소에서 저자 이름을 추출합니다."""
    # List elements directly passed, transform handles one or list, but let us handle if it is the element itself or a list of elements.
    # We will assume `element` is a single element matching the xpath
    surname = element.findtext(".//surname") or ""
    given = element.findtext(".//given-names") or ""
    name = f"{given} {surname}".strip()
    if name:
        return [{"name": name, "entityType": "person"}]
    return []


def _jats_author_names_list(elements: List[Any]) -> List[Dict[str, str]]:
    """JATS contrib 요소 리스트에서 저자 이름들을 추출합니다."""
    creators = []
    for el in elements:
        res = _extract_jats_author_name(el)
        if res:
            creators.extend(res)
    return creators


def _join_elements_text(elements: List[Any]) -> Optional[str]:
    """여러 요소의 텍스트를 하나로 합칩니다."""
    texts = ["".join(e.itertext()).strip() for e in elements]
    joined = "; ".join([t for t in texts if t])
    return joined if joined else None


def _s1000d_security(element: Any) -> Dict[str, str]:
    """보안 등급, 상업 등급, 주의사항을 종합해 accessRights와 상세 정보를 반환합니다."""
    sec_class = element.get("securityClassification", "")
    comm_class = element.get("commercialClassification", "")
    caveat = element.get("caveat", "")

    access = "public"
    if sec_class != "01" or comm_class or caveat:
        access = "restricted"

    parts = [p for p in (sec_class, comm_class, caveat) if p]
    license_val = "_".join(parts) if parts else "01"

    return {"accessRights": access, "license": license_val}


def _s1000d_info_code_to_type(element: Any) -> str:
    """dmCode의 infoCode를 DOC 스키마 dc_type으로 번역합니다."""
    dm_code = element.find(".//dmCode") if element.tag.endswith("Ident") else element
    if dm_code is None:
        dm_code = element
    info_code = dm_code.get("infoCode", "")

    mapping = {
        "040": "TechnicalReport",
        "520": "Procedure",
        "720": "Procedure",
    }
    return mapping.get(info_code[:3], "Procedure")


TRANSFORM_REGISTRY = {
    "dmc_assemble": _assemble_dmc_identifier,
    "date_from_attrs": _date_from_element_attrs,
    "text_extract": _extract_text_content,
    "jats_author_name": _jats_author_names_list,
    "join_text": _join_elements_text,
    "dmc_from_dmref": _dmc_from_dmref,
    "s1000d_security": _s1000d_security,
    "s1000d_info_code_to_type": _s1000d_info_code_to_type,
}
