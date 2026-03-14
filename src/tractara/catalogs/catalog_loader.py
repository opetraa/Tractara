"""XML 매핑 카탈로그 로더 모듈."""
# src/tractara/catalogs/catalog_loader.py
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

_CATALOG_DIR = Path(__file__).parent
_LOADED_CATALOGS: Dict[str, Dict[str, Any]] = {}
_BASE_CATALOG: Dict[str, Any] = {}


def load_all_catalogs() -> None:
    """_CATALOG_DIR 내의 모든 yaml 파일을 로드하여 메모리에 저장합니다."""

    if _LOADED_CATALOGS:
        return  # Already loaded

    for yaml_file in _CATALOG_DIR.glob("*.yaml"):
        try:
            with open(yaml_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if not data or "format_id" not in data:
                continue

            format_id = data["format_id"]
            if format_id == "_base":
                _BASE_CATALOG.clear()
                _BASE_CATALOG.update(data)
            else:
                _LOADED_CATALOGS[format_id] = data

        except (yaml.YAMLError, OSError) as e:
            logger.error("Failed to load catalog %s: %s", yaml_file.name, e)


def get_base_catalog() -> Dict[str, Any]:
    """공통 _base 카탈로그를 반환합니다."""
    if not _BASE_CATALOG:
        load_all_catalogs()
    return _BASE_CATALOG


def detect_catalog(root_tag: str) -> Optional[Dict[str, Any]]:
    """루트 태그 문자열에 매칭되는 카탈로그를 찾아 반환합니다."""
    if not _LOADED_CATALOGS:
        load_all_catalogs()

    root_tag_lower = root_tag.lower()

    for _, config in _LOADED_CATALOGS.items():
        detect_cfg = config.get("detect", {})
        root_contains = detect_cfg.get("root_tag_contains")

        if not root_contains:
            continue

        if isinstance(root_contains, str):
            if root_contains in root_tag_lower:
                return config
        elif isinstance(root_contains, list):
            if any(key in root_tag_lower for key in root_contains):
                return config

    return None
