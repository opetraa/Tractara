# src/clara_ssot/parsing/pdf_parser.py
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# PyMuPDF 임포트
import pymupdf

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """PDF 좌표 정보"""
    x0: float
    y0: float
    x1: float
    y1: float
    page: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
            "page": self.page
        }


@dataclass
class ParsedBlock:
    page: int
    block_type: str
    text: Optional[str] = None
    bbox: Optional[BoundingBox] = None
    table_data: Optional[Dict] = None
    confidence: float = 1.0


@dataclass
class ParsedDocument:
    source_path: str
    blocks: List[ParsedBlock]
    metadata: Dict = None


class PyMuPDFParser:
    """
    Docling(AI 파서)을 사용할 수 없는 환경(CUDA 에러 등)에서
    기본적인 텍스트 추출을 수행하는 Fallback 파서
    """

    def parse(self, pdf_path: Path) -> ParsedDocument:
        doc = pymupdf.open(pdf_path)
        blocks = []

        for page_index, page in enumerate(doc):
            # get_text("blocks") returns list of (x0, y0, x1, y1, "lines", block_no, block_type)
            raw_blocks = page.get_text("blocks")

            for b in raw_blocks:
                x0, y0, x1, y1, text, block_no, block_type = b

                # block_type 0 is text, 1 is image
                if block_type != 0 or not text.strip():
                    continue

                blocks.append(ParsedBlock(
                    page=page_index + 1,
                    block_type="paragraph",
                    text=text.strip(),
                    bbox=BoundingBox(x0=x0, y0=y0, x1=x1,
                                     y1=y1, page=page_index + 1),
                    confidence=0.5  # Rule-based라 신뢰도는 낮게 설정
                ))

        doc.close()

        return ParsedDocument(
            source_path=str(pdf_path),
            blocks=blocks,
            metadata={"parser": "pymupdf_fallback", "version": "1.0.0"}
        )


class DoclingParser:
    """Docling 기반 파서 (표 + 레이아웃 전문)"""

    def __init__(self):
        # 0) 필수 의존성 체크 (Torch)
        try:
            import torch  # noqa: F401
        except ImportError as e:
            # CUDA 라이브러리 누락 에러 핸들링 (libcusparse.so.12 등)
            if "libcusparse.so" in str(e) or "libcublas.so" in str(e):
                msg = (
                    "❌ PyTorch CUDA 라이브러리 로드 실패 (libcusparse/libcublas).\n"
                    "현재 환경에 GPU 라이브러리가 없거나 호환되지 않습니다.\n"
                    "CPU 환경이라면 다음 명령어로 PyTorch를 CPU 버전으로 재설치하세요:\n"
                    "👉 pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --force-reinstall"
                )
                logger.error(msg)
                raise ImportError(msg) from e

            msg = "PyTorch(torch)가 설치되지 않았습니다. 'make install'을 실행하여 의존성을 설치해주세요."
            logger.error(msg)
            raise ImportError(msg) from e

        # Docling Lazy Import (CUDA 라이브러리 에러 방지)
        try:
            from docling.document_converter import DocumentConverter
            from docling.datamodel.base_models import InputFormat
        except ImportError as e:
            # CUDA 라이브러리 누락 에러 핸들링 (libcusparse.so.12 등)
            if "libcusparse.so" in str(e) or "libcublas.so" in str(e):
                msg = (
                    "❌ PyTorch CUDA 라이브러리 로드 실패 (libcusparse/libcublas).\n"
                    "현재 환경에 GPU 라이브러리가 없거나 호환되지 않습니다.\n"
                    "CPU 환경이라면 다음 명령어로 PyTorch를 CPU 버전으로 재설치하세요:\n"
                    "👉 pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --force-reinstall"
                )
                logger.error(msg)
                raise ImportError(msg) from e
            raise e

        # TableFormer 활성화 옵션
        self.converter = DocumentConverter()

        # 기본적으로 표 구조 추출은 끄고 시작 (안전 모드)
        self.converter.format_to_options[InputFormat.PDF].pipeline_options.do_table_structure = False

        # OpenCV(libGL) 종속성 체크: 환경에 라이브러리가 없으면 표 추출 비활성화
        try:
            import cv2  # noqa: F401
            self.converter.format_to_options[InputFormat.PDF].pipeline_options.do_table_structure = True
        except ImportError:
            logger.warning(
                "OpenCV(cv2) 로드 실패. libGL.so.1 누락으로 인해 표 구조 추출(TableFormer)을 비활성화합니다.")

    def parse(self, pdf_path: Path) -> ParsedDocument:
        """Docling으로 PDF 파싱"""
        result = self.converter.convert(pdf_path)
        doc = result.document

        blocks = []

        # DoclingDocument 순회
        for item, level in doc.iterate_items():
            # Docling v2 item has label (Enum), not type. Converting to string for comparison.
            label = str(getattr(item, "label", "")).lower()

            # 텍스트 블록
            if any(x in label for x in ["text", "header", "paragraph", "title", "list_item", "caption", "footnote", "form"]):
                bbox = self._extract_bbox(item)
                page = item.prov[0].page_no if hasattr(
                    item, "prov") and item.prov else 1
                blocks.append(ParsedBlock(
                    page=page,
                    block_type="paragraph",
                    text=getattr(item, "text", ""),
                    bbox=bbox,
                    confidence=1.0
                ))

            # 표 블록
            elif "table" in label:
                bbox = self._extract_bbox(item)
                table_data = self._extract_table_data(item)
                page = item.prov[0].page_no if hasattr(
                    item, "prov") and item.prov else 1
                blocks.append(ParsedBlock(
                    page=page,
                    block_type="table",
                    text=self._table_to_markdown(table_data),
                    bbox=bbox,
                    table_data=table_data,
                    confidence=0.979  # TableFormer 평균
                ))

        return ParsedDocument(
            source_path=str(pdf_path),
            blocks=blocks,
            metadata={"parser": "docling", "version": "2.0.0"}
        )

    def _extract_bbox(self, item) -> Optional[BoundingBox]:
        """Docling item에서 bbox 추출"""
        # v2: prov list contains bbox
        if hasattr(item, "prov") and item.prov:
            p = item.prov[0]
            b = p.bbox
            # Docling uses l,r,t,b. Mapping to x0,y0,x1,y1.
            return BoundingBox(
                x0=getattr(b, "l", 0), y0=getattr(b, "b", 0),
                x1=getattr(b, "r", 0), y1=getattr(b, "t", 0),
                page=p.page_no
            )
        return None

    def _extract_table_data(self, table_item) -> Dict:
        """표 데이터 추출"""
        # Docling v2: export_to_dataframe() 사용
        if hasattr(table_item, "export_to_dataframe"):
            try:
                df = table_item.export_to_dataframe()
                return {
                    "headers": [str(h) for h in df.columns.tolist()],
                    "rows": [[str(c) for c in row] for row in df.values.tolist()]
                }
            except Exception as e:
                logger.warning(f"Table export failed: {e}")

        return {"headers": [], "rows": []}

    def _table_to_markdown(self, table_data: Dict) -> str:
        """표를 마크다운으로 변환"""
        if not table_data.get("rows"):
            return "[Empty Table]"

        md = []
        if table_data.get("headers"):
            md.append("| " + " | ".join(table_data["headers"]) + " |")
            md.append("| " + " | ".join(["---"] *
                      len(table_data["headers"])) + " |")

        for row in table_data["rows"]:
            md.append("| " + " | ".join(str(cell) for cell in row) + " |")

        return "\n".join(md)


class PyMuPDFCoordinateExtractor:
    """PyMuPDF로 정밀 좌표 추출 (보조 엔진)"""

    def enhance_with_coordinates(
        self,
        pdf_path: Path,
        docling_blocks: List[ParsedBlock]
    ) -> List[ParsedBlock]:
        """
        Docling 결과에 PyMuPDF의 정밀 좌표를 보강
        (Docling bbox가 부정확하거나 누락된 경우 대비)
        """
        doc = pymupdf.open(pdf_path)

        for block in docling_blocks:
            if block.bbox is None:
                # Docling에서 bbox를 찾지 못한 경우
                page = doc[block.page - 1]  # PyMuPDF는 0-based

                # 텍스트로 좌표 검색
                if block.text:
                    instances = page.search_for(block.text[:50])  # 앞 50자로 검색
                    if instances:
                        rect = instances[0]
                        block.bbox = BoundingBox(
                            x0=rect.x0, y0=rect.y0,
                            x1=rect.x1, y1=rect.y1,
                            page=block.page
                        )

        doc.close()
        return docling_blocks


def parse_pdf(path: Path) -> ParsedDocument:
    """
    첫 번째 문서 전략: Docling + PyMuPDF 멀티엔진

    1단계: Docling으로 표 + 레이아웃 추출 (TableFormer 97.9% 정확도)
    2단계: PyMuPDF로 좌표 보강 (누락 방지)
    3단계: ParsedDocument 반환
    """
    logger.info(f"Parsing PDF with Docling+PyMuPDF: {path}")

    try:
        # 1) Docling 파싱 (AI 기반, GPU 권장되나 CPU도 가능해야 함)
        docling_parser = DoclingParser()
        parsed = docling_parser.parse(path)

        # 2) PyMuPDF 좌표 보강
        coord_extractor = PyMuPDFCoordinateExtractor()
        parsed.blocks = coord_extractor.enhance_with_coordinates(
            path, parsed.blocks)

        logger.info(f"Parsed {len(parsed.blocks)} blocks from {path}")
        return parsed

    except (ImportError, Exception) as e:
        # 3) Fallback: 환경 문제로 Docling 실패 시 PyMuPDF 단독 사용
        logger.warning(
            f"⚠️ Docling AI 파서 실행 실패 ({e}).\n"
            "범용 호환성을 위해 PyMuPDF Fallback 모드로 전환합니다."
        )
        fallback_parser = PyMuPDFParser()
        return fallback_parser.parse(path)
