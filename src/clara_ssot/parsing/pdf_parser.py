# src/clara_ssot/parsing/pdf_parser.py
import io
import logging
import os
import re
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# PyMuPDF 임포트
import pymupdf
from PIL import Image

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
            "page": self.page,
        }


@dataclass
class ParsedBlock:
    page: int
    block_type: str
    text: Optional[str] = None
    bbox: Optional[BoundingBox] = None
    table_data: Optional[Dict] = None
    confidence: float = 1.0
    # 계층 구조 및 메타데이터 상속 필드 추가
    level: int = 999  # 0: Title, 1: Section, ... 999: Paragraph
    context_path: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    block_id: Optional[str] = None


@dataclass
class ParsedDocument:
    source_path: str
    blocks: List[ParsedBlock]
    metadata: Dict = None


class PyMuPDFParser:
    """
    기본 파서: 텍스트 기반 PDF 처리 (빠름, 정확, 무료)
    스택 기반 알고리즘으로 문서의 계층 구조(Hierarchy)를 복원하고 메타데이터를 상속함.
    """

    def parse(self, pdf_path: Path) -> ParsedDocument:
        doc = pymupdf.open(pdf_path)
        blocks = []

        # 0. 전처리: 문서 전체의 폰트 통계 분석 (본문 폰트 크기 추정)
        font_sizes = []
        for page in doc:
            blocks_raw = page.get_text("dict")["blocks"]
            for b in blocks_raw:
                if b["type"] == 0:  # text
                    for line in b["lines"]:
                        for span in line["spans"]:
                            if span["text"].strip():
                                font_sizes.append(round(span["size"], 1))

        # 가장 빈번한 폰트 크기를 본문 크기로 간주
        if font_sizes:
            body_font_size = Counter(font_sizes).most_common(1)[0][0]
        else:
            body_font_size = 10.0  # 기본값

        logger.info(f"Detected body font size: {body_font_size}pt")

        # 계층 구조 추적을 위한 스택
        # 구조: {'level': int, 'id': str, 'title': str}
        context_stack = []

        for page_index, page in enumerate(doc):
            # 폰트 정보를 얻기 위해 "dict" 모드 사용
            page_dict = page.get_text("dict")

            for block in page_dict.get("blocks", []):
                if block["type"] != 0:  # 0: text, 1: image
                    continue

                # 블록 내 텍스트 병합 및 스타일 대표값 추출
                block_text_parts = []
                max_font_size = 0.0
                is_bold = False

                for line in block["lines"]:
                    for span in line["spans"]:
                        block_text_parts.append(span["text"])
                        if span["size"] > max_font_size:
                            max_font_size = span["size"]
                        # PyMuPDF flags: 2^4 (16) is bold
                        if span["flags"] & 16:
                            is_bold = True

                clean_text = " ".join(block_text_parts).strip()
                if not clean_text:
                    continue

                # 1. 레벨 및 타입 판별 (Dynamic Heuristic)
                level, inferred_type = self._determine_structure(
                    clean_text, max_font_size, is_bold, body_font_size
                )
                block_id = str(uuid.uuid4())

                # 2. 스택 조정 (Pop): 현재 레벨보다 깊거나 같은 이전 섹션 닫기
                while context_stack and context_stack[-1]["level"] >= level:
                    context_stack.pop()

                # 3. 부모 연결 및 컨텍스트 상속
                parent_id = context_stack[-1]["id"] if context_stack else None
                current_context_path = [item["title"] for item in context_stack]

                # 4. 블록 생성
                blocks.append(
                    ParsedBlock(
                        page=page_index + 1,
                        block_type=inferred_type,
                        text=clean_text,
                        bbox=BoundingBox(
                            x0=block["bbox"][0],
                            y0=block["bbox"][1],
                            x1=block["bbox"][2],
                            y1=block["bbox"][3],
                            page=page_index + 1,
                        ),
                        confidence=1.0,
                        level=level,
                        context_path=current_context_path,
                        parent_id=parent_id,
                        block_id=block_id,
                    )
                )

                # 5. 스택 푸시 (Push): 섹션인 경우 스택에 추가하여 하위 블록의 부모가 됨
                if level < 999:
                    context_stack.append(
                        {"level": level, "id": block_id, "title": clean_text}
                    )

        doc.close()

        return ParsedDocument(
            source_path=str(pdf_path),
            blocks=blocks,
            metadata={"parser": "pymupdf_dynamic_stack", "version": "2.1.0"},
        )

    def _determine_structure(
        self, text: str, font_size: float, is_bold: bool, body_size: float
    ) -> Tuple[int, str]:
        """텍스트 패턴과 폰트 스타일로 레벨과 타입을 결정"""
        if re.match(
            r"^\s*(목\s*차|table of contents|contents|abstract|introduction|서\s*론)\s*$",
            text,
            re.IGNORECASE,
        ):
            return 0, "title"

        match = re.match(r"^(\d+(?:\.\d+)*)\.?\s+\w+", text)
        if match:
            depth = match.group(1).count(".") + 1
            return depth, "section"

        if font_size > body_size * 1.2:
            if font_size > body_size * 1.5:
                return 1, "section"
            return 2, "section"

        if is_bold and font_size > body_size * 1.05:
            return 3, "section"

        return 999, "paragraph"


class DoclingParser:
    """
    메인 파서: Docling 기반 (표 + 레이아웃 + 계층 구조 전문)
    Docling의 구조 분석 능력을 활용하여 context_path를 자동으로 생성함.
    """

    def __init__(self):
        # Docling Lazy Import (의존성 없을 시 Fallback 유도)
        try:
            import torch
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import (
                AcceleratorDevice,
                AcceleratorOptions,
                PdfPipelineOptions,
            )
            from docling.document_converter import DocumentConverter, PdfFormatOption

            if torch.cuda.is_available():
                logger.info(
                    f"🚀 GPU detected (CUDA: {torch.cuda.get_device_name(0)}). Using CUDA for Docling."
                )
                device = AcceleratorDevice.CUDA
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                logger.info("🚀 GPU detected (Apple MPS). Using MPS for Docling.")
                device = getattr(AcceleratorDevice, "MPS", AcceleratorDevice.CPU)
            else:
                logger.info(
                    "ℹ️ GPU not detected (CUDA/MPS unavailable). Using CPU for Docling."
                )
                device = AcceleratorDevice.CPU

            pipeline_options = PdfPipelineOptions()
            pipeline_options.accelerator_options = AcceleratorOptions(
                num_threads=4, device=device
            )

            self.converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )

            try:
                import cv2  # noqa: F401

                self.converter.format_to_options[
                    InputFormat.PDF
                ].pipeline_options.do_table_structure = True
            except ImportError:
                logger.warning("OpenCV(cv2) 없음. 표 구조 추출 기능이 제한될 수 있습니다.")
                self.converter.format_to_options[
                    InputFormat.PDF
                ].pipeline_options.do_table_structure = False

        except ImportError as e:
            raise ImportError(f"Docling 라이브러리가 설치되지 않았습니다: {e}")

    def parse(self, pdf_path: Path) -> ParsedDocument:
        result = self.converter.convert(pdf_path)
        doc = result.document
        blocks = []

        context_stack = []

        for item, level in doc.iterate_items():
            label = str(getattr(item, "label", "")).lower()
            text = getattr(item, "text", "").strip()

            if not text and "table" not in label:
                continue

            block_type = "paragraph"
            if "title" in label:
                block_type = "title"
            elif "header" in label:
                block_type = "section"
            elif "table" in label:
                block_type = "table"
            elif "list" in label:
                block_type = "list"

            if block_type in ["title", "section"] and level is not None:
                while context_stack and context_stack[-1]["level"] >= level:
                    context_stack.pop()

            parent_id = context_stack[-1]["id"] if context_stack else None
            current_context_path = [item["title"] for item in context_stack]
            block_id = str(uuid.uuid4())

            bbox = self._extract_bbox(item)

            parsed_block = ParsedBlock(
                page=item.prov[0].page_no if hasattr(item, "prov") and item.prov else 1,
                block_type=block_type,
                text=text,
                bbox=bbox,
                confidence=1.0,
                level=level if level is not None else 999,
                context_path=current_context_path,
                parent_id=parent_id,
                block_id=block_id,
            )

            if block_type == "table" and hasattr(item, "export_to_dataframe"):
                try:
                    df = item.export_to_dataframe()
                    parsed_block.table_data = {
                        "headers": [str(h) for h in df.columns.tolist()],
                        "rows": [[str(c) for c in row] for row in df.values.tolist()],
                    }
                    parsed_block.text = df.to_markdown(index=False)
                except Exception:
                    pass

            blocks.append(parsed_block)

            if block_type in ["title", "section"] and level is not None:
                context_stack.append({"level": level, "id": block_id, "title": text})

        return ParsedDocument(
            source_path=str(pdf_path),
            blocks=blocks,
            metadata={"parser": "docling", "version": "2.0.0"},
        )

    def _extract_bbox(self, item) -> Optional[BoundingBox]:
        if hasattr(item, "prov") and item.prov:
            p = item.prov[0]
            b = p.bbox
            return BoundingBox(
                x0=getattr(b, "l", 0),
                y0=getattr(b, "b", 0),
                x1=getattr(b, "r", 0),
                y1=getattr(b, "t", 0),
                page=p.page_no,
            )
        return None


class GeminiVisionParser:
    """
    백업 파서: 스캔된 문서나 복잡한 표 처리를 위한 VLM (Vision-Language Model)
    Gemini 1.5 Flash를 사용하여 이미지에서 구조화된 데이터를 추출
    """

    def __init__(self, api_key: str = None):
        from google import genai  # pylint: disable=no-name-in-module

        self.api_key = (
            api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        )
        if not self.api_key:
            raise ValueError("Gemini API Key is missing for Vision Parser.")

        self.client = genai.Client(api_key=self.api_key)
        self.model_name = "gemini-1.5-flash"

    # 수정 1: max_pages 파라미터 추가로 하드코딩 제거
    def parse(self, pdf_path: Path, max_pages: Optional[int] = None) -> ParsedDocument:
        """PDF를 이미지로 변환 후 Gemini에게 구조화 요청"""
        doc = pymupdf.open(pdf_path)
        blocks = []

        for page_index, page in enumerate(doc):
            if max_pages is not None and page_index >= max_pages:
                break

            pix = page.get_pixmap(dpi=150)
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))

            prompt = "Extract all text from this page. Return raw text."

            response = self.client.models.generate_content(
                model=self.model_name, contents=[prompt, image]
            )

            blocks.append(
                ParsedBlock(
                    page=page_index + 1,
                    block_type="paragraph",
                    text=response.text,
                    confidence=0.8,
                )
            )

        doc.close()

        return ParsedDocument(
            source_path=str(pdf_path),
            blocks=blocks,
            metadata={"parser": "gemini_vision", "version": "1.0.0"},
        )


# 수정 3: Docling 인스턴스 지연 초기화 (싱글톤 패턴 응용)를 위한 전역 변수
_DOCLING_PARSER_INSTANCE = None

def get_docling_parser() -> DoclingParser:
    global _DOCLING_PARSER_INSTANCE
    if _DOCLING_PARSER_INSTANCE is None:
        _DOCLING_PARSER_INSTANCE = DoclingParser()
    return _DOCLING_PARSER_INSTANCE


def parse_pdf(path: Path, max_vision_pages: Optional[int] = None) -> ParsedDocument:
    """
    하이브리드 파싱 전략: Docling (최우선) -> PyMuPDF (백업) -> Gemini Vision (스캔본)

    1. Docling 시도: 표, 레이아웃, 계층 구조 완벽 지원
    2. 실패 시 PyMuPDF: 빠르고 안정적인 텍스트 추출 (스택 기반 구조화 적용)
    3. 텍스트가 없거나 깨진 경우(스캔 문서) Gemini Vision으로 전환 (강력함, 비용 발생)
    """
    logger.info(f"Parsing PDF with Hybrid Strategy (PyMuPDF + Gemini): {path}")

    try:
        # 1. 텍스트 밀도 체크 (Digital PDF vs Scanned PDF 판별)
        doc = pymupdf.open(path)
        total_text_len = 0
        for page in doc:
            total_text_len += len(page.get_text())

        is_scanned_document = (len(doc) > 0) and (total_text_len / len(doc) < 50)
        doc.close()

        if not is_scanned_document:
            # 1순위: Docling
            try:
                logger.info("🚀 Docling 파서 시도 (표/구조 최적화)")
                # 수정 3 적용: 함수 호출 시마다 인스턴스를 만들지 않고 재사용합니다.
                parser = get_docling_parser()
                return parser.parse(path)
            except Exception as e:
                logger.warning(f"⚠️ Docling 실패 ({e}). PyMuPDF로 전환합니다.")
                # 2순위: PyMuPDF
                parser = PyMuPDFParser()
                return parser.parse(path)
        else:
            logger.info("🖼️ Scanned PDF 감지: Gemini Vision(VLM) 사용")
            # 수정 2: 불필요한 API 키 체크 로직 삭제. 
            # API 키가 없으면 GeminiVisionParser에서 알아서 ValueError를 발생시키고,
            # 아래의 예외 처리(except Exception as e:)가 잡아서 PyMuPDF로 넘겨줍니다.
            parser = GeminiVisionParser()
            return parser.parse(path, max_pages=max_vision_pages)

    except Exception as e:
        logger.warning(f"⚠️ 파싱 중 에러 발생 ({e}). PyMuPDF Fallback 모드로 전환합니다.")
        fallback_parser = PyMuPDFParser()
        return fallback_parser.parse(path)