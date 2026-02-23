# src/clara_ssot/parsing/pdf_parser.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import os
import re
import uuid
import io
from PIL import Image
from collections import Counter

# PyMuPDF 임포트
import pymupdf

from .section_classifier import (
    SectionClassifier,
    SectionFeatures,
    extract_section_label,
    _normalize,
)

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
    # 계층 구조 필드
    level: int = 999          # 0: Title, 1: Section, 2+: Subsection, 999: Paragraph
    context_path: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    block_id: Optional[str] = None
    # 섹션 메타데이터 (section_classifier 에서 추출)
    section_label: Optional[str] = None   # 예: "1.2.3", "제2장"
    section_title: Optional[str] = None   # 번호 이후 제목 텍스트


@dataclass
class ParsedDocument:
    source_path: str
    blocks: List[ParsedBlock]
    metadata: Dict = None


class PyMuPDFParser:
    """
    텍스트 기반 PDF 파서 (PyMuPDF).

    파이프라인:
      Phase 0 — 문서 레벨 전처리:
        - 본문 폰트 크기 추정
        - PDF 북마크(S급 힌트) 수집
        - ToC 페이지 파싱(S급 힌트) 시도
        - SectionClassifier 초기화
      Phase 1 — 블록 루프:
        - 블록 특징 추출 → SectionClassifier.classify()
        - 스택 기반 parent_id / context_path 추적
    """

    def parse(self, pdf_path: Path) -> ParsedDocument:
        doc = pymupdf.open(pdf_path)
        blocks: List[ParsedBlock] = []

        # ── Phase 0: 문서 레벨 전처리 ─────────────────────────────────────

        # 본문 폰트 크기 추정 (전체 span 폰트 크기 최빈값)
        font_sizes: List[float] = []
        for page in doc:
            for b in page.get_text("dict")["blocks"]:
                if b["type"] == 0:
                    for line in b["lines"]:
                        for span in line["spans"]:
                            if span["text"].strip():
                                font_sizes.append(round(span["size"], 1))

        body_font_size = Counter(font_sizes).most_common(1)[
            0][0] if font_sizes else 10.0
        logger.info(f"Detected body font size: {body_font_size}pt")

        # S급 힌트 1: PDF 북마크
        pdf_bookmarks = doc.get_toc()   # [(level, title, page_no), ...]
        logger.info(f"PDF bookmarks found: {len(pdf_bookmarks)}")

        # S급 힌트 2: ToC 페이지 파싱
        toc_entries = self._extract_toc_entries(doc)
        logger.info(f"ToC entries parsed: {len(toc_entries)}")

        # 분류기 초기화
        classifier = SectionClassifier(
            body_font_size, pdf_bookmarks, toc_entries)

        # ── Phase 1: 블록 루프 ────────────────────────────────────────────

        # 스택: [{"level": int, "id": str, "title": str}]
        context_stack: List[Dict] = []

        for page_index, page in enumerate(doc):
            page_dict = page.get_text("dict")
            page_width = page.rect.width

            for block in page_dict.get("blocks", []):
                if block["type"] != 0:   # 0: text, 1: image
                    continue

                # 블록 특징 추출
                text_parts: List[str] = []
                max_font_size = 0.0
                is_bold = False
                font_name_counter: Counter = Counter()

                for line in block["lines"]:
                    for span in line["spans"]:
                        text_parts.append(span["text"])
                        if span["size"] > max_font_size:
                            max_font_size = span["size"]
                        if span["flags"] & 16:   # bit 4 = bold
                            is_bold = True
                        if span["text"].strip():
                            font_name_counter[span["font"]] += 1

                clean_text = " ".join(text_parts).strip()
                if not clean_text:
                    continue

                dominant_font = (
                    font_name_counter.most_common(1)[0][0]
                    if font_name_counter else ""
                )
                bbox_x0, bbox_y0, bbox_x1, bbox_y1 = block["bbox"]

                # 분류기 호출
                features = SectionFeatures(
                    text=clean_text,
                    max_font_size=max_font_size,
                    is_bold=is_bold,
                    font_name=dominant_font,
                    page_width=page_width,
                    bbox_x0=bbox_x0,
                    bbox_x1=bbox_x1,
                )
                result = classifier.classify(features)

                # 스택 조정: 현재 레벨보다 깊거나 같은 이전 섹션 닫기
                level = result.level
                while context_stack and context_stack[-1]["level"] >= level:
                    context_stack.pop()

                # 부모 연결 및 컨텍스트 경로 수집
                parent_id = context_stack[-1]["id"] if context_stack else None
                current_context_path = [item["title"]
                                        for item in context_stack]
                block_id = str(uuid.uuid4())

                blocks.append(ParsedBlock(
                    page=page_index + 1,
                    block_type=result.block_type,
                    text=clean_text,
                    bbox=BoundingBox(
                        x0=bbox_x0, y0=bbox_y0,
                        x1=bbox_x1, y1=bbox_y1,
                        page=page_index + 1,
                    ),
                    confidence=result.confidence,
                    level=level,
                    context_path=current_context_path,
                    parent_id=parent_id,
                    block_id=block_id,
                    section_label=result.section_label,
                    section_title=result.section_title,
                ))

                # 섹션만 스택에 푸시 (paragraph는 부모가 될 수 없음)
                if level < 999:
                    context_stack.append({
                        "level": level,
                        "id": block_id,
                        "title": clean_text,
                    })

        doc.close()

        return ParsedDocument(
            source_path=str(pdf_path),
            blocks=blocks,
            metadata={"parser": "pymupdf_section_classifier",
                      "version": "3.0.0"},
        )

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────────

    def _extract_toc_entries(self, doc) -> List[Dict]:
        """
        ToC 페이지를 탐지하고 섹션 엔트리를 파싱한다.

        탐지 전략:
          - 첫 15페이지에서 "contents" / "목차" / "table of contents" 키워드 검색
          - 발견된 페이지의 텍스트 라인에서 점선+페이지번호 제거 후 섹션 라벨 추출

        반환: [{"label": "1.2", "title": "Background"}, ...]
        """
        entries: List[Dict] = []

        for page_index, page in enumerate(doc):
            if page_index > 15:
                break

            page_text = page.get_text().strip()
            first_300 = page_text[:300].lower()

            toc_keywords = ("contents", "목차", "table of contents")
            if not any(kw in first_300 for kw in toc_keywords):
                continue

            # ToC 페이지 발견 → 라인별 파싱
            for line in page_text.split("\n"):
                line = line.strip()
                if len(line) < 3:
                    continue

                # 점선 및 끝 페이지 번호 제거
                # 예: "1.2 Background ............. 45" → "1.2 Background"
                cleaned = re.sub(r"[.\s]{3,}\d+\s*$", "", line).strip()
                cleaned = re.sub(r"\.{3,}", "", cleaned).strip()

                if len(cleaned) < 3:
                    continue

                label, title = extract_section_label(cleaned)
                if label:
                    entries.append({"label": label, "title": title or cleaned})

            # 첫 번째 ToC 페이지만 처리
            break

        return entries


class DoclingParser:
    """
    메인 파서: Docling 기반 (표 + 레이아웃 + 계층 구조 전문).

    Docling은 자체적으로 계층 구조를 제공하므로 SectionClassifier를 우회한다.
    section/title 블록에 한해 extract_section_label()로 sectionLabel/sectionTitle을 추출한다.
    """

    def __init__(self):
        try:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import (
                PdfPipelineOptions,
                AcceleratorOptions,
                AcceleratorDevice,
            )
            import torch

            if torch.cuda.is_available():
                logger.info(
                    f"🚀 GPU detected (CUDA: {torch.cuda.get_device_name(0)}). Using CUDA for Docling.")
                device = AcceleratorDevice.CUDA
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                logger.info(
                    "🚀 GPU detected (Apple MPS). Using MPS for Docling.")
                device = getattr(AcceleratorDevice, "MPS",
                                 AcceleratorDevice.CPU)
            else:
                logger.info(
                    "ℹ️ GPU not detected (CUDA/MPS unavailable). Using CPU for Docling.")
                device = AcceleratorDevice.CPU

            pipeline_options = PdfPipelineOptions()
            pipeline_options.accelerator_options = AcceleratorOptions(
                num_threads=4, device=device
            )

            self.converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options)}
            )

            try:
                import cv2  # noqa: F401
                self.converter.format_to_options[InputFormat.PDF].pipeline_options.do_table_structure = True
            except ImportError:
                logger.warning("OpenCV(cv2) 없음. 표 구조 추출 기능이 제한될 수 있습니다.")
                self.converter.format_to_options[InputFormat.PDF].pipeline_options.do_table_structure = False

        except ImportError as e:
            raise ImportError(f"Docling 라이브러리가 설치되지 않았습니다: {e}")

    def parse(self, pdf_path: Path) -> ParsedDocument:
        result = self.converter.convert(pdf_path)
        doc = result.document
        blocks: List[ParsedBlock] = []

        context_stack: List[Dict] = []

        for item, level in doc.iterate_items():
            label = str(getattr(item, "label", "")).lower()
            text = getattr(item, "text", "").strip()

            if not text and "table" not in label:
                continue

            # 타입 매핑
            block_type = "paragraph"
            if "title" in label:
                block_type = "title"
            elif "header" in label:
                block_type = "section"
            elif "table" in label:
                block_type = "table"
            elif "list" in label:
                block_type = "list"

            # 스택 조정 (Docling level이 None인 본문은 스택 유지)
            if block_type in ["title", "section"] and level is not None:
                while context_stack and context_stack[-1]["level"] >= level:
                    context_stack.pop()

            parent_id = context_stack[-1]["id"] if context_stack else None
            current_context_path = [item["title"] for item in context_stack]
            block_id = str(uuid.uuid4())

            # sectionLabel / sectionTitle 추출 (Docling은 classifier 우회)
            sec_label, sec_title = None, None
            if block_type in ["title", "section"]:
                sec_label, sec_title = extract_section_label(text)
                if sec_title is None:
                    sec_title = text

            bbox = self._extract_bbox(item)

            parsed_block = ParsedBlock(
                page=item.prov[0].page_no if hasattr(
                    item, "prov") and item.prov else 1,
                block_type=block_type,
                text=text,
                bbox=bbox,
                confidence=1.0,
                level=level if level is not None else 999,
                context_path=current_context_path,
                parent_id=parent_id,
                block_id=block_id,
                section_label=sec_label,
                section_title=sec_title,
            )

            if block_type == "table" and hasattr(item, "export_to_dataframe"):
                try:
                    df = item.export_to_dataframe()
                    parsed_block.table_data = {
                        "headers": [str(h) for h in df.columns.tolist()],
                        "rows": [[str(c) for c in row] for row in df.values.tolist()]
                    }
                    parsed_block.text = df.to_markdown(index=False)
                except Exception:
                    pass

            blocks.append(parsed_block)

            if block_type in ["title", "section"] and level is not None:
                context_stack.append({
                    "level": level,
                    "id": block_id,
                    "title": text,
                })

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
                x0=getattr(b, "l", 0), y0=getattr(b, "b", 0),
                x1=getattr(b, "r", 0), y1=getattr(b, "t", 0),
                page=p.page_no,
            )
        return None


class GeminiVisionParser:
    """
    백업 파서: 스캔된 문서나 복잡한 표 처리를 위한 VLM (Vision-Language Model).
    gemini-3-flash-preview를 사용하여 이미지에서 구조화된 데이터를 추출.
    """

    def __init__(self, api_key: str = None):
        from google import genai

        self.api_key = api_key or os.getenv(
            "GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API Key is missing for Vision Parser.")

        self.client = genai.Client(api_key=self.api_key)
        self.model_name = "gemini-3-flash-preview"

    def parse(self, pdf_path: Path) -> ParsedDocument:
        """PDF를 이미지로 변환 후 Gemini에게 구조화 요청."""
        doc = pymupdf.open(pdf_path)
        blocks: List[ParsedBlock] = []

        for page_index, page in enumerate(doc):
            if page_index >= 3:
                break

            pix = page.get_pixmap(dpi=150)
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))

            prompt = "Extract all text from this page. Return raw text."
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt, image],
            )

            blocks.append(ParsedBlock(
                page=page_index + 1,
                block_type="paragraph",
                text=response.text,
                confidence=0.8,
            ))

        doc.close()

        return ParsedDocument(
            source_path=str(pdf_path),
            blocks=blocks,
            metadata={"parser": "gemini_vision", "version": "1.0.0"},
        )


def parse_pdf(path: Path) -> ParsedDocument:
    """
    하이브리드 파싱 전략: Docling (최우선) → PyMuPDF → Gemini Vision (스캔본)

    1. Docling: 표, 레이아웃, 계층 구조 완벽 지원 (SectionClassifier 우회)
    2. PyMuPDF: 안정적 텍스트 추출 + SectionClassifier 적용
    3. Gemini Vision: 스캔 문서 전용 (비용 발생)
    """
    logger.info(f"Parsing PDF with Hybrid Strategy: {path}")

    try:
        doc = pymupdf.open(path)
        total_text_len = sum(len(page.get_text()) for page in doc)
        is_scanned = (len(doc) > 0) and (total_text_len / len(doc) < 50)
        doc.close()

        if not is_scanned:
            try:
                logger.info("🚀 Docling 파서 시도 (표/구조 최적화)")
                return DoclingParser().parse(path)
            except Exception as e:
                logger.warning(
                    f"⚠️ Docling 실패 ({e}). PyMuPDF + SectionClassifier로 전환.")
                return PyMuPDFParser().parse(path)
        else:
            logger.info("🖼️ Scanned PDF 감지: Gemini Vision(VLM) 사용")
            if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
                logger.warning("⚠️ Gemini API Key 없음. PyMuPDF로 강제 진행")
                return PyMuPDFParser().parse(path)
            return GeminiVisionParser().parse(path)

    except Exception as e:
        logger.warning(f"⚠️ 파싱 중 에러 ({e}). PyMuPDF fallback 모드.")
        return PyMuPDFParser().parse(path)
