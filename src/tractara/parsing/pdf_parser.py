"""PDF 파싱 모듈: 하이브리드 전략 (Docling → PyMuPDF → Gemini Vision)."""
# src/tractara/parsing/pdf_parser.py
import io
import logging
import os
import re
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# PyMuPDF 임포트
import pymupdf
from PIL import Image

from .section_classifier import (
    SectionClassifier,
    SectionFeatures,
    extract_section_label,
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
        """자신을 Dictionary로 변환합니다."""
        return {
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
            "page": self.page,
        }


@dataclass
class ParsedBlock:
    """단일 파싱 블록."""

    page: int
    block_type: str
    text: Optional[str] = None
    bbox: Optional[BoundingBox] = None
    table_data: Optional[Dict] = None
    equation_data: Optional[Dict] = None
    confidence: float = 1.0
    # 계층 구조 필드
    level: int = 999  # 0: Title, 1: Section, 2+: Subsection, 999: Paragraph
    context_path: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    block_id: Optional[str] = None
    # 섹션 메타데이터 (section_classifier 에서 추출)
    section_label: Optional[str] = None  # 예: "1.2.3", "제2장"
    section_title: Optional[str] = None  # 번호 이후 제목 텍스트


@dataclass
class ParsedDocument:
    """파싱된 전체 문서 컨테이너."""

    source_path: str
    blocks: List[ParsedBlock]
    metadata: Optional[Dict] = None


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

        body_font_size = (
            Counter(font_sizes).most_common(1)[0][0] if font_sizes else 10.0
        )
        logger.info("Detected body font size: %spt", body_font_size)

        # S급 힌트 1: PDF 북마크
        pdf_bookmarks = doc.get_toc()  # [(level, title, page_no), ...]
        logger.info("PDF bookmarks found: %d", len(pdf_bookmarks))

        # S급 힌트 2: ToC 페이지 파싱
        toc_entries = self._extract_toc_entries(doc)
        logger.info("ToC entries parsed: %d", len(toc_entries))

        # 분류기 초기화
        classifier = SectionClassifier(body_font_size, pdf_bookmarks, toc_entries)

        # ── Phase 1: 블록 루프 ────────────────────────────────────────────

        # 스택: [{"level": int, "id": str, "title": str}]
        context_stack: List[Dict] = []

        for page_index, page in enumerate(doc):
            page_dict = page.get_text("dict")
            page_width = page.rect.width

            for block in page_dict.get("blocks", []):
                if block["type"] != 0:  # 0: text, 1: image
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
                        if span["flags"] & 16:  # bit 4 = bold
                            is_bold = True
                        if span["text"].strip():
                            font_name_counter[span["font"]] += 1

                clean_text = " ".join(text_parts).strip()
                if not clean_text:
                    continue

                dominant_font = (
                    font_name_counter.most_common(1)[0][0] if font_name_counter else ""
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

                # 수식 감지 덮어쓰기 (휴리스틱)
                equation_data = None
                if result.block_type == "paragraph" and self._is_equation(clean_text):
                    result.block_type = "equation"

                    # 수식 번호 추출 시도
                    eq_num_match = re.search(r"\((?P<num>\d+(\.\d+)*)\)$", clean_text)
                    eq_num = eq_num_match.group("num") if eq_num_match else None

                    equation_data = {
                        "latex": clean_text,  # 원시 텍스트 보존
                        "equationNumber": eq_num if eq_num else "",
                    }

                # 스택 조정: 현재 레벨보다 깊거나 같은 이전 섹션 닫기
                level = result.level
                while context_stack and context_stack[-1]["level"] >= level:
                    context_stack.pop()

                # 부모 연결 및 컨텍스트 경로 수집
                parent_id = context_stack[-1]["id"] if context_stack else None
                current_context_path = [item["title"] for item in context_stack]
                block_id = str(uuid.uuid4())

                blocks.append(
                    ParsedBlock(
                        page=page_index + 1,
                        block_type=result.block_type,
                        text=clean_text,
                        bbox=BoundingBox(
                            x0=bbox_x0,
                            y0=bbox_y0,
                            x1=bbox_x1,
                            y1=bbox_y1,
                            page=page_index + 1,
                        ),
                        table_data=None,
                        equation_data=equation_data,
                        confidence=result.confidence,
                        level=level,
                        context_path=current_context_path,
                        parent_id=parent_id,
                        block_id=block_id,
                        section_label=result.section_label,
                        section_title=result.section_title,
                    )
                )

                # 섹션만 스택에 푸시 (paragraph나 equation은 부모가 될 수 없음)
                if level < 999:
                    context_stack.append(
                        {
                            "level": level,
                            "id": block_id,
                            "title": clean_text,
                        }
                    )

        doc.close()

        # 후처리 1: 문맥 기반 분할 (인라인 수식 추출)
        blocks = _split_inline_equations(blocks)

        # 후처리 2: 문맥 기반 수식 탐지 및 재분류 (단독 블록 대상)
        blocks = _reclassify_equations(blocks)

        return ParsedDocument(
            source_path=str(pdf_path),
            blocks=blocks,
            metadata={"parser": "pymupdf_section_classifier", "version": "3.0.0"},
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

    def _is_equation(self, text: str) -> bool:
        """수식 여부 휴리스틱 탐지"""
        # 끝이 괄호 번호로 끝나는 경우: "E = mc^2 (1.1)"
        if re.search(r"\(\d+(\.\d+)*\)$", text.strip()):
            return True

        # 수학 기호 비율 확인
        math_symbols = set("∑∫√±αβγδεζηθικλμνξοπρστυφχψωΔΓΘΛΞΠΣΦΨΩ=+-*/<>≤≥≈≠")
        symbol_count = sum(1 for char in text if char in math_symbols)
        # 3개 이상의 수식 기호가 있거나 전체 길이 대비 기호 비율이 10% 이상이면 수식으로 간주 (방정식 등)
        if len(text) > 3 and (symbol_count >= 3 or (symbol_count / len(text)) > 0.1):
            return True

        return False


class DoclingParser:
    """
    메인 파서: Docling 기반 (표 + 레이아웃 + 계층 구조 전문).

    Docling은 자체적으로 계층 구조를 제공하므로 SectionClassifier를 우회한다.
    section/title 블록에 한해 extract_section_label()로 sectionLabel/sectionTitle을 추출한다.
    """

    def __init__(self):
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
                    "🚀 GPU detected (CUDA: %s). Using CUDA for Docling.",
                    torch.cuda.get_device_name(0),
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
                pass  # CV2 Optional dependency check removed for pylint compliance

                self.converter.format_to_options[
                    InputFormat.PDF
                ].pipeline_options.do_table_structure = True
            except ImportError:
                logger.warning("OpenCV(cv2) 없음. 표 구조 추출 기능이 제한될 수 있습니다.")
                self.converter.format_to_options[
                    InputFormat.PDF
                ].pipeline_options.do_table_structure = False

        except ImportError as e:
            raise ImportError(f"Docling 라이브러리가 설치되지 않았습니다: {e}") from e

    def parse(
        self, pdf_path: Path
    ) -> ParsedDocument:  # pylint: disable=too-many-locals
        """Docling 기반 PDF 파싱."""
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
            elif "formula" in label or "equation" in label:
                block_type = "equation"
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
                page=item.prov[0].page_no if hasattr(item, "prov") and item.prov else 1,
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
                        "rows": [[str(c) for c in row] for row in df.values.tolist()],
                    }
                    parsed_block.text = df.to_markdown(index=False)
                except (AttributeError, ValueError, KeyError):
                    pass
            elif block_type == "equation":
                parsed_block.equation_data = {"latex": text}

            blocks.append(parsed_block)

            if block_type in ["title", "section"] and level is not None:
                context_stack.append(
                    {
                        "level": level,
                        "id": block_id,
                        "title": text,
                    }
                )

        # 후처리 1: 문맥 기반 분할 (인라인 수식 추출)
        blocks = _split_inline_equations(blocks)

        # 후처리 2: 문맥 기반 수식 탐지 및 재분류 (단독 블록 대상)
        blocks = _reclassify_equations(blocks)

        return ParsedDocument(
            source_path=str(pdf_path),
            blocks=blocks,
            metadata={"parser": "docling", "version": "2.0.0"},
        )

    def _extract_bbox(self, item) -> Optional[BoundingBox]:
        """Docling 아이템에서 BoundingBox 추출."""
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
    백업 파서: 스캔된 문서나 복잡한 표 처리를 위한 VLM (Vision-Language Model).
    gemini-3-flash-preview를 사용하여 이미지에서 구조화된 데이터를 추출.
    """

    def __init__(self, api_key: Optional[str] = None):
        import google.generativeai as genai  # type: ignore

        self.api_key = (
            api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        )
        if not self.api_key:
            raise ValueError("Gemini API Key is missing for Vision Parser.")

        genai.configure(api_key=self.api_key)
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

            prompt = (
                "Extract all text from this page. Return raw text. "
                "For equations or mathematical formulas, "
                "extract them strictly in LaTeX format."
            )
            import google.generativeai as genai

            response = genai.GenerativeModel(self.model_name).generate_content(
                contents=[prompt, image],
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


def _ocr_equation_region(
    page: "pymupdf.Page",  # type: ignore[name-defined]
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    padding: int = 10,
) -> Optional[str]:
    """수식 BBox 영역을 이미지로 크롭 후 Gemini Vision으로 LaTeX를 추출한다.

    API 키가 없거나 호출 실패 시 None을 반환하여 호출자가 fallback 할 수 있게 한다.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None

    try:
        # 1. BBox → 크롭 이미지 (패딩 추가로 잘림 방지)
        clip = pymupdf.Rect(
            x0 - padding,
            y0 - padding,
            x1 + padding,
            y1 + padding,
        )
        pix = page.get_pixmap(dpi=300, clip=clip)
        img_bytes = pix.tobytes("png")

        # 2. Gemini Vision 호출
        import google.generativeai as genai  # pylint: disable=import-outside-toplevel

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-3-flash-preview")

        prompt = (
            "This image contains a single mathematical equation. "
            "Return ONLY the equation in LaTeX format, without equation numbers, "
            "without $$ delimiters, without any explanation. "
            "Example: F_{en} = \\exp(0.935 - T^* \\dot{\\varepsilon}^* O^*)"
        )

        image = Image.open(io.BytesIO(img_bytes))
        response = model.generate_content(contents=[prompt, image])
        latex = response.text.strip()

        # 빈 응답이나 오류 메시지 필터
        if not latex or len(latex) < 3:
            return None

        logger.info("Vision OCR for equation: %s", latex)
        return latex

    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Vision OCR failed, falling back to PyMuPDF text: %s", exc)
        return None


def _supplement_missing_equations(
    doc_path: Path, docling_blocks: List[ParsedBlock]
) -> List[ParsedBlock]:
    """
    Docling이 통째로 텍스트를 누락한 영역에 대해 PyMuPDF로 가볍게 스캔하여
    수식 패턴이 있는 블록을 보충한다.
    Gemini Vision API 키가 있으면 크롭 OCR로 정확한 LaTeX를 취득한다.
    """
    try:
        doc = pymupdf.open(doc_path)
    except Exception as e:
        logger.warning("Surgical supplement failed to open PDF: %s", e)
        return docling_blocks

    supplemented_blocks = list(docling_blocks)

    # 예: "(12)" 같은 괄호형 수식 번호로 끝나는 패턴
    eq_num_pattern = re.compile(r"\(\s*(?P<num>\d+(\.\d+)*[a-zA-Z]?)\s*\)\s*$")
    # 수학 함수명 패턴 (= 기호가 폰트 디코딩 오류로 소실된 경우 보조 탐지용)
    math_func_pattern = re.compile(r"\b(exp|ln|log|sin|cos|tan|sqrt)\b", re.IGNORECASE)

    # 페이지별로 docling block들의 bbox를 O(N) 비교를 위해 미리 구성
    from collections import defaultdict

    docling_bboxes_by_page = defaultdict(list)

    # PDF 페이지 높이(Height)를 구해서 Y좌표를 통일하기 위해 미리 한 번 스캔
    page_heights = {}
    for p_idx, p in enumerate(doc):
        page_heights[p_idx + 1] = p.rect.height

    for b in docling_blocks:
        if b.bbox and b.text and len(b.text.strip()) >= 5:
            ph = page_heights.get(b.bbox.page, 800)
            dy0 = ph - b.bbox.y1
            dy1 = ph - b.bbox.y0
            if dy0 < 0 or dy1 < 0:
                dy0, dy1 = b.bbox.y0, b.bbox.y1

            docling_bboxes_by_page[b.bbox.page].append(
                {
                    "x0": b.bbox.x0 - 5,
                    "y0": dy0 - 5,
                    "x1": b.bbox.x1 + 5,
                    "y1": dy1 + 5,
                    "text": b.text,
                }
            )

    added_count = 0
    for page_index, page in enumerate(doc):
        page_num = page_index + 1
        page_dict = page.get_text("dict")

        for block in page_dict.get("blocks", []):
            if block["type"] != 0:  # text block
                continue

            text_parts = []
            for line in block["lines"]:
                for span in line["spans"]:
                    text_parts.append(span["text"])

            text = " ".join(text_parts).strip()
            if not text or len(text) < 5:
                continue

            # 수식 번호 패턴이 끝에 있어야 함 (필수)
            eq_num_match = eq_num_pattern.search(text)
            if not eq_num_match:
                continue

            # '=' 기호 또는 수학 함수명이 있어야 수식으로 인정
            has_equals = "=" in text
            has_math_func = bool(math_func_pattern.search(text))
            if not has_equals and not has_math_func:
                continue

            bx0, by0, bx1, by1 = block["bbox"]

            # 겹침 확인 (Docling이 이미 잡았다면 패스)
            is_overlap = False
            for db in docling_bboxes_by_page[page_num]:
                dx0, dy0, dx1, dy1 = db["x0"], db["y0"], db["x1"], db["y1"]
                if not (bx1 < dx0 or bx0 > dx1 or by1 < dy0 or by0 > dy1):
                    is_overlap = True
                    break

            if not is_overlap:
                eq_num = eq_num_match.group("num").strip()

                # Gemini Vision으로 정확한 LaTeX 취득 시도
                vision_latex = _ocr_equation_region(page, bx0, by0, bx1, by1)

                if vision_latex:
                    latex_text = vision_latex
                    display_text = vision_latex
                else:
                    # fallback: PyMuPDF 텍스트에서 수식 번호 제거 후 사용
                    latex_text = eq_num_pattern.sub("", text).strip()
                    latex_text = re.sub(r"\s{2,}", " ", latex_text)
                    display_text = text

                new_block = ParsedBlock(
                    page=page_num,
                    block_type="equation",
                    text=display_text,
                    equation_data={"latex": latex_text, "equationNumber": eq_num},
                    bbox=BoundingBox(x0=bx0, y0=by0, x1=bx1, y1=by1, page=page_num),
                    level=999,
                    block_id=str(uuid.uuid4()),
                )
                supplemented_blocks.append(new_block)
                added_count += 1

    doc.close()

    if added_count > 0:
        logger.info(
            "Surgically supplemented %d missing equations using PyMuPDF.", added_count
        )
        # 페이지와 y 좌표 순으로 재정렬
        supplemented_blocks.sort(key=lambda b: (b.page, b.bbox.y0 if b.bbox else 0))

    return supplemented_blocks


def _split_inline_equations(blocks: List[ParsedBlock]) -> List[ParsedBlock]:
    """
    긴 문단(paragraph) 내에 끼어있는 수식을 분리하여 별도 equation 블록으로 추출한다.
    앵커: '(숫자)' 혹은 '(숫자.숫자)' 형태의 수식 번호가 문단 중간/끝에 등장하고,
          그 앞에 수식 기호(=)가 존재하는 패턴을 찾는다.
    """
    new_blocks: List[ParsedBlock] = []

    # 예: "(12)", "(1.1)", "(13a)" 등 수식 번호 패턴 (단어 경계 확인)
    # 텍스트 내에 줄바꿈이 있을 수 있으므로 re.DOTALL 적용
    eq_num_pattern = re.compile(
        r"(?P<eq_text>.+?)\(\s*(?P<num>\d+(\.\d+)*[a-zA-Z]?)\s*\)", re.DOTALL
    )

    for block in blocks:
        if block.block_type != "paragraph" or not block.text or len(block.text) < 30:
            new_blocks.append(block)
            continue

        text = block.text
        # = 기호가 없으면 인라인 수식으로 취급 안 함
        if "=" not in text:
            new_blocks.append(block)
            continue

        parts_handled = False
        remaining_text = text
        block_splits: List[ParsedBlock] = []

        while True:
            match = eq_num_pattern.search(remaining_text)
            if not match:
                break

            candidate_text = match.group("eq_text").strip()
            eq_num = match.group("num")

            # 수식 기호가 없으면 일반 참조일 수 있으므로 패스
            if "=" not in candidate_text:
                # 현재 매치의 끝 위치 다음부터 남은 텍스트 재탐색
                remaining_text = remaining_text[match.end() :].strip()
                continue

            # 역방향으로 도입 키워드 찾기 (예: given by)
            # 가장 가까운 도입 키워드나 문장 끝 부호 이후를 수식의 시작으로 간주
            best_start_idx = 0

            # 1. 도입 키워드 찾기
            for kw in ["where", "given by", "as follows", "defined as", "is:", "is "]:
                # 마지막 등장을 찾기
                idx = candidate_text.lower().rfind(kw)
                if idx != -1:
                    # 원본 텍스트에서의 인덱스를 기준으로 자름
                    best_start_idx = max(best_start_idx, idx + len(kw))

            # 2. 문장 끝 부호 찾기 (도입 키워드가 발견되지 않았거나 더 가까운 부품이 있을 때)
            for punc in [". ", ": ", "; ", ".\n", ":\n", ";\n"]:
                idx = candidate_text.rfind(punc)
                if idx != -1:
                    best_start_idx = max(best_start_idx, idx + len(punc))

            # 만약 찾지 못했다면 전체를 수식으로 볼지 판단.
            # 하지만 = 기호가 아주 앞쪽에 있다면 전체를 수식으로 삼을 수 있음.
            if best_start_idx == 0:
                eq_start_idx = candidate_text.find("=")
                if eq_start_idx > 80:  # 너무 멀면 오탐지(글이 길 때) 방지, 하지만 여유를 좀 둠
                    # 수식 분할 포기하고 스킵
                    remaining_text = remaining_text[match.end() :].strip()
                    continue

            prefix = remaining_text[: match.start() + best_start_idx].strip()
            equation_body = candidate_text[best_start_idx:].strip()

            # 수식 분할 조건: 최소한의 길이와 '=' 포함 검증
            if len(equation_body) < 3 or "=" not in equation_body:
                remaining_text = remaining_text[match.end() :].strip()
                continue

            # prefix가 있으면 앞부분 paragraph 생성
            if prefix:
                prefix_block = ParsedBlock(
                    page=block.page,
                    block_type="paragraph",
                    text=prefix,
                    bbox=block.bbox,
                    level=block.level,
                    context_path=block.context_path,
                    parent_id=block.parent_id,
                    block_id=str(uuid.uuid4()),
                )
                block_splits.append(prefix_block)

            # equation 블록 생성
            eq_block = ParsedBlock(
                page=block.page,
                block_type="equation",
                text=equation_body,
                equation_data={"latex": equation_body, "equationNumber": eq_num},
                bbox=block.bbox,
                level=block.level,
                context_path=block.context_path,
                parent_id=block.parent_id,
                block_id=str(uuid.uuid4()),
                section_label=block.section_label,
                section_title=block.section_title,
            )
            block_splits.append(eq_block)

            remaining_text = remaining_text[match.end() :].strip()
            parts_handled = True

        if parts_handled:
            # 남은 뒷부분 텍스트
            if remaining_text:
                suffix_block = ParsedBlock(
                    page=block.page,
                    block_type="paragraph",
                    text=remaining_text,
                    bbox=block.bbox,
                    level=block.level,
                    context_path=block.context_path,
                    parent_id=block.parent_id,
                    block_id=str(uuid.uuid4()),
                )
                block_splits.append(suffix_block)

            new_blocks.extend(block_splits)
        else:
            new_blocks.append(block)

    return new_blocks


def _reclassify_equations(blocks: List[ParsedBlock]) -> List[ParsedBlock]:
    """
    파싱된 블록 리스트를 순회하며 좌우 문맥(Context)을 평가하여 수식을 탐지/재분류한다.
    기존 paragraph 블록 중에서 수식일 가능성이 높은 것을 판단.
    """
    for i, block in enumerate(blocks):
        if block.block_type != "paragraph" or not block.text:
            continue

        text = block.text.strip()
        if not text:
            continue

        score = 0

        # S1: 아주 짧은 텍스트 길이 (일반 문단은 보통 길음)
        if len(text) < 100:
            score += 1

        # S2: 괄호형 수식 번호로 끝나는 패턴
        eq_num = None
        eq_num_match = re.search(r"\((?P<num>\d+(\.\d+)*[a-zA-Z]?)\)\s*$", text)
        if eq_num_match:
            eq_num = eq_num_match.group("num")
            score += 2  # 강한 시그널

        # 주변 블록 컨텍스트
        prev_block_text = blocks[i - 1].text if i > 0 else None
        next_block_text = blocks[i + 1].text if i < len(blocks) - 1 else None
        prev_text = prev_block_text.strip().lower() if prev_block_text else ""
        next_text = next_block_text.strip().lower() if next_block_text else ""

        # S3: 앞 문단 끝 키워드
        prev_keywords = (
            "where",
            "given by",
            "defined as",
            "as follows",
            "expressed as",
            "is",
            "equation",
            "equation:",
            "식",
            "다음과 같다",
        )
        if any(prev_text.endswith(kw) for kw in prev_keywords):
            score += 2

        # S4: 뒤 문단 시작 키워드
        next_keywords = ("where", "from", "in which", "here", "여기서")
        if any(next_text.startswith(kw) for kw in next_keywords):
            score += 1

        # S5: 주변 블록 수식 지칭 단어 (본문이 아니라 내부에 있을 때만)
        surrounding_text = prev_text + " " + next_text
        if re.search(r"\b(eq\.|eqs\.|equation|수식)\b", surrounding_text):
            score += 1

        # S6: 특수 수학 기호 (유니코드 및 연산자)
        # 하이픈(-)은 일반 텍스트에서도 많이 쓰이므로, 단독 하이픈만 있는 경우는 기호에서 제외하거나 가중치를 낮춤
        math_symbols = set("∑∫√±αβγδεζηθικλμνξοπρστυφχψωΔΓΘΛΞΠΣΦΨΩ=+-*/<>≤≥≈≠")
        symbol_count = sum(1 for char in text if char in math_symbols)
        # = 기호가 있거나 전체 기호가 2개 이상일 때
        if "=" in text or symbol_count >= 2:
            score += 2

        # S7 (Penalty) : 일반 문장형 텍스트 및 오탐지(이메일, 전화번호 등) 방지
        # 1. 텍스트 자체가 "given by", "where" 등으로 끝나는 건 보통 도입 문장
        if any(text.lower().endswith(kw) for kw in prev_keywords):
            score -= 3
        # 2. 너무 긴 문장은 수식 아님 (특수기호가 많지 않은 한)
        if len(text) > 200 and symbol_count < 3:
            score -= 3
        # 3. 이메일 주소 형태인 경우
        if "@" in text and re.search(r"[\w\.-]+@[\w\.-]+", text):
            score -= 5
        # 4. 전화번호/팩스 번호 형태 (예: Facsimile: 301-415-2289)
        if re.search(r"\d{2,3}-\d{3,4}-\d{4}", text) and "=" not in text:
            score -= 5

        # 최종 평가: 3점 이상이면 수식으로 판단
        if score >= 3:
            block.block_type = "equation"
            block.equation_data = {
                "latex": text,  # 향후 수식 정제 모델이 붙는다면 여기서 처리
                "equationNumber": eq_num if eq_num else "",
            }

    return blocks


def parse_pdf(path: Path) -> ParsedDocument:
    """
    하이브리드 파싱 전략: Docling (최우선) → PyMuPDF → Gemini Vision (스캔본)

    1. Docling: 표, 레이아웃, 계층 구조 완벽 지원 (SectionClassifier 우회)
    2. PyMuPDF: 안정적 텍스트 추출 + SectionClassifier 적용
    3. Gemini Vision: 스캔 문서 전용 (비용 발생)
    """
    logger.info("Parsing PDF with Hybrid Strategy: %s", path)

    try:
        doc = pymupdf.open(path)
        total_text_len = sum(len(page.get_text()) for page in doc)
        is_scanned = (len(doc) > 0) and (total_text_len / len(doc) < 50)
        doc.close()

        if not is_scanned:
            try:
                logger.info("🚀 Docling 파서 시도 (표/구조 최적화)")
                parsed_doc = DoclingParser().parse(path)

                # 외과적 보충 (Surgical Supplement): Docling 누락 수식 채우기
                parsed_doc.blocks = _supplement_missing_equations(
                    path, parsed_doc.blocks
                )

                return parsed_doc
            except (ImportError, OSError, RuntimeError) as e:
                logger.warning(
                    "⚠️ Docling 실패 (%s). PyMuPDF + SectionClassifier로 전환.", e
                )
                return PyMuPDFParser().parse(path)
        else:
            logger.info("🖼️ Scanned PDF 감지: Gemini Vision(VLM) 사용")
            if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
                logger.warning("⚠️ Gemini API Key 없음. PyMuPDF로 강제 진행")
                return PyMuPDFParser().parse(path)
            return GeminiVisionParser().parse(path)

    except (OSError, RuntimeError, ValueError) as e:
        logger.warning("⚠️ 파싱 중 에러 (%s). PyMuPDF fallback 모드.", e)
        return PyMuPDFParser().parse(path)
