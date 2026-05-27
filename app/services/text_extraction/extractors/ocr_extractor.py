# extractors/ocr_extractor.py
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import tempfile
import os

from .base import (
    BaseExtractor,
    ExtractionResult,
    ExtractedText,
    DocumentType
)
from ..preprocessors.image_enhancer import ImageEnhancer

logger = logging.getLogger(__name__)


class OCRExtractor(BaseExtractor):
    """
    Extract text from images and scanned documents using OCR.

    Uses Tesseract OCR engine with configurable preprocessing
    and language support.

    Configuration options:
        - language: OCR language(s) (default: 'eng')
        - psm: Page segmentation mode (default: 3 - auto)
        - oem: OCR engine mode (default: 3 - LSTM only)
        - dpi: Resolution for PDF conversion (default: 300)
        - preprocess: Enable image preprocessing (default: True)
    """

    SUPPORTED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.webp'}
    SUPPORTED_DOC_EXTENSIONS = {'.pdf'}

    # Tesseract Page Segmentation Modes
    PSM_MODES = {
        0: "Orientation and script detection only",
        1: "Automatic page segmentation with OSD",
        3: "Fully automatic page segmentation (default)",
        4: "Assume single column of text",
        6: "Assume single uniform block of text",
        7: "Treat image as single text line",
        11: "Sparse text, no particular order",
        12: "Sparse text with OSD",
    }

    def _setup(self):
        """Configure OCR settings."""
        self.language = self.config.get('language', 'eng')
        self.psm = self.config.get('psm', 3)
        self.oem = self.config.get('oem', 3)
        self.dpi = self.config.get('dpi', 300)
        self.preprocess = self.config.get('preprocess', True)

        # Initialize image enhancer
        self.enhancer = ImageEnhancer(self.config.get('enhancer_config'))

        # Verify Tesseract installation
        try:
            pytesseract.get_tesseract_version()
        except pytesseract.TesseractNotFoundError:
            raise RuntimeError(
                "Tesseract OCR not found. Install with: brew install tesseract"
            )

    def can_handle(self, file_path: str) -> bool:
        """Check if file is an image or scanned PDF."""
        suffix = Path(file_path).suffix.lower()
        return suffix in self.SUPPORTED_IMAGE_EXTENSIONS | self.SUPPORTED_DOC_EXTENSIONS

    def extract(self, file_path: str) -> ExtractionResult:
        """
        Extract text using OCR.

        Handles both images and PDFs. PDFs are first converted to
        images, then processed page by page.

        Args:
            file_path: Path to image or PDF file

        Returns:
            ExtractionResult with OCR-extracted text
        """
        self.validate_file(file_path)

        suffix = Path(file_path).suffix.lower()

        if suffix in self.SUPPORTED_IMAGE_EXTENSIONS:
            return self._extract_from_image(file_path)
        elif suffix == '.pdf':
            return self._extract_from_pdf(file_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    def _extract_from_image(self, file_path: str) -> ExtractionResult:
        """
        Extract text from a single image file.

        Args:
            file_path: Path to image file

        Returns:
            ExtractionResult with extracted text
        """
        texts: List[ExtractedText] = []
        errors: List[str] = []

        try:
            # Load image
            image = cv2.imread(file_path)
            if image is None:
                raise ValueError(f"Could not load image: {file_path}")

            # Preprocess if enabled
            if self.preprocess:
                processed = self.enhancer.enhance(image)
            else:
                processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Run OCR with detailed output
            ocr_result = self._run_ocr_with_details(processed)

            if ocr_result['text'].strip():
                texts.append(ExtractedText(
                    content=ocr_result['text'],
                    page_number=1,
                    confidence=ocr_result['confidence'],
                    metadata={
                        "extraction_method": "tesseract_ocr",
                        "language": self.language,
                        "psm_mode": self.psm,
                        "preprocessed": self.preprocess,
                        "image_quality": self.enhancer.estimate_quality(image)
                    }
                ))

            quality_score = ocr_result['confidence'] if ocr_result['confidence'] else 0.0

            return ExtractionResult(
                texts=texts,
                document_type=DocumentType.IMAGE,
                total_pages=1,
                extraction_method="ocr",
                quality_score=quality_score,
                errors=errors
            )

        except Exception as e:
            logger.error(f"Image OCR failed: {str(e)}")
            errors.append(str(e))
            return ExtractionResult(
                texts=[],
                document_type=DocumentType.IMAGE,
                extraction_method="ocr",
                errors=errors
            )

    def _extract_from_pdf(self, file_path: str) -> ExtractionResult:
        """
        Extract text from a scanned PDF using OCR.

        Converts each page to an image, then runs OCR.

        Args:
            file_path: Path to PDF file

        Returns:
            ExtractionResult with OCR-extracted text
        """
        texts: List[ExtractedText] = []
        errors: List[str] = []

        try:
            # Convert PDF pages to images
            logger.info(f"Converting PDF to images at {self.dpi} DPI")
            images = convert_from_path(
                file_path,
                dpi=self.dpi,
                fmt='png',
                thread_count=4  # Parallel conversion
            )

            total_pages = len(images)
            logger.info(f"Processing {total_pages} pages with OCR")

            total_confidence = 0.0
            pages_with_confidence = 0

            for page_num, pil_image in enumerate(images, start=1):
                # Convert PIL Image to OpenCV format
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

                # Preprocess
                if self.preprocess:
                    processed = self.enhancer.enhance(image)
                else:
                    processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Run OCR
                ocr_result = self._run_ocr_with_details(processed)

                if ocr_result['text'].strip():
                    texts.append(ExtractedText(
                        content=ocr_result['text'],
                        page_number=page_num,
                        confidence=ocr_result['confidence'],
                        metadata={
                            "extraction_method": "tesseract_ocr",
                            "language": self.language,
                            "dpi": self.dpi,
                            "preprocessed": self.preprocess
                        }
                    ))

                    if ocr_result['confidence']:
                        total_confidence += ocr_result['confidence']
                        pages_with_confidence += 1

                logger.debug(
                    f"Page {page_num}/{total_pages} complete, "
                    f"confidence: {ocr_result['confidence']:.2f}"
                )

            # Calculate average quality
            avg_confidence = (
                total_confidence / pages_with_confidence
                if pages_with_confidence > 0
                else 0.0
            )

            return ExtractionResult(
                texts=texts,
                document_type=DocumentType.PDF,
                total_pages=total_pages,
                extraction_method="ocr",
                quality_score=avg_confidence,
                errors=errors
            )

        except Exception as e:
            logger.error(f"PDF OCR failed: {str(e)}")
            errors.append(str(e))
            return ExtractionResult(
                texts=[],
                document_type=DocumentType.PDF,
                extraction_method="ocr",
                errors=errors
            )

    def _run_ocr_with_details(
        self,
        image: np.ndarray
    ) -> Dict[str, Any]:
        """
        Run Tesseract OCR and extract detailed results.

        Uses pytesseract.image_to_data() to get word-level
        confidence scores along with text.

        Args:
            image: Preprocessed grayscale image

        Returns:
            Dictionary with 'text' and 'confidence' keys
        """
        # Build Tesseract config string
        config = f'--psm {self.psm} --oem {self.oem}'

        # Get detailed OCR output with confidence scores
        data = pytesseract.image_to_data(
            image,
            lang=self.language,
            config=config,
            output_type=pytesseract.Output.DICT
        )

        # Build text and calculate confidence
        words = []
        confidences = []

        for i, word in enumerate(data['text']):
            conf = int(data['conf'][i])

            # Skip empty words and low-confidence noise
            if word.strip() and conf > 0:
                words.append(word)
                confidences.append(conf)

        text = ' '.join(words)
        avg_confidence = (
            sum(confidences) / len(confidences) / 100
            if confidences
            else 0.0
        )

        return {
            'text': text,
            'confidence': avg_confidence,
            'word_count': len(words)
        }

    def extract_with_layout(self, file_path: str) -> ExtractionResult:
        """
        Extract text preserving document layout using hOCR output.

        hOCR is an HTML-based format that includes bounding box
        coordinates for each word and line.

        Args:
            file_path: Path to image file

        Returns:
            ExtractionResult with layout-aware text blocks
        """
        self.validate_file(file_path)

        texts: List[ExtractedText] = []

        # Load and preprocess image
        image = cv2.imread(file_path)
        if self.preprocess:
            processed = self.enhancer.enhance(image)
        else:
            processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Get hOCR output
        config = f'--psm {self.psm} --oem {self.oem}'
        hocr = pytesseract.image_to_pdf_or_hocr(
            processed,
            lang=self.language,
            config=config,
            extension='hocr'
        )

        # Parse hOCR to extract text blocks with positions
        # (Simplified; full implementation would parse XML)
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(hocr, 'html.parser')

        for paragraph in soup.find_all('p', class_='ocr_par'):
            # Extract bounding box from title attribute
            title = paragraph.get('title', '')
            bbox = self._parse_bbox(title)

            text = paragraph.get_text(strip=True)
            if text:
                texts.append(ExtractedText(
                    content=text,
                    page_number=1,
                    bounding_box=bbox,
                    metadata={
                        "extraction_method": "tesseract_hocr",
                        "layout_preserved": True
                    }
                ))

        return ExtractionResult(
            texts=texts,
            document_type=DocumentType.IMAGE,
            total_pages=1,
            extraction_method="ocr_layout",
            quality_score=0.0  # Would calculate from word confidences
        )

    def _parse_bbox(self, title: str) -> Optional[List[float]]:
        """Parse bounding box from hOCR title attribute."""
        import re
        match = re.search(r'bbox (\d+) (\d+) (\d+) (\d+)', title)
        if match:
            return [float(x) for x in match.groups()]
        return None