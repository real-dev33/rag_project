import pymupdf  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from .base import(
    BaseExtractor,
    ExtractionResult,
    ExtractedText,
    DocumentType
)
logger=logging.getLogger(__name__)
class PDFExtractor(BaseExtractor):
     """
    Extract text from native PDF documents using PyMuPDF.

    This extractor handles PDFs with embedded text (not scanned images).
    For scanned PDFs, use OCRExtractor instead.

    Configuration options:
        - preserve_layout: Maintain spatial text positioning (default: True)
        - extract_images: Also extract embedded images (default: False)
        - password: Password for encrypted PDFs (default: None)
    """
     supported_extension={'.pdf'}
     def _setup(self):
       """configure extraction settings"""
       self.preserve_layout=self.config.get('preserve_layout',True)
       self.extract_image=self.config.get('extract_image',False)
       self.password=self.config.get('password',None)
     def can_handle(self,file_path:str)->bool:
       """check if a file is a pdf"""
       return Path(file_path).suffix.lower() in self.supported_extension
     def extract(self,file_path:str)->ExtractionResult:
        """
        Extract text from a PDF document.

        This method attempts native text extraction first. If the PDF
        appears to be scanned (low text density), it returns a result
        indicating OCR is needed.

        Args:
            file_path: Path to PDF file

        Returns:
            ExtractionResult with extracted text blocks
        """
        self.validate_file(file_path)
        texts:list[ExtractedText]
        errors:list[str]=[]
        try:
            doc=pymupdf.open(file_path,password=self.password)
            if doc.is_encrypted:
               if self.password:
                  if not doc.authenticate(self.password):
                     raise ValueError("Incorrect password for encrypted PDF")
               else:
                  raise ValueError("PDF is encrypted,Password required")
            total_pages=len(doc)
            logger.info(f"Processing PDF with {total_pages} pages: {file_path}")
            for page_num in range(total_pages):
                page = doc[page_num]

                # Extract text with layout preservation
                if self.preserve_layout:
                    # Use blocks to preserve structure
                    blocks = page.get_text("dict")["blocks"]
                    page_text = self._process_blocks(blocks, page_num + 1)
                    texts.extend(page_text)
                else:
                    # Simple text extraction
                    text = page.get_text("text")
                    if text.strip():
                        texts.append(ExtractedText(
                            content=text,
                            page_number=page_num + 1,
                            metadata={
                                "extraction_method": "pymupdf_simple",
                                "page_width": page.rect.width,
                                "page_height": page.rect.height
                            }
                        ))

            doc.close()

            # Calculate quality score based on extraction results
            quality_score = self._calculate_quality_score(texts, total_pages)
            return ExtractionResult(
                texts=texts,
                document_type=DocumentType.PDF,
                total_pages=total_pages,
                extraction_method="native_pdf",
                quality_score=quality_score,
                errors=errors
            )
        except Exception as e:
            logger.error(f"PDF extraction failed: {str(e)}")
            errors.append(str(e))
            return ExtractionResult(
                texts=[],
                document_type=DocumentType.PDF,
                extraction_method="native_pdf",
                errors=errors
            )
     def _process_blocks(self, blocks: List[Dict[str, Any]], page_number: int) -> List[ExtractedText]:
        """
        Process PDF text blocks preserving layout structure.

        Args:
            blocks: List of block dictionaries from PyMuPDF
            page_number: Current page number (1-indexed)

        Returns:
            List of ExtractedText objects
        """
        extracted = []

        for block in blocks:
            # Skip image blocks (type 1)
            if block.get("type") != 0:
                continue

            # Extract text from lines within the block
            block_text = []
            for line in block.get("lines", []):
                line_text = ""
                for span in line.get("spans", []):
                    line_text += span.get("text", "")
                block_text.append(line_text)

            content = "\n".join(block_text)
            if not content.strip():
                continue

            # Get bounding box coordinates
            bbox = block.get("bbox", [0, 0, 0, 0])

            extracted.append(ExtractedText(
                content=content,
                page_number=page_number,
                bounding_box=list(bbox),
                metadata={
                    "extraction_method": "pymupdf_blocks",
                    "block_type": "text",
                    "line_count": len(block_text)
                }
            ))

        return extracted
     def _calculate_quality_score(
        self,
        texts: List[ExtractedText],
        total_pages: int
    ) -> float:
        """
        Calculate extraction quality score.

        Factors considered:
        - Text density per page
        - Character diversity
        - Presence of common patterns

        Args:
            texts: Extracted text blocks
            total_pages: Total pages in document

        Returns:
            Quality score between 0.0 and 1.0
        """
        if not texts or total_pages == 0:
            return 0.0

        full_text = " ".join(t.content for t in texts)

        # Check text density (characters per page)
        chars_per_page = len(full_text) / total_pages
        density_score = min(chars_per_page / 1000, 1.0)  # Expect ~1000 chars/page

        # Check character diversity (unique chars / total chars)
        unique_chars = len(set(full_text))
        diversity_score = min(unique_chars / 50, 1.0)  # Expect ~50 unique chars

        # Check for garbled text patterns (common OCR artifacts)
        garbled_patterns = ['�', '\x00', '\ufffd']
        garbled_count = sum(full_text.count(p) for p in garbled_patterns)
        garbled_score = max(0, 1 - (garbled_count / max(len(full_text), 1)) * 10)

        # Weighted average
        return (density_score * 0.4 + diversity_score * 0.3 + garbled_score * 0.3)

     def is_scanned_pdf(self, file_path: str) -> bool:
        """
        Detect if PDF is scanned (image-based) rather than native text.

        A scanned PDF typically has very little or no extractable text
        despite having visible content on pages.

        Args:
            file_path: Path to PDF file

        Returns:
            True if PDF appears to be scanned/image-based
        """
        try:
            doc = pymupdf.open(file_path)
            total_text_chars = 0
            total_pages = len(doc)

            for page_num in range(min(total_pages, 5)):  # Check first 5 pages
                page = doc[page_num]
                text = page.get_text("text")
                total_text_chars += len(text.strip())

            doc.close()

            # If average chars per page is very low, likely scanned
            avg_chars = total_text_chars / min(total_pages, 5)
            return avg_chars < 100  # Threshold for scanned detection

        except Exception as e:
            logger.warning(f"Could not detect PDF type: {e}")
            return False

       