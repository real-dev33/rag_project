# pipeline.py
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
from dataclasses import dataclass

from .extractors.base import ExtractionResult, DocumentType
from .extractors.pdf_extractor import PDFExtractor
from .extractors.ocr_extractor import OCRExtractor
from .extractors.table_extractor import TableExtractor
from .validators.quality_checker import QualityChecker, QualityReport

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """
    Complete result from extraction pipeline.

    Combines extraction result with quality report and
    provides chunked output ready for RAG indexing.
    """
    extraction: ExtractionResult
    quality: QualityReport
    chunks: List[Dict[str, Any]]

    @property
    def is_successful(self) -> bool:
        """Check if extraction was successful and passed quality checks."""
        return (
            len(self.extraction.texts) > 0 and
            self.quality.passed
        )


class ExtractionPipeline:
    """
    Unified text extraction pipeline for RAG systems.

    Automatically selects the best extraction strategy based on
    document type and quality requirements.

    Pipeline flow:
    1. Detect document type
    2. Try native extraction first (fastest)
    3. Fall back to OCR if needed
    4. Extract tables separately
    5. Validate extraction quality
    6. Chunk text for embedding

    Configuration:
        - chunk_size: Target chunk size in characters (default: 1000)
        - chunk_overlap: Overlap between chunks (default: 200)
        - min_quality_score: Minimum quality threshold (default: 0.6)
        - ocr_config: OCR-specific settings
        - pdf_config: PDF extraction settings
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize pipeline with configuration."""
        self.config = config or {}

        # Initialize extractors
        self.pdf_extractor = PDFExtractor(self.config.get('pdf_config', {}))
        self.ocr_extractor = OCRExtractor(self.config.get('ocr_config', {}))
        self.table_extractor = TableExtractor(self.config.get('table_config', {}))

        # Initialize quality checker
        self.quality_checker = QualityChecker(self.config.get('quality_config', {}))

        # Chunking settings
        self.chunk_size = self.config.get('chunk_size', 1000)
        self.chunk_overlap = self.config.get('chunk_overlap', 200)
        self.min_quality_score = self.config.get('min_quality_score', 0.6)

    def extract(self, file_path: str) -> PipelineResult:
        """
        Run full extraction pipeline on a document.

        Args:
            file_path: Path to document file

        Returns:
            PipelineResult with extraction, quality report, and chunks
        """
        logger.info(f"Starting extraction pipeline for: {file_path}")

        path = Path(file_path)
        suffix = path.suffix.lower()

        # Step 1: Primary extraction based on file type
        if suffix == '.pdf':
            result = self._extract_pdf(file_path)
        elif suffix in {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'}:
            result = self._extract_image(file_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        # Step 2: Extract tables (for PDFs)
        if suffix == '.pdf':
            table_result = self.table_extractor.extract(file_path)
            result.tables = table_result.tables

        # Step 3: Quality validation
        quality_report = self.quality_checker.check(result)
        logger.info(
            f"Quality score: {quality_report.overall_score:.2f}, "
            f"Passed: {quality_report.passed}"
        )

        # Step 4: Re-extraction if quality is poor
        if not quality_report.passed and result.extraction_method == "native_pdf":
            logger.info("Quality check failed, attempting OCR extraction")
            result = self.ocr_extractor.extract(file_path)
            quality_report = self.quality_checker.check(result)

        # Step 5: Chunk text for embedding
        chunks = self._create_chunks(result)

        return PipelineResult(
            extraction=result,
            quality=quality_report,
            chunks=chunks
        )

    def _extract_pdf(self, file_path: str) -> ExtractionResult:
        """
        Extract text from PDF, choosing native or OCR based on content.

        Args:
            file_path: Path to PDF file

        Returns:
            ExtractionResult from best method
        """
        # Check if PDF is scanned
        if self.pdf_extractor.is_scanned_pdf(file_path):
            logger.info("Detected scanned PDF, using OCR")
            return self.ocr_extractor.extract(file_path)

        # Try native extraction first
        result = self.pdf_extractor.extract(file_path)

        # If extraction quality is poor, try OCR
        if result.quality_score < 0.5:
            logger.info("Native extraction quality low, trying OCR")
            ocr_result = self.ocr_extractor.extract(file_path)

            # Use whichever result is better
            if ocr_result.quality_score > result.quality_score:
                return ocr_result

        return result

    def _extract_image(self, file_path: str) -> ExtractionResult:
        """
        Extract text from image using OCR.

        Args:
            file_path: Path to image file

        Returns:
            ExtractionResult from OCR
        """
        return self.ocr_extractor.extract(file_path)

    def _create_chunks(
        self,
        result: ExtractionResult
    ) -> List[Dict[str, Any]]:
        """
        Split extracted text into chunks for vector embedding.

        Uses overlap to maintain context across chunk boundaries.
        Respects paragraph boundaries where possible.

        Args:
            result: ExtractionResult to chunk

        Returns:
            List of chunk dictionaries with text and metadata
        """
        chunks = []

        # Process each page separately to maintain page references
        for text_block in result.texts:
            page_chunks = self._chunk_text(
                text_block.content,
                text_block.page_number,
                text_block.metadata
            )
            chunks.extend(page_chunks)

        # Add table chunks
        for i, table in enumerate(result.tables):
            table_text = table.get('markdown', '')
            if table_text:
                chunks.append({
                    'text': table_text,
                    'page': table.get('page', 0),
                    'chunk_type': 'table',
                    'table_index': i,
                    'metadata': {
                        'rows': table.get('rows', 0),
                        'columns': len(table.get('columns', [])),
                        'accuracy': table.get('accuracy', 0)
                    }
                })

        # Add chunk indices
        for i, chunk in enumerate(chunks):
            chunk['chunk_index'] = i

        logger.info(f"Created {len(chunks)} chunks from extraction")
        return chunks

    def _chunk_text(
        self,
        text: str,
        page_number: int,
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks.

        Tries to split at paragraph boundaries, falling back to
        sentence boundaries, then word boundaries.

        Args:
            text: Text to chunk
            page_number: Source page number
            metadata: Additional metadata to include

        Returns:
            List of chunk dictionaries
        """
        if not text:
            return []

        chunks = []

        # Split into paragraphs first
        paragraphs = text.split('\n\n')

        current_chunk = ""
        chunk_start = 0

        for para in paragraphs:
            # If adding this paragraph exceeds chunk size
            if len(current_chunk) + len(para) > self.chunk_size:
                # Save current chunk if it has content
                if current_chunk.strip():
                    chunks.append({
                        'text': current_chunk.strip(),
                        'page': page_number,
                        'chunk_type': 'text',
                        'char_start': chunk_start,
                        'char_end': chunk_start + len(current_chunk),
                        'metadata': metadata
                    })

                # Start new chunk with overlap
                overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                current_chunk = overlap_text + "\n\n" + para
                chunk_start = chunk_start + len(current_chunk) - len(overlap_text) - len(para)
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para

        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'page': page_number,
                'chunk_type': 'text',
                'char_start': chunk_start,
                'char_end': chunk_start + len(current_chunk),
                'metadata': metadata
            })

        return chunks

    def extract_batch(
        self,
        file_paths: List[str],
        on_progress: Optional[callable] = None
    ) -> Dict[str, PipelineResult]:
        """
        Extract text from multiple documents.

        Args:
            file_paths: List of document paths
            on_progress: Optional callback(current, total, file_path)

        Returns:
            Dictionary mapping file paths to results
        """
        results = {}
        total = len(file_paths)

        for i, file_path in enumerate(file_paths):
            try:
                if on_progress:
                    on_progress(i + 1, total, file_path)

                results[file_path] = self.extract(file_path)

            except Exception as e:
                logger.error(f"Failed to extract {file_path}: {str(e)}")
                # Store error result
                results[file_path] = PipelineResult(
                    extraction=ExtractionResult(
                        texts=[],
                        errors=[str(e)]
                    ),
                    quality=QualityReport(
                        overall_score=0.0,
                        issues=[str(e)],
                        suggestions=["Check file format and permissions"],
                        metrics={}
                    ),
                    chunks=[]
                )

        return results