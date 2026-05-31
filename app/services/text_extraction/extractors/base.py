from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import hashlib


class DocumentType(Enum):
    """Supported document types for extraction."""
    PDF = "pdf"
    HTML = "html"
    IMAGE = "image"
    WORD = "word"
    EXCEL = "excel"
    UNKNOWN = "unknown"


@dataclass
class ExtractedText:
    """
    Container for extracted text and metadata.

    Attributes:
        content: The raw extracted text content
        page_number: Source page number (1-indexed)
        confidence: OCR confidence score (0.0 to 1.0), None for native text
        bounding_box: Optional coordinates of text region [x1, y1, x2, y2]
        metadata: Additional extraction metadata (font, size, etc.)
    """
    content: str
    page_number: int
    confidence: Optional[float] = None
    bounding_box: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Clean whitespace while preserving structure
        self.content = self.content.strip()

    @property
    def word_count(self) -> int:
        """Count the words in content."""
        return len(self.content.split())

    @property
    def content_hash(self) -> str:
        """Generate hash for deduplication."""
        return hashlib.md5(self.content.encode()).hexdigest()


@dataclass
class ExtractionResult:
    """
    Complete result from document extraction.

    Attributes:
        texts: List of extracted text blocks
        tables: List of extracted tables as dictionaries
        document_type: Detected document type
        total_pages: Total number of pages processed
        extraction_method: Method used (native, ocr, hybrid)
        quality_score: Overall extraction quality (0.0 to 1.0)
        errors: List of any errors encountered during extraction
    """
    texts: List[ExtractedText]
    # Fix: was List[Dict[ExtractedText]] — wrong type, and no default caused crashes
    # in all callers that omit tables= (every error-return path in OCR/PDF extractors)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    document_type: DocumentType = DocumentType.UNKNOWN
    total_pages: int = 0
    extraction_method: str = "unknown"
    quality_score: float = 0.0
    errors: List[str] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        """Concatenate all extracted text blocks."""
        return "\n\n".join(t.content for t in self.texts if t.content)

    @property
    def average_confidence(self) -> Optional[float]:
        """Calculate average OCR confidence across all blocks."""
        confidences = [t.confidence for t in self.texts if t.confidence is not None]
        return sum(confidences) / len(confidences) if confidences else None


class BaseExtractor(ABC):
    """
    Abstract base class for all text extractors.

    Subclasses must implement the extract() method to handle
    their specific document type.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize extractor with optional configuration.

        Args:
            config: Dictionary of extractor-specific settings
        """
        self.config = config or {}
        self._setup()

    def _setup(self):
        """Override in subclasses for initialization logic."""
        pass

    @abstractmethod
    def extract(self, file_path: str) -> ExtractionResult:
        """
        Extract text from a document.

        Args:
            file_path: Path to the document file

        Returns:
            ExtractionResult containing extracted text and metadata
        """
        pass

    @abstractmethod
    def can_handle(self, file_path: str) -> bool:
        """
        Check if this extractor can handle the given file.

        Args:
            file_path: Path to the document file

        Returns:
            True if this extractor supports the file type
        """
        pass

    def validate_file(self, file_path: str) -> bool:
        """
        Validate that file exists and is readable.

        Args:
            file_path: Path to the document file

        Returns:
            True if file is valid and readable
        """
        import os
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document not found: {file_path}")
        if not os.access(file_path, os.R_OK):
            raise PermissionError(f"Cannot read document: {file_path}")
        return True