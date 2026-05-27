from abc import ABC,abstractmethod
from dataclasses import dataclass,field
from typing import List,Optional,Dict,Any
from enum import Enum
import hashlib

class DocumentType(Enum):
    """supported document types for extraction"""
    PDF="pdf"
    HTML="html"
    IMAGE="image"
    WORD="word"
    EXCEL="excel"
    UNKNOWN="unknown"
@dataclass
class ExtractedText:
    """ 
    container for extrcted text meta data

attributes:
        content: the raw extracted text content
        page_number: source page number (1-indexed)
        confidence: ocr confidence score (0.0 to 1.0), None for native text
        bounding_box: optional coordinates of text region [x1, y1, x2, y2]
        metadata: additional extraction metadata (font, size, etc.)
    
    """
    content:str
    page_number:int
    confidence: Optional[float]=None
    bounding_box: Optional[List[float]]=None
    metadata:Dict[str,Any]=field(default_factory=dict)
    def __post_init__(self):
        # clean whitespace while preserving structure
        self.content=self.content.strip()
    @property
    def word_count(self) -> int:
        """counts the word"""
        return len(self.content.split())
    @property
    def content_hash(self)->str:
        """generate hash for duplication"""
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
    texts:List[ExtractedText]
    tables:List[Dict[ExtractedText]]
    document_type:DocumentType=DocumentType.UNKNOWN
    total_pages:int = 0
    extraction_method:str = "unknown"
    quality_score: float  = 0.0
    errors:List[str]=field(default_factory=list)
    @property
    def full_text(self)->str:
        """concanete all extracted text blocks"""
        return "\n\n".join(t.content for t in self.texts if t.content)
    @property
    def average_confidence(self)->Optional[float]:
        """calculate average OCR across the blocks"""
        confidences=[t.confidence for t in self.texts if t.confidence is not None]
        return sum(confidences)/len(confidences) if confidences else None
class BaseExtractor(ABC):
     """
    abstract base class for all text extractors.

    subclasses must implement the extract() method to handle
    their specific document type.
    """
     def __init__(self,config:Optional[Dict[str,Any]]=None):
         """
         initialize extractor with optional configuration
         args:
         config:dictionary of extractor specific setting
         """
         self.config = config or {}
         self._setup()
     def _setup(self):
         """override in subclassses for iinitialization logic"""
         pass
     @abstractmethod
     def extract(self,file_path:str)->ExtractionResult:
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
            raise FileNotFoundError(f"Document not found:{file_path}")
        if not os.access(file_path,os.R_OK):
            raise PermissionError(f"cannot read document:{file_path}")
        return True
    


     

