# validators/quality_checker.py
import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import logging

from ..extractors.base import ExtractionResult, ExtractedText

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """
    Detailed quality assessment report.

    Attributes:
        overall_score: Combined quality score (0.0 to 1.0)
        issues: List of detected quality issues
        suggestions: Recommended actions to improve extraction
        metrics: Detailed quality metrics
    """
    overall_score: float
    issues: List[str]
    suggestions: List[str]
    metrics: Dict[str, float]

    @property
    def passed(self) -> bool:
        """Check if quality meets minimum threshold."""
        return self.overall_score >= 0.6


class QualityChecker:
    """
    Validate extraction quality for RAG suitability.

    Checks for:
    - Text coherence and readability
    - OCR artifacts and garbled text
    - Language consistency
    - Content density
    - Structure preservation

    Configuration:
        - min_word_length: Minimum average word length (default: 3)
        - max_garbled_ratio: Maximum allowed garbled character ratio (default: 0.05)
        - min_alpha_ratio: Minimum alphabetic character ratio (default: 0.7)
    """

    # Common OCR error patterns
    GARBLED_PATTERNS = [
        r'[^\x00-\x7F]{3,}',  # Long non-ASCII sequences
        r'[!@#$%^&*]{3,}',     # Repeated special characters
        r'\b[bcdfghjklmnpqrstvwxz]{5,}\b',  # Consonant-only words
        r'(?:\d[A-Za-z]){3,}', # Alternating digit-letter patterns
    ]

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize quality checker with configuration."""
        self.config = config or {}
        self.min_word_length = self.config.get('min_word_length', 3)
        self.max_garbled_ratio = self.config.get('max_garbled_ratio', 0.05)
        self.min_alpha_ratio = self.config.get('min_alpha_ratio', 0.7)

    def check(self, result: ExtractionResult) -> QualityReport:
        """
        Perform comprehensive quality check on extraction result.

        Args:
            result: ExtractionResult to validate

        Returns:
            QualityReport with detailed assessment
        """
        issues = []
        suggestions = []
        metrics = {}

        # Get full text for analysis
        full_text = result.full_text

        if not full_text:
            return QualityReport(
                overall_score=0.0,
                issues=["No text extracted"],
                suggestions=["Try OCR extraction", "Check if document is readable"],
                metrics={'text_length': 0}
            )

        # Run individual checks
        coherence_score, coherence_issues = self._check_coherence(full_text)
        metrics['coherence'] = coherence_score
        issues.extend(coherence_issues)

        garbled_score, garbled_issues = self._check_garbled_text(full_text)
        metrics['garbled_text'] = garbled_score
        issues.extend(garbled_issues)

        density_score, density_issues = self._check_content_density(result)
        metrics['content_density'] = density_score
        issues.extend(density_issues)

        structure_score, structure_issues = self._check_structure(result)
        metrics['structure'] = structure_score
        issues.extend(structure_issues)

        # Check OCR confidence if available
        if result.average_confidence is not None:
            metrics['ocr_confidence'] = result.average_confidence
            if result.average_confidence < 0.7:
                issues.append(f"Low OCR confidence: {result.average_confidence:.2f}")
                suggestions.append("Try higher DPI or better image preprocessing")

        # Calculate overall score
        weights = {
            'coherence': 0.3,
            'garbled_text': 0.25,
            'content_density': 0.25,
            'structure': 0.2
        }

        overall_score = sum(
            metrics.get(key, 0) * weight
            for key, weight in weights.items()
        )

        # Generate suggestions based on issues
        suggestions.extend(self._generate_suggestions(issues, metrics))

        return QualityReport(
            overall_score=overall_score,
            issues=issues,
            suggestions=list(set(suggestions)),  # Deduplicate
            metrics=metrics
        )

    def _check_coherence(self, text: str) -> Tuple[float, List[str]]:
        """
        Check text coherence and readability.

        Analyzes:
        - Average word length
        - Sentence structure
        - Word validity
        """
        issues = []

        words = text.split()
        if not words:
            return 0.0, ["No words found in text"]

        # Average word length check
        avg_word_length = sum(len(w) for w in words) / len(words)
        if avg_word_length < self.min_word_length:
            issues.append(f"Short average word length: {avg_word_length:.1f}")

        word_length_score = min(avg_word_length / 5, 1.0)

        # Check for reasonable sentence structure
        sentences = re.split(r'[.!?]+', text)
        valid_sentences = [s for s in sentences if len(s.split()) >= 3]
        sentence_ratio = len(valid_sentences) / max(len(sentences), 1)

        if sentence_ratio < 0.5:
            issues.append("Poor sentence structure detected")

        # Combined coherence score
        coherence_score = (word_length_score * 0.5 + sentence_ratio * 0.5)

        return coherence_score, issues

    def _check_garbled_text(self, text: str) -> Tuple[float, List[str]]:
        """
        Detect OCR artifacts and garbled text.

        Looks for common OCR error patterns that indicate
        poor recognition quality.
        """
        issues = []

        total_chars = len(text)
        if total_chars == 0:
            return 0.0, ["Empty text"]

        # Count garbled characters
        garbled_count = 0
        for pattern in self.GARBLED_PATTERNS:
            matches = re.findall(pattern, text)
            garbled_count += sum(len(m) for m in matches)

        garbled_ratio = garbled_count / total_chars

        if garbled_ratio > self.max_garbled_ratio:
            issues.append(f"High garbled text ratio: {garbled_ratio:.2%}")

        # Check alphabetic ratio
        alpha_count = sum(1 for c in text if c.isalpha())
        alpha_ratio = alpha_count / total_chars

        if alpha_ratio < self.min_alpha_ratio:
            issues.append(f"Low alphabetic content: {alpha_ratio:.2%}")

        # Score (higher is better, so invert garbled ratio)
        garbled_score = max(0, 1 - (garbled_ratio * 10))
        alpha_score = min(alpha_ratio / self.min_alpha_ratio, 1.0)

        return (garbled_score * 0.6 + alpha_score * 0.4), issues

    def _check_content_density(
        self,
        result: ExtractionResult
    ) -> Tuple[float, List[str]]:
        """
        Check content density per page.

        Too little content per page may indicate extraction issues.
        """
        issues = []

        if result.total_pages == 0:
            return 0.0, ["No pages processed"]

        total_words = sum(t.word_count for t in result.texts)
        words_per_page = total_words / result.total_pages

        # Expect at least 100 words per page for typical documents
        if words_per_page < 50:
            issues.append(f"Low content density: {words_per_page:.0f} words/page")
            density_score = words_per_page / 100
        elif words_per_page < 100:
            issues.append(f"Below average content: {words_per_page:.0f} words/page")
            density_score = 0.5 + (words_per_page - 50) / 100
        else:
            density_score = min(words_per_page / 200, 1.0)

        return density_score, issues

    def _check_structure(
        self,
        result: ExtractionResult
    ) -> Tuple[float, List[str]]:
        """
        Check if document structure was preserved.

        Looks for:
        - Paragraph separation
        - List formatting
        - Header patterns
        """
        issues = []
        full_text = result.full_text

        # Check for paragraph breaks
        paragraphs = [p for p in full_text.split('\n\n') if p.strip()]
        if len(paragraphs) < 2 and len(full_text) > 500:
            issues.append("No paragraph structure detected")
            paragraph_score = 0.3
        else:
            paragraph_score = min(len(paragraphs) / 10, 1.0)

        # Check for list patterns
        list_patterns = [
            r'^\s*[\u2022\u2023\u25E6\u2043\u2219]\s',  # Bullet points
            r'^\s*\d+[.)]\s',  # Numbered lists
            r'^\s*[a-z][.)]\s',  # Lettered lists
        ]
        has_lists = any(
            re.search(pattern, full_text, re.MULTILINE)
            for pattern in list_patterns
        )
        list_score = 1.0 if has_lists else 0.7  # Not having lists is not necessarily bad

        # Combined structure score
        structure_score = (paragraph_score * 0.7 + list_score * 0.3)

        return structure_score, issues

    def _generate_suggestions(
        self,
        issues: List[str],
        metrics: Dict[str, float]
    ) -> List[str]:
        """Generate actionable suggestions based on detected issues."""
        suggestions = []

        if metrics.get('coherence', 1) < 0.5:
            suggestions.append("Consider using a different extraction method")
            suggestions.append("Check document language settings")

        if metrics.get('garbled_text', 1) < 0.6:
            suggestions.append("Increase image DPI for OCR")
            suggestions.append("Apply image preprocessing (denoising, contrast)")

        if metrics.get('content_density', 1) < 0.5:
            suggestions.append("Verify document is not password protected")
            suggestions.append("Check if document contains mostly images")

        if metrics.get('structure', 1) < 0.5:
            suggestions.append("Use layout-aware extraction")
            suggestions.append("Process document sections separately")

        return suggestions

    def quick_check(self, text: str) -> bool:
        """
        Perform quick quality check for simple pass/fail.

        Use this for filtering during batch processing.

        Args:
            text: Extracted text to check

        Returns:
            True if text passes basic quality threshold
        """
        if not text or len(text) < 50:
            return False

        # Quick garbled check
        non_ascii = sum(1 for c in text if ord(c) > 127)
        if non_ascii / len(text) > 0.1:
            return False

        # Quick word check
        words = text.split()
        if len(words) < 10:
            return False

        avg_word_len = sum(len(w) for w in words) / len(words)
        if avg_word_len < 2 or avg_word_len > 15:
            return False

        return True