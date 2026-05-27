# extractors/table_extractor.py
import camelot
import pdfplumber
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import json

from .base import BaseExtractor, ExtractionResult, DocumentType

logger = logging.getLogger(__name__)


class TableExtractor(BaseExtractor):
    """
    Extract tables from PDF documents.

    Uses multiple strategies:
    1. Camelot (lattice mode) for tables with visible borders
    2. Camelot (stream mode) for tables without borders
    3. pdfplumber as fallback

    Configuration options:
        - flavor: 'lattice' (bordered), 'stream' (borderless), or 'auto'
        - min_accuracy: Minimum accuracy threshold (default: 80)
        - pages: Pages to extract from (default: 'all')
    """

    def _setup(self):
        """Configure table extraction settings."""
        self.flavor = self.config.get('flavor', 'auto')
        self.min_accuracy = self.config.get('min_accuracy', 80)
        self.pages = self.config.get('pages', 'all')

    def can_handle(self, file_path: str) -> bool:
        """Check if file is a PDF."""
        return Path(file_path).suffix.lower() == '.pdf'

    def extract(self, file_path: str) -> ExtractionResult:
        """
        Extract tables from a PDF document.

        Tries multiple extraction methods and returns the best results.

        Args:
            file_path: Path to PDF file

        Returns:
            ExtractionResult with extracted tables
        """
        self.validate_file(file_path)

        tables: List[Dict[str, Any]] = []
        errors: List[str] = []

        # Try Camelot first (best for structured tables)
        camelot_tables = self._extract_with_camelot(file_path)
        tables.extend(camelot_tables)

        # If no tables found, try pdfplumber
        if not tables:
            logger.info("No tables found with Camelot, trying pdfplumber")
            plumber_tables = self._extract_with_pdfplumber(file_path)
            tables.extend(plumber_tables)

        # Calculate quality score
        quality_score = self._calculate_table_quality(tables)

        return ExtractionResult(
            texts=[],  # Table extractor focuses on tables
            tables=tables,
            document_type=DocumentType.PDF,
            extraction_method="table_extraction",
            quality_score=quality_score,
            errors=errors
        )

    def _extract_with_camelot(
        self,
        file_path: str
    ) -> List[Dict[str, Any]]:
        """
        Extract tables using Camelot library.

        Camelot uses computer vision to detect table structures
        and is very accurate for PDFs with clear table layouts.

        Args:
            file_path: Path to PDF

        Returns:
            List of table dictionaries
        """
        tables = []

        try:
            # Determine which flavor to use
            flavors = ['lattice', 'stream'] if self.flavor == 'auto' else [self.flavor]

            for flavor in flavors:
                logger.info(f"Extracting tables with Camelot ({flavor} mode)")

                extracted = camelot.read_pdf(
                    file_path,
                    pages=str(self.pages) if self.pages != 'all' else 'all',
                    flavor=flavor,
                    suppress_stdout=True
                )

                for i, table in enumerate(extracted):
                    # Check accuracy threshold
                    if table.accuracy < self.min_accuracy:
                        logger.debug(
                            f"Skipping table {i} with low accuracy: {table.accuracy}"
                        )
                        continue

                    # Convert to dictionary format
                    df = table.df

                    # Try to use first row as header if it looks like headers
                    if self._row_looks_like_header(df.iloc[0]):
                        df.columns = df.iloc[0]
                        df = df[1:].reset_index(drop=True)

                    tables.append({
                        'data': df.to_dict('records'),
                        'columns': list(df.columns),
                        'rows': len(df),
                        'page': table.page,
                        'accuracy': table.accuracy,
                        'extraction_method': f'camelot_{flavor}',
                        'markdown': df.to_markdown(index=False),
                        'csv': df.to_csv(index=False)
                    })

                # If we found tables with lattice, no need to try stream
                if tables and flavor == 'lattice':
                    break

        except Exception as e:
            logger.warning(f"Camelot extraction failed: {str(e)}")

        return tables

    def _extract_with_pdfplumber(
        self,
        file_path: str
    ) -> List[Dict[str, Any]]:
        """
        Extract tables using pdfplumber library.

        pdfplumber uses different algorithms and can catch tables
        that Camelot misses.

        Args:
            file_path: Path to PDF

        Returns:
            List of table dictionaries
        """
        tables = []

        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    # Skip pages if specific pages requested
                    if self.pages != 'all':
                        page_list = [int(p) for p in str(self.pages).split(',')]
                        if page_num not in page_list:
                            continue

                    # Extract tables from page
                    page_tables = page.extract_tables()

                    for i, table_data in enumerate(page_tables):
                        if not table_data or len(table_data) < 2:
                            continue

                        # Convert to DataFrame
                        header = table_data[0]
                        rows = table_data[1:]

                        # Clean up None values
                        header = [str(h) if h else f'col_{j}' for j, h in enumerate(header)]
                        rows = [[str(cell) if cell else '' for cell in row] for row in rows]

                        df = pd.DataFrame(rows, columns=header)

                        tables.append({
                            'data': df.to_dict('records'),
                            'columns': list(df.columns),
                            'rows': len(df),
                            'page': page_num,
                            'accuracy': 75.0,  # pdfplumber does not provide accuracy
                            'extraction_method': 'pdfplumber',
                            'markdown': df.to_markdown(index=False),
                            'csv': df.to_csv(index=False)
                        })

        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {str(e)}")

        return tables

    def _row_looks_like_header(self, row: pd.Series) -> bool:
        """
        Heuristic to detect if a row is a header row.

        Headers typically:
        - Have no numeric values
        - Are shorter than data rows
        - Have unique values
        """
        values = [str(v) for v in row.values if v]

        # Check if any values are purely numeric (unlikely for headers)
        numeric_count = sum(1 for v in values if v.replace('.', '').isdigit())
        if numeric_count > len(values) / 2:
            return False

        # Check for duplicate values (headers usually unique)
        if len(values) != len(set(values)):
            return False

        return True

    def _calculate_table_quality(
        self,
        tables: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate overall quality score for extracted tables.

        Considers:
        - Individual table accuracy scores
        - Table completeness (no empty cells)
        - Column consistency
        """
        if not tables:
            return 0.0

        scores = []
        for table in tables:
            accuracy = table.get('accuracy', 50) / 100

            # Check for empty cells
            data = table.get('data', [])
            if data:
                total_cells = len(data) * len(table.get('columns', []))
                empty_cells = sum(
                    1 for row in data
                    for val in row.values()
                    if not str(val).strip()
                )
                completeness = 1 - (empty_cells / max(total_cells, 1))
            else:
                completeness = 0

            scores.append((accuracy * 0.6 + completeness * 0.4))

        return sum(scores) / len(scores)

    def tables_to_text(
        self,
        tables: List[Dict[str, Any]],
        format: str = 'markdown'
    ) -> str:
        """
        Convert extracted tables to text format for RAG indexing.

        Args:
            tables: List of extracted table dictionaries
            format: Output format ('markdown', 'csv', 'json')

        Returns:
            Tables as formatted text
        """
        output_parts = []

        for i, table in enumerate(tables, start=1):
            header = f"## Table {i} (Page {table.get('page', 'unknown')})\n"

            if format == 'markdown':
                content = table.get('markdown', '')
            elif format == 'csv':
                content = table.get('csv', '')
            elif format == 'json':
                content = json.dumps(table.get('data', []), indent=2)
            else:
                content = table.get('markdown', '')

            output_parts.append(header + content)

        return '\n\n'.join(output_parts)