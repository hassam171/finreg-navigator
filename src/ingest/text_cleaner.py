"""
Text Cleaning Module for FinReg Navigator
Cleans extracted PDF text using frequency-based pattern detection
"""

import re
import json
from pathlib import Path
from collections import Counter
import logging
import os
logger = logging.getLogger(__name__)


class TextCleaner:
    """
    Cleans extracted PDF text for RAG processing

    Features:
    - Auto-detects repeated headers/footers (frequency-based)
    - Removes globally frequent boilerplate lines
    - Removes page numbers
    - Removes image references
    - Normalizes whitespace (removes 3+ consecutive blank lines)
    - Preserves semantic formatting (**bold**, #### headers)
    """

    def __init__(self, header_threshold=0.3, footer_threshold=0.3,
                 global_threshold=0.35, max_line_length=80):
        """
        Initialize text cleaner

        Args:
            header_threshold: Frequency threshold for header detection (default: 0.3 = 30%)
            footer_threshold: Frequency threshold for footer detection (default: 0.3 = 30%)
            global_threshold: Frequency threshold for globally frequent lines (default: 0.2 = 20%)
            max_line_length: Maximum line length to consider for global analysis (default: 80)
        """
        self.project_root = Path(__file__).resolve().parents[2]
        self.header_threshold = header_threshold
        self.footer_threshold = footer_threshold
        self.global_threshold = global_threshold
        self.max_line_length = max_line_length

        self.detected_headers = set()
        self.detected_footers = set()
        self.globally_frequent_lines = set()

    def analyze_document(self, json_data):
        """
        Analyze entire document to detect repeated headers/footers and globally frequent lines

        Args:
            json_data: Extracted JSON data from PDFExtractor

        Returns:
            dict: Analysis results with detected patterns
        """
        logger.info("=" * 80)
        logger.info("ANALYZING DOCUMENT FOR REPEATED PATTERNS")
        logger.info("=" * 80)

        pages = json_data.get('pages', [])
        total_pages = len(pages)

        if total_pages == 0:
            logger.warning("No pages found in document")
            return {'headers': set(), 'footers': set(), 'globally_frequent': set()}

        first_lines_counter = Counter()
        last_lines_counter = Counter()
        all_lines_counter = Counter()

        for page in pages:
            text = page.get('text', '')
            if not text.strip():
                continue

            lines = [line.strip() for line in text.split('\n') if line.strip()]

            if not lines:
                continue

            for line in lines[:3]:
                if line:
                    first_lines_counter[line] += 1

            for line in lines[-3:]:
                if line:
                    last_lines_counter[line] += 1

            for line in lines:
                if line and len(line) <= self.max_line_length:
                    all_lines_counter[line] += 1

        header_min_count = int(total_pages * self.header_threshold)
        footer_min_count = int(total_pages * self.footer_threshold)
        global_min_count = int(total_pages * self.global_threshold)

        self.detected_headers = {
            line for line, count in first_lines_counter.items()
            if count >= header_min_count
        }

        self.detected_footers = {
            line for line, count in last_lines_counter.items()
            if count >= footer_min_count
        }

        self.globally_frequent_lines = {
            line for line, count in all_lines_counter.items()
            if count >= global_min_count and not self._should_keep_line(line)
        }

        logger.info(f"Analyzed {total_pages} pages")
        logger.info(
            f"Header/Footer Threshold: {self.header_threshold * 100:.0f}% = {header_min_count} pages minimum"
        )
        logger.info(
            f"Global Threshold: {self.global_threshold * 100:.0f}% = {global_min_count} pages minimum"
        )

        if self.detected_headers:
            logger.info("DETECTED HEADERS (in first 3 lines):")
            for line in sorted(self.detected_headers):
                count = first_lines_counter[line]
                pct = (count / total_pages) * 100
                preview = line[:60] + "..." if len(line) > 60 else line
                logger.info(
                    f"'{preview}' ({count}/{total_pages} = {pct:.1f}%)"
                )
        else:
            logger.info("No repeated headers detected")

        if self.detected_footers:
            logger.info("DETECTED FOOTERS (in last 3 lines):")
            for line in sorted(self.detected_footers):
                count = last_lines_counter[line]
                pct = (count / total_pages) * 100
                preview = line[:60] + "..." if len(line) > 60 else line
                logger.info(
                    f"'{preview}' ({count}/{total_pages} = {pct:.1f}%)"
                )
        else:
            logger.info("No repeated footers detected")

        if self.globally_frequent_lines:
            logger.info("GLOBALLY FREQUENT LINES (anywhere in document):")
            for line in sorted(self.globally_frequent_lines):
                count = all_lines_counter[line]
                pct = (count / total_pages) * 100
                preview = line[:60] + "..." if len(line) > 60 else line
                logger.info(
                    f"'{preview}' ({count}/{total_pages} = {pct:.1f}%)"
                )
        else:
            logger.info("No globally frequent lines detected")

        return {
            'headers': self.detected_headers,
            'footers': self.detected_footers,
            'globally_frequent': self.globally_frequent_lines,
            'header_counts': dict(first_lines_counter),
            'footer_counts': dict(last_lines_counter),
            'global_counts': dict(all_lines_counter)
        }

    def _is_page_number(self, line):
        """
        Check if line is a page number

        Args:
            line: Text line to check

        Returns:
            bool: True if line looks like a page number
        """
        line = line.strip()

        if line.isdigit():
            return True

        if re.match(r'^Page\s+\d+', line, re.IGNORECASE):
            return True

        if re.match(r'^\d+\s*\|', line):
            return True

        return False

    def _is_image_reference(self, line):
        """
        Check if line is a markdown image reference

        Args:
            line: Text line to check

        Returns:
            bool: True if line is an image reference
        """
        return bool(re.match(r'^!\[\]\([^)]+\)', line.strip()))

    def _remove_image_references(self, text):
        """
        Remove all markdown image references from text

        Args:
            text: Text to clean

        Returns:
            str: Text with image references removed
        """
        return re.sub(r'!\[\]\([^)]+\)', '', text)

    def _should_keep_line(self, line):
        """
        Check if line should be kept even if frequent
        Protects important repeated content like data/dates

        Args:
            line: Text line to check

        Returns:
            bool: True if should keep (don't remove even if frequent)
        """
        line = line.strip()

        if re.search(r'\d+[,.]?\d*', line):
            return True

        if re.search(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b', line, re.IGNORECASE):
            return True

        if re.search(r'\d{4}', line):
            return True

        if '.' in line and not line.endswith('...'):
            return True

        if any(word in line.lower() for word in ['percent', 'million', 'billion', 'trillion', 'total', 'average']):
            return True

        return False

    def _normalize_whitespace(self, text):
        """
        Normalize whitespace while preserving structure

        Args:
            text: Text to normalize

        Returns:
            str: Normalized text
        """
        lines = text.split('\n')
        lines = [line.strip() for line in lines]

        cleaned = []
        blank_count = 0

        for line in lines:
            if line == '':
                blank_count += 1
                if blank_count <= 2:
                    cleaned.append(line)
            else:
                cleaned.append(line)
                blank_count = 0

        while cleaned and cleaned[0] == '':
            cleaned.pop(0)
        while cleaned and cleaned[-1] == '':
            cleaned.pop()

        return '\n'.join(cleaned)

    def clean_page_text(self, text, page_num=None):
        """
        Clean text from a single page

        Args:
            text: Raw text from page
            page_num: Page number (for logging)

        Returns:
            str: Cleaned text
        """
        if not text or not text.strip():
            return ""

        lines = [line.strip() for line in text.split('\n')]

        cleaned_lines = []

        for i, line in enumerate(lines):
            if not line:
                cleaned_lines.append(line)
                continue
            # ---------------------------------------------------------
            # Remove trailing page number patterns like:
            # "Page 10 of 52" at end of header/footer line (SAFE)
            # ---------------------------------------------------------
            line = re.sub(r'\s*Page\s+\d+\s+(of|/)\s+\d+\s*$',
                '',
                line,
                flags=re.IGNORECASE
            ).strip()

            if i < 3 and line in self.detected_headers:
                continue

            if i >= len(lines) - 3 and line in self.detected_footers:
                continue

            if line in self.globally_frequent_lines:
                continue

            if (i < 3 or i >= len(lines) - 3) and self._is_page_number(line):
                continue

            if self._is_image_reference(line):
                continue

            cleaned_lines.append(line)

        cleaned_text = '\n'.join(cleaned_lines)
        cleaned_text = self._remove_image_references(cleaned_text)
        cleaned_text = self._normalize_whitespace(cleaned_text)

        return cleaned_text

    def clean_document(self, json_data, output_path=None):
        """
        Clean entire document

        Args:
            json_data: Extracted JSON data from PDFExtractor
            output_path: Optional path to save cleaned JSON

        Returns:
            dict: Cleaned JSON data
        """
        logger.info("=" * 80)
        logger.info("CLEANING DOCUMENT")
        logger.info("=" * 80)

        analysis = self.analyze_document(json_data)

        logger.info("Cleaning pages...")
        cleaned_pages = []

        for page in json_data.get('pages', []):
            page_num = page.get('metadata', {}).get('page', '?')
            original_text = page.get('text', '')

            cleaned_text = self.clean_page_text(original_text, page_num)

            cleaned_page = {
                **page,
                'text': cleaned_text,
                'text_original': original_text
            }

            cleaned_pages.append(cleaned_page)

        cleaned_json = {
            **json_data,
            'pages': cleaned_pages,
            'cleaning_applied': True,
            'cleaning_summary': {
                'detected_headers': list(self.detected_headers),
                'detected_footers': list(self.detected_footers),
                'globally_frequent': list(self.globally_frequent_lines),
                'header_threshold': self.header_threshold,
                'footer_threshold': self.footer_threshold,
                'global_threshold': self.global_threshold
            }
        }

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(cleaned_json, f, indent=2, default=str)

            logger.info(f"Saved cleaned JSON: {output_path}")

        logger.info(f"Cleaned {len(cleaned_pages)} pages")

        return cleaned_json

    def clean_multiple(self, json_files, output_dir="cleaned"):
        """
        Clean multiple JSON files

        Args:
            json_files: List of JSON file paths
            output_dir: Directory to save cleaned files

        Returns:
            dict: Results for each file
        """
        output_dir = self.project_root / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        for json_file in json_files:
            json_path = Path(json_file)

            logger.info("=" * 80)
            logger.info(f"Processing: {json_path.name}")
            logger.info("=" * 80)

            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)

                output_path = output_dir / f"{json_path.stem}_cleaned.json"

                cleaned_data = self.clean_document(json_data, output_path)

                results[str(json_path)] = {
                    'success': True,
                    'output_path': str(output_path),
                    'pages_cleaned': len(cleaned_data['pages'])
                }

            except Exception as e:
                logger.error(f"Error processing {json_path.name}: {e}")
                results[str(json_path)] = {
                    'success': False,
                    'error': str(e)
                }

        return results


if __name__ == "__main__":
    from logs.logging_config import setup_logging

    setup_logging()

    logger.info("Running standalone TextCleaner test...")

    project_root = Path(__file__).resolve().parent.parent.parent
    TEST_JSON = project_root / "extracted" / "pk_sbp_financial_stability_review_2024_data.json"

    cleaner = TextCleaner(
        header_threshold=0.3,
        footer_threshold=0.3,
        global_threshold=0.2,
        max_line_length=80
    )

    if not TEST_JSON.exists():
        logger.error(f"JSON file not found: {TEST_JSON}")
    else:
        with open(TEST_JSON, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        # ✅ Absolute path to project-level cleaned directory
        output_path = project_root / "cleaned" / f"{TEST_JSON.stem}_cleaned.json"

        cleaned_data = cleaner.clean_document(json_data, output_path)

        logger.info("Cleaning complete.")
        logger.info(f"Total pages cleaned: {len(cleaned_data['pages'])}")

    logger.info("Done.")