"""
PDF Extraction Module
Extracts text, images, tables from PDFs with OCR support
"""

import pymupdf
# import pymupdf.layout
import pymupdf4llm
from pymupdf4llm.helpers.check_ocr import should_ocr_page
import json
import os
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class PDFExtractor:
    """
    Complete PDF extraction with image filtering, table extraction, and OCR.

    Features:
    - Text extraction with pymupdf4llm (layout mode)
    - Automatic image extraction and filtering
    - Table detection and image extraction
    - OCR for scanned pages using LLaVA
    - Header/footer exclusion
    - Structured JSON output
    """

    def __init__(self,
                 output_dir="extracted",
                 min_image_area=50000,
                 min_file_size_kb=10,
                 min_aspect_ratio=0.3,
                 max_aspect_ratio=3.0,
                 table_dpi=200,
                 table_padding=10,
                 ocr_dpi=200,
                 image_size_limit=0.05,
                 table_strategy="lines_strict"):
        """
        Initialize PDF extractor with customizable settings.

        Args:
            output_dir: Base directory for all outputs (default: "extracted")
            min_image_area: Minimum image area in pixels² (default: 50000)
            min_file_size_kb: Minimum image file size in KB (default: 10)
            min_aspect_ratio: Minimum width/height ratio (default: 0.3)
            max_aspect_ratio: Maximum width/height ratio (default: 3.0)
            table_dpi: DPI for table image extraction (default: 200)
            table_padding: Padding around tables in pixels (default: 10)
            ocr_dpi: DPI for OCR image extraction (default: 200)
            image_size_limit: Image size as fraction of page (default: 0.05)
            table_strategy: Table detection method (default: "lines_strict")
        """

        # Output directories — support both absolute and relative paths
        self.project_root = Path(__file__).resolve().parents[2]
        output_dir = Path(output_dir)
        self.output_dir = output_dir if output_dir.is_absolute() else self.project_root / output_dir
        self.charts_dir = self.output_dir / "charts"
        self.tables_dir = self.output_dir / "tables"


        # Image filter settings
        self.min_image_area = min_image_area
        self.min_file_size_kb = min_file_size_kb
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio

        # Table extraction settings
        self.table_dpi = table_dpi
        self.table_padding = table_padding

        # OCR settings
        self.ocr_dpi = ocr_dpi

        # pymupdf4llm settings
        self.image_size_limit = image_size_limit
        self.table_strategy = table_strategy

        # Create directories
        self._setup_directories()

    def _setup_directories(self):
        """Create all required output directories."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.charts_dir, exist_ok=True)
        os.makedirs(self.tables_dir, exist_ok=True)

    def _get_pdf_name(self, pdf_path):
        """Extract PDF filename without extension."""
        return Path(pdf_path).stem

    def _filter_image(self, image_bbox, page):
        """
        Check if image should be kept based on custom filters.

        Args:
            image_bbox: Tuple (x0, y0, x1, y1) - image location
            page: PyMuPDF page object

        Returns:
            Tuple (should_keep: bool, reason: str)
        """
        # Calculate dimensions
        x0, y0, x1, y1 = image_bbox
        width = x1 - x0
        height = y1 - y0
        area = width * height

        # Filter 1: Area check
        if area < self.min_image_area:
            return False, "too small area", None

        # Filter 2: File size check
        bbox_rect = pymupdf.Rect(image_bbox)
        pix = page.get_pixmap(clip=bbox_rect, dpi=150)
        file_size_kb = len(pix.tobytes("png")) / 1024

        if file_size_kb < self.min_file_size_kb:
            return False, "file too small (blank/low quality)", file_size_kb

        # Filter 3: Aspect ratio check
        aspect_ratio = width / height if height > 0 else 0

        if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
            return False, f"bad aspect ratio ({aspect_ratio:.2f})", file_size_kb

        # Passed all filters
        return True, "passed", file_size_kb

    def _filter_images(self, chunk, page, pdf_name):
        """
        Filter all images in a chunk and delete rejected ones.

        Args:
            chunk: Page chunk dict from pymupdf4llm
            page: PyMuPDF page object
            pdf_name: PDF filename without extension

        Returns:
            Updated chunk with filtered images
        """
        filtered_images = []
        page_index = chunk['metadata']['page'] - 1  # Convert to 0-based

        for img_index, image in enumerate(chunk.get('images', [])):
            bbox = image['bbox']

            # Apply filters
            should_keep, reason, file_size_kb = self._filter_image(bbox, page)

            # Build file path
            file_path = f"{self.charts_dir}/{pdf_name}-{page_index}-{img_index}.png"

            # Add metadata
            image['filtered'] = not should_keep
            image['filter_reason'] = reason
            image['file_path'] = file_path
            image['file_size_kb'] = round(file_size_kb, 1) if file_size_kb else None

            # Delete filtered images from disk
            if not should_keep:
                if os.path.exists(file_path):
                    os.remove(file_path)

            filtered_images.append(image)

        chunk['images'] = filtered_images
        return chunk

    def _extract_table_image(self, page, table_bbox, page_num, table_num, pdf_name):
        """
        Extract a single table region as image.

        Args:
            page: PyMuPDF page object
            table_bbox: Tuple (x0, y0, x1, y1) - table location
            page_num: Page number (1-based)
            table_num: Table index on page (1-based)
            pdf_name: PDF filename without extension

        Returns:
            str: Path where image was saved
        """
        # Add padding to bbox
        bbox = pymupdf.Rect(table_bbox)
        padded_bbox = bbox + (-self.table_padding, -self.table_padding,
                              self.table_padding, self.table_padding)

        # Extract as image
        pix = page.get_pixmap(clip=padded_bbox, dpi=self.table_dpi)

        # Save with descriptive name
        image_path = f"{self.tables_dir}/{pdf_name}_page{page_num}_table{table_num}.png"
        pix.save(image_path)

        return image_path

    def _extract_tables(self, chunk, page, pdf_name):
        """
        Extract all tables from a chunk as images.

        Args:
            chunk: Page chunk dict from pymupdf4llm
            page: PyMuPDF page object
            pdf_name: PDF filename without extension

        Returns:
            Updated chunk with table image paths
        """
        page_num = chunk['metadata']['page']

        for table_num, table in enumerate(chunk.get('tables', []), start=1):
            bbox = table['bbox']

            # Extract table as image
            image_path = self._extract_table_image(
                page, bbox, page_num, table_num, pdf_name
            )

            # Add image path to metadata
            table['image_path'] = image_path

        return chunk

    def _should_ocr(self, page):
        """
        Detect if page needs OCR using official pymupdf4llm detection.

        Args:
            page: PyMuPDF page object

        Returns:
            Tuple (needs_ocr: bool, decision: dict)
        """
        decision = should_ocr_page(page, dpi=self.ocr_dpi)
        return decision['should_ocr'], decision

    def _ocr_page(self, page, page_num, pdf_name):
        """
        OCR a scanned page using PyMuPDF Tesseract.

        Args:
            page: PyMuPDF page object
            page_num: Page number (1-based)
            pdf_name: PDF filename without extension

        Returns:
            Tuple (ocr_text: str, image_path: str)
        """

        # OCR with Tesseract (GOOD method)
        tp = page.get_textpage_ocr(language="eng")
        ocr_text = page.get_text(textpage=tp)

        return ocr_text, None

    def extract(self, pdf_path):
        """
        Main extraction method - orchestrates everything.

        Args:
            pdf_path: Path to PDF file

        Returns:
            dict: Extraction result with metadata
        """
        logger.info("=" * 80)
        logger.info(f"Processing: {pdf_path}")
        logger.info("=" * 80)

        # Open PDF with PyMuPDF
        doc = pymupdf.open(pdf_path)
        pdf_name = self._get_pdf_name(pdf_path)
        total_pages = doc.page_count
        # Extract with pymupdf4llm (Layout Mode)
        logger.info("Step 1: Extracting text, images, and tables...")
        chunks = pymupdf4llm.to_markdown(
            doc,
            page_chunks=True,
            write_images=True,
            image_path=self.charts_dir,
            image_size_limit=self.image_size_limit,
            table_strategy=self.table_strategy,
            use_ocr=False,  # Disable auto-OCR (we'll do manual Tesseract)
            header=True,  # Keep headers (clean later)
            footer=True  # Keep footers (clean later)
        )
        # Step 1 - strip .pdf only from regular image files (not full page renders)
        for f in Path(self.charts_dir).glob("*.pdf-*.png"):
            if f.stem.endswith('-full'):  # skip full page renders
                continue
            new_name = f.name.replace('.pdf-', '-')
            f.rename(f.parent / new_name)


        logger.info(f"Extracted {len(chunks)} pages")

        # Tracking variables
        ocr_count = 0
        scanned_pages = []
        images_filtered = 0
        tables_extracted = 0
        # Process each page
        logger.info("Step 2: Processing pages...")

        for i, chunk in enumerate(chunks):
            page = doc[i]
            page_num = i + 1

            # Check if needs OCR
            needs_ocr, decision = self._should_ocr(page)

            if needs_ocr:
                # SCANNED PAGE - Apply OCR
                logger.info(f"Page {page_num}: Scanned - applying OCR...")

                ocr_text, image_path = self._ocr_page(page, page_num, pdf_name)

                # Replace text with OCR'd version
                chunk['text'] = ocr_text
                chunk['metadata']['ocr_applied'] = True
                chunk['metadata']['ocr_method'] = 'tesseract'
                chunk['metadata']['scanned_image'] = image_path

                # Track
                ocr_count += 1
                scanned_pages.append(page_num)

            else:
                # DIGITAL PAGE - Process normally
                logger.info(f"Page {page_num}: Digital - processing...")

                chunk['metadata']['ocr_applied'] = False

                # Filter images
                chunk = self._filter_images(chunk, page, pdf_name)
                filtered_count = len([img for img in chunk.get('images', []) if img['filtered']])
                images_filtered += filtered_count

                # Extract table images
                chunk = self._extract_tables(chunk, page, pdf_name)
                tables_extracted += len(chunk.get('tables', []))

        logger.info("Processed all pages")

        # Build JSON output
        logger.info("Step 3: Building JSON output...")

        output = {
            "filename": doc.name,
            "pdf_name": pdf_name,
            "total_pages": total_pages,
            "extraction_date": datetime.now().isoformat(),
            "extraction_summary": {
                "scanned_pages": scanned_pages,
                "ocr_count": ocr_count,
                "images_filtered": images_filtered,
                "tables_extracted": tables_extracted
            },
            "pages": chunks
        }

        # Save JSON
        json_path = f"{self.output_dir}/{pdf_name}_data.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Saved: {json_path}")
        # ---------------------------------------------------------
        # Remove images if PDF is mostly scanned
        # ---------------------------------------------------------

        if total_pages > 0 and (ocr_count / total_pages) > 0.7:
            logger.info("PDF detected as mostly scanned. Removing extracted images...")

            for f in Path(self.charts_dir).glob(f"{pdf_name}-*.png"):
                try:
                    f.unlink()
                except Exception:
                    pass

            logger.info("Scanned PDF images removed.")
        # Print summary
        logger.info("=" * 80)
        logger.info("EXTRACTION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Data file: {pdf_name}_data.json")
        logger.info(f"Charts: {len(os.listdir(self.charts_dir))} files")
        logger.info(f"Tables: {len(os.listdir(self.tables_dir))} files")
        logger.info("=" * 80)

        # Close document
        doc.close()

        # Return summary
        return {
            "success": True,
            "pdf_name": pdf_name,
            "total_pages": total_pages,
            "scanned_pages": scanned_pages,
            "ocr_count": ocr_count,
            "images_filtered": images_filtered,
            "tables_extracted": tables_extracted,
            "output_file": json_path
        }

    def extract_multiple(self, pdf_paths):
        """
        Extract multiple PDFs.

        Args:
            pdf_paths: List of PDF file paths

        Returns:
            list: Results for all PDFs
        """
        logger.info("=" * 80)
        logger.info(f"BATCH EXTRACTION: {len(pdf_paths)} PDFs")
        logger.info("=" * 80)

        results = []

        for idx, pdf_path in enumerate(pdf_paths, start=1):
            logger.info(f"[{idx}/{len(pdf_paths)}] Processing: {pdf_path}")

            try:
                result = self.extract(pdf_path)
                results.append(result)

            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {e}")
                results.append({
                    "success": False,
                    "pdf_path": pdf_path,
                    "error": str(e)
                })

        # Print final summary
        logger.info("=" * 80)
        logger.info("BATCH EXTRACTION COMPLETE")
        logger.info("=" * 80)

        successful = len([r for r in results if r.get('success')])
        failed = len(results) - successful

        logger.info(f"Successful: {successful}/{len(pdf_paths)}")
        logger.info(f"Failed: {failed}/{len(pdf_paths)}")

        if successful > 0:
            total_pages = sum(r.get('total_pages', 0) for r in results if r.get('success'))
            total_ocr = sum(r.get('ocr_count', 0) for r in results if r.get('success'))
            logger.info(f"Total pages processed: {total_pages}")
            logger.info(f"Total pages OCR'd: {total_ocr}")

        logger.info("=" * 80)

        return results


if __name__ == "__main__":
    from logs.logging_config import setup_logging

    setup_logging()

    logger.info("Running standalone PDFExtractor test...")

    project_root = Path(__file__).resolve().parent.parent.parent
    test_pdf = project_root / "data/pk_fbr_sales_tax_act_1990_updated_2025.pdf"

    if not test_pdf.exists():
        logger.error(f"Test PDF not found: {test_pdf}")
    else:
        extractor = PDFExtractor(output_dir=str(project_root / "extracted"))

        result = extractor.extract(str(test_pdf))

        if result["success"]:
            logger.info("Extraction successful.")
            logger.info(f"Output file: {result['output_file']}")

    logger.info("Done.")