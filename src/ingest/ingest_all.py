"""
Master Ingestion Pipeline - FinReg Navigator

Pipeline:
1. Extract PDFs
2. Clean extracted JSON
3. Chunk cleaned text
4. Describe images
5. Embed everything into ChromaDB
"""

import logging
from pathlib import Path

from logs.logging_config import setup_logging
from src.ingest.pdf_extractor import PDFExtractor
from src.ingest.text_cleaner import TextCleaner
from src.ingest.document_chunker import DocumentChunker
from src.ingest.image_describer import ImageDescriber
from src.ingest.embed import Embedder


logger = logging.getLogger(__name__)


def main():

    setup_logging()

    logger.info("=" * 80)
    logger.info("FULL INGESTION PIPELINE STARTED")
    logger.info("=" * 80)

    # ---------------------------------------------------------
    # Resolve project root
    # ---------------------------------------------------------
    project_root = Path(__file__).resolve().parent.parent.parent

    data_dir = project_root / "data"
    extracted_dir = project_root / "extracted"
    cleaned_dir = project_root / "cleaned"
    text_chunk_dir = project_root / "chunked" / "text_chunk"
    image_chunk_dir = project_root / "chunked" / "image_chunk"
    chroma_dir = project_root / "chromadb"

    # ---------------------------------------------------------
    # 1️⃣ Extract PDFs
    # ---------------------------------------------------------
    pdf_files = list(data_dir.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in {data_dir}")
        return

    extractor = PDFExtractor(output_dir=str(extracted_dir))

    extraction_results = extractor.extract_multiple(
        [str(p) for p in pdf_files]
    )

    successful_extractions = [
        r for r in extraction_results if r.get("success")
    ]

    if not successful_extractions:
        logger.error("No PDFs extracted successfully. Stopping pipeline.")
        return

    # ---------------------------------------------------------
    # 2️⃣ Clean Extracted JSON
    # ---------------------------------------------------------
    extracted_json_files = list(extracted_dir.glob("*_data.json"))

    cleaner = TextCleaner()

    cleaning_results = cleaner.clean_multiple(
        extracted_json_files,
        output_dir=cleaned_dir
    )

    successful_cleaned = [
        k for k, v in cleaning_results.items() if v.get("success")
    ]

    if not successful_cleaned:
        logger.error("No files cleaned successfully. Stopping pipeline.")
        return

    # ---------------------------------------------------------
    # 3️⃣ Chunk Cleaned Text
    # ---------------------------------------------------------
    cleaned_json_files = list(cleaned_dir.glob("*_cleaned.json"))

    chunker = DocumentChunker()

    chunk_results = chunker.chunk_multiple(
        cleaned_json_files
    )

    successful_chunks = [
        k for k, v in chunk_results.items() if v.get("success")
    ]

    if not successful_chunks:
        logger.error("No files chunked successfully. Stopping pipeline.")
        return

    # ---------------------------------------------------------
    # 4️⃣ Describe Images
    # ---------------------------------------------------------
    describer = ImageDescriber(
        prompts_path=str(project_root / "prompts" / "prompts.yaml"),
        output_dir=str(image_chunk_dir),
        model="gpt-4o-mini",
        delay_seconds=2
    )

    describe_results = describer.describe_multiple(
        extracted_json_files
    )

    successful_descriptions = [
        k for k, v in describe_results.items() if v.get("success")
    ]

    if not successful_descriptions:
        logger.warning("No image descriptions generated.")

    # ---------------------------------------------------------
    # 5️⃣ Embed Everything
    # ---------------------------------------------------------
    embedder = Embedder(
        mode="regulatory"
    )

    embedder.run()

    logger.info("=" * 80)
    logger.info("FULL INGESTION PIPELINE COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()