"""
Document Chunker for FinReg Navigator
Uses LangChain's RecursiveCharacterTextSplitter
"""

import json
import os
from pathlib import Path
from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)

class DocumentChunker:
    """
    Chunks cleaned documents using LangChain's RecursiveCharacterTextSplitter

    Features:
    - Recursive splitting (paragraphs → lines → sentences → words)
    - Configurable chunk size and overlap
    - Preserves source page metadata
    - Saves chunks as JSON
    """

    def __init__(self, chunk_size=800, chunk_overlap=200, output_dir="chunked/text_chunk"):
        """
        Initialize document chunker

        Args:
            chunk_size: Target size of each chunk in characters (default: 1200)
            chunk_overlap: Number of characters to overlap between chunks (default: 200)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.project_root = Path(__file__).resolve().parents[2]
        self.output_dir = self.project_root / output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " "],
            length_function=len,
            is_separator_regex=False
        )

    def chunk_document(self, cleaned_json_path, output_path=None):
        """
        Chunk entire cleaned document (DOCUMENT-LEVEL splitting)
        """

        logger.info("=" * 80)
        logger.info("CHUNKING DOCUMENT (DOCUMENT-LEVEL MODE)")
        logger.info("=" * 80)

        cleaned_json_path = Path(cleaned_json_path)
        if not cleaned_json_path.is_absolute():
            cleaned_json_path = self.project_root / cleaned_json_path

        with open(cleaned_json_path, 'r', encoding='utf-8') as f:
            cleaned_json = json.load(f)

        pdf_name = cleaned_json.get('pdf_name', 'unknown')
        total_pages = cleaned_json.get('total_pages', 0)

        # ---------------------------------------------------
        # STEP 1: BUILD FULL TEXT + CHARACTER MAP
        # ---------------------------------------------------
        full_text = ""
        char_to_page_map = []

        for page in cleaned_json.get('pages', []):
            page_num = page.get('metadata', {}).get('page', '?')
            page_text = page.get('text', '')

            if not page_text.strip():
                continue

            page_content = page_text + "\n\n"
            full_text += page_content
            char_to_page_map.extend([page_num] * len(page_content))

        if not full_text.strip():
            raise ValueError("No content found to chunk.")

        # ----------------------------------------
        # STEP 2: SPLIT WHOLE DOCUMENT RECURSIVELY
        # ----------------------------------------
        chunks = self.splitter.split_text(full_text)

        # ----------------------------------------
        # STEP 3: ASSIGN METADATA TO CHUNKS
        # ----------------------------------------
        all_chunks = []
        chunk_counter = 0
        current_search_start = 0

        for chunk_text in chunks:
            if len(chunk_text.strip()) < 15:
                continue

            start_char = full_text.find(chunk_text, current_search_start)
            if start_char == -1:
                start_char = full_text.find(chunk_text)
                if start_char == -1: continue

            end_char = start_char + len(chunk_text)
            start_page = char_to_page_map[start_char]
            end_page = char_to_page_map[min(end_char - 1, len(char_to_page_map) - 1)]

            chunk_data = {
                "id": f"{pdf_name}_{chunk_counter}",
                "text": chunk_text,
                "char_count": len(chunk_text),
                "word_count": len(chunk_text.split()),
                "metadata": {
                    "pdf_name": pdf_name,
                    "start_page": start_page,
                    "end_page": end_page,
                    "chunk_index": chunk_counter,
                    "type": "text"
                }
            }

            all_chunks.append(chunk_data)
            chunk_counter += 1
            current_search_start = end_char

        # ---------------------------------------------------------
        # STEP 4: DEFINE CHUNKED_JSON (This was the missing piece!)
        # ---------------------------------------------------------
        chunked_json = {
            "pdf_name": pdf_name,
            "source_file": str(cleaned_json_path),
            "chunking_date": datetime.now().isoformat(),
            "total_pages": total_pages,
            "total_chunks": len(all_chunks),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "chunks": all_chunks
        }

        # ----------------------------------------
        # STEP 5: SAVE FILE
        # ----------------------------------------
        if output_path is None:
            output_path = self.output_dir / f"{pdf_name}_chunks.json"
        else:
            output_path = Path(output_path)
            if not output_path.is_absolute():
                output_path = self.project_root / output_path
            output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunked_json, f, indent=2, ensure_ascii=False)

        logger.info(f"Chunking Complete! Saved to: {output_path}")

        return chunked_json

    def chunk_multiple(self, cleaned_json_files):
        """
        Chunk multiple cleaned JSON files

        Args:
            cleaned_json_files: List of cleaned JSON file paths
            output_dir: Directory to save chunked files

        Returns:
            dict: Results for each file
        """
        output_dir = self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        for json_file in cleaned_json_files:
            json_path = Path(json_file)

            logger.info("=" * 80)
            logger.info(f"Processing: {json_path.name}")
            logger.info("=" * 80)

            try:
                output_path = output_dir / f"{json_path.stem.replace('_data_cleaned', '')}_chunks.json"

                chunked_data = self.chunk_document(json_path, output_path)

                results[str(json_path)] = {
                    'success': True,
                    'output_path': str(output_path),
                    'total_chunks': chunked_data['total_chunks']
                }

            except Exception as e:
                logger.error(f"Error processing {json_path.name}: {e}")
                results[str(json_path)] = {
                    'success': False,
                    'error': str(e)
                }

        return results

    def get_chunk_stats(self, chunked_json_path):
        """
        Get statistics about chunked document

        Args:
            chunked_json_path: Path to chunked JSON file

        Returns:
            dict: Statistics about chunks
        """
        with open(chunked_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        chunks = data.get('chunks', [])

        if not chunks:
            return {'error': 'No chunks found'}

        char_counts = [c['char_count'] for c in chunks]
        word_counts = [c['word_count'] for c in chunks]

        stats = {
            'total_chunks': len(chunks),
            'avg_char_count': sum(char_counts) / len(char_counts),
            'min_char_count': min(char_counts),
            'max_char_count': max(char_counts),
            'avg_word_count': sum(word_counts) / len(word_counts),
            'min_word_count': min(word_counts),
            'max_word_count': max(word_counts),
            'total_characters': sum(char_counts),
            'total_words': sum(word_counts)
        }

        return stats


if __name__ == "__main__":
    from logs.logging_config import setup_logging

    setup_logging()

    logger.info("Running standalone DocumentChunker test...")

    project_root = Path(__file__).resolve().parents[2]
    TEST_JSON = project_root / "cleaned" / "pk_sbp_financial_stability_review_2024_data_cleaned.json"

    if not TEST_JSON.exists():
        logger.error(f"Cleaned JSON not found: {TEST_JSON}")
    else:
        chunker = DocumentChunker(
            chunk_size=1200,
            chunk_overlap=200
        )

        chunked_data = chunker.chunk_document(TEST_JSON)

        output_path = chunker.output_dir / f"{TEST_JSON.stem.replace('_data_cleaned', '')}_chunks.json"

        stats = chunker.get_chunk_stats(output_path)

        logger.info(f"Total Chunks: {stats['total_chunks']}")
        logger.info(f"Avg Chars/Chunk: {stats['avg_char_count']:.0f}")
        logger.info(f"Avg Words/Chunk: {stats['avg_word_count']:.0f}")

    logger.info("Done.")