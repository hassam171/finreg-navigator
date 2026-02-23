"""
Embedding Pipeline for FinReg Navigator
Creates TWO vector stores:
    1. finreg_text_store
    2. finreg_image_store
"""

import json
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import logging

logger = logging.getLogger(__name__)


class Embedder:

    def __init__(
            self,
            mode="regulatory",
            upload_id=None,
            text_chunk_dir="chunked/text_chunk",
            image_chunk_dir="chunked/image_chunk",
            persist_dir="chromadb",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            reset_collections=False
    ):

        self.project_root = Path(__file__).resolve().parents[2]
        self.text_chunk_dir = self.project_root / text_chunk_dir
        self.image_chunk_dir = self.project_root / image_chunk_dir
        self.persist_dir = str(self.project_root / persist_dir)

        # --------------------------------------------
        # Determine collection names based on mode
        # --------------------------------------------
        if mode == "regulatory":
            text_collection_name = "finreg_regulatory_text_store"
            image_collection_name = "finreg_regulatory_image_store"

        elif mode == "uploaded":
            if not upload_id:
                from datetime import datetime
                upload_id = datetime.now().strftime("%Y%m%d_%H%M%S")

            text_collection_name = f"finreg_uploaded_text_{upload_id}"
            image_collection_name = f"finreg_uploaded_image_{upload_id}"

        else:
            raise ValueError("Mode must be 'regulatory' or 'uploaded'")

        logger.info("Loading embedding model...")
        self.embedder = SentenceTransformer(embedding_model)
        logger.info("Model loaded.")

        logger.info("Initializing ChromaDB...")
        self.client = chromadb.PersistentClient(path=self.persist_dir)

        # 🔥 NEW: optional reset (used for uploaded collections)
        if reset_collections:
            try:
                self.client.delete_collection(text_collection_name)
                logger.info(f"Deleted existing collection: {text_collection_name}")
            except:
                pass

            try:
                self.client.delete_collection(image_collection_name)
                logger.info(f"Deleted existing collection: {image_collection_name}")
            except:
                pass

        self.text_collection = self.client.get_or_create_collection(text_collection_name)
        self.image_collection = self.client.get_or_create_collection(image_collection_name)

        logger.info(f"Text collection: {text_collection_name}")
        logger.info(f"Image collection: {image_collection_name}")
    # ========================================
    # INTERNAL UTILITY
    # ========================================

    def _load_json_files(self, directory: Path):
        if not directory.exists():
            logger.error(f"Directory not found: {directory}")
            return []

        files = list(directory.glob("*.json"))
        logger.info(f"Found {len(files)} files in {directory}")
        return files

    # ========================================
    # TEXT EMBEDDING
    # ========================================

    def embed_text_chunks(self):

        files = self._load_json_files(self.text_chunk_dir)

        for file in files:
            logger.info(f"Processing TEXT file: {file.name}")

            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

            chunks = data.get("chunks", [])

            ids = []
            texts = []
            metadatas = []

            for chunk in chunks:
                ids.append(chunk["id"])
                texts.append(chunk["text"])
                metadatas.append(chunk["metadata"])

            if not texts:
                logger.warning("No text chunks found.")
                continue

            logger.info(f"Embedding {len(texts)} text chunks...")
            embeddings = self.embedder.encode(texts, show_progress_bar=True)

            self.text_collection.upsert(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )

            logger.info(f"Added {len(ids)} text embeddings.")

    # ========================================
    # IMAGE EMBEDDING
    # ========================================

    def embed_image_chunks(self):

        files = self._load_json_files(self.image_chunk_dir)

        for file in files:
            logger.info(f"Processing IMAGE file: {file.name}")

            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Support both formats
            chunks = data.get("image_chunks", data.get("chunks", []))

            ids = []
            texts = []
            metadatas = []

            for chunk in chunks:
                ids.append(chunk["chunk_id"])
                texts.append(chunk["text"])
                metadatas.append({
                    "pdf_name": chunk.get("source_pdf"),
                    "page_number": chunk.get("page_number"),
                    "image_path": chunk.get("image_path"),
                    "type": chunk.get("type", "image"),
                    "file_size_kb": chunk.get("file_size_kb")
                })

            if not texts:
                logger.warning("No image chunks found.")
                continue

            logger.info(f"Embedding {len(texts)} image descriptions...")
            embeddings = self.embedder.encode(texts, show_progress_bar=True)

            self.image_collection.upsert(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )

            logger.info(f"Added {len(ids)} image embeddings.")

    # ========================================
    # RUN BOTH
    # ========================================

    def run(self):
        logger.info("=" * 80)
        logger.info("EMBEDDING PIPELINE STARTED")
        logger.info("=" * 80)

        self.embed_text_chunks()
        self.embed_image_chunks()

        logger.info("=" * 80)
        logger.info("ALL EMBEDDINGS STORED SUCCESSFULLY")
        logger.info("=" * 80)


# ========================================
# MAIN
# ========================================

if __name__ == "__main__":
    from logs.logging_config import setup_logging

    setup_logging()

    logger.info("Running standalone Embedder test...")

    # Regulatory embedding by default
    embedder = Embedder(
        mode="regulatory"
    )

    embedder.run()

    logger.info("Done.")