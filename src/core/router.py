"""
Router - FinReg Navigator

Upload lifecycle:
  uploads/{upload_id}/   ← raw files from Streamlit
  temp/{upload_id}/      ← pipeline intermediates (deleted after query)
  chromadb/              ← ephemeral uploaded collections (deleted after query)
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from logs.logging_config import setup_logging
from src.graph.query_graph import build_query_graph
from src.ingest.pdf_extractor import PDFExtractor
from src.ingest.text_cleaner import TextCleaner
from src.ingest.document_chunker import DocumentChunker
from src.ingest.image_describer import ImageDescriber
from src.ingest.embed import Embedder

logger = logging.getLogger(__name__)

PDF_TYPES   = {".pdf"}
IMAGE_TYPES = {".png", ".jpg", ".jpeg"}


class Router:

    def __init__(self):
        # Always ensure logging is active — even if caller forgot setup_logging()
        setup_logging()

        self.graph        = build_query_graph()
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.chromadb_dir = self.project_root / "chromadb"
        self.uploads_dir  = self.project_root / "uploads"
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

        logger.info("[Router] Initialized. ChromaDB: %s", self.chromadb_dir)

    # ─── Save raw Streamlit bytes ──────────────────────────────────────────

    def save_uploaded_files(self, files: list, upload_id: str) -> List[Path]:
        """
        files = [(filename: str, file_bytes: bytes), ...]

        Streamlit usage:
            router.save_uploaded_files(
                [(f.name, f.read()) for f in st.file_uploader(...)],
                upload_id=st.session_state["session_id"]
            )
        """
        dest_dir = self.uploads_dir / upload_id
        dest_dir.mkdir(parents=True, exist_ok=True)

        saved = []
        for filename, file_bytes in files:
            dest = dest_dir / filename
            dest.write_bytes(file_bytes)
            logger.info(f"[Router] Saved: {dest.name} ({len(file_bytes)//1024} KB)")
            saved.append(dest)
        return saved

    # ─── Temp dirs ────────────────────────────────────────────────────────

    def _make_temp_dirs(self, upload_id: str) -> dict:
        base = self.project_root / "temp" / upload_id
        dirs = {
            "base":        base,
            "extracted":   base / "extracted",
            "cleaned":     base / "cleaned",
            "text_chunk":  base / "chunked" / "text_chunk",
            "image_chunk": base / "chunked" / "image_chunk",
        }
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        return dirs

    # ─── Cleanup ──────────────────────────────────────────────────────────

    def _cleanup(self, upload_id: str):
        logger.info(f"[Router] Cleaning up session: {upload_id}")

        for folder in [
            self.uploads_dir / upload_id,
            self.project_root / "temp" / upload_id,
        ]:
            if folder.exists():
                shutil.rmtree(folder)
                logger.info(f"[Router] Deleted: {folder}")

        try:
            import chromadb as _chroma
            client = _chroma.PersistentClient(path=str(self.chromadb_dir))
            for name in [
                f"finreg_uploaded_text_{upload_id}",
                f"finreg_uploaded_image_{upload_id}",
            ]:
                try:
                    client.delete_collection(name)
                    logger.info(f"[Router] Deleted collection: {name}")
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"[Router] ChromaDB cleanup error: {e}")

    # ─── Ingestion pipeline ───────────────────────────────────────────────

    def _run_ingestion(self, upload_id: str) -> dict:
        session_dir = self.uploads_dir / upload_id

        if not session_dir.exists():
            raise FileNotFoundError(
                f"Upload dir not found for {upload_id}. Call save_uploaded_files() first."
            )

        all_files = [f for f in session_dir.iterdir() if f.is_file()]
        pdfs      = [f for f in all_files if f.suffix.lower() in PDF_TYPES]
        images    = [f for f in all_files if f.suffix.lower() in IMAGE_TYPES]
        others    = [f for f in all_files if f not in pdfs and f not in images]

        if others:
            logger.warning(f"[Router] Skipping unsupported: {[f.name for f in others]}")

        if not pdfs and not images:
            raise ValueError("No supported files (PDF or image) found in upload dir.")

        logger.info(f"[Router] Ingesting — PDFs: {len(pdfs)}, Images: {len(images)}")

        dirs = self._make_temp_dirs(upload_id)

        extractor = PDFExtractor(output_dir=str(dirs["extracted"]))
        cleaner   = TextCleaner()
        chunker   = DocumentChunker(output_dir=str(dirs["text_chunk"]))
        describer = ImageDescriber(
            output_dir = str(dirs["image_chunk"]),
            charts_dir = str(dirs["extracted"] / "charts"),
            tables_dir = str(dirs["extracted"] / "tables"),
        )

        # ── PDFs ──────────────────────────────────────────────────────────
        for pdf in pdfs:
            logger.info(f"[Router] Processing PDF: {pdf.name}")
            result = extractor.extract(str(pdf))

            if not result.get("success"):
                logger.error(f"[Router] Extraction failed: {pdf.name}")
                continue

            json_path = result["output_file"]
            with open(json_path) as f:
                json_data = json.load(f)

            cleaned_path = dirs["cleaned"] / f"{pdf.stem}_cleaned.json"
            cleaner.clean_document(json_data, cleaned_path)
            chunker.chunk_document(cleaned_path)
            describer.describe_document(json_path)

        # ── Standalone images ─────────────────────────────────────────────
        for img in images:
            logger.info(f"[Router] Processing image: {img.name}")
            describer.describe_single_image(str(img))

        # ── Embed ─────────────────────────────────────────────────────────
        logger.info(f"[Router] Embedding into uploaded collections ({upload_id})")
        Embedder(
            mode              = "uploaded",
            upload_id         = upload_id,
            text_chunk_dir    = str(dirs["text_chunk"]),
            image_chunk_dir   = str(dirs["image_chunk"]),
            reset_collections = True,
        ).run()

        logger.info(f"[Router] Ingestion complete for {upload_id}")
        return dirs

    # ─── Main entry point ─────────────────────────────────────────────────

    def handle_input(
        self,
        query:     Optional[str]         = None,
        files:     Optional[List[tuple]] = None,   # [(filename, bytes), ...]
        upload_id: Optional[str]         = None,
    ) -> dict:
        """
        Pattern A — files + query (most common from Streamlit):
            router.handle_input(query="...", files=[("doc.pdf", bytes)])
            → ingest → query → cleanup → return result

        Pattern B — upload first, query later (multi-turn):
            r = router.handle_input(files=[("doc.pdf", bytes)])
            upload_id = r["upload_id"]
            result = router.handle_input(query="...", upload_id=upload_id)

        Pattern C — no files, pure regulatory query:
            router.handle_input(query="What is the EMI license fee?")
        """

        if files and not upload_id:
            upload_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # ── Pattern A / B-step-1: files provided ──────────────────────────
        if files:
            self.save_uploaded_files(files, upload_id)

            if query:
                try:
                    self._run_ingestion(upload_id)
                    logger.info(f"[Router] Running query | upload_id={upload_id}")
                    return self.graph.invoke({"query": query, "upload_id": upload_id})
                finally:
                    self._cleanup(upload_id)
            else:
                self._run_ingestion(upload_id)
                return {
                    "message": "Files ingested. Send your query with this upload_id.",
                    "upload_id": upload_id,
                }

        # ── Pattern B-step-2: follow-up query for prior upload ─────────────
        if query and upload_id:
            logger.info(f"[Router] Follow-up query | upload_id={upload_id}")
            try:
                return self.graph.invoke({"query": query, "upload_id": upload_id})
            finally:
                self._cleanup(upload_id)

        # ── Pattern C: pure regulatory query ──────────────────────────────
        if query:
            logger.info(f"[Router] Regulatory query: '{query[:80]}'")
            return self.graph.invoke({"query": query})

        return {"message": "No input provided."}