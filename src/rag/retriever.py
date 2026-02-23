import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


class Retriever:

    def __init__(
        self,
        mode="regulatory_only",
        upload_id=None,
        persist_dir="chromadb",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    ):

        self.mode = mode
        self.upload_id = upload_id

        self.project_root = Path(__file__).resolve().parents[2]
        self.persist_dir = self.project_root / persist_dir

        logger.info(f"Connecting to Chroma at: {self.persist_dir}")
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))

        self.embedder = SentenceTransformer(embedding_model)

        self.text_collections = []
        self.image_collections = []

        if mode in ["regulatory_only", "compare"]:
            self._load_regulatory_collections()

        if mode in ["uploaded_only", "compare"]:
            if not upload_id:
                raise ValueError("upload_id required for uploaded mode")

            self._load_uploaded_collections(upload_id)

    # --------------------------------------------
    # Regulatory Collections
    # --------------------------------------------
    def _load_regulatory_collections(self):
        # Try new name first, fall back to old name if empty or missing
        for names, bucket in [
            (["finreg_regulatory_text_store",  "finreg_text_store"],  "text"),
            (["finreg_regulatory_image_store", "finreg_image_store"], "image"),
        ]:
            loaded = False
            for name in names:
                try:
                    col = self.client.get_collection(name)
                    count = col.count()
                    if count > 0:
                        if bucket == "text":
                            self.text_collections.append(("regulatory", col))
                        else:
                            self.image_collections.append(("regulatory", col))
                        logger.info(f"[Retriever] ✓ '{name}' ({count} docs)")
                        loaded = True
                        break
                    else:
                        logger.warning(f"[Retriever] '{name}' exists but has 0 docs — trying next")
                except Exception:
                    logger.warning(f"[Retriever] '{name}' not found — trying next")
            if not loaded:
                logger.warning(f"[Retriever] No populated collection found for regulatory_{bucket}")

    # --------------------------------------------
    # Uploaded Collections
    # --------------------------------------------
    def _load_uploaded_collections(self, upload_id):

        text_name = f"finreg_uploaded_text_{upload_id}"
        image_name = f"finreg_uploaded_image_{upload_id}"

        try:
            up_text = self.client.get_collection(text_name)
            self.text_collections.append(("uploaded", up_text))
            logger.info(f"Loaded uploaded text collection: {text_name}")
        except Exception:
            logger.warning(f"Uploaded text collection not found: {text_name}")

        try:
            up_image = self.client.get_collection(image_name)
            self.image_collections.append(("uploaded", up_image))
            logger.info(f"Loaded uploaded image collection: {image_name}")
        except Exception:
            logger.warning(f"Uploaded image collection not found: {image_name}")

    # --------------------------------------------
    # Internal Search
    # --------------------------------------------
    def _search(self, collection, query_embedding, k=3):

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        logger.info(f"[Retriever] Distances returned: {distances}")
        SIMILARITY_THRESHOLD = 0.7
        filtered = []

        for d, m, dist in zip(docs, metas, distances):
            if dist <= SIMILARITY_THRESHOLD:
                filtered.append({
                    "text": d,
                    "metadata": m,
                    "distance": dist
                })

        return filtered

    # --------------------------------------------
    # Public Search
    # --------------------------------------------
    def search(self, query):
        # compare mode pulls more chunks for a proper side-by-side analysis
        k = 5 if self.mode == "compare" else 3

        logger.info(f"[Retriever] Query (k={k}): '{query[:80]}'")
        query_embedding = self.embedder.encode(query)

        output = {
            "uploaded_text":    [],
            "regulatory_text":  [],
            "uploaded_images":  [],
            "regulatory_images":[],
        }

        for label, col in self.text_collections:
            hits = self._search(col, query_embedding, k)
            key  = "uploaded_text" if label == "uploaded" else "regulatory_text"
            output[key].extend(hits)
            logger.info(f"[Retriever] {key}: {len(hits)} hit(s) from '{col.name}'")

        for label, col in self.image_collections:
            hits = self._search(col, query_embedding, k)
            key  = "uploaded_images" if label == "uploaded" else "regulatory_images"
            output[key].extend(hits)
            logger.info(f"[Retriever] {key}: {len(hits)} hit(s) from '{col.name}'")

        total = sum(len(v) for v in output.values())
        logger.info(f"[Retriever] Total hits: {total}")
        return output