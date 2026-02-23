import logging
from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)


class QueryEmbedder:
    """
    Handles embedding of user queries.
    Must use SAME model as document embedding.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.model_name = model_name

        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        logger.info("Embedding model loaded successfully.")

    # -------------------------------------------------------
    # Single Query Embedding
    # -------------------------------------------------------

    def embed(self, query: str):
        """
        Embed a single query string.
        Returns: vector (list or numpy array)
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        return self.model.encode(query)

    # -------------------------------------------------------
    # Batch Embedding (Future Use)
    # -------------------------------------------------------

    def embed_batch(self, queries: list):
        """
        Embed multiple queries at once.
        Returns: list of vectors
        """
        if not isinstance(queries, list) or not queries:
            raise ValueError("Queries must be a non-empty list of strings.")

        return self.model.encode(queries)