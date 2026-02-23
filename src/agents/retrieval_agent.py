import logging
from src.rag.retriever import Retriever

logger = logging.getLogger(__name__)


class RetrievalAgent:

    def run(self, state: dict) -> dict:

        query = state.get("query")
        mode = state.get("mode", "regulatory_only")
        upload_id = state.get("upload_id")

        logger.info(f"[RetrievalAgent] Mode: {mode} | upload_id: {upload_id}")

        state.setdefault("progress", [])
        state["progress"].append("🔎 Searching knowledge base...")

        retriever = Retriever(
            mode=mode,
            upload_id=upload_id
        )

        results = retriever.search(query)

        state["uploaded_text"]     = results.get("uploaded_text",    [])
        state["regulatory_text"]   = results.get("regulatory_text",  [])
        state["uploaded_images"]   = results.get("uploaded_images",  [])
        state["regulatory_images"] = results.get("regulatory_images",[])
        state["retrieval_results"] = results

        total_hits = sum(len(v) for v in results.values())
        logger.info(f"[RetrievalAgent] Total chunks retrieved: {total_hits}")

        # Log chunk previews so you can see exactly what the LLM receives
        for i, chunk in enumerate(state["regulatory_text"]):
            logger.info(f"[RetrievalAgent] reg_chunk[{i}]: {chunk['text'][:120]}")
        for i, chunk in enumerate(state["uploaded_text"]):
            logger.info(f"[RetrievalAgent] upl_chunk[{i}]: {chunk['text'][:120]}")

        if total_hits == 0:
            state["need_web"] = True
            state["progress"].append("⚠️  Nothing in KB — trying web search...")
            logger.warning("[RetrievalAgent] Zero results — web fallback.")
        else:
            state["need_web"] = False
            reg = len(state["regulatory_text"]) + len(state["regulatory_images"])
            upl = len(state["uploaded_text"])   + len(state["uploaded_images"])
            state["progress"].append(f"✅ Found {reg} regulatory chunk(s), {upl} uploaded chunk(s).")

        return state