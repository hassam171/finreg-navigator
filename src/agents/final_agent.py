import logging
import yaml
from pathlib import Path
from src.llm.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


class FinalAgent:

    def __init__(self, model="llama3:8b"):
        self.llm = OllamaClient(model=model)
        project_root = Path(__file__).resolve().parents[2]
        with open(project_root / "prompts" / "prompts.yaml", "r", encoding="utf-8") as f:
            self.prompts = yaml.safe_load(f)

    def run(self, state: dict) -> dict:
        query             = state["query"]
        mode              = state.get("mode", "regulatory_only")
        verbose           = state.get("verbose", False)
        regulatory_text   = state.get("regulatory_text",  [])
        uploaded_text     = state.get("uploaded_text",    [])
        regulatory_images = state.get("regulatory_images",[])
        uploaded_images   = state.get("uploaded_images",  [])
        web_results       = state.get("web_results",      [])

        state.setdefault("progress", [])
        state["progress"].append("🤖 Generating answer...")

        logger.info(
            f"[FinalAgent] mode={mode} | reg_text={len(regulatory_text)} | "
            f"upl_text={len(uploaded_text)} | web={len(web_results)}"
        )

        # ── Only include sections that actually have content ───────────────
        def fmt_blocks(blocks, label):
            if not blocks:
                return ""
            return "\n\n".join([f"[{label}]\n{b['text']}" for b in blocks])

        def fmt_web(results):
            if not results:
                return ""
            return "\n\n".join([
                f"[WEB]\nTitle: {r['title']}\nURL: {r['url']}\nSnippet: {r['snippet']}"
                for r in results
            ])

        sections = []

        reg_text = fmt_blocks(regulatory_text, "REGULATORY_KB")
        if reg_text:
            sections.append(f"--- Regulatory Knowledge Base ---\n{reg_text}")

        upl_text = fmt_blocks(uploaded_text, "UPLOADED_DOC")
        if upl_text:
            sections.append(f"--- Uploaded Document ---\n{upl_text}")

        reg_img = fmt_blocks(regulatory_images, "REGULATORY_IMAGE")
        if reg_img:
            sections.append(f"--- Regulatory Images ---\n{reg_img}")

        upl_img = fmt_blocks(uploaded_images, "UPLOADED_IMAGE")
        if upl_img:
            sections.append(f"--- Uploaded Images ---\n{upl_img}")

        web = fmt_web(web_results)
        if web:
            sections.append(f"--- Web Search Results ---\n{web}")

        context_body = "\n\n".join(sections) if sections else "No context available."

        context = f"""User Question: {query}

Execution Mode: {mode}
Verbose: {verbose}

{context_body}"""

        logger.info(f"[FinalAgent] Context: {len(context)} chars")

        response = self.llm.chat([
            {"role": "system", "content": self.prompts["final_answer_prompt"]},
            {"role": "user",   "content": context},
        ])

        state["progress"].append("✅ Answer ready.")
        logger.info("[FinalAgent] Done.")

        image_paths = (
            [i.get("image_path") for i in uploaded_images   if i.get("image_path")] +
            [i.get("image_path") for i in regulatory_images if i.get("image_path")]
        )

        return {
            **state,
            "answer": response,
            "images": [p for p in image_paths if p],
        }