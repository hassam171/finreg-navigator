import json
import logging
from pathlib import Path
import yaml
import re

from src.llm.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


class IntentAgent:

    def __init__(self):

        # --------------------------------------------
        # Project root (src → agents → file)
        # --------------------------------------------
        self.project_root = Path(__file__).resolve().parents[2]

        # --------------------------------------------
        # Load prompts.yaml
        # --------------------------------------------
        prompts_path = self.project_root / "prompts" / "prompts.yaml"

        with open(prompts_path, "r", encoding="utf-8") as f:
            prompts = yaml.safe_load(f)

        self.system_prompt = prompts["intent_classification"]["system_prompt"]
        self.user_prompt_template = prompts["intent_classification"]["user_prompt"]

        # --------------------------------------------
        # Initialize Ollama client
        # --------------------------------------------
        self.llm = OllamaClient()

        logger.info("IntentAgent initialized.")

    # -------------------------------------------------
    # MAIN ENTRY
    # -------------------------------------------------
    def run(self, state: dict) -> dict:

        query = state.get("query")
        upload_id = state.get("upload_id")
        has_upload = bool(upload_id)

        state.setdefault("progress", [])
        state["progress"].append("🧭 Determining query intent...")

        # --------------------------------------------
        # Build prompt
        # --------------------------------------------
        user_prompt = self.user_prompt_template.format(
            query=query,
            has_upload=has_upload
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # --------------------------------------------
        # Call Ollama (returns STRING)
        # --------------------------------------------
        response = self.llm.chat(messages)

        logger.info(f"[IntentAgent] Raw LLM output: {response}")

        # --------------------------------------------
        # Clean Markdown + <think> blocks
        # --------------------------------------------
        clean_res = re.sub(r"```json|```", "", response).strip()
        clean_res = re.sub(r"<think>.*?</think>", "", clean_res, flags=re.DOTALL).strip()

        # --------------------------------------------
        # Default mode
        # --------------------------------------------
        mode = "regulatory_only"

        # --------------------------------------------
        # Parse JSON safely
        # --------------------------------------------
        try:
            result = json.loads(clean_res)
            mode = result.get("mode", "regulatory_only")
        except Exception:
            logger.warning("[IntentAgent] JSON parsing failed. Using default mode.")

        # --------------------------------------------
        # Enforce allowed modes
        # --------------------------------------------
        allowed_modes = ["regulatory_only", "web"]

        if has_upload:
            allowed_modes = ["regulatory_only", "uploaded_only", "compare"]

        if mode == "web":
            state["need_web"] = True

        if mode not in allowed_modes:
            logger.warning(
                f"[IntentAgent] Invalid mode '{mode}'. Defaulting to regulatory_only."
            )
            mode = "regulatory_only"

        state["mode"] = mode

        logger.info(f"[IntentAgent] Final mode selected: {mode}")

        return state


# ==========================================================
# Standalone Test
# ==========================================================
if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    agent = IntentAgent()

    print("\n--- Test 1: No Upload ---")
    state1 = {
        "query": "What is EMI license fee?",
        "upload_id": None
    }
    result1 = agent.run(state1)
    print("Mode:", result1["mode"])

    print("\n--- Test 2: Uploaded Only ---")
    state2 = {
        "query": "Tell me only from this document what the capital requirement is.",
        "upload_id": "abc123"
    }
    result2 = agent.run(state2)
    print("Mode:", result2["mode"])

    print("\n--- Test 3: Compare ---")
    state3 = {
        "query": "Does this uploaded EMI policy comply with SBP rules?",
        "upload_id": "abc123"
    }
    result3 = agent.run(state3)
    print("Mode:", result3["mode"])