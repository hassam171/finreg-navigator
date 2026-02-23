import logging
from ddgs import DDGS

logger = logging.getLogger(__name__)


class WebAgent:

    def __init__(self, max_results=3):
        self.max_results = max_results

    def run(self, state: dict) -> dict:
        state.setdefault("progress", [])
        state.setdefault("web_results", [])

        if not state.get("need_web", False):
            logger.info("[WebAgent] Skipping — KB had results.")
            return state

        query = state["query"]
        state["progress"].append("🌐 Searching web as fallback...")
        logger.info(f"[WebAgent] Searching: '{query[:80]}'")

        results = []
        seen_urls = set()

        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=self.max_results):
                    url = r.get("href")
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    results.append({
                        "title":   r.get("title", ""),
                        "url":     url,
                        "snippet": r.get("body", "")[:800],
                    })

            logger.info(f"[WebAgent] Got {len(results)} result(s).")
            state["progress"].append(
                f"🌐 Web returned {len(results)} result(s)." if results
                else "🌐 Web search returned no results."
            )

        except Exception as e:
            logger.warning(f"[WebAgent] Failed: {e}")
            state["progress"].append(f"🌐 Web search failed: {e}")

        state["web_results"] = results
        return state