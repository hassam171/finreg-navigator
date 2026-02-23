from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END

from src.agents.intent import IntentAgent
from src.agents.retrieval_agent import RetrievalAgent
from src.agents.web_agent import WebAgent
from src.agents.final_agent import FinalAgent


class GraphState(TypedDict, total=False):
    query:             str
    upload_id:         Optional[str]
    mode:              str
    need_web:          bool
    uploaded_text:     List[dict]
    regulatory_text:   List[dict]
    uploaded_images:   List[dict]
    regulatory_images: List[dict]
    retrieval_results: dict
    web_results:       List[dict]
    answer:            str
    images:            List[str]
    progress:          List[str]


def _route_after_retrieval(state: dict) -> str:
    return "web" if state.get("need_web", False) else "final"


def build_query_graph():
    intent_agent    = IntentAgent()
    retrieval_agent = RetrievalAgent()
    web_agent       = WebAgent()
    final_agent     = FinalAgent()

    builder = StateGraph(GraphState)
    builder.add_node("intent",   intent_agent.run)
    builder.add_node("retrieve", retrieval_agent.run)
    builder.add_node("web",      web_agent.run)
    builder.add_node("final",    final_agent.run)

    builder.set_entry_point("intent")
    builder.add_edge("intent", "retrieve")
    builder.add_conditional_edges(
        "retrieve",
        _route_after_retrieval,
        {"web": "web", "final": "final"}
    )
    builder.add_edge("web",   "final")
    builder.add_edge("final", END)

    return builder.compile()