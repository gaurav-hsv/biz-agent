from typing import Dict, Any
from langgraph.graph import StateGraph, END
from .agents.intent_detector import detect_intent

# state shape (keep minimal for now)
# {"session_id": str, "text": str, "intent": dict}
def _intent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    intent = detect_intent(state["text"])
    return {**state, "intent": intent}


def build_graph():
    g = StateGraph(dict)
    g.add_node("intent_detector", _intent_node)
    g.set_entry_point("intent_detector")
    g.add_edge("intent_detector", END)
    return g.compile()

app_graph = build_graph()
