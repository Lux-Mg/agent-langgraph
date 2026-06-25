"""Definición del grafo LangGraph. Conecta los nodos y las transiciones."""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from app.state import AgentState
from app.nodes import (
    router_node,
    retrieve_node,
    reflect_node,
    answer_node,
    clarify_node,
    off_topic_node,
    after_router,
    after_reflect,
)


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("router",    router_node)
    g.add_node("retrieve",  retrieve_node)
    g.add_node("reflect",   reflect_node)
    g.add_node("answer",    answer_node)
    g.add_node("clarify",   clarify_node)
    g.add_node("off_topic", off_topic_node)

    g.add_edge(START, "router")
    g.add_conditional_edges("router", after_router, {
        "retrieve":  "retrieve",
        "clarify":   "clarify",
        "off_topic": "off_topic",
    })
    g.add_edge("retrieve", "reflect")
    g.add_conditional_edges("reflect", after_reflect, {
        "retrieve": "retrieve",
        "answer":   "answer",
    })
    g.add_edge("answer",    END)
    g.add_edge("clarify",   END)
    g.add_edge("off_topic", END)

    # MemorySaver guarda el estado por thread_id → memoria multi-turn
    return g.compile(checkpointer=MemorySaver())
