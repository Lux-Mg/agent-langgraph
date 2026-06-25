"""Smoke test del agente: 3 turnos consecutivos validando memoria multi-turn."""
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from app.graph import build_graph
from app.tools import health_check


def turn(graph, config, msg, label):
    print(f"\n{'='*70}\n{label}")
    print(f"USER  ▸ {msg}")
    result = graph.invoke(
        {"messages": [HumanMessage(content=msg)], "user_query": msg},
        config=config,
    )
    print(f"INTENT▸ {result.get('intent')}  (iter={result.get('iteration')}, "
          f"sufficient={result.get('sufficient')}, docs={len(result.get('retrieved_docs', []))})")
    print(f"AGENT ▸ {result.get('final_answer', '<no answer>')[:500]}")
    return result


def main():
    assert health_check(), "RAG API no responde — arrancalo antes"
    graph  = build_graph()
    config = {"configurable": {"thread_id": "smoke-test"}}

    # 1. Trends sobre una peli específica
    turn(graph, config, "What do Russian viewers say about Mickey 17?", "TURN 1 — trends RU")

    # 2. Follow-up que requiere memoria del turno anterior
    turn(graph, config, "And what about english reviews of that same movie?", "TURN 2 — follow-up (needs memory)")

    # 3. Switch a recommend en la misma conversación
    turn(graph, config, "Recommend me something similar but more lighthearted", "TURN 3 — recommend (different intent)")

    # 4. Off-topic
    turn(graph, config, "Who directed Pulp Fiction?", "TURN 4 — off-topic (not in corpus)")


if __name__ == "__main__":
    main()
