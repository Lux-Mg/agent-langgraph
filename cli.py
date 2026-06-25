"""
REPL interactivo del agente. Cada sesión es una "conversación" via thread_id.

Uso:
    python cli.py                       # nueva conversación
    python cli.py --thread mythread     # retomar (en-memoria, solo en una corrida)
    python cli.py --debug               # imprime decisiones intermedias
"""
import os
import sys
import argparse
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
from app.graph import build_graph
from app.tools import health_check


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thread", default="default")
    parser.add_argument("--debug",  action="store_true")
    args = parser.parse_args()

    if not health_check():
        print(f"❌ RAG API no responde en {os.getenv('RAG_API_URL', 'http://127.0.0.1:8000')}")
        print("   Arrancalo primero:  uvicorn app.api:app --port 8000  (en el repo rag-lenguatrends)")
        sys.exit(1)
    print("✓ RAG API healthy")

    graph = build_graph()
    config = {"configurable": {"thread_id": args.thread}}

    print(f"\nThread: {args.thread}")
    print("Escribí tu pregunta (vacía o 'exit' para salir).\n")

    while True:
        try:
            user_input = input("you ▸ ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input or user_input.lower() in ("exit", "quit"):
            break

        result = graph.invoke(
            {
                "messages":   [HumanMessage(content=user_input)],
                "user_query": user_input,
            },
            config=config,
        )

        if args.debug:
            print(f"  [intent={result.get('intent')} iter={result.get('iteration')} "
                  f"sufficient={result.get('sufficient')} docs={len(result.get('retrieved_docs', []))}]")

        print(f"\nagent ▸ {result['final_answer']}\n")


if __name__ == "__main__":
    main()
