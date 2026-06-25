"""
Wrapper del Demo #1 (RAG) como tool del agente.

El agente decide CUÁNDO llamar acá y CON QUÉ query. La separación entre
el agente (decisión) y el RAG (búsqueda) es la idea entera.
"""
import os
import requests

RAG_API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000")


def call_trends(query: str, language: str | None = None, top_k: int = 20) -> dict:
    """Llamada al endpoint /trends del RAG. Devuelve {answer, sources}."""
    resp = requests.post(
        f"{RAG_API_URL}/trends",
        json={"query": query, "language": language, "top_k": top_k},
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()


def call_recommend(query: str, language: str | None = None, top_k: int = 40) -> dict:
    """Llamada al endpoint /recommend del RAG. Devuelve {recommendations, caveats, sources}."""
    resp = requests.post(
        f"{RAG_API_URL}/recommend",
        json={"query": query, "language": language, "top_k": top_k},
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()


def health_check() -> bool:
    try:
        r = requests.get(f"{RAG_API_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False
