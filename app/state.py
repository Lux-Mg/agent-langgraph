"""Estado compartido del agente. Pasa por todos los nodos del grafo."""
from typing import TypedDict, Annotated, Literal
from operator import add
from langchain_core.messages import BaseMessage


class AgentState(TypedDict, total=False):
    # historial completo de la conversación (acumulativo entre turnos)
    messages: Annotated[list[BaseMessage], add]

    # último query del usuario (solo este turno)
    user_query: str

    # qué decidió el router
    intent: Literal["trends", "recommend", "clarify", "off_topic"]
    intent_reason: str

    # query refinada que se manda al RAG (el router la formula)
    rag_query: str
    rag_language: str | None

    # qué retrievió el RAG
    retrieved_docs: list[dict]
    rag_raw: dict  # respuesta cruda del API (incluye answer del Demo #1)

    # contra-loop infinito
    iteration: int

    # decisión del reflect
    sufficient: bool
    refined_query: str | None

    # respuesta final al usuario
    final_answer: str
