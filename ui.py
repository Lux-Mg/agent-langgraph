"""
Streamlit chat sobre el agente LangGraph. Conversación persistente con memoria
multi-turn (vía thread_id de LangGraph + MemorySaver).

Correr:  streamlit run ui.py --server.port 8502
"""
import os
import uuid
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd
from langchain_core.messages import HumanMessage

from app.graph import build_graph
from app.tools import health_check


st.set_page_config(
    page_title="RAG Agent — LenguaTrends",
    page_icon="🎬",
    layout="wide",
)


@st.cache_resource
def get_graph():
    return build_graph()


# --- session state ---
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"web-{uuid.uuid4().hex[:8]}"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_intent" not in st.session_state:
    st.session_state.last_intent = None
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []
if "last_iter" not in st.session_state:
    st.session_state.last_iter = None


# --- sidebar ---
with st.sidebar:
    st.markdown("## Agent · LenguaTrends")
    st.markdown(
        "Agente conversacional con memoria multi-turno sobre el "
        "[RAG de LenguaTrends](https://github.com/Lux-Mg/rag-lenguatrends). "
        "Routea entre `trends` y `recommend` automáticamente."
    )

    if health_check():
        st.success(f"RAG API OK ({os.getenv('RAG_API_URL', 'http://127.0.0.1:8000')})")
    else:
        st.error("RAG API no responde. Arrancalo primero.")
        st.stop()

    st.markdown(f"**Thread actual:** `{st.session_state.thread_id}`")

    if st.button("🔄 Nueva conversación", use_container_width=True):
        st.session_state.thread_id = f"web-{uuid.uuid4().hex[:8]}"
        st.session_state.messages = []
        st.session_state.last_intent = None
        st.session_state.last_sources = []
        st.session_state.last_iter = None
        st.rerun()

    st.markdown("---")
    st.markdown("### Último turno (debug)")
    if st.session_state.last_intent:
        st.markdown(f"**Intent:** `{st.session_state.last_intent}`")
        st.markdown(f"**Iteraciones del retrieve:** {st.session_state.last_iter}")
        st.markdown(f"**Docs retrievados:** {len(st.session_state.last_sources)}")
    else:
        st.caption("(después de tu primer mensaje aparecerá aquí)")

    st.markdown("---")
    st.caption("Stack: LangGraph + Qwen 2.5 7B + RAG (BGE-M3 + Qwen 3).")


# --- main: chat ---
st.markdown("### 💬 Conversación")
st.caption(
    "Probá preguntas encadenadas. El agente recuerda el contexto entre turnos "
    "y decide automáticamente si buscar tendencias o recomendar películas."
)

# render historial
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# input del usuario
user_input = st.chat_input("Escribí tu pregunta — ej: 'qué dicen los rusos sobre Mickey 17?'")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            graph = get_graph()
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            try:
                result = graph.invoke(
                    {
                        "messages": [HumanMessage(content=user_input)],
                        "user_query": user_input,
                    },
                    config=config,
                )
                answer = result.get("final_answer", "(sin respuesta)")
                st.markdown(answer)
            except Exception as e:
                answer = f"Error: {e}"
                st.error(answer)
                result = {}

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.last_intent = result.get("intent")
    st.session_state.last_iter = result.get("iteration")
    st.session_state.last_sources = result.get("retrieved_docs", []) or []
    st.rerun()


# sources del último turno
if st.session_state.last_sources:
    with st.expander(f"📋 Ver los {len(st.session_state.last_sources)} comentarios del último turno"):
        df = pd.DataFrame([
            {
                "Película": s["movie_title"],
                "Idioma": s["language"],
                "Sentiment": s["sentiment"],
                "Sim": round(s["similarity"], 3),
                "Comentario": s["text"][:200] + ("…" if len(s["text"]) > 200 else ""),
            }
            for s in st.session_state.last_sources
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)
