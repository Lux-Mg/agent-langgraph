"""
Los nodos del grafo. Cada uno recibe el state y devuelve un dict parcial
con los campos que actualiza.
"""
import os
import json
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama

from app.state import AgentState
from app.tools import call_trends, call_recommend

OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
MAX_ITER     = 2  # cuántas veces puede el reflect pedir otra búsqueda

llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_HOST, temperature=0.2)


CORPUS_INFO = """The corpus contains 17,130 YouTube comments (English, Spanish, Russian)
about 15 specific recent movies: Mickey 17, Avatar: Fire and Ash, Thunderbolts*,
Sinners, Lilo & Stitch, Project Hail Mary, Scream 7, The Super Mario Galaxy Movie,
Mission: Impossible - The Final Reckoning, Hoppers, GOAT, Send Help, Crime 101,
Pretty Lethal, Mike & Nick & Nick & Alice."""


# ---------- ROUTER ----------
ROUTER_SYS = f"""You are the planning step of a movie-comments assistant.

{CORPUS_INFO}

Given the latest user message AND the conversation history, decide ONE of:
- "trends":      analytical question about what users say (sentiment, opinions, themes about specific movies)
- "recommend":   user wants a movie recommendation based on something they describe
- "clarify":    the request is ambiguous and we need more info from the user
- "off_topic":  user asks about anything outside the corpus (movies not listed, general knowledge, math, etc)

For "trends" and "recommend", also produce:
- "rag_query": a clean self-contained search query for the retrieval system (English; rewrite the user message into a single declarative phrase that captures the intent. If the user followed up like "and in Russian?", combine with prior turn).
- "rag_language": "en", "es", "ru", or null (only set if user explicitly asks about a language or comments in a language).

Return strict JSON, no prose."""

ROUTER_SCHEMA = {
    "title": "RouterDecision",
    "description": "Classifies user intent and rewrites the query for retrieval.",
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": ["trends", "recommend", "clarify", "off_topic"],
            "description": "What the user wants.",
        },
        "rag_query":    {"type": "string", "description": "Self-contained search query for retrieval. Empty string if intent is clarify/off_topic."},
        "rag_language": {"type": "string", "enum": ["en", "es", "ru", ""], "description": "Language filter or empty string if none."},
        "reason":       {"type": "string", "description": "One-sentence explanation."},
    },
    "required": ["intent", "rag_query", "rag_language", "reason"],
}


def router_node(state: AgentState) -> dict:
    history = state.get("messages", [])
    user_query = state["user_query"]

    msgs = [SystemMessage(content=ROUTER_SYS)] + history + [
        HumanMessage(content=f"Latest user message: {user_query}\n\nReturn the JSON.")
    ]
    resp = None
    try:
        resp = llm.with_structured_output(ROUTER_SCHEMA).invoke(msgs)
    except Exception as e:
        print(f"[router] structured_output failed: {e}")

    if not resp:
        # Fallback heurístico: si menciona "recommend", asume recommend; sino trends
        intent = "recommend" if "recommend" in user_query.lower() else "trends"
        return {
            "intent": intent,
            "intent_reason": "router fallback (LLM did not return valid JSON)",
            "rag_query":  user_query,
            "rag_language": None,
            "iteration": 0,
        }

    lang_raw = resp.get("rag_language") or ""
    return {
        "intent":        resp["intent"],
        "intent_reason": resp.get("reason", ""),
        "rag_query":     resp.get("rag_query") or user_query,
        "rag_language":  lang_raw if lang_raw in ("en", "es", "ru") else None,
        "iteration":     0,
    }


# ---------- RETRIEVE ----------
def retrieve_node(state: AgentState) -> dict:
    q    = state.get("refined_query") or state["rag_query"]
    lang = state.get("rag_language")
    if state["intent"] == "trends":
        raw = call_trends(q, language=lang, top_k=20)
    else:
        raw = call_recommend(q, language=lang, top_k=40)
    return {
        "retrieved_docs": raw.get("sources", []),
        "rag_raw":        raw,
        "iteration":      state.get("iteration", 0) + 1,
    }


# ---------- REFLECT ----------
REFLECT_SYS = """You evaluate whether retrieved comments are enough to answer the user.

Given:
- The user's actual question
- The number of retrieved comments and which movies they belong to
- A short sample of those comments

Decide:
- "sufficient": true if the comments clearly cover the topic asked
- "sufficient": false if the comments are off-topic, too few, or mostly about a different movie

If insufficient, also give a "refined_query" — a sharper search phrase that would do better.

Return strict JSON."""

REFLECT_SCHEMA = {
    "title": "ReflectDecision",
    "description": "Decides whether retrieved docs are enough or a refined query is needed.",
    "type": "object",
    "properties": {
        "sufficient":    {"type": "boolean", "description": "True if the retrieved comments cover the topic."},
        "refined_query": {"type": "string", "description": "Sharper search query if insufficient; empty string if sufficient."},
        "reason":        {"type": "string", "description": "One-sentence explanation."},
    },
    "required": ["sufficient", "refined_query", "reason"],
}


def reflect_node(state: AgentState) -> dict:
    docs = state.get("retrieved_docs", [])
    if not docs:
        return {"sufficient": False, "refined_query": state["rag_query"]}

    movie_counts: dict[str, int] = {}
    for d in docs:
        movie_counts[d["movie_title"]] = movie_counts.get(d["movie_title"], 0) + 1
    movies_summary = ", ".join(f"{t}({n})" for t, n in sorted(movie_counts.items(), key=lambda kv: -kv[1]))

    sample = "\n".join(f"- [{d['movie_title']}] {d['text'][:140]}" for d in docs[:5])

    user_msg = f"""User question: {state['user_query']}

Retrieved {len(docs)} comments across these movies: {movies_summary}

Sample:
{sample}

Is this enough to answer? Return the JSON."""

    resp = None
    try:
        resp = llm.with_structured_output(REFLECT_SCHEMA).invoke([
            SystemMessage(content=REFLECT_SYS),
            HumanMessage(content=user_msg),
        ])
    except Exception as e:
        print(f"[reflect] structured_output failed: {e}")

    if not resp or "sufficient" not in resp:
        # Fallback: si retrievió suficiente cantidad, asumimos OK
        return {"sufficient": len(docs) >= 5}

    out = {"sufficient": bool(resp["sufficient"])}
    if not out["sufficient"]:
        refined = resp.get("refined_query") or ""
        out["refined_query"] = refined if refined.strip() else state["rag_query"]
    return out


# ---------- ANSWER ----------
ANSWER_SYS = """You are answering a user about movie comments. You have:
- The conversation history
- The latest user question
- Either a synthesized RAG answer (for trends) or structured recommendations (for recommend)
- The list of retrieved comments

Write a clear answer that:
1. Directly addresses the latest question
2. References specific movies from the retrieved comments — never invent
3. Acknowledges if the corpus support is weak
4. Stays concise (4-7 sentences for trends; bullet list for recommend)
5. Uses English."""


def answer_node(state: AgentState) -> dict:
    raw  = state.get("rag_raw", {})
    docs = state.get("retrieved_docs", [])

    if state["intent"] == "recommend":
        recs = raw.get("recommendations", [])
        if not recs:
            return {"final_answer": "I couldn't find strong matches in the corpus for that request.",
                    "messages": [AIMessage(content="I couldn't find strong matches in the corpus for that request.")]}
        rec_block = "\n".join(f"- {r['title']}: {r['reason']}" for r in recs)
        caveats   = raw.get("caveats", "")
        final = "Based on what real viewers say:\n\n" + rec_block + (f"\n\n_{caveats}_" if caveats else "")
        return {
            "final_answer": final,
            "messages": [AIMessage(content=final)],
        }

    # trends path: el RAG ya nos dio una respuesta sintetizada; la pulimos
    # con historial conversacional para resolver follow-ups.
    rag_answer = raw.get("answer", "")
    sample = "\n".join(f"- [{d['movie_title']} · {d['language']}] {d['text'][:160]}" for d in docs[:5])

    user_msg = f"""Latest user question: {state['user_query']}

RAG produced this draft answer:
{rag_answer}

Top retrieved comments:
{sample}

Write the final answer."""

    msgs = [SystemMessage(content=ANSWER_SYS)] + state.get("messages", []) + [HumanMessage(content=user_msg)]
    resp = llm.invoke(msgs)
    final = resp.content
    return {
        "final_answer": final,
        "messages": [AIMessage(content=final)],
    }


# ---------- CLARIFY / OFF-TOPIC ----------
def clarify_node(state: AgentState) -> dict:
    msg = (
        "I need a bit more context. Which movie or topic are you asking about? "
        "The corpus covers 15 recent titles (Mickey 17, Avatar: Fire and Ash, "
        "Thunderbolts*, Sinners, Lilo & Stitch, Mission: Impossible, Mario Galaxy, "
        "and others)."
    )
    return {"final_answer": msg, "messages": [AIMessage(content=msg)]}


def off_topic_node(state: AgentState) -> dict:
    msg = (
        "That's outside what I can answer with this corpus — I only have 17k YouTube "
        "comments about 15 specific recent movies. Try asking about one of them, "
        "or describe what kind of movie you'd like to watch."
    )
    return {"final_answer": msg, "messages": [AIMessage(content=msg)]}


# ---------- ROUTING FUNCTIONS ----------
def after_router(state: AgentState) -> str:
    intent = state["intent"]
    if intent in ("trends", "recommend"):
        return "retrieve"
    if intent == "clarify":
        return "clarify"
    return "off_topic"


def after_reflect(state: AgentState) -> str:
    if state.get("sufficient", True) or state.get("iteration", 0) >= MAX_ITER:
        return "answer"
    return "retrieve"
