# Agent over LenguaTrends RAG

A small LangGraph agent that wraps the [LenguaTrends RAG](https://github.com/Lux-Mg/rag-lenguatrends) and adds the things a single-shot RAG can't do: **multi-turn memory**, **intent routing**, and **self-reflection on retrieval quality**. Built as Demo #2 of my AI/ML portfolio.

The interesting bit isn't the LLM — it's the **graph**. Each user message flows through 3-4 decision points before producing an answer, and follow-up questions remember context from previous turns.

## What it does

```text
you ▸ what do russians say about Mickey 17?
agent ▸ Russian reviewers are split on Mickey 17. Several praise Bong Joon-ho's
        direction but feel the sci-fi side is weaker than his previous work…

you ▸ and the english reviews?
agent ▸ English-speaking commenters are more positive overall. Many highlight
        Robert Pattinson's performance…

you ▸ ok, recommend me something similar but more lighthearted
agent ▸ Based on what real viewers say:
        - The Super Mario Galaxy Movie: comments emphasize the nostalgic warmth…
        - Lilo & Stitch: viewers describe it as a comforting family rewatch…
```

Three things happened here that the previous demo couldn't do:
1. *"and the english reviews?"* — the agent recovered the prior topic (Mickey 17) from conversation memory and re-issued the query against the English corpus.
2. *"recommend me something similar but more lighthearted"* — the agent routed the same conversation through `/recommend` instead of `/trends`.
3. Self-correction: if the first retrieval misses, the agent re-queries with a refined phrase (up to 2 retries) before answering.

## Architecture

```
                          ┌─────────────┐
   user message  ──────▶  │   ROUTER    │ ◀── classifies intent + rewrites query
                          └──────┬──────┘
                                 │
                ┌────────────────┼────────────────┐
                ▼                ▼                ▼
        ┌────────────┐   ┌────────────┐   ┌──────────────┐
        │  RETRIEVE  │   │  CLARIFY   │   │  OFF_TOPIC   │
        │  (calls    │   │   (asks    │   │  (declines   │
        │  RAG API)  │   │   user)    │   │   politely)  │
        └─────┬──────┘   └─────┬──────┘   └──────┬───────┘
              │                │                 │
              ▼                ▼                 ▼
        ┌────────────┐                           │
        │  REFLECT   │ ◀── checks coverage       │
        │            │     of retrieved docs     │
        └─────┬──────┘                           │
              │                                  │
       insufficient                              │
              │                                  │
              └──▶ loop back to RETRIEVE         │
                   (max 2 retries)               │
              │                                  │
       sufficient                                │
              ▼                                  │
        ┌────────────┐                           │
        │   ANSWER   │ ◀── grounded answer +     │
        │            │     conversation memory   │
        └─────┬──────┘                           │
              │                                  │
              ▼                                  ▼
              ────────────── END ────────────────
```

The state passed between nodes (`AgentState`) carries the full message history (so each turn sees prior context), the current intent, the retrieved documents, and an iteration counter that prevents reflect/retrieve loops.

## Stack

- **LangGraph 0.2** — graph orchestration with built-in `MemorySaver` checkpointer (one thread = one conversation)
- **LangChain Core** — message types and structured output via `with_structured_output(schema)`
- **Ollama + Qwen 2.5 7B** — the current default local LLM for routing, reflection, and answer synthesis. The model is configurable through `OLLAMA_MODEL`.
- **Demo #1 RAG** — used as a tool over HTTP. The agent does NOT re-implement retrieval; it makes `POST /trends` and `POST /recommend` calls to the running RAG service.

## Running it

You need the [RAG service from Demo #1](https://github.com/Lux-Mg/rag-lenguatrends) running on port 8000. Then:

```bash
git clone https://github.com/Lux-Mg/agent-langgraph.git
cd agent-langgraph
cp .env.example .env

python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# CLI — REPL for quick testing
python cli.py
python cli.py --debug     # show intent / iteration / docs_count after each turn

# Web UI (Streamlit) — chat with persistent conversation
streamlit run ui.py --server.port 8502
# Open:  http://localhost:8502
```

### The Streamlit UI

`ui.py` exposes the agent as a chat. Each browser session is one conversation — the agent remembers prior turns via a `thread_id` stored in `st.session_state`. The sidebar shows the current intent, iteration count, and number of retrieved docs after each turn (a debug-friendly view of what the graph just did). An expander below the chat shows the comments that fed the last answer.

The UI talks to the agent in-process via `graph.invoke` — no HTTP. The graph itself talks to the RAG service over HTTP. So: UI → graph → RAG service. The graph object is created once per Streamlit process (`@st.cache_resource`) so the LLM and embeddings don't reload on each rerun.

## Repository layout

```
.
├── app/
│   ├── state.py           # AgentState TypedDict (shared across nodes)
│   ├── tools.py           # HTTP wrappers for the RAG API
│   ├── nodes.py           # router · retrieve · reflect · answer · clarify · off_topic
│   └── graph.py           # StateGraph wiring + MemorySaver checkpointer
├── cli.py                 # REPL
├── ui.py                  # Streamlit chat UI (browser-based front-end)
├── test_agent.py          # smoke test: 4 turns covering trends/follow-up/recommend/off-topic
├── requirements.txt
├── .env.example
└── README.md
```

`test_agent.py` is a live four-turn smoke-test script rather than an isolated unit test: it requires both Ollama and the RAG service (including its corpus) to be running.

## What I learned

In rough order of usefulness:

1. **Structured output is what makes deterministic routing possible.** Free-text "the intent is recommend" parsing breaks 5% of the time and ruins the graph. `llm.with_structured_output(schema)` makes the router behave like a function call, not a chat.
2. **The reflect node pays for itself.** When the first retrieval is off-topic (e.g. "nostalgic childhood games" hitting *Send Help* before *The Super Mario Galaxy Movie*), a reformulated query like *"comments about a Mario movie referencing childhood and old games"* lands the right movie. Without reflect, the answer would be confidently wrong.
3. **Conversation memory ≠ retrieval memory.** The agent remembers what the user said earlier; the RAG does NOT. Each retrieval is stateless — what changes is the query the agent CONSTRUCTS based on conversation context. This is the right separation.
4. **`MemorySaver` is enough for a demo; SQLite/Postgres is needed for production.** With `MemorySaver`, threads die when the process restarts. LangGraph's interface lets you swap to `SqliteSaver` or `PostgresSaver` without touching the graph.
5. **Keep the RAG as a separate service.** Tempting to import `app.rag` directly from this project. Going over HTTP forces a clean contract, lets the RAG scale independently, and makes the agent reusable over any RAG endpoint.

## Limitations

- **In-memory checkpointer.** Conversations don't survive process restart. SQLite checkpointer is a one-line swap if needed.
- **No streaming.** The CLI waits for the full answer before printing. LangGraph supports `astream_events` for token streaming — left out to keep the demo focused.
- **No tool registry.** The agent only knows about two tools (`trends`, `recommend`) hard-coded into nodes. A real product would expose tools to the LLM via tool-calling, which Qwen 2.5 supports.
- **No evaluation suite for the agent.** Evaluating multi-turn agents is its own rabbit hole (TRIAD scoring, conversational benchmarks). The retrieval inside is already evaluated in Demo #1.
- **The off-topic branch is a hard refusal.** A nicer version would acknowledge the question and pivot back to corpus.

---

Built by Luis Mendoza · [GitHub](https://github.com/Lux-Mg) · 2026
