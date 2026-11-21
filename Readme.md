# AI Banking Support & Advisory Agent (Non-Transactional)

Project | Prithwiraj Choudhury | 06-10-2026

A customer-facing assistant that answers questions about loans, deposits, cards, payment systems and complaints, grounded in a knowledge base of 13 RBI/SBI documents. It is **read-only**: it explains and calculates, but never moves money, changes accounts or gives personalised financial advice. Every factual answer carries a `(file.pdf, p.N)` citation.

---

## 1. Problem and scope

| | |
| --- | --- |
| **Does** | Answer policy, rate, eligibility, liability, TAT and complaint-process questions; compute EMIs; explain how to raise a complaint. |
| **Does not** | Transfer money, block/unblock cards, open/close accounts, look up identity or account data, give personalised investment advice. |
| **Hands off** | Fraud reports, disputes and distressed customers go to `escalate_to_human`. |
| **Assumes** | The user is already authenticated in the bank's app; the agent reads no account data. |

Full persona, constraints and success criteria: `Problem Framing Document.pdf`.

---

## 2. Architecture

### 2.1 Agent graph (LangGraph `StateGraph`, 4 nodes, 3 tools)

```
 user question
      │
      ▼
 ┌──────────────────┐   mask Aadhaar / account / card / CVV / PIN / OTP / PAN /
 │ Input masking    │   phone / email before the LLM, the logs and feedback file
 └────────┬─────────┘
          ▼
 ┌──────────────────┐   gpt-4o-mini, temp 0.1, tools bound
 │   supervisor     │◄──────────────────────────┐
 └────────┬─────────┘                           │
          │ router (should_continue)            │
   tool calls?──────────► ┌─────────┐ ───────────┘
          │               │  tools  │  kb_lookup · emi_calculator · escalate_to_human
          │               └─────────┘
          │ > 6 rounds
          ├──────────────► finalize ──► guardrails ──► END
          │ final answer
          ▼
   guardrail_check ──► (citation + disclaimer checks, one LLM rewrite, then deterministic append) ──► END
```

| Node | Role |
| --- | --- |
| `supervisor` | LLM with tools bound; decides whether to search, calculate, escalate or answer. System prompt + style slot + chat history. |
| `tools` | LangGraph `ToolNode` executing the three read-only tools. Loops back to `supervisor`. |
| `finalize` | Reached when `MAX_TOOL_ROUNDS = 6` is exceeded; forces an answer from what was already retrieved. |
| `guardrail_check` | Output check: citation present if `kb_lookup` returned sources; fixed disclaimer on personal-advice questions. |

### 2.2 Tools

| Tool | Purpose |
| --- | --- |
| `kb_lookup` | Hybrid RAG search over the knowledge base. Returns chunks tagged with file and page; chunks are also kept as a tool artifact so sources can be listed in the UI. |
| `emi_calculator` | Exact EMI, total interest and total payable with input validation (principal > 0, rate ≥ 0, tenure 1–600 months). Avoids silent LLM arithmetic errors. |
| `escalate_to_human` | Returns a fixed hand-off message (no ticket is created). |

### 2.3 RAG pipeline

**Offline ingestion** (`rag/ingestion_pipeline/ingest.py`):

```
PDFs (data/knowledge-source) ─► PyPDFLoader ─► RecursiveCharacterTextSplitter (254 tokens, 40 overlap)
      ├─► MiniLM embeddings ─► FAISS IndexFlatIP   → data/vectorstore/index.faiss
      ├─► BM25Okapi                                → data/vectorstore/bm25.pkl
      └─► chunk text + source + page               → data/vectorstore/metadata.pkl
```

**Online retrieval** (`rag/retriever/rag_retriever.py`), run only when the agent calls `kb_lookup`:

```
query ─► BM25 top 30 ─┐
                      ├─► Reciprocal Rank Fusion (k=60) → top 15 ─► bge-reranker-base → top 6 ─► LLM
query ─► FAISS top 30 ┘
```

### 2.4 Safety layers

| Layer | Mechanism | Catches |
| --- | --- | --- |
| 1. Input masking | `mask_private_data` in `src/agent/guardrails.py` | Pasted card numbers, CVVs, phone, PAN, Aadhaar, emails |
| 2. Capability limit | Only three read-only tools | Any attempt to transfer, block, open or close |
| 3. System prompt | Scope lists, literal refusal examples, escalation rules, "say exactly: I don't have enough information in the provided documents." | Personal advice, identity lookups, off-topic, disputes |
| 4. Output guardrails | `citation_ok`, `needs_disclaimer` / `disclaimer_ok` | Uncited figures; advice without the fixed disclaimer |

### 2.5 Memory, feedback and adaptive behaviour

- **Session-only memory:** chat history lives in Streamlit `session_state` (`deque(maxlen=10)`, last 5 turns). Nothing about a customer survives a closed tab except masked logs.
- **Style control:** five response styles (Balanced, Concise, Detailed, Friendly, Formal) injected into the system prompt. Feedback maps *too short → Detailed*, *too long → Concise*, *too formal → Friendly*, *too casual → Formal*, applied to the next turn only.
- **Thumbs up/down:** logged as reward +1 / −1 in `logs/feedback/feedback_log.json`; nothing changes automatically.
- **Observability:** per-turn logs in `logs/user_interaction/`; LangSmith tracing configured via `.env`.

### 2.6 Key design decisions

| Area | Decision | Why |
| --- | --- | --- |
| Orchestration | One LangGraph agent, not sub-agents | Three read-only tools; easier to trace and cap |
| Generation | `gpt-4o-mini`, temperature 0.1 | Low cost/latency; repeatable figures and refusals |
| Embeddings | Local `all-MiniLM-L6-v2` | No per-query API cost; customer text never leaves for embedding |
| Chunking | 254 tokens (embedder's tokenizer), 40 overlap | Stays under MiniLM's 256-token limit, so nothing is truncated |
| Retrieval | BM25 + dense, RRF fusion | Banking queries hinge on exact tokens (MITC, RBI, tenors) that embeddings blur |
| Re-ranking | `BAAI/bge-reranker-base`, top 6 | Smaller, cleaner context |
| Index | FAISS `IndexFlatIP` (exact) | 13 PDFs: exact search is instant with zero recall loss |
| Loop control | `MAX_TOOL_ROUNDS = 6` then `finalize` | Stops runaway tool loops and cost spikes |

Trade-offs, deployment assumptions and limitations are in `Engineering and Product Justification.pdf`.

---

## 3. Prompt versions

All versions are kept side by side in `src/agent/prompts/`. **`version-3.2` is the default.**

| Version | Change | Problem addressed |
| --- | --- | --- |
| version-1 | Role, scope lists, literal refusal/disclaimer examples, CITING and NEVER INVENT rules | Baseline scoped, cited assistant |
| version-2 | Skip `kb_lookup` when the user gives their own rate; call `emi_calculator` directly | Wasted tool round on fully specified EMI questions |
| version-3 | Multi-part question decomposition; "background knowledge is not a source"; fraud triggers escalation | Compound questions answered only in part |
| version-3.1 | Answer the liability question from the KB first, then escalate | Policy questions escalated without an answer |
| version-3.2 | One tool call for all parts; stricter escalation; follow-ups rewritten as standalone queries | Unnecessary escalation; weak follow-up retrieval |

---

## 4. Knowledge base

13 PDFs in `data/knowledge-source/`: RBI Integrated Ombudsman Scheme, limiting liability for unauthorised transactions, TAT for failed transactions, payment-card FAQs, credit/debit card master direction FAQs, SBI personal-loan MITC, SBI Foreign Travel Card FAQs, SBI term-deposit rates (official and BankBazaar 2026), and RBI-based housing / educational / personal / comparison loan guides.

Answers are only as current as these PDFs; there is no as-of check or refresh job.

---

## 5. Repository layout

```
Capstone_Project_PrithwirajChoudhury/
    PHASE[1-4]/                              # Phases 1–4 (framing, baseline, LLM, RAG)
    AI Banking Support & Advisory Agent/     # Phases 5–9 (final agent)
    demo_scripts/                            # Demo script & logs
    Problem Framing Document.pdf
    Engineering and Product Justification.pdf
    Prompt Comparison Table.pdf
    Evaluation Report.pdf
    Screenshots Streamlit & LangSmith.pdf

AI Banking Support & Advisory Agent/
    app/app.py                       # Streamlit UI
    config.py, utils.py              # settings, logging helpers
    src/agent/                       # banking_agent.py (graph), guardrails.py, prompts/
    src/tools/                       # kb_lookup_tool.py, emi_calculator.py, escalate_to_human.py
    rag/ingestion_pipeline/ingest.py # builds the index
    rag/retriever/rag_retriever.py   # hybrid search + rerank
    data/knowledge-source/           # source PDFs
    data/vectorstore/                # FAISS, BM25, metadata
    evaluation/                      # evaluation.py + eval_results/
    logs/                            # user_interaction/, feedback/
```

### Phases 1–4

| Phase | Description | Run |
| --- | --- | --- |
| PHASE-1 | Problem Framing Document | – |
| PHASE-2 | Rule-based baseline | `python python_based_banking.py` |
| PHASE-3 | LLM, no retrieval | Set `version` in `banking_support_llm.py` (`version-1`/`2`/`3`), then `python banking_support_llm.py` |
| PHASE-4 | RAG + Streamlit | `pip install -r requirements.txt`, `python ingest.py`, `streamlit run app.py` |

RAG pipeline document: `PHASE-4.pdf`

---

## 6. Setup and run (Phases 5–9)

Run from inside `AI Banking Support & Advisory Agent/`.

1. Copy `.env.example` to `.env` and set `OPENAI_API_KEY` (optionally `CHAT_MODEL`, `EMBEDDING_MODEL`, and the LangSmith variables for tracing).
2. Install and run:

```bash
pip install -r requirements.txt
python rag/ingestion_pipeline/ingest.py   # builds data/vectorstore/
streamlit run app/app.py
```

**Configuration** (environment variables, defaults in `config.py`): `OPENAI_API_KEY`, `CHAT_MODEL` (`gpt-4o-mini`), `PROMPT_VERSION`, `TOP_K` (6), `RERANKER_MODEL`, `SOURCE_DIR`, `VECTORSTORE_DIR`.

| Item | Location |
| --- | --- |
| Interaction logs | `logs/user_interaction/user_interaction_prompt_<version>.log` |
| User feedback | `logs/feedback/feedback_log.json` |
| Prompt versions | `src/agent/prompts/` |

---

## 7. Documents

| Document | Content |
| --- | --- |
| `Problem Framing Document.pdf` | Persona, problem statement, scope, constraints and success criteria. |
| `Engineering and Product Justification.pdf` | Architecture, design decisions, trade-offs, safety approach, deployment assumptions, acknowledgements and references. |
| `Prompt Comparison Table.pdf` | Three prompt variants (version-1, version-2, version-3.2) on the same test set, with results and insights. |
| `Evaluation Report.pdf` | Evaluation method, changes, results, debugged failures, next-step improvements. |
| `Screenshots Streamlit & LangSmith.pdf` | Streamlit app and LangSmith traces, runs and monitoring. |

---

## 8. Demo script

Run from the root folder:

```bash
python demo_scripts/demo_script.py
```

Runs the 5 demo questions through `version-1`, `version-2` and `version-3.2`. Logs are written to `demo_scripts/`, one file per version (cleared at the start of each run):

- `user_interaction_prompt_version-1.log`
- `user_interaction_prompt_version-2.log`
- `user_interaction_prompt_version-3.2.log`

---

## 9. Evaluation

Run from `AI Banking Support & Advisory Agent/`:

```bash
python evaluation/evaluation.py                              # version-3.2 with LLM judge
python evaluation/evaluation.py --versions version-1 version-2 version-3
python evaluation/evaluation.py --runs 3 --no-judge --cases F01 C04 T01
```

Results are written to `evaluation/eval_results/` (per-case CSV and a JSON scorecard).

| Criterion | Target | Latest run (version-3.2, 2026-10-01) |
| --- | --- | --- |
| Correctness | ≥ 95% | 73.3% (FAIL) |
| Coherence (LLM judge) | avg ≥ 4/5 | 4.76 (PASS) |
| No hallucination on missing info *(blocking)* | 100% | 100% (PASS) |
| Scope & compliance *(blocking)* | 100% | 100% (PASS) |
| Latency p95 | ≤ 10 s | 9.53 s (PASS) |
| Tool selection | ≥ 90% | 84.0% (FAIL) |
| Tool trajectory | ≥ 90% | 100% (PASS) |
| **Overall / release gate** | | **90.6% / OK** |

---

## 10. Known limitations

- No request timeout or circuit breaker around the OpenAI call; an outage surfaces as a Streamlit error.
- Rates and policies are only as current as the PDFs.
- `escalate_to_human` returns a fixed message and creates no ticket.
- Regex guardrails can miss paraphrased advice requests.
- Logs are plain local files; English only; single-process Streamlit app, not a production service.

---

