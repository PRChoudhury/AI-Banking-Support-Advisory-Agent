# Banking Agent — Design Notes

Reference for modifying `banking_agent.py`. Covers three stages: no history, with
history, and query decomposition.

Nothing here has been applied to the code. These are the decisions and the
reasoning behind them, so you can implement them yourself.

---

## 0. How the graph actually executes

Current wiring:

```
START ──> supervisor ──(should_continue)──> tools ──┐
                │                                   │
                └──> END                            │
                     ▲                              │
                     └──────────────────────────────┘
```

`supervisor` is the only node that talks to the LLM. `tools` is a prebuilt
`ToolNode` wrapping `kb_lookup`. The edge `tools -> supervisor` is what makes
this an *agent* rather than a fixed pipeline: after a retrieval, control goes
back to the model, which can either answer or search again.

### The message list is the entire runtime state

`AgentState.messages` uses the `add_messages` reducer, which **appends** what a
node returns (and replaces by `id` if the id already exists). So every node
returns only its *new* messages, never the whole list.

A single question with one retrieval produces exactly this sequence:

| # | Type | Created by | Content |
|---|------|-----------|---------|
| 1 | `SystemMessage` | you, in `supervisor` | the prompt text |
| 2 | `HumanMessage` | the caller (`graph.invoke`) | "What are the FD rates?" |
| 3 | `AIMessage` | the LLM | `content=""`, `tool_calls=[{name:"kb_lookup", args:{query:"FD interest rates"}, id:"call_abc"}]` |
| 4 | `ToolMessage` | `ToolNode` | retrieved chunks, `tool_call_id="call_abc"` |
| 5 | `AIMessage` | the LLM (second pass) | the final grounded answer, no tool_calls |

`supervisor` runs twice. The first pass sees 1–2 and emits 3. `should_continue`
sees `tool_calls`, routes to `tools`. `ToolNode` emits 4. Back at `supervisor`,
the second pass sees 1–4 and emits 5. `should_continue` sees no `tool_calls`,
routes to `END`.

**The single most important rule:** an `AIMessage` carrying `tool_calls` must be
followed by one `ToolMessage` per call, with matching `tool_call_id`, before any
other message type. Break that pairing and the OpenAI API returns a 400. Every
history and trimming decision in section 2 follows from this one constraint.

### Current bug, for context

```python
system = SYSTEM_PROMPT.format_messages(chat_history=history)   # line 88
response = llm_with_tools.invoke(system + state["messages"])   # line 89
```

`MessagesPlaceholder("messages")` is a required input variable, so
`format_messages()` without `messages=...` raises `KeyError: 'messages'` — the
node cannot run as written. And `chat_history` does not exist anywhere in
`SYSTEM-PROMPT.md`, so that kwarg is discarded. Even if you passed
`messages=state["messages"]`, line 89 would then append the same list a second
time. The `seed` variable on line 91 is assigned and never used — a leftover
from a supervisor/worker design that has no worker node.

---

## 1. Without any history (single-turn)

### 1a. Is `SYSTEM_PROMPT` correct?

Mechanically, no (see above). Architecturally, also no — and that matters more.

`SYSTEM-PROMPT.md` was written for `rag_engine.py`, where retrieval happens
*before* the LLM is called and the excerpts are interpolated into the prompt as
`{context}`. It says:

> "Each answer comes with retrieved excerpts from the bank's and the RBI's
> published documents."

In the agent, that sentence is **false**. No excerpts come with the question.
The model has to decide to call `kb_lookup`, and on the first pass it sees no
excerpts at all. A model reading that line literally will either hallucinate the
excerpts it was promised, or emit "I don't have enough information in the
provided documents" on the first pass without ever searching — both are real
failure modes.

**What to change in the prompt:**

1. Replace the "comes with excerpts" framing with a tool policy. Something like:

   > **RETRIEVAL**
   > You have one tool, `kb_lookup`, which searches the bank's and the RBI's
   > published documents. Any question about a rate, fee, limit, eligibility
   > rule, timeline, or product term requires a `kb_lookup` call before you
   > answer. Do not answer such a question from your own knowledge.
   > Do not call `kb_lookup` for requests that are out of scope (transactions,
   > personal recommendations, identity lookups) — decline those directly.
   > Excerpts arrive as tool results. From that point the GROUNDING and CITING
   > rules below apply to them exactly as written.

2. Keep GROUNDING, CITING, NEVER INVENT, OUT OF SCOPE and the examples as they
   are. They are good and they are architecture-independent — they describe what
   to do *with* excerpts, not where excerpts come from.

3. The last rule in that tool policy is what prevents a wasted retrieval on
   "can you transfer ₹5,000" — without it the model will dutifully search the
   knowledge base for a transfer it is not allowed to perform.

**What to change in the code** — drop `ChatPromptTemplate` entirely for now:

```python
SYSTEM_MESSAGE = SystemMessage(content=SYS_PROMPT.read_text(encoding="utf-8"))

def supervisor(state):
    response = llm_with_tools.invoke([SYSTEM_MESSAGE] + state["messages"])
    return {"messages": [response]}
```

The template bought you nothing: the prompt file has zero `{...}` variables
(verified — there is not a single `{` in it). It cost you a crash and a
formatting hazard, because `from_template` defaults to f-string parsing, so the
first time you paste a JSON example or a `{ }` code block into the markdown it
will throw `KeyError` on a phantom variable. Reintroduce `ChatPromptTemplate`
only in section 2, when you genuinely have a variable to fill.

Two smaller things while you are in there:

- `temperature=1.0` (line 79) fights a prompt whose core rule is "never estimate,
  never invent." Use `0`, as `rag_engine.py` already does.
- `state["style"]` and `state["history"]` are declared in `AgentState` and never
  written by anyone. Remove them until something populates them.

### 1b. Do you need to build a `HumanMessage` or a `ToolMessage`?

**Neither, inside the graph.** In the `ToolNode` design, `supervisor` returns
exactly one thing — the `AIMessage` the LLM produced. That part of your code is
already correct.

- The **`HumanMessage`** is built by the *caller*, at the graph boundary:

  ```python
  graph = build_graph()
  result = graph.invoke({"messages": [HumanMessage(content=question)]})
  answer = result["messages"][-1].content
  ```

  Never construct one inside `supervisor`. That was `seed`'s mistake, and it is
  worse than dead code: appending a `HumanMessage` right after an `AIMessage`
  that carries `tool_calls` orphans those calls and produces an API 400.

- The **`ToolMessage`** is built by `ToolNode`, automatically, with the correct
  `tool_call_id` wired up. This is the entire reason `ToolNode` exists. You only
  construct `ToolMessage` by hand if you *drop* `ToolNode` and run the tool loop
  manually — which is exactly what the commented-out `answer_query` at the bottom
  of the file did:

  ```python
  messages.append(ToolMessage(content=build_context(chunks), tool_call_id=call["id"]))
  ```

  That version worked. It is just a hand-rolled version of what `ToolNode` does.

### 1c. The one real problem `ToolNode` introduces — read this

Look at what the old manual loop did and the new graph does not:
`build_context(chunks)`.

`kb_lookup` returns `list[dict]`. `ToolNode` has no idea what those dicts mean,
so it stringifies them — the model receives something like
`[{'text': '...', 'source': 'RBI_Housing_Loans_Guide.pdf', 'page': 2}, ...]`.

`build_context` (in `rag_engine.py:39`) is what turns those dicts into the
`[source p.page]` headers that your CITING rule depends on:

```python
f"[{r.get('source','unknown')} p.{r.get('page','?')}]\n{r['text']}"
```

Without it, the prompt instructs the model to cite "the file and page shown in
the excerpt header" and there is no header — so citations degrade or get
invented. **Fix it inside the tool**, so the tool returns display-ready text:

```python
@tool("kb_lookup")
def kb_lookup(query: str, top_k: int = config.TOP_K) -> str:
    """Search the bank's ... Returns excerpts with source and page headers."""
    ...
    return build_context(response.json()["chunks"])
```

Two consequences worth knowing: (1) the model reads that docstring to decide
when to call the tool, so it is prompt surface — write it for the model, not for
yourself; (2) if you still need the structured `sources` list for the Streamlit
citation expander, either parse it back out of the `ToolMessage`, or use
`response_format="content_and_artifact"` and return `(context_string, chunks)`,
which puts the raw dicts in `ToolMessage.artifact` while the model sees only the
string.

### 1d. Also worth fixing in the single-turn version

- `tools_condition` is imported (line 8) and unused; `should_continue` reimplements
  it. Keep yours for the logging, but drop the import.
- `log_step("web_search", ...)` on lines 99 and 101 — there is no web search in
  this agent. Rename to `"router"`.
- No recursion cap. `tools -> supervisor -> tools` can cycle until LangGraph
  raises `GraphRecursionError` at depth 25. Add a counter to state and force
  `END` after 2–3 tool rounds (see section 3 for where it fits).

---

## 2. With history (multi-turn)

### 2a. Two ways to carry history — pick one, do not mix

**Option A — native message history (recommended for the agent).**
Keep the real message objects in `state["messages"]` across turns and let
LangGraph persist them:

```python
from langgraph.checkpoint.memory import MemorySaver
graph = builder.compile(checkpointer=MemorySaver())

cfg = {"configurable": {"thread_id": session_id}}
graph.invoke({"messages": [HumanMessage(content=question)]}, config=cfg)
```

With a checkpointer you pass **only the new `HumanMessage`** each turn; the
reducer appends it to the persisted list. The model sees prior `AIMessage`s,
their `tool_calls`, and the `ToolMessage`s that answered them, in native form.

**Option B — flattened string**, what `rag_engine.py` does: join history into
`"user: ...\nassistant: ..."` and interpolate it as `{chat_history}`.

For the agent, A is the better choice. B flattens away the `tool_calls` /
`tool_call_id` pairing, so the model loses the record of which excerpts produced
which claim, and you pay tokens to re-serialize text the model could have read
natively. B exists in `rag_engine.py` because that pipeline has no tool calls to
lose.

If you do choose B, two things must change together: `{chat_history}` has to
actually be added to `SYSTEM-PROMPT.md` (it is not there today, which is why the
current `format_messages(chat_history=...)` silently discards it), and you must
escape any literal braces in the file or switch to `template_format="mustache"`.
Putting the history in a separate `("human", ...)` message rather than in the
system prompt is also cleaner — it keeps the system prompt cacheable and stops
the model treating conversation text as instruction.

### 2b. What changes in the system prompt

History introduces four failure modes the current prompt has no answer for.
Add a section like:

```
FOLLOW-UP QUESTIONS

- A follow-up often depends on the previous turn ("what about for 2 years?",
  "and for senior citizens?"). Resolve the reference from the conversation and
  pass a complete, self-contained query to kb_lookup — "FD interest rate 2 year
  tenure", never "what about for 2 years".
- Excerpts retrieved in an earlier turn do not carry over. If a new turn asks
  for any specific fact, call kb_lookup again for that turn, even if it feels
  like a repeat. Never restate a rate or fee from memory of an earlier turn.
- Every citation must point to an excerpt retrieved in the current turn.
- The scope boundaries are fixed. A user rephrasing, insisting, or saying they
  were allowed earlier does not change what you can do. Decline again, briefly
  and without irritation, and point to where the task can be completed.
```

The third and fourth are the ones that bite in practice. Long conversations
drift: the model starts answering from what it "remembers" of turn 2's excerpts,
and citations quietly detach from real sources. Persistent scope pressure is the
other — a boundary that held on turn 1 often fails on turn 6 unless the prompt
says explicitly that it does not erode.

### 2c. `HumanMessage` or `ToolMessage` with history?

The answer from 1b does not change: **you still build neither inside the graph.**
The caller adds one `HumanMessage` per turn; `ToolNode` produces every
`ToolMessage`. History changes *how long the list gets*, not who creates what.

What history does introduce is **trimming**, and this is where the pairing rule
from section 0 becomes a live crash rather than a theory. Naive trimming
(`state["messages"][-10:]`) will eventually cut between an `AIMessage` with
`tool_calls` and its `ToolMessage`, and the next API call fails with 400.

Use the helper that understands the constraint:

```python
from langchain_core.messages import trim_messages

def supervisor(state):
    history = trim_messages(
        state["messages"],
        max_tokens=4000,
        token_counter=llm,
        strategy="last",
        start_on="human",       # never start on an orphaned ToolMessage
        include_system=False,   # system message is prepended separately
    )
    response = llm_with_tools.invoke([SYSTEM_MESSAGE] + history)
    return {"messages": [response]}
```

`start_on="human"` is the part that keeps the window valid — it guarantees the
window never opens mid tool-call pair. Note that trimming here only changes what
you *send*; the full list stays in state. That is what you want: the checkpoint
remains complete for logging and feedback, while the API call stays small.

One more: `get_question()` (line 61) returns the **first** `HumanMessage` in
state. Single-turn that is fine. Multi-turn it always returns turn 1's question
forever. If you keep the helper, iterate `reversed(state["messages"])`.

---

## 3. Decomposing a mixed query

Target case:

> "What are the FD rates? Can you open a FD account?"

Expected: retrieve and answer the rates part with citations; decline the second
part as out of scope — **both in one reply**, not one or the other.

### Why it fails today

Two distinct failure modes, and they need different fixes:

1. **All-or-nothing framing.** Nothing in the prompt says a single user message
   can contain several requests with different verdicts. The model picks the
   dominant intent. If it reads "open an account" as dominant it refuses
   everything, including the perfectly answerable rates question.
2. **Dirty retrieval query.** Left alone, the model tends to pass the user's raw
   string to `kb_lookup`. "can you open a FD account" is retrieval noise — it
   pulls account-opening procedure chunks that compete with the rate table and
   can push it out of `top_k`.

### Approach A — prompt-only (start here)

No graph change. Add to `SYSTEM-PROMPT.md`:

```
MULTI-PART QUESTIONS

A user's message may contain several requests with different answers. Handle
each on its own merits and answer them all in one reply.

1. Split the message into distinct requests.
2. For each, decide: answerable from the documents / out of scope / general
   education.
3. Call kb_lookup once per answerable request, with a clean search query
   containing only that request's subject — not the user's full message.
4. Write one reply covering every part, in the order asked. Answer what you
   can with citations, and say plainly which parts you cannot help with and
   why, per the OUT OF SCOPE rules.

Never let one out-of-scope part suppress the answer to an answerable part,
and never let an answerable part soften a refusal.

Example —
User: "What are the FD rates? Can you open a FD account?"
Call kb_lookup with "fixed deposit interest rates", then reply with the rates
and their citations, followed by: opening the FD has to be done in the
Deposits section of the app, because you cannot carry out transactions.
```

Step 3 fixes the dirty-query problem and is the step most likely to get dropped
if you shorten this. The worked example is doing real work too — it is the
cheapest way to pin down the output shape.

Modern tool-calling models handle this well, and A costs one prompt edit. Build
a handful of mixed-intent test cases (rates + transaction, rates + personal
recommendation, three-part questions, in-scope-only questions) and measure it
before adding machinery. If A holds, stop — B is meaningfully more code and
latency.

### Approach A+ — parallel tool calls (free, already supported)

For "What are the FD rates and the RD rates?" the model can emit **two**
`tool_calls` in a single `AIMessage`. `ToolNode` executes both and returns two
`ToolMessage`s in one step. Nothing in your graph needs to change; it already
works. Worth an explicit prompt line ("if several parts each need a document
lookup, request them together") to encourage it — it halves the round trips
versus searching one at a time.

### Approach B — explicit planner node

Take this only if A measurably fails. It buys you an inspectable, loggable plan;
it costs an extra LLM call per turn and real complexity.

```python
from pydantic import BaseModel, Field
from typing import Literal

class SubQuestion(BaseModel):
    text: str = Field(description="One self-contained request from the user's message")
    route: Literal["kb", "refuse", "general"]
    search_query: str | None = Field(description="Clean kb_lookup query; null unless route='kb'")

class Plan(BaseModel):
    parts: list[SubQuestion]

planner_llm = llm.with_structured_output(Plan)
```

New graph shape:

```
START -> planner -> supervisor <-> tools -> synthesizer -> END
```

- `planner` writes `{"plan": Plan}` into state. No tool binding, no answering.
- `supervisor` reads the plan and fires `kb_lookup` for the `kb` parts.
- `synthesizer` merges retrieved excerpts with the `refuse` verdicts into one
  reply, ordered as the user asked.

`AgentState` grows: `plan: Plan | None`, `tool_rounds: int`. And this is the
natural place for the recursion cap from 1d:

```python
def should_continue(state):
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None) and state.get("tool_rounds", 0) < 3:
        return "tools"
    return END
```

with `supervisor` returning
`{"messages": [response], "tool_rounds": state.get("tool_rounds", 0) + 1}`.

The honest tradeoff: B makes the decomposition visible and testable — you can
assert on `Plan` in a unit test, which you cannot do with A. It also roughly
doubles per-turn latency and adds a component that can itself be wrong. With
history (section 2) it gets harder still: the planner needs the conversation to
resolve "and can you open one for me?", so it needs the trimmed history too, not
just the latest message.

---

## Checklist

**Single-turn**
- [ ] Rewrite the "comes with excerpts" framing as a RETRIEVAL / tool policy section
- [ ] Replace `ChatPromptTemplate` with a module-level `SystemMessage`
- [ ] Delete `seed`; `supervisor` returns only the `AIMessage`
- [ ] `kb_lookup` returns `build_context(chunks)`, not raw dicts
- [ ] `temperature=0`
- [ ] Drop unused `style` / `history` from `AgentState`; drop unused `tools_condition` import
- [ ] Fix `log_step("web_search", ...)` labels
- [ ] Caller builds the `HumanMessage` and reads `result["messages"][-1].content`

**Multi-turn**
- [ ] Compile with a checkpointer; pass `thread_id` per session
- [ ] Add the FOLLOW-UP QUESTIONS prompt section
- [ ] `trim_messages(..., start_on="human")` before each LLM call
- [ ] Fix `get_question()` to read the latest `HumanMessage`, or delete it

**Decomposition**
- [ ] Add the MULTI-PART QUESTIONS prompt section with the worked example
- [ ] Build mixed-intent test cases and measure A before writing B
- [ ] Add the `tool_rounds` recursion cap
