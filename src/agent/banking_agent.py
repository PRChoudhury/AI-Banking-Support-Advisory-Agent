from functools import lru_cache
from typing import Annotated, TypedDict
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

import os
from pathlib import Path
from dotenv import load_dotenv
import config
load_dotenv()

from src.tools.kb_lookup_tool import kb_lookup
from src.tools.escalate_to_human import escalate_to_human
from src.tools.emi_calculator import emi_calculator
from src.agent.prompts.prompt_styles import prompt_style
from utils import log_interaction, reset_log
from src.agent import guardrails 

version = "version-3"
LOG_FILE = config.USER_INTERACTION_LOG_DIR / f"user_interaction_prompt_{version}.log"
SYS_PROMPT = Path(__file__).parent /f"prompts/{version}/SYSTEM-PROMPT.md"

_api_key = os.getenv("OPENAI_API_KEY")
_model= os.getenv("CHAT_MODEL")

llm = init_chat_model(
    api_key=_api_key,
    model=_model,
    temperature=0.2,
    max_tokens=1000
)
tools = [kb_lookup, escalate_to_human, emi_calculator]
SYSTEM_MESSAGE = ChatPromptTemplate.from_messages([
    ("system", SYS_PROMPT.read_text(encoding="utf-8")),
    MessagesPlaceholder("chat_history"),
])
DEFAULT_STYLE = "Balanced"
FEEDBACK_STYLE = {
    "too short": "Detailed",
    "too long": "Concise",
    "too formal": "Friendly",
    "too casual": "Formal", 
}
STYLE_INSTRUCTIONS = prompt_style.STYLE_PROMPTS

# agent = create_agent(llm, tools= tools)
# ---------------------------------------------------------------------------
# Callback to print tool calls as they happen
# ---------------------------------------------------------------------------

class PrintToolCallsCallback(BaseCallbackHandler):
    def on_tool_start(self, serialized, input_str, **kwargs):
        name = serialized.get("name", "tool")
        print("  [" + name + "] " + str(input_str)[:120])

    def on_tool_end(self, output, **kwargs):
        preview = str(output)[:80].replace("\n", " ")
        print("  -> " + preview + "...")
        
# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    style: str
    history: list[dict]


def get_question(state) -> str:
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""

# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

def build_graph(model=_model, verbose=True):

    def log_step(name, detail=""):
        if verbose:
            print("  [" + name + "] " + str(detail)[:120])

    llm = init_chat_model(
        model=model,
        temperature=0.1,
        api_key=_api_key)
    
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools, messages_key="messages")
    
    def tool_rounds(messages) -> int:
        """Assistant turns that requested tools. Parallel calls in one AIMessage
        count as a single round, which is the granularity we want to cap."""
        return sum(1 for m in messages
                if isinstance(m, AIMessage) and getattr(m, "tool_calls", None))
    def _run_guardrails(response, history, question, system, final_msg=None):
        sources = collect_sources(history)
        problems = []
        if not guardrails.citation_ok(response.text, sources):
            problems.append("add an inline (file.pdf, p.N) citation for every specific fact from the excerpts")
        needs_disclaimer = guardrails.needs_disclaimer(question)
        if needs_disclaimer and not guardrails.disclaimer_ok(question, response.text):
            problems.append(
                "add this exact line verbatim, on its own line: \"" + guardrails.DISCLAIMER_TEXT + "\"")

        if problems:
            log_step("guardrail", "retrying once: " + "; ".join(problems))
            correction = HumanMessage(content=(
                "Rewrite your last answer to fix this: " + "; and ".join(problems) +
                ". Use only the excerpts already retrieved — do not add new facts."))
            extra = [final_msg] if final_msg is not None else []
            response = llm.invoke(system + history + extra + [response, correction])

        if needs_disclaimer and not guardrails.disclaimer_ok(question, response.text):
            response.content = str(response.content).rstrip() + "\n\n" + guardrails.DISCLAIMER_TEXT
        return response

    def finalize(state):
        """
        Answer from what has already been retrieved.
        """
        msgs = state["messages"]
        last = msgs[-1]
        pending = bool(getattr(last, "tool_calls", None))
        history = msgs[:-1] if pending else msgs

        final_msg = HumanMessage(content=(
            "You have used all available document searches for this question. "
            "Answer now from the excerpts already retrieved, with citations. "
            "For anything they do not cover, say exactly: I don't have enough "
            "information in the provided documents."))

        style_text = STYLE_INSTRUCTIONS.get(state.get("style"), STYLE_INSTRUCTIONS[DEFAULT_STYLE])
        # print(style_text)
        system = SYSTEM_MESSAGE.format_messages(
            chat_history=state.get("history") or [], response_style=style_text,
        )
        # USED FOR STREAMING - COMMENTED TEMPORARILY
        # ========================================================================
        # response = None
        # for c in llm.stream(system + history + [final_msg]):
        #     response = c if response is None else response + c
        # response = llm.invoke(system + history + [final_msg])
        # log_step("finalize", ":: " + str(response.content))
        # out = [RemoveMessage(id=last.id), response] if pending else [response]
        # return {"messages": out}
        # ========================================================================
        response = llm.invoke(system + history + [final_msg])
        question = get_question(state)
        response = _run_guardrails(response, history, question, system, final_msg)
        log_step("finalize", ":: " + str(response.content))
        out = [RemoveMessage(id=last.id), response] if pending else [response]
        return {"messages": out}
    
    def guardrail_check(state):
        history = state["messages"]
        last = history[-1]
        question = get_question(state)
        style_text = STYLE_INSTRUCTIONS.get(state.get("style"), STYLE_INSTRUCTIONS[DEFAULT_STYLE])
        system = SYSTEM_MESSAGE.format_messages(chat_history=state.get("history") or [], response_style=style_text)
        response = _run_guardrails(last, history[:-1], question, system)
        if response is last:
            return {}
        return {"messages": [RemoveMessage(id=last.id), response]}

    def supervisor(state):
        log_step("supervisor", "analyzing query")
        history = state.get("history") or []
        style_text = STYLE_INSTRUCTIONS.get(state.get("style"), STYLE_INSTRUCTIONS[DEFAULT_STYLE])
        # print(style_text)
        system = SYSTEM_MESSAGE.format_messages(chat_history=history, response_style=style_text)
        response = llm_with_tools.invoke(system + state["messages"])
        log_step("supervisor", "-> " + str(response.content))
        return {"messages": [response]}
  
    def should_continue(state):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            rounds = tool_rounds(state["messages"])
            if rounds > config.MAX_TOOL_ROUNDS:
                log_step("router", f"-> cap of {config.MAX_TOOL_ROUNDS} rounds hit, forcing an answer")
                return "finalize"
            names = ", ".join(tc["name"] for tc in last.tool_calls)
            log_step("router", f"-> tools ({rounds}/{config.MAX_TOOL_ROUNDS}): " + names)
            return "tools"
        log_step("router", "-> routing to guardrail_check")
        return "guardrail_check"

    builder = StateGraph(AgentState)
    builder.add_node("supervisor", supervisor)
    builder.add_node("tools", tool_node)
    builder.add_node("finalize", finalize)
    builder.add_node("guardrail_check", guardrail_check)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
            "supervisor",
            should_continue,
            {"tools": "tools",
             "finalize": "finalize",
             "guardrail_check": "guardrail_check"},
        )
    builder.add_edge("tools", "supervisor")
    builder.add_edge("guardrail_check", END)

    return builder.compile()

def collect_sources(messages) -> list[dict]:
    """Citations from every kb_lookup call in the run, de-duplicated by (source, page)."""
    seen, sources = set(), []
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        for chunk in msg.artifact or []:
            key = (chunk.get("source"), chunk.get("page"))
            if key in seen:
                continue
            seen.add(key)
            sources.append({"source": chunk.get("source", "unknown"),
                            "page": chunk.get("page", "?")})
    return sources

@lru_cache(maxsize=4)
def _build_cached(model: str, verbose: bool):
    """Keyed on canonical args only — never call this directly."""
    return build_graph(model=model, verbose=verbose)

def get_graph(model: str , verbose: bool = True):
    # reset_log(LOG_FILE)
    return _build_cached(model or config.CHAT_MODEL, verbose)


def stream_research_agent(question, history=None, style=None, model=config.CHAT_MODEL,
                          verbose=True, sink=None):

    graph = get_graph(model=model, verbose=verbose)
    final_messages = []
    callbacks = [PrintToolCallsCallback()] if verbose else []
    question = guardrails.sanitize_and_respond(question)
    for mode, chunk in graph.stream(
        {
            "messages": [HumanMessage(content=question)],
            "history": history,
            "style": style or DEFAULT_STYLE,
        },
        stream_mode=["messages", "values"],
        config={"callbacks": callbacks},
    ):
        if mode == "messages":
            msg, meta = chunk
            if meta.get("langgraph_node") in ("supervisor","finalize") and msg.text:
                yield msg.text
        elif mode == "values":
            final_messages = chunk["messages"]

    raw = ""
    for msg in reversed(final_messages):
        if isinstance(msg, AIMessage) and msg.content and msg.text.strip():
            raw = msg.text.strip()
            break

    if verbose:
        print("\n" + "-" * 60)
        print(raw)
    log_interaction(LOG_FILE, version, question, raw, style)

    if sink is not None:
        sink["answer"] = raw
        sink["sources"] = collect_sources(final_messages)


def run_research_agent(question, history="", style=None, model=config.CHAT_MODEL, verbose=True):
    """Blocking wrapper — unchanged signature and return contract."""
    sink = {}
    for _ in stream_research_agent(
        question, history=history, style=style, model=model, verbose=verbose, sink=sink,
    ):
        pass
    return sink.get("answer", ""), sink.get("sources", [])


