# Standard Libraries
from collections import deque
from datetime import datetime, timezone
import json
import sys
import uuid
from pathlib import Path
# Import Third-party libraries
import streamlit as st
ROOT_DIR = Path(__file__).resolve().parent.parent   # PHASE-5
sys.path.insert(0, str(ROOT_DIR.parent))            # project root (utils.py)
sys.path.insert(0, str(ROOT_DIR))                   # PHASE-5 wins for `config`

# Local modules
import config
# from src.agent.banking_agent import stream_research_agent, FEEDBACK_STYLE, DEFAULT_STYLE
from src.agent.banking_agent import run_research_agent, FEEDBACK_STYLE, DEFAULT_STYLE
from src.agent import guardrails
import threading
from rag.retriever.rag_retriever import get_retriever


@st.cache_resource(show_spinner="Starting agent...")
def _warm_retriever():
    threading.Thread(target=get_retriever, daemon=True).start()

_warm_retriever()

st.set_page_config(
    page_title="AI Banking Support & Advisory Agent (Non-Transactional)",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {
          max-width: 100% !important;
          padding: 1.5rem 2rem 6rem 2rem;
      }
      /* Pull the thumbs-up/down pair tight together in the feedback row. */
      [class*="st-key-thumbs_"] [data-testid="stHorizontalBlock"] {
          gap: 0.15rem !important;
      }
      [class*="st-key-thumbs_"] button {
          padding: 0.25rem 0.5rem !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Starting agent...")
def warm_agent():
    from src.agent.banking_agent import get_graph
    return get_graph(config.CHAT_MODEL)


warm_agent()


# --------------------------------------------------------------------------- #
# Conversation store (kept in session state)
# --------------------------------------------------------------------------- #
def new_conversation() -> str:
    conv_id = f"conv_{st.session_state.conversation_counter}"
    st.session_state.conversation_counter += 1
    st.session_state.conversations[conv_id] = {
        "title": "New chat",
        "messages": [],
        "chat_history": deque(maxlen=10),
        "next_style": DEFAULT_STYLE,
    }
    st.session_state.active_conversation = conv_id
    return conv_id


if "conversations" not in st.session_state:
    st.session_state.conversations = {}
    st.session_state.conversation_counter = 1
    st.session_state.pending_question = None
    new_conversation()


def delete_conversation(conv_id: str) -> None:
    st.session_state.conversations.pop(conv_id, None)
    if not st.session_state.conversations:
        new_conversation()
    elif st.session_state.active_conversation == conv_id:
        st.session_state.active_conversation = next(iter(st.session_state.conversations))
    st.session_state.pending_question = None


def select_conversation(conv_id: str) -> None:
    st.session_state.active_conversation = conv_id
    st.session_state.pending_question = None


def conversation_title(question: str) -> str:
    title = " ".join(question.split())
    return title if len(title) <= 40 else title[:37] + "..."


# --------------------------------------------------------------------------- #
# Feedback helpers
# --------------------------------------------------------------------------- #
def write_json(filepath: str, data) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _append_feedback_record(record: dict) -> None:
    record["timestamp"] = datetime.now(timezone.utc).isoformat()
    try:
        with open(config.FEEDBACK_LOG, "r", encoding="utf-8") as f:
            records = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        records = []

    records.append(record)
    write_json(config.FEEDBACK_LOG, records)


def capture_feedback(query: str, turn_id: str, rating: str, default_style: str) -> None:
    reward = 1 if rating == "up" else -1
    query = guardrails.sanitize_and_respond(query)
    _append_feedback_record({
        "query": query,
        "turn_id": turn_id,
        "reward": reward,
        "default_style": default_style,
    })


def capture_style_feedback(query: str, turn_id: str, style: str, default_style: str) -> None:
    query = guardrails.sanitize_and_respond(query)
    _append_feedback_record({
        "query": query,
        "turn_id": turn_id,
        "style_feedback": style,
        "default_style": default_style,
    })


RESPONSE_STYLES = ["Looks good", "Too short", "Too long", "Too formal", "Too casual"]


def render_feedback(message: dict) -> None:
    # setdefault, not [] — covers assistant turns already in session_state
    # from before turn_id existed (e.g. a hot-reload mid-session).
    turn_id = message.setdefault("turn_id", str(uuid.uuid4()))
    message.setdefault("style_feedback", "Looks good")

    def _on_style_change():
        choice = st.session_state[f"style_{turn_id}"]
        previous_style = message["style_feedback"]
        message["style_feedback"] = choice
        capture_style_feedback(message.get("query"), turn_id, choice, previous_style)
        corrected_style = FEEDBACK_STYLE.get(choice.lower())
        if corrected_style:
            active_conv = st.session_state.conversations[st.session_state.active_conversation]
            active_conv["next_style"] = corrected_style

    col_thumbs, _, col_style = st.columns(
        [1.2, 5.8, 2.2], gap="small", vertical_alignment="bottom"
    )
    with col_thumbs:
        st.caption("Rate response")
        with st.container(key=f"thumbs_{turn_id}"):
            col_up, col_down = st.columns(2, gap="small")
            with col_up:
                if st.button(
                    "", icon=":material/thumb_up:",
                    key=f"feedback_up_{turn_id}", type="tertiary",
                ):
                    capture_feedback(
                        message.get("query"), turn_id, "up", message["style_feedback"],
                    )
            with col_down:
                if st.button(
                    "", icon=":material/thumb_down:",
                    key=f"feedback_down_{turn_id}", type="tertiary",
                ):
                    capture_feedback(
                        message.get("query"), turn_id, "down", message["style_feedback"],
                    )
    with col_style:
        st.selectbox(
            "Style feedback",
            RESPONSE_STYLES,
            index=RESPONSE_STYLES.index(message["style_feedback"]),
            key=f"style_{turn_id}",
            help="How would you rate this answer's style?",
            on_change=_on_style_change,
        )


def render_sources(sources) -> None:
    with st.expander("Source document citation"):
        for src in sources:
            st.write(f"- {src['source']} (page {src['page']})")


# --------------------------------------------------------------------------- #
# Layout: 20% conversation rail | 80% chat window
# --------------------------------------------------------------------------- #
rail, chat_pane = st.columns([1, 4], gap="medium")

with rail.container(border=True):
    st.subheader("Conversations")

    if st.button("➕ New chat", width="stretch"):
        new_conversation()
        st.rerun()

    st.divider()

    for conv_id, conversation in reversed(list(st.session_state.conversations.items())):
        is_active = conv_id == st.session_state.active_conversation
        col_open, col_delete = st.columns([5, 1], gap="small")
        with col_open:
            st.button(
                conversation["title"],
                key=f"open_{conv_id}",
                width="stretch",
                type="primary" if is_active else "tertiary",
                on_click=select_conversation,
                args=(conv_id,),
            )
        with col_delete:
            st.button(
                "🗑️",
                key=f"delete_{conv_id}",
                help="Delete this conversation",
                on_click=delete_conversation,
                args=(conv_id,),
            )

with chat_pane:
    st.title("AI Banking Support & Advisory Agent")
    st.caption("Non-Transactional")

    active_id = st.session_state.active_conversation
    active = st.session_state.conversations[active_id]

    transcript = st.container(height=560, border=False)
    with transcript:
        for message in active["messages"]:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    st.caption(f"Style: {message.get('style_used', DEFAULT_STYLE)}")
                st.write(message["content"])
                if message["role"] == "assistant":
                    if message.get("sources"):
                        render_sources(message["sources"])
                    render_feedback(message)

        # A question submitted on the previous run is answered here, so the
        # conversation rail already shows the updated title.
        if st.session_state.pending_question:
            question = guardrails.sanitize_and_respond(st.session_state.pending_question)
            history_snapshot = list(active["chat_history"])
            style_used = active.get("next_style", DEFAULT_STYLE)
            print(history_snapshot)
            with st.chat_message("assistant"):
                st.caption(f"Style: {style_used}")
                # sink = {}
                # answer = st.write_stream(
                #     stream_research_agent(question, history_snapshot, style=style_used, sink=sink)
                # )
                # sources = sink.get("sources", [])
                with st.spinner("Thinking..."):
                    answer, sources = run_research_agent(
                        question, history_snapshot, style=style_used
                    )
                st.write(answer)
            active["messages"].append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "query": question,
                    "turn_id": str(uuid.uuid4()),
                    "style_feedback": "Looks good",
                    "style_used": style_used,
                }
            )
            active["chat_history"].append({"role": "user", "content": question})
            active["chat_history"].append({"role": "assistant", "content": answer})
            st.session_state.pending_question = None
            st.rerun()

    question = st.chat_input("Ask a question...")

if question:
    question = guardrails.sanitize_and_respond(question)
    active = st.session_state.conversations[st.session_state.active_conversation]
    if not active["messages"]:
        active["title"] = conversation_title(question)
    active["messages"].append({"role": "user", "content": question})
    st.session_state.pending_question = question
    st.rerun()
