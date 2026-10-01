import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_DIR = Path(__file__).resolve().parent

# Python package root: src/ holds api/, tools/, agent/ and the pipeline modules.
SRC_DIR = PROJECT_DIR / "src"

# Everything the RAG pipeline reads or writes on disk lives under data/.
DATA_DIR = PROJECT_DIR / "data"

# Folder containing the source PDFs ingested into the knowledge base.
SOURCE_DIR = Path(os.getenv("SOURCE_DIR", DATA_DIR / "knowledge-source"))

# Where the FAISS index, BM25 store and chunk metadata are written by the
# ingestion pipeline and read back by the retriever.
VECTORSTORE_DIR = Path(os.getenv("VECTORSTORE_DIR", DATA_DIR / "vectorstore"))

# Prompt folder - versioned per phase, owned by the agent layer.
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "version-3")
ROLES_DIR = SRC_DIR / "agent" / "prompts" / PROMPT_VERSION

# Logs Folder - feedback and per-turn user-interaction logs.
LOGS_DIR = PROJECT_DIR / "logs"
FEEDBACK_LOG = LOGS_DIR / "feedback" / "feedback_log.json"
USER_INTERACTION_LOG_DIR = LOGS_DIR / "user_interaction"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_MODEL_ST = "sentence-transformers/all-MiniLM-L6-v2"
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# Cross-encoder that re-scores hybrid-retrieval candidates before they reach the LLM.
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
RERANKER_MAX_LENGTH = int(os.getenv("RERANKER_MAX_LENGTH", "512"))

# Number of chunks retrieved per query.
TOP_K = int(os.getenv("TOP_K", "6"))
MAX_TOOL_ROUNDS = 6

# Default message for NO ANSWER
NO_ANSWER_MESSAGE = "I don't have enough information in the provided documents."
