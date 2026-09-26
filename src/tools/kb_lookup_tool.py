# Import Third-party libraries
from langchain_core.tools import tool
# Local modules
import config
from rag.retriever.rag_retriever import get_retriever

def build_context(results: list[dict]) -> str:
    return "\n\n".join(
        f"[{r.get('source', 'unknown')} p.{r.get('page', '?')}]\n{r['text']}"
        for r in results
    )

def trim(chunk: dict) -> dict:
    return {
        "text": chunk["text"],
        "source": chunk.get("source", "unknown"),
        "page": chunk.get("page", "?"),
        "rrf_score": chunk.get("rrf_score"),
        "rerank_score": chunk.get("rerank_score"),
    }

@tool("kb_lookup", response_format="content_and_artifact")
def kb_lookup(query: str, top_k: int = config.TOP_K) -> tuple[str, list[dict]]:
    """
    Search the bank's knowledge base documents for relevant content to a customer question.

    Use this for questions about loans, interest rates, deposits, cards,
    payment systems or complaints. Returns the matching chunks.
    """
    results = get_retriever().hybrid_search(query, final_k=top_k)
    if not results:
        return config.NO_ANSWER_MESSAGE, []
    chunks = [trim(c) for c in results]
    return build_context(chunks), chunks
