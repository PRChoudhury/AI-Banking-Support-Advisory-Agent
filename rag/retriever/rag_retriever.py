# Standard library imports
import os
import pickle
import re
from functools import lru_cache

# Third-party imports
import faiss
import numpy as np
from langchain_core.runnables import RunnableLambda, RunnableParallel
from sentence_transformers import CrossEncoder, SentenceTransformer

# Local application imports
import config

def check_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your .env file before starting the app."
        )

def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())

def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[int, float]]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """
    ranked_lists: e.g. [bm25_results, ann_results], each a list of
                  (chunk_id, score) already sorted best-first.
    Returns [(chunk_id, rrf_score), ...] sorted best-first.
    """
    fused_scores: dict[int, float] = {}
    for results in ranked_lists:
        for rank, (chunk_id, _score) in enumerate(results):  # rank is 0-indexed
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)

    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

def load_indices():
    with open(config.VECTORSTORE_DIR / "bm25.pkl", "rb") as f:
        bm25_store = pickle.load(f)
    faiss_index = faiss.read_index(str(config.VECTORSTORE_DIR / "index.faiss"))
    with open(config.VECTORSTORE_DIR / "metadata.pkl", "rb") as f:
        chunk_records = pickle.load(f)
    return bm25_store, faiss_index, chunk_records


class HybridRetriever:
    def __init__(self, embed_model: SentenceTransformer | None = None,
                 reranker: CrossEncoder | None = None,
                 k_per_retriever: int = 30,
                 fused_top_n: int = 15):
        if embed_model is None:
            embed_model = SentenceTransformer(config.EMBEDDING_MODEL_ST)
        if reranker is None:
            reranker = CrossEncoder(config.RERANKER_MODEL, max_length=config.RERANKER_MAX_LENGTH)
        self.bm25_store, self.faiss_index, self.chunk_records = load_indices()
        self.embed_model = embed_model
        self.reranker = reranker
        self.k_per_retriever = k_per_retriever
        self.fused_top_n = fused_top_n
        self.retrieve_parallel = RunnableParallel(
        bm25_results = RunnableLambda(lambda q: self.bm25_search(q["query"], k = q["k"])),
        ann_results =  RunnableLambda(lambda q: self.ann_search(q["query"], k = q["k"])))

    def bm25_search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Returns [(chunk_id, bm25_score), ...] sorted best-first."""
        scores = self.bm25_store["bm25"].get_scores(tokenize(query))
        k = min(k, len(scores))
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [(int(i), float(scores[i])) for i in top_idx]

    def ann_search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Returns [(chunk_id, ip_score), ...] sorted best-first."""
        q_emb = self.embed_query(query)
        scores, idx = self.faiss_index.search(q_emb, k) 
        return [
            (int(i), float(s))
            for i, s in zip(idx[0], scores[0])
            if i != -1
        ]
    def embed_query(self, query: str) -> np.ndarray:
        embeddings = self.embed_model.encode([query], normalize_embeddings=True)
        return embeddings.astype("float32")
    
    def hybrid_retrieve(self, query: str, k_per_retriever: int, fused_top_n: int):
        retrieve =  self.retrieve_parallel.invoke({"query": query, "k": k_per_retriever})
        fused = reciprocal_rank_fusion([retrieve["bm25_results"], retrieve["ann_results"]])[:fused_top_n]
        return [
            {**self.chunk_records[chunk_id], "rrf_score": score}
            for chunk_id, score in fused
        ]
    def hybrid_search(self, query: str, final_k: int):
        candidates = self.hybrid_retrieve(query, k_per_retriever=self.k_per_retriever, fused_top_n=self.fused_top_n)
        return self.rerank(query, candidates, top_k=final_k)
    
    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.reranker.predict(pairs)  # higher = more relevant
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_k]

@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    """
    One retriever per process. Building it loads the embedding model, the
    cross-encoder and the three vectorstore files, so this must not run
    per call.
    """
    return HybridRetriever()
