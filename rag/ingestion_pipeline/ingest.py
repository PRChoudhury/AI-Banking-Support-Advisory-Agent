# Standard library imports
import pickle
import re
import sys
from pathlib import Path
from typing import Optional

# Third-party imports
import faiss
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# Local application imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

# UTIL METHODS
def capture_metadata(chunks):
    """Capture and Store metadata"""
    print(f"Writing metadata to: {config.VECTORSTORE_DIR}/metadata.pkl")
    with open(config.VECTORSTORE_DIR/"metadata.pkl", "wb") as f:
        pickle.dump([{**c.metadata, "text": c.page_content} for c in chunks], f)

def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())

def find_pdfs(*roots):
    """Recursively find PDF files under the given root directories."""
    pdfs = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.pdf"):
            pdfs.append(path)
    return pdfs

def extract_metadata(pdf_path: str):
    # Relative path from the root folder
    relative = pdf_path.relative_to(config.SOURCE_DIR)
    filename = relative.name
    metadata = {
        "filename": filename,
        "source": str(pdf_path)
    }
    return metadata
    

class RagIngestionPipeline:
    """
    Reads PDFs under config.SOURCE_DIR, splits each page into chunks, and
    writes three files to config.VECTORSTORE_DIR:

        index.faiss   - IndexFlatIP for semantic search
        bm25.pkl      - BM25Okapi for lexical search
        metadata.pkl  - per-chunk source, page and raw text
    """

    def __init__(self, chunk_size: int = 254, chunk_overlap:int = 40):
        self.embed_model_name= config.EMBEDDING_MODEL_ST
        self._tokenizer = None
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embed_model: Optional[SentenceTransformer] = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.embed_model_name)
        return self._tokenizer
    
    def load_embed_model(self):
        if self.embed_model is None:
            self.embed_model = SentenceTransformer(self.embed_model_name)
            
    def embed_corpus(self, texts: list[str]) -> np.ndarray:
        embeddings = self.embed_model.encode(texts, normalize_embeddings=True)
        return embeddings.astype("float32")
  
    def build_bm25_store(self, chunks):
        corpus_tokens = [tokenize(c.page_content) for c in chunks]
        bm25 = BM25Okapi(corpus_tokens)
        bm25_store = {
            "bm25" : bm25
        }
        print(f"Writing bm25 data: {config.VECTORSTORE_DIR}/bm25.pkl")
        with open(config.VECTORSTORE_DIR/"bm25.pkl", "wb") as f:
            pickle.dump(bm25_store, f)
    
    def token_len_st(self, text) ->int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def load_pages(self, pdf_paths):
        """Load every page of every PDF as a LangChain Document, tagging metadata with the source file."""
        docs = []
        for pdf_path in pdf_paths:
            try:
                loader = PyPDFLoader(str(pdf_path))
                pages = loader.load()
            except Exception as exc:
                print(f"  Skipping {pdf_path.name}: {exc}")
                continue
            for page in pages:
                page.metadata["page"] = page.metadata.get("page", 0) + 1
                page.metadata["source"] = pdf_path.name
            docs.extend(pages)
            print(f"  Loaded {pdf_path.name} ({len(pages)} pages)")
        return docs

    def build_index(self):
        self.load_embed_model()
        print(f"Source folder: {config.SOURCE_DIR}")
        if not config.SOURCE_DIR.exists():
            print(f"ERROR: source folder does not exist: {config.SOURCE_DIR}")

        pdf_paths = find_pdfs(config.SOURCE_DIR)
        print(f"\nFound {len(pdf_paths)} PDF(s). Loading...")

        pages = self.load_pages(pdf_paths)
        print(f"\nLoaded {len(pages)} total pages. Splitting into semantic chunks...")

        splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    length_function=self.token_len_st,
                    separators=["\n\n", "\n", ". ", " ", ""],
                )
        chunks = splitter.split_documents(pages)
        capture_metadata(chunks)
        texts = [chunk.page_content for chunk in chunks]
        vectors = self.embed_corpus(texts)
        self.build_bm25_store(chunks)
        dim = vectors.shape[1]
        print(f"Produced {len(chunks)} chunks. Embedding and building FAISS index...")
        
        vectorstore = faiss.IndexFlatIP(dim)
        vectorstore.add(vectors)
        config.VECTORSTORE_DIR.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(vectorstore, str(config.VECTORSTORE_DIR/"index.faiss"))
        print(f"\nDone. Index saved to {config.VECTORSTORE_DIR}")

if __name__ == "__main__":
    RagIngestionPipeline().build_index()