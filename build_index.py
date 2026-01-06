"""
build_index.py
----------------
Step 1 of the pipeline: turn a folder of .txt documents into a searchable
FAISS vector index.

What this script does, in order:
1. Reads every .txt file in the documents/ folder.
2. Splits ("chunks") each document into smaller overlapping pieces, because
   LLMs and embedding models work better on focused paragraphs than on
   whole documents at once.
3. Calls the Gemini API to turn each chunk into an "embedding" -- a list
   of numbers that represents the meaning of that chunk.
4. Stores all those embeddings in a FAISS index (a fast similarity-search
   structure) and saves it to disk, along with the original chunk text,
   so app.py can load it later without re-embedding everything.

Run this once whenever you add/change documents:
    python build_index.py
"""

import os
import glob
import pickle

import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()  # reads GEMINI_API_KEY from a local .env file

DOCUMENTS_DIR = "documents"
INDEX_FILE = "vector_store.faiss"
CHUNKS_FILE = "chunks.pkl"

EMBEDDING_MODEL = "gemini-embedding-001"

# How big each chunk is (in characters) and how much consecutive chunks
# overlap. Overlap helps avoid cutting a relevant sentence in half between
# two chunks.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def load_documents(folder: str) -> list[dict]:
    """Read every .txt file in `folder` and return a list of
    {"source": filename, "text": file contents} dicts."""
    docs = []
    for path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        docs.append({"source": os.path.basename(path), "text": text})
    return docs


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks of roughly `chunk_size` characters."""
    chunks = []
    start = 0
    text = text.strip()
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def embed_texts(client: genai.Client, texts: list[str]) -> np.ndarray:
    """Call the Gemini embedding model on a list of strings and return a
    numpy array of shape (len(texts), embedding_dim)."""
    embeddings = []
    for text in texts:
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        embeddings.append(result.embeddings[0].values)
    return np.array(embeddings, dtype="float32")


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    client = genai.Client(api_key=api_key)

    print("Loading documents...")
    docs = load_documents(DOCUMENTS_DIR)
    if not docs:
        raise SystemExit(f"No .txt files found in {DOCUMENTS_DIR}/")
    print(f"  Found {len(docs)} document(s): {[d['source'] for d in docs]}")

    print("Chunking documents...")
    all_chunks = []  # list of {"source": ..., "text": ...}
    for doc in docs:
        pieces = chunk_text(doc["text"])
        for piece in pieces:
            all_chunks.append({"source": doc["source"], "text": piece})
    print(f"  Produced {len(all_chunks)} chunks")

    print("Generating embeddings via Gemini API (this calls the network once per chunk)...")
    texts = [c["text"] for c in all_chunks]
    embeddings = embed_texts(client, texts)
    print(f"  Got embeddings of shape {embeddings.shape}")

    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # simple exact nearest-neighbor search
    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"Done. Saved index to '{INDEX_FILE}' and chunk metadata to '{CHUNKS_FILE}'.")


if __name__ == "__main__":
    main()
