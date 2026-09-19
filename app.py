"""
app.py
------
Step 2 of the pipeline: a small Flask REST API that answers questions
using the documents indexed by build_index.py.

How a request flows through this file:
1. A client POSTs {"question": "..."} to /ask.
2. We embed the question using the same Gemini embedding model used to
   build the index (embeddings only make sense when compared to other
   embeddings from the same model).
3. We search the FAISS index for the chunks whose embeddings are closest
   to the question's embedding -- these are the most relevant snippets
   from our documents.
4. We build a prompt that includes those snippets plus the original
   question, and send it to a Gemini chat model.
5. We return the model's answer, along with which document(s) it was
   grounded in, so the response is traceable back to a source.

Run with:
    python app.py
Then, in another terminal:
    curl -X POST http://localhost:5000/ask \
         -H "Content-Type: application/json" \
         -d '{"question": "How many days of annual leave do I get?"}'
"""

import os
import pickle

import faiss
import numpy as np
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

INDEX_FILE = "vector_store.faiss"
CHUNKS_FILE = "chunks.pkl"
EMBEDDING_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.6-flash"
TOP_K = 3  # how many chunks to retrieve per question

app = Flask(__name__)

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise SystemExit("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")
client = genai.Client(api_key=api_key)

# Load the index and chunk metadata built by build_index.py.
if not os.path.exists(INDEX_FILE) or not os.path.exists(CHUNKS_FILE):
    raise SystemExit(
        f"Could not find '{INDEX_FILE}' / '{CHUNKS_FILE}'. Run `python build_index.py` first."
    )

index = faiss.read_index(INDEX_FILE)
with open(CHUNKS_FILE, "rb") as f:
    chunks = pickle.load(f)


def embed_query(text: str) -> np.ndarray:
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return np.array([result.embeddings[0].values], dtype="float32")


def retrieve_relevant_chunks(question: str, top_k: int = TOP_K) -> list[dict]:
    query_vector = embed_query(question)
    distances, indices = index.search(query_vector, top_k)
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx == -1:
            continue
        chunk = chunks[idx]
        results.append({**chunk, "distance": float(dist)})
    return results


def build_prompt(question: str, retrieved: list[dict]) -> str:
    context_text = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in retrieved
    )
    return f"""You are a helpful HR assistant. Answer the question using ONLY the
context provided below. If the answer is not contained in the context,
say you don't have that information rather than guessing.

Context:
{context_text}

Question: {question}

Answer:"""


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Missing 'question' in request body"}), 400

    retrieved = retrieve_relevant_chunks(question)
    if not retrieved:
        return jsonify({"error": "No indexed documents to search"}), 500

    prompt = build_prompt(question, retrieved)
    response = client.models.generate_content(model=CHAT_MODEL, contents=prompt)

    return jsonify({
        "question": question,
        "answer": response.text.strip(),
        "sources": sorted({c["source"] for c in retrieved}),
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "chunks_indexed": len(chunks)})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
