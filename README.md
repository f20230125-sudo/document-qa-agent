# Document Q&A Agent

A small Retrieval-Augmented Generation (RAG) service: it answers questions
about a set of internal documents (here, sample HR policies) by retrieving
the most relevant passages and asking an LLM to answer grounded in that
retrieved context, instead of relying on the model's own memory.

## How it works

1. **Chunking** (`build_index.py`) — splits each `.txt` document in
   `documents/` into overlapping ~500-character chunks.
2. **Embeddings** — each chunk is converted into a vector using Google's
   `gemini-embedding-001` model via the Gemini API.
3. **Vector store** — all chunk vectors are stored in a FAISS index
   (`vector_store.faiss`) for fast similarity search, saved alongside the
   original chunk text (`chunks.pkl`).
4. **Retrieval + generation** (`app.py`) — a Flask endpoint `/ask` embeds
   an incoming question, retrieves the top-3 most similar chunks from
   FAISS, and passes them plus the question to Gemini (`gemini-3.6-flash`)
   to produce a grounded answer, along with which source document(s) it
   used.

```
question -> embed -> FAISS similarity search -> top-k chunks
                                                     |
                                                     v
                              prompt = chunks + question -> Gemini -> answer
```

## Stack

Python, Flask, Google Gemini API (`google-generativeai`), FAISS
(`faiss-cpu`), NumPy.

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com/)
   (no credit card required).
3. Copy `.env.example` to `.env` and paste in your key:
   ```
   cp .env.example .env
   ```
4. Build the vector index from the sample documents:
   ```
   python build_index.py
   ```
5. Start the API:
   ```
   python app.py
   ```

## Usage

With `app.py` running in one terminal, open a second terminal and either:

**Option A — the included helper script (works the same on Windows/Mac/Linux):**
```bash
python ask.py "How many days of annual leave do I get?"
```

**Option B — curl:**
```bash
curl -X POST http://localhost:5000/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "How many days of annual leave do I get?"}'
```

Example response:
```json
{
  "question": "How many days of annual leave do I get?",
  "answer": "You are entitled to 25 working days of paid annual leave per calendar year.",
  "sources": ["leave_policy.txt"]
}
```

Health check:
```bash
curl http://localhost:5000/health
```

## Tests

```bash
pip install pytest
pytest
```

The tests swap Gemini for a stub client and build a small FAISS index in a
temporary folder, so they run without an API key and make no network calls.
They cover chunking, embedding, the saved index, retrieval order, the prompt,
and the `/ask` and `/health` routes.

## Notes

- Swap in your own documents by dropping `.txt` files into `documents/`
  and re-running `build_index.py`.
- `vector_store.faiss` and `chunks.pkl` are generated files (not committed)
  — regenerate them locally with `build_index.py` before running `app.py`.
- Chunk size, overlap, and `TOP_K` (how many chunks are retrieved per
  question) are configurable constants at the top of `build_index.py` and
  `app.py`.
