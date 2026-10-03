"""Tests for app.py: retrieval, the prompt, and the /ask and /health routes.

app.py loads its index when it is imported, so each test gets a small index
built in a temporary folder and a freshly imported app with a stubbed client.
"""

import importlib
import pickle
import sys

import faiss
import numpy as np
import pytest

from conftest import StubClient

CHUNKS = [
    {"source": "leave_policy.txt", "text": "Employees get 25 days of annual leave."},
    {"source": "remote_work_policy.txt", "text": "Remote work is allowed 3 days a week."},
    {"source": "expense_reimbursement_policy.txt", "text": "Submit expenses within 30 days."},
]
VECTORS = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
ANSWER = "You get 25 days of annual leave."


def embed_question(text):
    """Place a question closest to the chunk about its topic."""
    if "leave" in text:
        return [0.9, 0.1, 0.0]
    if "remote" in text:
        return [0.1, 0.9, 0.0]
    return [0.0, 0.1, 0.9]


def load_app(tmp_path, monkeypatch, chunks=CHUNKS, vectors=VECTORS):
    index = faiss.IndexFlatL2(3)
    if vectors:
        index.add(np.array(vectors, dtype="float32"))
    faiss.write_index(index, str(tmp_path / "vector_store.faiss"))
    with open(tmp_path / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    sys.modules.pop("app", None)
    module = importlib.import_module("app")
    module.client = StubClient(embed=embed_question, answer=f"  {ANSWER}\n")
    return module


@pytest.fixture
def qa(tmp_path, monkeypatch):
    yield load_app(tmp_path, monkeypatch)
    sys.modules.pop("app", None)


def test_import_fails_clearly_when_the_index_has_not_been_built(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    sys.modules.pop("app", None)

    with pytest.raises(SystemExit, match="build_index.py"):
        importlib.import_module("app")


def test_retrieval_returns_the_closest_chunk_first(qa):
    results = qa.retrieve_relevant_chunks("How much annual leave do I get?", top_k=2)

    assert [r["source"] for r in results] == ["leave_policy.txt", "remote_work_policy.txt"]
    assert results[0]["distance"] < results[1]["distance"]


def test_retrieval_embeds_the_question_as_a_retrieval_query(qa):
    qa.retrieve_relevant_chunks("Can I work remote on Fridays?", top_k=1)

    call = qa.client.embed_calls[0]
    assert call["model"] == qa.EMBEDDING_MODEL
    assert call["config"].task_type == "RETRIEVAL_QUERY"


def test_retrieval_skips_empty_slots_when_the_index_is_smaller_than_top_k(tmp_path, monkeypatch):
    qa = load_app(tmp_path, monkeypatch, chunks=CHUNKS[:1], vectors=VECTORS[:1])

    results = qa.retrieve_relevant_chunks("How much annual leave do I get?", top_k=3)

    assert [r["source"] for r in results] == ["leave_policy.txt"]


def test_prompt_contains_the_question_and_each_source(qa):
    prompt = qa.build_prompt("How much annual leave do I get?", CHUNKS[:2])

    assert "Question: How much annual leave do I get?" in prompt
    assert "[Source: leave_policy.txt]\nEmployees get 25 days of annual leave." in prompt
    assert "[Source: remote_work_policy.txt]" in prompt
    assert "ONLY" in prompt


def test_ask_returns_the_answer_with_its_sources(qa):
    response = qa.app.test_client().post("/ask", json={"question": " How much annual leave do I get? "})

    assert response.status_code == 200
    assert response.get_json() == {
        "question": "How much annual leave do I get?",
        "answer": ANSWER,
        "sources": sorted(c["source"] for c in CHUNKS),
    }
    chat = qa.client.chat_calls[0]
    assert chat["model"] == qa.CHAT_MODEL
    assert "Employees get 25 days of annual leave." in chat["contents"]


@pytest.mark.parametrize("body", [{}, {"question": "   "}, {"other": "field"}])
def test_ask_rejects_a_missing_question(qa, body):
    response = qa.app.test_client().post("/ask", json=body)

    assert response.status_code == 400
    assert "question" in response.get_json()["error"]
    assert qa.client.chat_calls == []


def test_ask_rejects_a_body_that_is_not_json(qa):
    response = qa.app.test_client().post("/ask", data="not json", content_type="text/plain")

    assert response.status_code == 400


def test_ask_reports_an_empty_index(tmp_path, monkeypatch):
    qa = load_app(tmp_path, monkeypatch, chunks=[], vectors=[])

    response = qa.app.test_client().post("/ask", json={"question": "How much annual leave do I get?"})

    assert response.status_code == 500
    assert response.get_json() == {"error": "No indexed documents to search"}


def test_health_reports_how_many_chunks_are_indexed(qa):
    response = qa.app.test_client().get("/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok", "chunks_indexed": len(CHUNKS)}
