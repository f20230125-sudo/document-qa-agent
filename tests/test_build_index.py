"""Tests for build_index.py: loading, chunking, embedding and the saved index."""

import pickle

import faiss
import pytest

import build_index
from build_index import chunk_text, embed_texts, load_documents
from conftest import StubClient


def test_chunk_text_returns_nothing_for_blank_text():
    assert chunk_text("") == []
    assert chunk_text("   \n\t  ") == []


def test_chunk_text_keeps_short_text_as_one_chunk():
    assert chunk_text("  Annual leave is 25 days.  ") == ["Annual leave is 25 days."]


def test_chunk_text_respects_the_chunk_size():
    text = "".join(str(i % 10) for i in range(1234))

    chunks = chunk_text(text, chunk_size=200, overlap=50)

    assert all(len(chunk) <= 200 for chunk in chunks)
    assert chunks[0] == text[:200]


def test_chunk_text_overlaps_consecutive_chunks():
    text = "".join(str(i % 10) for i in range(1000))

    chunks = chunk_text(text, chunk_size=200, overlap=50)

    for previous, current in zip(chunks, chunks[1:]):
        assert previous[-50:] == current[:50]


def test_chunk_text_covers_the_whole_document():
    text = "".join(str(i % 10) for i in range(1000))

    chunks = chunk_text(text, chunk_size=200, overlap=50)

    # Each chunk starts 150 characters after the previous one, so dropping the
    # 50-character overlap from every later chunk rebuilds the original text.
    rebuilt = chunks[0] + "".join(chunk[50:] for chunk in chunks[1:])
    assert rebuilt == text


def test_load_documents_reads_only_txt_files_in_name_order(tmp_path):
    (tmp_path / "b_policy.txt").write_text("second", encoding="utf-8")
    (tmp_path / "a_policy.txt").write_text("first", encoding="utf-8")
    (tmp_path / "notes.md").write_text("ignored", encoding="utf-8")

    assert load_documents(str(tmp_path)) == [
        {"source": "a_policy.txt", "text": "first"},
        {"source": "b_policy.txt", "text": "second"},
    ]


def test_embed_texts_returns_one_float32_row_per_text():
    client = StubClient(embed=lambda text: [float(len(text)), 1.0, 0.0])

    vectors = embed_texts(client, ["ab", "abcd"])

    assert vectors.shape == (2, 3)
    assert vectors.dtype == "float32"
    assert vectors[:, 0].tolist() == [2.0, 4.0]


def test_embed_texts_embeds_chunks_as_retrieval_documents():
    client = StubClient(embed=lambda text: [1.0, 0.0])

    embed_texts(client, ["some chunk"])

    call = client.embed_calls[0]
    assert call["model"] == build_index.EMBEDDING_MODEL
    assert call["contents"] == "some chunk"
    assert call["config"].task_type == "RETRIEVAL_DOCUMENT"


def test_main_needs_an_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(SystemExit, match="GEMINI_API_KEY is not set"):
        build_index.main()


def test_main_stops_when_there_are_no_documents(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(build_index.genai, "Client", lambda api_key: StubClient(embed=lambda t: [1.0]))

    with pytest.raises(SystemExit, match="No .txt files found"):
        build_index.main()


def test_main_writes_an_index_with_one_vector_per_chunk(monkeypatch, tmp_path):
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "leave_policy.txt").write_text("L" * 900, encoding="utf-8")
    (documents / "remote_work_policy.txt").write_text("Remote work is allowed.", encoding="utf-8")
    client = StubClient(embed=lambda text: [float(len(text)), 1.0, 0.0, 0.0])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(build_index.genai, "Client", lambda api_key: client)

    build_index.main()

    with open(tmp_path / build_index.CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)
    index = faiss.read_index(str(tmp_path / build_index.INDEX_FILE))

    # 900 characters at size 500 / overlap 100 gives chunks starting at 0, 400 and 800.
    assert [c["source"] for c in chunks] == ["leave_policy.txt"] * 3 + ["remote_work_policy.txt"]
    assert chunks[-1]["text"] == "Remote work is allowed."
    assert index.ntotal == len(chunks) == len(client.embed_calls)
    assert index.d == 4
