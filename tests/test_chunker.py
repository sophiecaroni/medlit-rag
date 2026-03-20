import math
from ingestion.chunker import chunk_text, tokenizer


def test_chunk_text_count():
    text = "Metformin is a first-line treatment for type 2 diabetes. It works by decreasing hepatic glucose production..."
    chunk_size = 10
    overlap = 2
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    expect = math.ceil(len(tokens) / (chunk_size-overlap))
    assert expect == len(chunks)


def test_chunk_text_empty_input():
    text = ""
    chunks = chunk_text(text)
    assert chunks == []


def test_chunk_text_overlap(monkeypatch):
    text_ex = "Metformin is a first-line treatment for type 2 diabetes"

    def mock_token_ids(text, add_special_tokens=False):
        return list(range(len(text.split(' '))))

    def mock_decode(ids):
        return ids  # ids already computed in mock_token_ids are returned unchanged
    chunk_size = 4
    overlap = 2
    monkeypatch.setattr(tokenizer, "encode", mock_token_ids)
    monkeypatch.setattr(tokenizer, "decode", mock_decode)
    chunks = chunk_text(text_ex, chunk_size=chunk_size, overlap=overlap)

    # Check end of a token equals start of next token
    for chunk_i in range(len(chunks)-1):
        this_chunk = chunks[chunk_i]
        next_chunk = chunks[chunk_i+1]
        assert this_chunk[-overlap:] == next_chunk[:overlap]

