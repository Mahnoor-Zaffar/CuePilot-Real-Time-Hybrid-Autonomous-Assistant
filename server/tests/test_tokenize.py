from app.retrieval.tokenize import char_ngrams, tokenize


def test_snake_case_keeps_compound_and_parts() -> None:
    tokens = tokenize("stream_window_ms")
    assert "stream_window_ms" in tokens
    assert {"stream", "window", "ms"} <= set(tokens)


def test_camel_case_splitting() -> None:
    tokens = tokenize("getUserID")
    assert {"getuserid", "get", "user", "id"} <= set(tokens)


def test_error_code_variants_match() -> None:
    doc_tokens = set(tokenize("CUE-E1042 means the context window overflowed"))
    query_tokens = set(tokenize("E1042"))
    assert query_tokens & doc_tokens, "error code forms must share at least one token"


def test_stopwords_and_short_tokens_dropped() -> None:
    tokens = tokenize("the a of is to it")
    assert tokens == []


def test_punctuation_and_paths() -> None:
    tokens = tokenize("/ws/stream and retry_backoff_cap_ms.")
    assert "ws" in tokens
    assert "stream" in tokens
    assert "retry_backoff_cap_ms" in tokens


def test_char_ngrams_are_bounded() -> None:
    grams = char_ngrams("x" * 5000, n=3, max_ngrams=50)
    assert len(grams) == 50
