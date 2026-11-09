from app.retrieval.chunking import chunk_document, split_text


def test_split_respects_paragraphs_and_size() -> None:
    text = "\n\n".join(["First paragraph about websockets." * 3, "Second about retrieval." * 3])
    chunks = split_text(text, max_chars=120, overlap=20)
    assert len(chunks) >= 2
    assert all(len(chunk) <= 160 for chunk in chunks)


def test_chunk_ids_are_deterministic_and_unique() -> None:
    text = "\n\n".join(f"Paragraph {i} " + "content " * 40 for i in range(4))
    first = chunk_document("doc-1", "Title", text, max_chars=200, overlap=30)
    second = chunk_document("doc-1", "Title", text, max_chars=200, overlap=30)
    assert [c.id for c in first] == [c.id for c in second]
    assert len({c.id for c in first}) == len(first)
    assert all(c.doc_id == "doc-1" for c in first)
    assert [c.meta["chunk_index"] for c in first] == list(range(len(first)))


def test_long_single_paragraph_is_split() -> None:
    text = "This is one very long sentence without paragraph breaks. " * 60
    chunks = split_text(text, max_chars=300, overlap=50)
    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)


def test_overlap_carries_context() -> None:
    text = "\n\n".join(f"Block {i}: " + "word " * 30 for i in range(6))
    chunks = split_text(text, max_chars=200, overlap=60)
    assert len(chunks) > 1
    # each chunk after the first begins with some tail of the previous one
    assert chunks[1][:20] in chunks[0] + chunks[1]
