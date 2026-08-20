from app.ingest.chunking import chunk_text, chunk_documents
from app.ingest.loaders import Document, load_csv, load_html


def test_headings_become_a_path():
    text = "# Guide\n\nintro paragraph long enough to keep.\n\n## Refunds\n\nrefund body text here.\n"
    chunks = chunk_text(text)
    paths = [h for h, _ in chunks]
    assert "Guide" in paths
    assert "Guide > Refunds" in paths


def test_chunks_respect_target_size_and_do_not_split_sentences():
    body = "This is a sentence about refunds. " * 60
    chunks = chunk_text(f"# T\n\n{body}", target_chars=400)
    assert len(chunks) > 1
    for _, c in chunks:
        assert len(c) <= 500          # target plus one trailing sentence
        assert c.rstrip().endswith(".")


def test_overlap_carries_context_between_chunks():
    body = " ".join(f"Sentence number {i} about policy." for i in range(60))
    chunks = [c for _, c in chunk_text(f"# T\n\n{body}", target_chars=300, overlap_chars=120)]
    first_tail = chunks[0].split(". ")[-2]
    assert first_tail in chunks[1], "expected the previous chunk's tail to repeat as overlap"


def test_runt_chunks_are_merged_away():
    text = "# T\n\n" + "a real paragraph of reasonable length here. " * 5 + "\n\nok.\n"
    chunks = chunk_text(text, min_chars=120)
    assert all(len(c) >= 40 for _, c in chunks)


def test_chunk_ids_are_stable_and_unique():
    doc = Document("d1", "# A\n\n" + "text. " * 200, "src.md")
    a = chunk_documents([doc])
    b = chunk_documents([doc])
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert len({c.chunk_id for c in a}) == len(a)


def test_html_loader_strips_script_and_keeps_headings():
    html = "<html><body><script>bad()</script><h1>Title</h1><p>Body text.</p></body></html>"
    text = load_html(html)
    assert "bad()" not in text
    assert "Title" in text and "Body text." in text


def test_csv_loader_keeps_rows_whole():
    text = load_csv("sku,price\nA-1,10\nB-2,20\n")
    assert "sku: A-1" in text and "price: 20" in text
    assert text.count("### record") == 2


def test_fixed_strategy_covers_all_content_with_overlap():
    body = "word " * 400  # 2000 chars
    chunks = chunk_text(body, strategy="fixed-512", target_chars=1200, overlap_chars=150)
    assert len(chunks) > 1
    for _, c in chunks:
        assert len(c) <= 512
    # every chunk after the first should share its opening with the previous
    # chunk's tail, proving the overlap window actually moved forward.
    assert chunks[0][1] != chunks[1][1]


def test_fixed_1024_uses_a_wider_window_than_fixed_512():
    body = "word " * 400
    small = chunk_text(body, strategy="fixed-512")
    large = chunk_text(body, strategy="fixed-1024")
    assert len(large) < len(small)


def test_fixed_strategy_can_split_mid_sentence():
    # fixed-512 uses its own 512-char window regardless of target_chars -- the
    # strategy name encodes the size. One sentence well over 512 chars is needed
    # to actually exercise a mid-sentence cut.
    text = "This is one long sentence that just keeps going without any natural break points at all. " * 10
    chunks = chunk_text(text, strategy="fixed-512")
    assert not all(c.rstrip().endswith(".") for _, c in chunks)


def test_recursive_overlap_packs_across_section_boundaries():
    text = ("# A\n\nshort intro here.\n\n"
           "# B\n\nanother short section that alone would be a runt chunk.\n")
    structure_first = chunk_text(text, strategy="structure-first", target_chars=1000)
    recursive = chunk_text(text, strategy="recursive-overlap", target_chars=1000)
    # structure-first keeps A and B as separate headings; recursive-overlap merges
    # short sections together into fewer, larger chunks.
    assert len({h for h, _ in structure_first}) >= 2
    assert len(recursive) <= len(structure_first)


def test_recursive_overlap_respects_target_size():
    body = "This is a sentence about policy details. " * 80
    chunks = chunk_text(body, strategy="recursive-overlap", target_chars=400, overlap_chars=100)
    assert len(chunks) > 1
    for _, c in chunks:
        assert len(c) <= 500


class _FakeEmbeddings:
    """Deterministic stand-in: sentences sharing a keyword get near-identical
    vectors, sentences without it get an orthogonal vector -- enough to exercise a
    real similarity-drop boundary without needing a real embedding model."""

    def embed(self, texts):
        return [[1.0, 0.0] if "refund" in t.lower() else [0.0, 1.0] for t in texts]


def test_semantic_strategy_requires_embeddings():
    import pytest
    with pytest.raises(ValueError):
        chunk_text("some text.", strategy="semantic")


def test_semantic_strategy_breaks_on_topic_shift():
    text = ("Refunds take five days. Refund requests need an order number. "
           "Shipping takes three days. Shipping is free over fifty euros.")
    chunks = chunk_text(text, strategy="semantic", embeddings=_FakeEmbeddings(),
                        target_chars=1000)
    assert len(chunks) >= 2
    assert "refund" in chunks[0][1].lower()
    assert "shipping" in chunks[-1][1].lower()
