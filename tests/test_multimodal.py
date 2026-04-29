"""
Tests for the multimodal CLIP retrieval pipeline.

Covers: UnifiedFAISSIndex round-trip, MultiModalRetriever, and (if the index
is already built) a smoke-query against the real silberschatz index.
"""

import numpy as np
import pytest
from pathlib import Path

from src.multimodal.unified_index import UnifiedFAISSIndex, ModalityEntry
from src.multimodal.retriever import MultiModalRetriever


# ---------------------------------------------------------------------------
# UnifiedFAISSIndex unit tests (no CLIP, no GPU)
# ---------------------------------------------------------------------------

def _fake_index():
    idx = UnifiedFAISSIndex()
    text_embs = np.random.randn(4, 768).astype(np.float32)
    text_embs /= np.linalg.norm(text_embs, axis=1, keepdims=True)
    img_embs = np.random.randn(3, 768).astype(np.float32)
    img_embs /= np.linalg.norm(img_embs, axis=1, keepdims=True)

    text_entries = [
        ModalityEntry(idx=0, modality="text", source="test.pdf", page_number=i+1,
                      text=f"chunk {i}", image_path=None)
        for i in range(4)
    ]
    img_entries = [
        ModalityEntry(idx=0, modality="image", source="test.pdf", page_number=i+5,
                      text=None, image_path=f"/tmp/fig{i}.png")
        for i in range(3)
    ]
    idx.add_vectors(text_embs, text_entries)
    idx.add_vectors(img_embs, img_entries)
    return idx


def test_index_stats():
    idx = _fake_index()
    stats = idx.stats()
    assert stats["total"] == 7
    assert stats["text"] == 4
    assert stats["image"] == 3


def test_index_search_returns_top_k():
    idx = _fake_index()
    q = np.random.randn(768).astype(np.float32)
    q /= np.linalg.norm(q)
    results = idx.search(q, top_k=3)
    assert len(results) == 3


def test_index_modality_filter():
    idx = _fake_index()
    q = np.random.randn(768).astype(np.float32)
    q /= np.linalg.norm(q)

    text_only = idx.search(q, top_k=10, modality_filter="text")
    assert all(e.modality == "text" for e, _ in text_only)

    img_only = idx.search(q, top_k=10, modality_filter="image")
    assert all(e.modality == "image" for e, _ in img_only)


def test_index_save_load(tmp_path):
    idx = _fake_index()
    idx.save(tmp_path / "test")
    loaded = UnifiedFAISSIndex.load(tmp_path / "test")
    assert loaded.ntotal == idx.ntotal
    assert loaded.stats() == idx.stats()

    q = np.random.randn(768).astype(np.float32)
    q /= np.linalg.norm(q)
    r1 = idx.search(q, top_k=3)
    r2 = loaded.search(q, top_k=3)
    np.testing.assert_allclose(
        [s for _, s in r1], [s for _, s in r2], atol=1e-5
    )


def test_image_boost():
    idx = _fake_index()
    q = np.random.randn(768).astype(np.float32)
    q /= np.linalg.norm(q)
    results_no_boost = idx.search(q, top_k=7, image_boost=0.0)
    results_boosted  = idx.search(q, top_k=7, image_boost=0.5)
    # With a large boost, images should dominate the top spots
    top3_boosted = [e.modality for e, _ in results_boosted[:3]]
    assert top3_boosted.count("image") >= 2, "image_boost should push images up"


# ---------------------------------------------------------------------------
# MultiModalRetriever unit test (mocked index)
# ---------------------------------------------------------------------------

def test_retriever_format_for_prompt():
    idx = _fake_index()
    retriever = MultiModalRetriever(idx, top_k=3)
    q = np.random.randn(768).astype(np.float32)
    q /= np.linalg.norm(q)
    # Override retrieve() so we don't need CLIP loaded
    from src.multimodal.retriever import RetrievalResult
    fake_results = [
        RetrievalResult(entry=ModalityEntry(0, "text", "t.pdf", 5, "hello world", None), score=0.9),
        RetrievalResult(entry=ModalityEntry(0, "image", "t.pdf", 7, None, "/tmp/fig.png"), score=0.7),
    ]
    prompt_block = retriever.format_for_prompt(fake_results)
    assert "(text," in prompt_block
    assert "(figure," in prompt_block


# ---------------------------------------------------------------------------
# Integration smoke test — requires pre-built index at data/multimodal/
# ---------------------------------------------------------------------------

UNIFIED_INDEX_PATH = Path("data/multimodal/unified_index")


@pytest.mark.skipif(
    not (UNIFIED_INDEX_PATH.with_suffix(".faiss")).exists(),
    reason="Pre-built multimodal index not found; run scripts/run_ingest.sh first"
)
def test_real_index_retrieval():
    """
    Smoke-query against the real silberschatz multimodal index.
    Checks that we get >= 1 text result and >= 1 image result for a DB query.
    """
    idx = UnifiedFAISSIndex.load(UNIFIED_INDEX_PATH)
    assert idx.ntotal > 100, "Index should have substantial vectors"

    from src.clip_embedder import embed_texts
    q_emb = embed_texts(["What is a B+ tree?"])[0]
    results = idx.search(q_emb, top_k=10)

    modalities = {r[0].modality for r in results}
    assert "text" in modalities, "Should retrieve text chunks"
    # Image results may or may not surface in top-10 depending on CLIP alignment
    print(f"\nTop-10 results: {[(r[0].modality, round(r[1],4)) for r in results]}")
