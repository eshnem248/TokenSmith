"""
UnifiedFAISSIndex — single FAISS IndexFlatIP index for both text and image modalities.

All vectors are 768-dim CLIP embeddings, L2-normalized so that inner product == cosine similarity.
Metadata tracks modality ('text' or 'image'), source file, page number, and chunk text or image path.
"""

import json
import pickle
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import faiss
import numpy as np


EMBEDDING_DIM = 768


@dataclass
class ModalityEntry:
    idx: int              # position in FAISS index
    modality: str         # 'text' or 'image'
    source: str           # originating file path
    page_number: int      # 1-indexed; 0 if unknown
    text: Optional[str]   # chunk text (text entries only)
    image_path: Optional[str]  # PNG path (image entries only)
    extra: dict = field(default_factory=dict)


class UnifiedFAISSIndex:
    """
    Wraps a FAISS IndexFlatIP for joint text+image retrieval with CLIP embeddings.

    Usage:
        idx = UnifiedFAISSIndex()
        idx.add_vectors(embeddings, entries)
        results = idx.search(query_emb, top_k=10)
        idx.save("data/multimodal_index")   # writes .faiss + _meta.pkl
        idx = UnifiedFAISSIndex.load("data/multimodal_index")
    """

    def __init__(self):
        self._index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self._entries: List[ModalityEntry] = []

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def add_vectors(self, embeddings: np.ndarray, entries: List[ModalityEntry]) -> None:
        """
        Add a batch of L2-normalized 768-dim embeddings and their metadata entries.
        embeddings shape: (N, 768), dtype float32.
        """
        if embeddings.shape[0] != len(entries):
            raise ValueError(
                f"embeddings rows ({embeddings.shape[0]}) != entries ({len(entries)})"
            )
        if embeddings.shape[1] != EMBEDDING_DIM:
            raise ValueError(
                f"Expected {EMBEDDING_DIM}-dim embeddings, got {embeddings.shape[1]}"
            )
        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
        base = len(self._entries)
        for i, e in enumerate(entries):
            e.idx = base + i
        self._index.add(embeddings)
        self._entries.extend(entries)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def search(
        self,
        query_emb: np.ndarray,
        top_k: int = 10,
        modality_filter: Optional[str] = None,
        image_boost: float = 0.0,
    ) -> List[Tuple[ModalityEntry, float]]:
        """
        Return top-k (entry, score) pairs sorted by descending cosine similarity.

        Args:
            query_emb:       (1, 768) or (768,) float32 L2-normalized vector.
            top_k:           Number of results to return.
            modality_filter: If 'text' or 'image', restrict results to that modality.
            image_boost:     Additive bonus applied to image scores (addresses CLIP modality gap).
        """
        if self._index.ntotal == 0:
            return []

        q = np.ascontiguousarray(
            query_emb.reshape(1, EMBEDDING_DIM).astype(np.float32)
        )
        # Over-fetch so we can filter and still return top_k
        fetch_k = min(self._index.ntotal, top_k * 5 if modality_filter else top_k)
        scores, indices = self._index.search(q, fetch_k)
        scores = scores[0]
        indices = indices[0]

        results = []
        for score, idx in zip(scores, indices):
            if idx < 0:
                continue
            entry = self._entries[idx]
            adjusted = float(score) + (image_boost if entry.modality == "image" else 0.0)
            if modality_filter and entry.modality != modality_filter:
                continue
            results.append((entry, adjusted))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, prefix: Union[str, Path]) -> None:
        """
        Write index to {prefix}.faiss and metadata to {prefix}_meta.pkl.
        """
        prefix = Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(prefix.with_suffix(".faiss")))
        with open(str(prefix) + "_meta.pkl", "wb") as f:
            pickle.dump(self._entries, f)

    @classmethod
    def load(cls, prefix: Union[str, Path]) -> "UnifiedFAISSIndex":
        """
        Load from {prefix}.faiss and {prefix}_meta.pkl.
        """
        prefix = Path(prefix)
        obj = cls.__new__(cls)
        obj._index = faiss.read_index(str(prefix.with_suffix(".faiss")))
        with open(str(prefix) + "_meta.pkl", "rb") as f:
            obj._entries = pickle.load(f)
        return obj

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    @property
    def ntotal(self) -> int:
        return self._index.ntotal

    def stats(self) -> dict:
        text_n = sum(1 for e in self._entries if e.modality == "text")
        image_n = sum(1 for e in self._entries if e.modality == "image")
        return {"total": self.ntotal, "text": text_n, "image": image_n}
