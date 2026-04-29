"""
MultiModalRetriever — query the UnifiedFAISSIndex using CLIP text embeddings.

The query is encoded with CLIP's text encoder (same space as both text and image
embeddings), so a single vector search spans both modalities simultaneously.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from src.clip_embedder import embed_texts
from src.multimodal.unified_index import UnifiedFAISSIndex, ModalityEntry


@dataclass
class RetrievalResult:
    entry: ModalityEntry
    score: float

    @property
    def modality(self) -> str:
        return self.entry.modality

    @property
    def snippet(self) -> str:
        if self.entry.modality == "text":
            return (self.entry.text or "")[:300]
        return f"[Image] {Path(self.entry.image_path).name}  (p.{self.entry.page_number})"

    def __repr__(self) -> str:
        return f"RetrievalResult(modality={self.modality!r}, score={self.score:.4f}, snippet={self.snippet!r})"


class MultiModalRetriever:
    """
    Thin wrapper that encodes a text query with CLIP and searches the UnifiedFAISSIndex.

    Args:
        index:         A loaded UnifiedFAISSIndex (call UnifiedFAISSIndex.load() or use
                       IngestionPipeline.index directly).
        top_k:         Default number of results to return.
        image_boost:   Additive score bonus for image results to compensate for the
                       CLIP modality gap (images embed slightly lower than matching text).
                       Typical range: 0.0–0.1.

    Example:
        retriever = MultiModalRetriever(UnifiedFAISSIndex.load("data/multimodal/unified_index"))
        results = retriever.retrieve("What is a B+ tree?")
        for r in results:
            print(r)
    """

    def __init__(
        self,
        index: UnifiedFAISSIndex,
        top_k: int = 10,
        image_boost: float = 0.05,
    ):
        self.index = index
        self.top_k = top_k
        self.image_boost = image_boost

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        modality_filter: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """
        Encode query with CLIP text encoder and return top-k results.

        Args:
            query:           Natural-language question.
            top_k:           Override instance default.
            modality_filter: 'text', 'image', or None for both.

        Returns:
            List of RetrievalResult sorted by descending score.
        """
        k = top_k if top_k is not None else self.top_k
        query_emb = embed_texts([query])[0]  # (768,)
        raw = self.index.search(
            query_emb,
            top_k=k,
            modality_filter=modality_filter,
            image_boost=self.image_boost,
        )
        return [RetrievalResult(entry=entry, score=score) for entry, score in raw]

    def retrieve_text_only(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        return self.retrieve(query, top_k=top_k, modality_filter="text")

    def retrieve_images_only(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        return self.retrieve(query, top_k=top_k, modality_filter="image")

    # ------------------------------------------------------------------
    # Convenience: format results for LLM prompt injection
    # ------------------------------------------------------------------

    def format_for_prompt(self, results: List[RetrievalResult]) -> str:
        """
        Format retrieval results as a context block suitable for an LLM prompt.
        Image results become a caption-style line; text results are quoted directly.
        """
        lines = []
        for i, r in enumerate(results, 1):
            if r.modality == "text":
                lines.append(f"[{i}] (text, p.{r.entry.page_number}) {r.snippet}")
            else:
                lines.append(
                    f"[{i}] (figure, p.{r.entry.page_number}) "
                    f"{Path(r.entry.image_path).name}"
                )
        return "\n".join(lines)
