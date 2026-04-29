from typing import List, Dict, Optional, Any
from difflib import SequenceMatcher
from tests.metrics.base import MetricBase


class ChunkRetrievalMetric(MetricBase):
    """Chunk retrieval evaluation metric.

    Returns a 0–1 recall score: fraction of ideal chunks found in the
    retrieved set.  Returns 0.0 gracefully when ideal_retrieved_chunks or
    retrieved_chunks is None/empty.
    """

    def __init__(self, similarity_threshold: float = 0.95):
        self.similarity_threshold = similarity_threshold

    @property
    def name(self) -> str:
        return "chunk_retrieval"

    @property
    def weight(self) -> float:
        return 0.5

    def calculate(self,
                  ideal_retrieved_chunks: Optional[List[int]],
                  retrieved_chunks) -> float:
        if not ideal_retrieved_chunks or not retrieved_chunks:
            return 0.0

        retrieved_ids = {chunk["chunk_id"] for chunk in retrieved_chunks}
        found = sum(1 for cid in ideal_retrieved_chunks if cid in retrieved_ids)
        return found / len(ideal_retrieved_chunks)
