"""
IngestionPipeline — ingest a PDF into the UnifiedFAISSIndex.

Two passes per document:
  1. Text pass  — chunk raw text with the existing DocumentChunker, embed with CLIP text encoder.
  2. Image pass — extract embedded figures with PDFImageExtractor, embed with CLIP image encoder.

Both passes write into the same UnifiedFAISSIndex so queries span both modalities.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from src.clip_embedder import embed_texts, embed_images
from src.pdf_image_extractor import extract_images_from_pdf, ExtractedImage
from src.multimodal.unified_index import UnifiedFAISSIndex, ModalityEntry


# Characters per text chunk fed to CLIP.
# CLIP's text encoder silently truncates at 77 tokens (~300-400 chars).
_TEXT_CHUNK_SIZE = 350
_TEXT_CHUNK_OVERLAP = 50


def _simple_chunks(text: str, size: int = _TEXT_CHUNK_SIZE, overlap: int = _TEXT_CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping character-level windows."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start += size - overlap
    return chunks


class IngestionPipeline:
    """
    Orchestrate PDF → UnifiedFAISSIndex for both text and image modalities.

    Example:
        pipeline = IngestionPipeline(index_dir="data/multimodal")
        pipeline.ingest_pdf("data/chapters/silberschatz.pdf")
        pipeline.save()           # writes data/multimodal/unified_index.faiss + _meta.pkl

        # Later:
        pipeline = IngestionPipeline.load("data/multimodal")
    """

    def __init__(
        self,
        index_dir: Union[str, Path] = "data/multimodal",
        image_output_dir: Optional[Union[str, Path]] = None,
        min_image_width: int = 100,
        min_image_height: int = 100,
        text_batch_size: int = 64,
        image_batch_size: int = 16,
    ):
        self.index_dir = Path(index_dir)
        self.image_output_dir = Path(image_output_dir or self.index_dir / "figures")
        self.min_image_width = min_image_width
        self.min_image_height = min_image_height
        self.text_batch_size = text_batch_size
        self.image_batch_size = image_batch_size
        self.index = UnifiedFAISSIndex()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_pdf(self, pdf_path: Union[str, Path]) -> dict:
        """
        Ingest one PDF — both text chunks and embedded figures — into self.index.

        Returns a summary dict with counts of vectors added per modality.
        """
        pdf_path = Path(pdf_path)
        text_added = self._ingest_text(pdf_path)
        image_added = self._ingest_images(pdf_path)
        summary = {"pdf": str(pdf_path), "text_vectors": text_added, "image_vectors": image_added}
        print(
            f"Ingested {pdf_path.name}: "
            f"{text_added} text vectors, {image_added} image vectors"
        )
        return summary

    def save(self) -> None:
        """Persist the index to disk."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index.save(self.index_dir / "unified_index")
        print(f"Saved unified index to {self.index_dir}  ({self.index.stats()})")

    @classmethod
    def load(cls, index_dir: Union[str, Path]) -> "IngestionPipeline":
        """Load a previously saved pipeline (index only, not config)."""
        obj = cls.__new__(cls)
        obj.index_dir = Path(index_dir)
        obj.index = UnifiedFAISSIndex.load(obj.index_dir / "unified_index")
        return obj

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ingest_text(self, pdf_path: Path, page_stride: int = 200) -> int:
        """Extract text from PDF, chunk, embed with CLIP, add to index.

        Processes pages in strided batches to cap peak memory usage.
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            print("PyMuPDF not installed — skipping text pass")
            return 0

        doc = fitz.open(str(pdf_path))
        n_pages = len(doc)
        total_added = 0

        for batch_start in range(0, n_pages, page_stride):
            batch_end = min(batch_start + page_stride, n_pages)
            chunks: List[str] = []
            entries: List[ModalityEntry] = []

            for page_num in range(batch_start, batch_end):
                text = re.sub(r"\s+", " ", doc[page_num].get_text("text").strip())
                if not text:
                    continue
                for chunk in _simple_chunks(text):
                    chunks.append(chunk)
                    entries.append(
                        ModalityEntry(
                            idx=0,
                            modality="text",
                            source=str(pdf_path),
                            page_number=page_num + 1,
                            text=chunk,
                            image_path=None,
                        )
                    )

            if chunks:
                embeddings = embed_texts(chunks, batch_size=self.text_batch_size)
                self.index.add_vectors(embeddings, entries)
                total_added += len(chunks)
                print(f"  text pages {batch_start+1}-{batch_end}: {len(chunks)} chunks embedded")

        doc.close()
        return total_added

    def _ingest_images(self, pdf_path: Path) -> int:
        """Extract figures from PDF, embed with CLIP image encoder, add to index."""
        images: List[ExtractedImage] = extract_images_from_pdf(
            pdf_path=pdf_path,
            output_dir=self.image_output_dir,
            min_width=self.min_image_width,
            min_height=self.min_image_height,
        )

        if not images:
            return 0

        image_paths = [img.image_path for img in images]
        embeddings = embed_images(image_paths, batch_size=self.image_batch_size)

        entries = [
            ModalityEntry(
                idx=0,
                modality="image",
                source=img.source_pdf,
                page_number=img.page_number,
                text=None,
                image_path=img.image_path,
                extra={
                    "width": img.width,
                    "height": img.height,
                    "image_id": img.image_id,
                },
            )
            for img in images
        ]

        self.index.add_vectors(embeddings, entries)
        return len(images)
