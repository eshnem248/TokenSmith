#!/usr/bin/env python3
"""
Offline multimodal ingestion — two-phase architecture.

Phase 1 (fitz-only subprocess):
  Extract text chunks and figure paths from a PDF, write to a temp JSON file.

Phase 2 (torch-only subprocess):
  Load the JSON, embed everything with CLIP, build the UnifiedFAISSIndex, save.

This two-process split avoids a crash caused by PyMuPDF's MuPDF C library and
PyTorch's Accelerate/Metal bindings conflicting when both are loaded in the same
process on macOS Apple Silicon.

Usage (from project root):
  bash scripts/run_ingest.sh
  bash scripts/run_ingest.sh --pdf data/chapters/silberschatz.pdf
  bash scripts/run_ingest.sh --pdf path/to/book.pdf --index-dir data/multimodal
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

PYTHON = sys.executable
PROJECT_ROOT = str(Path(__file__).parent.parent)


# ---------------------------------------------------------------------------
# Phase 1: PDF text + image extraction (fitz, no torch)
# ---------------------------------------------------------------------------

PHASE1_CODE = """
import sys, json, re
from pathlib import Path
sys.path.insert(0, {root!r})

import fitz
from src.pdf_image_extractor import extract_images_from_pdf

CHUNK_SIZE    = 350
CHUNK_OVERLAP = 50

def simple_chunks(text):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

pdf_path      = {pdf!r}
image_dir     = {image_dir!r}
out_json      = {out_json!r}
min_w, min_h  = {min_w}, {min_h}

# --- text ---
doc = fitz.open(pdf_path)
text_records = []
for page_num in range(len(doc)):
    raw = re.sub(r'\\s+', ' ', doc[page_num].get_text('text').strip())
    for chunk in simple_chunks(raw):
        text_records.append({{
            "modality": "text",
            "source": pdf_path,
            "page_number": page_num + 1,
            "text": chunk,
            "image_path": None,
        }})
doc.close()

# --- images ---
images = extract_images_from_pdf(
    pdf_path=pdf_path,
    output_dir=image_dir,
    min_width=min_w,
    min_height=min_h,
)
image_records = [{{
    "modality": "image",
    "source": img.source_pdf,
    "page_number": img.page_number,
    "text": None,
    "image_path": img.image_path,
    "extra": {{"width": img.width, "height": img.height, "image_id": img.image_id}},
}} for img in images]

records = text_records + image_records
with open(out_json, "w") as f:
    json.dump(records, f)

print(f"Phase1: {{len(text_records)}} text chunks, {{len(image_records)}} images -> {{out_json}}", flush=True)
"""

# ---------------------------------------------------------------------------
# Phase 2: CLIP embedding + FAISS index (torch, no fitz)
# ---------------------------------------------------------------------------

PHASE2_CODE = """
import sys, json
from pathlib import Path
sys.path.insert(0, {root!r})

import numpy as np
from src.clip_embedder import embed_texts, embed_images
from src.multimodal.unified_index import UnifiedFAISSIndex, ModalityEntry

in_json    = {in_json!r}
index_dir  = {index_dir!r}
text_batch = {text_batch}
img_batch  = {img_batch}

with open(in_json) as f:
    records = json.load(f)

text_recs  = [r for r in records if r["modality"] == "text"]
image_recs = [r for r in records if r["modality"] == "image"]
print(f"Phase2: embedding {{len(text_recs)}} text chunks, {{len(image_recs)}} images", flush=True)

index = UnifiedFAISSIndex()

# -- text --
STRIDE = 500
for i in range(0, len(text_recs), STRIDE):
    batch = text_recs[i:i+STRIDE]
    texts  = [r["text"] for r in batch]
    embs   = embed_texts(texts, batch_size=text_batch)
    entries = [
        ModalityEntry(idx=0, modality="text", source=r["source"],
                      page_number=r["page_number"], text=r["text"], image_path=None)
        for r in batch
    ]
    index.add_vectors(embs, entries)
    print(f"  text {{i+1}}-{{i+len(batch)}} embedded", flush=True)

# -- images --
if image_recs:
    img_paths = [r["image_path"] for r in image_recs]
    embs = embed_images(img_paths, batch_size=img_batch)
    entries = [
        ModalityEntry(idx=0, modality="image", source=r["source"],
                      page_number=r["page_number"], text=None,
                      image_path=r["image_path"],
                      extra=r.get("extra", {{}}))
        for r in image_recs
    ]
    index.add_vectors(embs, entries)
    print(f"  {{len(image_recs)}} images embedded", flush=True)

Path(index_dir).mkdir(parents=True, exist_ok=True)
index.save(Path(index_dir) / "unified_index")
print(f"Index saved to {{index_dir}}  stats={{index.stats()}}", flush=True)
"""


def run_phase(code: str, label: str) -> None:
    result = subprocess.run(
        [PYTHON, "-c", code],
        check=False,
    )
    if result.returncode != 0:
        print(f"ERROR: {label} failed (exit {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Build multimodal CLIP index from a PDF")
    parser.add_argument("--pdf",          default="data/chapters/silberschatz.pdf")
    parser.add_argument("--index-dir",    default="data/multimodal")
    parser.add_argument("--image-dir",    default="data/figures")
    parser.add_argument("--min-image-width",  type=int, default=100)
    parser.add_argument("--min-image-height", type=int, default=100)
    parser.add_argument("--text-batch",   type=int, default=64)
    parser.add_argument("--image-batch",  type=int, default=16)
    args = parser.parse_args()

    pdf = Path(args.pdf)
    if not pdf.exists():
        print(f"ERROR: PDF not found: {pdf}", file=sys.stderr)
        sys.exit(1)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        tmp_json = tf.name

    print(f"=== Phase 1: extracting text and images from {pdf.name} ===")
    run_phase(
        PHASE1_CODE.format(
            root=PROJECT_ROOT,
            pdf=str(pdf),
            image_dir=args.image_dir,
            out_json=tmp_json,
            min_w=args.min_image_width,
            min_h=args.min_image_height,
        ),
        "Phase 1 (PDF extraction)"
    )

    print(f"\n=== Phase 2: embedding with CLIP and building index ===")
    run_phase(
        PHASE2_CODE.format(
            root=PROJECT_ROOT,
            in_json=tmp_json,
            index_dir=args.index_dir,
            text_batch=args.text_batch,
            img_batch=args.image_batch,
        ),
        "Phase 2 (CLIP embedding)"
    )

    Path(tmp_json).unlink(missing_ok=True)
    print(f"\nIngestion complete. Index at: {args.index_dir}/")


if __name__ == "__main__":
    main()
