"""
CLIP-based embedder for multi-modal retrieval.

Projects both text and images into a shared 768-dim embedding space
using openai/clip-vit-large-patch14 via HuggingFace Transformers.
Inner product on L2-normalized vectors equals cosine similarity,
making the output directly compatible with a FAISS IndexFlatIP index.
"""

import numpy as np
import torch
from PIL import Image
from pathlib import Path
from typing import List, Union
from transformers import CLIPProcessor, CLIPModel


_MODEL_ID = "openai/clip-vit-large-patch14"
_EMBEDDING_DIM = 768

_model: CLIPModel = None
_processor: CLIPProcessor = None


def _load():
    global _model, _processor
    if _model is None:
        _model = CLIPModel.from_pretrained(_MODEL_ID)
        _processor = CLIPProcessor.from_pretrained(_MODEL_ID)
        _model.eval()


def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return x / norms


def embed_texts(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Encode a list of strings with CLIP's text encoder.

    Returns an (N, 768) float32 array of L2-normalized embeddings.
    CLIP's text encoder has a hard 77-token context window; longer
    inputs are silently truncated by the processor.
    """
    _load()
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = _processor(
            text=batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        with torch.no_grad():
            out = _model.get_text_features(**inputs)
            emb = out.pooler_output if hasattr(out, "pooler_output") else out
        all_embeddings.append(emb.cpu().numpy())

    result = np.vstack(all_embeddings).astype(np.float32)
    return _normalize(result)


def embed_images(
    images: List[Union[str, Path, Image.Image]], batch_size: int = 16
) -> np.ndarray:
    """
    Encode a list of images with CLIP's image encoder.

    Each image can be a file path (str/Path) or a PIL Image.
    Returns an (N, 768) float32 array of L2-normalized embeddings.
    """
    _load()
    all_embeddings = []

    pil_images = []
    for img in images:
        if isinstance(img, (str, Path)):
            pil_images.append(Image.open(img).convert("RGB"))
        else:
            pil_images.append(img.convert("RGB"))

    for i in range(0, len(pil_images), batch_size):
        batch = pil_images[i : i + batch_size]
        inputs = _processor(images=batch, return_tensors="pt")
        with torch.no_grad():
            out = _model.get_image_features(**inputs)
            emb = out.pooler_output if hasattr(out, "pooler_output") else out
        all_embeddings.append(emb.cpu().numpy())

    result = np.vstack(all_embeddings).astype(np.float32)
    return _normalize(result)


def caption_image(
    image: Union[str, Path, Image.Image], candidates: List[str]
) -> str:
    """
    Zero-shot caption: return the candidate string most similar to the image.

    Useful for injecting image context into text-only LLM prompts.
    """
    img_emb = embed_images([image])           # (1, 768)
    txt_emb = embed_texts(candidates)         # (K, 768)
    scores = (img_emb @ txt_emb.T).squeeze()  # (K,)
    return candidates[int(np.argmax(scores))]


@property
def embedding_dim() -> int:
    return _EMBEDDING_DIM
