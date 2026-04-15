"""
PDF image extractor for multi-modal ingestion.

Uses PyMuPDF to extract embedded images from PDF pages, filtering out
decorative elements (page borders, small icons) by minimum dimensions.
Each extracted image is saved as a PNG and returned with its source
metadata (PDF path, page number, image index on page).
"""

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Union

import fitz  # PyMuPDF


@dataclass
class ExtractedImage:
    image_path: str        # absolute path to saved PNG
    source_pdf: str        # originating PDF file
    page_number: int       # 1-indexed page
    image_index: int       # position on page (0-indexed)
    width: int
    height: int
    image_id: str          # stable MD5 hash of pixel content


def extract_images_from_pdf(
    pdf_path: Union[str, Path] = None,
    output_dir: Union[str, Path] = None,
    min_width: int = 100,
    min_height: int = 100,
) -> List[ExtractedImage]:
    """
    Extract all embedded images from a PDF that meet the minimum size filter.

    Args:
        pdf_path:   Path to the source PDF.
        output_dir: Directory to save extracted PNGs.
                    Created if it does not exist.
        min_width:  Minimum pixel width to keep (filters icons/borders).
        min_height: Minimum pixel height to keep.

    Returns:
        List of ExtractedImage metadata objects.
    """

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    extracted: List[ExtractedImage] = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)

        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
            except Exception:
                continue

            width = base_image["width"]
            height = base_image["height"]

            if width < min_width or height < min_height:
                continue

            image_bytes = base_image["image"]
            image_ext = base_image.get("ext", "png")

            image_id = hashlib.md5(image_bytes).hexdigest()[:12]
            filename = f"p{page_num + 1:04d}_i{img_idx:02d}_{image_id}.png"
            image_path = output_dir / filename

            if not image_path.exists():
                # Convert to PNG for uniform downstream handling
                try:
                    import PIL.Image
                    import io
                    pil_img = PIL.Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    pil_img.save(str(image_path), "PNG")
                except Exception:
                    # Fall back to raw bytes if PIL conversion fails
                    image_path.write_bytes(image_bytes)

            extracted.append(
                ExtractedImage(
                    image_path=str(image_path),
                    source_pdf=str(pdf_path),
                    page_number=page_num + 1,
                    image_index=img_idx,
                    width=width,
                    height=height,
                    image_id=image_id,
                )
            )

    doc.close()
    return extracted


def save_manifest(images: List[ExtractedImage], manifest_path: Union[str, Path]):
    """Save extraction results to a JSON manifest for downstream use."""
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump([asdict(img) for img in images], f, indent=2)


def load_manifest(manifest_path: Union[str, Path]) -> List[ExtractedImage]:
    """Load a previously saved manifest."""
    with open(manifest_path) as f:
        return [ExtractedImage(**item) for item in json.load(f)]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract images from a PDF")
    parser.add_argument("pdf", help="Path to input PDF")
    parser.add_argument("--output_dir", default="data/figures", help="Output directory for PNGs")
    parser.add_argument("--min_width", type=int, default=100)
    parser.add_argument("--min_height", type=int, default=100)
    args = parser.parse_args()

    images = extract_images_from_pdf(
        pdf_path=args.pdf,
        output_dir=args.output_dir,
        min_width=args.min_width,
        min_height=args.min_height,
    )

    manifest_path = Path(args.output_dir) / "manifest.json"
    save_manifest(images, manifest_path)

    print(f"Extracted {len(images)} images -> {args.output_dir}")
    print(f"Manifest saved to {manifest_path}")
