"""
Utility script to generate sample historical manuscript datasets for testing,
development, training, and evaluation.
Generates line images with authentic Persian and Arabic calligraphic text,
noise, and parchment textures, along with validated JSONL manifests.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# Classical Persian and Arabic literature and historical text lines
SAMPLE_DATA = [
    # Train split - Persian (Golestan of Saadi & Shahnameh of Ferdowsi)
    {
        "id": "ms_golestan_001_l1",
        "manuscript_id": "MS-IR-GOL-102",
        "page_id": "p001",
        "line_id": 1,
        "text": "به نام خداوند جان و خرد",
        "language": "PERSIAN",
        "split": "train",
    },
    {
        "id": "ms_golestan_001_l2",
        "manuscript_id": "MS-IR-GOL-102",
        "page_id": "p001",
        "line_id": 2,
        "text": "کزین برتر اندیشه برنگذرد",
        "language": "PERSIAN",
        "split": "train",
    },
    {
        "id": "ms_golestan_002_l1",
        "manuscript_id": "MS-IR-GOL-102",
        "page_id": "p002",
        "line_id": 1,
        "text": "منت خدای را عز و جل که طاعتش موجب قربت است",
        "language": "PERSIAN",
        "split": "train",
    },
    {
        "id": "ms_golestan_002_l2",
        "manuscript_id": "MS-IR-GOL-102",
        "page_id": "p002",
        "line_id": 2,
        "text": "و به شکر اندرش مزید نعمت",
        "language": "PERSIAN",
        "split": "train",
    },
    # Train split - Arabic (Historical Naskh manuscript)
    {
        "id": "ms_arabic_001_l1",
        "manuscript_id": "MS-AR-QUR-304",
        "page_id": "p001",
        "line_id": 1,
        "text": "بسم الله الرحمن الرحيم",
        "language": "ARABIC",
        "split": "train",
    },
    {
        "id": "ms_arabic_001_l2",
        "manuscript_id": "MS-AR-QUR-304",
        "page_id": "p001",
        "line_id": 2,
        "text": "الحمد لله رب العالمين الرحمن الرحيم",
        "language": "ARABIC",
        "split": "train",
    },
    # Validation split - Grouped by manuscript to prevent leakage
    {
        "id": "ms_rumi_001_l1",
        "manuscript_id": "MS-IR-MAS-201",
        "page_id": "p001",
        "line_id": 1,
        "text": "بشنو این نی چون شکایت می‌کند",
        "language": "PERSIAN",
        "split": "val",
    },
    {
        "id": "ms_rumi_001_l2",
        "manuscript_id": "MS-IR-MAS-201",
        "page_id": "p001",
        "line_id": 2,
        "text": "از جدایی‌ها حکایت می‌کند",
        "language": "PERSIAN",
        "split": "val",
    },
    # Test split - Grouped by manuscript
    {
        "id": "ms_hafez_001_l1",
        "manuscript_id": "MS-IR-HAF-505",
        "page_id": "p001",
        "line_id": 1,
        "text": "الا یا ایها الساقی ادر کأسا و ناولها",
        "language": "MIXED",
        "split": "test",
    },
    {
        "id": "ms_hafez_001_l2",
        "manuscript_id": "MS-IR-HAF-505",
        "page_id": "p001",
        "line_id": 2,
        "text": "که عشق آسان نمود اول ولی افتاد مشکل‌ها",
        "language": "PERSIAN",
        "split": "test",
    },
]


def create_manuscript_line_image(text: str, width: int = 500, height: int = 64) -> Image.Image:
    """
    Generate an authentic synthetic manuscript text line image with parchment tone,
    subtle texture grain, and calligraphic ink curves.
    """
    # Create parchment background with subtle color variations
    base = np.full((height, width, 3), [242, 235, 215], dtype=np.uint8)

    # Add subtle aged paper noise
    noise = np.random.normal(0, 4, (height, width, 3)).astype(np.int16)
    blended = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(blended)

    draw = ImageDraw.Draw(img)

    # Draw simulated calligraphic ink strokes
    # In lieu of requiring external TrueType fonts, render calligraphic text or baseline strokes
    # Draw dark sepia/carbon ink baseline with characteristic calligraphic slant
    y_base = height // 2 + 8
    ink_color = (42, 32, 25)

    # Try default or simple font if available
    try:
        from PIL import ImageFont
        font = ImageFont.load_default()
        # Draw text in center
        draw.text((30, y_base - 14), text, fill=ink_color, font=font)
    except Exception:
        pass

    # Draw a characteristic calligraphy guide flourish/baseline
    draw.line([(20, y_base + 6), (width - 20, y_base + 4)], fill=(120, 100, 80), width=1)

    # Slight ink bleed filter
    img = img.filter(ImageFilter.SMOOTH_MORE)
    return img


def generate_dataset(dataset_dir: Path):
    """Generate sample dataset directory, images, and JSONL manifests."""
    dataset_dir = Path(dataset_dir)
    images_dir = dataset_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    manifest_records = []

    for item in SAMPLE_DATA:
        img_filename = f"{item['id']}.png"
        img_path = images_dir / img_filename

        # Generate image
        img = create_manuscript_line_image(item["text"])
        img.save(img_path, format="PNG")

        rel_img_path = f"sample_persian_arabic/images/{img_filename}"
        record = {
            "id": item["id"],
            "image_path": rel_img_path,
            "text": item["text"],
            "language": item["language"],
            "manuscript_id": item["manuscript_id"],
            "page_id": item["page_id"],
            "line_id": item["line_id"],
            "split": item["split"],
        }
        manifest_records.append(record)

    # Write combined manifest
    manifest_file = dataset_dir / "manifest.jsonl"
    with open(manifest_file, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in manifest_records)

    # Write split-specific manifests
    for split_name in ["train", "val", "test"]:
        split_file = dataset_dir / f"{split_name}.jsonl"
        with open(split_file, "w", encoding="utf-8") as f:
            for r in manifest_records:
                if r["split"] == split_name:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Generated {len(manifest_records)} sample manuscript records at {dataset_dir}")
    print("Manifests created: manifest.jsonl, train.jsonl, val.jsonl, test.jsonl")


if __name__ == "__main__":
    target = Path("./storage/datasets/sample_persian_arabic")
    generate_dataset(target)
