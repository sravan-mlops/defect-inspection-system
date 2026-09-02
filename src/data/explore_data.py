"""
explore_data.py

Purpose: Before we train anything, understand what we're working with.
Assumes the NEU-DET dataset layout after download:

    data/raw/
        train/
            images/*.jpg
            annotations/*.xml      (Pascal VOC format)
        validation/
            images/*.jpg
            annotations/*.xml

Run: python src/data/explore_data.py
"""

import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from PIL import Image

RAW_DIR = Path("data/raw")


def count_classes_in_split(split: str) -> Counter:
    """
    Parse every XML annotation in a split (train/validation) and count
    how many bounding boxes exist per defect class.

    WHY a Counter and not just len(images): one image can contain
    MULTIPLE defects of different classes, so counting images alone
    would hide class imbalance at the object level, which is what
    actually matters for a detector.
    """
    ann_dir = RAW_DIR / split / "annotations"
    class_counts = Counter()

    for xml_file in ann_dir.glob("*.xml"):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        # Pascal VOC format: each <object> tag is one labeled defect instance
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            class_counts[class_name] += 1

    return class_counts


def check_image_health(split: str) -> None:
    """
    Open every image to confirm it's not corrupt, and record size
    consistency. Catching a corrupt file NOW (seconds) is much cheaper
    than discovering it mid-training (could be hours in).
    """
    img_dir = RAW_DIR / split / "images"
    sizes = set()
    corrupt = []

    for img_path in img_dir.rglob("*.jpg"):
        try:
            with Image.open(img_path) as img:
                sizes.add(img.size)
        except Exception as e:
            corrupt.append((img_path.name, str(e)))

    print(f"  Unique image sizes found: {sizes}")
    if corrupt:
        print(f"  WARNING: {len(corrupt)} corrupt files: {corrupt}")
    else:
        print("  No corrupt images found.")


if __name__ == "__main__":
    for split in ["train", "validation"]:
        print(f"\n=== {split.upper()} ===")

        counts = count_classes_in_split(split)
        total = sum(counts.values())
        print(f"Total defect instances: {total}")
        for cls, n in sorted(counts.items(), key=lambda x: -x[1]):
            pct = 100 * n / total
            print(f"  {cls:20s}: {n:4d}  ({pct:.1f}%)")

        check_image_health(split)
