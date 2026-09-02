"""
convert_to_yolo.py

Purpose: Convert Pascal VOC XML annotations (NEU-DET format) into YOLO-format
label .txt files, and reorganize images + labels into the flat structure
Ultralytics YOLO expects:

    data/processed/
        images/train/*.jpg
        images/val/*.jpg
        labels/train/*.txt      (same basename as matching image)
        labels/val/*.txt
        data.yaml               (tells YOLO where data is + class names)

Run: python src/data/convert_to_yolo.py
"""

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")

# Fixed order -> class_id mapping. This MUST stay identical between
# training and inference, or the model's predicted numbers will point
# to the wrong class names later.
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}


def convert_box(img_w: int, img_h: int, xmin: float, ymin: float, xmax: float, ymax: float):
    """
    Pascal VOC box (pixel corners) -> YOLO box (normalized center/size).

    Why normalize? YOLO works across any input resolution the model is
    trained/run at. A box defined as "20% from the left, 35% wide" is
    resolution-independent; raw pixel coordinates are not.
    """
    x_center = (xmin + xmax) / 2.0 / img_w
    y_center = (ymin + ymax) / 2.0 / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h
    return x_center, y_center, width, height


def find_image_file(images_dir: Path, filename: str) -> Optional[Path]:
    """
    Images live in per-class subfolders (images/crazing/, images/patches/...),
    so we search recursively for the filename the XML references.

    Some annotation XMLs (a known quirk of this dataset, affects some
    'patches' entries) omit the file extension entirely - e.g. filename is
    'patches_104' instead of 'patches_104.jpg'. We try the name as-is first,
    then fall back to common image extensions.
    """
    has_extension = any(filename.lower().endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".bmp"))
    candidates = [filename] if has_extension else [filename + ext for ext in (".jpg", ".JPG", ".png", ".bmp")]

    for candidate in candidates:
        matches = list(images_dir.rglob(candidate))
        if matches:
            return matches[0]
    return None


def process_split(split: str) -> None:
    ann_dir = RAW_DIR / split / "annotations"
    img_dir = RAW_DIR / split / "images"

    # NEU-DET calls the second split "validation"; YOLO convention is "val"
    out_split = "val" if split == "validation" else "train"
    out_img_dir = OUT_DIR / "images" / out_split
    out_lbl_dir = OUT_DIR / "labels" / out_split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    n_converted, n_missing = 0, 0

    for xml_file in ann_dir.glob("*.xml"):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        filename = root.find("filename").text
        size_tag = root.find("size")
        img_w = int(size_tag.find("width").text)
        img_h = int(size_tag.find("height").text)

        src_img = find_image_file(img_dir, filename)
        if src_img is None:
            n_missing += 1
            if n_missing <= 10:  # print only the first 10, avoid flooding the console
                print(f"  MISSING: xml={xml_file.name}  filename_in_xml='{filename}'")
            continue

        lines = []
        for obj in root.findall("object"):
            class_id = CLASS_TO_ID[obj.find("name").text]
            bnd = obj.find("bndbox")
            xmin = float(bnd.find("xmin").text)
            ymin = float(bnd.find("ymin").text)
            xmax = float(bnd.find("xmax").text)
            ymax = float(bnd.find("ymax").text)

            xc, yc, w, h = convert_box(img_w, img_h, xmin, ymin, xmax, ymax)
            lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        # Use the ACTUAL found image's name/stem, not the raw XML filename -
        # the XML value is sometimes extension-less (see find_image_file),
        # and using it directly here would silently write files with no
        # extension, breaking YOLO's image<->label matching.
        label_path = out_lbl_dir / (src_img.stem + ".txt")
        label_path.write_text("\n".join(lines))

        shutil.copy(src_img, out_img_dir / src_img.name)
        n_converted += 1

    print(f"{split}: converted {n_converted} images, {n_missing} missing")


if __name__ == "__main__":
    process_split("train")
    process_split("validation")

    # data.yaml tells YOLO: where the images live, and what each class_id means.
    yaml_lines = [f"path: {OUT_DIR.resolve()}", "train: images/train", "val: images/val", "", "names:"]
    for i, name in enumerate(CLASSES):
        yaml_lines.append(f"  {i}: {name}")

    (OUT_DIR / "data.yaml").write_text("\n".join(yaml_lines))
    print(f"\ndata.yaml written to {OUT_DIR / 'data.yaml'}")