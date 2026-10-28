
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

"""Convert MVTec AD pixel masks -> YOLO bounding-box labels
and build the train/val dataset split.

"""

import os
import shutil
import random
import cv2
import numpy as np
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent
DATA_ROOT    = BASE_DIR / "metal_nut"
DATASET_DIR  = BASE_DIR / "dataset"
RANDOM_SEED  = 42
VAL_RATIO    = 0.30          # 30 % of defective images go to val
MIN_AREA_PX  = 200           # ignore contours smaller than this (noise filter)

# Class mapping — must match order used in data.yaml
CLASS_NAMES  = ["bent", "color", "flip", "scratch"]
CLASS_MAP    = {name: idx for idx, name in enumerate(CLASS_NAMES)}
# ──────────────────────────────────────────────────────────────────────────────


def mask_to_yolo_boxes(mask_path: Path, img_w: int, img_h: int) -> list[str]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"  [WARN] Could not read mask: {mask_path}")
        return []

    # Binarise — MVTec masks are 0/255 but threshold makes it robust
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < MIN_AREA_PX:
            continue  # skip tiny noise blobs

        # Normalise to [0, 1]
        x_c = (x + w / 2) / img_w
        y_c = (y + h / 2) / img_h
        w_n = w / img_w
        h_n = h / img_h

        # Clamp to valid range (safety)
        x_c = max(0.0, min(1.0, x_c))
        y_c = max(0.0, min(1.0, y_c))
        w_n = max(0.0, min(1.0, w_n))
        h_n = max(0.0, min(1.0, h_n))

        boxes.append((x_c, y_c, w_n, h_n))

    return boxes


def write_label_file(label_path: Path, class_id: int, boxes: list) -> None:
    """Write YOLO label file. One line per box: class x_c y_c w h"""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as f:
        for (x_c, y_c, w_n, h_n) in boxes:
            f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")


def copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def write_empty_label(label_path: Path) -> None:
    """Empty label file = negative sample (no defects)."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.touch()


def make_data_yaml() -> None:
    yaml_path = DATASET_DIR / "data.yaml"
    content = f"""# YOLOv8 dataset config — Metal Nut Defect Detection
path: {DATASET_DIR.as_posix()}
train: images/train
val: images/val

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    yaml_path.write_text(content)
    print(f"  Wrote {yaml_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    random.seed(RANDOM_SEED)

    # Verify paths
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_ROOT}")

    # Clean & recreate dataset dir
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    DATASET_DIR.mkdir(parents=True)

    # ── Step 1: Process defective images (test/ + ground_truth/) ────────────
    print("\n=== Step 1: Converting masks → YOLO labels ===")

    # Stats tracker
    stats = {cls: {"train": 0, "val": 0, "boxes": 0} for cls in CLASS_NAMES}

    for defect_type in CLASS_NAMES:
        class_id  = CLASS_MAP[defect_type]
        img_dir   = DATA_ROOT / "test"  / defect_type
        mask_dir  = DATA_ROOT / "ground_truth" / defect_type

        img_files = sorted(img_dir.glob("*.png"))
        if not img_files:
            print(f"  [WARN] No images found for class: {defect_type}")
            continue

        # Stratified 70/30 split
        random.shuffle(img_files)
        n_val   = max(1, round(len(img_files) * VAL_RATIO))
        val_set = set(f.name for f in img_files[:n_val])

        print(f"\n  [{defect_type}]  total={len(img_files)}  "
              f"train={len(img_files)-n_val}  val={n_val}")

        for img_path in img_files:
            stem      = img_path.stem                      # e.g. "000"
            mask_path = mask_dir / f"{stem}_mask.png"
            split     = "val" if img_path.name in val_set else "train"

            # Read image to get dimensions
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"    [WARN] Cannot read image: {img_path}")
                continue
            h, w = img.shape[:2]

            # Convert mask → boxes
            boxes = mask_to_yolo_boxes(mask_path, w, h)
            if not boxes:
                print(f"    [WARN] No valid boxes for {img_path.name} — skipping")
                continue

            # Unique filename: <defect_type>_<stem>.png
            out_name   = f"{defect_type}_{stem}"
            img_dst    = DATASET_DIR / "images" / split / f"{out_name}.png"
            label_dst  = DATASET_DIR / "labels" / split / f"{out_name}.txt"

            copy_image(img_path, img_dst)
            write_label_file(label_dst, class_id, boxes)

            stats[defect_type][split] += 1
            stats[defect_type]["boxes"] += len(boxes)

    # ── Step 2: Add good images as negatives ────────────────────────────────
    print("\n=== Step 2: Adding negative samples (good images) ===")

    good_train = sorted((DATA_ROOT / "train" / "good").glob("*.png"))
    good_test  = sorted((DATA_ROOT / "test"  / "good").glob("*.png"))

    # Use 110 good/train images as training negatives
    neg_train = random.sample(good_train, min(110, len(good_train)))
    # Use all good/test images as val negatives
    neg_val   = good_test

    for img_path in neg_train:
        out_name  = f"good_{img_path.stem}"
        img_dst   = DATASET_DIR / "images" / "train" / f"{out_name}.png"
        label_dst = DATASET_DIR / "labels" / "train" / f"{out_name}.txt"
        copy_image(img_path, img_dst)
        write_empty_label(label_dst)

    for img_path in neg_val:
        out_name  = f"good_{img_path.stem}"
        img_dst   = DATASET_DIR / "images" / "val" / f"{out_name}.png"
        label_dst = DATASET_DIR / "labels" / "val" / f"{out_name}.txt"
        copy_image(img_path, img_dst)
        write_empty_label(label_dst)

    print(f"  Negatives added — train: {len(neg_train)}, val: {len(neg_val)}")

    # ── Step 3: Write data.yaml ──────────────────────────────────────────────
    print("\n=== Step 3: Writing data.yaml ===")
    make_data_yaml()

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n=== Summary ===")
    print(f"{'Class':<10} {'Train':>6} {'Val':>5} {'Boxes':>7}")
    print("-" * 30)
    total_train = total_val = total_boxes = 0
    for cls in CLASS_NAMES:
        s = stats[cls]
        print(f"{cls:<10} {s['train']:>6} {s['val']:>5} {s['boxes']:>7}")
        total_train += s["train"]
        total_val   += s["val"]
        total_boxes += s["boxes"]
    print("-" * 30)
    print(f"{'Defective':<10} {total_train:>6} {total_val:>5} {total_boxes:>7}")
    print(f"{'Negatives':<10} {len(neg_train):>6} {len(neg_val):>5}")
    print(f"{'TOTAL':<10} {total_train+len(neg_train):>6} {total_val+len(neg_val):>5}")
    print(f"\nDataset ready at: {DATASET_DIR}")


if __name__ == "__main__":
    main()
