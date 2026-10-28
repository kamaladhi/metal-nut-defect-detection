import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

"""
Draws YOLO bounding boxes on images so you can visually verify
that mask -> bbox conversion worked correctly.
"""

import cv2
import numpy as np
import random
from pathlib import Path

BASE_DIR     = Path(__file__).resolve().parent
DATASET_DIR  = BASE_DIR / "dataset"
OUTPUT_DIR   = BASE_DIR / "results" / "label_check"
CLASS_NAMES  = ["bent", "color", "flip", "scratch"]
SAMPLES_PER_CLASS = 4   # how many images to show per defect class

# One distinct color per class (BGR)
COLORS = {
    "bent":    (0,   200, 255),   # orange
    "color":   (0,   255, 100),   # green
    "flip":    (255, 100, 0  ),   # blue
    "scratch": (100, 0,   255),   # purple
}
# ──────────────────────────────────────────────────────────────────────────────


def draw_boxes(img: np.ndarray, label_path: Path) -> np.ndarray:
    img = img.copy()
    h, w = img.shape[:2]

    if not label_path.exists() or label_path.stat().st_size == 0:
        cv2.putText(img, "GOOD (no defect)", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 205, 50), 2)
        return img

    with open(label_path) as f:
        lines = f.read().strip().splitlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id, x_c, y_c, bw, bh = int(parts[0]), *map(float, parts[1:])
        cls_name = CLASS_NAMES[cls_id]
        color    = COLORS[cls_name]

        # Convert normalised -> pixel coords
        x1 = int((x_c - bw / 2) * w)
        y1 = int((y_c - bh / 2) * h)
        x2 = int((x_c + bw / 2) * w)
        y2 = int((y_c + bh / 2) * h)

        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

        # Draw label background + text
        label_txt = cls_name
        (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img, label_txt, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return img


def make_grid(images: list[np.ndarray], cols: int = 4) -> np.ndarray:
    """Stack annotated images into a single grid image."""
    # Resize all to same size
    size = 350
    resized = [cv2.resize(im, (size, size)) for im in images]

    rows = []
    for i in range(0, len(resized), cols):
        row_imgs = resized[i:i+cols]
        # Pad last row if needed
        while len(row_imgs) < cols:
            row_imgs.append(np.zeros((size, size, 3), dtype=np.uint8))
        rows.append(np.hstack(row_imgs))
    return np.vstack(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(42)

    for split in ["train", "val"]:
        img_dir   = DATASET_DIR / "images" / split
        label_dir = DATASET_DIR / "labels" / split

        # ── Per-class grid ───────────────────────────────────────────────────
        for cls_name in CLASS_NAMES:
            cls_imgs = sorted(img_dir.glob(f"{cls_name}_*.png"))
            if not cls_imgs:
                continue

            samples = random.sample(cls_imgs, min(SAMPLES_PER_CLASS, len(cls_imgs)))
            annotated = []
            for img_path in samples:
                img = cv2.imread(str(img_path))
                lbl = label_dir / (img_path.stem + ".txt")
                annotated.append(draw_boxes(img, lbl))

            grid = make_grid(annotated, cols=SAMPLES_PER_CLASS)
            out  = OUTPUT_DIR / f"{split}_{cls_name}.png"
            cv2.imwrite(str(out), grid)
            print(f"  Saved: {out.name}  ({len(samples)} samples)")

        # ── Negatives sample ─────────────────────────────────────────────────
        good_imgs = sorted(img_dir.glob("good_*.png"))
        if good_imgs:
            samples   = random.sample(good_imgs, min(4, len(good_imgs)))
            annotated = []
            for img_path in samples:
                img = cv2.imread(str(img_path))
                lbl = label_dir / (img_path.stem + ".txt")
                annotated.append(draw_boxes(img, lbl))
            grid = make_grid(annotated, cols=4)
            out  = OUTPUT_DIR / f"{split}_good_negatives.png"
            cv2.imwrite(str(out), grid)
            print(f"  Saved: {out.name}  ({len(samples)} samples)")

    print(f"\nAll label check images saved to: {OUTPUT_DIR}")
    print("Open the folder and verify boxes look correct before training.")


if __name__ == "__main__":
    main()
