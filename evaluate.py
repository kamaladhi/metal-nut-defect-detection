import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np
import json
from pathlib import Path
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent
WEIGHTS      = BASE_DIR / "runs" / "detect" / "metal_nut_yolov8n" / "weights" / "best.pt"
if not WEIGHTS.exists() and (BASE_DIR / "best.pt").exists():
    WEIGHTS  = BASE_DIR / "best.pt"
DATASET_YAML = BASE_DIR / "dataset" / "data.yaml"
VAL_IMG_DIR  = BASE_DIR / "dataset" / "images" / "val"
VAL_LBL_DIR  = BASE_DIR / "dataset" / "labels" / "val"
OUTPUT_DIR   = BASE_DIR / "results" / "evaluation"
FAILURE_DIR  = OUTPUT_DIR / "failure_cases"

CLASS_NAMES  = ["bent", "color", "flip", "scratch"]
CONF_THRESH  = 0.25    # confidence threshold for predictions
IOU_THRESH   = 0.5     # IoU threshold for matching pred to GT

# Colors per class (BGR)
COLORS = {
    "bent":    (0,   200, 255),
    "color":   (0,   255, 100),
    "flip":    (255, 100, 0  ),
    "scratch": (100, 0,   255),
    "good":    (50,  205, 50 ),
}
# ──────────────────────────────────────────────────────────────────────────────


def load_gt_boxes(label_path: Path, img_w: int, img_h: int):
    """Load YOLO ground-truth boxes -> list of (class_id, x1, y1, x2, y2)."""
    if not label_path.exists() or label_path.stat().st_size == 0:
        return []
    boxes = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, xc, yc, w, h = int(parts[0]), *map(float, parts[1:])
        x1 = int((xc - w / 2) * img_w)
        y1 = int((yc - h / 2) * img_h)
        x2 = int((xc + w / 2) * img_w)
        y2 = int((yc + h / 2) * img_h)
        boxes.append((cls, x1, y1, x2, y2))
    return boxes


def iou(boxA, boxB):
    """Compute IoU between two boxes (x1,y1,x2,y2)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    aA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    aB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / float(aA + aB - inter)


def draw_box(img, x1, y1, x2, y2, label, color, thickness=2):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(img, label, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return img


def add_legend(img):
    """Add GT=green / PRED=red legend to image."""
    cv2.putText(img, "GT (ground truth)", (10, img.shape[0] - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(img, "PRED (prediction)", (10, img.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return img


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FAILURE_DIR.mkdir(parents=True, exist_ok=True)

    if not WEIGHTS.exists():
        raise FileNotFoundError(
            f"Weights not found: {WEIGHTS}\nRun train.py first."
        )

    print("=" * 60)
    print("  Metal Nut Defect Detection -- Evaluation")
    print("=" * 60)
    print(f"Weights   : {WEIGHTS}")
    print(f"Val set   : {VAL_IMG_DIR}")
    print()

    model = YOLO(str(WEIGHTS))

    # ── Official YOLO validation (for metrics) ───────────────────────────────
    print("--- Running official YOLO validation ---")
    val_results = model.val(
        data=str(DATASET_YAML),
        split="val",
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        plots=True,
        save_json=False,
        project=str(OUTPUT_DIR),
        name="yolo_val",
        exist_ok=True,
    )

    # Extract metrics
    mp  = val_results.box.mp          # mean precision
    mr  = val_results.box.mr          # mean recall
    map50    = val_results.box.map50  # mAP@0.5
    map5095  = val_results.box.map    # mAP@0.5:0.95
    per_class_ap50 = val_results.box.ap50   # per-class AP@0.5

    print("\n--- Metrics Summary ---")
    print(f"{'Class':<12} {'AP@0.5':>8}")
    print("-" * 22)
    for i, cls in enumerate(CLASS_NAMES):
        ap = per_class_ap50[i] if i < len(per_class_ap50) else float("nan")
        print(f"{cls:<12} {ap:>8.4f}")
    print("-" * 22)
    print(f"{'mAP@0.5':<12} {map50:>8.4f}")
    print(f"{'Precision':<12} {mp:>8.4f}")
    print(f"{'Recall':<12} {mr:>8.4f}")
    print(f"{'mAP@0.5:0.95':<12} {map5095:>8.4f}")

    # Save metrics to file
    metrics_path = OUTPUT_DIR / "metrics_table.txt"
    with open(metrics_path, "w") as f:
        f.write("Metal Nut Defect Detection -- Evaluation Metrics\n")
        f.write("=" * 50 + "\n")
        f.write(f"{'Class':<12} {'AP@0.5':>8}\n")
        f.write("-" * 22 + "\n")
        for i, cls in enumerate(CLASS_NAMES):
            ap = per_class_ap50[i] if i < len(per_class_ap50) else float("nan")
            f.write(f"{cls:<12} {ap:>8.4f}\n")
        f.write("-" * 22 + "\n")
        f.write(f"{'mAP@0.5':<12} {map50:>8.4f}\n")
        f.write(f"{'Precision':<12} {mp:>8.4f}\n")
        f.write(f"{'Recall':<12} {mr:>8.4f}\n")
        f.write(f"{'mAP@0.5:0.95':<12} {map5095:>8.4f}\n")
    print(f"\nMetrics saved to: {metrics_path}")

    # ── Failure Case Analysis ─────────────────────────────────────────────────
    print("\n--- Collecting Failure Cases ---")

    failures = []   # list of (img_path, reason, annotated_img)

    for img_path in sorted(VAL_IMG_DIR.glob("*.png")):
        stem      = img_path.stem
        lbl_path  = VAL_LBL_DIR / (stem + ".txt")
        is_good   = stem.startswith("good_")

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # Ground truth boxes
        gt_boxes = load_gt_boxes(lbl_path, w, h)

        # Model predictions
        preds = model.predict(str(img_path), conf=CONF_THRESH,
                              verbose=False)[0]
        pred_boxes = []
        for box in preds.boxes:
            cls_id = int(box.cls)
            conf   = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            pred_boxes.append((cls_id, x1, y1, x2, y2, conf))

        # ── Detect false positives (pred on good image) ──────────────────
        if is_good and len(pred_boxes) > 0:
            vis = img.copy()
            for (cls_id, x1, y1, x2, y2, conf) in pred_boxes:
                cls_name = CLASS_NAMES[cls_id]
                vis = draw_box(vis, x1, y1, x2, y2,
                               f"FP:{cls_name}({conf:.2f})", (0, 0, 255))
            vis = add_legend(vis)
            reason = (f"False Positive: model detected {CLASS_NAMES[pred_boxes[0][0]]} "
                      f"on a clean nut. Likely texture/shadow mistaken for defect.")
            failures.append((stem, "FP", reason, vis))

        # ── Detect false negatives (missed GT box) ────────────────────────
        elif not is_good and len(gt_boxes) > 0:
            matched = [False] * len(gt_boxes)
            for (pcls, px1, py1, px2, py2, _) in pred_boxes:
                for gi, (gcls, gx1, gy1, gx2, gy2) in enumerate(gt_boxes):
                    if pcls == gcls and iou((px1,py1,px2,py2),(gx1,gy1,gx2,gy2)) >= IOU_THRESH:
                        matched[gi] = True

            if not all(matched):  # at least one GT box missed
                vis = img.copy()
                # Draw GT boxes in green
                for (gcls, gx1, gy1, gx2, gy2) in gt_boxes:
                    vis = draw_box(vis, gx1, gy1, gx2, gy2,
                                   f"GT:{CLASS_NAMES[gcls]}", (0, 255, 0))
                # Draw predictions in red
                for (pcls, px1, py1, px2, py2, conf) in pred_boxes:
                    vis = draw_box(vis, px1, py1, px2, py2,
                                   f"PRED:{CLASS_NAMES[pcls]}({conf:.2f})", (0, 0, 255))
                vis = add_legend(vis)
                defect = CLASS_NAMES[gt_boxes[0][0]]
                reason = (f"False Negative: {defect} defect missed. "
                          f"Likely low contrast or small defect region below confidence threshold.")
                failures.append((stem, "FN", reason, vis))

    # Save top 10 failure cases (at least 5 needed for submission)
    print(f"Total failure cases found: {len(failures)}")
    saved = 0
    failure_log = []
    for (stem, ftype, reason, vis) in failures[:10]:
        out_path = FAILURE_DIR / f"{ftype}_{stem}.png"
        cv2.imwrite(str(out_path), vis)
        failure_log.append({"image": stem, "type": ftype, "reason": reason})
        print(f"  [{ftype}] {stem}: {reason[:80]}...")
        saved += 1

    # Save failure log as JSON
    log_path = OUTPUT_DIR / "failure_cases.json"
    with open(log_path, "w") as f:
        json.dump(failure_log, f, indent=2)

    print(f"\n{saved} failure cases saved to: {FAILURE_DIR}")
    print(f"Failure log saved to        : {log_path}")
    print("\nNext step: run export_onnx.py")


if __name__ == "__main__":
    main()
