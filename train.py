import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
from pathlib import Path
from ultralytics import YOLO

BASE_DIR     = Path(__file__).resolve().parent
DATASET_YAML = BASE_DIR / "dataset" / "data.yaml"
OUTPUT_DIR   = BASE_DIR / "runs" / "detect"
RUN_NAME     = "metal_nut_yolov8n"
WEIGHTS      = BASE_DIR / "yolov8n.pt" if (BASE_DIR / "yolov8n.pt").exists() else "yolov8n.pt"
# ──────────────────────────────────────────────────────────────────────────────

def get_device():
    """Auto-detect best available device: CUDA GPU > CPU."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU detected : {gpu_name}")
        print("Device       : CUDA (GPU) -- training will be fast!")
        return "0", 16          # device id, batch size
    else:
        print("No CUDA GPU detected -- falling back to CPU")
        print("Tip: Install CUDA PyTorch for ~10x faster training.")
        return "cpu", 8         # device, batch size


def main():
    print("=" * 60)
    print("  Metal Nut Defect Detection -- Training (YOLOv8n)")
    print("=" * 60)

    if not DATASET_YAML.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {DATASET_YAML}\n"
            "Run prepare_labels.py first."
        )

    # Auto-detect device
    device, batch_size = get_device()

    # Load pretrained YOLOv8n (downloads from ultralytics if not cached)
    model = YOLO(WEIGHTS)

    print(f"\nModel     : {WEIGHTS} (COCO pretrained, transfer learning)")
    print(f"Dataset   : {DATASET_YAML}")
    print(f"Device    : {device}  |  Batch: {batch_size}")
    print(f"Output    : {OUTPUT_DIR / RUN_NAME}")
    print("-" * 60)

    results = model.train(
        # ── Core ──────────────────────────────────────────────────────────
        data        = str(DATASET_YAML),
        epochs      = 100,
        imgsz       = 640,
        batch       = batch_size,       # 16 on GPU, 8 on CPU
        device      = device,           # "0" (GPU) or "cpu" (auto-detected)
        workers     = 4,

        # ── Output ────────────────────────────────────────────────────────
        project     = str(OUTPUT_DIR),
        name        = RUN_NAME,
        exist_ok    = False,            # fail if run already exists (clean start)

        # ── Transfer learning: freeze backbone ────────────────────────────
        pretrained  = True,
        freeze      = 10,               # freeze first 10 backbone layers

        # ── Training control ──────────────────────────────────────────────
        patience    = 30,               # early stopping
        optimizer   = "AdamW",
        lr0         = 0.001,            # lower LR for fine-tuning
        lrf         = 0.01,
        cos_lr      = True,             # cosine LR schedule
        warmup_epochs = 3,

        # ── Augmentations (cranked up for tiny dataset) ───────────────────
        mosaic      = 1.0,              # always mosaic
        copy_paste  = 0.3,              # paste defect patches onto backgrounds
        flipud      = 0.5,              # vertical flip
        fliplr      = 0.5,              # horizontal flip
        degrees     = 45.0,             # rotation (nuts can be any orientation)
        translate   = 0.1,
        scale       = 0.5,
        hsv_h       = 0.015,            # hue jitter
        hsv_s       = 0.7,              # saturation jitter
        hsv_v       = 0.4,              # brightness jitter
        erasing     = 0.4,              # random erasing

        # ── Regularization ────────────────────────────────────────────────
        weight_decay    = 0.0005,
        dropout         = 0.1,
        label_smoothing = 0.1,

        # ── Misc ──────────────────────────────────────────────────────────
        seed        = 42,
        plots       = True,             # save training curves + confusion matrix
        verbose     = True,
        close_mosaic = 10,              # disable mosaic in last 10 epochs for stability
    )

    best = OUTPUT_DIR / RUN_NAME / "weights" / "best.pt"
    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  Best weights : {best}")
    print(f"  mAP@0.5      : {results.results_dict.get('metrics/mAP50(B)', 'see results.csv'):.4f}")
    print("=" * 60)
    print("\nNext step: run evaluate.py")


if __name__ == "__main__":
    main()
