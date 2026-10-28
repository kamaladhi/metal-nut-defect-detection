import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import time
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError("Run: pip install onnxruntime")

BASE_DIR       = Path(__file__).resolve().parent
DATASET_YAML   = BASE_DIR / "dataset" / "data.yaml"
OUTPUT_DIR     = BASE_DIR / "runs" / "detect"
RESULTS_DIR    = BASE_DIR / "results"

BASELINE_WEIGHTS = OUTPUT_DIR / "metal_nut_yolov8n" / "weights" / "best.pt"
if not BASELINE_WEIGHTS.exists() and (BASE_DIR / "best.pt").exists():
    BASELINE_WEIGHTS = BASE_DIR / "best.pt"
EXP_RUN_NAME     = "metal_nut_yolov8s"
EXP_WEIGHTS_NAME = BASE_DIR / "yolov8s.pt" if (BASE_DIR / "yolov8s.pt").exists() else "yolov8s.pt"

IMG_SIZE    = 640
CONF_THRESH = 0.25
IOU_THRESH  = 0.5
N_WARMUP    = 20
N_BENCH     = 100
# ──────────────────────────────────────────────────────────────────────────────


def get_device():
    if torch.cuda.is_available():
        return "0", 16
    return "cpu", 8


def train_experiment(device, batch_size):
    """Train YOLOv8s with identical settings to the baseline."""
    exp_weights = OUTPUT_DIR / EXP_RUN_NAME / "weights" / "best.pt"

    if exp_weights.exists():
        print(f"YOLOv8s already trained -- skipping training.")
        print(f"Weights: {exp_weights}")
        return exp_weights

    print("\n" + "=" * 60)
    print(f"  Training YOLOv8s experiment")
    print("=" * 60)

    model = YOLO(EXP_WEIGHTS_NAME)
    model.train(
        data        = str(DATASET_YAML),
        epochs      = 100,
        imgsz       = IMG_SIZE,
        batch       = batch_size,
        device      = device,
        workers     = 4,
        project     = str(OUTPUT_DIR),
        name        = EXP_RUN_NAME,
        exist_ok    = False,
        pretrained  = True,
        freeze      = 10,
        patience    = 30,
        optimizer   = "AdamW",
        lr0         = 0.001,
        lrf         = 0.01,
        cos_lr      = True,
        warmup_epochs = 3,
        mosaic      = 1.0,
        copy_paste  = 0.3,
        flipud      = 0.5,
        fliplr      = 0.5,
        degrees     = 45.0,
        translate   = 0.1,
        scale       = 0.5,
        hsv_h       = 0.015,
        hsv_s       = 0.7,
        hsv_v       = 0.4,
        erasing     = 0.4,
        weight_decay    = 0.0005,
        dropout         = 0.1,
        label_smoothing = 0.1,
        seed        = 42,
        plots       = True,
        verbose     = True,
        close_mosaic = 10,
    )
    return exp_weights


def evaluate_model(weights_path: Path, label: str):
    """Run YOLO val and return metrics dict."""
    print(f"\nEvaluating {label}...")
    model = YOLO(str(weights_path))
    results = model.val(
        data  = str(DATASET_YAML),
        split = "val",
        conf  = CONF_THRESH,
        iou   = IOU_THRESH,
        plots = False,
        verbose = False,
    )
    return {
        "label"     : label,
        "weights"   : weights_path,
        "precision" : results.box.mp,
        "recall"    : results.box.mr,
        "map50"     : results.box.map50,
        "map5095"   : results.box.map,
    }


def export_and_benchmark(weights_path: Path, label: str):
    """Export to ONNX and benchmark CPU inference."""
    print(f"\nExporting {label} to ONNX...")
    model = YOLO(str(weights_path))
    onnx_path = Path(model.export(
        format="onnx", imgsz=IMG_SIZE, simplify=True, opset=12, dynamic=False
    ))

    size_mb = onnx_path.stat().st_size / (1024 * 1024)

    # CPU benchmark
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1
    sess_opts.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)

    for _ in range(N_WARMUP):
        session.run(None, {input_name: dummy})

    times = []
    for _ in range(N_BENCH):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        times.append((time.perf_counter() - t0) * 1000)

    mean_ms = np.mean(times)
    return {
        "onnx_path" : onnx_path,
        "size_mb"   : size_mb,
        "mean_ms"   : mean_ms,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not BASELINE_WEIGHTS.exists():
        raise FileNotFoundError(
            f"Baseline weights not found: {BASELINE_WEIGHTS}\n"
            "Run train.py first."
        )

    device, batch_size = get_device()
    print(f"Device: {device} | Batch: {batch_size}")

    # ── Step 1: Train experiment model ───────────────────────────────────────
    exp_weights = train_experiment(device, batch_size)

    # ── Step 2: Evaluate both models ─────────────────────────────────────────
    baseline_metrics = evaluate_model(BASELINE_WEIGHTS, "YOLOv8n (baseline)")
    exp_metrics      = evaluate_model(exp_weights,      "YOLOv8s (experiment)")

    # ── Step 3: Export & benchmark both ──────────────────────────────────────
    baseline_speed = export_and_benchmark(BASELINE_WEIGHTS, "YOLOv8n")
    exp_speed      = export_and_benchmark(exp_weights,      "YOLOv8s")

    # ── Step 4: Print & save comparison report ────────────────────────────────
    map_delta  = exp_metrics["map50"] - baseline_metrics["map50"]
    speed_delta = exp_speed["mean_ms"] - baseline_speed["mean_ms"]
    size_delta  = exp_speed["size_mb"] - baseline_speed["size_mb"]

    lines = [
        "=" * 65,
        "  Improvement Experiment: YOLOv8n vs YOLOv8s",
        "=" * 65,
        f"{'Metric':<22} {'YOLOv8n (base)':>16} {'YOLOv8s (exp)':>16} {'Delta':>8}",
        "-" * 65,
        f"{'mAP@0.5':<22} {baseline_metrics['map50']:>16.4f} {exp_metrics['map50']:>16.4f} {map_delta:>+8.4f}",
        f"{'Precision':<22} {baseline_metrics['precision']:>16.4f} {exp_metrics['precision']:>16.4f} {exp_metrics['precision']-baseline_metrics['precision']:>+8.4f}",
        f"{'Recall':<22} {baseline_metrics['recall']:>16.4f} {exp_metrics['recall']:>16.4f} {exp_metrics['recall']-baseline_metrics['recall']:>+8.4f}",
        f"{'CPU Inference (ms)':<22} {baseline_speed['mean_ms']:>16.1f} {exp_speed['mean_ms']:>16.1f} {speed_delta:>+8.1f}",
        f"{'ONNX Size (MB)':<22} {baseline_speed['size_mb']:>16.1f} {exp_speed['size_mb']:>16.1f} {size_delta:>+8.1f}",
        "-" * 65,
        "",
        "Conclusion:",
    ]

    # Auto-generate conclusion
    if map_delta > 0.02:
        acc_verdict = f"YOLOv8s gives +{map_delta:.3f} mAP@0.5 improvement over YOLOv8n."
    elif map_delta > 0:
        acc_verdict = f"YOLOv8s gives marginal +{map_delta:.3f} mAP@0.5 improvement."
    else:
        acc_verdict = f"YOLOv8s does NOT improve accuracy ({map_delta:.3f} mAP@0.5 delta)."

    if exp_speed["mean_ms"] < 10:
        speed_verdict = "Both models meet the 10ms edge budget."
    elif baseline_speed["mean_ms"] < 10 and exp_speed["mean_ms"] >= 10:
        speed_verdict = ("YOLOv8n fits within the 10ms edge budget, "
                         "but YOLOv8s does not.")
    else:
        speed_verdict = ("Neither model meets a strict 10ms budget -- "
                         "quantization or smaller input would be needed.")

    edge_verdict = (
        "For edge deployment, YOLOv8n is the better choice: "
        f"{baseline_speed['size_mb']:.0f} MB vs {exp_speed['size_mb']:.0f} MB, "
        f"and {baseline_speed['mean_ms']:.0f} ms vs {exp_speed['mean_ms']:.0f} ms CPU latency."
    )

    lines += [acc_verdict, speed_verdict, edge_verdict, "=" * 65]

    for line in lines:
        print(line)

    report_path = RESULTS_DIR / "improvement_report.txt"
    report_path.write_text("\n".join(lines))
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
