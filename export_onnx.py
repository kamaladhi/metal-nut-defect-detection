import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import time
import numpy as np
import platform
import cpuinfo          # pip install py-cpuinfo
from pathlib import Path
from ultralytics import YOLO

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError("Run: pip install onnxruntime")

BASE_DIR    = Path(__file__).resolve().parent
WEIGHTS     = BASE_DIR / "runs" / "detect" / "metal_nut_yolov8n" / "weights" / "best.pt"
if not WEIGHTS.exists() and (BASE_DIR / "best.pt").exists():
    WEIGHTS = BASE_DIR / "best.pt"
ONNX_OUT    = BASE_DIR / "best.onnx"
REPORT_OUT  = BASE_DIR / "results" / "onnx_benchmark.txt"
IMG_SIZE    = 640        # must match training imgsz
N_WARMUP    = 20         # warmup runs (excluded from measurement)
N_BENCH     = 100        # benchmark runs
# ──────────────────────────────────────────────────────────────────────────────


def get_cpu_name():
    try:
        info = cpuinfo.get_cpu_info()
        return info.get("brand_raw", platform.processor())
    except Exception:
        return platform.processor()


def export_onnx(weights_path: Path, img_size: int) -> Path:
    """Export .pt -> .onnx using Ultralytics built-in export."""
    print(f"Exporting {weights_path.name} -> ONNX (imgsz={img_size})...")
    model = YOLO(str(weights_path))
    export_path = model.export(
        format   = "onnx",
        imgsz    = img_size,
        simplify = True,          # simplify ONNX graph
        opset    = 12,            # broad device compatibility
        dynamic  = False,         # fixed batch=1 for edge deployment
    )
    print(f"Exported to: {export_path}")
    return Path(export_path)


def benchmark_onnx(onnx_path: Path, img_size: int, n_warmup: int, n_bench: int):
    """
    Benchmark ONNX inference on CPU using ONNXRuntime.
    Returns (mean_ms, std_ms, min_ms, max_ms).
    """
    # Force CPU execution
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1      # single thread (edge-realistic)
    sess_opts.inter_op_num_threads = 1

    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(1, 3, img_size, img_size).astype(np.float32)

    # Warmup
    print(f"Warming up ({n_warmup} runs)...")
    for _ in range(n_warmup):
        session.run(None, {input_name: dummy})

    # Benchmark
    print(f"Benchmarking ({n_bench} runs)...")
    times = []
    for _ in range(n_bench):
        start = time.perf_counter()
        session.run(None, {input_name: dummy})
        times.append((time.perf_counter() - start) * 1000)  # ms

    times = np.array(times)
    return times.mean(), times.std(), times.min(), times.max()


def main():
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)

    if not WEIGHTS.exists():
        raise FileNotFoundError(
            f"Weights not found: {WEIGHTS}\nRun train.py first."
        )

    print("=" * 60)
    print("  Metal Nut Defect Detection -- ONNX Export & Benchmark")
    print("=" * 60)

    # ── Export ───────────────────────────────────────────────────────────────
    onnx_path = export_onnx(WEIGHTS, IMG_SIZE)

    # ── Benchmark ────────────────────────────────────────────────────────────
    cpu_name = get_cpu_name()
    mean_ms, std_ms, min_ms, max_ms = benchmark_onnx(
        onnx_path, IMG_SIZE, N_WARMUP, N_BENCH
    )

    # ── Report ───────────────────────────────────────────────────────────────
    onnx_size_mb = onnx_path.stat().st_size / (1024 * 1024)

    report_lines = [
        "=" * 60,
        "  ONNX Export & CPU Inference Benchmark",
        "=" * 60,
        f"Model (ONNX)  : {onnx_path.name}",
        f"Model size    : {onnx_size_mb:.1f} MB",
        f"Image size    : {IMG_SIZE} x {IMG_SIZE}",
        f"CPU           : {cpu_name}",
        f"Threads       : 1 (single-thread, edge-realistic)",
        f"Warmup runs   : {N_WARMUP}",
        f"Benchmark runs: {N_BENCH}",
        "-" * 60,
        f"Mean latency  : {mean_ms:.1f} ms",
        f"Std deviation : {std_ms:.1f} ms",
        f"Min latency   : {min_ms:.1f} ms",
        f"Max latency   : {max_ms:.1f} ms",
        f"Throughput    : {1000/mean_ms:.1f} FPS (single image)",
        "=" * 60,
    ]

    for line in report_lines:
        print(line)

    REPORT_OUT.write_text("\n".join(report_lines))
    print(f"\nReport saved to: {REPORT_OUT}")


if __name__ == "__main__":
    main()
