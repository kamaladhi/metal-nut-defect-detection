# Metal Nut Defect Detection

> Take-home assignment for **InSiSo Technologies** — Edge-deployable object detector for industrial metal nut defects using the [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) dataset.

---

## Problem Statement

A customer makes metal nuts. Some come off the line with defects (**bent, scratched, color flaw, flipped**). This project builds a camera station that flags defective parts and localises the defect — designed to run on a small edge device next to the line.

---

## Project Structure

```
metal-nut-defect-detection/
├── prepare_labels.py        # Convert MVTec masks → YOLO bounding boxes
├── train.py                 # YOLOv8n training with transfer learning
├── evaluate.py              # Evaluation metrics + failure case analysis
├── export_onnx.py           # ONNX export + CPU inference benchmarking
├── improvement.py           # Improvement experiment
├── dataset/
│   └── data.yaml            # YOLO dataset config
├── results/                 # Training curves, confusion matrix, failure cases
├── best.onnx                # Exported model (or see Releases)
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/<your-username>/metal-nut-defect-detection.git
cd metal-nut-defect-detection

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

### Dataset

Download the **metal_nut** category from [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) and place it at:
```
metal_nut/
├── train/good/
├── test/{bent,color,flip,scratch,good}/
└── ground_truth/{bent,color,flip,scratch}/
```

---

## Reproduce Results

```bash
# Step 1: Convert masks to YOLO bounding boxes + build dataset split
python prepare_labels.py

# Step 2: Train YOLOv8n with transfer learning
python train.py

# Step 3: Evaluate + failure analysis
python evaluate.py

# Step 4: Export to ONNX + benchmark CPU inference
python export_onnx.py

# Step 5: Run improvement experiment
python improvement.py
```

---

## Approach

*To be filled after experiments.*

---

## Results

*To be filled after training.*

| Class   | Precision | Recall | mAP@0.5 |
|---------|-----------|--------|---------|
| bent    | —         | —      | —       |
| color   | —         | —      | —       |
| flip    | —         | —      | —       |
| scratch | —         | —      | —       |
| **All** | —         | —      | —       |

---

## Failure Cases

*To be filled after evaluation.*

---

## ONNX Export & Inference Speed

*To be filled after export.*

---

## Improvement Experiment

*To be filled after experiment.*

---

## Written Questions

### A. Only 30 defective images vs 3,000 good images — how to build a reliable system?

*To be filled.*

### B. Model starts missing defects 2 weeks after going live — what to check?

*To be filled.*

### C. Model takes 40 ms but budget is 10 ms — what are your options?

*To be filled.*

---

## AI Tools Used

This project was developed with the assistance of AI coding tools (Antigravity / Claude). All code has been reviewed and understood by the author. The approach, decisions, and written analysis are original.

---

## License

This project is for non-commercial research/internship purposes. Dataset subject to [MVTec AD License](https://www.mvtec.com/company/research/datasets/mvtec-ad).
