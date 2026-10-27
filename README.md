# Metal Nut Defect Detection

Take-home assignment for **InSiSo Technologies**. An edge-sized object detector (YOLOv8n) that flags and localises defects (bent, color, flip, scratch) on metal nuts from the [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) `metal_nut` category.

## Summary

| Item | Result |
|---|---|
| Model | YOLOv8n (COCO-pretrained, 3.0M params), input 640x640 |
| Held-out mAP@0.5 | **0.624** (mAP@0.5:0.95 = 0.284) |
| Precision / Recall | 0.655 / 0.693 |
| ONNX size | 11.7 MB (`best.onnx`) |
| CPU latency | 119 ms/img, 1 thread, i5-12500H, 640x640 (see note in Speed section) |
| Improvement tried | YOLOv8s: worse (-0.070 mAP) and 3x slower. Not adopted. |

The model is not production-ready. It is very good at `flip`, usable on `bent`, and weak on `scratch` and `color`. It also misses the 10 ms budget by a wide margin. Both points are analysed below.

## Project structure

```
prepare_labels.py   MVTec masks -> YOLO boxes, train/val split
visualize_labels.py draws labels on images to check them (results/label_check/)
train.py            YOLOv8n fine-tuning
evaluate.py         P / R / mAP, failure-case images (results/evaluation/)
export_onnx.py      ONNX export + CPU benchmark (results/onnx_benchmark.txt)
improvement.py      YOLOv8n vs YOLOv8s experiment (results/improvement_report.txt)
best.onnx           exported model
```

## Reproduce

```bash
pip install -r requirements.txt
# put MVTec metal_nut/ (train/good, test/*, ground_truth/*) next to the scripts
python prepare_labels.py
python train.py
python evaluate.py
python export_onnx.py
python improvement.py
```
Paths at the top of each script are absolute (`D:\Nxtwave\InsiSo\...`). Change them for your machine.

## 1. Labels

MVTec ships pixel masks, not boxes. I **derived boxes from the masks** instead of hand-labelling:

- Binarise each mask, find external contours, and take `cv2.boundingRect` of each contour. One mask can give several boxes.
- Contours with a box area under 200 px are dropped as noise.
- The class is the defect folder name (`bent`, `color`, `flip`, `scratch`).

Why masks: the boxes are exact and reproducible, and hand-labelling around 93 images would take hours and add inconsistency. The trade-off is that a box around a thin diagonal scratch or a curved bend contains a lot of background. This is a likely reason `scratch` scores low, because IoU with loose boxes is harsh.
I checked the labels visually (`results/label_check/`).

**Split:** the 93 defective test images were split 70/30 per class (stratified, seed 42). Good images were added as negatives (no label file): 110 in train and 22 in val. Result: 174 train / 51 val images, 39 defect instances in val (bent 13, color 8, flip 7, scratch 11).

## 2. Training

- YOLOv8n, COCO-pretrained, first 10 layers frozen, AdamW lr 1e-3 with cosine decay, 100 epochs max, patience 30, batch 16, RTX 3050 laptop GPU.
- Heavy augmentation because there is very little data: mosaic, copy-paste 0.3, HSV jitter, rotation 45 deg, flips, random erasing. Also dropout 0.1 and label smoothing 0.1.
- **Caveat on flips:** `flipud` and `fliplr` are 0.5. A mirrored nut can resemble the `flip` defect, so this augmentation may blur the `flip` class. It scored AP 0.995, which is suspiciously high (see Limitations).

## 3. Evaluation (held-out val split, conf 0.25, IoU 0.5)

| Class | AP@0.5 | Val instances |
|---|---|---|
| bent | 0.695 | 13 |
| color | 0.480 | 8 |
| flip | 0.995 | 7 |
| scratch | 0.325 | 11 |
| **All** | **0.624** | 39 |

Overall precision 0.655, recall 0.693, mAP@0.5:0.95 0.284.

From the confusion matrix (`results/evaluation/yolo_val/confusion_matrix.png`): 12 of 39 defects were missed (3 bent, 3 color, 6 scratch). There were 15 false-positive boxes on good areas (4 bent, 5 color, 6 scratch). 2 defects were given the wrong class (bent predicted as color or scratch).

**Honest caveats:** the val set is only 51 images and 39 instances, so one image moves a class AP by several points. `best.pt` was also selected on this same split, so numbers are slightly optimistic. There is no separate test set.

### Failure cases

Images are in `results/evaluation/failure_cases/`. Green = ground truth, red = prediction.

| Image | Type | What happened and why |
|---|---|---|
| `bent_010` | Miss / duplicate boxes | Several overlapping low-confidence (about 0.26) boxes sit on the bent tab, but none matches the ground-truth box well. The defect is a 3D deformation seen at a glancing angle, and the three GT boxes are nested and ambiguous, so boxes drawn from masks are inconsistent here. |
| `bent_012`, `bent_018` | Miss | _(fill in after viewing: same pattern, bent tab with subtle shape change and little texture contrast)_ |
| `color_004` | Miss | A thin dark-blue streak next to the hole. Its colour is the only cue, and it covers a tiny fraction of the box. The model has only a handful of color examples, and the HSV jitter augmentation can wash out the exact signal that defines this class. |
| `color_015`, `color_019` | Miss | _(fill in after viewing)_ |
| `scratch_000` | Miss | Very fine, low-contrast scratches on the textured metal. They look like the normal surface grain, and at 640 px they cover few pixels. |
| `scratch_001`, `scratch_009`, `scratch_012` | Miss | _(fill in after viewing)_ |
| False positives | FP | The matrix shows 15 FP boxes on good regions (mostly scratch and color) from texture and reflections on the nut. `evaluate.py` currently saves only misses. Add 2 to 3 FP images here (run predictions on the val good images and draw the boxes). |

Please open each image and replace the `fill in` rows with one sentence based on what you actually see. The three explained rows above are from direct inspection.

## 4. ONNX export and speed

- Exported with Ultralytics `model.export(format="onnx")`, run with ONNX Runtime on CPU.
- CPU: 12th Gen Intel Core i5-12500H. Input 640x640. **1 thread**, 20 warmup + 100 timed runs.
- Mean **119.3 ms** (std 19.8, min 85.5, max 181.1), about 8.4 FPS. Size 11.7 MB.
- Note: `improvement.py` measured 79.7 ms for the same model, because its thread and run settings differ. Use the 119 ms single-thread figure as the edge-realistic number.

## 5. Improvement experiment: YOLOv8n vs YOLOv8s

| Metric | YOLOv8n | YOLOv8s | Delta |
|---|---|---|---|
| mAP@0.5 | 0.624 | 0.554 | -0.070 |
| Precision | 0.655 | 0.594 | -0.062 |
| Recall | 0.693 | 0.625 | -0.068 |
| CPU latency | 79.7 ms | 253.5 ms | +174 ms |
| ONNX size | 11.7 MB | 42.7 MB | +31 MB |

**Did it work? No.** The larger model was worse and about 3x slower. Likely reason: with about 93 defective images, extra capacity overfits or cannot be trained well, and the val noise (39 instances) also limits what the comparison can show. This confirms the data is the bottleneck, not the model size. For edge use, YOLOv8n stays the choice.
Next things I would try: input 960 for small defects, turning off flips and colour jitter (to protect the `flip` and `color` classes), and tuning the confidence threshold per class.

## Written answers

### A. 30 defective images, 3,000 good images

A detector trained on 30 defects will not generalise, so I would lean on the 3,000 good images. First, build an anomaly-detection model that only needs good parts (PatchCore or PaDiM style: store features of normal parts and flag anything that differs). This gives an anomaly heatmap, so it also shows where the defect is. MVTec itself is built for this. Second, use the 30 defective images to pick the alert threshold and to measure recall, never for training alone. Third, enlarge the defect set cautiously: augmentation (not flips or colour changes that alter defect meaning), copy-paste of defect crops onto good parts, and a few synthetic defects, while keeping every real defect image for validation only. Fourth, add a human-in-the-loop step: flagged parts are reviewed, and confirmed defects flow back into a growing labelled set so a supervised detector can replace the anomaly model once there are enough examples. Report recall with confidence intervals, because 30 samples give a wide error bar.

### B. Model starts missing defects two weeks after going live

In order, cheapest and most likely first:
1. **Is the system running correctly?** Check model version, preprocessing, resolution, camera feed and confidence threshold are unchanged. A config change is the most common cause.
2. **Camera and lighting:** lens dirt, focus drift, exposure, light ageing, ambient light, camera moved. Compare recent frames with training images (brightness and sharpness statistics).
3. **Part and process change:** new batch, supplier, finish, orientation or position on the line, or a new defect type.
4. **Data drift:** compare image feature or embedding distributions and score distributions with the training set to quantify the shift.
5. **Real label check:** pull a sample of missed parts (use QA or customer returns) and review them to confirm they are true misses, not label disagreement.
6. **Fix:** retrain with new images, recalibrate the threshold, and add monitoring (score histogram, defect rate, periodic golden-sample checks) so it is caught in days, not weeks.

### C. 40 ms model, 10 ms budget

A 4x speed-up is needed. Options, roughly by effort and risk:
1. **Check the baseline:** measure on the real controller with its runtime (ONNX Runtime, OpenVINO, TensorRT, NPU SDK) because desktop numbers mislead. Fix any slow pre- and post-processing.
2. **Quantise:** INT8 post-training quantisation typically gives 2 to 3x on CPU or NPU. Validate recall afterwards, since small defects suffer first.
3. **Reduce input size:** 640 to 416 or 320 gives about 2 to 4x, but small scratches may vanish. Alternatively crop to the nut region first and run on a tight crop at lower resolution.
4. **Smaller architecture or pruning:** a lighter backbone (MobileNet, NanoDet-style) and distillation from the current model.
5. **Pipeline changes:** skip frames if the line allows, run a tiny "is it defective" classifier first and the detector only on suspects, or use a fixed-shape, fused model.
6. **Hardware:** a stronger edge accelerator if the above still miss the target or lose too much accuracy.
I would try 2, then 3, and measure both latency and recall after each step.

## Limitations

- Very small val set (51 images, 39 defects). Metrics are noisy, and no separate test set was used.
- `scratch` and `color` are weak (AP 0.33 and 0.48), so the model would let defects through. The latency (119 ms) misses a 10 ms budget by about 12x.
- Flip augmentation may interact with the `flip` class. The 0.995 AP should be treated with care.
- Boxes from masks are loose around thin or curved defects.

## AI tools used

I used an AI coding assistant (Claude / Antigravity) to help write the scripts (label conversion, training, evaluation, export) and to draft this report. I reviewed the code and can walk through every decision. Failure-case explanations come from inspecting the output images.

## Dataset license

MVTec AD is free for non-commercial use. This project is for an internship assessment only.
