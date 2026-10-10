# Detector validation review — 10 October 2026

Status: validation analysis complete; final test performance is unknown.

## Evidence and model selection

Both runs used YOLOv8n pretrained weights, seed 42, batch 8, maximum 50 epochs,
patience 15, the same prepared data and disabled geometric/colour augmentation.
The data exporter retained 3,313 training images from 1,073 inferred sources,
119 validation images and 80 test images. This is not a verified tray-independent split.

| Metric | 320 pixels | 640 pixels |
|---|---:|---:|
| Best epoch | 13 | 8 |
| Completed epochs | 28 | 23 |
| Training duration, CSV seconds | 1207.88 | 1036.90 |
| Re-evaluated validation mAP50 | 0.425224 | 0.326646 |
| Re-evaluated validation mAP50–95 | 0.180976 | 0.177776 |
| Selected confidence | 0.65 | 0.30 |
| Pooled precision at selected confidence | 0.779070 | 0.781609 |
| Pooled recall at selected confidence | 0.558333 | 0.566667 |
| Pooled F1 at selected confidence | 0.650485 | 0.657005 |
| Correct detections | 67 | 68 |
| False detections | 19 | 19 |
| Missed labelled defects | 53 | 52 |
| Defect-count MAE | 0.386555 | 0.394958 |

Select **320 pixels**, using the higher validation mAP50–95 as the detector
selection metric; mAP50 supports this choice. The mAP50–95 difference is small
and one seed provides no evidence of statistical significance. The 640 model
has slightly higher pooled F1 at its own selected threshold; there is no claim
that 320 wins every metric. No inference-speed winner is established by the
short, separately timed runs. The 320 training took longer because it completed
more epochs and its later epochs slowed down.

Both curves show strong evidence of overfitting: training loss continues to
fall while validation losses rise and detection metrics decline after their
early peaks. Early stopping retained the best checkpoint. Disabling further
augmentation is a limitation of these initial experiments, not an optimized
training policy. Increasing resolution alone did not resolve generalisation.

## Threshold analysis

The validation exports contain image filenames, pixel XYWH boxes, confidence
scores and one-based class IDs. The analysis maps category_id minus one to the
manifest's zero-based classes, verified against Ultralytics v8.4.175 source.
All exported filenames were checked against the 119-image validation manifest.
Images with no retained predictions still contribute false negatives/count errors.

Search confidence thresholds 0.05, 0.10, ..., 0.95. At each threshold, order
predictions by descending confidence and match each to the highest-IoU unused
ground-truth box of the same class, requiring IoU >= 0.5. Duplicate detections
are false positives. Choose highest pooled F1; break ties toward higher confidence.
These are custom operating-point scores, distinct from Ultralytics class-averaged
precision/recall and threshold-integrated AP. This is an exploratory choice made
after inspecting initial validation results, not a preregistered protocol.

For 320 pixels, increasing confidence from 0.25 to 0.65 reduces false detections
from 36 to 19 but increases missed defects from 48 to 53. This favours a balance
of precision and recall; it does not prioritize maximum defect recall.

| Class at 320 / 0.65 | Labelled boxes | Correct | False detections | Missed |
|---|---:|---:|---:|---:|
| loose-meat | 3 | 0 | 0 | 3 |
| packaging-error | 34 | 7 | 14 | 27 |
| twisted-meat | 3 | 1 | 0 | 2 |
| unsealed | 80 | 59 | 5 | 21 |
| wrinkle | 0 | 0 | 0 | 0 |

Unsealed precision is 59/64 = 92.19%; recall is 59/80 = 73.75%. These figures
are validation estimates. Wrinkle recall is undefined, not perfect; it has no
validation examples. The rare classes have too few examples for stable estimates.
Pooled metrics are dominated by unsealed, with 80/120 annotations.

For the chosen operating point, 1/14 images with no labels is flagged and
27/105 images with at least one label have no predictions. These image-level
counts ignore whether the predicted class/location is correct and therefore
must not replace the detection results. No labels means no annotated defect,
not certified good packaging or food safety.

## Comparison with taught regression baseline

The taught-method pipeline predicts annotated defect count. Its selected
training-median baseline always predicts one and obtains validation MAE
0.243697. Detector count MAE at the selected detection threshold is 0.386555,
so the detector does **not** beat that simple baseline on count error.
It does provide locations and categories, which the count baseline cannot.
Its threshold was chosen for detection F1, not count MAE. These are separate tasks.

## Frozen final evaluation

`frozen_selection.json` records the 320 checkpoint SHA-256, image size 320,
confidence 0.65 and matching criterion before any test predictions are evaluated.
Use the final-test export for threshold-integrated Ultralytics AP and apply
the same frozen confidence/matching rules for operating-point scores.
The wrapper's --conf parameter only affects predict mode; final-test exports
low-confidence predictions for AP, then this threshold is applied offline.
Validation uses multi-label NMS; the ordinary predict display path may use
different postprocessing. Report scores as validation/final-test export scores,
not as a verified deployment benchmark.

Do not tune settings after inspecting test results and then present that same
test as untouched. More development would require preserving an independent
evaluation set. Current evidence supports an experimental coursework prototype,
not a reliable general packaging inspection system.

## Reproduction and checks

Run from repository root:

```text
python scripts/analyze_export.py --manifest data/prepared/manifest.json --exports results/detector_validation --out results/detector_validation
```

The input exports and resolved Windows environment are retained alongside the
output CSVs. Training args retain the user's original Windows paths for provenance.
No weights or dataset images are added to Git by this analysis.

Three analytical test functions in tests/test_export_analysis.py passed by direct
execution: known box overlap, duplicate/wrong-class detections, and empty-image/
threshold-boundary cases. Pytest was unavailable in the current analysis runtime;
the existing earlier 14-test report refers to the previous development environment.

Source for JSON category mapping and serialization:
https://github.com/ultralytics/ultralytics/blob/v8.4.175/ultralytics/models/yolo/detect/val.py
