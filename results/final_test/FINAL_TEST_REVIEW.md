# Final test results — 10 October 2026

The frozen YOLOv8n model trained at 320 pixels was evaluated on 80 held-out
images containing 79 labelled defects. The uploaded best.pt SHA-256 matches
the checkpoint recorded before test evaluation. Confidence remained 0.65;
no test-set threshold search or model selection was performed.

| Metric | Validation | Final test |
|---|---:|---:|
| Images | 119 | 80 |
| Labelled defects | 120 | 79 |
| Correct detections | 67 | 37 |
| False detections | 19 | 9 |
| Missed defects | 53 | 42 |
| Pooled precision | 77.91% | 80.43% |
| Pooled recall | 55.83% | 46.84% |
| Pooled F1 | 0.6505 | 0.5920 |
| Ultralytics mAP50 | 0.4252 | 0.2444 |
| Ultralytics mAP50–95 | 0.1810 | 0.0995 |
| Detector count MAE | 0.3866 | 0.4875 |
| Constant-one count baseline MAE | 0.2437 | 0.2875 |

## Interpretation

Precision means 37 of the 46 retained detections matched a labelled defect of
the same class with intersection over union (IoU) at least 0.5. Recall means
37 of the 79 labelled defects were matched. Thus a precision of 80.43% does
not mean the system detected 80% of defects: it missed more than half.

| Class | Test annotations | Correct | False detections | Missed | Recall |
|---|---:|---:|---:|---:|---:|
| loose-meat | 3 | 0 | 0 | 3 | 0% |
| packaging-error | 23 | 5 | 4 | 18 | 21.74% |
| twisted-meat | 4 | 0 | 0 | 4 | 0% |
| unsealed | 49 | 32 | 5 | 17 | 65.31% |
| wrinkle | 0 | 0 | 0 | 0 | Not measurable |

Unsealed packaging is the strongest category, with precision 32/37 = 86.49%
and recall 32/49 = 65.31%. All seven labelled loose/twisted-meat defects were
missed at the frozen operating point. Their sample sizes are too small for
stable population estimates. Wrinkle has no test labels and cannot be assessed.
Where there are no predictions, per-class precision is mathematically undefined;
the machine-readable scoring output uses a zero convention for that case.

The exported PR plot reports AP50 approximately 0.771 for unsealed, 0.206 for
packaging-error, and 0.000 (rounded) for loose and twisted meat. The overall AP
averages classes represented in the labels, excluding the absent wrinkle class.
These threshold-integrated AP scores differ from the pooled metrics above.
Ultralytics' own summary precision/recall also uses a different averaging and
operating-point procedure; it must not replace the frozen-threshold scores.

Count RMSE is 0.8441 for the detector versus 0.5809 for the selected constant-one
baseline. The detector predicts the exact total count on 48/80 images, but a
correct count alone does not establish correct locations or classes. One of 12
images with no labels is flagged, and 26 of 68 images with labels receive no
retained prediction. These image-level results ignore localization and class.

## Engineering conclusion

The experiment demonstrates a functioning data preparation, model training,
validation selection and held-out evaluation workflow. It shows useful but
incomplete recognition of unsealed packaging. It does not demonstrate reliable
multi-class inspection or an improvement over the simple count baseline.

The test decline is consistent with the generalisation concerns observed during
validation. This experiment does not isolate the causes: limited unique source
images, rare categories, annotation ambiguity, disabled online augmentation and
possible train/validation/test scene differences are candidates for future study.
Adjacent video frames may share a physical tray despite disjoint inferred source
names, so this is not a verified tray-independent industrial benchmark.

No annotation means no labelled defect, not certified acceptable packaging or
food safety. The test has now been evaluated. Future tuning should use training/
validation data and should not reuse this test as an untouched final benchmark.

## Reproduction and evidence

`summary.json` stores exact scores, baseline errors and the frozen selection.
`image_results.csv` stores all 80 image-level records. `export/` retains the
original prediction JSON, Ultralytics metrics and precision-recall plot.

Run from repository root with the existing prepared dataset and checkpoint:

```text
python -m scripts.score_final_test --manifest data/prepared/manifest.json --export results/final_test/export --weights runs/detect/runs/detect/yolov8n_320/weights/best.pt
```

The scorer uses the same analytically checked matching function as validation:
descending-confidence, class-aware, one-to-one matching at IoU >= 0.5. It verifies
the checkpoint hash and rejects exported filenames outside the test manifest.
No new model inference was performed in this analysis environment. Results are
based on the user's supplied GPU evaluation exports, whose validation pipeline
uses multi-label postprocessing. A separate deployment predictor may differ.
