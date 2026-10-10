# Implementation verification and final score reproduction

Finalisation check, 10 October 2026:

- `python -m pytest -q --junitxml=results/tests.xml`: **17 passed** in 2.76 s.
- Regenerated HTML documentation for `rasher`, `scripts.analyze_export` and `scripts.score_final_test` using pdoc 16.0.0.
- Re-scored the included final-test prediction export using `results/prepared_manifest.json`, the included selected checkpoint and frozen confidence 0.65.
- The reproduced summary matched `results/final_test/summary.json` exactly as parsed JSON.

Tests cover analytically known COCO-to-YOLO coordinates, boundary clipping,
invalid/nonfinite/degenerate boxes, source grouping and transitive quarantine,
training-augmentation retention, synthetic RGB/gradient descriptors, feature-setting
validation, known MAE/RMSE/R2, synthetic ZIP preparation, the final-test guard,
box IoU and class-aware prediction matching including duplicates and empty images.

The earlier development report had 14 passing tests. Three export-scoring tests
were subsequently added, bringing the current pytest suite to 17. The report's
historical verification account predates this finalisation run.

The selected weights' checksum was verified. Saved-prediction reproduction does
not re-run GPU inference, retrain the model or create a new independent test set.
Actual training/evaluation evidence remains in the detector-validation and final-test
records. No tray-independent validation or production-line deployment test was performed.
