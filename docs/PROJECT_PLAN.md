# Scope and experimental plan

Deadline supplied by student: 23 October 2026. Presentation booking is separate.

## Engineering question
Can image-based inspection identify and locate annotated packaging defects, and
what accuracy/runtime trade-off is achievable on the available hardware?

## Two related outputs, evaluated separately
1. **Course-aligned baseline:** estimate annotated defect count per RGB image with
   linear regression, decision-tree regression, support vector regression and a
   small multilayer perceptron. This uses Weeks 5.1-5.4 directly. It cannot localise
   defects. The target is number of annotation boxes, NOT the number of rashers.
2. **Detection extension:** fine-tune pretrained YOLOv8n to output class, bounding
   box and confidence. Compare at least two input resolutions (320 and 640), using
   identical data and seed. Compare its predicted count against the regression
   baseline on the same images, alongside detection-specific metrics.

The count task is a limited image-level inspection indicator, not a substitute for
localisation or verified food safety. No unlabelled image is assumed certified good.
No claims about freshness, microbial safety or physical defect severity are supported.

## Course mapping
- Weeks 5.1-5.4: OLS linear regression, tree depth/leaf control, scaled RBF SVR,
  scaled neural regression.
- Weeks 5.5-5.6 and 3.2: baseline comparison, MAE/RMSE/R2, overfitting, split
  independence, train-only preprocessing, validation selection and held-out testing.
- Week 2.1: timing, profiling, feature-resolution trade-offs; report timing boundaries.
- Weeks 1.2-1.5: real Git history, docstrings/HTML documentation, tests and AI reflection.

## Locked development rules
- Dataset prepare stage audits all labels/images but never trains on test images.
- Original predefined splits are preserved except duplicate quarantine: test takes
  priority over validation over training. Connected source/hash groups cannot cross splits.
- CPU count baseline uses one lexicographically chosen image per retained source to
  avoid treating repeated exposure variants as independent observations.
- Detection training uses all retained training images. Additional random geometric
  and colour augmentations are disabled initially so changes are controlled.
- Count model/feature-grid selection uses validation MAE; no test scores yet.
- Detector selection uses validation mAP50-95 plus measured runtime. Choose any
  operational confidence threshold on validation data only.
- Final test is run once after model/settings/threshold are frozen. A revised model
  after examining test outcomes must be labelled exploratory, not independently tested.

## Known validity limits requiring investigation
Adjacent video frames can represent the same tray even after filename/hash cleaning.
Before final claims, inspect frame sequences and perform a temporally separated or
tray-grouped robustness experiment if meaningful group boundaries can be established.
Do not label the current split tray-independent. Wrinkle has no validation/test labels;
its recall cannot be established. Rare-class uncertainty must be reported.

## Remaining work
1. Run actual detector training at 320/640 on the student's RTX 4060 8 GB.
2. Inspect failed detections, assess temporal leakage, and compare common count output.
3. Freeze settings; run final test and record the exact local GPU environment.
4. Write 5-10 page report from real evidence, include references and AI reflection.
5. Prepare presentation and interview explanations; export final repository clone ZIP.
