# MMA3001 Pork rasher packaging inspection

**Status: experiments completed, frozen final test scored, report and presentation included.**
Student ID: 33318719. Deadline supplied by student: 23 October 2026.

This project locates visible packaging and meat-placement defects in RGB images.
All five supplied classes were trained: loose-meat, packaging-error, twisted-meat,
unsealed and wrinkle. The export treats these as separate labels, although an
unsealed package is a packaging error in ordinary engineering language. The precise
annotation rule separating the labels was not established. The model does not
assess freshness, microbial safety or certified package pass/fail.

## Final evidence

- Original dataset: 3,549 images. Duplicate-source/hash quarantine excluded 37.
- Retained images: 3,313 training, 119 validation, 80 test. Physical-tray independence is unknown.
- Selected model: YOLOv8n, 320-pixel input, best epoch 13, frozen confidence 0.65.
- Test: 37 true positives, 9 false positives, 42 missed annotations.
- Pooled precision 80.4%, recall 46.8%, F1 0.592 at matching IoU >= 0.50.
- Ultralytics test mAP50 0.2444, mAP50–95 0.0995.
- Unsealed: 32/49 found. Packaging-error: 5/23. Loose-meat: 0/3. Twisted-meat: 0/4.
- No held-out wrinkle examples, so its performance cannot be assessed.
- Test count MAE: constant-one baseline 0.2875, detector 0.4875. Lower is better.

See [validation analysis](results/detector_validation/REVIEW.md),
[final-test analysis](results/final_test/FINAL_TEST_REVIEW.md), and
[dataset audit](results/dataset_audit.json). AP uses Ultralytics evaluation;
the fixed operating-point scores use our separate class-aware greedy matcher.
Validation exports and ordinary prediction mode can use different postprocessing.

## Repository structure

| Location | Purpose |
|---|---|
| `rasher/` | Dataset preparation, image descriptors, regression and detector CLI |
| `scripts/` | Validation export analysis and frozen final-test scoring |
| `models/` | Selected checkpoint, hash and attribution |
| `tests/` | Analytical implementation tests |
| `docs/api/index.html` | Generated HTML documentation, open locally in a browser |
| `docs/AI_USE.md` | AI contribution disclosure and personal reflection |
| `results/` | Metrics, prediction exports, provenance and test report |
| `submission/` | Report, presentation and submission guidance |

## Setup

Windows with Python 3.12, from a terminal in the repository:
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
Use `.\.venv\Scripts\python.exe` in place of `python` below. On Linux/macOS,
create the environment with `python3 -m venv .venv` and use `.venv/bin/python`.
CPU requirements record the original baseline environment. The GPU run environment
is recorded in `results/detector_validation/gpu-environment.txt`.

## Review the final results without a GPU or dataset download

The included manifest records the annotation boxes and split membership. It contains
no image pixels. Re-score the saved predictions with the frozen checkpoint checksum:
```powershell
python -m scripts.score_final_test --manifest results/prepared_manifest.json --export results/final_test/export --weights models/yolov8n_320_best.pt --out results/reproduced_final_test
```
This reproduces the score from saved predictions. It does not re-run inference or
perform a new independent experiment. Compare its `summary.json` with
`results/final_test/summary.json`. Do not tune using the inspected test results.

## Obtain images and reproduce the experiments

Download **version 4, COCO JSON ZIP** from:
https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging/dataset/4

The raw archive and image dataset are excluded from Git to keep the clone manageable.
The original archive hash is recorded in the dataset audit.
```powershell
python -m rasher prepare --zip "C:\path\Pork Rasher Error-Packaging-.v4i.coco.zip" --out data/prepared
python -m rasher baseline --data data/prepared --out results/my_baseline
```
Preparation/baseline output directories must be new or empty. Preparation creates
machine-specific paths in `data.yaml`, so regenerate it after moving machines.
The regression task counts defect annotation boxes, not meat slices. Its training
uses one representative per source and selects configurations by validation MAE.

For GPU inference/training, first install an appropriate CUDA-enabled PyTorch build
using https://pytorch.org/get-started/locally/ and then:
```powershell
python -m pip install -r requirements-detector.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -m rasher.detector predict --weights models/yolov8n_320_best.pt --imgsz 320 --conf 0.65 --source data/prepared/images/valid --name demo_320
```
The recorded laptop used PyTorch 2.11.0+cu128, Ultralytics 8.4.175 and an RTX 4060
Laptop GPU with 8 GB VRAM. The default device is GPU 0; use `--device cpu` if needed.
`--conf` controls prediction mode only; AP evaluation retains its default confidence handling.

To repeat the two development runs:
```powershell
python -m rasher.detector train --imgsz 320 --epochs 50 --batch 8 --name rerun_320
python -m rasher.detector train --imgsz 640 --epochs 50 --batch 8 --name rerun_640
```
Generic pretrained weights download on first use. Workers are zero for Windows.
Use the actual output directory printed by Ultralytics. In the recorded environment,
relative project paths resolved to `runs/detect/runs/detect/...`; do not assume a
shorter path. The selected original checkpoint is provided under `models/`.
A new run may differ due to software/hardware changes despite seed 42.

## Tests and documentation

```powershell
python -m pytest -q --junitxml=results/tests.xml
python -m pdoc rasher scripts.analyze_export scripts.score_final_test -o docs/api
```
See the [current test report](results/TEST_REPORT.md). Tests verify implementation,
not the reliability of predictions. Original development findings retain their
historical scope; final results above supersede their pending-experiment statements.

## Documents and presentation demonstration

- [Word report](submission/MMA3001_Project_Report.docx)
- [Presentation with speaker notes](submission/MMA3001_Presentation.pptx)
- [Submission and live demonstration guide](submission/README.md)

The five-minute presentation must include a brief live view of the GitHub structure,
HTML documentation and testing evidence, followed by three minutes of questions.
Appointment booking/attendance and Moodle submission are the student's remaining actions.

## Licensing and AI

Original code: [MIT](LICENSE). Dataset annotations/images: CC BY 4.0, provider Hello.
The included trained checkpoint derives from Ultralytics YOLOv8n and retains upstream
licensing conditions. See [third-party attribution](THIRD_PARTY.md) and [model details](models/README.md).
Course PDFs and credentials are not included. AI generated most initial code and prose.
The student ran the GPU experiments and inspected shell outputs and epoch losses.
See [AI contribution and reflection](docs/AI_USE.md).
