# Chest X-ray YOLO research baseline

Four-class image classification: `normal`, `pneumonia`, `tuberculosis`, `covid19`.
Includes manifest validation, image preparation, transfer learning, held-out evaluation,
and single-image inference. No datasets or trained disease weights are included.

This is a research prototype, not a diagnostic device. A chest X-ray classifier's
output cannot establish or exclude these infections. Clinical validation is required
before considering any patient-care use.

## Label design comes first

COVID-19 can cause pneumonia, and diseases can coexist. This baseline uses one
mutually exclusive, curated label per image. Define `pneumonia` as non-COVID
pneumonia for this experiment and exclude ambiguous/co-infected examples rather
than silently relabeling them. If co-occurring labels must be preserved, use a
multi-label architecture/loss instead of this softmax YOLO classifier.

Include normal controls. `normal` means the dataset's normal label, not proof of
absence of disease. Other abnormalities are outside this model's scope; it always
chooses one of the four classes, even for unrelated inputs. Scores are not calibrated
clinical probabilities. Image classification does not produce abnormality boxes;
localization requires location annotations and a separate detection pipeline.

## Windows setup

Install Python 3.11 or 3.12, then run in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Use `py -3.12` instead if that is the installed version. Dependencies are bounded
but not locked; freeze the successfully tested environment for reproducible runs.
The model checkpoint downloads on first training use and requires internet access.
CPU is the default; for an NVIDIA GPU, install the appropriate PyTorch build from
https://pytorch.org/get-started/locally/ and use `--device 0`.

## Prepare Kaggle data

Keep downloads under `data/raw/` (ignored by version control). Supply the actual
dataset links or folder paths before writing a dataset-specific importer: Kaggle
datasets vary in label spelling, directory layout, patient identifiers, and splits.
Check each dataset's license and original source, and retain its provenance.

Create `data/manifest.csv` following `manifest.example.csv`. The example shows the
schema only; it is not a complete dataset. Columns:

| Column | Meaning |
| --- | --- |
| path | PNG/JPEG path, relative to the manifest directory or absolute |
| label | Exactly `normal`, `pneumonia`, `tuberculosis`, or `covid19` |
| patient_id | Stable, pseudonymous patient identifier |
| source | Dataset/site identifier for auditing |
| split | Exactly `train`, `val`, or `test` |

Assign all images of a patient to one split, typically approximately 70/15/15 by
patient while preserving class coverage. Keep an official held-out test set intact.
Identifiers must be globally consistent: namespace unrelated IDs by their original
source, and reconcile the same patient across repackaged datasets. Do not invent
one patient ID per image when patient metadata is absent; patient-independent
evaluation cannot be claimed in that situation.

Use frontal chest X-rays with compatible acquisition views/populations. Do not mix
CT scans, lateral views, segmentation masks, or annotated copies into this baseline.
Convert DICOM with reviewed windowing and photometric interpretation before import;
this script intentionally accepts PNG/JPEG only. Remove identifying information
before adding images. The audit file contains patient IDs and local paths: keep it private.

```powershell
.\.venv\Scripts\python.exe xray.py prepare --manifest data/manifest.csv --output data/prepared
```

Preparation checks required metadata, class coverage in each split, patient overlap,
repeated paths, and decoded-pixel duplicates crossing splits or conflicting labels.
It resizes and pads to 320×320 while preserving aspect ratio. It refuses to overwrite
an existing output directory. Validation completes before output creation, though
an interrupted image write can leave a partial output; use a fresh output path.

Checks cannot identify all repackaged patients or resized/recompressed duplicates.
Review near-duplicates and original metadata manually. Keep `data/prepared` immutable
after preparation; training trusts its audit. Source/site confounding remains a major
risk when each disease comes from a different dataset. Include multiple sources per
class, inspect class/source counts, and reserve an independent site for external validation.

## Train, evaluate, predict

```powershell
.\.venv\Scripts\python.exe xray.py train --data data/prepared --epochs 50 --device cpu
.\.venv\Scripts\python.exe xray.py evaluate --data data/prepared --weights runs/classify/chest_xray/weights/best.pt
.\.venv\Scripts\python.exe xray.py predict --weights runs/classify/chest_xray/weights/best.pt --image data/example.png
```

Ultralytics can increment the run directory when the name already exists; use the
actual path printed by training. Choose hyperparameters using validation data only.
The `evaluate` command explicitly uses `test` and saves Ultralytics evaluation plots
and metrics. Do not repeatedly tune against test performance.

The pretrained checkpoint is `yolo26n-cls.pt`; override with `--model` as needed.
Training disables random erasing, automatic augmentation, color jitter and flips.
Square padded inputs and disabled scaling avoid the default random crop discarding
lung regions. Inference repeats the square padding and uses the saved model input size;
if using an externally produced checkpoint, pass the correct `--size` explicitly.

Before drawing conclusions, report per-class sensitivity, specificity, precision,
F1, AUROC, confusion matrices, patient-level confidence intervals, calibration,
and performance on independent sites and demographic/acquisition subgroups.
Aggregate top-1 accuracy from this baseline is insufficient to assess clinical utility.
No accuracy or clinical performance is claimed here.

## References

- [Ultralytics classification and preprocessing](https://docs.ultralytics.com/tasks/classify/)
- [Ultralytics classification folder format](https://docs.ultralytics.com/datasets/classify/)
- [VinDr-CXR original dataset: pneumonia and tuberculosis annotations](https://physionet.org/content/vindr-cxr/1.0.0/)
- [COVID-19 Image Data Collection and evaluation cautions](https://github.com/ieee8023/covid-chestxray-dataset)

Review Ultralytics' AGPL-3.0/enterprise licensing terms before distribution or deployment.

## Verification status

The project was created in an environment without an installed Python interpreter.
Dependency installation, runtime checks, training, and evaluation have not been run.
Actual dataset mapping and training await the Kaggle links/files.
