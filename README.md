# Chest X-ray YOLO research baseline

Three classes: **normal, pneumonia, tuberculosis**. UNKNOWN and COVID-19 are excluded.
This is an experimental classifier, not a clinically validated diagnostic device.

`xray.py` is the active pipeline. `xray.original.py` preserves the original starter.

## Corrected data assignments

Use `data/splits/manifest_pixels.csv`, not the original physical train/val/test folders.
The new manifest excludes 1,357 UNKNOWN images and preserves original metadata.

| Class | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| normal | 3842 | 823 | 824 |
| pneumonia | 2991 | 641 | 641 |
| tuberculosis | 2949 | 622 | 626 |
| Total | 9782 | 2086 | 2091 |

The decoded-pixel scan found additional identical images with different file bytes.
`repair-pixel-splits.py` moved 12 assignments to keep their connected groups together.
The prior manifest and per-split CSVs remain for provenance; use manifest_pixels.csv
for preparation. The repair details are in data/splits/pixel_repair.json.

Duplicate files and matching patient filename groups remain in one split. Patient
IDs inferred from filenames are marked unverified; 7,303 images have unavailable
IDs, which remain blank. group_id is a technical grouping key, not a confirmed
patient identity. See data/splits/README.md for grouping details and limitations.

## Environment

Python 3.12 and a local virtual environment have been installed. To recreate:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Choose `.venv\Scripts\python.exe` as the VS Code interpreter. Activation is optional.
The tested dependencies are recorded in requirements.lock.txt when installation completes.

## Prepare and train

```powershell
.\.venv\Scripts\python.exe xray.py prepare
.\.venv\Scripts\python.exe xray.py smoke
.\.venv\Scripts\python.exe xray.py train --epochs 50 --device cpu
```

Preparation follows the corrected CSV and writes padded 320x320 PNGs to
`data/prepared_three_class/`. It verifies current file hashes, patient/group overlap,
and decoded-pixel duplicates across splits or labels. The original images stay
unchanged. Source metadata and patient ID status are retained in audit.json.
Existing outputs are never overwritten. An interrupted run leaves a .incomplete
folder for inspection; a complete directory appears only after success.

The smoke test uses 36 training and 18 validation images at 64 pixels for one epoch.
It selects unique groups within each class/split and never loads test images. Its
scores are for pipeline checking only. It refuses to overwrite its existing subset.

To repeat the smoke test, pass a fresh subset folder with `--output data/smoke_next`.
The verified run uses data/smoke_verified and saved its checkpoint under
runs/classify/smoke_three_class/weights/best.pt. It is not the full trained model.

Full training uses a pretrained yolo26n-cls checkpoint, batch 16, seed 42, early
stopping patience 15, and validation for checkpoint selection. Random scaling,
flips, automatic augmentation, erasing, and color jitter are disabled. The checkpoint
downloads on first use. Repeated runs can have incremented output directory names.
Use the actual checkpoint path printed by training.

CPU is the default. An RTX 2070 (8 GB) is present, but CUDA training requires a
compatible CUDA-enabled PyTorch installation. Check before using --device 0:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

For a GPU build, follow https://pytorch.org/get-started/locally/.

## Evaluate and predict

```powershell
.\.venv\Scripts\python.exe xray.py evaluate --weights runs/classify/chest_xray/weights/best.pt
.\.venv\Scripts\python.exe xray.py predict --weights runs/classify/chest_xray/weights/best.pt --image data/example.png
```

Evaluate only after selecting settings with validation data. Evaluation explicitly
uses the prepared test split. Prediction applies square padding and uses the saved
training input size, or an explicit --size. Outputs are three class scores, not
calibrated clinical probabilities. There are no abnormality boxes or multi-label outputs.

## Checks and limitations

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests cover missing patient IDs, group leakage, identical decoded pixels with
different file bytes, changed images, and full-image padding. Audit and preparation
metadata are stored under data/; smoke artifacts are under runs/classify/.

Verified locally: six tests passed, all 13,959 images prepared successfully, and a
one-epoch smoke run saved/reloaded its three-class checkpoint. Single-image prediction
and the custom validator passed using validation images only. No held-out test metrics
or full 50-epoch results have been produced. The installed torch build is CPU-only.

Patient independence is not verified. Near-duplicates and patients with unrelated
filenames may remain; source/population confounding is unresolved. Identical images
within one split remain included and can overweight some examples in metrics.
Report per-class performance, calibration, uncertainty, and external-site evaluation
before drawing conclusions; aggregate accuracy alone is insufficient.

Data, audit metadata, environments and checkpoints are ignored by version control.
Review dataset and Ultralytics AGPL-3.0/enterprise licensing before distribution.

References:
- https://docs.ultralytics.com/tasks/classify/
- https://docs.ultralytics.com/datasets/classify/
- https://www.kaggle.com/datasets/rifatulmajumder23/combined-unknown-pneumonia-and-tuberculosis
