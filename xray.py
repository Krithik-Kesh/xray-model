"""Research-only chest X-ray classification CLI. See README.md before use."""

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CLASSES = {"normal", "pneumonia", "tuberculosis"}
SPLITS = {"train", "val", "test"}


def read_manifest(path):
    """Validate all records before writing anything; keep patients and duplicates together."""
    from PIL import Image, ImageOps

    path = Path(path).resolve()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"path", "label", "patient_id", "patient_id_status", "source", "split", "group_id", "sha256"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest requires columns: {sorted(required)}")
        records = list(reader)
    if not records:
        raise ValueError("Manifest is empty")
    patients, groups, hashes, paths = {}, {}, {}, set()
    for line, row in enumerate(records, 2):
        row.update({key: row[key].strip() for key in required})
        if any(not row[key] for key in required - {"patient_id"}):
            raise ValueError(f"Line {line}: required values cannot be empty")
        if row["label"] not in CLASSES or row["split"] not in SPLITS:
            raise ValueError(f"Line {line}: unsupported label or split")
        image_path = (path.parent / row["path"]).resolve()
        if image_path in paths:
            raise ValueError(f"Line {line}: repeated image path: {image_path}")
        paths.add(image_path)
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            raise ValueError(f"Line {line}: convert images to PNG/JPEG first")
        if row["patient_id_status"] not in {"filename_inferred_unverified", "unavailable", "verified"}:
            raise ValueError(f"Line {line}: invalid patient_id_status")
        if bool(row["patient_id"]) != (row["patient_id_status"] != "unavailable"):
            raise ValueError(f"Line {line}: patient ID and status disagree")
        if groups.setdefault(row["group_id"], row["split"]) != row["split"]:
            raise ValueError(f"Line {line}: connected group appears in multiple splits")
        patient = row["patient_id"]
        if patient and patients.setdefault(patient, row["split"]) != row["split"]:
            raise ValueError(f"Line {line}: patient appears in multiple splits")
        row["resolved_path"] = str(image_path)
    counts = Counter((row["split"], row["label"]) for row in records)
    missing = [(s, c) for s in sorted(SPLITS) for c in sorted(CLASSES) if not counts[s, c]]
    if missing:
        raise ValueError(f"Every split must include every class; missing: {missing}")

    def inspect(row):
        image_path = Path(row["resolved_path"])
        with image_path.open("rb") as handle:
            actual_hash = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual_hash.lower() != row["sha256"].lower():
            raise ValueError(f"Image changed since split audit: {image_path}")
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()

    with ThreadPoolExecutor(max_workers=4) as executor:
        for index, (row, digest) in enumerate(zip(records, executor.map(inspect, records)), 1):
            previous = hashes.setdefault(digest, (row["split"], row["label"]))
            if previous != (row["split"], row["label"]):
                raise ValueError(f"Line {index + 1}: duplicate pixels cross splits or labels")
            row["pixel_sha256"] = digest
            if index % 500 == 0:
                print(f"Validated {index}/{len(records)} images", flush=True)
    return records


def square_image(image, size):
    from PIL import Image, ImageOps

    image = ImageOps.exif_transpose(image).convert("RGB")
    return ImageOps.pad(image, (size, size), method=Image.Resampling.LANCZOS, color=0)


def prepare(args):
    from PIL import Image

    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError("Output already exists; choose a new directory to avoid stale images")
    records = read_manifest(args.manifest)
    staging = output.with_name(output.name + ".incomplete")
    if staging.exists():
        raise ValueError(f"Previous partial preparation exists: {staging}; inspect it or choose a new output")
    staging.mkdir(parents=True)
    def write_image(item):
        index, row = item
        destination = staging / row["split"] / row["label"] / f"{index:07d}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(row["resolved_path"]) as image:
            square_image(image, args.size).save(destination)
        row["prepared_path"] = str(destination.relative_to(staging))
    with ThreadPoolExecutor(max_workers=4) as executor:
        for index, _ in enumerate(executor.map(write_image, enumerate(records)), 1):
            if index % 500 == 0:
                print(f"Prepared {index}/{len(records)} images", flush=True)
    # This local audit contains paths and patient IDs: do not publish it.
    (staging / "audit.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (staging / "preprocessing.json").write_text(json.dumps({"size": args.size, "classes": sorted(CLASSES)}), encoding="utf-8")
    shutil.copy2(args.manifest, staging / "source_manifest.csv")
    staging.rename(output)
    print(f"Prepared {len(records)} images in {output}")


def dataset_size(data):
    data = Path(data).resolve()
    if not (data / "audit.json").is_file():
        raise ValueError("Use the prepare command before training or evaluation")
    config = json.loads((data / "preprocessing.json").read_text())
    if set(config.get("classes", [])) != CLASSES:
        raise ValueError("Prepared data must contain exactly the three supported classes")
    for split in SPLITS:
        folders = {p.name for p in (data / split).iterdir() if p.is_dir()}
        if folders != CLASSES:
            raise ValueError(f"Unexpected class folders in {split}: {folders}")
    return data, config["size"]


def train(args):
    data, size = dataset_size(args.data)
    run_training(args, data, size)


def run_training(args, data, size):
    from ultralytics import YOLO
    from xray_transforms import FullImageTrainer

    model = YOLO(args.model)
    if model.task != "classify":
        raise ValueError("Choose a classification checkpoint ending in -cls.pt")
    model.train(
        trainer=FullImageTrainer, data=str(data), epochs=args.epochs, imgsz=size, batch=args.batch,
        device=args.device, workers=args.workers, seed=42, deterministic=True,
        project=str(Path("runs/classify").resolve()), name=args.name, patience=15,
        fliplr=0.0, flipud=0.0, degrees=0.0, translate=0.0,
        scale=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        auto_augment=None, erasing=0.0,
    )


def smoke(args):
    """Exercise training on a small train/val subset; never read held-out test images."""
    data, _ = dataset_size(args.data)
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError(f"Smoke subset exists: {output}; choose a new --output directory")
    records = json.loads((data / "audit.json").read_text())
    selected = []
    for split, limit in (("train", 12), ("val", 6)):
        for label in sorted(CLASSES):
            seen = set()
            for row in records:
                if row["split"] != split or row["label"] != label or row["group_id"] in seen:
                    continue
                seen.add(row["group_id"])
                target = output / row["prepared_path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(data / row["prepared_path"], target)
                selected.append(row)
                if len(seen) == limit:
                    break
            if len(seen) < limit:
                raise ValueError(f"Not enough unique groups in {split}/{label} for smoke test")
    (output / "audit.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    args.epochs = 1
    run_training(args, output, 64)


def evaluate(args):
    from ultralytics import YOLO
    from xray_transforms import FullImageValidator

    data, size = dataset_size(args.data)
    # Test is explicit, never used to choose training hyperparameters.
    model = YOLO(args.weights)
    if model.task != "classify" or set(model.names.values()) != CLASSES:
        raise ValueError("Weights must be a trained three-class chest X-ray classifier")
    results = model.val(data=str(data), split="test", imgsz=size, device=args.device,
                        workers=0, plots=True, validator=FullImageValidator)
    print(json.dumps(results.results_dict, indent=2))


def predict(args):
    from PIL import Image
    from ultralytics import YOLO

    model = YOLO(args.weights)
    if model.task != "classify" or set(model.names.values()) != CLASSES:
        raise ValueError("Weights must be a trained three-class chest X-ray classifier")
    # Match the stored training input size unless an explicit size is supplied.
    size = args.size or model.overrides.get("imgsz", 320)
    with Image.open(args.image) as image:
        prepared = square_image(image, size)
    result = model.predict(prepared, imgsz=size, device=args.device, verbose=False)[0]
    scores = {result.names[i]: float(score) for i, score in enumerate(result.probs.data.cpu())}
    print(json.dumps({"research_only": True, "predicted_class": max(scores, key=scores.get),
                      "scores": scores, "note": "Scores are not calibrated disease probabilities."}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Validate manifest and create square PNG inputs")
    prep.add_argument("--manifest", default="data/splits/manifest_pixels.csv")
    prep.add_argument("--output", default="data/prepared_three_class")
    prep.add_argument("--size", type=int, default=320)
    prep.set_defaults(run=prepare)
    fit = commands.add_parser("train")
    fit.add_argument("--data", default="data/prepared_three_class")
    fit.add_argument("--model", default="yolo26n-cls.pt")
    fit.add_argument("--epochs", type=int, default=50)
    fit.add_argument("--batch", type=int, default=16)
    fit.add_argument("--workers", type=int, default=0)
    fit.add_argument("--name", default="chest_xray")
    fit.set_defaults(run=train)
    check = commands.add_parser("smoke", help="One epoch on 36 training / 18 validation images at 64 pixels")
    check.add_argument("--data", default="data/prepared_three_class")
    check.add_argument("--output", default="data/smoke_three_class")
    check.add_argument("--model", default="yolo26n-cls.pt")
    check.add_argument("--batch", type=int, default=6)
    check.add_argument("--workers", type=int, default=0)
    check.add_argument("--name", default="smoke_three_class")
    check.set_defaults(run=smoke)
    val = commands.add_parser("evaluate")
    val.add_argument("--data", default="data/prepared_three_class")
    val.add_argument("--weights", required=True)
    val.set_defaults(run=evaluate)
    infer = commands.add_parser("predict")
    infer.add_argument("--weights", required=True)
    infer.add_argument("--image", required=True)
    infer.add_argument("--size", type=int)
    infer.set_defaults(run=predict)
    for command in (fit, check, val, infer):
        command.add_argument("--device", default="cpu", help="cpu or GPU index, e.g. 0")
    args = parser.parse_args()
    # Keep runtime caches in the project instead of unwritable user directories.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(".cache/ultralytics").resolve()))
    if getattr(args, "size", None) is not None and args.size <= 0:
        parser.error("--size must be positive")
    try:
        args.run(args)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
