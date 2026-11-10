"""Research-only chest X-ray classification CLI. See README.md before use."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

CLASSES = {"normal", "pneumonia", "tuberculosis", "covid19"}
SPLITS = {"train", "val", "test"}


def read_manifest(path):
    """Validate all records before writing anything; keep patients and duplicates together."""
    from PIL import Image, ImageOps

    path = Path(path).resolve()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"path", "label", "patient_id", "source", "split"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest requires columns: {sorted(required)}")
        records = list(reader)
    if not records:
        raise ValueError("Manifest is empty")
    patients, hashes, paths = {}, {}, set()
    for line, row in enumerate(records, 2):
        row.update({key: row[key].strip() for key in required})
        if any(not row[key] for key in required):
            raise ValueError(f"Line {line}: required values cannot be empty")
        if row["label"] not in CLASSES or row["split"] not in SPLITS:
            raise ValueError(f"Line {line}: unsupported label or split")
        image_path = (path.parent / row["path"]).resolve()
        if image_path in paths:
            raise ValueError(f"Line {line}: repeated image path: {image_path}")
        paths.add(image_path)
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            raise ValueError(f"Line {line}: convert images to PNG/JPEG first")
        # IDs must be globally consistent even when patients occur in multiple sources.
        patient = row["patient_id"]
        if patients.setdefault(patient, row["split"]) != row["split"]:
            raise ValueError(f"Line {line}: patient appears in multiple splits")
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            digest = hashlib.sha256(
                str(image.size).encode() + image.tobytes()
            ).hexdigest()
        previous = hashes.setdefault(digest, (row["split"], row["label"]))
        if previous != (row["split"], row["label"]):
            raise ValueError(f"Line {line}: duplicate pixels cross splits or labels")
        row["resolved_path"] = str(image_path)
        row["sha256"] = digest
    counts = Counter((row["split"], row["label"]) for row in records)
    missing = [(s, c) for s in sorted(SPLITS) for c in sorted(CLASSES) if not counts[s, c]]
    if missing:
        raise ValueError(f"Every split must include every class; missing: {missing}")
    return records


def square_image(image, size):
    from PIL import Image, ImageOps

    image = ImageOps.exif_transpose(image).convert("RGB")
    return ImageOps.pad(image, (size, size), method=Image.Resampling.LANCZOS, color=0)


def prepare(args):
    from PIL import Image

    records = read_manifest(args.manifest)
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError("Output already exists; choose a new directory to avoid stale images")
    output.mkdir(parents=True)
    for index, row in enumerate(records):
        destination = output / row["split"] / row["label"] / f"{index:07d}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(row["resolved_path"]) as image:
            square_image(image, args.size).save(destination)
    # This local audit contains paths and patient IDs: do not publish it.
    (output / "audit.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (output / "preprocessing.json").write_text(json.dumps({"size": args.size}), encoding="utf-8")
    print(f"Prepared {len(records)} images in {output}")


def dataset_size(data):
    data = Path(data).resolve()
    if not (data / "audit.json").is_file():
        raise ValueError("Use the prepare command before training or evaluation")
    return data, json.loads((data / "preprocessing.json").read_text())["size"]


def train(args):
    from ultralytics import YOLO

    data, size = dataset_size(args.data)
    model = YOLO(args.model)
    if model.task != "classify":
        raise ValueError("Choose a classification checkpoint ending in -cls.pt")
    model.train(
        data=str(data), epochs=args.epochs, imgsz=size, batch=args.batch,
        device=args.device, workers=args.workers, seed=42, deterministic=True,
        project="runs/classify", name=args.name, patience=15,
        fliplr=0.0, flipud=0.0, degrees=0.0, translate=0.0,
        scale=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        auto_augment=None, erasing=0.0,
    )


def evaluate(args):
    from ultralytics import YOLO

    data, size = dataset_size(args.data)
    # Test is explicit, never used to choose training hyperparameters.
    model = YOLO(args.weights)
    results = model.val(data=str(data), split="test", imgsz=size, device=args.device,
                        workers=0, plots=True)
    print(json.dumps(results.results_dict, indent=2))


def predict(args):
    from PIL import Image
    from ultralytics import YOLO

    model = YOLO(args.weights)
    if model.task != "classify" or set(model.names.values()) != CLASSES:
        raise ValueError("Weights must be a trained four-class chest X-ray classifier")
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
    prep.add_argument("--manifest", required=True)
    prep.add_argument("--output", default="data/prepared")
    prep.add_argument("--size", type=int, default=320)
    prep.set_defaults(run=prepare)
    fit = commands.add_parser("train")
    fit.add_argument("--data", default="data/prepared")
    fit.add_argument("--model", default="yolo26n-cls.pt")
    fit.add_argument("--epochs", type=int, default=50)
    fit.add_argument("--batch", type=int, default=16)
    fit.add_argument("--workers", type=int, default=0)
    fit.add_argument("--name", default="chest_xray")
    fit.set_defaults(run=train)
    val = commands.add_parser("evaluate")
    val.add_argument("--data", default="data/prepared")
    val.add_argument("--weights", required=True)
    val.set_defaults(run=evaluate)
    infer = commands.add_parser("predict")
    infer.add_argument("--weights", required=True)
    infer.add_argument("--image", required=True)
    infer.add_argument("--size", type=int)
    infer.set_defaults(run=predict)
    for command in (fit, val, infer):
        command.add_argument("--device", default="cpu", help="cpu or GPU index, e.g. 0")
    args = parser.parse_args()
    if getattr(args, "size", None) is not None and args.size <= 0:
        parser.error("--size must be positive")
    try:
        args.run(args)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
