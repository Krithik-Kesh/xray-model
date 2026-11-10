"""Refine existing split groups using decoded-pixel identity; preserve input CSV."""

import csv
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps


def main():
    manifest = Path("data/splits/manifest.csv").resolve()
    output = manifest.with_name("manifest_pixels.csv")
    if output.exists():
        raise ValueError(f"Refusing to overwrite {output}")
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    def inspect(row):
        path = manifest.parent / row["path"]
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest.lower() != row["sha256"].lower():
            raise ValueError(f"File changed since audit: {path}")
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()

    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def join(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    owners = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        for index, digest in enumerate(executor.map(inspect, rows)):
            row = rows[index]
            row["pixel_sha256"] = digest
            for key in (("group", row["group_id"]), ("pixels", digest)):
                if key in owners:
                    join(index, owners[key])
                else:
                    owners[key] = index
            if (index + 1) % 500 == 0:
                print(f"Checked pixels: {index + 1}/{len(rows)}", flush=True)
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault(find(index), []).append(row)
    conflicts, changes = [], []
    for members in groups.values():
        if len({row["label"] for row in members}) > 1:
            conflicts.append(members)
            continue
        counts = Counter(row["split"] for row in members)
        chosen = max(("train", "val", "test"), key=lambda s: counts[s])
        group_id = hashlib.sha256("|".join(sorted({r["group_id"] for r in members})).encode()).hexdigest()
        for row in members:
            row["previous_split"] = row["split"]
            if row["split"] != chosen:
                changes.append({"path": row["path"], "from": row["split"], "to": chosen})
            row["split"] = chosen
            row["group_id"] = group_id
    if conflicts:
        output.with_name("pixel_label_conflicts.json").write_text(json.dumps(conflicts, indent=2))
        raise ValueError("Conflicting labels in connected pixel groups; review pixel_label_conflicts.json")
    for field in ("group_id", "pixel_sha256", "sha256", "patient_id"):
        seen = {}
        for row in rows:
            if row[field] and seen.setdefault(row[field], row["split"]) != row["split"]:
                raise ValueError(f"Overlap remains in {field}")
    for split in ("train", "val", "test"):
        if {r["label"] for r in rows if r["split"] == split} != {"normal", "pneumonia", "tuberculosis"}:
            raise ValueError(f"Class coverage missing in {split}")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    report = {"images": len(rows), "changed_assignments": len(changes), "changes": changes,
              "counts": {f"{s}/{c}": n for (s, c), n in sorted(Counter((r["split"], r["label"]) for r in rows).items())},
              "note": "Connected groups preserved; cross-split exact-pixel overlap removed. Patient IDs remain unverified."}
    output.with_name("pixel_repair.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "changes"}, indent=2))


if __name__ == "__main__":
    main()
