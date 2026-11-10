import csv
import argparse
import hashlib
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from xray import CLASSES, SPLITS, prepare, dataset_size, read_manifest, square_image


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / "manifest.csv"
        self.rows = []
        for split in sorted(SPLITS):
            for label in sorted(CLASSES):
                index = len(self.rows)
                path = self.root / f"{index}.png"
                Image.new("RGB", (4, 6), (index * 20, 10, 30)).save(path)
                self.rows.append(dict(
                    path=path.name, label=label, split=split, source="fixture",
                    patient_id="", patient_id_status="unavailable",
                    group_id=f"group-{index}", sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                ))

    def save(self):
        with self.manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.rows[0].keys())
            writer.writeheader()
            writer.writerows(self.rows)

    def test_missing_patient_ids_remain_missing(self):
        self.save()
        records = read_manifest(self.manifest)
        self.assertEqual(len(records), 9)
        self.assertTrue(all(row["patient_id"] == "" for row in records))

    def test_group_crossing_splits_rejected(self):
        self.rows[3]["group_id"] = self.rows[0]["group_id"]
        self.save()
        with self.assertRaisesRegex(ValueError, "connected group"):
            read_manifest(self.manifest)

    def test_pixel_duplicate_with_different_bytes_rejected(self):
        source = self.root / self.rows[0]["path"]
        target = self.root / self.rows[3]["path"]
        with Image.open(source) as image:
            image.save(target, compress_level=0)
        self.rows[3]["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
        self.save()
        with self.assertRaisesRegex(ValueError, "duplicate pixels"):
            read_manifest(self.manifest)

    def test_changed_file_rejected(self):
        (self.root / self.rows[0]["path"]).write_bytes(b"changed")
        self.save()
        with self.assertRaisesRegex(ValueError, "changed since split audit"):
            read_manifest(self.manifest)

    def test_letterbox_preserves_full_image(self):
        result = square_image(Image.new("RGB", (4, 8), "white"), 8)
        self.assertEqual(result.size, (8, 8))
        self.assertEqual(result.getpixel((0, 4)), (0, 0, 0))
        self.assertEqual(result.getpixel((4, 0)), (255, 255, 255))
        self.assertEqual(result.getpixel((4, 7)), (255, 255, 255))

    def test_preparation_uses_assignments_and_preserves_metadata(self):
        self.save()
        output = self.root / "prepared"
        prepare(argparse.Namespace(manifest=str(self.manifest), output=str(output), size=32))
        self.assertEqual(dataset_size(output), (output, 32))
        self.assertEqual(len(list(output.glob("*/*/*.png"))), 9)
        self.assertTrue((output / "test" / "normal" / "0000000.png").is_file())
        self.assertTrue((self.root / "0.png").is_file())
        self.assertIn('"patient_id_status": "unavailable"', (output / "audit.json").read_text())
        self.assertFalse(output.with_name("prepared.incomplete").exists())
        with self.assertRaisesRegex(ValueError, "already exists"):
            prepare(argparse.Namespace(manifest=str(self.manifest), output=str(output), size=32))


if __name__ == "__main__":
    unittest.main()
