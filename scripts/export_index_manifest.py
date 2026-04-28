#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _unique_paintings(mapping: list[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[object] = set()
    rows: list[dict[str, object]] = []
    for item in mapping:
        painting_id = item.get("painting_id")
        if painting_id in seen:
            continue
        seen.add(painting_id)
        rows.append(item)
    return rows


def _write_manifest(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "painting_id",
        "source",
        "filename",
        "painter_name",
        "painting_name",
        "year",
        "style",
        "image_path",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _copy_samples(rows: list[dict[str, object]], samples_dir: Path, per_source: int) -> None:
    if per_source <= 0:
        return
    samples_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for row in rows:
        source = str(row.get("source") or "unknown")
        if counts.get(source, 0) >= per_source:
            continue
        image_path = Path(str(row.get("image_path") or ""))
        if not image_path.exists():
            continue
        target_dir = samples_dir / source
        target_dir.mkdir(parents=True, exist_ok=True)
        target_name = f"{int(row.get('painting_id') or 0):04d}_{image_path.name}"
        shutil.copy2(image_path, target_dir / target_name)
        counts[source] = counts.get(source, 0) + 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a CSV manifest and sample images from the built identity index.")
    parser.add_argument("--mapping", default="data/index_mapping.json")
    parser.add_argument("--output", default="data/index_manifest.csv")
    parser.add_argument("--samples-dir", default="data/test_samples/indexed")
    parser.add_argument("--samples-per-source", type=int, default=20)
    args = parser.parse_args()

    mapping_path = Path(args.mapping)
    if not mapping_path.exists():
        raise SystemExit(f"Mapping not found: {mapping_path}. Build the index first.")

    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    rows = _unique_paintings(mapping)
    _write_manifest(rows, Path(args.output))
    _copy_samples(rows, Path(args.samples_dir), args.samples_per_source)

    counts: dict[str, int] = {}
    for row in rows:
        source = str(row.get("source") or "unknown")
        counts[source] = counts.get(source, 0) + 1
    print(f"Wrote manifest: {args.output}")
    print(f"Copied samples to: {args.samples_dir}")
    print(f"Unique indexed paintings: {len(rows)}")
    print("Sources:", counts)


if __name__ == "__main__":
    main()
