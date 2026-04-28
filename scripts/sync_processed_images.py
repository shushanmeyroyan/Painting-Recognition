#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from art_recognition.config import ProjectPaths


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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


def _source_root(paths: ProjectPaths, source: str) -> Path:
    if source == "wikiart":
        return paths.wikiart_raw_dir
    if source == "armenian_local":
        return paths.armenian_images_dir
    return paths.datasets_dir


def _relative_source_path(project_root: Path, paths: ProjectPaths, row: dict[str, object]) -> tuple[Path, Path] | None:
    source = str(row.get("source") or "unknown")
    image_path = Path(str(row.get("image_path") or ""))
    if not image_path.is_absolute():
        image_path = project_root / image_path
    if not image_path.exists() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        return None

    root = _source_root(paths, source)
    try:
        relative_path = image_path.relative_to(root)
    except ValueError:
        relative_path = Path(image_path.name)

    return image_path, relative_path


def _clear_output(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _write_manifest(copied_rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "painting_id",
        "source",
        "filename",
        "painting_name",
        "painter_name",
        "style",
        "source_path",
        "processed_path",
        "dataset_path",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(copied_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy every unique painting in the current identity index into data/processed_images."
    )
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--mapping", default="data/index_mapping.json", help="Index mapping JSON path")
    parser.add_argument(
        "--source",
        choices=["all", "wikiart", "armenian_local"],
        default="all",
        help="Which indexed source to sync",
    )
    parser.add_argument(
        "--manifest",
        default="data/processed_images/processed_manifest.csv",
        help="CSV showing source image to processed image mapping",
    )
    parser.add_argument(
        "--dataset-copy",
        default="",
        help="Optional folder to also populate with the same indexed files, for example data/datasets/wikiart",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the selected output folder before copying so it exactly matches the current index",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    paths = ProjectPaths(project_root)
    mapping_path = project_root / args.mapping
    if not mapping_path.exists():
        raise SystemExit(f"Mapping not found: {mapping_path}. Build the index first.")

    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    rows = _unique_paintings(mapping)
    if args.source != "all":
        rows = [row for row in rows if row.get("source") == args.source]

    processed_roots = {str(row.get("source") or "unknown") for row in rows}
    if args.clean:
        for source in processed_roots:
            _clear_output(paths.processed_dir / source)
        if args.dataset_copy:
            _clear_output(project_root / args.dataset_copy)

    dataset_copy_root = project_root / args.dataset_copy if args.dataset_copy else None

    copied = 0
    dataset_copied = 0
    skipped = 0
    manifest_rows: list[dict[str, object]] = []
    for row in rows:
        source = str(row.get("source") or "unknown")
        source_info = _relative_source_path(project_root, paths, row)
        if source_info is None:
            skipped += 1
            continue
        image_path, relative_path = source_info
        target_path = paths.processed_dir / source / relative_path

        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, target_path)
        copied += 1
        dataset_path = ""
        if dataset_copy_root is not None:
            dataset_target = dataset_copy_root / relative_path
            dataset_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, dataset_target)
            dataset_path = str(dataset_target)
            dataset_copied += 1
        manifest_rows.append(
            {
                "painting_id": row.get("painting_id"),
                "source": row.get("source"),
                "filename": row.get("filename"),
                "painting_name": row.get("painting_name"),
                "painter_name": row.get("painter_name"),
                "style": row.get("style"),
                "source_path": str(image_path),
                "processed_path": str(target_path),
                "dataset_path": dataset_path,
            }
        )

    _write_manifest(manifest_rows, project_root / args.manifest)

    print(f"Synced processed images: {copied}")
    if dataset_copy_root is not None:
        print(f"Synced dataset copy: {dataset_copied}")
    print(f"Skipped missing/unreadable paths: {skipped}")
    print(f"Manifest: {args.manifest}")
    print(f"Output folder: {paths.processed_dir}")


if __name__ == "__main__":
    main()
