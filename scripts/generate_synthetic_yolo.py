#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from art_recognition.datasets import load_armenian_records, load_wikiart_records
from art_recognition.synthetic_yolo import write_yolo_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic wall/frame images for YOLO segmentation training.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-dir", default="data/synthetic_yolo_paintings")
    parser.add_argument("--samples-per-image", type=int, default=12)
    parser.add_argument("--include-wikiart", action="store_true", help="Also synthesize YOLO training images from WikiArt")
    parser.add_argument("--wikiart-limit", type=int, default=1000, help="Number of WikiArt images to include when enabled")
    parser.add_argument("--wikiart-metadata-path", default="", help="Optional metadata file path inside the Kaggle dataset")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_root = Path(args.project_root)
    records = load_armenian_records(project_root)
    if args.include_wikiart:
        records.extend(
            load_wikiart_records(
                sample_size=args.wikiart_limit,
                metadata_path=args.wikiart_metadata_path,
                output_dir=project_root / "data" / "datasets" / "wikiart_raw",
            )
        )
    image_paths = [Path(record.image_path) for record in records]
    write_yolo_dataset(
        image_paths=image_paths,
        output_dir=project_root / args.output_dir,
        samples_per_image=args.samples_per_image,
        seed=args.seed,
    )
    print(f"Wrote YOLO dataset to {project_root / args.output_dir}")


if __name__ == "__main__":
    main()
