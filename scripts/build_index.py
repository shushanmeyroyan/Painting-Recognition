#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from art_recognition.identity import DEFAULT_DINOV2_MODEL
from art_recognition.pipeline import ArtRecognitionPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the painting recognition index.")
    parser.add_argument("--project-root", default=".", help="Project root containing data/datasets/armenian and the codebase")
    parser.add_argument("--embedding-model", default=DEFAULT_DINOV2_MODEL, help="DINOv2 model name")
    parser.add_argument("--augmentations", type=int, default=16, help="Embeddings to store per Armenian painting")
    parser.add_argument("--include-wikiart", action="store_true", help="Also include WikiArt records in the identity index")
    parser.add_argument("--wikiart-limit", type=int, default=4500, help="Number of WikiArt images to include when enabled")
    parser.add_argument("--wikiart-metadata-path", default="", help="Optional metadata file path inside the Kaggle dataset")
    parser.add_argument("--progress-interval", type=int, default=100, help="Print progress every N embeddings")
    args = parser.parse_args()

    pipeline = ArtRecognitionPipeline(project_root=Path(args.project_root))
    summary = pipeline.build_index(
        embedding_model=args.embedding_model,
        augmentations_per_painting=args.augmentations,
        include_wikiart=args.include_wikiart,
        wikiart_sample_size=args.wikiart_limit,
        wikiart_metadata_path=args.wikiart_metadata_path,
        progress_interval=args.progress_interval,
    )
    print(json.dumps(summary.__dict__, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
