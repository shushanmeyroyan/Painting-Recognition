#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from art_recognition.pipeline import ArtRecognitionPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the painting recognition index.")
    parser.add_argument("--project-root", default=".", help="Project root containing data/datasets/armenian and the codebase")
    parser.add_argument("--wikiart-limit", type=int, default=4500, help="Number of WikiArt images to include")
    parser.add_argument("--wikiart-metadata-path", default="", help="Optional metadata file path inside the Kaggle dataset")
    parser.add_argument("--embedding-model", choices=["clip", "resnet50"], default="clip")
    parser.add_argument("--skip-wikiart", action="store_true", help="Build an index using only local Armenian paintings")
    args = parser.parse_args()

    pipeline = ArtRecognitionPipeline(project_root=Path(args.project_root))
    summary = pipeline.build_index(
        wikiart_sample_size=args.wikiart_limit,
        wikiart_metadata_path=args.wikiart_metadata_path,
        embedding_model=args.embedding_model,
        include_wikiart=not args.skip_wikiart,
    )
    print(json.dumps(summary.__dict__, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
