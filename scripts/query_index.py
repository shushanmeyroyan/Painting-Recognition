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
    parser = argparse.ArgumentParser(description="Query the Armenian painting identity index.")
    parser.add_argument("image_path", help="Path to the query image")
    parser.add_argument("--project-root", default=".", help="Project root containing the built data directory")
    parser.add_argument("--embedding-model", default=DEFAULT_DINOV2_MODEL, help="DINOv2 model name")
    parser.add_argument("--top-k", type=int, default=20, help="Number of augmented embeddings to retrieve")
    args = parser.parse_args()

    pipeline = ArtRecognitionPipeline(project_root=Path(args.project_root))
    result = pipeline.query(
        args.image_path,
        embedding_model=args.embedding_model,
        top_k=args.top_k,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
