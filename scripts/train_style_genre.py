#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from art_recognition.config import ProjectPaths
from art_recognition.database import ArtVectorDatabase
from art_recognition.style_genre import StyleGenrePredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DINOv2 style and genre classifiers from indexed WikiArt embeddings.")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--source", choices=["wikiart", "all"], default="wikiart", help="Indexed source to train from")
    parser.add_argument("--output", default="", help="Output classifier path")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    paths = ProjectPaths(project_root)
    output_path = Path(args.output) if args.output else paths.style_genre_classifier_path

    vector_db = ArtVectorDatabase(
        index_path=paths.faiss_index_path,
        mapping_path=paths.mapping_path,
        embeddings_path=paths.embeddings_path,
    ).load()
    embeddings = np.asarray(vector_db.load_embeddings(), dtype=np.float32)
    predictor = StyleGenrePredictor.fit_from_index(embeddings, vector_db.mapping, source=args.source)
    predictor.save(output_path)

    print(
        json.dumps(
            {
                "output": str(output_path),
                **predictor.metadata,
                "style_classes": len(predictor.style_model.classes_),
                "genre_classes": len(predictor.genre_model.classes_),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
