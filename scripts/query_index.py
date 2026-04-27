#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from art_recognition.database import ArtVectorDatabase
from art_recognition.ml_models import (
    ClipZeroShotStylePredictor,
    EmbeddingExtractor,
    StyleClassifier,
    predict_style_with_fallback,
)
from art_recognition.pipeline import build_query_response, preprocess_query_image_variants


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the painting recognition index.")
    parser.add_argument("image_path", help="Path to the query image")
    parser.add_argument("--project-root", default=".", help="Project root containing the built data directory")
    parser.add_argument("--embedding-model", choices=["clip", "resnet50"], default="clip")
    parser.add_argument("--top-k", type=int, default=3, help="Number of similar paintings to return")
    args = parser.parse_args()

    project_root = Path(args.project_root)
    extractor = EmbeddingExtractor(model_name=args.embedding_model)
    classifier_path = project_root / "data" / "style_classifier.pkl"
    classifier = None
    if classifier_path.exists():
        classifier = StyleClassifier.load(classifier_path)

    vector_db = ArtVectorDatabase(
        index_path=project_root / "data" / "faiss_index.idx",
        mapping_path=project_root / "data" / "index_mapping.json",
        embeddings_path=project_root / "data" / "embeddings.npy",
    ).load()
    indexed_model = str(vector_db.mapping[0].get("embedding_model") or "").lower() if vector_db.mapping else ""
    if indexed_model and indexed_model != args.embedding_model.lower():
        raise RuntimeError(
            f"The saved index was built with {indexed_model}. Rebuild it with {args.embedding_model} before querying."
        )
    zero_shot = None
    if extractor.model_name == "clip":
        labels = classifier.classes_ if classifier is not None else sorted(
            {str(record.get("style")) for record in vector_db.mapping if record.get("style")}
        )
        if labels:
            zero_shot = ClipZeroShotStylePredictor(extractor, labels)

    best_result = None
    for variant in preprocess_query_image_variants(args.image_path):
        query_embedding = extractor.extract(variant["processed_rgb"])
        predicted_style, predicted_style_confidence, predicted_style_source = predict_style_with_fallback(
            query_embedding,
            classifier,
            zero_shot,
        )

        matches = vector_db.export_matches_with_numpy(query_embedding, k=args.top_k)
        result = build_query_response(
            image_path=args.image_path,
            matches=matches,
            predicted_style=predicted_style,
            predicted_style_confidence=predicted_style_confidence,
            predicted_style_source=predicted_style_source,
        )
        result["query_variant"] = variant["name"]
        if result["is_recognized"]:
            best_result = result
            break
        if best_result is None or result["recognition_score"] > best_result["recognition_score"]:
            best_result = result

    result = best_result or build_query_response(image_path=args.image_path, matches=[])
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
