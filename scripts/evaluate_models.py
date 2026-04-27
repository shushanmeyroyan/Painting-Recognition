#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

from art_recognition.database import ArtVectorDatabase
from art_recognition.ml_models import (
    ClipZeroShotStylePredictor,
    EmbeddingExtractor,
    StyleClassifier,
    predict_style_with_fallback,
)
from art_recognition.pipeline import RECOGNITION_SCORE_THRESHOLD, preprocess_query_image_variants_from_bgr


THRESHOLD_CANDIDATES = [0.80, 0.84, 0.86, 0.90, 0.94, 0.96, 0.98]


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _stable_split(count: int, query_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(count)
    rng.shuffle(indices)
    query_count = max(1, int(round(count * query_fraction)))
    query_indices = np.sort(indices[:query_count])
    index_indices = np.sort(indices[query_count:])
    return index_indices, query_indices


def _sample_indices(indices: np.ndarray, max_count: int | None, seed: int) -> np.ndarray:
    if max_count is None or len(indices) <= max_count:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=max_count, replace=False))


def _record_key(record: dict[str, object]) -> str:
    return str(record.get("image_path") or record.get("filename") or "")


def _query_embeddings_for_record(
    record: dict[str, object],
    fallback_embedding: np.ndarray,
    extractor: EmbeddingExtractor | None,
) -> list[tuple[str, np.ndarray]]:
    if extractor is None:
        return [("stored_embedding", fallback_embedding)]

    image_path = record.get("image_path")
    if not image_path:
        return [("stored_embedding", fallback_embedding)]

    image = cv2.imread(str(image_path))
    if image is None:
        return [("stored_embedding", fallback_embedding)]

    variants = preprocess_query_image_variants_from_bgr(image)
    if not variants:
        return [("stored_embedding", fallback_embedding)]
    return [
        (str(variant.get("name") or "unknown_variant"), extractor.extract(variant["processed_rgb"]))
        for variant in variants
    ]


def _top_matches(scores: np.ndarray, mapping: list[dict[str, object]], top_k: int) -> list[dict[str, object]]:
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [
        {
            "rank": rank,
            "score": float(scores[int(index)]),
            "image_path": _record_key(mapping[int(index)]),
            "title": mapping[int(index)].get("title"),
            "artist": mapping[int(index)].get("artist"),
            "style": mapping[int(index)].get("style") or mapping[int(index)].get("predicted_style"),
        }
        for rank, index in enumerate(top_indices, start=1)
    ]


def _topk_accuracy(
    embeddings: np.ndarray,
    mapping: list[dict[str, object]],
    query_indices: np.ndarray,
    top_k: int,
    extractor: EmbeddingExtractor | None = None,
) -> dict[str, object]:
    top1_correct = 0
    top5_correct = 0
    positive_scores: list[float] = []
    best_negative_scores: list[float] = []
    examples = []
    failures = []
    threshold_rejections = []

    for query_index in query_indices:
        correct_key = _record_key(mapping[int(query_index)])
        best_variant_name = "unknown"
        best_scores = None
        best_ranked = None
        best_correct_rank = None

        for variant_name, query_embedding in _query_embeddings_for_record(
            mapping[int(query_index)],
            embeddings[int(query_index)],
            extractor,
        ):
            query_embedding = query_embedding.astype(np.float32)
            norm = np.linalg.norm(query_embedding)
            if norm != 0:
                query_embedding = query_embedding / norm

            scores = embeddings @ query_embedding
            ranked = np.argsort(scores)[::-1]
            correct_positions = np.where(ranked == int(query_index))[0]
            correct_rank = int(correct_positions[0] + 1) if len(correct_positions) else None
            top_score = float(scores[int(ranked[0])])
            current_best_score = float("-inf") if best_scores is None else float(best_scores[int(best_ranked[0])])
            if (
                best_scores is None
                or (correct_rank == 1 and (best_correct_rank != 1 or top_score > current_best_score))
                or (best_correct_rank != 1 and top_score > current_best_score)
            ):
                best_variant_name = variant_name
                best_scores = scores
                best_ranked = ranked
                best_correct_rank = correct_rank

        if best_scores is None or best_ranked is None:
            continue

        scores = best_scores
        ranked = best_ranked
        top_indices = ranked[:top_k]

        top1 = int(top_indices[0])
        top_keys = [_record_key(mapping[int(index)]) for index in top_indices]
        if _record_key(mapping[top1]) == correct_key:
            top1_correct += 1
        if correct_key in top_keys:
            top5_correct += 1

        positive_scores.append(float(scores[int(query_index)]))
        negative_ranked = [int(index) for index in ranked if int(index) != int(query_index)]
        if negative_ranked:
            best_negative_scores.append(float(scores[negative_ranked[0]]))

        top1_score = float(scores[top1])
        if _record_key(mapping[top1]) == correct_key and top1_score < RECOGNITION_SCORE_THRESHOLD:
            threshold_rejections.append(
                {
                    "query_image_id": correct_key,
                    "ground_truth_label": correct_key,
                    "predicted_label": _record_key(mapping[top1]),
                    "threshold_used": RECOGNITION_SCORE_THRESHOLD,
                    "top_1_score": top1_score,
                    "variant": best_variant_name,
                    "top_5": _top_matches(scores, mapping, top_k),
                }
            )
        elif _record_key(mapping[top1]) != correct_key:
            failures.append(
                {
                    "query_image_id": correct_key,
                    "ground_truth_label": correct_key,
                    "predicted_label": _record_key(mapping[top1]),
                    "correct_rank": best_correct_rank,
                    "threshold_used": RECOGNITION_SCORE_THRESHOLD,
                    "variant": best_variant_name,
                    "top_5": _top_matches(scores, mapping, top_k),
                }
            )

        if len(examples) < 12:
            examples.append(
                {
                    "query": correct_key,
                    "top_1": _record_key(mapping[top1]),
                    "top_1_score": float(scores[top1]),
                    "correct_score": float(scores[int(query_index)]),
                    "correct_rank": best_correct_rank,
                    "variant": best_variant_name,
                }
            )

    total = int(len(query_indices))
    return {
        "query_count": total,
        "top_1_accuracy": float(top1_correct / total) if total else 0.0,
        "top_5_accuracy": float(top5_correct / total) if total else 0.0,
        "positive_scores": positive_scores,
        "best_negative_scores": best_negative_scores,
        "failure_count": len(failures),
        "threshold_rejection_count": len(threshold_rejections),
        "failures": failures[:50],
        "threshold_rejections": threshold_rejections[:50],
        "examples": examples,
    }


def _score_summary(scores: list[float]) -> dict[str, float | None]:
    if not scores:
        return {"min": None, "p05": None, "median": None, "p95": None, "max": None}
    values = np.asarray(scores, dtype=np.float32)
    return {
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _threshold_analysis(positive_scores: list[float], negative_scores: list[float]) -> list[dict[str, object]]:
    rows = []
    for threshold in THRESHOLD_CANDIDATES:
        false_rejects = sum(score < threshold for score in positive_scores)
        false_accepts = sum(score >= threshold for score in negative_scores)
        rows.append(
            {
                "threshold": threshold,
                "false_reject_count": int(false_rejects),
                "false_accept_count": int(false_accepts),
                "false_reject_rate": float(false_rejects / len(positive_scores)) if positive_scores else None,
                "false_accept_rate": float(false_accepts / len(negative_scores)) if negative_scores else None,
            }
        )
    return rows


def _style_rows(mapping: list[dict[str, object]], indices: np.ndarray) -> list[tuple[int, str]]:
    rows = []
    for index in indices:
        style = mapping[int(index)].get("style")
        if style:
            rows.append((int(index), str(style)))
    return rows


def _evaluate_style_classifier(
    embeddings: np.ndarray,
    mapping: list[dict[str, object]],
    index_indices: np.ndarray,
    query_indices: np.ndarray,
) -> dict[str, object]:
    train_rows = _style_rows(mapping, index_indices)
    query_rows = _style_rows(mapping, query_indices)
    train_counts = Counter(label for _, label in train_rows)
    train_rows = [(index, label) for index, label in train_rows if train_counts[label] >= 2]
    train_labels = sorted({label for _, label in train_rows})
    query_rows = [(index, label) for index, label in query_rows if label in train_labels]

    if len(train_rows) < 10 or not query_rows:
        return {
            "backend": None,
            "query_count": len(query_rows),
            "accuracy": None,
            "labels": [],
            "confusion_matrix": [],
            "reason": "not_enough_labeled_style_data",
        }

    classifier = StyleClassifier()
    classifier.fit(
        embeddings[[index for index, _ in train_rows]],
        [label for _, label in train_rows],
    )

    y_true = [label for _, label in query_rows]
    y_pred = [classifier.predict(embeddings[index])[0] for index, _ in query_rows]
    labels = sorted(set(y_true) | {str(label) for label in y_pred if label})
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "backend": classifier.backend,
        "query_count": len(query_rows),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "labels": labels,
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def _evaluate_zero_shot(
    embeddings: np.ndarray,
    mapping: list[dict[str, object]],
    query_indices: np.ndarray,
    extractor: EmbeddingExtractor | None,
) -> dict[str, object]:
    if extractor is None or extractor.model_name != "clip":
        return {"query_count": 0, "accuracy": None, "reason": "clip_extractor_not_enabled"}

    rows = _style_rows(mapping, query_indices)
    labels = sorted({label for _, label in rows})
    if not rows or not labels:
        return {"query_count": 0, "accuracy": None, "reason": "not_enough_labeled_style_data"}

    predictor = ClipZeroShotStylePredictor(extractor, labels)
    y_true = [label.replace("_", " ") for _, label in rows]
    y_pred = [predictor.predict(embeddings[index])[0] for index, _ in rows]
    labels_for_matrix = sorted(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=labels_for_matrix)
    return {
        "query_count": len(rows),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "labels": labels_for_matrix,
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate painting recognition and style prediction.")
    parser.add_argument("--project-root", default=".", help="Project root containing the built data directory")
    parser.add_argument("--query-fraction", type=float, default=0.2, help="Fraction of records to reserve as query samples")
    parser.add_argument("--max-query", type=int, default=500, help="Maximum query samples for recognition evaluation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-model", choices=["clip", "resnet50"], default="clip")
    parser.add_argument(
        "--recompute-query-embeddings",
        action="store_true",
        help="Re-read query images and recompute embeddings. Slower, but closer to real uploads.",
    )
    parser.add_argument("--output", default="data/evaluation_report.json")
    args = parser.parse_args()

    project_root = Path(args.project_root)
    data_dir = project_root / "data"
    vector_db = ArtVectorDatabase(
        index_path=data_dir / "faiss_index.idx",
        mapping_path=data_dir / "index_mapping.json",
        embeddings_path=data_dir / "embeddings.npy",
    ).load()
    mapping = vector_db.mapping
    embeddings = _normalize_matrix(vector_db.load_embeddings())

    indexed_model = str(mapping[0].get("embedding_model") or "").lower() if mapping else ""
    if indexed_model and indexed_model != args.embedding_model:
        raise RuntimeError(
            f"The saved index was built with {indexed_model}. Rebuild it with {args.embedding_model} before evaluating."
        )

    index_indices, query_indices = _stable_split(len(mapping), args.query_fraction, args.seed)
    query_indices = _sample_indices(query_indices, args.max_query, args.seed)
    extractor = EmbeddingExtractor(args.embedding_model) if args.recompute_query_embeddings or args.embedding_model == "clip" else None

    recognition = _topk_accuracy(
        embeddings=embeddings,
        mapping=mapping,
        query_indices=query_indices,
        top_k=5,
        extractor=extractor if args.recompute_query_embeddings else None,
    )
    style_classifier = _evaluate_style_classifier(embeddings, mapping, index_indices, query_indices)
    zero_shot = _evaluate_zero_shot(embeddings, mapping, query_indices, extractor)

    positive_scores = recognition.pop("positive_scores")
    negative_scores = recognition.pop("best_negative_scores")
    positive_summary = _score_summary(positive_scores)
    negative_summary = _score_summary(negative_scores)
    suggested_threshold = None
    if positive_summary["p05"] is not None and negative_summary["p95"] is not None:
        suggested_threshold = float((positive_summary["p05"] + negative_summary["p95"]) / 2)

    report = {
        "embedding_model": args.embedding_model,
        "index_size": len(mapping),
        "split": {
            "index_count": int(len(index_indices)),
            "query_count": int(len(query_indices)),
            "query_fraction": args.query_fraction,
            "seed": args.seed,
        },
        "recognition": recognition,
        "style_classifier": style_classifier,
        "clip_zero_shot_style": zero_shot,
        "similarity_score_distribution": {
            "current_threshold": RECOGNITION_SCORE_THRESHOLD,
            "positive_correct_match_scores": positive_summary,
            "best_negative_match_scores": negative_summary,
            "suggested_threshold": suggested_threshold,
            "threshold_candidates": _threshold_analysis(positive_scores, negative_scores),
            "note": "Use recompute-query-embeddings for a stronger threshold estimate from real image loading.",
        },
    }

    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
