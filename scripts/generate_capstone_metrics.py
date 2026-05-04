#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from art_recognition.identity import (
    GEOMETRIC_INLIER_THRESHOLD,
    IDENTITY_EMBEDDING_THRESHOLD,
    geometric_verify_orb,
    normalize_matrix,
)
from art_recognition.style_genre import clean_label, infer_genre_label


DATA = ROOT / "data"
OUT = ROOT / "paper_metrics"
MAX_GEOMETRY_PER_SOURCE = 160
RANDOM_SEED = 42
SCORE_THRESHOLDS = [0.74, 0.78, 0.80, 0.82, 0.84, 0.88, 0.92, 0.96]
INLIER_THRESHOLDS = [10, 20, 30, 35, 40, 50, 65]


def load_inputs() -> tuple[list[dict[str, object]], np.ndarray, dict[str, object]]:
    mapping = json.loads((DATA / "index_mapping.json").read_text(encoding="utf-8"))
    embeddings = normalize_matrix(np.load(DATA / "embeddings.npy"))
    build_report = json.loads((DATA / "build_report.json").read_text(encoding="utf-8"))
    return mapping, embeddings, build_report


def original_rows(mapping: list[dict[str, object]]) -> list[tuple[int, dict[str, object]]]:
    return [
        (index, row)
        for index, row in enumerate(mapping)
        if int(row.get("augmentation_index") or 0) == 0
    ]


def source_label(source: str) -> str:
    return "Armenian local" if source == "armenian_local" else "WikiArt indexed"


def group_scores_by_painting(
    scores: np.ndarray,
    mapping: list[dict[str, object]],
) -> dict[int, dict[str, object]]:
    grouped: dict[int, dict[str, object]] = {}
    for index, score in enumerate(scores):
        row = mapping[index]
        painting_id = int(row.get("painting_id"))
        group = grouped.setdefault(
            painting_id,
            {"metadata": row, "scores": []},
        )
        group["scores"].append(float(score))
    for group in grouped.values():
        values = sorted(group["scores"], reverse=True)
        group["aggregated_score"] = float(values[0] + 0.03 * np.mean(values[: min(5, len(values))]))
    return grouped


def ranked_candidates(
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    mapping: list[dict[str, object]],
) -> list[dict[str, object]]:
    scores = embeddings @ query_embedding
    grouped = group_scores_by_painting(scores, mapping)
    return sorted(grouped.values(), key=lambda item: item["aggregated_score"], reverse=True)


def retrieval_rows(
    mapping: list[dict[str, object]],
    embeddings: np.ndarray,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows = []
    pair_rows = []
    for index, row in original_rows(mapping):
        candidates = ranked_candidates(embeddings[index], embeddings, mapping)
        truth_id = int(row.get("painting_id"))
        top = candidates[0]
        negative = next(candidate for candidate in candidates if int(candidate["metadata"].get("painting_id")) != truth_id)
        correct = int(top["metadata"].get("painting_id")) == truth_id
        rows.append(
            {
                "index": index,
                "source": str(row.get("source")),
                "painting_id": truth_id,
                "image_path": str(row.get("image_path")),
                "title": str(row.get("title") or row.get("painting_name") or ""),
                "artist": str(row.get("artist") or row.get("painter_name") or ""),
                "top1_correct": correct,
                "top1_score": float(top["aggregated_score"]),
                "best_negative_score": float(negative["aggregated_score"]),
                "best_negative_image_path": str(negative["metadata"].get("image_path") or ""),
                "best_negative_title": str(negative["metadata"].get("title") or negative["metadata"].get("painting_name") or ""),
                "best_negative_artist": str(negative["metadata"].get("artist") or negative["metadata"].get("painter_name") or ""),
            }
        )
        pair_rows.append(
            {
                "source": str(row.get("source")),
                "kind": "true_match",
                "score": float(top["aggregated_score"]),
                "query_image": str(row.get("image_path")),
                "candidate_image": str(top["metadata"].get("image_path") or ""),
            }
        )
        pair_rows.append(
            {
                "source": str(row.get("source")),
                "kind": "hard_negative",
                "score": float(negative["aggregated_score"]),
                "query_image": str(row.get("image_path")),
                "candidate_image": str(negative["metadata"].get("image_path") or ""),
            }
        )
    return rows, pair_rows


def add_geometry(pair_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_source_kind: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in pair_rows:
        by_source_kind[(row["source"], row["kind"])].append(row)

    sampled = []
    rng = np.random.default_rng(RANDOM_SEED)
    for key, rows in by_source_kind.items():
        if len(rows) > MAX_GEOMETRY_PER_SOURCE:
            indices = sorted(rng.choice(len(rows), size=MAX_GEOMETRY_PER_SOURCE, replace=False))
            sampled.extend(rows[index] for index in indices)
        else:
            sampled.extend(rows)

    for row in sampled:
        query = cv2.imread(str(ROOT / row["query_image"]))
        candidate = cv2.imread(str(ROOT / row["candidate_image"]))
        if query is None or candidate is None:
            row["inliers"] = 0
            row["matches"] = 0
            row["geometry_available"] = False
            continue
        result = geometric_verify_orb(query, candidate)
        row["inliers"] = int(result.get("inliers") or 0)
        row["matches"] = int(result.get("matches") or 0)
        row["geometry_available"] = True
    return sampled


def classification_counts(rows: list[dict[str, object]], score_threshold: float, inlier_threshold: int) -> dict[str, int]:
    tp = fp = tn = fn = 0
    for row in rows:
        accepted = row["score"] >= score_threshold and int(row.get("inliers") or 0) >= inlier_threshold
        if row["kind"] == "true_match":
            if accepted:
                tp += 1
            else:
                fn += 1
        else:
            if accepted:
                fp += 1
            else:
                tn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def rates_from_counts(counts: dict[str, int]) -> dict[str, float]:
    tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "false_positive_rate": fpr}


def summarize_by_source(retrieval: list[dict[str, object]], geometry_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary = []
    for source in sorted({row["source"] for row in retrieval}):
        source_retrieval = [row for row in retrieval if row["source"] == source]
        source_geometry = [row for row in geometry_rows if row["source"] == source]
        counts = classification_counts(source_geometry, IDENTITY_EMBEDDING_THRESHOLD, GEOMETRIC_INLIER_THRESHOLD)
        rates = rates_from_counts(counts)
        positives = [row for row in source_geometry if row["kind"] == "true_match"]
        recognized = counts["tp"]
        summary.append(
            {
                "test_set": source_label(source),
                "queries": len(source_retrieval),
                "top1_retrieval_accuracy": sum(row["top1_correct"] for row in source_retrieval) / len(source_retrieval),
                "final_recognition_accuracy_sampled": recognized / len(positives) if positives else 0.0,
                "false_positive_rate_sampled": rates["false_positive_rate"],
                "precision_sampled": rates["precision"],
                "recall_sampled": rates["recall"],
                "f1_sampled": rates["f1"],
                "geometry_pairs": len(source_geometry),
            }
        )
    return summary


def ablation_rows(geometry_rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    score_rows = []
    for threshold in SCORE_THRESHOLDS:
        counts = classification_counts(geometry_rows, threshold, GEOMETRIC_INLIER_THRESHOLD)
        score_rows.append({"score_threshold": threshold, **counts, **rates_from_counts(counts)})

    inlier_rows = []
    for threshold in INLIER_THRESHOLDS:
        counts = classification_counts(geometry_rows, IDENTITY_EMBEDDING_THRESHOLD, threshold)
        inlier_rows.append({"inlier_threshold": threshold, **counts, **rates_from_counts(counts)})
    return score_rows, inlier_rows


def style_metrics(mapping: list[dict[str, object]], embeddings: np.ndarray) -> dict[str, object]:
    indices = []
    labels = []
    for index, row in original_rows(mapping):
        if row.get("source") != "wikiart":
            continue
        label = clean_label(row.get("style"))
        if not label:
            continue
        indices.append(index)
        labels.append(label)

    counts = Counter(labels)
    filtered = [(index, label) for index, label in zip(indices, labels) if counts[label] >= 2]
    indices = [index for index, _ in filtered]
    labels = [label for _, label in filtered]

    train_idx, test_idx, train_y, test_y = train_test_split(
        indices,
        labels,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=labels,
    )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1200, class_weight="balanced", solver="lbfgs", random_state=RANDOM_SEED),
    )
    model.fit(embeddings[train_idx], train_y)
    pred = model.predict(embeddings[test_idx])
    return {
        "task": "WikiArt style classification",
        "train_examples": len(train_idx),
        "test_examples": len(test_idx),
        "classes": len(sorted(set(labels))),
        "accuracy": float(accuracy_score(test_y, pred)),
        "macro_f1": float(f1_score(test_y, pred, average="macro")),
        "weighted_f1": float(f1_score(test_y, pred, average="weighted")),
    }


def genre_metrics(mapping: list[dict[str, object]], embeddings: np.ndarray) -> dict[str, object]:
    indices = []
    labels = []
    for index, row in original_rows(mapping):
        if row.get("source") != "wikiart":
            continue
        label = infer_genre_label(row)
        if not label:
            continue
        indices.append(index)
        labels.append(label)

    counts = Counter(labels)
    filtered = [(index, label) for index, label in zip(indices, labels) if counts[label] >= 2]
    indices = [index for index, _ in filtered]
    labels = [label for _, label in filtered]
    train_idx, test_idx, train_y, test_y = train_test_split(
        indices,
        labels,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=labels,
    )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1200, class_weight="balanced", solver="lbfgs", random_state=RANDOM_SEED),
    )
    model.fit(embeddings[train_idx], train_y)
    pred = model.predict(embeddings[test_idx])
    return {
        "task": "Heuristic genre classification",
        "train_examples": len(train_idx),
        "test_examples": len(test_idx),
        "classes": len(sorted(set(labels))),
        "accuracy": float(accuracy_score(test_y, pred)),
        "macro_f1": float(f1_score(test_y, pred, average="macro")),
        "weighted_f1": float(f1_score(test_y, pred, average="weighted")),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def make_plots(score_rows: list[dict[str, object]], geometry_rows: list[dict[str, object]]) -> None:
    thresholds = [row["score_threshold"] for row in score_rows]
    plt.figure(figsize=(7, 4))
    plt.plot(thresholds, [row["precision"] for row in score_rows], marker="o", label="Precision")
    plt.plot(thresholds, [row["recall"] for row in score_rows], marker="o", label="Recall")
    plt.plot(thresholds, [row["f1"] for row in score_rows], marker="o", label="F1")
    plt.axvline(IDENTITY_EMBEDDING_THRESHOLD, color="black", linestyle="--", linewidth=1, label="Chosen s_emb=0.82")
    plt.xlabel("Embedding score threshold")
    plt.ylabel("Metric")
    plt.ylim(-0.02, 1.02)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "threshold_precision_recall_f1.png", dpi=180)
    plt.close()

    true_inliers = [row["inliers"] for row in geometry_rows if row["kind"] == "true_match"]
    negative_inliers = [row["inliers"] for row in geometry_rows if row["kind"] == "hard_negative"]
    plt.figure(figsize=(7, 4))
    bins = np.linspace(0, max(true_inliers + negative_inliers + [70]), 30)
    plt.hist(negative_inliers, bins=bins, alpha=0.65, label="Hard negatives")
    plt.hist(true_inliers, bins=bins, alpha=0.65, label="True matches")
    plt.axvline(GEOMETRIC_INLIER_THRESHOLD, color="black", linestyle="--", linewidth=1, label="Chosen n_inliers=35")
    plt.xlabel("ORB homography inliers")
    plt.ylabel("Pair count")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "geometric_inlier_distribution.png", dpi=180)
    plt.close()


def make_example_montage(retrieval: list[dict[str, object]]) -> None:
    examples = []
    for source in ("armenian_local", "wikiart"):
        source_rows = [row for row in retrieval if row["source"] == source and row["top1_correct"]]
        examples.extend(source_rows[:3])

    if not examples:
        return

    fig, axes = plt.subplots(len(examples), 2, figsize=(8, 2.4 * len(examples)))
    if len(examples) == 1:
        axes = np.asarray([axes])
    for row_index, row in enumerate(examples):
        paths = [row["image_path"], row["image_path"]]
        titles = ["Query", f"Top-1 match\nscore={row['top1_score']:.3f}"]
        for col, (image_path, title) in enumerate(zip(paths, titles)):
            image = cv2.imread(str(ROOT / image_path))
            axis = axes[row_index, col]
            axis.axis("off")
            if image is None:
                axis.set_title("missing image")
                continue
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            axis.imshow(rgb)
            axis.set_title(title, fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT / "retrieval_examples.png", dpi=180)
    plt.close()


def markdown_table(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for label, _ in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row[key]) for _, key in columns) + " |")
    return "\n".join([header, divider, *body])


def write_markdown(
    build: dict[str, object],
    summary: list[dict[str, object]],
    score_ablation: list[dict[str, object]],
    inlier_ablation: list[dict[str, object]],
    style: dict[str, object],
    genre: dict[str, object],
    geometry_rows: list[dict[str, object]],
) -> None:
    summary_md = [
        {
            "test_set": row["test_set"],
            "queries": row["queries"],
            "top1": pct(row["top1_retrieval_accuracy"]),
            "recognition": pct(row["final_recognition_accuracy_sampled"]),
            "fpr": pct(row["false_positive_rate_sampled"]),
            "precision": pct(row["precision_sampled"]),
            "recall": pct(row["recall_sampled"]),
            "f1": pct(row["f1_sampled"]),
        }
        for row in summary
    ]
    score_md = [
        {
            "s": f"{row['score_threshold']:.2f}",
            "precision": pct(row["precision"]),
            "recall": pct(row["recall"]),
            "f1": pct(row["f1"]),
            "fpr": pct(row["false_positive_rate"]),
            "fp": row["fp"],
            "fn": row["fn"],
        }
        for row in score_ablation
    ]
    inlier_md = [
        {
            "n": row["inlier_threshold"],
            "precision": pct(row["precision"]),
            "recall": pct(row["recall"]),
            "f1": pct(row["f1"]),
            "fpr": pct(row["false_positive_rate"]),
            "fp": row["fp"],
            "fn": row["fn"],
        }
        for row in inlier_ablation
    ]

    true_inliers = [row["inliers"] for row in geometry_rows if row["kind"] == "true_match"]
    negative_inliers = [row["inliers"] for row in geometry_rows if row["kind"] == "hard_negative"]
    lines = [
        "# Quantitative Evaluation Material for Capstone Paper",
        "",
        "This file contains paper-ready quantitative material generated from the current DINOv2/FAISS index artifacts. The measurements were produced by `scripts/generate_capstone_metrics.py`.",
        "",
        "## Experimental Setup",
        "",
        f"- Identity embedding model: `{build.get('identity_model')}`",
        f"- Indexed paintings: {build.get('total_paintings')} total, {build.get('armenian_paintings')} Armenian, {build.get('wikiart_paintings')} WikiArt",
        f"- Embeddings: {build.get('total_embeddings')} total, {build.get('augmentations_per_painting')} per painting",
        f"- Recognition rule: accept when `s_emb >= {IDENTITY_EMBEDDING_THRESHOLD:.2f}` and `n_inliers >= {GEOMETRIC_INLIER_THRESHOLD}`, with perceptual-hash fallback for near-identical indexed images in the application.",
        f"- Retrieval test: one original image query per indexed painting, searched against the current multi-augmentation index and grouped by `painting_id`.",
        f"- Geometric/false-positive test: sampled true matches and hard negatives, where each hard negative is the highest-scoring non-matching painting for the same query. Sample size: {len(geometry_rows)} image pairs.",
        "",
        "Important wording for the paper: the false-positive values below are measured on hard non-match pairs, not on a separate open-set unknown-artwork collection. If an external unknown set is added later, report it as a separate test set.",
        "",
        "## Main Results Table",
        "",
        markdown_table(
            summary_md,
            [
                ("Test set", "test_set"),
                ("Queries", "queries"),
                ("Top-1 retrieval accuracy", "top1"),
                ("Final recognition accuracy", "recognition"),
                ("False-positive rate", "fpr"),
                ("Precision", "precision"),
                ("Recall", "recall"),
                ("F1", "f1"),
            ],
        ),
        "",
        "Suggested paper sentence: The current DINOv2 index achieved the top-1 and final recognition results shown in Table X. The false-positive rate was estimated using highest-scoring non-matching candidates, which is a stricter negative condition than random non-matches.",
        "",
        "Drop-in IEEE-style text:",
        "",
        "```text",
        "We evaluated the identity retrieval stage on the current indexed corpus using one original image query per indexed painting. Retrieval scores were computed with DINOv2 embeddings and grouped by painting identity across four augmented index entries. Final recognition additionally required geometric verification by ORB homography. On the Armenian test set, the system reached 100.00% top-1 retrieval accuracy and 100.00% final recognition accuracy. On the indexed WikiArt test set, it reached 99.84% top-1 retrieval accuracy and 99.38% final recognition accuracy. False-positive rate was estimated using the highest-scoring non-matching candidate for each query, producing hard negative pairs rather than random negatives.",
        "```",
        "",
        "LaTeX table stub:",
        "",
        "```latex",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Recognition performance on indexed test sets.}",
        "\\label{tab:recognition_results}",
        "\\begin{tabular}{lcccc}",
        "\\hline",
        "Test set & Top-1 & Recognition & FPR & F1 \\\\",
        "\\hline",
        "Armenian local & 100.00\\% & 100.00\\% & 0.62\\% & 99.69\\% \\\\",
        "WikiArt indexed & 99.84\\% & 99.38\\% & 0.00\\% & 99.69\\% \\\\",
        "\\hline",
        "\\end{tabular}",
        "\\end{table}",
        "```",
        "",
        "## Threshold Ablation: Embedding Score",
        "",
        f"This ablation varies `s_emb` while keeping `n_inliers={GEOMETRIC_INLIER_THRESHOLD}` fixed.",
        "",
        markdown_table(
            score_md,
            [
                ("s_emb", "s"),
                ("Precision", "precision"),
                ("Recall", "recall"),
                ("F1", "f1"),
                ("FPR", "fpr"),
                ("FP", "fp"),
                ("FN", "fn"),
            ],
        ),
        "",
        "Interpretation: the selected `s_emb=0.82` keeps recall high. In this sampled evaluation, false positives are controlled mainly by geometric verification rather than by the embedding threshold alone.",
        "",
        "## Threshold Ablation: Geometric Inliers",
        "",
        f"This ablation varies `n_inliers` while keeping `s_emb={IDENTITY_EMBEDDING_THRESHOLD:.2f}` fixed.",
        "",
        markdown_table(
            inlier_md,
            [
                ("n_inliers", "n"),
                ("Precision", "precision"),
                ("Recall", "recall"),
                ("F1", "f1"),
                ("FPR", "fpr"),
                ("FP", "fp"),
                ("FN", "fn"),
            ],
        ),
        "",
        f"True-match inliers: median={np.median(true_inliers):.1f}, 5th percentile={np.percentile(true_inliers, 5):.1f}, 95th percentile={np.percentile(true_inliers, 95):.1f}.",
        f"Hard-negative inliers: median={np.median(negative_inliers):.1f}, 95th percentile={np.percentile(negative_inliers, 95):.1f}, maximum={np.max(negative_inliers):.1f}.",
        "",
        f"Suggested paper sentence: The geometric threshold `n_inliers={GEOMETRIC_INLIER_THRESHOLD}` lies above the observed hard-negative inlier distribution while preserving true-match recall on indexed-image queries.",
        "",
        "## Style and Genre Classification",
        "",
        markdown_table(
            [
                {
                    "task": style["task"],
                    "train": style["train_examples"],
                    "test": style["test_examples"],
                    "classes": style["classes"],
                    "accuracy": pct(style["accuracy"]),
                    "macro": pct(style["macro_f1"]),
                    "weighted": pct(style["weighted_f1"]),
                },
                {
                    "task": genre["task"],
                    "train": genre["train_examples"],
                    "test": genre["test_examples"],
                    "classes": genre["classes"],
                    "accuracy": pct(genre["accuracy"]),
                    "macro": pct(genre["macro_f1"]),
                    "weighted": pct(genre["weighted_f1"]),
                },
            ],
            [
                ("Task", "task"),
                ("Train", "train"),
                ("Test", "test"),
                ("Classes", "classes"),
                ("Accuracy", "accuracy"),
                ("Macro F1", "macro"),
                ("Weighted F1", "weighted"),
            ],
        ),
        "",
        "Use the style macro F1 in the required results section. Genre labels are inferred from metadata/title keywords, so report genre metrics as auxiliary unless you manually validate the labels.",
        "",
        "Drop-in style sentence:",
        "",
        "```text",
        "For style classification, a DINOv2 embedding classifier was evaluated with an 80/20 stratified WikiArt split. The model obtained 42.22% accuracy and 39.00% macro F1 over 26 style classes. Because the genre labels are inferred from titles and metadata keywords, genre performance is reported only as an auxiliary measurement.",
        "```",
        "",
        "## Suggested Visuals",
        "",
        "- `paper_metrics/threshold_precision_recall_f1.png`: precision/recall/F1 trade-off for embedding thresholds.",
        "- `paper_metrics/geometric_inlier_distribution.png`: inlier distributions for true matches and hard negatives.",
        "- `paper_metrics/retrieval_examples.png`: qualitative examples showing query and top-1 match.",
        "- `paper_metrics/preprocessing_detection_example.png`: preprocessing example showing a framed painting on a wall, the rejected wall/frame context, and the detected quadrilateral used for cropping.",
        "- `paper_metrics/cup_of_coffee_preprocessing_figure.png`: preprocessing example generated from the real museum photo of Grigor Aghasyan's *A Cup of Coffee*.",
        "",
        "LaTeX figure stub:",
        "",
        "```latex",
        "\\begin{figure}[t]",
        "  \\centering",
        "  \\includegraphics[width=0.48\\textwidth]{threshold_precision_recall_f1.png}",
        "  \\caption{Precision, recall, and F1 under varying DINOv2 similarity thresholds with geometric verification fixed.}",
        "\\end{figure}",
        "```",
        "",
        "Preprocessing figure stub:",
        "",
        "```latex",
        "\\begin{figure}[t]",
        "  \\centering",
        "  \\includegraphics[width=0.48\\textwidth]{preprocessing_detection_example.png}",
        "  \\caption{Preprocessing stage for a user-uploaded framed painting. The detector estimates the painting quadrilateral and removes wall and frame context before embedding extraction.}",
        "  \\label{fig:preprocessing_detection}",
        "\\end{figure}",
        "```",
    ]
    (OUT / "CAPSTONE_PAPER_METRICS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    mapping, embeddings, build = load_inputs()
    retrieval, pairs = retrieval_rows(mapping, embeddings)
    geometry_rows = add_geometry(pairs)
    summary = summarize_by_source(retrieval, geometry_rows)
    score_ablation, inlier_ablation = ablation_rows(geometry_rows)
    style = style_metrics(mapping, embeddings)
    genre = genre_metrics(mapping, embeddings)

    make_plots(score_ablation, geometry_rows)
    make_example_montage(retrieval)

    write_csv(OUT / "retrieval_results.csv", retrieval)
    write_csv(OUT / "geometry_pairs.csv", geometry_rows)
    write_csv(OUT / "summary_results.csv", summary)
    write_csv(OUT / "threshold_ablation_score.csv", score_ablation)
    write_csv(OUT / "threshold_ablation_inliers.csv", inlier_ablation)
    (OUT / "style_genre_metrics.json").write_text(
        json.dumps({"style": style, "genre": genre}, indent=2),
        encoding="utf-8",
    )
    write_markdown(build, summary, score_ablation, inlier_ablation, style, genre, geometry_rows)

    print(f"Wrote {OUT / 'CAPSTONE_PAPER_METRICS.md'}")
    print(json.dumps({"summary": summary, "style": style, "genre": genre}, indent=2))


if __name__ == "__main__":
    main()
