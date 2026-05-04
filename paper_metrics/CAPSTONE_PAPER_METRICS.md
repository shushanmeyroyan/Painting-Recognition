# Quantitative Evaluation Material for Capstone Paper

This file contains paper-ready quantitative material generated from the current DINOv2/FAISS index artifacts. The measurements were produced by `scripts/generate_capstone_metrics.py`.

## Experimental Setup

- Identity embedding model: `facebook/dinov2-base`
- Indexed paintings: 4709 total, 209 Armenian, 4500 WikiArt
- Embeddings: 18836 total, 4 per painting
- Recognition rule: accept when `s_emb >= 0.82` and `n_inliers >= 35`, with perceptual-hash fallback for near-identical indexed images in the application.
- Retrieval test: one original image query per indexed painting, searched against the current multi-augmentation index and grouped by `painting_id`.
- Geometric/false-positive test: sampled true matches and hard negatives, where each hard negative is the highest-scoring non-matching painting for the same query. Sample size: 640 image pairs.

Important wording for the paper: the false-positive values below are measured on hard non-match pairs, not on a separate open-set unknown-artwork collection. If an external unknown set is added later, report it as a separate test set.

## Main Results Table

| Test set | Queries | Top-1 retrieval accuracy | Final recognition accuracy | False-positive rate | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Armenian local | 209 | 100.00% | 100.00% | 0.62% | 99.38% | 100.00% | 99.69% |
| WikiArt indexed | 4500 | 99.84% | 99.38% | 0.00% | 100.00% | 99.38% | 99.69% |

Suggested paper sentence: The current DINOv2 index achieved the top-1 and final recognition results shown in Table X. The false-positive rate was estimated using highest-scoring non-matching candidates, which is a stricter negative condition than random non-matches.

Drop-in IEEE-style text:

```text
We evaluated the identity retrieval stage on the current indexed corpus using one original image query per indexed painting. Retrieval scores were computed with DINOv2 embeddings and grouped by painting identity across four augmented index entries. Final recognition additionally required geometric verification by ORB homography. On the Armenian test set, the system reached 100.00% top-1 retrieval accuracy and 100.00% final recognition accuracy. On the indexed WikiArt test set, it reached 99.84% top-1 retrieval accuracy and 99.38% final recognition accuracy. False-positive rate was estimated using the highest-scoring non-matching candidate for each query, producing hard negative pairs rather than random negatives.
```

LaTeX table stub:

```latex
\begin{table}[t]
\centering
\caption{Recognition performance on indexed test sets.}
\label{tab:recognition_results}
\begin{tabular}{lcccc}
\hline
Test set & Top-1 & Recognition & FPR & F1 \\
\hline
Armenian local & 100.00\% & 100.00\% & 0.62\% & 99.69\% \\
WikiArt indexed & 99.84\% & 99.38\% & 0.00\% & 99.69\% \\
\hline
\end{tabular}
\end{table}
```

## Threshold Ablation: Embedding Score

This ablation varies `s_emb` while keeping `n_inliers=35` fixed.

| s_emb | Precision | Recall | F1 | FPR | FP | FN |
| --- | --- | --- | --- | --- | --- | --- |
| 0.74 | 99.69% | 99.69% | 99.69% | 0.31% | 1 | 1 |
| 0.78 | 99.69% | 99.69% | 99.69% | 0.31% | 1 | 1 |
| 0.80 | 99.69% | 99.69% | 99.69% | 0.31% | 1 | 1 |
| 0.82 | 99.69% | 99.69% | 99.69% | 0.31% | 1 | 1 |
| 0.84 | 99.69% | 99.69% | 99.69% | 0.31% | 1 | 1 |
| 0.88 | 99.69% | 99.69% | 99.69% | 0.31% | 1 | 1 |
| 0.92 | 100.00% | 99.69% | 99.84% | 0.00% | 0 | 1 |
| 0.96 | 100.00% | 99.69% | 99.84% | 0.00% | 0 | 1 |

Interpretation: the selected `s_emb=0.82` keeps recall high. In this sampled evaluation, false positives are controlled mainly by geometric verification rather than by the embedding threshold alone.

## Threshold Ablation: Geometric Inliers

This ablation varies `n_inliers` while keeping `s_emb=0.82` fixed.

| n_inliers | Precision | Recall | F1 | FPR | FP | FN |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | 99.38% | 100.00% | 99.69% | 0.62% | 2 | 0 |
| 20 | 99.69% | 99.69% | 99.69% | 0.31% | 1 | 1 |
| 30 | 99.69% | 99.69% | 99.69% | 0.31% | 1 | 1 |
| 35 | 99.69% | 99.69% | 99.69% | 0.31% | 1 | 1 |
| 40 | 99.69% | 99.69% | 99.69% | 0.31% | 1 | 1 |
| 50 | 99.69% | 99.38% | 99.53% | 0.31% | 1 | 2 |
| 65 | 99.69% | 99.06% | 99.37% | 0.31% | 1 | 3 |

True-match inliers: median=2500.0, 5th percentile=364.9, 95th percentile=2500.0.
Hard-negative inliers: median=4.0, 95th percentile=7.0, maximum=443.0.

Suggested paper sentence: The geometric threshold `n_inliers=35` lies above the observed hard-negative inlier distribution while preserving true-match recall on indexed-image queries.

## Style and Genre Classification

| Task | Train | Test | Classes | Accuracy | Macro F1 | Weighted F1 |
| --- | --- | --- | --- | --- | --- | --- |
| WikiArt style classification | 3599 | 900 | 26 | 42.22% | 39.00% | 42.12% |
| Heuristic genre classification | 1496 | 375 | 9 | 54.67% | 43.95% | 54.99% |

Use the style macro F1 in the required results section. Genre labels are inferred from metadata/title keywords, so report genre metrics as auxiliary unless you manually validate the labels.

Drop-in style sentence:

```text
For style classification, a DINOv2 embedding classifier was evaluated with an 80/20 stratified WikiArt split. The model obtained 42.22% accuracy and 39.00% macro F1 over 26 style classes. Because the genre labels are inferred from titles and metadata keywords, genre performance is reported only as an auxiliary measurement.
```

## Suggested Visuals

- `paper_metrics/threshold_precision_recall_f1.png`: precision/recall/F1 trade-off for embedding thresholds.
- `paper_metrics/geometric_inlier_distribution.png`: inlier distributions for true matches and hard negatives.
- `paper_metrics/retrieval_examples.png`: qualitative examples showing query and top-1 match.
- `paper_metrics/preprocessing_detection_example.png`: preprocessing example showing a framed painting on a wall, the rejected wall/frame context, and the detected quadrilateral used for cropping.
- `paper_metrics/cup_of_coffee_preprocessing_figure.png`: preprocessing example generated from the real museum photo of Grigor Aghasyan's *A Cup of Coffee*.

LaTeX figure stub:

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.48\textwidth]{threshold_precision_recall_f1.png}
  \caption{Precision, recall, and F1 under varying DINOv2 similarity thresholds with geometric verification fixed.}
\end{figure}
```

Preprocessing figure stub:

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.48\textwidth]{preprocessing_detection_example.png}
  \caption{Preprocessing stage for a user-uploaded framed painting. The detector estimates the painting quadrilateral and removes wall and frame context before embedding extraction.}
  \label{fig:preprocessing_detection}
\end{figure}
```
