from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from art_recognition.config import ProjectPaths
from art_recognition.database import ArtVectorDatabase
from art_recognition.datasets import ArtworkRecord, load_armenian_records, load_wikiart_records
from art_recognition.ml_models import (
    ClipZeroShotStylePredictor,
    EmbeddingExtractor,
    StyleClassifier,
    predict_style_with_fallback,
)
from art_recognition.preprocessing import preprocess_gallery_image


RECOGNITION_SCORE_THRESHOLD = 0.96
NEAR_THRESHOLD_MARGIN = 0.04
ARTIST_SCORE_THRESHOLD = 0.55
ARTIST_CONFIDENCE_THRESHOLD = 0.72


def crop_border(image: np.ndarray, border_ratio: float = 0.04) -> np.ndarray:
    height, width = image.shape[:2]
    y_margin = int(height * border_ratio)
    x_margin = int(width * border_ratio)
    if y_margin * 2 >= height or x_margin * 2 >= width:
        return image
    return image[y_margin : height - y_margin, x_margin : width - x_margin]


def preprocess_painting_image(image_bgr: np.ndarray, border_ratio: float = 0.04) -> np.ndarray:
    cropped = crop_border(image_bgr, border_ratio=border_ratio)
    rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    return rgb


def preprocess_query_image(image_path: str | Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"could not read query image: {image_path}")

    return preprocess_query_image_variants_from_bgr(image)[0]["processed_rgb"]


def preprocess_query_image_variants_from_bgr(image: np.ndarray) -> list[dict[str, object]]:
    variants: list[dict[str, object]] = [
        {
            "name": "full_image",
            "processed_rgb": preprocess_painting_image(image),
        }
    ]

    result = preprocess_gallery_image(image)
    if result["candidates"]:
        best = max(result["candidates"], key=lambda candidate: candidate.crop.shape[0] * candidate.crop.shape[1])
        variants.append(
            {
                "name": "detected_crop",
                "processed_rgb": preprocess_painting_image(best.crop),
            }
        )

    return variants


def preprocess_query_image_variants(image_path: str | Path) -> list[dict[str, object]]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"could not read query image: {image_path}")
    return preprocess_query_image_variants_from_bgr(image)


def _clean_label(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "unknown"}:
        return None
    return text


def _title_text(metadata: dict[str, object]) -> str:
    return " ".join(
        str(metadata.get(key) or "").lower()
        for key in ("title", "filename", "image_path")
    )


def infer_genre_from_matches(matches: list[dict[str, object]]) -> tuple[str | None, float]:
    genre_keywords = {
        "Portrait": ("portrait", "self-portrait", "head of", "woman with", "girl with", "boy with"),
        "Landscape": ("landscape", "mount", "mountain", "valley", "village", "road", "field", "garden"),
        "Seascape": ("sea", "bay", "coast", "beach", "harbor", "boat", "sail", "river", "lake", "sevan"),
        "Still life": ("still life", "flowers", "fruit", "pears", "peaches", "pomegranate", "pitcher", "skull"),
        "Cityscape": ("street", "city", "bridge", "yerevan", "constantinople", "odessa", "church", "cathedral"),
        "Historical or religious scene": ("battle", "saint", "church", "cathedral", "shrine", "exodus", "execution"),
        "Figure scene": ("dancer", "family", "mother", "workers", "shepherd", "servants", "musician"),
    }

    scores: Counter[str] = Counter()
    total_weight = 0.0
    for match in matches:
        metadata = match.get("metadata", {})
        if not isinstance(metadata, dict):
            continue

        explicit_genre = _clean_label(metadata.get("genre"))
        weight = max(float(match.get("score") or 0.0), 0.0)
        if explicit_genre:
            scores[explicit_genre] += weight
            total_weight += weight
            continue

        text = _title_text(metadata)
        for genre, keywords in genre_keywords.items():
            if any(keyword in text for keyword in keywords):
                scores[genre] += weight
                total_weight += weight
                break

    if not scores or total_weight <= 0:
        return None, 0.0

    genre, weight = scores.most_common(1)[0]
    return genre, float(weight / total_weight)


def infer_artist_from_matches(matches: list[dict[str, object]]) -> tuple[str | None, float]:
    if not matches:
        return None, 0.0

    artist_scores: Counter[str] = Counter()
    total_weight = 0.0
    for match in matches:
        score = max(float(match.get("score") or 0.0), 0.0)
        metadata = match.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        artist = _clean_label(metadata.get("artist"))
        if not artist:
            continue
        artist_scores[artist] += score
        total_weight += score

    if not artist_scores or total_weight <= 0:
        return None, 0.0

    artist, weight = artist_scores.most_common(1)[0]
    confidence = float(weight / total_weight)
    best_score = float(matches[0].get("score") or 0.0)
    if best_score >= ARTIST_SCORE_THRESHOLD and confidence >= ARTIST_CONFIDENCE_THRESHOLD:
        return artist, confidence
    return None, confidence


def build_query_response(
    image_path: str | Path,
    matches: list[dict[str, object]],
    predicted_style: str | None = None,
    predicted_style_confidence: float = 0.0,
    predicted_style_source: str | None = None,
) -> dict[str, object]:
    best_match = matches[0]["metadata"] if matches else None
    best_score = float(matches[0]["score"]) if matches else 0.0
    is_recognized = best_match is not None and best_score >= RECOGNITION_SCORE_THRESHOLD

    possible_artist, possible_artist_confidence = infer_artist_from_matches(matches[:8])
    inferred_genre, inferred_genre_confidence = infer_genre_from_matches(matches[:8])

    fallback_style = None
    if best_match is not None:
        fallback_style = _clean_label(best_match.get("style")) or _clean_label(best_match.get("predicted_style"))

    response = {
        "query_image": str(image_path),
        "is_recognized": is_recognized,
        "recognition_status": "recognized" if is_recognized else "not_found",
        "recognition_score": best_score,
        "recognition_threshold": RECOGNITION_SCORE_THRESHOLD,
        "near_threshold_margin": NEAR_THRESHOLD_MARGIN,
        "is_near_threshold": (
            best_match is not None
            and not is_recognized
            and best_score >= RECOGNITION_SCORE_THRESHOLD - NEAR_THRESHOLD_MARGIN
        ),
        "near_match_candidate": best_match if best_match is not None and best_score >= RECOGNITION_SCORE_THRESHOLD - NEAR_THRESHOLD_MARGIN else None,
        "recognized_painting": best_match.get("title") if is_recognized and best_match else None,
        "artist": best_match.get("artist") if is_recognized and best_match else None,
        "year": best_match.get("year") if is_recognized and best_match else None,
        "predicted_style": predicted_style or fallback_style,
        "predicted_style_confidence": predicted_style_confidence,
        "predicted_style_source": predicted_style_source,
        "inferred_genre": inferred_genre,
        "inferred_genre_confidence": inferred_genre_confidence,
        "possible_artist": possible_artist if not is_recognized else None,
        "possible_artist_confidence": possible_artist_confidence if not is_recognized else 0.0,
        "similar_paintings": matches,
    }

    return response


def _write_processed_image(processed_dir: Path, record: ArtworkRecord, image_rgb: np.ndarray) -> str:
    target_dir = processed_dir / record.source
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{Path(record.filename).stem}.jpg"
    cv2.imwrite(str(target_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    return str(target_path)


def _fit_style_classifier(
    records: list[dict[str, object]],
    embeddings: np.ndarray,
    classifier_path: Path,
) -> StyleClassifier | None:
    style_rows = [
        (index, record["style"])
        for index, record in enumerate(records)
        if record.get("source") == "wikiart" and record.get("style")
    ]
    if not style_rows:
        return None

    style_counter = Counter(style for _, style in style_rows)
    eligible_indices = [index for index, style in style_rows if style_counter[style] >= 2]
    if len(eligible_indices) < 10:
        return None

    classifier = StyleClassifier()
    classifier.fit(embeddings[eligible_indices], [str(records[index]["style"]) for index in eligible_indices])
    classifier.save(classifier_path)
    return classifier


@dataclass
class BuildSummary:
    total_records: int
    armenian_records: int
    wikiart_records: int
    style_classes: int
    style_classifier_trained: bool
    style_classifier_backend: str | None
    index_path: str
    mapping_path: str
    classifier_path: str | None
    armenian_style_predictions_path: str | None


class ArtRecognitionPipeline:
    def __init__(self, project_root: str | Path = ".") -> None:
        self.project_root = Path(project_root)
        self.paths = ProjectPaths(self.project_root)
        self.data_dir = self.paths.data_dir
        self.processed_dir = self.paths.processed_dir
        self.index_path = self.paths.faiss_index_path
        self.mapping_path = self.paths.mapping_path
        self.embeddings_path = self.paths.embeddings_path
        self.classifier_path = self.paths.classifier_path
        self.build_report_path = self.paths.build_report_path
        self.armenian_style_predictions_path = self.data_dir / "armenian_style_predictions.csv"

    def build_index(
        self,
        wikiart_sample_size: int = 4500,
        wikiart_metadata_path: str | None = "",
        embedding_model: str = "clip",
        include_wikiart: bool = True,
    ) -> BuildSummary:
        armenian_records = load_armenian_records(self.project_root)
        wikiart_records: list[ArtworkRecord] = []

        if include_wikiart:
            wikiart_records = load_wikiart_records(
                sample_size=wikiart_sample_size,
                metadata_path=wikiart_metadata_path,
                output_dir=self.paths.wikiart_raw_dir,
            )

        all_records = armenian_records + wikiart_records
        if not all_records:
            raise ValueError("no artwork records found to index")

        extractor = EmbeddingExtractor(model_name=embedding_model)
        embeddings: list[np.ndarray] = []
        mapping: list[dict[str, object]] = []

        for record in all_records:
            image_bgr = cv2.imread(record.image_path)
            if image_bgr is None:
                continue

            processed_rgb = preprocess_painting_image(image_bgr)
            embedding = extractor.extract(processed_rgb)
            processed_path = _write_processed_image(self.processed_dir, record, processed_rgb)

            entry = record.to_mapping()
            entry.update(
                {
                    "processed_image_path": processed_path,
                    "embedding_model": embedding_model,
                }
            )
            embeddings.append(embedding)
            mapping.append(entry)

        if not embeddings:
            raise ValueError("all candidate images failed during embedding extraction")

        embeddings_array = np.vstack(embeddings).astype(np.float32)
        classifier = _fit_style_classifier(mapping, embeddings_array, self.classifier_path)

        if classifier is not None:
            for index, record in enumerate(mapping):
                if record.get("source") != "armenian_local":
                    continue
                predicted_style, confidence = classifier.predict(embeddings_array[index])
                record["predicted_style"] = predicted_style
                record["predicted_style_confidence"] = confidence
            self._save_armenian_style_predictions(mapping)

        vector_db = ArtVectorDatabase(
            index_path=self.index_path,
            mapping_path=self.mapping_path,
            embeddings_path=self.embeddings_path,
        )
        vector_db.build(embeddings_array, mapping)

        summary = BuildSummary(
            total_records=len(mapping),
            armenian_records=sum(1 for record in mapping if record["source"] == "armenian_local"),
            wikiart_records=sum(1 for record in mapping if record["source"] == "wikiart"),
            style_classes=len({record["style"] for record in mapping if record.get("style")}),
            style_classifier_trained=classifier is not None,
            style_classifier_backend=classifier.backend if classifier is not None else None,
            index_path=str(self.index_path),
            mapping_path=str(self.mapping_path),
            classifier_path=str(self.classifier_path) if classifier is not None else None,
            armenian_style_predictions_path=(
                str(self.armenian_style_predictions_path) if classifier is not None else None
            ),
        )
        self.build_report_path.write_text(json.dumps(summary.__dict__, indent=2), encoding="utf-8")
        self.mapping_path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
        return summary

    def _save_armenian_style_predictions(self, mapping: list[dict[str, object]]) -> None:
        armenian_rows = []
        for record in mapping:
            if record.get("source") != "armenian_local":
                continue
            armenian_rows.append(
                {
                    "filename": record.get("filename"),
                    "title": record.get("title"),
                    "artist": record.get("artist"),
                    "year": record.get("year"),
                    "predicted_style": record.get("predicted_style"),
                    "predicted_style_confidence": record.get("predicted_style_confidence"),
                    "processed_image_path": record.get("processed_image_path"),
                }
            )

        if armenian_rows:
            self.armenian_style_predictions_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(armenian_rows).to_csv(self.armenian_style_predictions_path, index=False)

    def query(
        self,
        image_path: str | Path,
        embedding_model: str = "clip",
        top_k: int = 3,
    ) -> dict[str, object]:
        extractor = EmbeddingExtractor(model_name=embedding_model)
        query_variants = preprocess_query_image_variants(image_path)
        vector_db = ArtVectorDatabase(
            index_path=self.index_path,
            mapping_path=self.mapping_path,
            embeddings_path=self.embeddings_path,
        ).load()
        indexed_model = str(vector_db.mapping[0].get("embedding_model") or "").lower() if vector_db.mapping else ""
        if indexed_model and indexed_model != embedding_model.lower():
            raise RuntimeError(
                f"The saved index was built with {indexed_model}. Rebuild it with {embedding_model} before querying."
            )
        classifier = StyleClassifier.load(self.classifier_path) if self.classifier_path.exists() else None
        zero_shot = None
        if extractor.model_name == "clip":
            labels = classifier.classes_ if classifier is not None else sorted(
                {str(record.get("style")) for record in vector_db.mapping if record.get("style")}
            )
            if labels:
                zero_shot = ClipZeroShotStylePredictor(extractor, labels)

        best_result: dict[str, object] | None = None
        for variant in query_variants:
            query_embedding = extractor.extract(variant["processed_rgb"])
            matches = vector_db.export_matches_with_numpy(query_embedding, k=top_k)

            predicted_style, style_confidence, style_source = predict_style_with_fallback(
                query_embedding,
                classifier,
                zero_shot,
            )

            result = build_query_response(
                image_path=image_path,
                matches=matches,
                predicted_style=predicted_style,
                predicted_style_confidence=style_confidence,
                predicted_style_source=style_source,
            )
            result["query_variant"] = variant["name"]
            if result["is_recognized"]:
                return result
            if best_result is None or result["recognition_score"] > best_result["recognition_score"]:
                best_result = result

        if best_result is None:
            return build_query_response(image_path=image_path, matches=[])

        return best_result
