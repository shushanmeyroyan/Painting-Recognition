from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import faiss
import numpy as np

from art_recognition.config import ProjectPaths
from art_recognition.cropping import PaintingCropper, remove_border
from art_recognition.datasets import ArtworkRecord, load_armenian_records, load_wikiart_records
from art_recognition.database import ArtVectorDatabase
from art_recognition.identity import (
    DEFAULT_DINOV2_MODEL,
    GEOMETRIC_INLIER_THRESHOLD,
    IDENTITY_EMBEDDING_THRESHOLD,
    PERCEPTUAL_HASH_THRESHOLD,
    Dinov2EmbeddingExtractor,
    aggregate_identity_matches,
    augment_clean_painting,
    geometric_verify,
    hash_distance,
    normalize_matrix,
    perceptual_hash,
)


RECOGNITION_SCORE_THRESHOLD = IDENTITY_EMBEDDING_THRESHOLD
NEAR_THRESHOLD_MARGIN = 0.04


def crop_border(image: np.ndarray, border_ratio: float = 0.05) -> np.ndarray:
    return remove_border(image, border_ratio)


def preprocess_painting_image(image_bgr: np.ndarray, border_ratio: float = 0.0) -> np.ndarray:
    cropped = crop_border(image_bgr, border_ratio=border_ratio)
    return cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)


def preprocess_query_image(image_path: str | Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"could not read query image: {image_path}")
    cropper = PaintingCropper()
    crop = cropper.crop(image).image_bgr
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def preprocess_query_image_variants(image_path: str | Path) -> list[dict[str, object]]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"could not read query image: {image_path}")
    return preprocess_query_image_variants_from_bgr(image)


def preprocess_query_image_variants_from_bgr(image: np.ndarray) -> list[dict[str, object]]:
    orientation_inputs = [
        ("rot0", image),
        ("rot90", cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)),
        ("rot180", cv2.rotate(image, cv2.ROTATE_180)),
        ("rot270", cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)),
    ]

    variants: list[dict[str, object]] = []
    seen_shapes: set[tuple[str, tuple[int, int, int]]] = set()

    for orientation_name, oriented_image in orientation_inputs:
        key = (orientation_name, oriented_image.shape)
        if key in seen_shapes:
            continue
        seen_shapes.add(key)
        variants.append(
            {
                "name": f"{orientation_name}_original_full",
                "processed_rgb": cv2.cvtColor(oriented_image, cv2.COLOR_BGR2RGB),
                "processed_bgr": oriented_image,
                "crop_confidence": 1.0,
            }
        )
        for ratio in (0.03, 0.05):
            cropped = remove_border(oriented_image, ratio)
            variants.append(
                {
                    "name": f"{orientation_name}_original_border_{ratio:.2f}",
                    "processed_rgb": cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB),
                    "processed_bgr": cropped,
                    "crop_confidence": 1.0,
                }
            )

    # Cropper variants are most useful for normal camera orientation. Full-image
    # rotation variants above handle phone/EXIF rotation without multiplying the
    # slower cropper path four times.
    variants.extend(_cropper_query_variants(image))
    return variants


def _cropper_query_variants(image: np.ndarray) -> list[dict[str, object]]:
    variants = []
    cropper = PaintingCropper()
    crop_result = cropper.crop(image)
    variants.append(
        {
            "name": crop_result.method,
            "processed_rgb": cv2.cvtColor(crop_result.image_bgr, cv2.COLOR_BGR2RGB),
            "processed_bgr": crop_result.image_bgr,
            "crop_confidence": crop_result.confidence,
        }
    )
    for ratio in (0.03, 0.08):
        cropped = remove_border(crop_result.image_bgr, ratio)
        variants.append(
            {
                "name": f"{crop_result.method}_border_{ratio:.2f}",
                "processed_rgb": cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB),
                "processed_bgr": cropped,
                "crop_confidence": crop_result.confidence,
            }
        )
    return variants


def _identity_mapping(
    record: ArtworkRecord,
    painting_id: int,
    augmentation_index: int,
    embedding_model: str,
) -> dict[str, object]:
    original_bgr = cv2.imread(record.image_path)
    return {
        "painting_id": painting_id,
        "augmentation_index": augmentation_index,
        "source": record.source,
        "filename": record.filename,
        "painter_name": record.artist,
        "painting_name": record.title,
        "year": record.year,
        "image_path": record.image_path,
        "title": record.title,
        "artist": record.artist,
        "style": record.style,
        "genre": record.genre,
        "perceptual_hash": perceptual_hash(original_bgr) if original_bgr is not None else None,
        "embedding_model": embedding_model,
    }


def _matches_from_embeddings(
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    mapping: list[dict[str, object]],
    top_k: int,
) -> list[dict[str, object]]:
    query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(query)
    scores = embeddings @ query.reshape(-1)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [
        {
            "rank": rank,
            "score": float(scores[index]),
            "metadata": mapping[int(index)],
        }
        for rank, index in enumerate(top_indices, start=1)
    ]


def build_query_response(
    image_path: str | Path,
    matches: list[dict[str, object]],
    predicted_style: str | None = None,
    predicted_style_confidence: float = 0.0,
    predicted_style_source: str | None = None,
) -> dict[str, object]:
    best = matches[0] if matches else None
    metadata = best.get("metadata", {}) if best else {}
    score = float(best.get("score") or 0.0) if best else 0.0
    geometric = best.get("geometric_verification", {}) if best else {}
    inliers = int(geometric.get("inliers") or 0)
    hash_match = bool(best and geometric.get("method") == "perceptual_hash")
    recognized = bool(
        hash_match
        or (
            best
            and score >= IDENTITY_EMBEDDING_THRESHOLD
            and inliers >= GEOMETRIC_INLIER_THRESHOLD
        )
    )

    return {
        "query_image": str(image_path),
        "is_recognized": recognized,
        "recognition_status": "recognized" if recognized else "not_found",
        "recognition_score": score,
        "recognition_threshold": IDENTITY_EMBEDDING_THRESHOLD,
        "geometric_inliers": inliers,
        "geometric_inlier_threshold": GEOMETRIC_INLIER_THRESHOLD,
        "geometric_verification": geometric,
        "perceptual_hash_threshold": PERCEPTUAL_HASH_THRESHOLD,
        "near_threshold_margin": NEAR_THRESHOLD_MARGIN,
        "is_near_threshold": bool(best and not recognized and score >= IDENTITY_EMBEDDING_THRESHOLD - NEAR_THRESHOLD_MARGIN),
        "near_match_candidate": metadata if best and score >= IDENTITY_EMBEDDING_THRESHOLD - NEAR_THRESHOLD_MARGIN else None,
        "recognized_painting": metadata.get("painting_name") if recognized else None,
        "artist": metadata.get("painter_name") if recognized else None,
        "year": metadata.get("year") if recognized else None,
        "predicted_style": predicted_style,
        "predicted_style_confidence": predicted_style_confidence,
        "predicted_style_source": predicted_style_source,
        "inferred_genre": None,
        "inferred_genre_confidence": 0.0,
        "possible_artist": None,
        "possible_artist_confidence": 0.0,
        "similar_paintings": matches,
    }


@dataclass
class BuildSummary:
    identity_model: str
    total_paintings: int
    armenian_paintings: int
    wikiart_paintings: int
    total_embeddings: int
    augmentations_per_painting: int
    include_wikiart: bool
    wikiart_sample_size: int
    index_path: str
    mapping_path: str
    embeddings_path: str
    yolo_model_path: str


class ArtRecognitionPipeline:
    def __init__(self, project_root: str | Path = ".") -> None:
        self.project_root = Path(project_root)
        self.paths = ProjectPaths(self.project_root)
        self.index_path = self.paths.faiss_index_path
        self.mapping_path = self.paths.mapping_path
        self.embeddings_path = self.paths.embeddings_path
        self.build_report_path = self.paths.build_report_path
        self.yolo_model_path = self.paths.data_dir / "models" / "painting_yolo_seg.pt"
        self._hash_rows: list[dict[str, object]] | None = None

    @staticmethod
    def _unique_mapping_rows(mapping: list[dict[str, object]]) -> list[dict[str, object]]:
        seen: set[object] = set()
        rows: list[dict[str, object]] = []
        for item in mapping:
            painting_id = item.get("painting_id")
            if painting_id in seen:
                continue
            seen.add(painting_id)
            rows.append(item)
        return rows

    def _hash_matches(
        self,
        variant: dict[str, object],
        mapping: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        query_hash = perceptual_hash(variant["processed_bgr"])
        if not query_hash:
            return []

        best_distance = 10**9
        best_rows: list[dict[str, object]] = []
        if self._hash_rows is None:
            self._hash_rows = []
            for row in self._unique_mapping_rows(mapping):
                row_hash = row.get("perceptual_hash")
                if not row_hash:
                    image_bgr = cv2.imread(str(row.get("image_path")))
                    row_hash = perceptual_hash(image_bgr) if image_bgr is not None else ""
                enriched = dict(row)
                enriched["perceptual_hash"] = row_hash
                self._hash_rows.append(enriched)
                distance = hash_distance(query_hash, str(row_hash))
                if distance == 0:
                    best_distance = 0
                    best_rows = [enriched]
                    break

        if best_distance != 0:
            for row in self._hash_rows:
                row_hash = row.get("perceptual_hash")
                distance = hash_distance(query_hash, str(row_hash))
                if distance < best_distance:
                    best_distance = distance
                    best_rows = [row]
                elif distance == best_distance:
                    best_rows.append(row)

        if best_distance > PERCEPTUAL_HASH_THRESHOLD:
            return []

        matches = []
        for rank, row in enumerate(best_rows[:5], start=1):
            matches.append(
                {
                    "rank": rank,
                    "score": float(1.0 - best_distance / 64.0),
                    "metadata": row,
                    "hit_count": 1,
                    "geometric_verification": {
                        "method": "perceptual_hash",
                        "distance": int(best_distance),
                        "inliers": GEOMETRIC_INLIER_THRESHOLD,
                        "matches": 1,
                        "verified": True,
                    },
                }
            )
        return matches

    def build_index(
        self,
        wikiart_sample_size: int = 0,
        wikiart_metadata_path: str | None = "",
        embedding_model: str = DEFAULT_DINOV2_MODEL,
        include_wikiart: bool = False,
        augmentations_per_painting: int = 16,
        progress_interval: int = 100,
    ) -> BuildSummary:
        print("Loading Armenian painting records...", flush=True)
        records = load_armenian_records(self.project_root)
        if include_wikiart:
            print(f"Loading WikiArt records (limit={wikiart_sample_size})...", flush=True)
            records.extend(
                load_wikiart_records(
                    sample_size=wikiart_sample_size,
                    metadata_path=wikiart_metadata_path,
                    output_dir=self.paths.wikiart_raw_dir,
                )
            )
        if not records:
            raise ValueError("no artwork records found")

        armenian_count = sum(1 for record in records if record.source == "armenian_local")
        wikiart_count = sum(1 for record in records if record.source == "wikiart")
        expected_embeddings = len(records) * augmentations_per_painting
        print(
            f"Loaded {len(records)} paintings: {armenian_count} Armenian, {wikiart_count} WikiArt.",
            flush=True,
        )
        print(f"Loading DINOv2 model: {embedding_model}", flush=True)
        extractor = Dinov2EmbeddingExtractor(model_name=embedding_model)
        print(
            f"Building FAISS embeddings with {augmentations_per_painting} augmentations per painting "
            f"({expected_embeddings} embeddings expected).",
            flush=True,
        )
        embeddings: list[np.ndarray] = []
        mapping: list[dict[str, object]] = []

        for painting_id, record in enumerate(records):
            if progress_interval > 0 and (painting_id == 0 or painting_id % max(progress_interval // max(augmentations_per_painting, 1), 1) == 0):
                print(
                    f"Processing painting {painting_id + 1} / {len(records)}: "
                    f"{record.source} / {record.filename}",
                    flush=True,
                )
            image_bgr = cv2.imread(record.image_path)
            if image_bgr is None:
                print(f"Skipping unreadable image: {record.image_path}", flush=True)
                continue

            variants = augment_clean_painting(
                image_bgr,
                count=augmentations_per_painting,
                seed=painting_id + 1729,
            )
            for augmentation_index, variant_bgr in enumerate(variants):
                image_rgb = cv2.cvtColor(variant_bgr, cv2.COLOR_BGR2RGB)
                embeddings.append(extractor.extract(image_rgb))
                mapping.append(_identity_mapping(record, painting_id, augmentation_index, embedding_model))
                if progress_interval > 0 and (len(embeddings) == 1 or len(embeddings) % progress_interval == 0):
                    print(f"Embedded {len(embeddings)} / {expected_embeddings} images...", flush=True)

        if not embeddings:
            raise ValueError("all Armenian images failed during DINOv2 embedding extraction")

        print("Writing FAISS index and metadata mapping...", flush=True)
        embeddings_array = normalize_matrix(np.vstack(embeddings).astype(np.float32))
        vector_db = ArtVectorDatabase(
            index_path=self.index_path,
            mapping_path=self.mapping_path,
            embeddings_path=self.embeddings_path,
        )
        vector_db.build(embeddings_array, mapping)

        summary = BuildSummary(
            identity_model=embedding_model,
            total_paintings=len({entry["painting_id"] for entry in mapping}),
            armenian_paintings=sum(1 for entry in mapping if entry.get("source") == "armenian_local" and entry.get("augmentation_index") == 0),
            wikiart_paintings=sum(1 for entry in mapping if entry.get("source") == "wikiart" and entry.get("augmentation_index") == 0),
            total_embeddings=len(mapping),
            augmentations_per_painting=augmentations_per_painting,
            include_wikiart=include_wikiart,
            wikiart_sample_size=wikiart_sample_size if include_wikiart else 0,
            index_path=str(self.index_path),
            mapping_path=str(self.mapping_path),
            embeddings_path=str(self.embeddings_path),
            yolo_model_path=str(self.yolo_model_path),
        )
        self.build_report_path.write_text(json.dumps(summary.__dict__, indent=2), encoding="utf-8")
        print(
            f"Done. Indexed {summary.total_paintings} paintings "
            f"({summary.total_embeddings} embeddings).",
            flush=True,
        )
        return summary

    def query(
        self,
        image_path: str | Path,
        embedding_model: str = DEFAULT_DINOV2_MODEL,
        top_k: int = 20,
    ) -> dict[str, object]:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise FileNotFoundError(f"could not read query image: {image_path}")

        vector_db = ArtVectorDatabase(
            index_path=self.index_path,
            mapping_path=self.mapping_path,
            embeddings_path=self.embeddings_path,
        ).load()
        indexed_model = str(vector_db.mapping[0].get("embedding_model") or "") if vector_db.mapping else ""
        if indexed_model and indexed_model != embedding_model:
            raise RuntimeError(
                f"The saved index was built with {indexed_model}. Rebuild it with {embedding_model} before querying."
            )

        best_result: dict[str, object] | None = None
        query_variants = preprocess_query_image_variants_from_bgr(image_bgr)
        for variant in query_variants:
            hash_matches = self._hash_matches(variant, vector_db.mapping)
            if hash_matches:
                result = build_query_response(image_path=image_path, matches=hash_matches)
                result["query_variant"] = variant["name"]
                result["crop_confidence"] = variant["crop_confidence"]
                return result

        embeddings = normalize_matrix(vector_db.load_embeddings())
        extractor = Dinov2EmbeddingExtractor(model_name=embedding_model)
        for variant in query_variants:
            query_embedding = extractor.extract(variant["processed_rgb"])
            raw_matches = _matches_from_embeddings(query_embedding, embeddings, vector_db.mapping, top_k=top_k)
            candidates = aggregate_identity_matches(raw_matches, max_candidates=5)
            matches = []
            for candidate in candidates:
                candidate_bgr = cv2.imread(str(candidate.metadata.get("image_path")))
                verification = {"method": "missing_candidate_image", "inliers": 0, "matches": 0, "verified": False}
                if candidate_bgr is not None:
                    verification = geometric_verify(variant["processed_bgr"], candidate_bgr)
                matches.append(
                    {
                        "rank": candidate.rank,
                        "score": candidate.score,
                        "metadata": candidate.metadata,
                        "hit_count": candidate.hit_count,
                        "geometric_verification": verification,
                    }
                )

            result = build_query_response(image_path=image_path, matches=matches)
            result["query_variant"] = variant["name"]
            result["crop_confidence"] = variant["crop_confidence"]
            if result["is_recognized"]:
                return result
            if best_result is None or result["recognition_score"] > best_result["recognition_score"]:
                best_result = result

        return best_result or build_query_response(image_path=image_path, matches=[])
