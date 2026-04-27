from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2

from art_recognition.config import ProjectPaths
from art_recognition.database import ArtVectorDatabase
from art_recognition.ml_models import (
    ClipZeroShotStylePredictor,
    EmbeddingExtractor,
    StyleClassifier,
    predict_style_with_fallback,
)
from art_recognition.pipeline import build_query_response, preprocess_query_image_variants


LOGGER = logging.getLogger(__name__)


class QueryService:
    def __init__(self, project_root: str | Path = ".", embedding_model: str = "clip", top_k: int = 5) -> None:
        self.project_root = Path(project_root)
        self.embedding_model = embedding_model
        self.default_top_k = top_k
        self.paths = ProjectPaths(self.project_root)
        self.vector_db = ArtVectorDatabase(
            index_path=self.paths.faiss_index_path,
            mapping_path=self.paths.mapping_path,
            embeddings_path=self.paths.embeddings_path,
        )
        self._extractor: EmbeddingExtractor | None = None
        self._classifier: StyleClassifier | None = None
        self._zero_shot: ClipZeroShotStylePredictor | None = None
        self._artifacts_loaded = False

    def _base_response(self, image_path: str | Path, top_k: int) -> dict[str, Any]:
        return {
            "query_path": str(image_path),
            "model_name": self.embedding_model,
            "top_k": int(top_k),
            "best_match": None,
            "top_matches": [],
            "recognition_decision": {
                "recognized": False,
                "reason": "query_not_processed",
                "score": None,
            },
            "style_prediction": None,
            "variant_used": None,
            "scores": {
                "best_variant_score": None,
                "best_variant_name": None,
                "recognition_threshold": None,
                "variant_scores": {},
            },
            "warnings": [],
            "errors": [],
        }

    def _ensure_artifacts_loaded(self) -> list[str]:
        missing = []
        for artifact in (
            self.paths.faiss_index_path,
            self.paths.mapping_path,
            self.paths.embeddings_path,
        ):
            if not artifact.exists():
                missing.append(str(artifact))

        if missing:
            return missing

        if not self._artifacts_loaded:
            LOGGER.info("Loading query artifacts from %s", self.paths.data_dir)
            self.vector_db.load()
            # Also validate that embeddings.npy exists and is readable.
            self.vector_db.load_embeddings()
            indexed_model = str(self.vector_db.mapping[0].get("embedding_model") or "").lower() if self.vector_db.mapping else ""
            if indexed_model and indexed_model != self.embedding_model.lower():
                raise RuntimeError(
                    f"The saved index was built with {indexed_model}. Rebuild it with {self.embedding_model} before querying."
                )
            self._artifacts_loaded = True

        if self._extractor is None:
            LOGGER.info("Loading embedding extractor: %s", self.embedding_model)
            self._extractor = EmbeddingExtractor(model_name=self.embedding_model)

        if self._classifier is None and self.paths.classifier_path.exists():
            LOGGER.info("Loading style classifier from %s", self.paths.classifier_path)
            self._classifier = StyleClassifier.load(self.paths.classifier_path)

        if self._zero_shot is None and self._extractor is not None and self._extractor.model_name == "clip":
            labels = self._classifier.classes_ if self._classifier is not None else sorted(
                {str(record.get("style")) for record in self.vector_db.mapping if record.get("style")}
            )
            if labels:
                LOGGER.info("Preparing CLIP zero-shot style prompts for %d styles", len(labels))
                self._zero_shot = ClipZeroShotStylePredictor(self._extractor, labels)

        return []

    def _response_from_pipeline_result(
        self,
        image_path: str | Path,
        top_k: int,
        variant_name: str,
        pipeline_result: dict[str, Any],
    ) -> dict[str, Any]:
        recognition_score = pipeline_result.get("recognition_score")
        threshold = pipeline_result.get("recognition_threshold")
        recognized = bool(pipeline_result.get("is_recognized"))

        if recognized:
            reason = "score_above_threshold"
        elif pipeline_result.get("similar_paintings"):
            reason = "best_score_below_threshold"
        else:
            reason = "no_matches_found"

        style_label = pipeline_result.get("predicted_style")
        style_confidence = float(pipeline_result.get("predicted_style_confidence") or 0.0)
        style_prediction = None
        if style_label:
            style_prediction = {
                "label": str(style_label),
                "confidence": style_confidence,
                "source": pipeline_result.get("predicted_style_source"),
            }

        return {
            "query_path": str(image_path),
            "model_name": self.embedding_model,
            "top_k": int(top_k),
            "best_match": pipeline_result.get("similar_paintings", [{}])[0].get("metadata")
            if pipeline_result.get("similar_paintings")
            else None,
            "top_matches": pipeline_result.get("similar_paintings", []),
            "recognition_decision": {
                "recognized": recognized,
                "reason": reason,
                "score": float(recognition_score) if recognition_score is not None else None,
            },
            "style_prediction": style_prediction,
            "variant_used": variant_name,
            "scores": {
                "best_variant_score": float(recognition_score) if recognition_score is not None else None,
                "best_variant_name": variant_name,
                "recognition_threshold": float(threshold) if threshold is not None else None,
                "variant_scores": {variant_name: float(recognition_score) if recognition_score is not None else None},
            },
            "warnings": [],
            "errors": [],
        }

    def query_image(self, image_path: str | Path, top_k: int = 5) -> dict[str, Any]:
        requested_top_k = int(top_k or self.default_top_k)
        response = self._base_response(image_path=image_path, top_k=requested_top_k)
        query_path = Path(image_path)

        if not query_path.exists():
            message = f"Query image does not exist: {query_path}"
            LOGGER.error(message)
            response["recognition_decision"]["reason"] = "missing_query_image"
            response["errors"].append(message)
            return response

        if not query_path.is_file():
            message = f"Query path is not a file: {query_path}"
            LOGGER.error(message)
            response["recognition_decision"]["reason"] = "invalid_query_path"
            response["errors"].append(message)
            return response

        image = cv2.imread(str(query_path))
        if image is None:
            message = f"Could not read query image: {query_path}"
            LOGGER.error(message)
            response["recognition_decision"]["reason"] = "unreadable_query_image"
            response["errors"].append(message)
            return response

        try:
            missing_artifacts = self._ensure_artifacts_loaded()
        except Exception as exc:  # pragma: no cover - defensive wrapper
            LOGGER.exception("Failed to load query artifacts for %s", query_path)
            response["recognition_decision"]["reason"] = "artifact_loading_failed"
            response["errors"].append(f"Failed to load query artifacts: {exc}")
            return response

        if missing_artifacts:
            message = f"Required artifacts are missing: {', '.join(missing_artifacts)}"
            LOGGER.error(message)
            response["recognition_decision"]["reason"] = "missing_artifacts"
            response["errors"].append(message)
            return response

        if self._extractor is None:
            message = "Embedding extractor was not initialized."
            LOGGER.error(message)
            response["recognition_decision"]["reason"] = "extractor_unavailable"
            response["errors"].append(message)
            return response

        classifier = self._classifier
        if classifier is None:
            warning = f"Style classifier not found at {self.paths.classifier_path}; style prediction will be skipped."
            LOGGER.warning(warning)
            response["warnings"].append(warning)

        try:
            variants = preprocess_query_image_variants(query_path)
        except Exception as exc:
            LOGGER.exception("Failed to preprocess query image variants for %s", query_path)
            response["recognition_decision"]["reason"] = "preprocessing_failed"
            response["errors"].append(f"Failed to preprocess query image: {exc}")
            return response

        if not variants:
            message = "No query variants were generated for the image."
            LOGGER.error(message)
            response["recognition_decision"]["reason"] = "no_query_variants"
            response["errors"].append(message)
            return response

        LOGGER.info("Running query over %d variants for %s", len(variants), query_path)
        best_response: dict[str, Any] | None = None

        for variant in variants:
            variant_name = str(variant.get("name") or "unknown_variant")
            processed_rgb = variant.get("processed_rgb")
            try:
                query_embedding = self._extractor.extract(processed_rgb)
                matches = self.vector_db.export_matches_with_numpy(query_embedding, k=requested_top_k)

                predicted_style, predicted_style_confidence, predicted_style_source = predict_style_with_fallback(
                    query_embedding,
                    classifier,
                    self._zero_shot,
                )

                pipeline_result = build_query_response(
                    image_path=query_path,
                    matches=matches,
                    predicted_style=predicted_style,
                    predicted_style_confidence=predicted_style_confidence,
                    predicted_style_source=predicted_style_source,
                )
                candidate_response = self._response_from_pipeline_result(
                    image_path=query_path,
                    top_k=requested_top_k,
                    variant_name=variant_name,
                    pipeline_result=pipeline_result,
                )
                candidate_response["scores"]["variant_scores"] = {
                    variant_name: candidate_response["recognition_decision"]["score"]
                }
            except Exception as exc:
                LOGGER.exception("Variant %s failed for %s", variant_name, query_path)
                response["warnings"].append(f"Variant '{variant_name}' failed: {exc}")
                continue

            current_score = candidate_response["recognition_decision"]["score"] or float("-inf")
            response["scores"]["variant_scores"][variant_name] = (
                float(current_score) if current_score != float("-inf") else None
            )

            if candidate_response["recognition_decision"]["recognized"]:
                best_response = candidate_response
                break

            if best_response is None:
                best_response = candidate_response
                continue

            best_score = best_response["recognition_decision"]["score"]
            best_score_value = float(best_score) if best_score is not None else float("-inf")
            if current_score > best_score_value:
                best_response = candidate_response

        if best_response is None:
            response["recognition_decision"]["reason"] = "all_variants_failed"
            if not response["warnings"]:
                response["errors"].append("No query result could be produced.")
            else:
                response["errors"].append("All query variants failed.")
            return response

        best_response["warnings"].extend(response["warnings"])
        best_response["errors"].extend(response["errors"])
        best_response["scores"]["variant_scores"] = response["scores"]["variant_scores"]
        LOGGER.info(
            "Best query result for %s used variant %s with score %s",
            query_path,
            best_response.get("variant_used"),
            best_response["recognition_decision"].get("score"),
        )
        return best_response
