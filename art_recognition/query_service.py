from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from art_recognition.identity import DEFAULT_DINOV2_MODEL
from art_recognition.pipeline import ArtRecognitionPipeline


LOGGER = logging.getLogger(__name__)


class QueryService:
    def __init__(
        self,
        project_root: str | Path = ".",
        embedding_model: str = DEFAULT_DINOV2_MODEL,
        top_k: int = 20,
    ) -> None:
        self.project_root = Path(project_root)
        self.embedding_model = embedding_model
        self.default_top_k = top_k
        self.pipeline = ArtRecognitionPipeline(project_root=self.project_root)

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
                "geometric_inliers": None,
                "geometric_inlier_threshold": None,
                "variant_scores": {},
            },
            "warnings": [],
            "errors": [],
        }

    def query_image(self, image_path: str | Path, top_k: int = 20) -> dict[str, Any]:
        requested_top_k = int(top_k or self.default_top_k)
        response = self._base_response(image_path=image_path, top_k=requested_top_k)
        query_path = Path(image_path)

        if not query_path.exists() or not query_path.is_file():
            response["recognition_decision"]["reason"] = "invalid_query_path"
            response["errors"].append(f"Query image does not exist or is not a file: {query_path}")
            return response

        try:
            result = self.pipeline.query(
                query_path,
                embedding_model=self.embedding_model,
                top_k=requested_top_k,
            )
        except Exception as exc:  # pragma: no cover - service boundary
            LOGGER.exception("Failed to query image %s", query_path)
            response["recognition_decision"]["reason"] = "query_failed"
            response["errors"].append(str(exc))
            return response

        recognized = bool(result.get("is_recognized"))
        score = result.get("recognition_score")
        best_match = None
        if result.get("similar_paintings"):
            best_match = result["similar_paintings"][0].get("metadata")

        return {
            "query_path": str(query_path),
            "model_name": self.embedding_model,
            "top_k": requested_top_k,
            "best_match": best_match,
            "top_matches": result.get("similar_paintings", []),
            "recognition_decision": {
                "recognized": recognized,
                "reason": "score_and_geometry_passed" if recognized else "score_or_geometry_below_threshold",
                "score": float(score) if score is not None else None,
            },
            "style_prediction": None,
            "variant_used": result.get("query_variant"),
            "scores": {
                "best_variant_score": float(score) if score is not None else None,
                "best_variant_name": result.get("query_variant"),
                "recognition_threshold": result.get("recognition_threshold"),
                "geometric_inliers": result.get("geometric_inliers"),
                "geometric_inlier_threshold": result.get("geometric_inlier_threshold"),
                "variant_scores": {str(result.get("query_variant")): float(score) if score is not None else None},
            },
            "warnings": [],
            "errors": [],
        }
