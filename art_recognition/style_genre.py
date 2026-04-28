from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


STYLE_MIN_CLASS_COUNT = 2
GENRE_MIN_CLASS_COUNT = 2


GENRE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "portrait": ("portrait", "self-portrait", "head of", "bust of", "lady", "woman", "man", "girl", "boy"),
    "landscape": ("landscape", "view of", "mountain", "valley", "forest", "river", "lake", "field", "garden", "tree"),
    "seascape": ("sea", "seashore", "coast", "harbor", "harbour", "beach", "boat", "ship", "marine", "wave"),
    "cityscape": ("city", "street", "square", "bridge", "paris", "venice", "village", "town", "church"),
    "still life": ("still-life", "still life", "flowers", "fruit", "vase", "bottle", "apples", "table"),
    "religious": ("madonna", "christ", "crucifixion", "saint", "angel", "virgin", "annunciation", "adoration"),
    "mythological": ("venus", "apollo", "diana", "bacchus", "nymph", "myth", "cupid", "mars"),
    "interior": ("interior", "room", "studio", "window", "kitchen"),
    "abstract": ("abstract", "composition", "untitled", "study", "improvisation"),
}


def clean_label(value: object) -> str | None:
    text = str(value or "").replace("_", " ").strip()
    if not text or text.lower() in {"none", "nan", "unknown"}:
        return None
    return re.sub(r"\s+", " ", text).title()


def infer_genre_label(metadata: dict[str, object]) -> str | None:
    explicit = clean_label(metadata.get("genre"))
    if explicit:
        return explicit

    text = " ".join(
        str(metadata.get(key) or "")
        for key in ("painting_name", "title", "filename")
    ).replace("_", " ").replace("-", " ").lower()
    for genre, keywords in GENRE_KEYWORDS.items():
        if any(keyword.replace("-", " ") in text for keyword in keywords):
            return genre.title()
    return None


@dataclass
class LabelPrediction:
    label: str | None
    confidence: float
    source: str


class _LabelModel:
    def __init__(self, model=None, label_encoder: LabelEncoder | None = None) -> None:
        self.model = model
        self.label_encoder = label_encoder

    @property
    def is_trained(self) -> bool:
        return self.model is not None and self.label_encoder is not None

    @property
    def classes_(self) -> list[str]:
        if self.label_encoder is None:
            return []
        return [str(label) for label in self.label_encoder.classes_]

    def fit(self, embeddings: np.ndarray, labels: list[str]) -> None:
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(labels)
        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1200,
                class_weight="balanced",
                solver="lbfgs",
                random_state=42,
            ),
        )
        self.model.fit(np.asarray(embeddings, dtype=np.float32), y)

    def predict(self, embedding: np.ndarray, source: str) -> LabelPrediction:
        if not self.is_trained:
            return LabelPrediction(None, 0.0, "not_trained")
        sample = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        encoded = int(self.model.predict(sample)[0])
        label = str(self.label_encoder.inverse_transform([encoded])[0])
        confidence = 0.0
        if hasattr(self.model, "predict_proba"):
            confidence = float(np.max(self.model.predict_proba(sample)[0]))
        return LabelPrediction(label, confidence, source)


class StyleGenrePredictor:
    def __init__(
        self,
        style_model: _LabelModel | None = None,
        genre_model: _LabelModel | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.style_model = style_model or _LabelModel()
        self.genre_model = genre_model or _LabelModel()
        self.metadata = metadata or {}

    @staticmethod
    def _enough_labels(labels: list[str], min_count: int) -> bool:
        counts: dict[str, int] = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        return sum(1 for count in counts.values() if count >= min_count) >= 2

    @classmethod
    def fit_from_index(
        cls,
        embeddings: np.ndarray,
        mapping: list[dict[str, object]],
        source: str = "wikiart",
    ) -> "StyleGenrePredictor":
        style_embeddings: list[np.ndarray] = []
        style_labels: list[str] = []
        genre_embeddings: list[np.ndarray] = []
        genre_labels: list[str] = []

        for index, metadata in enumerate(mapping):
            if source != "all" and metadata.get("source") != source:
                continue
            style = clean_label(metadata.get("style"))
            if style:
                style_embeddings.append(embeddings[index])
                style_labels.append(style)

            genre = infer_genre_label(metadata)
            if genre:
                genre_embeddings.append(embeddings[index])
                genre_labels.append(genre)

        predictor = cls(
            metadata={
                "training_source": source,
                "style_examples": len(style_labels),
                "genre_examples": len(genre_labels),
            }
        )
        if style_labels and cls._enough_labels(style_labels, STYLE_MIN_CLASS_COUNT):
            predictor.style_model.fit(np.vstack(style_embeddings), style_labels)
            predictor.metadata["style_classes"] = predictor.style_model.classes_
        if genre_labels and cls._enough_labels(genre_labels, GENRE_MIN_CLASS_COUNT):
            predictor.genre_model.fit(np.vstack(genre_embeddings), genre_labels)
            predictor.metadata["genre_classes"] = predictor.genre_model.classes_
        return predictor

    def predict_style(self, embedding: np.ndarray) -> LabelPrediction:
        return self.style_model.predict(embedding, "dinov2_wikiart_style_classifier")

    def predict_genre(self, embedding: np.ndarray) -> LabelPrediction:
        return self.genre_model.predict(embedding, "dinov2_title_genre_classifier")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: str | Path) -> "StyleGenrePredictor":
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        if isinstance(payload, cls):
            return payload
        raise ValueError(f"unsupported style/genre classifier artifact: {path}")
