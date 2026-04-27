from __future__ import annotations

import os
import pickle
import contextlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from torchvision.models import ResNet50_Weights, resnet50

os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from transformers import CLIPModel, CLIPProcessor
from transformers.utils import logging as transformers_logging

transformers_logging.set_verbosity_error()

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - depends on optional local install
    XGBClassifier = None


DEFAULT_CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"


@contextlib.contextmanager
def _silence_model_loading():
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


class EmbeddingExtractor:
    def __init__(self, model_name: str = "clip") -> None:
        self.model_name = model_name.lower()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim: int
        cache_root = Path(os.environ.get("ART_RECOGNITION_CACHE_DIR", "data/model_cache"))
        cache_root.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TORCH_HOME", str(cache_root / "torch"))

        if self.model_name == "clip":
            clip_name = DEFAULT_CLIP_MODEL_NAME
            with _silence_model_loading():
                self.model = self._load_clip_model(clip_name).to(self.device)
                self.processor = self._load_clip_processor(clip_name)
            self.model.eval()
            self.embedding_dim = self.model.config.projection_dim
        elif self.model_name == "resnet50":
            weights = ResNet50_Weights.DEFAULT
            self.model = resnet50(weights=weights)
            self.model.fc = torch.nn.Identity()
            self.transforms = weights.transforms()
            self.model = self.model.to(self.device)
            self.model.eval()
            self.embedding_dim = 2048
        else:
            raise ValueError("model_name must be 'clip' or 'resnet50'")

    def _load_clip_model(self, clip_name: str) -> CLIPModel:
        previous_offline = os.environ.get("HF_HUB_OFFLINE")
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            return CLIPModel.from_pretrained(clip_name, local_files_only=True, use_safetensors=False)
        except OSError:
            if previous_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = previous_offline
            return CLIPModel.from_pretrained(clip_name, use_safetensors=False)

    def _load_clip_processor(self, clip_name: str) -> CLIPProcessor:
        try:
            return CLIPProcessor.from_pretrained(clip_name, local_files_only=True)
        except OSError:
            return CLIPProcessor.from_pretrained(clip_name)

    def extract(self, image_rgb: np.ndarray) -> np.ndarray:
        if image_rgb is None or image_rgb.size == 0:
            raise ValueError("image_rgb must be a non-empty RGB array")

        pil_image = Image.fromarray(image_rgb)

        with torch.no_grad():
            if self.model_name == "clip":
                inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
                features = self.model.get_image_features(**inputs)
                if not isinstance(features, torch.Tensor):
                    if hasattr(features, "image_embeds") and features.image_embeds is not None:
                        features = features.image_embeds
                    elif hasattr(features, "pooler_output") and features.pooler_output is not None:
                        features = features.pooler_output
                    else:
                        raise TypeError("CLIP image encoder did not return tensor features")
            else:
                tensor = self.transforms(pil_image).unsqueeze(0).to(self.device)
                features = self.model(tensor)

        embedding = features.detach().cpu().numpy().astype(np.float32).reshape(-1)
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm

    def extract_texts(self, texts: list[str]) -> np.ndarray:
        if self.model_name != "clip":
            raise ValueError("zero-shot text embeddings require the CLIP embedding model")
        if not texts:
            raise ValueError("texts cannot be empty")

        with torch.no_grad():
            inputs = self.processor(text=texts, padding=True, return_tensors="pt").to(self.device)
            features = self.model.get_text_features(**inputs)
            if not isinstance(features, torch.Tensor):
                if hasattr(features, "text_embeds") and features.text_embeds is not None:
                    features = features.text_embeds
                elif hasattr(features, "pooler_output") and features.pooler_output is not None:
                    features = features.pooler_output
                else:
                    raise TypeError("CLIP text encoder did not return tensor features")

        embeddings = features.detach().cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


class StyleClassifier:
    def __init__(
        self,
        model: object | None = None,
        label_encoder: LabelEncoder | None = None,
        backend: str | None = None,
    ) -> None:
        self.model = model
        self.label_encoder = label_encoder
        self.backend = backend

    @property
    def is_trained(self) -> bool:
        return self.model is not None and self.label_encoder is not None

    @property
    def classes_(self) -> list[str]:
        if self.label_encoder is None:
            return []
        return [str(label) for label in self.label_encoder.classes_]

    def fit(self, embeddings: np.ndarray, labels: list[str]) -> None:
        if len(embeddings) == 0:
            raise ValueError("embeddings cannot be empty")

        x = np.asarray(embeddings, dtype=np.float32)
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(labels)

        if XGBClassifier is not None:
            self.backend = "xgboost"
            self.model = XGBClassifier(
                objective="multi:softprob",
                eval_metric="mlogloss",
                n_estimators=350,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                tree_method="hist",
                random_state=42,
                n_jobs=-1,
            )
        else:
            self.backend = "hist_gradient_boosting"
            self.model = HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                l2_regularization=0.1,
                random_state=42,
            )

        self.model.fit(x, y)

    def predict(self, embedding: np.ndarray) -> tuple[str | None, float]:
        if self.model is None or self.label_encoder is None:
            return None, 0.0

        sample = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        encoded_label = int(self.model.predict(sample)[0])
        label = str(self.label_encoder.inverse_transform([encoded_label])[0])
        if hasattr(self.model, "predict_proba"):
            probability = float(np.max(self.model.predict_proba(sample)[0]))
        else:
            probability = 0.0
        return label, probability

    def save(self, path: str | Path) -> None:
        if self.model is None or self.label_encoder is None:
            raise ValueError("cannot save an untrained classifier")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(
                {
                    "model": self.model,
                    "label_encoder": self.label_encoder,
                    "backend": self.backend,
                },
                fh,
            )

    @classmethod
    def load(cls, path: str | Path) -> "StyleClassifier":
        with Path(path).open("rb") as fh:
            payload = pickle.load(fh)
        if isinstance(payload, dict) and "model" in payload:
            return cls(
                model=payload.get("model"),
                label_encoder=payload.get("label_encoder"),
                backend=payload.get("backend"),
            )

        # Backward compatibility for older LogisticRegression pipeline artifacts.
        legacy_classes = getattr(payload, "classes_", None)
        if legacy_classes is None and hasattr(payload, "named_steps"):
            classifier = payload.named_steps.get("classifier")
            legacy_classes = getattr(classifier, "classes_", None)
        label_encoder = None
        if legacy_classes is not None:
            label_encoder = LabelEncoder()
            label_encoder.fit([str(label) for label in legacy_classes])
        return cls(model=payload, label_encoder=label_encoder, backend="legacy_pipeline")


class ClipZeroShotStylePredictor:
    def __init__(
        self,
        extractor: EmbeddingExtractor,
        style_labels: list[str],
        prompt_template: str = "a painting in the style of {style}",
    ) -> None:
        if extractor.model_name != "clip":
            raise ValueError("CLIP zero-shot style prediction requires a CLIP extractor")
        cleaned_labels = sorted({str(label).replace("_", " ").strip() for label in style_labels if str(label).strip()})
        if not cleaned_labels:
            raise ValueError("style_labels cannot be empty")
        self.extractor = extractor
        self.style_labels = cleaned_labels
        self.prompts = [prompt_template.format(style=label.lower()) for label in self.style_labels]
        self.text_embeddings = extractor.extract_texts(self.prompts)

    def predict(self, image_embedding: np.ndarray) -> tuple[str, float]:
        query = normalize_embedding(image_embedding)
        scores = self.text_embeddings @ query
        probabilities = _softmax(scores)
        index = int(np.argmax(probabilities))
        return self.style_labels[index], float(probabilities[index])


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores.astype(np.float32) - float(np.max(scores))
    exp_scores = np.exp(shifted)
    total = float(np.sum(exp_scores))
    if total == 0:
        return np.ones_like(exp_scores) / len(exp_scores)
    return exp_scores / total


def predict_style_with_fallback(
    embedding: np.ndarray,
    classifier: StyleClassifier | None,
    zero_shot: ClipZeroShotStylePredictor | None = None,
    min_classifier_confidence: float = 0.35,
) -> tuple[str | None, float, str]:
    classifier_label = None
    classifier_confidence = 0.0
    if classifier is not None:
        classifier_label, classifier_confidence = classifier.predict(embedding)

    if classifier_label and classifier_confidence >= min_classifier_confidence:
        return classifier_label, classifier_confidence, "classifier"

    if zero_shot is not None:
        zero_shot_label, zero_shot_confidence = zero_shot.predict(embedding)
        if not classifier_label or zero_shot_confidence > classifier_confidence:
            return zero_shot_label, zero_shot_confidence, "clip_zero_shot"

    return classifier_label, classifier_confidence, "classifier" if classifier_label else "none"
