from __future__ import annotations

import contextlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import cv2
import faiss
import numpy as np
from PIL import Image

import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.utils import logging as transformers_logging

from art_recognition.cropping import remove_border

transformers_logging.set_verbosity_error()


DEFAULT_DINOV2_MODEL = "facebook/dinov2-base"
IDENTITY_EMBEDDING_THRESHOLD = 0.82
GEOMETRIC_INLIER_THRESHOLD = 35
PERCEPTUAL_HASH_THRESHOLD = 8


@contextlib.contextmanager
def _silence_model_loading():
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def perceptual_hash(image_bgr: np.ndarray, hash_size: int = 8, highfreq_factor: int = 4) -> str:
    if image_bgr is None or image_bgr.size == 0:
        return ""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    size = hash_size * highfreq_factor
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)
    low_freq = dct[:hash_size, :hash_size]
    median = np.median(low_freq[1:, 1:])
    bits = (low_freq > median).astype(np.uint8).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:0{hash_size * hash_size // 4}x}"


def hash_distance(first: str | None, second: str | None) -> int:
    if not first or not second:
        return 10**9
    return int((int(first, 16) ^ int(second, 16)).bit_count())


class Dinov2EmbeddingExtractor:
    def __init__(self, model_name: str = DEFAULT_DINOV2_MODEL) -> None:
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cache_root = Path(os.environ.get("ART_RECOGNITION_CACHE_DIR", "data/model_cache"))
        cache_root.mkdir(parents=True, exist_ok=True)

        with _silence_model_loading():
            self.processor = self._load_processor(model_name)
            self.model = self._load_model(model_name).to(self.device)
        self.model.eval()
        self.embedding_dim = int(getattr(self.model.config, "hidden_size", 768))

    @staticmethod
    def _load_processor(model_name: str):
        previous_offline = os.environ.get("HF_HUB_OFFLINE")
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            return AutoImageProcessor.from_pretrained(model_name, local_files_only=True)
        except OSError:
            if previous_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = previous_offline
            return AutoImageProcessor.from_pretrained(model_name)

    @staticmethod
    def _load_model(model_name: str):
        previous_offline = os.environ.get("HF_HUB_OFFLINE")
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            return AutoModel.from_pretrained(model_name, local_files_only=True)
        except OSError:
            if previous_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = previous_offline
            return AutoModel.from_pretrained(model_name)

    def extract(self, image_rgb: np.ndarray) -> np.ndarray:
        if image_rgb is None or image_rgb.size == 0:
            raise ValueError("image_rgb must be a non-empty RGB image")
        pil_image = Image.fromarray(image_rgb)
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, return_dict=True)
        features = getattr(outputs, "pooler_output", None)
        if not isinstance(features, torch.Tensor):
            hidden = outputs.last_hidden_state
            features = hidden[:, 0] if hidden.ndim == 3 else hidden
        return normalize_vector(features.detach().cpu().numpy().astype(np.float32))


def _jpeg_roundtrip(image_bgr: np.ndarray, quality: int) -> np.ndarray:
    ok, encoded = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return image_bgr
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return decoded if decoded is not None else image_bgr


def augment_clean_painting(image_bgr: np.ndarray, count: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    variants = [image_bgr]
    height, width = image_bgr.shape[:2]

    for _ in range(max(count - 1, 0)):
        image = image_bgr.copy()

        crop_ratio = float(rng.uniform(0.0, 0.06))
        image = remove_border(image, crop_ratio)

        alpha = float(rng.uniform(0.85, 1.18))
        beta = float(rng.uniform(-14, 14))
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        if rng.random() < 0.45:
            ksize = int(rng.choice([3, 5]))
            image = cv2.GaussianBlur(image, (ksize, ksize), 0)

        if rng.random() < 0.45:
            h, w = image.shape[:2]
            jitter = min(h, w) * float(rng.uniform(0.005, 0.025))
            src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
            dst = src + rng.normal(0, jitter, src.shape).astype(np.float32)
            matrix = cv2.getPerspectiveTransform(src, dst)
            image = cv2.warpPerspective(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

        if rng.random() < 0.5:
            image = _jpeg_roundtrip(image, int(rng.integers(55, 95)))

        if image.shape[0] != height or image.shape[1] != width:
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        variants.append(image)

    return variants


@dataclass
class AggregatedCandidate:
    metadata: dict[str, object]
    score: float
    rank: int
    hit_count: int


def aggregate_identity_matches(
    matches: list[dict[str, object]],
    max_candidates: int = 5,
) -> list[AggregatedCandidate]:
    grouped: dict[int, dict[str, object]] = {}
    for match in matches:
        metadata = match.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        painting_id = int(metadata.get("painting_id", metadata.get("row_index", -1)))
        if painting_id < 0:
            continue
        group = grouped.setdefault(
            painting_id,
            {
                "metadata": metadata,
                "scores": [],
            },
        )
        group["scores"].append(float(match.get("score") or 0.0))

    candidates: list[AggregatedCandidate] = []
    for group in grouped.values():
        scores = sorted(group["scores"], reverse=True)
        score = float(scores[0] + 0.03 * np.mean(scores[: min(5, len(scores))]))
        candidates.append(
            AggregatedCandidate(
                metadata=group["metadata"],
                score=score,
                rank=0,
                hit_count=len(scores),
            )
        )

    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    for rank, candidate in enumerate(candidates, start=1):
        candidate.rank = rank
    return candidates[:max_candidates]


def geometric_verify_orb(query_bgr: np.ndarray, candidate_bgr: np.ndarray) -> dict[str, object]:
    gray_query = cv2.cvtColor(query_bgr, cv2.COLOR_BGR2GRAY)
    gray_candidate = cv2.cvtColor(candidate_bgr, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=2500)
    kp1, des1 = orb.detectAndCompute(gray_query, None)
    kp2, des2 = orb.detectAndCompute(gray_candidate, None)
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return {"method": "orb_homography", "inliers": 0, "matches": 0, "verified": False}

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = matcher.knnMatch(des1, des2, k=2)
    good = []
    for pair in raw_matches:
        if len(pair) != 2:
            continue
        first, second = pair
        if first.distance < 0.75 * second.distance:
            good.append(first)

    if len(good) < 8:
        return {"method": "orb_homography", "inliers": 0, "matches": len(good), "verified": False}

    src = np.float32([kp1[match.queryIdx].pt for match in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[match.trainIdx].pt for match in good]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    inliers = int(mask.sum()) if mask is not None else 0
    return {
        "method": "orb_homography",
        "inliers": inliers,
        "matches": len(good),
        "verified": inliers >= GEOMETRIC_INLIER_THRESHOLD,
    }


def geometric_verify(query_bgr: np.ndarray, candidate_bgr: np.ndarray) -> dict[str, object]:
    lightglue_result = geometric_verify_lightglue(query_bgr, candidate_bgr)
    if lightglue_result is not None:
        return lightglue_result
    return geometric_verify_orb(query_bgr, candidate_bgr)


def geometric_verify_lightglue(query_bgr: np.ndarray, candidate_bgr: np.ndarray) -> dict[str, object] | None:
    try:
        from lightglue import LightGlue, SuperPoint
        from lightglue.utils import load_image, rbd
    except ImportError:
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    query_path = save_temp_image(query_bgr)
    candidate_path = save_temp_image(candidate_bgr)
    try:
        extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
        matcher = LightGlue(features="superpoint").eval().to(device)
        image0 = load_image(query_path).to(device)
        image1 = load_image(candidate_path).to(device)
        feats0 = extractor.extract(image0)
        feats1 = extractor.extract(image1)
        matches01 = matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, matches01 = [rbd(item) for item in [feats0, feats1, matches01]]
        matches = matches01["matches"].detach().cpu().numpy()
        if len(matches) < 8:
            return {"method": "lightglue_superpoint", "inliers": 0, "matches": int(len(matches)), "verified": False}

        points0 = feats0["keypoints"][matches[:, 0]].detach().cpu().numpy()
        points1 = feats1["keypoints"][matches[:, 1]].detach().cpu().numpy()
        _, mask = cv2.findHomography(points0.reshape(-1, 1, 2), points1.reshape(-1, 1, 2), cv2.RANSAC, 5.0)
        inliers = int(mask.sum()) if mask is not None else 0
        return {
            "method": "lightglue_superpoint",
            "inliers": inliers,
            "matches": int(len(matches)),
            "verified": inliers >= GEOMETRIC_INLIER_THRESHOLD,
        }
    except Exception:
        return None
    finally:
        for path in (query_path, candidate_path):
            with contextlib.suppress(OSError):
                Path(path).unlink()


def save_temp_image(image_bgr: np.ndarray) -> str:
    handle = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    handle.close()
    cv2.imwrite(handle.name, image_bgr)
    return handle.name
