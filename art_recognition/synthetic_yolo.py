from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _random_wall(rng: np.random.Generator, height: int, width: int) -> np.ndarray:
    base = rng.integers(170, 235, size=(1, 1, 3), dtype=np.uint8)
    noise = rng.normal(0, 8, size=(height, width, 3)).astype(np.int16)
    wall = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(wall, (15, 15), 0)


def _yolo_seg_line(points: np.ndarray, width: int, height: int) -> str:
    normalized = points.astype(np.float32).copy()
    normalized[:, 0] /= float(width)
    normalized[:, 1] /= float(height)
    coords = " ".join(f"{value:.6f}" for value in normalized.reshape(-1))
    return f"0 {coords}"


def make_synthetic_detection_sample(
    painting_bgr: np.ndarray,
    rng: np.random.Generator,
    canvas_size: tuple[int, int] = (768, 768),
) -> tuple[np.ndarray, np.ndarray]:
    canvas_h, canvas_w = canvas_size
    wall = _random_wall(rng, canvas_h, canvas_w)

    src_h, src_w = painting_bgr.shape[:2]
    scale = float(rng.uniform(0.45, 0.82) * min(canvas_w / src_w, canvas_h / src_h))
    new_w = max(int(src_w * scale), 32)
    new_h = max(int(src_h * scale), 32)
    painting = cv2.resize(painting_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    frame = int(rng.integers(10, 45))
    framed = cv2.copyMakeBorder(
        painting,
        frame,
        frame,
        frame,
        frame,
        cv2.BORDER_CONSTANT,
        value=tuple(int(v) for v in rng.integers(25, 95, size=3)),
    )
    fh, fw = framed.shape[:2]
    max_fh = canvas_h - 60
    max_fw = canvas_w - 60
    if fh > max_fh or fw > max_fw:
        resize_scale = min(max_fw / fw, max_fh / fh)
        fw = max(int(fw * resize_scale), 32)
        fh = max(int(fh * resize_scale), 32)
        framed = cv2.resize(framed, (fw, fh), interpolation=cv2.INTER_AREA)
        frame = max(int(frame * resize_scale), 1)
        new_w = max(fw - 2 * frame, 8)
        new_h = max(fh - 2 * frame, 8)
    x = int(rng.integers(20, max(21, canvas_w - fw - 20)))
    y = int(rng.integers(20, max(21, canvas_h - fh - 20)))

    src = np.array([[0, 0], [fw - 1, 0], [fw - 1, fh - 1], [0, fh - 1]], dtype=np.float32)
    jitter = min(fw, fh) * float(rng.uniform(0.01, 0.06))
    dst = src + np.array([x, y], dtype=np.float32) + rng.normal(0, jitter, src.shape).astype(np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(framed, matrix, (canvas_w, canvas_h), borderMode=cv2.BORDER_TRANSPARENT)
    mask = cv2.warpPerspective(np.full((fh, fw), 255, dtype=np.uint8), matrix, (canvas_w, canvas_h))
    wall[mask > 0] = warped[mask > 0]

    painting_src = np.array(
        [[frame, frame], [frame + new_w - 1, frame], [frame + new_w - 1, frame + new_h - 1], [frame, frame + new_h - 1]],
        dtype=np.float32,
    )
    painting_polygon = cv2.perspectiveTransform(painting_src.reshape(-1, 1, 2), matrix).reshape(-1, 2)

    alpha = float(rng.uniform(0.85, 1.15))
    beta = float(rng.uniform(-18, 18))
    wall = cv2.convertScaleAbs(wall, alpha=alpha, beta=beta)
    if rng.random() < 0.5:
        wall = cv2.GaussianBlur(wall, (3, 3), 0)
    if rng.random() < 0.55:
        ok, encoded = cv2.imencode(".jpg", wall, [int(cv2.IMWRITE_JPEG_QUALITY), int(rng.integers(45, 90))])
        if ok:
            wall = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    return wall, painting_polygon


def write_yolo_dataset(
    image_paths: list[Path],
    output_dir: Path,
    samples_per_image: int = 12,
    seed: int = 42,
) -> None:
    rng = np.random.default_rng(seed)
    train_images = output_dir / "images" / "train"
    train_labels = output_dir / "labels" / "train"
    train_images.mkdir(parents=True, exist_ok=True)
    train_labels.mkdir(parents=True, exist_ok=True)

    sample_index = 0
    for image_path in image_paths:
        painting = cv2.imread(str(image_path))
        if painting is None:
            continue
        for _ in range(samples_per_image):
            synthetic, polygon = make_synthetic_detection_sample(painting, rng)
            image_name = f"synthetic_{sample_index:06d}.jpg"
            label_name = f"synthetic_{sample_index:06d}.txt"
            cv2.imwrite(str(train_images / image_name), synthetic)
            (train_labels / label_name).write_text(
                _yolo_seg_line(polygon, synthetic.shape[1], synthetic.shape[0]) + "\n",
                encoding="utf-8",
            )
            sample_index += 1

    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {output_dir.resolve()}",
                "train: images/train",
                "val: images/train",
                "names:",
                "  0: painting",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
