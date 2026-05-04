#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from art_recognition.cropping import PaintingCropper
from art_recognition.preprocessing import preprocess_gallery_image


SOURCE = ROOT / "data/datasets/armenian/images/a_cup_of_coffee_2001_grigor_aghasyan.JPG"
OUT = ROOT / "paper_metrics"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def fit_for_display(image_bgr: np.ndarray, max_width: int = 1600) -> tuple[np.ndarray, float]:
    height, width = image_bgr.shape[:2]
    scale = min(1.0, max_width / width)
    if scale == 1.0:
        return image_bgr.copy(), 1.0
    resized = cv2.resize(image_bgr, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def polygon_for_drawing(corners: np.ndarray) -> np.ndarray:
    if len(corners) != 4:
        return corners
    # preprocess_gallery_image returns top-left, top-right, bottom-left, bottom-right.
    return corners[[0, 1, 3, 2]]


def perspective_crop_from_corners(image_bgr: np.ndarray, corners: np.ndarray) -> np.ndarray:
    if len(corners) != 4:
        return image_bgr.copy()
    ordered = polygon_for_drawing(corners).astype(np.float32)
    tl, tr, br, bl = ordered
    width = max(int(np.linalg.norm(tr - tl)), int(np.linalg.norm(br - bl)), 16)
    height = max(int(np.linalg.norm(bl - tl)), int(np.linalg.norm(br - tr)), 16)
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(ordered, dst)
    return cv2.warpPerspective(image_bgr, matrix, (width, height))


def remove_fractional_border(image_bgr: np.ndarray, ratio: float = 0.08) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    mx, my = int(w * ratio), int(h * ratio)
    if mx * 2 >= w or my * 2 >= h:
        return image_bgr
    return image_bgr[my : h - my, mx : w - mx]


def draw_detection_overlay(image_bgr: np.ndarray) -> tuple[np.ndarray, dict[str, object], np.ndarray]:
    result = preprocess_gallery_image(image_bgr)
    candidates = result.get("candidates") or []
    if not candidates:
        raise RuntimeError("no painting candidate found")
    candidate = max(candidates, key=lambda item: item.bounding_rect[2] * item.bounding_rect[3])
    display, scale = fit_for_display(image_bgr)

    x, y, w, h = candidate.bounding_rect
    rect = np.array([x, y, x + w, y + h], dtype=np.float32) * scale
    original_corners = candidate.painting_corners.astype(np.float32)
    corners = polygon_for_drawing(original_corners) * scale

    overlay = display.copy()
    cv2.rectangle(
        overlay,
        (int(rect[0]), int(rect[1])),
        (int(rect[2]), int(rect[3])),
        (0, 145, 255),
        8,
    )
    if len(corners) == 4:
        cv2.polylines(overlay, [corners.astype(np.int32).reshape(-1, 1, 2)], True, (255, 220, 0), 9)
        for point in corners.astype(np.int32):
            cv2.circle(overlay, tuple(point), 18, (255, 220, 0), -1)
            cv2.circle(overlay, tuple(point), 18, (255, 255, 255), 4)

    pil = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)).convert("RGBA")
    draw = ImageDraw.Draw(pil)
    title_font = font(42, True)
    label_font = font(26, True)
    body_font = font(22)
    draw.rounded_rectangle([34, 34, 812, 135], radius=20, fill=(18, 24, 30, 218))
    draw.text((62, 52), "Preprocessing on real museum photo", font=title_font, fill=(255, 255, 255, 255))
    draw.text((64, 99), "detect frame, isolate painting, normalize input", font=body_font, fill=(220, 232, 238, 255))

    draw.rounded_rectangle([48, 146, 454, 238], radius=16, fill=(255, 255, 255, 230), outline=(0, 145, 255, 255), width=5)
    draw.text((72, 162), "Frame / wall candidate", font=label_font, fill=(119, 70, 0, 255))
    draw.text((72, 199), "orange box is removed from embedding", font=body_font, fill=(45, 45, 45, 255))

    draw.rounded_rectangle([918, 144, 1544, 238], radius=16, fill=(255, 255, 255, 230), outline=(255, 220, 0, 255), width=5)
    draw.text((946, 162), "Detected painting quadrilateral", font=label_font, fill=(36, 92, 112, 255))
    draw.text((946, 199), "corners define crop and perspective correction", font=body_font, fill=(45, 45, 45, 255))

    return cv2.cvtColor(np.asarray(pil.convert("RGB")), cv2.COLOR_RGB2BGR), {
        "method": "legacy_contour",
        "candidate_count": len(candidates),
        "bounding_rect": candidate.bounding_rect,
        "painting_corners": original_corners.tolist() if len(original_corners) == 4 else [],
    }, original_corners


def make_combined_figure(overlay_bgr: np.ndarray, crop_bgr: np.ndarray, final_bgr: np.ndarray) -> Image.Image:
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)

    overlay_img = Image.fromarray(overlay_rgb)
    overlay_img.thumbnail((1500, 870), Image.Resampling.LANCZOS)
    crop_img = Image.fromarray(crop_rgb)
    crop_img.thumbnail((510, 350), Image.Resampling.LANCZOS)
    final_img = Image.fromarray(final_rgb)
    final_img.thumbnail((650, 350), Image.Resampling.LANCZOS)

    width = 1600
    height = overlay_img.height + 640
    canvas = Image.new("RGB", (width, height), (246, 246, 244))
    canvas.paste(overlay_img, ((width - overlay_img.width) // 2, 24))

    draw = ImageDraw.Draw(canvas)
    title_font = font(30, True)
    body_font = font(22)
    y0 = overlay_img.height + 56
    draw.rounded_rectangle([70, y0, 510, y0 + 114], radius=18, fill=(232, 240, 243), outline=(0, 145, 255), width=4)
    draw.text((98, y0 + 22), "1. Crop detected region", font=title_font, fill=(20, 80, 101))
    draw.text((98, y0 + 64), "frame and wall are excluded", font=body_font, fill=(52, 60, 66))
    canvas.paste(crop_img, (600, y0 - 30))
    draw.rectangle([600, y0 - 30, 600 + crop_img.width, y0 - 30 + crop_img.height], outline=(0, 145, 255), width=5)

    y1 = y0 + 220
    draw.rounded_rectangle([70, y1, 510, y1 + 114], radius=18, fill=(232, 240, 243), outline=(255, 190, 0), width=4)
    draw.text((98, y1 + 22), "2. Remove border context", font=title_font, fill=(96, 70, 0))
    draw.text((98, y1 + 64), "normalized painting input for DINOv2", font=body_font, fill=(52, 60, 66))
    canvas.paste(final_img, (600, y1 - 30))
    draw.rectangle([600, y1 - 30, 600 + final_img.width, y1 - 30 + final_img.height], outline=(255, 190, 0), width=5)
    return canvas


def main() -> None:
    OUT.mkdir(exist_ok=True)
    image = cv2.imread(str(SOURCE))
    if image is None:
        raise FileNotFoundError(SOURCE)

    crop_result = PaintingCropper().crop(image)
    overlay, metadata, corners = draw_detection_overlay(image)
    cropped = perspective_crop_from_corners(image, corners)
    final_crop = remove_fractional_border(cropped, ratio=0.08)

    overlay_path = OUT / "cup_of_coffee_preprocessing_overlay.png"
    crop_path = OUT / "cup_of_coffee_preprocessed_crop.png"
    final_crop_path = OUT / "cup_of_coffee_preprocessed_final_input.png"
    figure_path = OUT / "cup_of_coffee_preprocessing_figure.png"
    metadata_path = OUT / "cup_of_coffee_preprocessing_metadata.json"

    cv2.imwrite(str(overlay_path), overlay)
    cv2.imwrite(str(crop_path), cropped)
    cv2.imwrite(str(final_crop_path), final_crop)
    make_combined_figure(overlay, cropped, final_crop).save(figure_path)
    metadata_path.write_text(
        __import__("json").dumps(
            {
                **metadata,
                "cropper_method": crop_result.method,
                "cropper_confidence": crop_result.confidence,
                "crop_shape_hwc": list(cropped.shape),
                "final_crop_shape_hwc": list(final_crop.shape),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(overlay_path)
    print(crop_path)
    print(final_crop_path)
    print(figure_path)
    print(metadata_path)


if __name__ == "__main__":
    main()
