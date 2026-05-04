#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_metrics"
PAINTING = ROOT / "data/datasets/armenian/images/still-life-1953.jpg!PinterestSmall.jpg"


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


def textured_wall(width: int, height: int) -> Image.Image:
    rng = np.random.default_rng(7)
    base = np.zeros((height, width, 3), dtype=np.uint8)
    base[:, :] = np.array([212, 205, 192], dtype=np.uint8)
    noise = rng.normal(0, 7, base.shape).astype(np.int16)
    wall = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    for y in range(0, height, 26):
        wall[y : y + 1, :, :] = np.clip(wall[y : y + 1, :, :] - 9, 0, 255)
    return Image.fromarray(wall).filter(ImageFilter.GaussianBlur(radius=0.7))


def paste_perspective(
    canvas: Image.Image,
    source: Image.Image,
    dst_quad: np.ndarray,
) -> None:
    src = np.array([[0, 0], [source.width - 1, 0], [source.width - 1, source.height - 1], [0, source.height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst_quad.astype(np.float32))
    warped = cv2.warpPerspective(np.asarray(source), matrix, canvas.size, flags=cv2.INTER_CUBIC)
    mask = cv2.warpPerspective(np.full((source.height, source.width), 255, dtype=np.uint8), matrix, canvas.size)
    canvas.paste(Image.fromarray(warped), (0, 0), Image.fromarray(mask))


def draw_polyline(draw: ImageDraw.ImageDraw, points: np.ndarray, fill: tuple[int, int, int], width: int) -> None:
    pts = [tuple(map(int, point)) for point in points]
    draw.line(pts + [pts[0]], fill=fill, width=width, joint="curve")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    width, height = 1500, 940
    canvas = textured_wall(width, height)

    painting = Image.open(PAINTING).convert("RGB")
    painting.thumbnail((560, 470), Image.Resampling.LANCZOS)
    inner_w, inner_h = painting.size

    mat = Image.new("RGB", (inner_w + 96, inner_h + 96), (236, 231, 220))
    mat.paste(painting, (48, 48))
    frame = Image.new("RGB", (mat.width + 90, mat.height + 90), (92, 67, 43))
    frame_draw = ImageDraw.Draw(frame)
    for i, color in enumerate([(72, 49, 31), (124, 90, 54), (158, 119, 75), (61, 42, 28)]):
        inset = i * 12
        frame_draw.rectangle([inset, inset, frame.width - inset - 1, frame.height - inset - 1], outline=color, width=12)
    frame.paste(mat, (45, 45))

    frame_quad = np.array([[315, 126], [1018, 86], [1102, 691], [244, 724]], dtype=np.float32)
    painting_quad = np.array([[420, 225], [922, 201], [975, 594], [373, 622]], dtype=np.float32)
    crop_quad = np.array([[411, 216], [931, 192], [986, 604], [364, 632]], dtype=np.float32)

    shadow_quad = frame_quad + np.array([[28, 34], [28, 34], [28, 34], [28, 34]], dtype=np.float32)
    shadow_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow_layer)
    shadow_draw.polygon([tuple(point) for point in shadow_quad], fill=(0, 0, 0, 70))
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=18))
    canvas = Image.alpha_composite(canvas.convert("RGBA"), shadow_layer).convert("RGB")

    paste_perspective(canvas, frame, frame_quad)

    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw_polyline(draw, frame_quad, (255, 155, 38, 255), 5)
    draw_polyline(draw, crop_quad, (0, 184, 255, 255), 7)
    draw.polygon([tuple(point) for point in crop_quad], fill=(0, 184, 255, 24))

    for point in crop_quad:
        x, y = map(int, point)
        draw.ellipse([x - 10, y - 10, x + 10, y + 10], fill=(0, 184, 255, 255), outline=(255, 255, 255, 255), width=3)

    title_font = font(42, bold=True)
    label_font = font(25, bold=True)
    small_font = font(21)
    draw.rounded_rectangle([54, 48, 716, 145], radius=18, fill=(26, 32, 38, 210))
    draw.text((82, 67), "Preprocessing: painting detection", font=title_font, fill=(255, 255, 255, 255))
    draw.text((82, 112), "wall + frame are rejected before recognition", font=small_font, fill=(219, 232, 240, 255))

    draw.rounded_rectangle([965, 142, 1418, 246], radius=16, fill=(255, 255, 255, 230), outline=(0, 184, 255, 255), width=4)
    draw.text((990, 162), "Detected painting region", font=label_font, fill=(11, 76, 108, 255))
    draw.text((990, 199), "quadrilateral used for crop/normalization", font=small_font, fill=(42, 54, 63, 255))
    draw.line([(963, 246), tuple(map(int, crop_quad[1]))], fill=(0, 184, 255, 255), width=4)

    draw.rounded_rectangle([74, 736, 466, 829], radius=16, fill=(255, 255, 255, 230), outline=(255, 155, 38, 255), width=4)
    draw.text((98, 754), "Frame and wall context", font=label_font, fill=(118, 67, 0, 255))
    draw.text((98, 790), "kept out of the identity embedding", font=small_font, fill=(52, 52, 52, 255))
    draw.line([(466, 759), tuple(map(int, frame_quad[3]))], fill=(255, 155, 38, 255), width=4)

    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay)

    crop = np.asarray(canvas.convert("RGB"))
    source_points = crop_quad.astype(np.float32)
    crop_w, crop_h = 420, 300
    target_points = np.array([[0, 0], [crop_w - 1, 0], [crop_w - 1, crop_h - 1], [0, crop_h - 1]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(source_points, target_points)
    normalized = cv2.warpPerspective(crop, transform, (crop_w, crop_h))
    crop_img = Image.fromarray(normalized)
    crop_img = crop_img.resize((336, 240), Image.Resampling.LANCZOS)

    final = Image.new("RGB", (width, height + 320), (246, 246, 244))
    final.paste(canvas.convert("RGB"), (0, 0))
    draw = ImageDraw.Draw(final)
    draw.rounded_rectangle([575, height + 28, 925, height + 190], radius=20, fill=(229, 238, 242), outline=(0, 184, 255), width=4)
    draw.text((608, height + 50), "Perspective crop", font=label_font, fill=(11, 76, 108))
    draw.text((608, height + 88), "detected polygon -> normalized input", font=small_font, fill=(42, 54, 63))
    draw.line([(740, height + 140), (740, height + 185)], fill=(0, 184, 255), width=5)
    draw.polygon([(740, height + 198), (724, height + 176), (756, height + 176)], fill=(0, 184, 255))
    crop_x, crop_y = 994, height + 20
    final.paste(crop_img, (crop_x, crop_y))
    draw.rectangle([crop_x, crop_y, crop_x + crop_img.width, crop_y + crop_img.height], outline=(0, 184, 255), width=5)
    draw.text((crop_x + 32, crop_y + crop_img.height + 18), "normalized crop used by DINOv2", font=small_font, fill=(42, 54, 63))

    output = OUT / "preprocessing_detection_example.png"
    final.save(output)
    print(output)


if __name__ == "__main__":
    main()
