from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class PaintingCandidate:
    bounding_rect: tuple[int, int, int, int]
    frame_contour: np.ndarray
    painting_corners: np.ndarray
    painting_points: np.ndarray
    crop: np.ndarray
    mask_crop: np.ndarray


def mean_shift_segmentation(
    image: np.ndarray,
    spatial_radius: int = 7,
    color_radius: int = 13,
    max_pyramid_level: int = 1,
) -> np.ndarray:
    return cv2.pyrMeanShiftFiltering(
        image,
        sp=spatial_radius,
        sr=color_radius,
        maxLevel=max_pyramid_level,
    )


def get_mask_of_largest_segment(
    image: np.ndarray,
    color_difference: tuple[int, int, int] = (2, 2, 2),
) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("image must be a non-empty BGR image")

    segmented = image.copy()
    mask = np.zeros((image.shape[0] + 2, image.shape[1] + 2), dtype=np.uint8)
    largest_area = 0
    wall_color = None
    rng = np.random.default_rng(42)

    lo_diff = color_difference
    up_diff = color_difference

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if mask[y + 1, x + 1] != 0:
                continue

            new_val = tuple(int(v) for v in rng.integers(0, 256, size=3))
            _, _, _, rect = cv2.floodFill(
                segmented,
                mask,
                (x, y),
                new_val,
                loDiff=lo_diff,
                upDiff=up_diff,
                flags=4,
            )
            area = rect[2] * rect[3]
            if area > largest_area:
                largest_area = area
                wall_color = new_val

    if wall_color is None:
        raise RuntimeError("failed to determine largest segment")

    lower = np.array(wall_color, dtype=np.uint8)
    upper = np.array(wall_color, dtype=np.uint8)
    return cv2.inRange(segmented, lower, upper)


def dilate_image(image: np.ndarray, kernel_size: int, shape: int = cv2.MORPH_RECT) -> np.ndarray:
    kernel = cv2.getStructuringElement(shape, (kernel_size, kernel_size))
    return cv2.dilate(image, kernel)


def erode_image(image: np.ndarray, kernel_size: int, shape: int = cv2.MORPH_RECT) -> np.ndarray:
    kernel = cv2.getStructuringElement(shape, (kernel_size, kernel_size))
    return cv2.erode(image, kernel)


def invert_image(image: np.ndarray) -> np.ndarray:
    return cv2.bitwise_not(image)


def median_filter(image: np.ndarray, blur_size: int) -> np.ndarray:
    return cv2.medianBlur(image, blur_size)


def canny_edge_detection(image: np.ndarray) -> np.ndarray:
    otsu_threshold, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    lower_threshold = 0.5 * otsu_threshold
    return cv2.Canny(image, lower_threshold, otsu_threshold)


def could_be_painting(
    image: np.ndarray,
    bounder: tuple[int, int, int, int],
    contour: np.ndarray,
    min_width: int,
    min_height: int,
    min_area_percentage: float,
) -> bool:
    _, _, width, height = bounder
    rect_area = width * height
    image_area = image.shape[0] * image.shape[1]

    if not (rect_area < image_area and rect_area > min_width * min_height):
        return False

    return cv2.contourArea(contour) > rect_area * min_area_percentage


def get_possible_painting_contours(
    image: np.ndarray,
    contours: list[np.ndarray],
    min_width: int = 150,
    min_height: int = 150,
    min_area_percentage: float = 0.6,
) -> list[np.ndarray]:
    painting_contours: list[np.ndarray] = []
    for contour in contours:
        bounder = cv2.boundingRect(contour)
        if could_be_painting(
            image,
            bounder,
            contour,
            min_width,
            min_height,
            min_area_percentage,
        ):
            painting_contours.append(contour)
    return painting_contours


def order_corners(corners: np.ndarray) -> np.ndarray:
    corners = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    corners = corners[np.argsort(corners[:, 1])]
    top = corners[:2][np.argsort(corners[:2, 0])]
    bottom = corners[2:][np.argsort(corners[2:, 0])]
    return np.vstack([top, bottom])


def extend_lines_across_image(image: np.ndarray, lines: np.ndarray, color: int | tuple[int, int, int]) -> np.ndarray:
    result = image.copy()
    if lines is None:
        return result

    height, width = image.shape[:2]
    length = max(height, width)

    for line in lines.reshape(-1, 4):
        x1, y1, x2, y2 = map(int, line)
        angle = np.degrees(np.arctan2(y1 - y2, x1 - x2))

        p1 = np.array([x1, y1], dtype=np.float32)
        direction = np.array(
            [
                np.cos(np.radians(angle)),
                np.sin(np.radians(angle)),
            ],
            dtype=np.float32,
        )
        p2 = np.round(p1 + length * direction).astype(int)
        p3 = np.round(p1 - length * direction).astype(int)
        cv2.line(result, tuple(p3), tuple(p2), color, 10, cv2.LINE_8)

    return result


def _largest_contour(contours: list[np.ndarray]) -> np.ndarray | None:
    if not contours:
        return None
    return max(contours, key=lambda contour: contour.shape[0])


def _extract_painting_shape(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eroded = erode_image(mask, 60)
    blurred = median_filter(eroded, 31)
    edges = canny_edge_detection(blurred)

    painting_ratio = int(max(image.shape[:2]) * 0.1)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=0,
        minLineLength=int(painting_ratio * 1.5),
        maxLineGap=painting_ratio,
    )

    sudoku = np.zeros(mask.shape, dtype=np.uint8)
    sudoku = extend_lines_across_image(sudoku, lines, 255)

    contours, _ = cv2.findContours(
        invert_image(sudoku),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_NONE,
    )
    if len(contours) < 9:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 1, 2), dtype=np.int32)

    largest = _largest_contour(contours)
    painting_contour_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(painting_contour_mask, [largest], -1, 255, cv2.FILLED)

    corners = cv2.goodFeaturesToTrack(
        painting_contour_mask,
        maxCorners=4,
        qualityLevel=0.001,
        minDistance=20,
    )
    if corners is None or len(corners) != 4:
        corners_array = np.empty((0, 2), dtype=np.float32)
    else:
        corners_array = order_corners(corners.reshape(-1, 2))

    return corners_array, largest


def preprocess_gallery_image(
    image: np.ndarray,
    spatial_radius: int = 7,
    color_radius: int = 13,
    max_pyramid_level: int = 1,
    wall_kernel_size: int = 18,
    min_width: int = 150,
    min_height: int = 150,
    min_area_percentage: float = 0.6,
) -> dict[str, object]:
    segmented = mean_shift_segmentation(
        image,
        spatial_radius=spatial_radius,
        color_radius=color_radius,
        max_pyramid_level=max_pyramid_level,
    )
    wall_mask = get_mask_of_largest_segment(segmented, color_difference=(2, 2, 2))
    dilated_wall_mask = dilate_image(wall_mask, wall_kernel_size)
    inverted_wall_mask = invert_image(dilated_wall_mask)

    contours, _ = cv2.findContours(
        inverted_wall_mask.copy(),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_NONE,
    )
    frame_contours = get_possible_painting_contours(
        inverted_wall_mask,
        contours,
        min_width=min_width,
        min_height=min_height,
        min_area_percentage=min_area_percentage,
    )

    candidates: list[PaintingCandidate] = []
    for contour in frame_contours:
        x, y, w, h = cv2.boundingRect(contour)
        crop = image[y : y + h, x : x + w]
        mask_crop = inverted_wall_mask[y : y + h, x : x + w]
        corners, painting_points = _extract_painting_shape(crop, mask_crop)

        # if len(corners) != 4:
        #     continue

        translated_points = painting_points + np.array([[[x, y]]], dtype=np.int32)
        candidates.append(
            PaintingCandidate(
                bounding_rect=(x, y, w, h),
                frame_contour=contour,
                painting_corners=corners + np.array([x, y], dtype=np.float32) if len(corners) == 4 else np.empty((0, 2), dtype=np.float32),
                painting_points=translated_points,
                crop=crop,
                mask_crop=mask_crop,
            )
        )

    return {
        "segmented": segmented,
        "wall_mask": wall_mask,
        "dilated_wall_mask": dilated_wall_mask,
        "inverted_wall_mask": inverted_wall_mask,
        "frame_contours": frame_contours,
        "candidates": candidates,
    }


def draw_preprocessing_result(image: np.ndarray, candidates: list[PaintingCandidate]) -> np.ndarray:
    output = image.copy()
    for index, candidate in enumerate(candidates, start=1):
        x, y, w, h = candidate.bounding_rect
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)

        corners = candidate.painting_corners.astype(int)
        if len(corners) == 4:
            cv2.polylines(output, [corners.reshape(-1, 1, 2)], True, (0, 255, 0), 2)

        cv2.putText(
            output,
            f"candidate {index}",
            (x, max(25, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return output


def load_image(image_path: str | Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"could not read image: {image_path}")
    return image


def save_debug_images(result: dict[str, object], output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path / "01_segmented.jpg"), result["segmented"])
    cv2.imwrite(str(output_path / "02_wall_mask.jpg"), result["wall_mask"])
    cv2.imwrite(str(output_path / "03_dilated_wall_mask.jpg"), result["dilated_wall_mask"])
    cv2.imwrite(str(output_path / "04_inverted_wall_mask.jpg"), result["inverted_wall_mask"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess a gallery image to find painting candidates.")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument(
        "--output-dir",
        default="preprocessing_output",
        help="Directory for debug images",
    )
    args = parser.parse_args()

    image = load_image(args.image_path)
    result = preprocess_gallery_image(image)
    annotated = draw_preprocessing_result(image, result["candidates"])
    save_debug_images(result, args.output_dir)
    cv2.imwrite(str(Path(args.output_dir) / "05_annotated_candidates.jpg"), annotated)
    print(f"Found {len(result['candidates'])} painting candidate(s).")
