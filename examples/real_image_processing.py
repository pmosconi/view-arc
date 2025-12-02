"""Real image processing example for the view_arc pipeline.

This script shows how the obstacle detector can be paired with simple
image processing logic to extract contours from a real photograph. The
demo loads ``images/background.jpeg`` that ships with the repository,
then visualises the winning obstacle and angular coverage overlay.

Run with::

    uv run python examples/real_image_processing.py
"""

from __future__ import annotations

from pathlib import Path
from typing import List, cast

import numpy as np
from numpy.typing import NDArray
from skimage import color, filters, measure, morphology, io, util

from view_arc import find_largest_obstacle
from view_arc.api import ObstacleResult
from view_arc.visualize import draw_complete_visualization, HAS_CV2

EXAMPLES_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXAMPLES_DIR.parent
IMAGE_PATH = PROJECT_ROOT / "images" / "background.jpeg"
OUTPUT_DIR = EXAMPLES_DIR / "output"
OUTPUT_PATH = OUTPUT_DIR / "real_image_demo.png"


def _polygon_area(vertices: NDArray[np.float32]) -> float:
    """Compute the absolute area of a closed polygon using the shoelace formula."""

    if vertices.shape[0] < 3:
        return 0.0
    x = vertices[:, 0]
    y = vertices[:, 1]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def extract_obstacle_contours(
    image: NDArray[np.uint8],
    max_obstacles: int = 6,
) -> List[NDArray[np.float32]]:
    """Segment bright regions from an image and return polygonal contours."""

    grayscale = color.rgb2gray(image)
    blurred = filters.gaussian(grayscale, sigma=1.2)
    threshold = filters.threshold_otsu(blurred)
    mask = blurred > threshold
    mask = morphology.remove_small_objects(mask, min_size=500)
    mask = morphology.binary_closing(mask, footprint=np.ones((5, 5), dtype=bool))

    contours = measure.find_contours(mask.astype(np.float32), level=0.5)

    processed: List[NDArray[np.float32]] = []
    for contour in contours:
        if contour.shape[0] < 10:
            continue

        # Down-sample long contours to keep the polygon manageable
        step = max(1, contour.shape[0] // 150)
        contour = contour[::step]

        # Convert from (row, col) to (x, y) with y growing downward (image coords)
        polygon = np.column_stack((contour[:, 1], contour[:, 0])).astype(np.float32)
        processed.append(polygon)

    # Keep largest contours by projected area
    processed.sort(key=_polygon_area, reverse=True)
    return processed[:max_obstacles]


def summarise_result(result: ObstacleResult) -> None:
    """Print summary information for the detection result."""

    print(result.summary())
    print()
    if result.all_coverage:
        print("Coverage per obstacle (degrees):")
        for obstacle_id, coverage in sorted(result.all_coverage.items()):
            coverage_deg = np.rad2deg(coverage)
            min_distance = result.all_distances.get(obstacle_id, float("inf")) if result.all_distances else float("inf")
            print(
                f"  {obstacle_id}: {coverage_deg:.2f}° coverage, min_distance={min_distance:.2f}"
            )


def load_scene_image() -> NDArray[np.uint8]:
    """Load the demo background from ``images/background.jpeg`` as uint8 RGB."""

    if not IMAGE_PATH.exists():
        raise SystemExit(
            f"Sample image not found at {IMAGE_PATH}. Add the file or adjust IMAGE_PATH."
        )

    raw_image = io.imread(IMAGE_PATH)
    image = np.asarray(raw_image)
    if image.ndim == 2:
        image = color.gray2rgb(image)
    if image.dtype != np.uint8:
        image = util.img_as_ubyte(image)
    return cast(NDArray[np.uint8], image.astype(np.uint8, copy=False))


def main() -> None:
    """Run obstacle detection on a real image and optionally visualise the result."""

    image = load_scene_image()
    height, width, _ = image.shape

    viewer_point = np.array([width / 2.0, 40.0], dtype=np.float32)
    view_direction = np.array([0.378, 0.925], dtype=np.float32)  # looking toward the bottom-right of the image
    field_of_view_deg = 45.0
    max_range = height * 0.4

    obstacles = extract_obstacle_contours(image, max_obstacles=6)
    if not obstacles:
        raise SystemExit("No contours detected; adjust segmentation parameters in extract_obstacle_contours().")

    result = find_largest_obstacle(
        viewer_point=viewer_point,
        view_direction=view_direction,
        field_of_view_deg=field_of_view_deg,
        max_range=max_range,
        obstacle_contours=obstacles,
        return_intervals=True,
        return_all_coverage=True,
    )

    summarise_result(result)

    if not HAS_CV2:
        print("OpenCV not installed; skipping visualization output.")
        return

    intervals = [(interval.angle_start, interval.angle_end) for interval in result.get_all_intervals()]

    visualization = draw_complete_visualization(
        image.astype(np.uint8),
        viewer_point=viewer_point,
        view_direction=view_direction,
        field_of_view_deg=field_of_view_deg,
        max_range=max_range,
        obstacle_contours=obstacles,
        winner_id=result.obstacle_id,
        intervals=intervals,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    import cv2  # Imported lazily to keep dependency optional at module import time

    cv2.imwrite(str(OUTPUT_PATH), visualization)
    print(f"Saved visualization to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
