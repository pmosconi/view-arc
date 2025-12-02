"""Real image processing example for the view_arc pipeline.

This script demonstrates how the obstacle detector can work with
manually annotated polygons. The demo loads the background image from
``images/background.jpeg`` plus polygons stored in
``images/polygon_vertices.json`` and then visualises the winning
obstacle and angular coverage overlay.

Run with::

    uv run python examples/real_image_processing.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple, cast

import numpy as np
from numpy.typing import NDArray
from skimage import color, io, util

from view_arc import find_largest_obstacle
from view_arc.api import ObstacleResult
from view_arc.visualize import draw_complete_visualization, HAS_CV2

EXAMPLES_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXAMPLES_DIR.parent
IMAGE_PATH = PROJECT_ROOT / "images" / "background.jpeg"
POLYGON_PATH = PROJECT_ROOT / "images" / "polygon_vertices.json"
OUTPUT_DIR = EXAMPLES_DIR / "output"
OUTPUT_PATH = OUTPUT_DIR / "real_image_demo.png"


def summarise_result(
    result: ObstacleResult,
    index_to_id: Mapping[int, int] | None = None,
) -> None:
    """Print summary information for the detection result."""

    def _map_id(idx: int) -> int:
        if index_to_id is not None:
            return index_to_id.get(idx, idx)
        return idx

    if result.obstacle_id is not None:
        winner_id = _map_id(result.obstacle_id)
        print(f"Winner: Obstacle {winner_id}")
        print(f"  Coverage: {result.angular_coverage_deg:.2f}°")
        print(f"  Min Distance: {result.min_distance:.2f}")
    else:
        print("No obstacle visible in the view arc")

    print()
    if result.all_coverage:
        print("Coverage per obstacle (by annotated id):")
        for obstacle_index, coverage in sorted(result.all_coverage.items()):
            display_id = _map_id(obstacle_index)
            coverage_deg = np.rad2deg(coverage)
            min_distance = (
                result.all_distances.get(obstacle_index, float("inf"))
                if result.all_distances
                else float("inf")
            )
            print(
                f"  {display_id}: {coverage_deg:.2f}° coverage, min_distance={min_distance:.2f}"
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


def load_obstacle_contours(
    polygon_file: Path,
) -> Tuple[List[NDArray[np.float32]], Dict[int, int]]:
    """Load manually annotated polygons from JSON and return contours with ID mapping.

    Returns:
        A tuple of (obstacles, index_to_id) where obstacles is a list of polygon
        arrays and index_to_id maps list indices to the original annotation IDs.
    """

    if not polygon_file.exists():
        raise SystemExit(
            f"Polygon annotation file not found at {polygon_file}. "
            "Create it or adjust POLYGON_PATH."
        )

    with polygon_file.open("r", encoding="utf-8") as handle:
        try:
            data = json.load(handle)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Failed to parse {polygon_file}: {exc}") from exc

    obstacles: List[NDArray[np.float32]] = []
    index_to_id: Dict[int, int] = {}

    for entry in data:
        if not isinstance(entry, dict):
            continue
        vertices: Any = entry.get("vertices")
        if not isinstance(vertices, list) or len(vertices) < 3:
            continue
        try:
            polygon = np.array(vertices, dtype=np.float32)
        except ValueError as exc:
            raise SystemExit(
                f"Invalid vertex data for entry {entry.get('id')}: {exc}"
            ) from exc
        if polygon.ndim != 2 or polygon.shape[1] != 2:
            raise SystemExit(
                f"Polygon {entry.get('id')} does not contain 2D vertices:"
                f" shape={polygon.shape}"
            )
        obstacles.append(polygon)
        try:
            index_to_id[len(obstacles) - 1] = int(entry["id"])
        except (KeyError, TypeError, ValueError):
            index_to_id[len(obstacles) - 1] = len(obstacles) - 1

    if not obstacles:
        raise SystemExit(
            f"No valid polygons found in {polygon_file}. Ensure the JSON contains "
            "objects with 'id' and 'vertices' keys."
        )

    return obstacles, index_to_id


def main() -> None:
    """Run obstacle detection on a real image and optionally visualise the result."""

    image = load_scene_image()
    height, width, _ = image.shape

    viewer_point = np.array([350, 200.0], dtype=np.float32)
    view_direction = np.array([0.378, -0.925], dtype=np.float32)
    field_of_view_deg = 45.0
    max_range = 60.0

    obstacles, index_to_id = load_obstacle_contours(POLYGON_PATH)

    result = find_largest_obstacle(
        viewer_point=viewer_point,
        view_direction=view_direction,
        field_of_view_deg=field_of_view_deg,
        max_range=max_range,
        obstacle_contours=obstacles,
        return_intervals=True,
        return_all_coverage=True,
    )

    summarise_result(result, index_to_id=index_to_id)

    if not HAS_CV2:
        print("OpenCV not installed; skipping visualization output.")
        return

    intervals = [
        (interval.angle_start, interval.angle_end)
        for interval in result.get_all_intervals()
    ]

    # Build labels from the annotation IDs
    obstacle_labels = [str(index_to_id.get(i, i)) for i in range(len(obstacles))]

    visualization = draw_complete_visualization(
        image.astype(np.uint8),
        viewer_point=viewer_point,
        view_direction=view_direction,
        field_of_view_deg=field_of_view_deg,
        max_range=max_range,
        obstacle_contours=obstacles,
        winner_id=result.obstacle_id,
        intervals=intervals,
        obstacle_labels=obstacle_labels,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    import cv2  # Imported lazily to keep dependency optional at module import time

    cv2.imwrite(str(OUTPUT_PATH), visualization)
    print(f"Saved visualization to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
