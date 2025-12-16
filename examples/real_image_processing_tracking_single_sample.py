"""Real image processing example using the process_single_sample tracking API.

This script demonstrates how the tracking API can work with manually annotated
polygons. This is equivalent to ``real_image_processing.py`` but uses the
``process_single_sample`` wrapper around ``find_largest_obstacle``.

The demo loads the background image from ``images/background.jpeg`` plus
polygons stored in ``images/polygon_vertices.json`` and then visualises
the winning obstacle and angular coverage overlay.

Run with::

    uv run python examples/real_image_processing_tracking.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple, cast

import numpy as np
from numpy.typing import NDArray
from skimage import color, io, util

from view_arc import find_largest_obstacle
from view_arc.tracking import (
    AOI,
    SingleSampleResult,
    ViewerSample,
    process_single_sample,
)
from view_arc.obstacle.visualize import draw_complete_visualization, HAS_CV2

EXAMPLES_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXAMPLES_DIR.parent
IMAGE_PATH = PROJECT_ROOT / "images" / "background.jpeg"
POLYGON_PATH = PROJECT_ROOT / "images" / "polygon_vertices.json"
OUTPUT_DIR = EXAMPLES_DIR / "output"
OUTPUT_PATH = OUTPUT_DIR / "real_image_demo_tracking.png"


def summarise_result(
    result: SingleSampleResult,
    aois: List[AOI],
) -> None:
    """Print summary information for the detection result."""

    if result.winning_aoi_id is not None:
        print(f"Winner: AOI {result.winning_aoi_id}")
        print(f"  Coverage: {np.rad2deg(result.angular_coverage):.2f}°")
        print(f"  Min Distance: {result.min_distance:.2f}")
    else:
        print("No AOI visible in the view arc")

    print()
    if result.all_coverage:
        print("Coverage per AOI (by annotated id):")
        for aoi_id, coverage in sorted(result.all_coverage.items(), key=lambda x: str(x[0])):
            coverage_deg = np.rad2deg(coverage)
            print(f"  {aoi_id}: {coverage_deg:.2f}° coverage")


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


def load_aois(
    polygon_file: Path,
) -> Tuple[List[AOI], Dict[int, int]]:
    """Load manually annotated polygons from JSON and return AOI objects.

    Returns:
        A tuple of (aois, index_to_id) where aois is a list of AOI objects
        and index_to_id maps list indices to the original annotation IDs.
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

    aois: List[AOI] = []
    index_to_id: Dict[int, int] = {}

    for entry in data:
        if not isinstance(entry, dict):
            continue
        vertices: Any = entry.get("vertices")
        if not isinstance(vertices, list) or len(vertices) < 3:
            continue
        try:
            polygon = np.array(vertices, dtype=np.float64)
        except ValueError as exc:
            raise SystemExit(
                f"Invalid vertex data for entry {entry.get('id')}: {exc}"
            ) from exc
        if polygon.ndim != 2 or polygon.shape[1] != 2:
            raise SystemExit(
                f"Polygon {entry.get('id')} does not contain 2D vertices:"
                f" shape={polygon.shape}"
            )

        # Get the annotation ID
        try:
            annotation_id = int(entry["id"])
        except (KeyError, TypeError, ValueError):
            annotation_id = len(aois)

        # Create AOI with the annotation ID
        aois.append(AOI(id=annotation_id, contour=polygon))
        index_to_id[len(aois) - 1] = annotation_id

    if not aois:
        raise SystemExit(
            f"No valid polygons found in {polygon_file}. Ensure the JSON contains "
            "objects with 'id' and 'vertices' keys."
        )

    return aois, index_to_id


def main() -> None:
    """Run obstacle detection on a real image using the tracking API."""

    image = load_scene_image()
    height, width, _ = image.shape

    # Create ViewerSample
    sample = ViewerSample(
        position=(350.0, 200.0),
        direction=(0.378, -0.925),
    )
    field_of_view_deg = 45.0
    max_range = 60.0

    aois, index_to_id = load_aois(POLYGON_PATH)

    # Use process_single_sample with return_details=True
    result = process_single_sample(
        sample=sample,
        aois=aois,
        field_of_view_deg=field_of_view_deg,
        max_range=max_range,
        return_details=True,
    )

    assert isinstance(result, SingleSampleResult)
    summarise_result(result, aois)

    if not HAS_CV2:
        print("OpenCV not installed; skipping visualization output.")
        return

    # For visualization, we need to use the original find_largest_obstacle
    # to get the interval details (which process_single_sample doesn't expose)
    viewer_point = np.array([350, 200.0], dtype=np.float32)
    view_direction = np.array([0.378, -0.925], dtype=np.float32)
    obstacle_contours = [aoi.contour.astype(np.float32) for aoi in aois]

    viz_result = find_largest_obstacle(
        viewer_point=viewer_point,
        view_direction=view_direction,
        field_of_view_deg=field_of_view_deg,
        max_range=max_range,
        obstacle_contours=obstacle_contours,
        return_intervals=True,
        return_all_coverage=True,
    )

    intervals = [
        (interval.angle_start, interval.angle_end)
        for interval in viz_result.get_all_intervals()
    ]

    # Build labels from the annotation IDs
    obstacle_labels = [str(aoi.id) for aoi in aois]

    visualization = draw_complete_visualization(
        image.astype(np.uint8),
        viewer_point=viewer_point,
        view_direction=view_direction,
        field_of_view_deg=field_of_view_deg,
        max_range=max_range,
        obstacle_contours=obstacle_contours,
        winner_id=viz_result.obstacle_id,
        intervals=intervals,
        obstacle_labels=obstacle_labels,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    import cv2  # Imported lazily to keep dependency optional at module import time

    cv2.imwrite(str(OUTPUT_PATH), visualization)
    print(f"Saved visualization to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
