"""Visualization demo for the view_arc obstacle detection pipeline.

The script runs the detector on a synthetic scene, renders the
field-of-view wedge, obstacle contours, and resolved angular intervals,
then writes the output to ``examples/output/visualization_demo.png``.

Run with::

    uv run python examples/visualization_demo.py
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from view_arc import find_largest_obstacle
from view_arc.visualize import draw_complete_visualization, HAS_CV2

SceneTuple = Tuple[
    NDArray[np.uint8],
    NDArray[np.float32],
    NDArray[np.float32],
    float,
    float,
    List[NDArray[np.float32]],
]

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_PATH = OUTPUT_DIR / "visualization_demo.png"


def create_background(width: int = 640, height: int = 480) -> NDArray[np.uint8]:
    """Create a soft gradient background to make overlays stand out."""

    x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)
    xv, yv = np.meshgrid(x, y)
    base = 200 * (0.5 + 0.5 * yv)  # vertical gradient
    image = np.stack([base, base * 0.95, base * 0.9], axis=-1)
    return image.astype(np.uint8)


def build_demo_scene() -> SceneTuple:
    """Construct a synthetic scene along with the viewer configuration."""

    image = create_background()
    viewer = np.array([320.0, 360.0], dtype=np.float32)
    view_direction = np.array([0.0, -1.0], dtype=np.float32)  # looking toward image top
    field_of_view_deg = 90.0
    max_range = 260.0

    obstacles: List[NDArray[np.float32]] = [
        np.array([[260.0, 250.0], [310.0, 150.0], [360.0, 250.0]], dtype=np.float32),
        np.array([[420.0, 280.0], [500.0, 340.0], [470.0, 410.0], [410.0, 360.0]], dtype=np.float32),
        np.array([[180.0, 260.0], [240.0, 280.0], [220.0, 360.0], [160.0, 330.0]], dtype=np.float32),
    ]

    return image, viewer, view_direction, field_of_view_deg, max_range, obstacles


def run_detector(
    viewer_point: NDArray[np.float32],
    view_direction: NDArray[np.float32],
    field_of_view_deg: float,
    max_range: float,
    obstacles: List[NDArray[np.float32]],
) -> tuple[
    int | None,
    List[tuple[float, float]],
]:
    """Execute the detector and prepare data for visualization overlays."""

    result = find_largest_obstacle(
        viewer_point=viewer_point,
        view_direction=view_direction,
        field_of_view_deg=field_of_view_deg,
        max_range=max_range,
        obstacle_contours=obstacles,
        return_intervals=True,
    )

    intervals: List[tuple[float, float]] = []
    for interval in result.get_all_intervals():
        intervals.append((interval.angle_start, interval.angle_end))

    print(result.summary())
    return result.obstacle_id, intervals


def main() -> None:
    """Generate and save a visualization demo image."""

    if not HAS_CV2:
        raise SystemExit(
            "OpenCV (cv2) is required for this example. Install the optional "
                'dependency with `uv pip install -e ".[dev]"` or the `opencv-python` package.'
        )

    image, viewer, direction, fov_deg, max_range, obstacles = build_demo_scene()
    winner_id, intervals = run_detector(viewer, direction, fov_deg, max_range, obstacles)

    visualization = draw_complete_visualization(
        image,
        viewer_point=viewer,
        view_direction=direction,
        field_of_view_deg=fov_deg,
        max_range=max_range,
        obstacle_contours=obstacles,
        winner_id=winner_id,
        intervals=intervals,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    import cv2  # Imported lazily to keep dependency optional at module import time

    cv2.imwrite(str(OUTPUT_PATH), visualization)
    print(f"Visualization saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
