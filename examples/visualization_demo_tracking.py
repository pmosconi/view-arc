"""Visualization demo using the process_single_sample tracking API.

This script demonstrates how to use the tracking API for obstacle detection
and then renders the field-of-view wedge, obstacle contours, and resolved
angular intervals. This is equivalent to ``visualization_demo.py`` but uses
``process_single_sample`` to identify the winning AOI.

The script writes the output to ``examples/output/visualization_demo_tracking.png``.

Run with::

    uv run python examples/visualization_demo_tracking.py
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from view_arc import find_largest_obstacle
from view_arc.tracking import (
    AOI,
    SingleSampleResult,
    ViewerSample,
    process_single_sample,
)
from view_arc.visualize import draw_complete_visualization, HAS_CV2

SceneTuple = Tuple[
    NDArray[np.uint8],
    ViewerSample,
    float,
    float,
    List[AOI],
]

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_PATH = OUTPUT_DIR / "visualization_demo_tracking.png"


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

    # Create ViewerSample
    sample = ViewerSample(
        position=(320.0, 360.0),
        direction=(0.0, -1.0),  # looking toward image top
    )
    field_of_view_deg = 90.0
    max_range = 260.0

    # Create AOIs from obstacle contours
    aois: List[AOI] = [
        AOI(
            id=0,
            contour=np.array(
                [[260.0, 250.0], [310.0, 150.0], [360.0, 250.0]], dtype=np.float64
            ),
        ),
        AOI(
            id=1,
            contour=np.array(
                [[420.0, 280.0], [500.0, 340.0], [470.0, 410.0], [410.0, 360.0]],
                dtype=np.float64,
            ),
        ),
        AOI(
            id=2,
            contour=np.array(
                [[180.0, 260.0], [240.0, 280.0], [220.0, 360.0], [160.0, 330.0]],
                dtype=np.float64,
            ),
        ),
    ]

    return image, sample, field_of_view_deg, max_range, aois


def run_detector(
    sample: ViewerSample,
    field_of_view_deg: float,
    max_range: float,
    aois: List[AOI],
) -> tuple[
    int | str | None,
    SingleSampleResult,
]:
    """Execute the detector using process_single_sample."""

    result = process_single_sample(
        sample=sample,
        aois=aois,
        fov_deg=field_of_view_deg,
        max_range=max_range,
        return_details=True,
    )

    assert isinstance(result, SingleSampleResult)

    # Print summary
    if result.winning_aoi_id is not None:
        print(f"Winner: AOI {result.winning_aoi_id}")
        print(f"  Coverage: {np.rad2deg(result.angular_coverage):.2f}°")
        print(f"  Min Distance: {result.min_distance:.2f}")
    else:
        print("No AOI visible in the view arc")

    if result.all_coverage:
        print("\nAll AOIs:")
        for aoi_id, coverage in sorted(result.all_coverage.items(), key=lambda x: str(x[0])):
            print(f"  [{aoi_id}] Coverage: {np.rad2deg(coverage):.2f}°")

    return result.winning_aoi_id, result


def main() -> None:
    """Generate and save a visualization demo image."""

    if not HAS_CV2:
        raise SystemExit(
            "OpenCV (cv2) is required for this example. Install the optional "
            'dependency with `uv pip install -e ".[dev]"` or the `opencv-python` package.'
        )

    image, sample, fov_deg, max_range, aois = build_demo_scene()
    winner_id, tracking_result = run_detector(sample, fov_deg, max_range, aois)

    # For visualization, we need the interval breakdown from find_largest_obstacle
    # since process_single_sample doesn't expose interval details
    viewer_point = np.array(sample.position, dtype=np.float32)
    view_direction = np.array(sample.direction, dtype=np.float32)
    obstacle_contours = [aoi.contour.astype(np.float32) for aoi in aois]

    viz_result = find_largest_obstacle(
        viewer_point=viewer_point,
        view_direction=view_direction,
        field_of_view_deg=fov_deg,
        max_range=max_range,
        obstacle_contours=obstacle_contours,
        return_intervals=True,
    )

    intervals: List[tuple[float, float]] = []
    for interval in viz_result.get_all_intervals():
        intervals.append((interval.angle_start, interval.angle_end))

    # Map the winner_id back to obstacle index for visualization
    winner_index: int | None = None
    if winner_id is not None:
        for idx, aoi in enumerate(aois):
            if aoi.id == winner_id:
                winner_index = idx
                break

    visualization = draw_complete_visualization(
        image,
        viewer_point=viewer_point,
        view_direction=view_direction,
        field_of_view_deg=fov_deg,
        max_range=max_range,
        obstacle_contours=obstacle_contours,
        winner_id=winner_index,
        intervals=intervals,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    import cv2  # Imported lazily to keep dependency optional at module import time

    cv2.imwrite(str(OUTPUT_PATH), visualization)
    print(f"\nVisualization saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
