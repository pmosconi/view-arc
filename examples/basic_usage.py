"""Basic usage example for the view_arc obstacle detection API.

This script demonstrates how to call ``find_largest_obstacle`` with a tiny
synthetic scene and print the resulting summary. Run it with ``uv run`` to
ensure the project's virtual environment and dependencies are used::

    uv run python examples/basic_usage.py
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from view_arc import find_largest_obstacle
from view_arc.obstacle.api import ObstacleResult

SceneTuple = Tuple[
    NDArray[np.float32],
    NDArray[np.float32],
    float,
    float,
    List[NDArray[np.float32]],
]


def build_sample_scene() -> SceneTuple:
    """Create a small deterministic scene for demonstration purposes.

    Returns:
        Tuple containing ``viewer_point``, ``view_direction``, ``field_of_view``
        in degrees, ``max_range`` in pixels/units, and a list of obstacle
        contours expressed as ``(N, 2)`` arrays of ``float32`` values.
    """

    viewer_point = np.array([100.0, 100.0], dtype=np.float32)
    view_direction = np.array([0.0, 1.0], dtype=np.float32)  # facing "up"
    field_of_view_deg = 60.0
    max_range = 150.0

    # Two overlapping obstacles with different footprints
    triangle = np.array(
        [[90.0, 150.0], [110.0, 150.0], [100.0, 190.0]], dtype=np.float32
    )
    rectangle = np.array(
        [[70.0, 130.0], [130.0, 130.0], [130.0, 180.0], [70.0, 180.0]],
        dtype=np.float32,
    )

    return viewer_point, view_direction, field_of_view_deg, max_range, [triangle, rectangle]


def print_result(result: ObstacleResult) -> None:
    """Print useful information about the obstacle detection result."""

    print(result.summary())

    if result.interval_details:
        print("\nWinner intervals (degrees):")
        for interval in result.get_winner_intervals():
            print(
                f"  {interval.angle_start_deg:.1f}° → {interval.angle_end_deg:.1f}° "
                f"span={interval.angular_span_deg:.1f}° min_dist={interval.min_distance:.2f}"
            )
    else:
        print("No interval breakdown requested. Pass return_intervals=True to obtain it.")



def main() -> None:
    """Execute the basic usage workflow."""

    viewer_point, view_direction, fov_deg, max_range, obstacles = build_sample_scene()

    result = find_largest_obstacle(
        viewer_point=viewer_point,
        view_direction=view_direction,
        field_of_view_deg=fov_deg,
        max_range=max_range,
        obstacle_contours=obstacles,
        return_intervals=True,
        return_all_coverage=True,
    )

    print_result(result)


if __name__ == "__main__":
    main()
