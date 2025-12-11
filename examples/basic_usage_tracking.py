"""Basic usage example using the process_single_sample tracking API.

This script demonstrates how to call ``process_single_sample`` with a tiny
synthetic scene and print the resulting summary. This is equivalent to
``basic_usage.py`` but uses the tracking API wrapper around ``find_largest_obstacle``.

Run it with ``uv run`` to ensure the project's virtual environment and
dependencies are used::

    uv run python examples/basic_usage_tracking.py
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from view_arc.tracking import (
    AOI,
    SingleSampleResult,
    ViewerSample,
    process_single_sample,
)

SceneTuple = Tuple[
    ViewerSample,
    float,
    float,
    List[AOI],
]


def build_sample_scene() -> SceneTuple:
    """Create a small deterministic scene for demonstration purposes.

    Returns:
        Tuple containing ``ViewerSample``, ``field_of_view`` in degrees,
        ``max_range`` in pixels/units, and a list of AOI objects.
    """

    # Create viewer sample (position + direction as unit vector)
    sample = ViewerSample(
        position=(100.0, 100.0),
        direction=(0.0, 1.0),  # facing "up"
    )
    field_of_view_deg = 60.0
    max_range = 150.0

    # Two overlapping obstacles with different footprints
    triangle = np.array(
        [[90.0, 150.0], [110.0, 150.0], [100.0, 190.0]], dtype=np.float64
    )
    rectangle = np.array(
        [[70.0, 130.0], [130.0, 130.0], [130.0, 180.0], [70.0, 180.0]],
        dtype=np.float64,
    )

    # Wrap contours as AOI objects with IDs
    aois = [
        AOI(id=0, contour=triangle),
        AOI(id=1, contour=rectangle),
    ]

    return sample, field_of_view_deg, max_range, aois


def print_result(result: SingleSampleResult, aois: List[AOI]) -> None:
    """Print useful information about the single sample result."""

    if result.winning_aoi_id is not None:
        print(f"Winner: AOI {result.winning_aoi_id}")
        print(f"  Angular Coverage: {np.rad2deg(result.angular_coverage):.2f}°")
        print(f"  Min Distance: {result.min_distance:.2f}")
    else:
        print("No AOI visible in the view arc")

    if result.all_coverage:
        print("\nAll AOI Coverage:")
        for aoi_id, coverage in sorted(result.all_coverage.items(), key=lambda x: str(x[0])):
            print(f"  AOI {aoi_id}: {np.rad2deg(coverage):.2f}°")


def main() -> None:
    """Execute the basic usage workflow using process_single_sample."""

    sample, fov_deg, max_range, aois = build_sample_scene()

    # Use process_single_sample with return_details=True to get full info
    result = process_single_sample(
        sample=sample,
        aois=aois,
        fov_deg=fov_deg,
        max_range=max_range,
        return_details=True,
    )

    # Type assertion for detailed result
    assert isinstance(result, SingleSampleResult)

    print_result(result, aois)

    # Also demonstrate simple usage (just getting the winner ID)
    print("\n--- Simple usage (winner ID only) ---")
    winner_id = process_single_sample(
        sample=sample,
        aois=aois,
        fov_deg=fov_deg,
        max_range=max_range,
        return_details=False,
    )
    print(f"Winner AOI ID: {winner_id}")


if __name__ == "__main__":
    main()
