#!/usr/bin/env python3
"""
Profile script for view_arc module to identify performance bottlenecks.

Includes:
- Single-frame obstacle detection profiling
- Tracking (batch processing) profiling with golden baseline comparison
- CSV output for trend tracking
"""

import argparse
import cProfile
import csv
import io
import os
import pstats
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

import numpy as np
from numpy.typing import NDArray

from view_arc.obstacle.api import find_largest_obstacle
from view_arc.tracking import AOI, ViewerSample, compute_attention_seconds


def generate_random_polygon(
    center: NDArray[np.float32],
    radius: float,
    n_vertices: int = 5
) -> NDArray[np.float32]:
    """Generate a random polygon roughly centered at center."""
    angles = np.sort(np.random.uniform(0, 2 * np.pi, n_vertices))
    radii = np.random.uniform(0.5 * radius, 1.5 * radius, n_vertices)
    x = center[0] + radii * np.cos(angles)
    y = center[1] + radii * np.sin(angles)
    return np.column_stack([x, y]).astype(np.float32)


def generate_typical_workload(
    n_obstacles: int = 5,
    vertices_per_obstacle: int = 5
) -> tuple[NDArray[np.float32], NDArray[np.float32], float, float, List[NDArray[np.float32]]]:
    """
    Generate a typical workload for profiling.
    Simulates a viewer at center looking at obstacles in front.
    """
    viewer = np.array([500.0, 500.0], dtype=np.float32)
    direction = np.array([0.0, 1.0], dtype=np.float32)  # Looking up
    fov = 60.0
    max_range = 300.0
    
    # Generate obstacles in the viewing direction
    obstacles = []
    for i in range(n_obstacles):
        # Place obstacles in front of viewer
        angle = np.random.uniform(-0.4, 0.4)  # Within FOV roughly
        dist = np.random.uniform(50, 250)
        center = (viewer + dist * np.array([np.sin(angle), np.cos(angle)])).astype(np.float32)
        polygon = generate_random_polygon(center, 30.0, vertices_per_obstacle)
        obstacles.append(polygon)
    
    return viewer, direction, fov, max_range, obstacles


def run_typical_workload(n_iterations: int = 100) -> None:
    """Run typical workload multiple times for profiling."""
    np.random.seed(42)  # For reproducibility
    
    for _ in range(n_iterations):
        viewer, direction, fov, max_range, obstacles = generate_typical_workload(
            n_obstacles=5, vertices_per_obstacle=5
        )
        find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=fov,
            max_range=max_range,
            obstacle_contours=obstacles
        )


def run_many_obstacles_workload(n_iterations: int = 20) -> None:
    """Run workload with many obstacles for profiling."""
    np.random.seed(42)
    
    for _ in range(n_iterations):
        viewer, direction, fov, max_range, obstacles = generate_typical_workload(
            n_obstacles=50, vertices_per_obstacle=8
        )
        find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=fov,
            max_range=max_range,
            obstacle_contours=obstacles
        )


def profile_function(func: Callable[[], None], description: str) -> None:
    """Profile a function and print statistics."""
    print(f"\n{'=' * 60}")
    print(f"Profiling: {description}")
    print('=' * 60)
    
    # Time the execution
    start = time.perf_counter()
    
    profiler = cProfile.Profile()
    profiler.enable()
    func()
    profiler.disable()
    
    elapsed = time.perf_counter() - start
    
    # Get stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)
    print(s.getvalue())
    
    print(f"\nTotal time: {elapsed:.3f}s")


# =============================================================================
# Tracking Workload Functions (Step 6.1)
# =============================================================================


@dataclass
class BaselineResult:
    """Result from a baseline tracking run."""

    scenario: str
    n_samples: int
    n_aois: int
    runtime_seconds: float
    samples_per_second: float
    total_hits: int
    hit_rate: float

    def __repr__(self) -> str:
        return (
            f"BaselineResult(scenario={self.scenario!r}, "
            f"n_samples={self.n_samples}, n_aois={self.n_aois}, "
            f"runtime={self.runtime_seconds:.3f}s, "
            f"throughput={self.samples_per_second:.1f} samples/s, "
            f"hits={self.total_hits}, hit_rate={self.hit_rate:.1%})"
        )


def generate_tracking_workload(
    n_samples: int = 300,
    n_aois: int = 20,
    seed: int = 42,
) -> tuple[list[ViewerSample], list[AOI]]:
    """Generate a realistic tracking workload with viewer moving through a space.

    Args:
        n_samples: Number of viewer samples (e.g., 300 = 5 minute session at 1 Hz)
        n_aois: Number of areas of interest (shelves)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (samples, aois)
    """
    rng = np.random.default_rng(seed)

    # Create AOIs distributed in a grid-like pattern (like store shelves)
    aois = []
    grid_size = int(np.ceil(np.sqrt(n_aois)))
    for i in range(n_aois):
        row = i // grid_size
        col = i % grid_size

        # Each AOI is a rectangular shelf
        x_center = 200 + col * 150
        y_center = 200 + row * 150
        width = rng.uniform(60, 100)
        height = rng.uniform(40, 60)

        contour = np.array(
            [
                [x_center - width / 2, y_center - height / 2],
                [x_center + width / 2, y_center - height / 2],
                [x_center + width / 2, y_center + height / 2],
                [x_center - width / 2, y_center + height / 2],
            ],
            dtype=np.float32,
        )

        aois.append(AOI(id=f"shelf_{i:02d}", contour=contour))

    # Create viewer samples walking through the space
    samples = []
    for i in range(n_samples):
        # Viewer walks along a path
        t = i / n_samples
        x = 100 + t * 800 + rng.normal(0, 10)
        y = 400 + np.sin(t * 4 * np.pi) * 200 + rng.normal(0, 10)

        # Direction is generally forward with some variation
        base_angle = t * 0.5 + rng.normal(0, 0.2)
        dx = np.cos(base_angle)
        dy = np.sin(base_angle)

        # Normalize direction
        mag = np.sqrt(dx * dx + dy * dy)
        direction = (dx / mag, dy / mag)

        samples.append(ViewerSample(position=(x, y), direction=direction))

    return samples, aois


def run_tracking_baseline(
    n_samples: int = 300, n_aois: int = 20, seed: int = 42
) -> BaselineResult:
    """Run a tracking baseline workload and return metrics.

    Args:
        n_samples: Number of samples (default 300 = 5 min at 1 Hz)
        n_aois: Number of AOIs (default 20)
        seed: Random seed for reproducibility

    Returns:
        BaselineResult with performance and accuracy metrics
    """
    samples, aois = generate_tracking_workload(n_samples, n_aois, seed)

    start = time.perf_counter()
    result = compute_attention_seconds(samples, aois, enable_profiling=True)
    elapsed = time.perf_counter() - start

    throughput = n_samples / elapsed if elapsed > 0 else 0
    total_hits = result.samples_with_hits
    hit_rate = total_hits / n_samples if n_samples > 0 else 0

    return BaselineResult(
        scenario=f"tracking_{n_samples}samples_{n_aois}aois",
        n_samples=n_samples,
        n_aois=n_aois,
        runtime_seconds=elapsed,
        samples_per_second=throughput,
        total_hits=total_hits,
        hit_rate=hit_rate,
    )


def compare_with_golden_baseline(
    result: BaselineResult, golden_runtime: float, golden_hits: int
) -> tuple[bool, str]:
    """Compare a result against golden baseline values.

    Args:
        result: The result to check
        golden_runtime: Expected runtime (seconds)
        golden_hits: Expected hit count

    Returns:
        Tuple of (passed, message)
    """
    # Allow 20% runtime variance
    runtime_threshold = golden_runtime * 1.2
    runtime_passed = result.runtime_seconds <= runtime_threshold

    # Accuracy must match exactly
    accuracy_passed = result.total_hits == golden_hits

    messages = []
    if not runtime_passed:
        messages.append(
            f"Runtime regression: {result.runtime_seconds:.3f}s > {runtime_threshold:.3f}s threshold"
        )
    if not accuracy_passed:
        messages.append(
            f"Accuracy mismatch: {result.total_hits} hits != {golden_hits} expected"
        )

    if runtime_passed and accuracy_passed:
        return True, "✓ Passed baseline checks"
    else:
        return False, "✗ Failed: " + "; ".join(messages)


def save_profile_run_to_csv(result: BaselineResult, csv_path: Path) -> None:
    """Append a profile run result to CSV for trend tracking.

    Args:
        result: The profile result to save
        csv_path: Path to CSV file (will be created if missing)
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    is_new = not csv_path.exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "scenario",
                "n_samples",
                "n_aois",
                "runtime_seconds",
                "samples_per_second",
                "total_hits",
                "hit_rate",
            ],
        )

        if is_new:
            writer.writeheader()

        writer.writerow(
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "scenario": result.scenario,
                "n_samples": result.n_samples,
                "n_aois": result.n_aois,
                "runtime_seconds": f"{result.runtime_seconds:.3f}",
                "samples_per_second": f"{result.samples_per_second:.1f}",
                "total_hits": result.total_hits,
                "hit_rate": f"{result.hit_rate:.3f}",
            }
        )


def run_all_tracking_baselines(save_csv: bool = False) -> list[BaselineResult]:
    """Run all tracking baseline scenarios and optionally save to CSV.

    Args:
        save_csv: If True, append results to examples/output/profile_runs.csv

    Returns:
        List of BaselineResult objects
    """
    print("\n" + "=" * 60)
    print("Tracking Performance Baselines")
    print("=" * 60)

    scenarios = [
        # (n_samples, n_aois, seed, golden_runtime, golden_hits)
        (100, 10, 42, 0.2, 53),  # Small: 100 samples, 10 AOIs
        (300, 20, 42, 1.2, 275),  # Medium: 300 samples (5 min), 20 AOIs
        (600, 50, 42, 4.0, 600),  # Large: 600 samples (10 min), 50 AOIs
    ]

    results = []
    csv_path = Path("examples/output/profile_runs.csv")

    for n_samples, n_aois, seed, golden_runtime, golden_hits in scenarios:
        print(f"\nScenario: {n_samples} samples × {n_aois} AOIs")
        result = run_tracking_baseline(n_samples, n_aois, seed)
        print(f"  Runtime: {result.runtime_seconds:.3f}s")
        print(f"  Throughput: {result.samples_per_second:.1f} samples/s")
        print(f"  Hits: {result.total_hits} ({result.hit_rate:.1%})")

        # Compare with golden baseline
        passed, message = compare_with_golden_baseline(
            result, golden_runtime, golden_hits
        )
        print(f"  {message}")

        results.append(result)

        if save_csv:
            save_profile_run_to_csv(result, csv_path)

    if save_csv:
        print(f"\n✓ Results saved to {csv_path}")

    return results



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Profile view_arc performance"
    )
    parser.add_argument(
        "--scenario",
        choices=["typical", "many_obstacles", "tracking_baseline", "all"],
        default="all",
        help="Which scenario to run",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Save tracking results to examples/output/profile_runs.csv",
    )

    args = parser.parse_args()

    print("View Arc Performance Profiling")
    print("=" * 60)

    if args.scenario in ["typical", "all"]:
        # Profile typical workload (5 obstacles, 5 vertices each)
        profile_function(
            lambda: run_typical_workload(100),
            "Typical workload (5 obstacles x 5 vertices, 100 iterations)"
        )

    if args.scenario in ["many_obstacles", "all"]:
        # Profile many obstacles workload
        profile_function(
            lambda: run_many_obstacles_workload(20),
            "Many obstacles (50 obstacles x 8 vertices, 20 iterations)"
        )

    if args.scenario in ["tracking_baseline", "all"]:
        # Run tracking baseline scenarios
        run_all_tracking_baselines(save_csv=args.save_csv)

