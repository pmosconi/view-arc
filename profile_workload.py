#!/usr/bin/env python3
"""
Profile script for view_arc module to identify performance bottlenecks.
"""

import cProfile
import pstats
import io
import numpy as np
from numpy.typing import NDArray
import time
from typing import List
from view_arc.api import find_largest_obstacle


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


def profile_function(func, description: str) -> None:
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


if __name__ == "__main__":
    print("View Arc Performance Profiling")
    print("=" * 60)
    
    # Profile typical workload (5 obstacles, 5 vertices each)
    profile_function(
        lambda: run_typical_workload(100),
        "Typical workload (5 obstacles x 5 vertices, 100 iterations)"
    )
    
    # Profile many obstacles workload
    profile_function(
        lambda: run_many_obstacles_workload(20),
        "Many obstacles (50 obstacles x 8 vertices, 20 iterations)"
    )
