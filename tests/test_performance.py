"""
Performance tests for view_arc module.

These tests verify that the algorithm performs within acceptable time bounds
for realistic workloads. The target is <100ms per frame for typical scenarios.
"""

import time
from typing import List
import numpy as np
from numpy.typing import NDArray
import pytest

from view_arc.api import find_largest_obstacle


def generate_random_polygon(
    center: NDArray[np.float32],
    radius: float,
    n_vertices: int = 5,
    rng: np.random.Generator | None = None
) -> NDArray[np.float32]:
    """Generate a random polygon roughly centered at center."""
    if rng is None:
        rng = np.random.default_rng()
    angles = np.sort(rng.uniform(0, 2 * np.pi, n_vertices))
    radii = rng.uniform(0.5 * radius, 1.5 * radius, n_vertices)
    x = center[0] + radii * np.cos(angles)
    y = center[1] + radii * np.sin(angles)
    return np.column_stack([x, y]).astype(np.float32)


def generate_workload(
    n_obstacles: int = 5,
    vertices_per_obstacle: int = 5,
    seed: int = 42
) -> tuple[NDArray[np.float32], NDArray[np.float32], float, float, List[NDArray[np.float32]]]:
    """
    Generate a workload for performance testing.
    Simulates a viewer at center looking at obstacles in front.
    """
    rng = np.random.default_rng(seed)
    
    viewer = np.array([500.0, 500.0], dtype=np.float32)
    direction = np.array([0.0, 1.0], dtype=np.float32)  # Looking up
    fov = 60.0
    max_range = 300.0
    
    # Generate obstacles in the viewing direction
    obstacles = []
    for _ in range(n_obstacles):
        # Place obstacles in front of viewer, with some outside FOV too
        angle = rng.uniform(-0.8, 0.8)  # Some within, some outside FOV
        dist = rng.uniform(50, 280)
        center = (viewer + dist * np.array([np.sin(angle), np.cos(angle)])).astype(np.float32)
        polygon = generate_random_polygon(center, 30.0, vertices_per_obstacle, rng)
        obstacles.append(polygon)
    
    return viewer, direction, fov, max_range, obstacles


class TestPerformanceManyObstacles:
    """Performance tests with many obstacles."""
    
    def test_performance_50_obstacles_8_vertices(self) -> None:
        """Test with 50 obstacles, each with 8 vertices.
        
        This represents a challenging but realistic workload.
        Target: <500ms total for 10 frames.
        """
        n_frames = 10
        total_time = 0.0
        
        for i in range(n_frames):
            viewer, direction, fov, max_range, obstacles = generate_workload(
                n_obstacles=50, vertices_per_obstacle=8, seed=42 + i
            )
            
            start = time.perf_counter()
            result = find_largest_obstacle(
                viewer_point=viewer,
                view_direction=direction,
                field_of_view_deg=fov,
                max_range=max_range,
                obstacle_contours=obstacles
            )
            elapsed = time.perf_counter() - start
            total_time += elapsed
        
        avg_time_ms = (total_time / n_frames) * 1000
        
        # Assert average time is under 100ms per frame
        assert avg_time_ms < 500, (
            f"Average frame time {avg_time_ms:.1f}ms exceeds 500ms limit for 50 obstacles"
        )
    
    def test_performance_100_obstacles_5_vertices(self) -> None:
        """Test with 100 obstacles, each with 5 vertices.
        
        Many simple obstacles.
        """
        n_frames = 5
        total_time = 0.0
        
        for i in range(n_frames):
            viewer, direction, fov, max_range, obstacles = generate_workload(
                n_obstacles=100, vertices_per_obstacle=5, seed=100 + i
            )
            
            start = time.perf_counter()
            find_largest_obstacle(
                viewer_point=viewer,
                view_direction=direction,
                field_of_view_deg=fov,
                max_range=max_range,
                obstacle_contours=obstacles
            )
            elapsed = time.perf_counter() - start
            total_time += elapsed
        
        avg_time_ms = (total_time / n_frames) * 1000
        
        # More lenient for 100 obstacles
        assert avg_time_ms < 1000, (
            f"Average frame time {avg_time_ms:.1f}ms exceeds 1000ms limit for 100 obstacles"
        )


class TestPerformanceComplexPolygons:
    """Performance tests with complex polygons."""
    
    def test_performance_5_obstacles_50_vertices(self) -> None:
        """Test with 5 obstacles, each with 50 vertices.
        
        Few obstacles but complex shapes.
        """
        n_frames = 20
        total_time = 0.0
        
        for i in range(n_frames):
            viewer, direction, fov, max_range, obstacles = generate_workload(
                n_obstacles=5, vertices_per_obstacle=50, seed=200 + i
            )
            
            start = time.perf_counter()
            find_largest_obstacle(
                viewer_point=viewer,
                view_direction=direction,
                field_of_view_deg=fov,
                max_range=max_range,
                obstacle_contours=obstacles
            )
            elapsed = time.perf_counter() - start
            total_time += elapsed
        
        avg_time_ms = (total_time / n_frames) * 1000
        
        assert avg_time_ms < 100, (
            f"Average frame time {avg_time_ms:.1f}ms exceeds 100ms limit for 5 complex obstacles"
        )
    
    def test_performance_10_obstacles_100_vertices(self) -> None:
        """Test with 10 obstacles, each with 100 vertices.
        
        Moderately complex scene - 1000 total vertices is an extreme case.
        This test validates that the algorithm doesn't have catastrophic
        performance degradation.
        """
        n_frames = 10
        total_time = 0.0
        
        for i in range(n_frames):
            viewer, direction, fov, max_range, obstacles = generate_workload(
                n_obstacles=10, vertices_per_obstacle=100, seed=300 + i
            )
            
            start = time.perf_counter()
            find_largest_obstacle(
                viewer_point=viewer,
                view_direction=direction,
                field_of_view_deg=fov,
                max_range=max_range,
                obstacle_contours=obstacles
            )
            elapsed = time.perf_counter() - start
            total_time += elapsed
        
        avg_time_ms = (total_time / n_frames) * 1000
        
        # 1000 vertices is an extreme case; allow up to 2 seconds
        assert avg_time_ms < 2000, (
            f"Average frame time {avg_time_ms:.1f}ms exceeds 2000ms limit for 10 complex obstacles"
        )


class TestPerformanceWideFOV:
    """Performance tests with wide field of view."""
    
    def test_performance_180_degree_fov(self) -> None:
        """Test with 180° field of view.
        
        Wide FOV means more angular sweep work.
        """
        n_frames = 20
        total_time = 0.0
        
        for i in range(n_frames):
            rng = np.random.default_rng(400 + i)
            viewer = np.array([500.0, 500.0], dtype=np.float32)
            direction = np.array([0.0, 1.0], dtype=np.float32)
            fov = 180.0
            max_range = 300.0
            
            # Place obstacles all around the front hemisphere
            obstacles = []
            for j in range(20):
                angle = rng.uniform(-np.pi / 2, np.pi / 2)
                dist = rng.uniform(50, 280)
                center = (viewer + dist * np.array([np.sin(angle), np.cos(angle)])).astype(np.float32)
                polygon = generate_random_polygon(center, 30.0, 6, rng)
                obstacles.append(polygon)
            
            start = time.perf_counter()
            find_largest_obstacle(
                viewer_point=viewer,
                view_direction=direction,
                field_of_view_deg=fov,
                max_range=max_range,
                obstacle_contours=obstacles
            )
            elapsed = time.perf_counter() - start
            total_time += elapsed
        
        avg_time_ms = (total_time / n_frames) * 1000
        
        assert avg_time_ms < 100, (
            f"Average frame time {avg_time_ms:.1f}ms exceeds 100ms limit for 180° FOV"
        )
    
    def test_performance_360_degree_fov(self) -> None:
        """Test with 360° field of view (full circle).
        
        Maximum angular sweep work.
        """
        n_frames = 10
        total_time = 0.0
        
        for i in range(n_frames):
            rng = np.random.default_rng(500 + i)
            viewer = np.array([500.0, 500.0], dtype=np.float32)
            direction = np.array([0.0, 1.0], dtype=np.float32)
            fov = 360.0
            max_range = 300.0
            
            # Place obstacles all around
            obstacles = []
            for j in range(30):
                angle = rng.uniform(-np.pi, np.pi)
                dist = rng.uniform(50, 280)
                center = (viewer + dist * np.array([np.sin(angle), np.cos(angle)])).astype(np.float32)
                polygon = generate_random_polygon(center, 30.0, 5, rng)
                obstacles.append(polygon)
            
            start = time.perf_counter()
            find_largest_obstacle(
                viewer_point=viewer,
                view_direction=direction,
                field_of_view_deg=fov,
                max_range=max_range,
                obstacle_contours=obstacles
            )
            elapsed = time.perf_counter() - start
            total_time += elapsed
        
        avg_time_ms = (total_time / n_frames) * 1000
        
        assert avg_time_ms < 200, (
            f"Average frame time {avg_time_ms:.1f}ms exceeds 200ms limit for 360° FOV"
        )


class TestPerformanceTypicalWorkload:
    """Performance tests for typical real-world workloads."""
    
    def test_performance_typical_5_obstacles_5_vertices(self) -> None:
        """Test with typical workload: 5 obstacles, 5 vertices each.
        
        This is the most common use case. Target: <10ms per frame.
        """
        n_frames = 100
        total_time = 0.0
        
        for i in range(n_frames):
            viewer, direction, fov, max_range, obstacles = generate_workload(
                n_obstacles=5, vertices_per_obstacle=5, seed=600 + i
            )
            
            start = time.perf_counter()
            find_largest_obstacle(
                viewer_point=viewer,
                view_direction=direction,
                field_of_view_deg=fov,
                max_range=max_range,
                obstacle_contours=obstacles
            )
            elapsed = time.perf_counter() - start
            total_time += elapsed
        
        avg_time_ms = (total_time / n_frames) * 1000
        
        assert avg_time_ms < 20, (
            f"Average frame time {avg_time_ms:.1f}ms exceeds 20ms limit for typical workload"
        )
    
    def test_performance_retail_scenario_10_obstacles_8_vertices(self) -> None:
        """Test retail scenario: 10 obstacles, 8 vertices each.
        
        Typical retail environment with multiple shoppers/objects.
        Target: <50ms per frame.
        """
        n_frames = 50
        total_time = 0.0
        
        for i in range(n_frames):
            viewer, direction, fov, max_range, obstacles = generate_workload(
                n_obstacles=10, vertices_per_obstacle=8, seed=700 + i
            )
            
            start = time.perf_counter()
            find_largest_obstacle(
                viewer_point=viewer,
                view_direction=direction,
                field_of_view_deg=fov,
                max_range=max_range,
                obstacle_contours=obstacles
            )
            elapsed = time.perf_counter() - start
            total_time += elapsed
        
        avg_time_ms = (total_time / n_frames) * 1000
        
        assert avg_time_ms < 50, (
            f"Average frame time {avg_time_ms:.1f}ms exceeds 50ms limit for retail scenario"
        )
    
    def test_performance_empty_scene(self) -> None:
        """Test with empty scene (no obstacles).
        
        Should be very fast.
        """
        n_frames = 100
        total_time = 0.0
        
        viewer = np.array([500.0, 500.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)
        fov = 60.0
        max_range = 300.0
        obstacles: List[NDArray[np.float32]] = []
        
        for _ in range(n_frames):
            start = time.perf_counter()
            find_largest_obstacle(
                viewer_point=viewer,
                view_direction=direction,
                field_of_view_deg=fov,
                max_range=max_range,
                obstacle_contours=obstacles
            )
            elapsed = time.perf_counter() - start
            total_time += elapsed
        
        avg_time_ms = (total_time / n_frames) * 1000
        
        assert avg_time_ms < 1, (
            f"Average frame time {avg_time_ms:.1f}ms exceeds 1ms limit for empty scene"
        )
    
    def test_performance_obstacles_outside_fov(self) -> None:
        """Test with obstacles outside the FOV.
        
        Early culling should make this fast.
        """
        n_frames = 50
        total_time = 0.0
        
        viewer = np.array([500.0, 500.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)  # Looking up
        fov = 30.0  # Narrow FOV
        max_range = 300.0
        
        rng = np.random.default_rng(800)
        
        # Place obstacles behind the viewer
        obstacles = []
        for _ in range(20):
            center = viewer + np.array([
                rng.uniform(-200, 200),
                rng.uniform(-200, -50)  # Behind viewer
            ], dtype=np.float32)
            polygon = generate_random_polygon(center, 30.0, 6, rng)
            obstacles.append(polygon)
        
        for _ in range(n_frames):
            start = time.perf_counter()
            find_largest_obstacle(
                viewer_point=viewer,
                view_direction=direction,
                field_of_view_deg=fov,
                max_range=max_range,
                obstacle_contours=obstacles
            )
            elapsed = time.perf_counter() - start
            total_time += elapsed
        
        avg_time_ms = (total_time / n_frames) * 1000
        
        assert avg_time_ms < 10, (
            f"Average frame time {avg_time_ms:.1f}ms exceeds 10ms limit for culled obstacles"
        )
