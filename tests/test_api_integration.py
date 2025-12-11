"""
Integration tests for the obstacle detection API with real-world scenarios.

These tests simulate realistic usage patterns and validate that the algorithm
behaves correctly in practical situations.
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from view_arc.obstacle.api import find_largest_obstacle, ObstacleResult


# =============================================================================
# Helper functions for creating test fixtures
# =============================================================================


def make_triangle(center: tuple[float, float], size: float = 20.0) -> NDArray[np.float32]:
    """Create a triangle centered at given point."""
    cx, cy = center
    return np.array(
        [
            [cx, cy + size],  # top
            [cx - size, cy - size],  # bottom-left
            [cx + size, cy - size],  # bottom-right
        ],
        dtype=np.float32,
    )


def make_square(center: tuple[float, float], half_size: float = 15.0) -> NDArray[np.float32]:
    """Create a square centered at given point."""
    cx, cy = center
    return np.array(
        [
            [cx - half_size, cy - half_size],  # bottom-left
            [cx + half_size, cy - half_size],  # bottom-right
            [cx + half_size, cy + half_size],  # top-right
            [cx - half_size, cy + half_size],  # top-left
        ],
        dtype=np.float32,
    )


def make_rectangle(
    center: tuple[float, float], width: float, height: float
) -> NDArray[np.float32]:
    """Create a rectangle centered at given point."""
    cx, cy = center
    hw, hh = width / 2, height / 2
    return np.array(
        [
            [cx - hw, cy - hh],
            [cx + hw, cy - hh],
            [cx + hw, cy + hh],
            [cx - hw, cy + hh],
        ],
        dtype=np.float32,
    )


def make_polygon(center: tuple[float, float], n_sides: int, radius: float) -> NDArray[np.float32]:
    """Create a regular polygon with n sides centered at given point."""
    cx, cy = center
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    vertices = np.column_stack(
        [cx + radius * np.cos(angles), cy + radius * np.sin(angles)]
    )
    return vertices.astype(np.float32)


def normalize_direction(x: float, y: float) -> NDArray[np.float32]:
    """Normalize a direction vector to unit length."""
    vec = np.array([x, y], dtype=np.float32)
    return (vec / np.linalg.norm(vec)).astype(np.float32)


# =============================================================================
# Scenario: Person looking up (view_direction=[0, 1])
# =============================================================================


class TestScenarioPersonLookingUp:
    """
    Scenarios simulating a person looking up (positive Y direction).
    
    This is a common scenario in surveillance/tracking where the camera
    is positioned above and the person is looking upward in the image.
    """

    def test_single_obstacle_directly_ahead(self) -> None:
        """Person looking up sees a single obstacle directly ahead."""
        viewer = np.array([200.0, 200.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)  # Looking UP

        # Obstacle directly in the line of sight
        contours = [make_square((200, 280), half_size=25)]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=60.0,
            max_range=150.0,
            obstacle_contours=contours,
        )

        assert result.obstacle_id == 0
        assert result.angular_coverage > 0
        assert result.min_distance < 100  # Should be relatively close

    def test_multiple_obstacles_at_different_distances(self) -> None:
        """Person looking up sees multiple obstacles at varying distances.
        
        The near obstacle (id=0) is closer (distance ~40 vs ~120) and even
        though it's smaller (half_size=10 vs 40), its proximity gives it
        larger angular coverage. Angular coverage scales inversely with distance.
        """
        viewer = np.array([200.0, 200.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        # Near obstacle (small but close) - angular coverage ~ 2*atan(10/40) ~ 28°
        near_obstacle = make_square((200, 240), half_size=10)
        # Far obstacle (large but distant) - angular coverage ~ 2*atan(40/120) ~ 37°
        # However, near obstacle occludes part of far obstacle, reducing its visible coverage
        far_obstacle = make_square((200, 320), half_size=40)

        contours = [near_obstacle, far_obstacle]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=90.0,
            max_range=200.0,
            obstacle_contours=contours,
        )

        # The near obstacle should win because its angular coverage (~37°)
        # exceeds the far obstacle's remaining visible coverage after occlusion
        assert result.obstacle_id == 0, (
            f"Expected near obstacle (id=0) to win due to proximity advantage, "
            f"but got obstacle {result.obstacle_id}"
        )
        assert result.angular_coverage > 0
        # Near obstacle's min distance should be around 30 (240 - 200 - 10)
        assert result.min_distance < 40, (
            f"Expected min_distance < 40 for near obstacle, got {result.min_distance}"
        )

    def test_obstacles_on_either_side(self) -> None:
        """Person looking up with obstacles to the left and right."""
        viewer = np.array([200.0, 200.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        # Left obstacle
        left_obs = make_square((140, 280), half_size=20)
        # Right obstacle
        right_obs = make_square((260, 280), half_size=20)

        contours = [left_obs, right_obs]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=90.0,
            max_range=150.0,
            obstacle_contours=contours,
        )

        # Should detect one of the obstacles
        assert result.obstacle_id is not None


# =============================================================================
# Scenario: Person looking left (view_direction=[-1, 0])
# =============================================================================


class TestScenarioPersonLookingLeft:
    """
    Scenarios simulating a person looking left (negative X direction).
    """

    def test_obstacle_to_the_left(self) -> None:
        """Person looking left sees an obstacle to their left."""
        viewer = np.array([300.0, 200.0], dtype=np.float32)
        direction = np.array([-1.0, 0.0], dtype=np.float32)  # Looking LEFT

        # Obstacle to the left
        contours = [make_square((200, 200), half_size=25)]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=60.0,
            max_range=150.0,
            obstacle_contours=contours,
        )

        assert result.obstacle_id == 0
        assert result.angular_coverage > 0

    def test_obstacle_behind_not_visible(self) -> None:
        """Person looking left should not see obstacle to the right."""
        viewer = np.array([200.0, 200.0], dtype=np.float32)
        direction = np.array([-1.0, 0.0], dtype=np.float32)  # Looking LEFT

        # Obstacle to the right (behind the viewer's view)
        contours = [make_square((350, 200), half_size=25)]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=60.0,
            max_range=200.0,
            obstacle_contours=contours,
        )

        # Should not detect the obstacle
        assert result.obstacle_id is None

    def test_peripheral_vision_edge(self) -> None:
        """Obstacle at the edge of peripheral vision."""
        viewer = np.array([200.0, 200.0], dtype=np.float32)
        direction = np.array([-1.0, 0.0], dtype=np.float32)

        # Obstacle slightly up from the left direction
        # At about 30° offset from the primary direction
        contours = [make_square((120, 260), half_size=20)]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=90.0,  # Wide enough to catch it
            max_range=150.0,
            obstacle_contours=contours,
        )

        assert result.obstacle_id == 0


# =============================================================================
# Scenario: Person looking diagonal (view_direction=[-0.37, 0.92])
# =============================================================================


class TestScenarioPersonLookingDiagonal:
    """
    Scenarios simulating a person looking in a diagonal direction.
    
    This uses the example from the spec: view_direction=[-0.37, 0.92]
    which is up-left at approximately 112° from the +x axis.
    """

    def test_obstacle_in_diagonal_direction(self) -> None:
        """Person looking up-left sees obstacle in that direction."""
        viewer = np.array([200.0, 200.0], dtype=np.float32)
        direction = normalize_direction(-0.37, 0.92)

        # Obstacle in the up-left direction
        contours = [make_square((140, 300), half_size=25)]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=60.0,
            max_range=150.0,
            obstacle_contours=contours,
        )

        assert result.obstacle_id == 0
        assert result.angular_coverage > 0

    def test_obstacle_off_diagonal_axis(self) -> None:
        """Obstacles that are not aligned with the diagonal viewing direction."""
        viewer = np.array([200.0, 200.0], dtype=np.float32)
        direction = normalize_direction(-0.37, 0.92)

        # Multiple obstacles at different positions
        contours = [
            make_square((120, 280), half_size=15),  # Near diagonal
            make_square((250, 320), half_size=20),  # Off to the right
            make_square((100, 350), half_size=18),  # Further up-left
        ]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=90.0,
            max_range=200.0,
            obstacle_contours=contours,
        )

        # Should detect one of the visible obstacles
        assert result.obstacle_id is not None

    def test_diagonal_with_narrow_fov(self) -> None:
        """Diagonal viewing with a narrow field of view."""
        viewer = np.array([200.0, 200.0], dtype=np.float32)
        direction = normalize_direction(-0.37, 0.92)

        # Obstacle precisely in the viewing direction
        # Calculate position along the direction vector
        target_dist = 80.0
        target_x = 200.0 + direction[0] * target_dist
        target_y = 200.0 + direction[1] * target_dist
        contours = [make_square((target_x, target_y), half_size=20)]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=30.0,  # Narrow FOV
            max_range=150.0,
            obstacle_contours=contours,
        )

        assert result.obstacle_id == 0


# =============================================================================
# Scenario: Close vs Far obstacles (distance-based selection)
# =============================================================================


class TestScenarioCloseVsFarObstacles:
    """
    Tests for scenarios where distance plays a key role in obstacle selection.
    """

    def test_closer_obstacle_occludes_farther(self) -> None:
        """A closer obstacle should occlude a farther one behind it.
        
        The close obstacle (half_size=8 at distance ~30) has larger angular
        coverage than its absolute size would suggest. Angular coverage
        scales as ~2*atan(size/distance), so proximity is a major factor.
        The close obstacle also blocks part of the far obstacle.
        """
        viewer = np.array([100.0, 100.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        # Close small obstacle - angular coverage ~ 2*atan(8/22) ~ 40°
        close_obs = make_square((100, 130), half_size=8)
        # Far large obstacle - angular coverage ~ 2*atan(30/70) ~ 46° but reduced by occlusion
        far_obs = make_square((100, 200), half_size=30)

        contours = [close_obs, far_obs]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=90.0,
            max_range=150.0,
            obstacle_contours=contours,
        )

        # The close obstacle wins because its angular coverage exceeds
        # the far obstacle's remaining visible coverage after occlusion
        assert result.obstacle_id == 0, (
            f"Expected close obstacle (id=0) to win due to proximity advantage, "
            f"but got obstacle {result.obstacle_id}"
        )
        assert result.angular_coverage > 0
        # Close obstacle's min distance should be around 22 (130 - 100 - 8)
        assert result.min_distance < 30, (
            f"Expected min_distance < 30 for close obstacle, got {result.min_distance}"
        )

    def test_tie_breaking_closer_wins(self) -> None:
        """When angular coverage is similar, closer obstacle should win.
        
        Two identically-sized obstacles at different distances but placed
        such that they have similar angular coverage (closer one is smaller
        in absolute size but appears similar due to distance).
        """
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        # Two obstacles designed to have approximately equal angular coverage:
        # - Close obstacle at distance ~50 with half_size=12.5
        # - Far obstacle at distance ~100 with half_size=25 (twice the size at twice the distance)
        # Angular coverage ≈ 2 * atan(half_size / distance) should be similar
        close_left = make_square((-25, 50), half_size=12)
        far_right = make_square((50, 100), half_size=24)  # ~2x size at ~2x distance

        contours = [close_left, far_right]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=120.0,
            max_range=150.0,
            obstacle_contours=contours,
        )

        # With similar angular coverage, the closer obstacle should win as tie-breaker
        assert result.obstacle_id == 0, (
            f"Expected closer obstacle (id=0) to win tie-breaker, "
            f"but got obstacle {result.obstacle_id}"
        )
        # Verify the min_distance corresponds to the close obstacle (~38 = 50 - 12)
        assert result.min_distance < 50, (
            f"Expected min_distance < 50 for close obstacle, got {result.min_distance}"
        )

    def test_very_close_obstacle_large_angular_coverage(self) -> None:
        """Very close obstacle should have large angular coverage."""
        viewer = np.array([100.0, 100.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        # Very close obstacle (almost touching)
        close_obs = make_square((100, 115), half_size=10)

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=90.0,
            max_range=50.0,
            obstacle_contours=[close_obs],
        )

        assert result.obstacle_id == 0
        # Very close obstacle should have substantial angular coverage
        assert result.angular_coverage > np.deg2rad(20)


# =============================================================================
# Scenario: Large vs Narrow obstacles (angular coverage comparison)
# =============================================================================


class TestScenarioLargeVsNarrowObstacles:
    """
    Tests comparing obstacles with different shapes and angular footprints.
    """

    def test_wide_vs_narrow_at_same_distance(self) -> None:
        """Wide obstacle should have more angular coverage than narrow one."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        # Wide obstacle
        wide_obs = make_rectangle((-40, 60), width=60, height=10)
        # Narrow obstacle at same distance
        narrow_obs = make_rectangle((40, 60), width=10, height=40)

        contours = [wide_obs, narrow_obs]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=120.0,
            max_range=100.0,
            obstacle_contours=contours,
        )

        # Wide obstacle should win based on angular coverage
        assert result.obstacle_id == 0

    def test_tall_vs_wide_obstacle(self) -> None:
        """Compare tall obstacle (perpendicular) vs wide obstacle.
        
        The wide obstacle (id=1) is closer and has greater horizontal extent,
        which translates to more angular coverage. The tall obstacle's height
        doesn't contribute to angular coverage (only width matters for the
        angular span from the viewer's perspective).
        """
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        # Tall but narrow (extends perpendicular to view direction)
        # Angular coverage ≈ 2 * atan(5 / 70) ≈ 8.2°
        tall_narrow = make_rectangle((0, 70), width=10, height=50)
        # Wide but short - closer and wider
        # Angular coverage ≈ 2 * atan(25 / 40) ≈ 64°
        wide_short = make_rectangle((0, 40), width=50, height=10)

        contours = [tall_narrow, wide_short]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=90.0,
            max_range=100.0,
            obstacle_contours=contours,
        )

        # The closer wide obstacle (id=1) should win due to greater angular coverage
        assert result.obstacle_id == 1, (
            f"Expected wide obstacle (id=1) to win with greater angular coverage, "
            f"but got obstacle {result.obstacle_id}"
        )
        # Verify min_distance corresponds to the wide obstacle (~35 = 40 - 5)
        assert result.min_distance < 40, (
            f"Expected min_distance < 40 for wide obstacle, got {result.min_distance}"
        )

    def test_small_obstacle_filling_fov(self) -> None:
        """Small obstacle that fills most of a narrow FOV."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        # Small obstacle directly ahead
        small_centered = make_square((0, 30), half_size=8)

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=20.0,  # Very narrow FOV
            max_range=50.0,
            obstacle_contours=[small_centered],
        )

        assert result.obstacle_id == 0
        # Should fill most of the narrow FOV
        max_coverage = np.deg2rad(20.0)
        assert result.angular_coverage > max_coverage * 0.5


# =============================================================================
# Scenario: Obstacle at arc boundary (partial visibility)
# =============================================================================


class TestScenarioObstacleAtArcBoundary:
    """
    Tests for obstacles that are partially visible at the edge of the FOV.
    """

    def test_obstacle_partially_in_fov(self) -> None:
        """Obstacle that is only partially within the field of view."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        # Large obstacle that extends beyond the FOV edges
        wide_obs = make_rectangle((0, 50), width=200, height=20)

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=60.0,
            max_range=100.0,
            obstacle_contours=[wide_obs],
        )

        assert result.obstacle_id == 0
        # Angular coverage should be limited by the FOV
        max_fov_coverage = np.deg2rad(60.0)
        assert result.angular_coverage <= max_fov_coverage + 0.01

    def test_obstacle_at_left_edge_of_fov(self) -> None:
        """Obstacle at the left edge of the field of view."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        # 60° FOV centered at 90° spans from 60° to 120° (from +x axis)
        # Place obstacle at roughly 115° so it's near the left edge but still visible
        # At 115°: cos(115°) ≈ -0.42, sin(115°) ≈ 0.91
        # Position at distance 60: (-25, 55) approximately
        contours = [make_square((-25, 55), half_size=15)]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=60.0,
            max_range=100.0,
            obstacle_contours=contours,
        )

        # Should still be partially visible
        assert result.obstacle_id is not None

    def test_obstacle_at_right_edge_of_fov(self) -> None:
        """Obstacle at the right edge of the field of view."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        # 60° FOV centered at 90° spans from 60° to 120° (from +x axis)
        # Place obstacle at roughly 65° so it's near the right edge but still visible
        # At 65°: cos(65°) ≈ 0.42, sin(65°) ≈ 0.91
        # Position at distance 60: (25, 55) approximately
        contours = [make_square((25, 55), half_size=15)]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=60.0,
            max_range=100.0,
            obstacle_contours=contours,
        )

        # Should still be partially visible
        assert result.obstacle_id is not None

    def test_obstacle_just_outside_fov(self) -> None:
        """Obstacle that is just outside the field of view."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        # Obstacle well outside the 30° FOV
        contours = [make_square((100, 20), half_size=10)]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=30.0,
            max_range=150.0,
            obstacle_contours=contours,
        )

        assert result.obstacle_id is None


# =============================================================================
# Scenario: Max range limit (distant obstacles rejected)
# =============================================================================


class TestScenarioMaxRangeLimit:
    """
    Tests for scenarios where obstacles are rejected due to max range.
    """

    def test_obstacle_beyond_max_range(self) -> None:
        """Obstacle beyond max_range should not be detected."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        # Obstacle at distance 100, but max_range is 50
        contours = [make_square((0, 100), half_size=20)]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=90.0,
            max_range=50.0,
            obstacle_contours=contours,
        )

        assert result.obstacle_id is None

    def test_obstacle_partially_beyond_max_range(self) -> None:
        """Obstacle partially beyond max_range should be clipped."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        # Obstacle centered at 60, extending from 40 to 80
        # With max_range of 70, the far part should be clipped
        contours = [make_rectangle((0, 60), width=30, height=40)]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=90.0,
            max_range=70.0,
            obstacle_contours=contours,
        )

        assert result.obstacle_id == 0

    def test_obstacle_exactly_at_max_range(self) -> None:
        """Obstacle at exactly max_range boundary."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        # Obstacle starting exactly at max_range
        contours = [make_square((0, 50), half_size=10)]  # Closest point at 40

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=90.0,
            max_range=45.0,  # Cuts through the obstacle
            obstacle_contours=contours,
        )

        # Part of the obstacle should still be visible
        assert result.obstacle_id == 0

    def test_multiple_obstacles_some_beyond_range(self) -> None:
        """Mix of obstacles within and beyond max range."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        contours = [
            make_square((0, 40), half_size=10),  # Within range
            make_square((30, 150), half_size=20),  # Beyond range
            make_square((-30, 60), half_size=12),  # Within range
        ]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=120.0,
            max_range=80.0,
            obstacle_contours=contours,
        )

        # Should detect one of the in-range obstacles
        assert result.obstacle_id in [0, 2]


# =============================================================================
# Scenario: Complex contours (polygons with many vertices)
# =============================================================================


class TestScenarioComplexContours:
    """
    Tests for obstacles with complex polygon shapes (many vertices).
    """

    def test_octagon_obstacle(self) -> None:
        """Obstacle with octagonal (8-sided) shape."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        # Create an octagon
        octagon = make_polygon((0, 60), n_sides=8, radius=20)
        contours = [octagon]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=90.0,
            max_range=100.0,
            obstacle_contours=contours,
        )

        assert result.obstacle_id == 0
        assert result.angular_coverage > 0

    def test_hexagon_obstacle(self) -> None:
        """Obstacle with hexagonal (6-sided) shape."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([1.0, 0.0], dtype=np.float32)  # Looking RIGHT

        hexagon = make_polygon((70, 0), n_sides=6, radius=25)
        contours = [hexagon]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=60.0,
            max_range=100.0,
            obstacle_contours=contours,
        )

        assert result.obstacle_id == 0

    def test_high_polygon_count(self) -> None:
        """Obstacle approximating a circle (many vertices)."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        # 32-sided polygon approximating a circle
        circle_approx = make_polygon((0, 50), n_sides=32, radius=15)
        contours = [circle_approx]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=90.0,
            max_range=80.0,
            obstacle_contours=contours,
        )

        assert result.obstacle_id == 0
        assert result.angular_coverage > 0

    def test_irregular_polygon(self) -> None:
        """Obstacle with irregular (non-regular) polygon shape."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        # Irregular polygon
        irregular = np.array(
            [
                [10, 40],
                [30, 50],
                [25, 70],
                [5, 80],
                [-15, 60],
                [-10, 45],
            ],
            dtype=np.float32,
        )
        contours = [irregular]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=90.0,
            max_range=100.0,
            obstacle_contours=contours,
        )

        assert result.obstacle_id == 0

    def test_multiple_complex_polygons(self) -> None:
        """Multiple obstacles with varying polygon complexities."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        contours = [
            make_polygon((-40, 60), n_sides=5, radius=15),  # Pentagon
            make_polygon((0, 80), n_sides=12, radius=20),  # Dodecagon
            make_polygon((40, 55), n_sides=7, radius=18),  # Heptagon
        ]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=120.0,
            max_range=120.0,
            obstacle_contours=contours,
        )

        assert result.obstacle_id is not None
        assert result.angular_coverage > 0


# =============================================================================
# Scenario: Edge cases and boundary conditions
# =============================================================================


class TestScenarioEdgeCases:
    """
    Tests for edge cases and boundary conditions.
    """

    def test_obstacle_centered_on_viewer(self) -> None:
        """Edge case: obstacle overlapping with viewer position."""
        viewer = np.array([50.0, 50.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        # Obstacle centered on the viewer (viewer is inside)
        contours = [make_square((50, 50), half_size=30)]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=90.0,
            max_range=100.0,
            obstacle_contours=contours,
        )

        # Behavior may vary - obstacle may be detected or not
        # The important thing is it doesn't crash
        assert isinstance(result, ObstacleResult)

    def test_very_small_obstacle(self) -> None:
        """Edge case: very small obstacle."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        # Very small obstacle
        contours = [make_square((0, 50), half_size=1)]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=90.0,
            max_range=100.0,
            obstacle_contours=contours,
        )

        assert result.obstacle_id == 0
        # Coverage should be very small but positive
        assert result.angular_coverage > 0

    def test_very_large_obstacle(self) -> None:
        """Edge case: very large obstacle filling most of the view."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        # Very large obstacle
        contours = [make_rectangle((0, 50), width=500, height=100)]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=90.0,
            max_range=100.0,
            obstacle_contours=contours,
        )

        assert result.obstacle_id == 0
        # Should fill most of the FOV
        max_coverage = np.deg2rad(90.0)
        assert result.angular_coverage > max_coverage * 0.8

    def test_minimal_triangle_obstacle(self) -> None:
        """Edge case: minimal valid polygon (3 vertices)."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        # Minimal triangle
        triangle = np.array(
            [
                [0, 40],
                [10, 60],
                [-10, 55],
            ],
            dtype=np.float32,
        )
        contours = [triangle]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=90.0,
            max_range=100.0,
            obstacle_contours=contours,
        )

        assert result.obstacle_id == 0

    def test_many_obstacles_performance(self) -> None:
        """Test with many obstacles to check reasonable performance."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        # Create 20 obstacles in a grid pattern
        contours = []
        for i in range(5):
            for j in range(4):
                x = -100 + i * 50
                y = 30 + j * 40
                contours.append(make_square((x, y), half_size=10))

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=180.0,
            max_range=200.0,
            obstacle_contours=contours,
        )

        # Should complete without error and find an obstacle
        assert result.obstacle_id is not None


# =============================================================================
# Scenario: Return intervals verification
# =============================================================================


class TestScenarioReturnIntervals:
    """
    Tests verifying the return_intervals functionality in realistic scenarios.
    """

    def test_single_obstacle_interval_bounds(self) -> None:
        """Verify interval bounds for a single obstacle."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        contours = [make_square((0, 50), half_size=20)]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=90.0,
            max_range=100.0,
            obstacle_contours=contours,
            return_intervals=True,
        )

        assert result.obstacle_id == 0
        assert result.intervals is not None
        assert len(result.intervals) >= 1

        # Verify interval format
        for start, end in result.intervals:
            assert isinstance(start, float)
            assert isinstance(end, float)

    def test_multiple_intervals_for_complex_scene(self) -> None:
        """Complex scene may result in multiple intervals for winner."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        # Large obstacle with a smaller one in front creating shadow
        large_back = make_rectangle((0, 80), width=100, height=20)
        small_front = make_square((0, 40), half_size=10)

        contours = [large_back, small_front]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=90.0,
            max_range=120.0,
            obstacle_contours=contours,
            return_intervals=True,
        )

        assert result.intervals is not None
        # The back obstacle (if it wins) may have multiple disconnected intervals
        # due to the front obstacle's shadow

    def test_intervals_sum_to_coverage(self) -> None:
        """Verify that intervals sum approximately to total coverage."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        contours = [make_square((0, 50), half_size=25)]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=90.0,
            max_range=100.0,
            obstacle_contours=contours,
            return_intervals=True,
        )

        if result.intervals:
            interval_sum = sum(end - start for start, end in result.intervals)
            # Should be approximately equal to angular_coverage
            assert abs(interval_sum - result.angular_coverage) < 0.01
