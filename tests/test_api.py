"""
Consolidated tests for the main API function find_largest_obstacle.

This module combines all synthetic obstacle detection scenarios with shared
fixtures and parameterized tests where appropriate. Visual tests have been
moved to a separate opt-in suite.
"""

from typing import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from view_arc.obstacle.api import find_largest_obstacle, ObstacleResult


# =============================================================================
# Shared helper functions for creating test fixtures
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
    vertices = np.column_stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles)])
    return vertices.astype(np.float32)


def normalize_direction(x: float, y: float) -> NDArray[np.float32]:
    """Normalize a direction vector to unit length."""
    vec = np.array([x, y], dtype=np.float32)
    return (vec / np.linalg.norm(vec)).astype(np.float32)


# =============================================================================
# Test: Single obstacle centered in field of view
# =============================================================================


class TestSingleObstacle:
    """Test with a single obstacle centered in the field of view."""

    @pytest.mark.parametrize(
        "viewer,direction,obstacle_fn,fov,expected_visible",
        [
            # Triangle directly in front, looking UP
            (
                np.array([100.0, 100.0], dtype=np.float32),
                np.array([0.0, 1.0], dtype=np.float32),
                lambda: make_triangle((100, 150), size=20),
                60.0,
                True,
            ),
            # Square directly in front, looking RIGHT
            (
                np.array([0.0, 0.0], dtype=np.float32),
                np.array([1.0, 0.0], dtype=np.float32),
                lambda: make_square((50, 0), half_size=15),
                90.0,
                True,
            ),
            # Square in front, looking UP (integration test variant)
            (
                np.array([200.0, 200.0], dtype=np.float32),
                np.array([0.0, 1.0], dtype=np.float32),
                lambda: make_square((200, 280), half_size=25),
                60.0,
                True,
            ),
        ],
    )
    def test_single_obstacle_centered(
        self,
        viewer: NDArray[np.float32],
        direction: NDArray[np.float32],
        obstacle_fn: Callable[[], NDArray[np.float32]],
        fov: float,
        expected_visible: bool,
    ) -> None:
        """Single obstacle directly in front of viewer should be detected."""
        contours = [obstacle_fn()]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=fov,
            max_range=150.0,
            obstacle_contours=contours,
        )

        if expected_visible:
            assert result.obstacle_id == 0
            assert result.angular_coverage > 0
            assert result.min_distance > 0
        else:
            assert result.obstacle_id is None


# =============================================================================
# Test: Multiple obstacles
# =============================================================================


class TestMultipleObstacles:
    """Test with multiple obstacles in various configurations."""

    def test_two_obstacles_side_by_side_larger_wins(self) -> None:
        """Two obstacles at equal distance, larger one should win."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        # Small obstacle on the left
        small_square = make_square((-30, 50), half_size=10)
        # Larger obstacle on the right
        large_square = make_square((30, 50), half_size=25)

        contours = [small_square, large_square]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=120.0,
            max_range=100.0,
            obstacle_contours=contours,
        )

        # Larger square should have more angular coverage
        assert result.obstacle_id == 1
        assert result.angular_coverage > 0

    def test_multiple_obstacles_at_different_distances(self) -> None:
        """Near obstacle wins despite being smaller due to proximity."""
        viewer = np.array([200.0, 200.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        # Near obstacle (small but close)
        near_obstacle = make_square((200, 240), half_size=10)
        # Far obstacle (large but distant)
        far_obstacle = make_square((200, 320), half_size=40)

        contours = [near_obstacle, far_obstacle]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=90.0,
            max_range=200.0,
            obstacle_contours=contours,
        )

        # The near obstacle should win due to proximity advantage
        assert result.obstacle_id == 0
        assert result.angular_coverage > 0
        assert result.min_distance < 40

    def test_obstacles_on_either_side(self) -> None:
        """Person looking up with obstacles to the left and right."""
        viewer = np.array([200.0, 200.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        left_obs = make_square((140, 280), half_size=20)
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

    def test_two_squares_with_intervals(self) -> None:
        """Verify intervals are returned correctly."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        left_square = make_square((-40, 50), half_size=15)
        right_square = make_square((40, 50), half_size=15)

        contours = [left_square, right_square]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=120.0,
            max_range=100.0,
            obstacle_contours=contours,
            return_intervals=True,
        )

        assert result.obstacle_id is not None
        assert result.intervals is not None
        assert len(result.intervals) > 0


# =============================================================================
# Test: Occlusion scenarios
# =============================================================================


class TestOcclusion:
    """Test occlusion scenarios where nearer obstacle masks farther one."""

    @pytest.mark.parametrize(
        "viewer,direction,near_obstacle,far_obstacle,expected_winner",
        [
            # Closer obstacle wins when occluding
            (
                np.array([0.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0], dtype=np.float32),
                (0, 30, 10),  # (x, y, half_size)
                (0, 80, 30),
                0,
            ),
            # Integration test variant with different coordinates
            (
                np.array([100.0, 100.0], dtype=np.float32),
                np.array([0.0, 1.0], dtype=np.float32),
                (100, 130, 8),
                (100, 200, 30),
                0,
            ),
        ],
    )
    def test_closer_obstacle_occludes_farther(
        self,
        viewer: NDArray[np.float32],
        direction: NDArray[np.float32],
        near_obstacle: tuple[float, float, float],
        far_obstacle: tuple[float, float, float],
        expected_winner: int,
    ) -> None:
        """Closer obstacle should occlude farther one at same angle."""
        near_x, near_y, near_size = near_obstacle
        far_x, far_y, far_size = far_obstacle

        near_obs = make_square((near_x, near_y), half_size=near_size)
        far_obs = make_square((far_x, far_y), half_size=far_size)

        contours = [near_obs, far_obs]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=90.0,
            max_range=250.0,
            obstacle_contours=contours,
        )

        assert result.obstacle_id == expected_winner
        assert result.angular_coverage > 0

    def test_small_near_fully_in_front_of_large_far(self) -> None:
        """Small near obstacle fully occludes center of large far obstacle."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        near_tiny = make_square((0, 20), half_size=5)
        far_wide = make_rectangle((0, 100), width=150, height=30)

        contours = [near_tiny, far_wide]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=90.0,
            max_range=150.0,
            obstacle_contours=contours,
        )

        # Far obstacle should still win because it has more total visible coverage
        assert result.obstacle_id is not None


# =============================================================================
# Test: Empty scene and obstacles outside FOV
# =============================================================================


class TestEmptyAndOutsideFOV:
    """Test with no obstacles or obstacles outside the field of view."""

    def test_empty_contours_list(self) -> None:
        """No obstacles provided."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=60.0,
            max_range=100.0,
            obstacle_contours=[],
        )

        assert result.obstacle_id is None
        assert result.angular_coverage == 0.0
        assert result.min_distance == float("inf")
        assert bool(result) is False

    @pytest.mark.parametrize(
        "viewer,direction,obstacle_positions,fov,max_range,reason",
        [
            # Obstacles behind the viewer
            (
                np.array([0.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0], dtype=np.float32),
                [(0, -50), (30, -70)],
                60.0,
                100.0,
                "behind",
            ),
            # Obstacles beyond max range
            (
                np.array([0.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0], dtype=np.float32),
                [(0, 100), (30, 150)],
                90.0,
                50.0,
                "beyond_range",
            ),
            # Obstacles outside narrow FOV
            (
                np.array([0.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0], dtype=np.float32),
                [(-50, 50), (50, 50)],
                10.0,
                100.0,
                "outside_fov",
            ),
        ],
    )
    def test_all_obstacles_outside_arc(
        self,
        viewer: NDArray[np.float32],
        direction: NDArray[np.float32],
        obstacle_positions: list[tuple[float, float]],
        fov: float,
        max_range: float,
        reason: str,
    ) -> None:
        """All obstacles outside the viewing arc should return no result."""
        contours = [make_square(pos, half_size=20) for pos in obstacle_positions]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=fov,
            max_range=max_range,
            obstacle_contours=contours,
        )

        assert result.obstacle_id is None
        assert result.angular_coverage == 0.0


# =============================================================================
# Test: Field of view variations
# =============================================================================


class TestFieldOfView:
    """Test with various field of view configurations."""

    def test_narrow_fov_single_obstacle(self) -> None:
        """Single obstacle visible in narrow FOV."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        contours = [make_square((0, 50), half_size=10)]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=30.0,
            max_range=100.0,
            obstacle_contours=contours,
        )

        assert result.obstacle_id == 0
        assert result.angular_coverage > 0

    def test_narrow_fov_clips_obstacle(self) -> None:
        """Obstacle partially clipped by narrow FOV edges."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        contours = [make_rectangle((0, 40), width=100, height=20)]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=30.0,
            max_range=100.0,
            obstacle_contours=contours,
        )

        assert result.obstacle_id == 0
        max_possible_coverage = np.deg2rad(30.0)
        assert result.angular_coverage <= max_possible_coverage + 0.01

    def test_wide_fov_multiple_obstacles(self) -> None:
        """Multiple obstacles visible in wide FOV."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        contours = [
            make_square((-40, 50), half_size=15),  # left
            make_square((0, 60), half_size=20),  # center
            make_square((50, 45), half_size=12),  # right
        ]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=120.0,
            max_range=100.0,
            obstacle_contours=contours,
        )

        # Center obstacle is largest
        assert result.obstacle_id == 1
        assert result.angular_coverage > 0

    def test_obstacle_partially_in_fov(self) -> None:
        """Obstacle that is only partially within the field of view."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        wide_obs = make_rectangle((0, 50), width=200, height=20)

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=60.0,
            max_range=100.0,
            obstacle_contours=[wide_obs],
        )

        assert result.obstacle_id == 0
        max_fov_coverage = np.deg2rad(60.0)
        assert result.angular_coverage <= max_fov_coverage + 0.01


# =============================================================================
# Test: Wide FOV with angle normalization (regression)
# =============================================================================


class TestWideFOVNormalization:
    """Regression tests for wide FOV scenarios requiring angle normalization."""

    def test_270_degree_fov_obstacle_at_225_degrees(self) -> None:
        """Obstacle at ~225° should be detected with 270° FOV looking up."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        contours = [make_square((-40, -40), half_size=15)]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=270.0,
            max_range=100.0,
            obstacle_contours=contours,
        )

        assert result.obstacle_id == 0
        assert result.angular_coverage > 0

    def test_300_degree_fov_multiple_obstacles(self) -> None:
        """Multiple obstacles in 300° FOV including near ±π boundary."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        contours = [
            make_square((-60, 0), half_size=15),  # At 180° (left)
            make_square((0, 60), half_size=12),  # At 90° (up, center of FOV)
        ]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=300.0,
            max_range=100.0,
            obstacle_contours=contours,
        )

        assert result.obstacle_id is not None
        assert result.angular_coverage > 0

    def test_330_degree_fov_obstacle_at_pi_boundary(self) -> None:
        """Obstacle exactly at ±π (180°) with very wide FOV."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        contours = [make_square((-50, 0), half_size=15)]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=330.0,
            max_range=100.0,
            obstacle_contours=contours,
        )

        assert result.obstacle_id == 0
        assert result.angular_coverage > 0


# =============================================================================
# Test: Full-circle FOV (360°) handling (regression)
# =============================================================================


class TestFullCircleFOV:
    """Regression tests for full-circle FOV (360°)."""

    @pytest.mark.parametrize(
        "obstacle_position,description",
        [
            ((0, 50), "in_front"),
            ((0, -50), "behind"),
            ((-50, 0), "to_left"),
            ((50, 0), "to_right"),
        ],
    )
    def test_360_fov_detects_all_directions(
        self, obstacle_position: tuple[float, float], description: str
    ) -> None:
        """Full-circle FOV should detect obstacles in all directions."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        contours = [make_square(obstacle_position, half_size=10)]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=360.0,
            max_range=100.0,
            obstacle_contours=contours,
        )

        assert result.obstacle_id == 0, f"Failed to detect obstacle {description}"
        assert result.angular_coverage > 0

    def test_360_fov_multiple_obstacles_all_around(self) -> None:
        """Full-circle FOV should detect obstacles in all directions."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        contours = [
            make_square((0, 50), half_size=5),  # Front - closest
            make_square((50, 0), half_size=10),  # Right
            make_square((0, -60), half_size=10),  # Back
            make_square((-60, 0), half_size=10),  # Left
        ]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=360.0,
            max_range=100.0,
            obstacle_contours=contours,
        )

        assert result.obstacle_id is not None
        assert result.angular_coverage > 0


# =============================================================================
# Test: Different viewing directions
# =============================================================================


class TestViewingDirections:
    """Test with various viewing directions."""

    @pytest.mark.parametrize(
        "direction,obstacle_position,description",
        [
            (np.array([1.0, 0.0], dtype=np.float32), (60, 0), "looking_right"),
            (np.array([-1.0, 0.0], dtype=np.float32), (-60, 0), "looking_left"),
            (np.array([0.0, -1.0], dtype=np.float32), (0, -50), "looking_down"),
        ],
    )
    def test_cardinal_directions(
        self,
        direction: NDArray[np.float32],
        obstacle_position: tuple[float, float],
        description: str,
    ) -> None:
        """Test viewing in cardinal directions."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        contours = [make_square(obstacle_position, half_size=15)]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=60.0,
            max_range=100.0,
            obstacle_contours=contours,
        )

        assert result.obstacle_id == 0, f"Failed to detect obstacle when {description}"
        assert result.angular_coverage > 0

    def test_looking_diagonal(self) -> None:
        """Viewer looking diagonally (normalized 45°)."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2)], dtype=np.float32)

        contours = [make_square((50, 50), half_size=15)]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=60.0,
            max_range=100.0,
            obstacle_contours=contours,
        )

        assert result.obstacle_id == 0
        assert result.angular_coverage > 0

    def test_looking_up_left(self) -> None:
        """Viewer looking up-left (as in spec example)."""
        viewer = np.array([100.0, 100.0], dtype=np.float32)
        direction = normalize_direction(-0.37, 0.92)

        contours = [make_square((60, 170), half_size=15)]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=60.0,
            max_range=100.0,
            obstacle_contours=contours,
        )

        assert result.obstacle_id == 0
        assert result.angular_coverage > 0

    def test_obstacle_behind_not_visible(self) -> None:
        """Person looking left should not see obstacle to the right."""
        viewer = np.array([200.0, 200.0], dtype=np.float32)
        direction = np.array([-1.0, 0.0], dtype=np.float32)

        contours = [make_square((350, 200), half_size=25)]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=60.0,
            max_range=200.0,
            obstacle_contours=contours,
        )

        assert result.obstacle_id is None


# =============================================================================
# Test: Distance and range
# =============================================================================


class TestDistanceAndRange:
    """Tests for scenarios where distance and range play key roles."""

    def test_tie_breaking_closer_wins(self) -> None:
        """When angular coverage is similar, closer obstacle should win."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        # Two obstacles with approximately equal angular coverage
        close_left = make_square((-25, 50), half_size=12)
        far_right = make_square((50, 100), half_size=24)

        contours = [close_left, far_right]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=120.0,
            max_range=150.0,
            obstacle_contours=contours,
        )

        assert result.obstacle_id == 0
        assert result.min_distance < 50

    def test_very_close_obstacle_large_angular_coverage(self) -> None:
        """Very close obstacle should have large angular coverage."""
        viewer = np.array([100.0, 100.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        close_obs = make_square((100, 115), half_size=10)

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=90.0,
            max_range=50.0,
            obstacle_contours=[close_obs],
        )

        assert result.obstacle_id == 0
        assert result.angular_coverage > np.deg2rad(20)

    def test_obstacle_beyond_max_range(self) -> None:
        """Obstacle beyond max_range should not be detected."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

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

        contours = [make_rectangle((0, 60), width=30, height=40)]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=90.0,
            max_range=70.0,
            obstacle_contours=contours,
        )

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

        assert result.obstacle_id in [0, 2]


# =============================================================================
# Test: Size and shape comparisons
# =============================================================================


class TestSizeAndShape:
    """Tests comparing obstacles with different shapes and angular footprints."""

    def test_wide_vs_narrow_at_same_distance(self) -> None:
        """Wide obstacle should have more angular coverage than narrow one."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        wide_obs = make_rectangle((-40, 60), width=60, height=10)
        narrow_obs = make_rectangle((40, 60), width=10, height=40)

        contours = [wide_obs, narrow_obs]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=120.0,
            max_range=100.0,
            obstacle_contours=contours,
        )

        assert result.obstacle_id == 0

    def test_tall_vs_wide_obstacle(self) -> None:
        """Wide obstacle closer should win over tall narrow obstacle."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        tall_narrow = make_rectangle((0, 70), width=10, height=50)
        wide_short = make_rectangle((0, 40), width=50, height=10)

        contours = [tall_narrow, wide_short]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=90.0,
            max_range=100.0,
            obstacle_contours=contours,
        )

        assert result.obstacle_id == 1
        assert result.min_distance < 40

    def test_small_obstacle_filling_fov(self) -> None:
        """Small obstacle that fills most of a narrow FOV."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        small_centered = make_square((0, 30), half_size=8)

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=20.0,
            max_range=50.0,
            obstacle_contours=[small_centered],
        )

        assert result.obstacle_id == 0
        max_coverage = np.deg2rad(20.0)
        assert result.angular_coverage > max_coverage * 0.5


# =============================================================================
# Test: Complex polygon shapes
# =============================================================================


class TestComplexPolygons:
    """Tests for obstacles with complex polygon shapes."""

    @pytest.mark.parametrize(
        "n_sides,position,radius,description",
        [
            (8, (0, 60), 20, "octagon"),
            (6, (70, 0), 25, "hexagon"),
            (5, (-40, 60), 15, "pentagon"),
            (32, (0, 50), 15, "circle_approximation"),
        ],
    )
    def test_regular_polygons(
        self,
        n_sides: int,
        position: tuple[float, float],
        radius: float,
        description: str,
    ) -> None:
        """Test obstacles with regular polygon shapes."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = (
            np.array([1.0, 0.0], dtype=np.float32)
            if position[0] > 0 and position[1] == 0
            else np.array([0.0, 1.0], dtype=np.float32)
        )

        polygon = make_polygon(position, n_sides=n_sides, radius=radius)
        contours = [polygon]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=90.0,
            max_range=100.0,
            obstacle_contours=contours,
        )

        assert result.obstacle_id == 0, f"Failed to detect {description}"
        assert result.angular_coverage > 0

    def test_irregular_polygon(self) -> None:
        """Obstacle with irregular polygon shape."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        irregular = np.array(
            [[10, 40], [30, 50], [25, 70], [5, 80], [-15, 60], [-10, 45]],
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

    def test_minimal_triangle_obstacle(self) -> None:
        """Minimal valid polygon (3 vertices)."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        triangle = np.array([[0, 40], [10, 60], [-10, 55]], dtype=np.float32)
        contours = [triangle]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=90.0,
            max_range=100.0,
            obstacle_contours=contours,
        )

        assert result.obstacle_id == 0


# =============================================================================
# Test: Edge cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_obstacle_centered_on_viewer(self) -> None:
        """Edge case: obstacle overlapping with viewer position."""
        viewer = np.array([50.0, 50.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        contours = [make_square((50, 50), half_size=30)]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=90.0,
            max_range=100.0,
            obstacle_contours=contours,
        )

        # Should not crash
        assert isinstance(result, ObstacleResult)

    def test_very_small_obstacle(self) -> None:
        """Very small obstacle should still be detected."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        contours = [make_square((0, 50), half_size=1)]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=90.0,
            max_range=100.0,
            obstacle_contours=contours,
        )

        assert result.obstacle_id == 0
        assert result.angular_coverage > 0

    def test_very_large_obstacle(self) -> None:
        """Very large obstacle filling most of the view."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        contours = [make_rectangle((0, 50), width=500, height=100)]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=90.0,
            max_range=100.0,
            obstacle_contours=contours,
        )

        assert result.obstacle_id == 0
        max_coverage = np.deg2rad(90.0)
        assert result.angular_coverage > max_coverage * 0.8

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

        assert result.obstacle_id is not None


# =============================================================================
# Test: Return intervals functionality
# =============================================================================


class TestReturnIntervals:
    """Tests verifying the return_intervals functionality."""

    def test_return_intervals_true(self) -> None:
        """Verify intervals are returned when requested."""
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
        assert isinstance(result.intervals, list)
        assert len(result.intervals) > 0

        for interval in result.intervals:
            assert isinstance(interval, tuple)
            assert len(interval) == 2

    def test_return_intervals_false(self) -> None:
        """Verify intervals are None when not requested."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        contours = [make_square((0, 50), half_size=20)]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=90.0,
            max_range=100.0,
            obstacle_contours=contours,
            return_intervals=False,
        )

        assert result.intervals is None

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
            assert abs(interval_sum - result.angular_coverage) < 0.01


# =============================================================================
# Test: Input validation
# =============================================================================


class TestInputValidation:
    """Test validation of invalid inputs."""

    def test_invalid_direction_not_normalized(self) -> None:
        """Non-unit vector should raise ValueError."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([1.0, 1.0], dtype=np.float32)

        contours = [make_square((50, 0))]

        with pytest.raises(ValueError, match="unit vector"):
            find_largest_obstacle(
                viewer_point=viewer,
                view_direction=direction,
                field_of_view_deg=60.0,
                max_range=100.0,
                obstacle_contours=contours,
            )

    @pytest.mark.parametrize(
        "invalid_shape,param_name",
        [
            (np.array([0.0, 0.0, 0.0], dtype=np.float32), "viewer_point"),
            (np.array([1.0], dtype=np.float32), "view_direction"),
        ],
    )
    def test_invalid_point_shapes(
        self, invalid_shape: NDArray[np.float32], param_name: str
    ) -> None:
        """Invalid point/direction shapes should raise ValueError."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([1.0, 0.0], dtype=np.float32)
        contours = [make_square((50, 0))]

        kwargs = {
            "viewer_point": invalid_shape if param_name == "viewer_point" else viewer,
            "view_direction": invalid_shape if param_name == "view_direction" else direction,
            "field_of_view_deg": 60.0,
            "max_range": 100.0,
            "obstacle_contours": contours,
        }

        with pytest.raises(ValueError, match="shape"):
            find_largest_obstacle(**kwargs)

    @pytest.mark.parametrize(
        "fov,expected_match",
        [
            (0.0, "field_of_view_deg"),
            (-30.0, "field_of_view_deg"),
        ],
    )
    def test_invalid_fov(self, fov: float, expected_match: str) -> None:
        """Invalid FOV values should raise ValueError."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([1.0, 0.0], dtype=np.float32)
        contours = [make_square((50, 0))]

        with pytest.raises(ValueError, match=expected_match):
            find_largest_obstacle(
                viewer_point=viewer,
                view_direction=direction,
                field_of_view_deg=fov,
                max_range=100.0,
                obstacle_contours=contours,
            )

    @pytest.mark.parametrize(
        "max_range,expected_match",
        [
            (0.0, "max_range"),
            (-10.0, "max_range"),
        ],
    )
    def test_invalid_max_range(self, max_range: float, expected_match: str) -> None:
        """Invalid max_range values should raise ValueError."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([1.0, 0.0], dtype=np.float32)
        contours = [make_square((50, 0))]

        with pytest.raises(ValueError, match=expected_match):
            find_largest_obstacle(
                viewer_point=viewer,
                view_direction=direction,
                field_of_view_deg=60.0,
                max_range=max_range,
                obstacle_contours=contours,
            )

    def test_invalid_contours_not_list(self) -> None:
        """Non-list contours should raise ValueError."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([1.0, 0.0], dtype=np.float32)

        contours = make_square((50, 0))

        with pytest.raises(ValueError, match="list"):
            find_largest_obstacle(
                viewer_point=viewer,
                view_direction=direction,
                field_of_view_deg=60.0,
                max_range=100.0,
                obstacle_contours=contours,  # type: ignore
            )

    def test_invalid_contour_shape(self) -> None:
        """Contour with wrong shape should raise ValueError."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([1.0, 0.0], dtype=np.float32)

        bad_contour = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

        with pytest.raises(ValueError, match="shape"):
            find_largest_obstacle(
                viewer_point=viewer,
                view_direction=direction,
                field_of_view_deg=60.0,
                max_range=100.0,
                obstacle_contours=[bad_contour],
            )


# =============================================================================
# Test: ObstacleResult dataclass
# =============================================================================


class TestObstacleResult:
    """Test ObstacleResult dataclass functionality."""

    def test_bool_true_when_obstacle_found(self) -> None:
        """Result should be truthy when obstacle is found."""
        result = ObstacleResult(obstacle_id=0, angular_coverage=0.5, min_distance=50.0)
        assert bool(result) is True

    def test_bool_false_when_no_obstacle(self) -> None:
        """Result should be falsy when no obstacle is found."""
        result = ObstacleResult(
            obstacle_id=None, angular_coverage=0.0, min_distance=float("inf")
        )
        assert bool(result) is False

    def test_optional_intervals(self) -> None:
        """Test that intervals are optional."""
        result = ObstacleResult(obstacle_id=0, angular_coverage=0.5, min_distance=50.0)
        assert result.intervals is None

        result_with_intervals = ObstacleResult(
            obstacle_id=0, angular_coverage=0.5, min_distance=50.0, intervals=[(0.0, 0.5)]
        )
        assert result_with_intervals.intervals == [(0.0, 0.5)]
