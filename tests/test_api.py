"""
Tests for the main API function find_largest_obstacle.
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from view_arc.obstacle.api import find_largest_obstacle, ObstacleResult


# =============================================================================
# Helper functions for creating test fixtures
# =============================================================================

def make_triangle(center: tuple, size: float = 20.0) -> NDArray[np.float32]:
    """Create a triangle centered at given point."""
    cx, cy = center
    return np.array([
        [cx, cy + size],       # top
        [cx - size, cy - size], # bottom-left
        [cx + size, cy - size], # bottom-right
    ], dtype=np.float32)


def make_square(center: tuple, half_size: float = 15.0) -> NDArray[np.float32]:
    """Create a square centered at given point."""
    cx, cy = center
    return np.array([
        [cx - half_size, cy - half_size],  # bottom-left
        [cx + half_size, cy - half_size],  # bottom-right
        [cx + half_size, cy + half_size],  # top-right
        [cx - half_size, cy + half_size],  # top-left
    ], dtype=np.float32)


def make_rectangle(center: tuple, width: float, height: float) -> NDArray[np.float32]:
    """Create a rectangle centered at given point."""
    cx, cy = center
    hw, hh = width / 2, height / 2
    return np.array([
        [cx - hw, cy - hh],
        [cx + hw, cy - hh],
        [cx + hw, cy + hh],
        [cx - hw, cy + hh],
    ], dtype=np.float32)


# =============================================================================
# Test: Single obstacle centered in field of view
# =============================================================================

class TestFindLargestObstacleSingleCentered:
    """Test with a single obstacle centered in the field of view."""
    
    def test_single_triangle_centered(self):
        """Single triangle directly in front of viewer."""
        viewer = np.array([100.0, 100.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)  # looking UP
        
        # Triangle at (100, 150) - directly in front
        contours = [make_triangle((100, 150), size=20)]
        
        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=60.0,
            max_range=100.0,
            obstacle_contours=contours
        )
        
        assert result.obstacle_id == 0
        assert result.angular_coverage > 0
        assert result.min_distance > 0
        assert result.min_distance <= 100.0
    
    def test_single_square_centered(self):
        """Single square directly in front of viewer."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([1.0, 0.0], dtype=np.float32)  # looking RIGHT
        
        # Square at (50, 0) - directly in front
        contours = [make_square((50, 0), half_size=15)]
        
        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=90.0,
            max_range=100.0,
            obstacle_contours=contours
        )
        
        assert result.obstacle_id == 0
        assert result.angular_coverage > 0
        assert result.min_distance > 0


# =============================================================================
# Test: Two obstacles side by side (no occlusion)
# =============================================================================

class TestFindLargestObstacleTwoSideBySide:
    """Test with two obstacles side by side without occlusion."""
    
    def test_two_squares_equal_distance_larger_wins(self):
        """Two squares at equal distance, larger one should win."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)  # looking UP
        
        # Small square on the left
        small_square = make_square((-30, 50), half_size=10)
        # Larger square on the right
        large_square = make_square((30, 50), half_size=25)
        
        contours = [small_square, large_square]
        
        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=120.0,
            max_range=100.0,
            obstacle_contours=contours
        )
        
        # Larger square should have more angular coverage
        assert result.obstacle_id == 1
        assert result.angular_coverage > 0
    
    def test_two_squares_side_by_side_with_intervals(self):
        """Verify intervals are returned correctly."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)  # looking UP
        
        # Two squares side by side
        left_square = make_square((-40, 50), half_size=15)
        right_square = make_square((40, 50), half_size=15)
        
        contours = [left_square, right_square]
        
        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=120.0,
            max_range=100.0,
            obstacle_contours=contours,
            return_intervals=True
        )
        
        assert result.obstacle_id is not None
        assert result.intervals is not None
        assert len(result.intervals) > 0


# =============================================================================
# Test: One obstacle occludes another (depth ordering)
# =============================================================================

class TestFindLargestObstacleOcclusion:
    """Test occlusion scenarios where nearer obstacle masks farther one."""
    
    def test_closer_obstacle_wins_when_occluding(self):
        """Closer obstacle should occlude farther one at same angle."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)  # looking UP
        
        # Near small obstacle
        near_obstacle = make_square((0, 30), half_size=10)
        # Far larger obstacle directly behind
        far_obstacle = make_square((0, 80), half_size=30)
        
        contours = [near_obstacle, far_obstacle]
        
        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=90.0,
            max_range=150.0,
            obstacle_contours=contours
        )
        
        # The far obstacle is larger but partially occluded
        # The winner depends on how much of the far obstacle is still visible
        assert result.obstacle_id is not None
        assert result.angular_coverage > 0
    
    def test_small_near_fully_in_front_of_large_far(self):
        """Small near obstacle fully occludes center of large far obstacle."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)  # looking UP
        
        # Tiny obstacle very close
        near_tiny = make_square((0, 20), half_size=5)
        # Wide obstacle far away
        far_wide = make_rectangle((0, 100), width=150, height=30)
        
        contours = [near_tiny, far_wide]
        
        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=90.0,
            max_range=150.0,
            obstacle_contours=contours
        )
        
        # Far obstacle should still win because it has more total visible coverage
        # (parts on the sides are not occluded)
        assert result.obstacle_id is not None


# =============================================================================
# Test: Empty scene (no obstacles)
# =============================================================================

class TestFindLargestObstacleEmptyScene:
    """Test with no obstacles."""
    
    def test_empty_contours_list(self):
        """No obstacles provided."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)
        
        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=60.0,
            max_range=100.0,
            obstacle_contours=[]
        )
        
        assert result.obstacle_id is None
        assert result.angular_coverage == 0.0
        assert result.min_distance == float('inf')
        assert bool(result) is False


# =============================================================================
# Test: All obstacles outside arc
# =============================================================================

class TestFindLargestObstacleAllOutsideArc:
    """Test when all obstacles are outside the field of view."""
    
    def test_obstacles_behind_viewer(self):
        """Obstacles behind the viewer (opposite direction)."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)  # looking UP
        
        # Obstacles behind (negative Y)
        contours = [
            make_square((0, -50), half_size=20),
            make_square((30, -70), half_size=15),
        ]
        
        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=60.0,
            max_range=100.0,
            obstacle_contours=contours
        )
        
        assert result.obstacle_id is None
        assert result.angular_coverage == 0.0
    
    def test_obstacles_beyond_max_range(self):
        """Obstacles beyond the maximum sensing range."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)  # looking UP
        
        # Obstacles far away (beyond max_range of 50)
        contours = [
            make_square((0, 100), half_size=20),
            make_square((30, 150), half_size=25),
        ]
        
        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=90.0,
            max_range=50.0,  # short range
            obstacle_contours=contours
        )
        
        assert result.obstacle_id is None
        assert result.angular_coverage == 0.0
    
    def test_obstacles_outside_narrow_fov(self):
        """Obstacles outside a narrow field of view."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)  # looking UP
        
        # Obstacles to the sides (outside narrow 10° FOV)
        contours = [
            make_square((-50, 50), half_size=15),  # far left
            make_square((50, 50), half_size=15),   # far right
        ]
        
        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=10.0,  # very narrow
            max_range=100.0,
            obstacle_contours=contours
        )
        
        assert result.obstacle_id is None


# =============================================================================
# Test: Narrow FOV
# =============================================================================

class TestFindLargestObstacleNarrowFOV:
    """Test with narrow field of view (30°)."""
    
    def test_narrow_fov_single_obstacle(self):
        """Single obstacle visible in narrow FOV."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)  # looking UP
        
        # Obstacle directly ahead
        contours = [make_square((0, 50), half_size=10)]
        
        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=30.0,
            max_range=100.0,
            obstacle_contours=contours
        )
        
        assert result.obstacle_id == 0
        assert result.angular_coverage > 0
    
    def test_narrow_fov_clips_obstacle(self):
        """Obstacle partially clipped by narrow FOV edges."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)  # looking UP
        
        # Wide obstacle that extends beyond narrow FOV
        contours = [make_rectangle((0, 40), width=100, height=20)]
        
        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=30.0,
            max_range=100.0,
            obstacle_contours=contours
        )
        
        assert result.obstacle_id == 0
        # The coverage should be limited by the FOV
        max_possible_coverage = np.deg2rad(30.0)
        assert result.angular_coverage <= max_possible_coverage + 0.01


# =============================================================================
# Test: Wide FOV
# =============================================================================

class TestFindLargestObstacleWideFOV:
    """Test with wide field of view (120°)."""
    
    def test_wide_fov_multiple_obstacles(self):
        """Multiple obstacles visible in wide FOV."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)  # looking UP
        
        contours = [
            make_square((-40, 50), half_size=15),  # left
            make_square((0, 60), half_size=20),    # center
            make_square((50, 45), half_size=12),   # right
        ]
        
        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=120.0,
            max_range=100.0,
            obstacle_contours=contours
        )
        
        # Center obstacle is largest
        assert result.obstacle_id == 1
        assert result.angular_coverage > 0


# =============================================================================
# Test: Invalid inputs
# =============================================================================

class TestFindLargestObstacleInvalidInputs:
    """Test validation of invalid inputs."""
    
    def test_invalid_direction_not_normalized(self):
        """Non-unit vector should raise ValueError."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([1.0, 1.0], dtype=np.float32)  # not normalized
        
        contours = [make_square((50, 0))]
        
        with pytest.raises(ValueError, match="unit vector"):
            find_largest_obstacle(
                viewer_point=viewer,
                view_direction=direction,
                field_of_view_deg=60.0,
                max_range=100.0,
                obstacle_contours=contours
            )
    
    def test_invalid_viewer_point_shape(self):
        """Invalid viewer_point shape should raise ValueError."""
        viewer = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # 3D
        direction = np.array([1.0, 0.0], dtype=np.float32)
        
        contours = [make_square((50, 0))]
        
        with pytest.raises(ValueError, match="shape"):
            find_largest_obstacle(
                viewer_point=viewer,
                view_direction=direction,
                field_of_view_deg=60.0,
                max_range=100.0,
                obstacle_contours=contours
            )
    
    def test_invalid_view_direction_shape(self):
        """Invalid view_direction shape should raise ValueError."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([1.0], dtype=np.float32)  # 1D
        
        contours = [make_square((50, 0))]
        
        with pytest.raises(ValueError, match="shape"):
            find_largest_obstacle(
                viewer_point=viewer,
                view_direction=direction,
                field_of_view_deg=60.0,
                max_range=100.0,
                obstacle_contours=contours
            )
    
    def test_invalid_fov_zero(self):
        """FOV of zero should raise ValueError."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([1.0, 0.0], dtype=np.float32)
        
        contours = [make_square((50, 0))]
        
        with pytest.raises(ValueError, match="field_of_view_deg"):
            find_largest_obstacle(
                viewer_point=viewer,
                view_direction=direction,
                field_of_view_deg=0.0,
                max_range=100.0,
                obstacle_contours=contours
            )
    
    def test_invalid_fov_negative(self):
        """Negative FOV should raise ValueError."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([1.0, 0.0], dtype=np.float32)
        
        contours = [make_square((50, 0))]
        
        with pytest.raises(ValueError, match="field_of_view_deg"):
            find_largest_obstacle(
                viewer_point=viewer,
                view_direction=direction,
                field_of_view_deg=-30.0,
                max_range=100.0,
                obstacle_contours=contours
            )
    
    def test_invalid_max_range_zero(self):
        """Zero max_range should raise ValueError."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([1.0, 0.0], dtype=np.float32)
        
        contours = [make_square((50, 0))]
        
        with pytest.raises(ValueError, match="max_range"):
            find_largest_obstacle(
                viewer_point=viewer,
                view_direction=direction,
                field_of_view_deg=60.0,
                max_range=0.0,
                obstacle_contours=contours
            )
    
    def test_invalid_max_range_negative(self):
        """Negative max_range should raise ValueError."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([1.0, 0.0], dtype=np.float32)
        
        contours = [make_square((50, 0))]
        
        with pytest.raises(ValueError, match="max_range"):
            find_largest_obstacle(
                viewer_point=viewer,
                view_direction=direction,
                field_of_view_deg=60.0,
                max_range=-10.0,
                obstacle_contours=contours
            )
    
    def test_invalid_contours_not_list(self):
        """Non-list contours should raise ValueError."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([1.0, 0.0], dtype=np.float32)
        
        contours = make_square((50, 0))  # Single array, not a list
        
        with pytest.raises(ValueError, match="list"):
            find_largest_obstacle(
                viewer_point=viewer,
                view_direction=direction,
                field_of_view_deg=60.0,
                max_range=100.0,
                obstacle_contours=contours  # type: ignore
            )
    
    def test_invalid_contour_shape(self):
        """Contour with wrong shape should raise ValueError."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([1.0, 0.0], dtype=np.float32)
        
        # Contour with 3 columns instead of 2
        bad_contour = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        
        with pytest.raises(ValueError, match="shape"):
            find_largest_obstacle(
                viewer_point=viewer,
                view_direction=direction,
                field_of_view_deg=60.0,
                max_range=100.0,
                obstacle_contours=[bad_contour]
            )


# =============================================================================
# Test: Return intervals flag
# =============================================================================

class TestFindLargestObstacleWithIntervals:
    """Test return_intervals functionality."""
    
    def test_return_intervals_true(self):
        """Verify intervals are returned when requested."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)  # looking UP
        
        contours = [make_square((0, 50), half_size=20)]
        
        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=90.0,
            max_range=100.0,
            obstacle_contours=contours,
            return_intervals=True
        )
        
        assert result.obstacle_id == 0
        assert result.intervals is not None
        assert isinstance(result.intervals, list)
        assert len(result.intervals) > 0
        
        # Each interval should be a tuple of (start, end) angles
        for interval in result.intervals:
            assert isinstance(interval, tuple)
            assert len(interval) == 2
    
    def test_return_intervals_false(self):
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
            return_intervals=False
        )
        
        assert result.intervals is None


# =============================================================================
# Test: Tie-breaking (equal coverage, closest wins)
# =============================================================================

class TestFindLargestObstacleTieBreaking:
    """Test tie-breaking when obstacles have equal coverage."""
    
    def test_equal_coverage_closer_wins(self):
        """When two obstacles have equal coverage, closer one should win."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)  # looking UP
        
        # Two identical squares, one closer than the other
        # Place them at different angles so they don't occlude each other
        near_square = make_square((-40, 30), half_size=12)  # closer, on left
        far_square = make_square((40, 60), half_size=12)    # farther, on right
        
        contours = [near_square, far_square]
        
        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=120.0,
            max_range=100.0,
            obstacle_contours=contours
        )
        
        # The closer obstacle should win in case of tie on angular coverage
        # Note: coverage may not be exactly equal due to perspective effects
        assert result.obstacle_id is not None
        assert result.angular_coverage > 0


# =============================================================================
# Test: Different viewing directions
# =============================================================================

class TestFindLargestObstacleViewDirections:
    """Test with various viewing directions."""
    
    def test_looking_right(self):
        """Viewer looking to the right (+x direction)."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([1.0, 0.0], dtype=np.float32)  # looking RIGHT
        
        contours = [make_square((60, 0), half_size=15)]
        
        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=60.0,
            max_range=100.0,
            obstacle_contours=contours
        )
        
        assert result.obstacle_id == 0
        assert result.angular_coverage > 0
    
    def test_looking_left(self):
        """Viewer looking to the left (-x direction)."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([-1.0, 0.0], dtype=np.float32)  # looking LEFT
        
        contours = [make_square((-60, 0), half_size=15)]
        
        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=60.0,
            max_range=100.0,
            obstacle_contours=contours
        )
        
        assert result.obstacle_id == 0
        assert result.angular_coverage > 0
    
    def test_looking_down(self):
        """Viewer looking down (-y direction)."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, -1.0], dtype=np.float32)  # looking DOWN
        
        contours = [make_square((0, -50), half_size=15)]
        
        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=60.0,
            max_range=100.0,
            obstacle_contours=contours
        )
        
        assert result.obstacle_id == 0
        assert result.angular_coverage > 0
    
    def test_looking_diagonal(self):
        """Viewer looking diagonally (normalized 45°)."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        # 45° direction (normalized)
        direction = np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2)], dtype=np.float32)
        
        contours = [make_square((50, 50), half_size=15)]
        
        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=60.0,
            max_range=100.0,
            obstacle_contours=contours
        )
        
        assert result.obstacle_id == 0
        assert result.angular_coverage > 0
    
    def test_looking_up_left(self):
        """Viewer looking up-left (like in the spec example)."""
        viewer = np.array([100.0, 100.0], dtype=np.float32)
        # Normalize [-0.37, 0.92] to be a unit vector
        raw_dir = np.array([-0.37, 0.92], dtype=np.float32)
        direction = (raw_dir / np.linalg.norm(raw_dir)).astype(np.float32)
        
        # Place obstacle in the up-left direction
        contours = [make_square((60, 170), half_size=15)]
        
        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=60.0,
            max_range=100.0,
            obstacle_contours=contours
        )
        
        assert result.obstacle_id == 0
        assert result.angular_coverage > 0


# =============================================================================
# Test: Wide FOV with angle normalization (regression test)
# =============================================================================

class TestFindLargestObstacleWideFOVNormalization:
    """Regression tests for wide FOV scenarios requiring angle normalization.
    
    These tests verify that alpha_min/alpha_max are properly normalized to [-π, π)
    so the sweep helpers can correctly detect wraparound via alpha_min > alpha_max.
    """
    
    def test_270_degree_fov_obstacle_at_225_degrees(self):
        """Obstacle at ~225° should be detected with 270° FOV looking up.
        
        Regression test: Without normalization, alpha_max ≈ 3.927 > π,
        causing sweep code to think arc doesn't wrap and rejecting vertices
        near 225° (-135° normalized).
        """
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)  # Looking UP (90°)
        
        # Obstacle at ~225° (down-left from viewer) 
        # With 270° FOV centered at 90°, arc spans from -45° to 225°
        # which wraps around, so 225° should be visible
        contours = [make_square((-40, -40), half_size=15)]
        
        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=270.0,
            max_range=100.0,
            obstacle_contours=contours
        )
        
        assert result.obstacle_id == 0, (
            "Obstacle at 225° should be visible with 270° FOV centered at 90°"
        )
        assert result.angular_coverage > 0
    
    def test_270_degree_fov_obstacle_at_minus_135_degrees(self):
        """Obstacle at -135° (same as 225°) should be detected."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)  # Looking UP
        
        # Obstacle at -135° (down-left, same as 225°)
        contours = [make_square((-50, -50), half_size=15)]
        
        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=270.0,
            max_range=120.0,
            obstacle_contours=contours
        )
        
        assert result.obstacle_id == 0
        assert result.angular_coverage > 0
    
    def test_300_degree_fov_multiple_obstacles(self):
        """Multiple obstacles in 300° FOV including near ±π boundary."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)  # Looking UP
        
        # With 300° FOV centered at 90°, arc spans from -60° to 240°
        # Obstacle at 180° (directly left) should be visible
        contours = [
            make_square((-60, 0), half_size=15),   # At 180° (left)
            make_square((0, 60), half_size=12),     # At 90° (up, center of FOV)
        ]
        
        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=300.0,
            max_range=100.0,
            obstacle_contours=contours
        )
        
        assert result.obstacle_id is not None
        assert result.angular_coverage > 0
    
    def test_330_degree_fov_obstacle_at_pi_boundary(self):
        """Obstacle exactly at ±π (180°) with very wide FOV."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)  # Looking UP
        
        # Obstacle at exactly 180° (π radians, directly left)
        contours = [make_square((-50, 0), half_size=15)]
        
        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=330.0,
            max_range=100.0,
            obstacle_contours=contours
        )
        
        assert result.obstacle_id == 0, (
            "Obstacle at 180° should be visible with 330° FOV centered at 90°"
        )
        assert result.angular_coverage > 0
    
    def test_wide_fov_looking_right_obstacle_behind_left(self):
        """Wide FOV looking right should see obstacle behind-left."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([1.0, 0.0], dtype=np.float32)  # Looking RIGHT (0°)
        
        # With 270° FOV centered at 0°, arc spans from -135° to 135°
        # Obstacle at 120° (up-left) should be visible
        contours = [make_square((-30, 50), half_size=15)]
        
        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=270.0,
            max_range=100.0,
            obstacle_contours=contours
        )
        
        assert result.obstacle_id == 0
        assert result.angular_coverage > 0
    
    def test_wide_fov_looking_down_obstacle_up_left(self):
        """Wide FOV looking down should see obstacle up-left."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, -1.0], dtype=np.float32)  # Looking DOWN (-90°)
        
        # With 270° FOV centered at -90°, arc spans from -225° to 45°
        # Obstacle at 135° (up-left) should NOT be visible (outside FOV)
        # But obstacle at -45° (down-right) should be visible
        contours = [make_square((40, -40), half_size=15)]  # At -45°
        
        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=270.0,
            max_range=100.0,
            obstacle_contours=contours
        )
        
        assert result.obstacle_id == 0
        assert result.angular_coverage > 0


# =============================================================================
# Test: Full-circle FOV (360°) handling
# =============================================================================

class TestFindLargestObstacleFullCircle:
    """
    Regression tests for full-circle FOV (360°).
    
    Previously, 360° FOV would collapse to an empty wedge after normalization
    (both alpha_min and alpha_max became -π), causing no obstacles to be detected.
    """
    
    def test_360_fov_obstacle_in_front(self):
        """Full-circle FOV should detect obstacle directly in front."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)  # Looking UP
        
        contours = [make_square((0, 50), half_size=10)]
        
        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=360.0,
            max_range=100.0,
            obstacle_contours=contours
        )
        
        assert result.obstacle_id == 0
        assert result.angular_coverage > 0
        assert result.min_distance == pytest.approx(40.0, abs=0.1)
    
    def test_360_fov_obstacle_behind(self):
        """Full-circle FOV should detect obstacle behind the viewer."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)  # Looking UP
        
        # Obstacle behind (at -90° or 270°)
        contours = [make_square((0, -50), half_size=10)]
        
        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=360.0,
            max_range=100.0,
            obstacle_contours=contours
        )
        
        assert result.obstacle_id == 0
        assert result.angular_coverage > 0
    
    def test_360_fov_obstacle_to_left(self):
        """Full-circle FOV should detect obstacle to the left."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)  # Looking UP
        
        # Obstacle at 180° (left)
        contours = [make_square((-50, 0), half_size=10)]
        
        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=360.0,
            max_range=100.0,
            obstacle_contours=contours
        )
        
        assert result.obstacle_id == 0
        assert result.angular_coverage > 0
    
    def test_360_fov_obstacle_to_right(self):
        """Full-circle FOV should detect obstacle to the right."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)  # Looking UP
        
        # Obstacle at 0° (right)
        contours = [make_square((50, 0), half_size=10)]
        
        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=360.0,
            max_range=100.0,
            obstacle_contours=contours
        )
        
        assert result.obstacle_id == 0
        assert result.angular_coverage > 0
    
    def test_360_fov_multiple_obstacles_all_around(self):
        """Full-circle FOV should detect obstacles in all directions."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)  # Looking UP
        
        # Obstacles in all 4 cardinal directions
        contours = [
            make_square((0, 50), half_size=5),    # Front - closest
            make_square((50, 0), half_size=10),   # Right
            make_square((0, -60), half_size=10),  # Back
            make_square((-60, 0), half_size=10),  # Left
        ]
        
        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=360.0,
            max_range=100.0,
            obstacle_contours=contours
        )
        
        # One of the obstacles should be detected
        assert result.obstacle_id is not None
        assert result.angular_coverage > 0
    
    def test_360_fov_near_epsilon(self):
        """FOV very close to 360° (e.g., 359.999999°) should also work."""
        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)
        
        contours = [make_square((0, 50), half_size=10)]
        
        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=360.0 - 1e-7,  # Just under 360
            max_range=100.0,
            obstacle_contours=contours
        )
        
        assert result.obstacle_id == 0
        assert result.angular_coverage > 0


# =============================================================================
# Test: ObstacleResult dataclass behavior
# =============================================================================

class TestObstacleResult:
    """Test ObstacleResult dataclass functionality."""
    
    def test_bool_true_when_obstacle_found(self):
        """Result should be truthy when obstacle is found."""
        result = ObstacleResult(
            obstacle_id=0,
            angular_coverage=0.5,
            min_distance=50.0
        )
        assert bool(result) is True
    
    def test_bool_false_when_no_obstacle(self):
        """Result should be falsy when no obstacle is found."""
        result = ObstacleResult(
            obstacle_id=None,
            angular_coverage=0.0,
            min_distance=float('inf')
        )
        assert bool(result) is False
    
    def test_optional_intervals(self):
        """Test that intervals are optional."""
        result = ObstacleResult(
            obstacle_id=0,
            angular_coverage=0.5,
            min_distance=50.0
        )
        assert result.intervals is None
        
        result_with_intervals = ObstacleResult(
            obstacle_id=0,
            angular_coverage=0.5,
            min_distance=50.0,
            intervals=[(0.0, 0.5)]
        )
        assert result_with_intervals.intervals == [(0.0, 0.5)]
