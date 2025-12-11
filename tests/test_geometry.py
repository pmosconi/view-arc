"""
Tests for geometry utilities - Phase 1 of implementation.

Step 1.1: Basic Operations
- test_normalize_angle_*
- test_to_viewer_frame_*
- test_to_polar_*
- test_validate_direction_*

Step 1.2: Ray Intersection
- test_intersect_ray_*
- test_handle_angle_discontinuity_*
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_almost_equal

from view_arc.obstacle.geometry import (
    normalize_angle,
    to_viewer_frame,
    to_polar,
    validate_and_get_direction_angle,
    intersect_ray_segment,
    handle_angle_discontinuity,
)


# =============================================================================
# Step 1.1: Basic Operations Tests
# =============================================================================

class TestNormalizeAngle:
    """Tests for normalize_angle() function."""
    
    def test_normalize_angle_already_normalized(self):
        """Angle already in [-π, π) should remain unchanged."""
        assert_allclose(normalize_angle(0.0), 0.0)
        assert_allclose(normalize_angle(1.0), 1.0)
        assert_allclose(normalize_angle(-1.0), -1.0)
        assert_allclose(normalize_angle(np.pi / 2), np.pi / 2)
        assert_allclose(normalize_angle(-np.pi / 2), -np.pi / 2)
    
    def test_normalize_angle_various_ranges(self):
        """Test wrapping from different input ranges."""
        # Angles > π should wrap to negative
        assert_allclose(normalize_angle(np.pi + 0.5), -np.pi + 0.5)
        assert_allclose(normalize_angle(2 * np.pi), 0.0, atol=1e-10)
        assert_allclose(normalize_angle(3 * np.pi), -np.pi, atol=1e-10)
        
        # Angles < -π should wrap to positive
        assert_allclose(normalize_angle(-np.pi - 0.5), np.pi - 0.5)
        assert_allclose(normalize_angle(-2 * np.pi), 0.0, atol=1e-10)
        
        # Large positive angles
        assert_allclose(normalize_angle(4 * np.pi + 0.1), 0.1, atol=1e-10)
        
        # Large negative angles
        assert_allclose(normalize_angle(-4 * np.pi - 0.1), -0.1, atol=1e-10)
    
    def test_normalize_angle_boundary_cases(self):
        """Test behavior exactly at ±π boundaries."""
        # Exactly at π should wrap to -π (since range is [-π, π))
        result = normalize_angle(np.pi)
        assert result >= -np.pi and result < np.pi
        assert_allclose(abs(result), np.pi, atol=1e-10)
        
        # Just below π should remain
        assert_allclose(normalize_angle(np.pi - 1e-10), np.pi - 1e-10, atol=1e-9)
        
        # At -π should remain (it's included in the range)
        assert_allclose(normalize_angle(-np.pi), -np.pi, atol=1e-10)


class TestToViewerFrame:
    """Tests for to_viewer_frame() function."""
    
    def test_to_viewer_frame_single_point(self):
        """Single point coordinate translation."""
        point = np.array([5.0, 3.0], dtype=np.float32)
        origin = np.array([2.0, 1.0], dtype=np.float32)
        
        result = to_viewer_frame(point, origin)
        
        assert_array_almost_equal(result, np.array([3.0, 2.0]))
    
    def test_to_viewer_frame_multiple_points(self):
        """Batch translation with multiple points."""
        points = np.array([
            [5.0, 3.0],
            [2.0, 1.0],
            [0.0, 0.0],
            [-1.0, 4.0],
        ], dtype=np.float32)
        origin = np.array([2.0, 1.0], dtype=np.float32)
        
        result = to_viewer_frame(points, origin)
        
        expected = np.array([
            [3.0, 2.0],
            [0.0, 0.0],
            [-2.0, -1.0],
            [-3.0, 3.0],
        ])
        assert_array_almost_equal(result, expected)
    
    def test_to_viewer_frame_at_origin(self):
        """Point at viewer origin should become (0, 0)."""
        point = np.array([10.0, 20.0], dtype=np.float32)
        origin = np.array([10.0, 20.0], dtype=np.float32)
        
        result = to_viewer_frame(point, origin)
        
        assert_array_almost_equal(result, np.array([0.0, 0.0]))
    
    def test_to_viewer_frame_preserves_shape(self):
        """Output shape should match input shape."""
        # Single point
        single = np.array([1.0, 2.0], dtype=np.float32)
        origin = np.array([0.0, 0.0], dtype=np.float32)
        assert to_viewer_frame(single, origin).shape == (2,)
        
        # Multiple points
        multiple = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        assert to_viewer_frame(multiple, origin).shape == (2, 2)


class TestToPolar:
    """Tests for to_polar() function."""
    
    def test_to_polar_basic(self):
        """Convert known Cartesian points to polar."""
        # Point at (1, 0) -> r=1, θ=0
        points = np.array([[1.0, 0.0]], dtype=np.float32)
        radii, angles = to_polar(points)
        assert_allclose(radii[0], 1.0)
        assert_allclose(angles[0], 0.0, atol=1e-6)
        
        # Point at (0, 1) -> r=1, θ=π/2
        points = np.array([[0.0, 1.0]], dtype=np.float32)
        radii, angles = to_polar(points)
        assert_allclose(radii[0], 1.0)
        assert_allclose(angles[0], np.pi / 2, atol=1e-6)
        
        # Point at (1, 1) -> r=√2, θ=π/4
        points = np.array([[1.0, 1.0]], dtype=np.float32)
        radii, angles = to_polar(points)
        assert_allclose(radii[0], np.sqrt(2), atol=1e-6)
        assert_allclose(angles[0], np.pi / 4, atol=1e-6)
    
    def test_to_polar_angles(self):
        """Verify angle ranges and quadrants."""
        # All four quadrants
        points = np.array([
            [1.0, 0.0],    # 0° (positive x-axis)
            [0.0, 1.0],    # 90° (positive y-axis)
            [-1.0, 0.0],   # 180° (negative x-axis)
            [0.0, -1.0],   # -90° (negative y-axis)
            [1.0, 1.0],    # 45° (first quadrant)
            [-1.0, 1.0],   # 135° (second quadrant)
            [-1.0, -1.0],  # -135° (third quadrant)
            [1.0, -1.0],   # -45° (fourth quadrant)
        ], dtype=np.float32)
        
        radii, angles = to_polar(points)
        
        expected_angles = np.array([
            0.0,
            np.pi / 2,
            np.pi,
            -np.pi / 2,
            np.pi / 4,
            3 * np.pi / 4,
            -3 * np.pi / 4,
            -np.pi / 4,
        ])
        
        assert_allclose(angles, expected_angles, atol=1e-6)
    
    def test_to_polar_distances(self):
        """Verify distance calculations."""
        points = np.array([
            [3.0, 4.0],   # 3-4-5 triangle
            [5.0, 12.0],  # 5-12-13 triangle
            [0.0, 0.0],   # origin
        ], dtype=np.float32)
        
        radii, angles = to_polar(points)
        
        assert_allclose(radii[0], 5.0, atol=1e-6)
        assert_allclose(radii[1], 13.0, atol=1e-6)
        assert_allclose(radii[2], 0.0, atol=1e-6)
    
    def test_to_polar_single_point(self):
        """Handle single point input (shape (2,))."""
        point = np.array([3.0, 4.0], dtype=np.float32)
        radii, angles = to_polar(point)
        
        # Should return scalars or 0-d arrays
        assert_allclose(float(radii), 5.0, atol=1e-6)
        assert_allclose(float(angles), np.arctan2(4.0, 3.0), atol=1e-6)


class TestValidateAndGetDirectionAngle:
    """Tests for validate_and_get_direction_angle() function."""
    
    def test_validate_direction_normalized(self):
        """Accept valid unit vectors."""
        # Standard unit vectors
        angle = validate_and_get_direction_angle(np.array([1.0, 0.0]))
        assert_allclose(angle, 0.0, atol=1e-6)
        
        angle = validate_and_get_direction_angle(np.array([0.0, 1.0]))
        assert_allclose(angle, np.pi / 2, atol=1e-6)
        
        angle = validate_and_get_direction_angle(np.array([-1.0, 0.0]))
        assert_allclose(angle, np.pi, atol=1e-6)
        
        # Diagonal unit vectors
        sqrt2_inv = 1.0 / np.sqrt(2)
        angle = validate_and_get_direction_angle(np.array([sqrt2_inv, sqrt2_inv]))
        assert_allclose(angle, np.pi / 4, atol=1e-6)
    
    def test_validate_direction_not_normalized(self):
        """Reject non-unit vectors."""
        with pytest.raises(ValueError, match="unit vector"):
            validate_and_get_direction_angle(np.array([2.0, 0.0]))
        
        with pytest.raises(ValueError, match="unit vector"):
            validate_and_get_direction_angle(np.array([0.5, 0.5]))
        
        with pytest.raises(ValueError, match="unit vector"):
            validate_and_get_direction_angle(np.array([0.0, 0.0]))
    
    def test_validate_direction_angle_extraction(self):
        """Verify atan2 computation for various angles."""
        test_angles = [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, 
                       2*np.pi/3, 3*np.pi/4, np.pi,
                       -np.pi/4, -np.pi/2, -3*np.pi/4]
        
        for expected_angle in test_angles:
            direction = np.array([np.cos(expected_angle), np.sin(expected_angle)])
            actual_angle = validate_and_get_direction_angle(direction)
            assert_allclose(actual_angle, expected_angle, atol=1e-6)
    
    def test_validate_direction_tolerance(self):
        """Test custom tolerance parameter."""
        # Slightly non-unit vector that passes with loose tolerance
        direction = np.array([1.001, 0.0])
        
        # Should fail with default tolerance (1e-3)
        with pytest.raises(ValueError):
            validate_and_get_direction_angle(direction)
        
        # Should pass with looser tolerance
        angle = validate_and_get_direction_angle(direction, tolerance=0.01)
        assert_allclose(angle, 0.0, atol=1e-6)


# =============================================================================
# Step 1.2: Ray Intersection Tests
# =============================================================================

class TestIntersectRaySegment:
    """Tests for intersect_ray_segment() function."""
    
    def test_intersect_ray_horizontal_segment(self):
        """Ray hits horizontal edge."""
        # Ray pointing up (90°) hitting horizontal segment at y=2
        segment_start = np.array([-1.0, 2.0], dtype=np.float32)
        segment_end = np.array([1.0, 2.0], dtype=np.float32)
        
        result = intersect_ray_segment(np.pi / 2, segment_start, segment_end, 10.0)
        
        assert result is not None
        assert_allclose(result, 2.0, atol=1e-6)
    
    def test_intersect_ray_vertical_segment(self):
        """Ray hits vertical edge."""
        # Ray pointing right (0°) hitting vertical segment at x=3
        segment_start = np.array([3.0, -1.0], dtype=np.float32)
        segment_end = np.array([3.0, 1.0], dtype=np.float32)
        
        result = intersect_ray_segment(0.0, segment_start, segment_end, 10.0)
        
        assert result is not None
        assert_allclose(result, 3.0, atol=1e-6)
    
    def test_intersect_ray_diagonal_segment(self):
        """General case with diagonal segment."""
        # Ray at 45° hitting a diagonal segment
        segment_start = np.array([2.0, 0.0], dtype=np.float32)
        segment_end = np.array([0.0, 2.0], dtype=np.float32)
        
        result = intersect_ray_segment(np.pi / 4, segment_start, segment_end, 10.0)
        
        assert result is not None
        # Ray at 45° hits the line x+y=2 at point (1, 1), distance = √2
        assert_allclose(result, np.sqrt(2), atol=1e-6)
    
    def test_intersect_ray_no_intersection(self):
        """Ray misses segment."""
        # Ray pointing right, segment is above and to the left
        segment_start = np.array([-2.0, 1.0], dtype=np.float32)
        segment_end = np.array([-1.0, 2.0], dtype=np.float32)
        
        result = intersect_ray_segment(0.0, segment_start, segment_end, 10.0)
        
        assert result is None
    
    def test_intersect_ray_behind_origin(self):
        """Intersection at negative r (behind ray origin)."""
        # Ray pointing right, segment is behind (at negative x)
        segment_start = np.array([-3.0, -1.0], dtype=np.float32)
        segment_end = np.array([-3.0, 1.0], dtype=np.float32)
        
        result = intersect_ray_segment(0.0, segment_start, segment_end, 10.0)
        
        assert result is None
    
    def test_intersect_ray_beyond_max_range(self):
        """Intersection too far (beyond max_range)."""
        # Ray pointing up, segment at y=5, but max_range=3
        segment_start = np.array([-1.0, 5.0], dtype=np.float32)
        segment_end = np.array([1.0, 5.0], dtype=np.float32)
        
        result = intersect_ray_segment(np.pi / 2, segment_start, segment_end, 3.0)
        
        assert result is None
    
    def test_intersect_ray_parallel(self):
        """Ray parallel to segment (no intersection)."""
        # Ray pointing right, horizontal segment not on ray path
        segment_start = np.array([1.0, 1.0], dtype=np.float32)
        segment_end = np.array([3.0, 1.0], dtype=np.float32)
        
        result = intersect_ray_segment(0.0, segment_start, segment_end, 10.0)
        
        assert result is None
    
    def test_intersect_ray_at_segment_endpoint(self):
        """Ray hits exactly at segment endpoint."""
        # Ray pointing at 45° towards point (1, 1)
        segment_start = np.array([1.0, 1.0], dtype=np.float32)
        segment_end = np.array([2.0, 0.0], dtype=np.float32)
        
        result = intersect_ray_segment(np.pi / 4, segment_start, segment_end, 10.0)
        
        assert result is not None
        assert_allclose(result, np.sqrt(2), atol=1e-6)
    
    def test_intersect_ray_grazing(self):
        """Ray just barely hits segment edge."""
        # Segment from (1, -0.001) to (1, 0.001), ray at angle 0
        segment_start = np.array([1.0, -0.001], dtype=np.float32)
        segment_end = np.array([1.0, 0.001], dtype=np.float32)
        
        result = intersect_ray_segment(0.0, segment_start, segment_end, 10.0)
        
        assert result is not None
        assert_allclose(result, 1.0, atol=1e-3)


class TestHandleAngleDiscontinuity:
    """Tests for handle_angle_discontinuity() function."""
    
    def test_handle_angle_discontinuity_no_wrap(self):
        """Arc doesn't cross ±π (alpha_min < alpha_max)."""
        angles = np.array([0.0, 0.5, 1.0, -0.5], dtype=np.float32)
        alpha_min = -np.pi / 2
        alpha_max = np.pi / 2
        
        result = handle_angle_discontinuity(angles, alpha_min, alpha_max)
        
        # No remapping needed, should be unchanged
        assert_array_almost_equal(result, angles)
    
    def test_handle_angle_discontinuity_with_wrap(self):
        """Arc crosses ±π boundary (alpha_min > alpha_max)."""
        # Arc from 3π/4 (135°) to -3π/4 (-135°) going through π
        angles = np.array([
            2.5,      # Near π, should stay
            -2.5,     # Near -π, should get 2π added
            3.0,      # Very close to π, should stay
            -3.0,     # Very close to -π, should get 2π added
        ], dtype=np.float32)
        
        alpha_min = 3 * np.pi / 4   # 135°
        alpha_max = -3 * np.pi / 4  # -135°
        
        result = handle_angle_discontinuity(angles, alpha_min, alpha_max)
        
        # Angles below alpha_max (-135°) should have 2π added
        # -2.5 is below -2.356 (-3π/4), so should get 2π added
        # -3.0 is below -2.356, so should get 2π added
        assert_allclose(result[0], 2.5, atol=1e-6)  # Unchanged
        assert_allclose(result[1], -2.5 + 2*np.pi, atol=1e-6)  # Remapped
        assert_allclose(result[2], 3.0, atol=1e-6)  # Unchanged
        assert_allclose(result[3], -3.0 + 2*np.pi, atol=1e-6)  # Remapped
    
    def test_handle_angle_discontinuity_remapping(self):
        """Verify 2π addition creates continuous range."""
        # Arc from 170° to -170° (spanning 20° across ±180°)
        alpha_min = np.deg2rad(170)   # ~2.967 rad
        alpha_max = np.deg2rad(-170)  # ~-2.967 rad
        
        # Angles in the arc: 175°, 180°, -175°
        angles = np.array([
            np.deg2rad(175),   # Should stay
            np.deg2rad(180),   # At boundary, depends on exact value
            np.deg2rad(-175),  # Should be remapped
        ], dtype=np.float32)
        
        result = handle_angle_discontinuity(angles, alpha_min, alpha_max)
        
        # After remapping, -175° should become 185° (in radians)
        # Check that the remapped angle is now > alpha_min
        assert result[2] > alpha_min  # -175° + 360° = 185° > 170°
    
    def test_handle_angle_discontinuity_empty_array(self):
        """Handle empty input array."""
        angles = np.array([], dtype=np.float32)
        result = handle_angle_discontinuity(angles, 0.0, np.pi)
        assert len(result) == 0
    
    def test_handle_angle_discontinuity_preserves_input(self):
        """Original array should not be modified."""
        original = np.array([0.5, -0.5, 1.0], dtype=np.float32)
        angles = original.copy()
        
        # Case with wrapping
        handle_angle_discontinuity(angles, np.pi/2, -np.pi/2)
        
        # Original should be unchanged
        assert_array_almost_equal(original, np.array([0.5, -0.5, 1.0]))


# =============================================================================
# Edge Cases and Integration Tests
# =============================================================================

class TestGeometryEdgeCases:
    """Edge cases and integration tests for geometry functions."""
    
    def test_polar_roundtrip(self):
        """Converting to polar and back should preserve coordinates."""
        original = np.array([
            [3.0, 4.0],
            [-2.0, 1.0],
            [0.5, -0.5],
        ], dtype=np.float32)
        
        radii, angles = to_polar(original)
        
        # Convert back
        reconstructed = np.column_stack([
            radii * np.cos(angles),
            radii * np.sin(angles)
        ])
        
        assert_array_almost_equal(reconstructed, original, decimal=5)
    
    def test_viewer_frame_then_polar(self):
        """Test workflow: translate to viewer frame then convert to polar."""
        viewer_pos = np.array([100.0, 100.0], dtype=np.float32)
        world_point = np.array([103.0, 104.0], dtype=np.float32)
        
        # Translate
        local = to_viewer_frame(world_point, viewer_pos)
        assert_array_almost_equal(local, np.array([3.0, 4.0]))
        
        # To polar
        r, theta = to_polar(local)
        assert_allclose(float(r), 5.0, atol=1e-6)
        assert_allclose(float(theta), np.arctan2(4.0, 3.0), atol=1e-6)
    
    def test_direction_validation_with_viewing_angles(self):
        """Test common viewing direction scenarios."""
        # Looking straight up (typical in image coordinates)
        up = np.array([0.0, 1.0], dtype=np.float32)
        assert_allclose(validate_and_get_direction_angle(up), np.pi / 2)
        
        # Looking at typical diagonal (-37°, 92°) normalized
        diag = np.array([-0.37, 0.929], dtype=np.float32)
        # Normalize properly
        diag = (diag / np.linalg.norm(diag)).astype(np.float32)
        angle = validate_and_get_direction_angle(diag)
        assert angle > np.pi / 2  # Should be in second quadrant
        assert angle < np.pi
