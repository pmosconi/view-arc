"""
Tests for polygon clipping operations - Phase 2 of implementation.

Step 2.1: Half-Plane Clipping
- test_is_valid_polygon_*
- test_compute_bounding_box_*
- test_clip_halfplane_*

Step 2.2: Circle and Wedge Clipping (to be added)
- test_clip_circle_*
- test_clip_wedge_*
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_almost_equal

from view_arc.clipping import (
    is_valid_polygon,
    compute_bounding_box,
    clip_polygon_halfplane,
)


# =============================================================================
# Step 2.1: Half-Plane Clipping Tests
# =============================================================================

class TestIsValidPolygon:
    """Tests for is_valid_polygon() function."""
    
    def test_is_valid_polygon_sufficient_vertices_triangle(self):
        """Triangle with exactly 3 vertices is valid."""
        triangle = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0],
        ], dtype=np.float32)
        assert is_valid_polygon(triangle) is True
    
    def test_is_valid_polygon_sufficient_vertices_quadrilateral(self):
        """Quadrilateral with 4 vertices is valid."""
        quad = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ], dtype=np.float32)
        assert is_valid_polygon(quad) is True
    
    def test_is_valid_polygon_sufficient_vertices_pentagon(self):
        """Pentagon with 5 vertices is valid."""
        pentagon = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.5, 0.8],
            [0.5, 1.5],
            [-0.5, 0.8],
        ], dtype=np.float32)
        assert is_valid_polygon(pentagon) is True
    
    def test_is_valid_polygon_insufficient_vertices_zero(self):
        """Empty polygon (0 vertices) is invalid."""
        empty = np.array([], dtype=np.float32).reshape(0, 2)
        assert is_valid_polygon(empty) is False
    
    def test_is_valid_polygon_insufficient_vertices_one(self):
        """Single point (1 vertex) is invalid."""
        point = np.array([[0.0, 0.0]], dtype=np.float32)
        assert is_valid_polygon(point) is False
    
    def test_is_valid_polygon_insufficient_vertices_two(self):
        """Line segment (2 vertices) is invalid."""
        line = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
        ], dtype=np.float32)
        assert is_valid_polygon(line) is False


class TestComputeBoundingBox:
    """Tests for compute_bounding_box() function."""
    
    def test_compute_bounding_box_square(self):
        """Axis-aligned unit square at origin."""
        square = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ], dtype=np.float32)
        
        min_pt, max_pt = compute_bounding_box(square)
        
        assert_array_almost_equal(min_pt, np.array([0.0, 0.0]))
        assert_array_almost_equal(max_pt, np.array([1.0, 1.0]))
    
    def test_compute_bounding_box_triangle(self):
        """Non-axis-aligned triangle."""
        triangle = np.array([
            [1.0, 2.0],
            [4.0, 1.0],
            [3.0, 5.0],
        ], dtype=np.float32)
        
        min_pt, max_pt = compute_bounding_box(triangle)
        
        assert_array_almost_equal(min_pt, np.array([1.0, 1.0]))
        assert_array_almost_equal(max_pt, np.array([4.0, 5.0]))
    
    def test_compute_bounding_box_negative_coords(self):
        """Polygon spanning negative and positive coordinates."""
        polygon = np.array([
            [-2.0, -1.0],
            [3.0, -2.0],
            [1.0, 4.0],
            [-1.0, 2.0],
        ], dtype=np.float32)
        
        min_pt, max_pt = compute_bounding_box(polygon)
        
        assert_array_almost_equal(min_pt, np.array([-2.0, -2.0]))
        assert_array_almost_equal(max_pt, np.array([3.0, 4.0]))
    
    def test_compute_bounding_box_single_point(self):
        """Degenerate case: single point polygon."""
        point = np.array([[5.0, 3.0]], dtype=np.float32)
        
        min_pt, max_pt = compute_bounding_box(point)
        
        assert_array_almost_equal(min_pt, np.array([5.0, 3.0]))
        assert_array_almost_equal(max_pt, np.array([5.0, 3.0]))
    
    def test_compute_bounding_box_returns_float32(self):
        """Verify output arrays are float32."""
        triangle = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0],
        ], dtype=np.float32)
        
        min_pt, max_pt = compute_bounding_box(triangle)
        
        assert min_pt.dtype == np.float32
        assert max_pt.dtype == np.float32

    def test_compute_bounding_box_empty_polygon_error(self):
        """Empty polygons should raise a clear error instead of np.min failure."""
        empty = np.array([], dtype=np.float32).reshape(0, 2)

        with pytest.raises(ValueError, match="at least one vertex"):
            compute_bounding_box(empty)


class TestClipPolygonHalfplane:
    """Tests for clip_polygon_halfplane() function."""
    
    def test_clip_halfplane_fully_inside(self):
        """Polygon entirely on the kept side - no clipping needed."""
        # Square in the first quadrant
        square = np.array([
            [1.0, 1.0],
            [2.0, 1.0],
            [2.0, 2.0],
            [1.0, 2.0],
        ], dtype=np.float32)
        
        # Clip with ray along positive x-axis, keep left (upper half-plane)
        result = clip_polygon_halfplane(square, plane_angle=0.0, keep_left=True)
        
        # All vertices should be preserved
        assert result.shape[0] == 4
        assert_array_almost_equal(result, square)
    
    def test_clip_halfplane_fully_outside(self):
        """Polygon entirely on the clipped side - complete removal."""
        # Square in the third quadrant (negative x and y)
        square = np.array([
            [-2.0, -2.0],
            [-1.0, -2.0],
            [-1.0, -1.0],
            [-2.0, -1.0],
        ], dtype=np.float32)
        
        # Clip with ray along positive x-axis, keep left (upper half-plane y > 0)
        result = clip_polygon_halfplane(square, plane_angle=0.0, keep_left=True)
        
        # All vertices should be clipped
        assert result.shape[0] == 0
    
    def test_clip_halfplane_partial(self):
        """Polygon partially inside - some vertices clipped."""
        # Square centered at origin: from (-1,-1) to (1,1)
        square = np.array([
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
        ], dtype=np.float32)
        
        # Clip with ray along positive x-axis, keep left (upper half-plane y >= 0)
        result = clip_polygon_halfplane(square, plane_angle=0.0, keep_left=True)
        
        # Should clip to upper half: 4 vertices forming rectangle
        assert result.shape[0] == 4
        
        # Verify the result contains the expected points
        # Top vertices should be preserved
        assert any(np.allclose(v, [-1.0, 1.0]) for v in result)
        assert any(np.allclose(v, [1.0, 1.0]) for v in result)
        # Bottom should be clipped to y=0
        assert any(np.allclose(v, [-1.0, 0.0]) for v in result)
        assert any(np.allclose(v, [1.0, 0.0]) for v in result)
    
    def test_clip_halfplane_edge_intersection(self):
        """Verify intersection points are computed correctly."""
        # Triangle with one vertex above y=0 and two below
        triangle = np.array([
            [0.0, 2.0],   # Above
            [-2.0, -1.0], # Below
            [2.0, -1.0],  # Below
        ], dtype=np.float32)
        
        # Clip with ray along positive x-axis, keep left (y >= 0)
        result = clip_polygon_halfplane(triangle, plane_angle=0.0, keep_left=True)
        
        # Result should be a triangle with apex at (0, 2) and two 
        # intersection points on the x-axis
        assert result.shape[0] == 3
        
        # One vertex should be the original apex
        assert any(np.allclose(v, [0.0, 2.0]) for v in result)
        
        # The other two should be on the x-axis (y=0)
        y_values = result[:, 1]
        count_on_axis = np.sum(np.abs(y_values) < 1e-6)
        assert count_on_axis == 2
    
    def test_clip_halfplane_ccw_preservation(self):
        """Verify that CCW winding order is maintained."""
        # CCW square
        square = np.array([
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
        ], dtype=np.float32)
        
        # Clip with ray along positive x-axis, keep left
        result = clip_polygon_halfplane(square, plane_angle=0.0, keep_left=True)
        
        # Compute signed area to verify CCW (positive area)
        def signed_area(poly):
            n = len(poly)
            area = 0.0
            for i in range(n):
                j = (i + 1) % n
                area += poly[i, 0] * poly[j, 1]
                area -= poly[j, 0] * poly[i, 1]
            return area / 2.0
        
        assert signed_area(result) > 0, "Winding order should remain CCW (positive area)"
    
    def test_clip_halfplane_diagonal_plane(self):
        """Clip with a diagonal half-plane (45 degrees)."""
        # Square from (0,0) to (2,2)
        square = np.array([
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
        ], dtype=np.float32)
        
        # Clip with ray at 45 degrees (π/4), keep left
        # The line y = x divides the square; keep points where y >= x
        result = clip_polygon_halfplane(square, plane_angle=np.pi/4, keep_left=True)
        
        # The kept region is a triangle: (0,0), (2,2), (0,2)
        # Due to boundary handling, we may have duplicates, but key vertices should exist
        # Vertices (0,0) and (2,2) are on the boundary, (0,2) is inside
        
        # Verify key vertices are present
        assert any(np.allclose(v, [0.0, 0.0], atol=1e-6) for v in result)
        assert any(np.allclose(v, [0.0, 2.0], atol=1e-6) for v in result)
        assert any(np.allclose(v, [2.0, 2.0], atol=1e-6) for v in result)
        
        # Verify no vertices from the clipped region (2, 0) are present
        assert not any(np.allclose(v, [2.0, 0.0], atol=1e-6) for v in result)
    
    def test_clip_halfplane_keep_right(self):
        """Test clipping with keep_left=False (keep right side)."""
        # Square centered at origin
        square = np.array([
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
        ], dtype=np.float32)
        
        # Clip with ray along positive x-axis, keep right (lower half-plane y <= 0)
        result = clip_polygon_halfplane(square, plane_angle=0.0, keep_left=False)
        
        # Should clip to lower half: 4 vertices forming rectangle
        assert result.shape[0] == 4
        
        # Bottom vertices should be preserved
        assert any(np.allclose(v, [-1.0, -1.0]) for v in result)
        assert any(np.allclose(v, [1.0, -1.0]) for v in result)
        # Top should be clipped to y=0
        assert any(np.allclose(v, [-1.0, 0.0]) for v in result)
        assert any(np.allclose(v, [1.0, 0.0]) for v in result)
    
    def test_clip_halfplane_empty_input(self):
        """Empty polygon input should return empty output."""
        empty = np.array([], dtype=np.float32).reshape(0, 2)
        
        result = clip_polygon_halfplane(empty, plane_angle=0.0, keep_left=True)
        
        assert result.shape == (0, 2)
    
    def test_clip_halfplane_negative_angle(self):
        """Test with negative plane angle."""
        # Square in first quadrant
        square = np.array([
            [1.0, 0.5],
            [2.0, 0.5],
            [2.0, 1.5],
            [1.0, 1.5],
        ], dtype=np.float32)
        
        # Clip with ray at -π/4 (pointing into fourth quadrant), keep left
        result = clip_polygon_halfplane(square, plane_angle=-np.pi/4, keep_left=True)
        
        # Square is entirely in the first quadrant, above the line y = -x
        # which passes through origin, so all vertices should be kept
        assert result.shape[0] == 4
    
    def test_clip_halfplane_vertex_on_boundary(self):
        """Test when a vertex lies exactly on the clipping boundary."""
        # Triangle with one vertex on the x-axis
        triangle = np.array([
            [1.0, 0.0],   # On boundary
            [2.0, 1.0],   # Above
            [0.0, 1.0],   # Above
        ], dtype=np.float32)
        
        # Clip with ray along positive x-axis, keep left (y >= 0)
        result = clip_polygon_halfplane(triangle, plane_angle=0.0, keep_left=True)
        
        # All vertices should be kept (boundary is included with >= 0)
        assert result.shape[0] == 3
        assert any(np.allclose(v, [1.0, 0.0]) for v in result)

    def test_clip_halfplane_preserves_boundary_edge_with_tolerance(self):
        """Entire edges infinitesimally below boundary remain after tolerance handling."""
        eps = np.float32(5e-7)
        polygon = np.array([
            [-1.5, -eps],
            [1.5, -eps],
            [1.5, 1.0],
            [-1.5, 1.0],
        ], dtype=np.float32)

        result = clip_polygon_halfplane(polygon, plane_angle=0.0, keep_left=True)

        # Polygon should be preserved (no extra vertices introduced)
        assert result.shape[0] == 4
        # Bottom edge should still be present within the tolerance band
        bottom_vertices = result[result[:, 1] < 0.0]
        assert bottom_vertices.shape[0] == 2
        assert np.allclose(bottom_vertices[:, 1], -eps, atol=1e-6)
