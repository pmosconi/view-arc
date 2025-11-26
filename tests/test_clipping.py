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
    clip_polygon_circle,
    clip_polygon_to_wedge,
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


# =============================================================================
# Step 2.2: Circle and Wedge Clipping Tests
# =============================================================================

class TestClipPolygonCircle:
    """Tests for clip_polygon_circle() function."""
    
    def test_clip_circle_fully_inside(self):
        """Polygon entirely within the circle - no clipping needed."""
        # Small square inside a circle of radius 5
        square = np.array([
            [1.0, 1.0],
            [2.0, 1.0],
            [2.0, 2.0],
            [1.0, 2.0],
        ], dtype=np.float32)
        
        result = clip_polygon_circle(square, radius=5.0)
        
        # All vertices should be preserved
        assert result.shape[0] == 4
        assert_array_almost_equal(result, square)
    
    def test_clip_circle_fully_outside(self):
        """Polygon entirely outside the circle - complete removal."""
        # Square far from origin
        square = np.array([
            [10.0, 10.0],
            [12.0, 10.0],
            [12.0, 12.0],
            [10.0, 12.0],
        ], dtype=np.float32)
        
        result = clip_polygon_circle(square, radius=5.0)
        
        # All vertices should be clipped
        assert result.shape[0] == 0
    
    def test_clip_circle_partial(self):
        """Polygon partially inside - some vertices beyond radius."""
        # Square centered at origin extending beyond radius
        square = np.array([
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
        ], dtype=np.float32)
        
        # Small radius that cuts through the square
        result = clip_polygon_circle(square, radius=1.2)
        
        # Result should have more vertices due to intersections
        assert result.shape[0] >= 4
        
        # All resulting vertices should be within or on the circle
        distances = np.sqrt(result[:, 0]**2 + result[:, 1]**2)
        assert np.all(distances <= 1.2 + 1e-5)
    
    def test_clip_circle_edge_intersections(self):
        """Verify intersection points are computed accurately."""
        # Triangle with vertices at different distances from origin
        triangle = np.array([
            [0.0, 0.0],   # Inside (at origin)
            [3.0, 0.0],   # Outside (distance 3)
            [0.0, 3.0],   # Outside (distance 3)
        ], dtype=np.float32)
        
        result = clip_polygon_circle(triangle, radius=2.0)
        
        # Origin should be preserved
        assert any(np.allclose(v, [0.0, 0.0], atol=1e-5) for v in result)
        
        # Should have intersection points at distance 2
        distances = np.sqrt(result[:, 0]**2 + result[:, 1]**2)
        
        # Check that some points are exactly on the circle (within tolerance)
        on_circle = np.abs(distances - 2.0) < 1e-5
        assert np.sum(on_circle) >= 2  # At least 2 intersection points
    
    def test_clip_circle_degenerate_to_point(self):
        """Handle edge case where polygon degenerates to a point."""
        # Very small triangle at origin
        tiny_triangle = np.array([
            [0.0, 0.0],
            [0.001, 0.0],
            [0.0, 0.001],
        ], dtype=np.float32)
        
        # Clip with very small radius - should still work
        result = clip_polygon_circle(tiny_triangle, radius=0.01)
        
        # Should preserve the tiny triangle
        assert result.shape[0] >= 3
    
    def test_clip_circle_vertex_on_boundary(self):
        """Test when a vertex lies exactly on the circle boundary."""
        # Triangle with one vertex exactly on the circle
        triangle = np.array([
            [0.0, 0.0],   # Inside
            [5.0, 0.0],   # On boundary (radius=5)
            [0.0, 3.0],   # Inside
        ], dtype=np.float32)
        
        result = clip_polygon_circle(triangle, radius=5.0)
        
        # All vertices should be preserved
        assert result.shape[0] == 3
        assert any(np.allclose(v, [5.0, 0.0], atol=1e-5) for v in result)
    
    def test_clip_circle_empty_input(self):
        """Empty polygon input should return empty output."""
        empty = np.array([], dtype=np.float32).reshape(0, 2)
        
        result = clip_polygon_circle(empty, radius=5.0)
        
        assert result.shape == (0, 2)
    
    def test_clip_circle_edge_passes_through(self):
        """Test when an edge passes through the circle but both endpoints are outside."""
        # Rectangle with endpoints outside but edge crossing through origin area
        rectangle = np.array([
            [-5.0, -1.0],
            [5.0, -1.0],
            [5.0, 1.0],
            [-5.0, 1.0],
        ], dtype=np.float32)
        
        result = clip_polygon_circle(rectangle, radius=2.0)
        
        # Should have intersection points
        assert result.shape[0] >= 4
        
        # All points should be within the circle
        distances = np.sqrt(result[:, 0]**2 + result[:, 1]**2)
        assert np.all(distances <= 2.0 + 1e-5)

    def test_clip_circle_polygon_encloses_circle_returns_sampled_arc(self):
        """Polygons that fully contain the circle should return an arc-sampled disk."""
        square = np.array([
            [-5.0, -5.0],
            [5.0, -5.0],
            [5.0, 5.0],
            [-5.0, 5.0],
        ], dtype=np.float32)

        radius = 1.5
        result = clip_polygon_circle(square, radius=radius)

        # Expect many vertices approximating the circle
        assert result.shape[0] >= 16

        distances = np.sqrt(result[:, 0] ** 2 + result[:, 1] ** 2)
        assert np.allclose(distances, radius, atol=5e-3)

    def test_clip_circle_retains_arc_when_polygon_outside(self):
        """Circle boundary segments should be preserved as sampled arcs."""
        rectangle = np.array([
            [-1.0, 0.0],
            [1.0, 0.0],
            [1.0, 3.0],
            [-1.0, 3.0],
        ], dtype=np.float32)

        result = clip_polygon_circle(rectangle, radius=1.0)

        # The intersection should include points along the circular arc (y > 0)
        assert result.shape[0] >= 6
        assert np.any(result[:, 1] > 0.5)

        # Ensure all points lie within the circle tolerance
        distances = np.sqrt(result[:, 0] ** 2 + result[:, 1] ** 2)
        assert np.all(distances <= 1.0 + 1e-5)


class TestClipPolygonToWedge:
    """Tests for clip_polygon_to_wedge() function."""
    
    def test_clip_wedge_full_pipeline(self):
        """Integration test of the 3-stage clipping pipeline."""
        # Square in first quadrant
        square = np.array([
            [1.0, 1.0],
            [3.0, 1.0],
            [3.0, 3.0],
            [1.0, 3.0],
        ], dtype=np.float32)
        
        # Wedge from 0° to 90° (first quadrant) with radius 5
        alpha_min = 0.0
        alpha_max = np.pi / 2
        max_range = 5.0
        
        result = clip_polygon_to_wedge(square, alpha_min, alpha_max, max_range)
        
        # Square is entirely in the wedge, should be preserved
        assert result is not None
        assert result.shape[0] == 4
        assert_array_almost_equal(result, square)
    
    def test_clip_wedge_narrow_arc(self):
        """Test with narrow 30° FOV."""
        # Square spanning from 0° to 45° approximately
        square = np.array([
            [2.0, 0.5],
            [3.0, 0.5],
            [3.0, 1.5],
            [2.0, 1.5],
        ], dtype=np.float32)
        
        # Narrow wedge: 15° to 45° (30° FOV)
        alpha_min = np.radians(15)
        alpha_max = np.radians(45)
        max_range = 5.0
        
        result = clip_polygon_to_wedge(square, alpha_min, alpha_max, max_range)
        
        # Should produce a valid clipped polygon
        assert result is not None
        assert result.shape[0] >= 3
        
        # All points should be in the wedge
        distances = np.sqrt(result[:, 0]**2 + result[:, 1]**2)
        angles = np.arctan2(result[:, 1], result[:, 0])
        
        assert np.all(distances <= max_range + 1e-5)
        assert np.all(angles >= alpha_min - 1e-5)
        assert np.all(angles <= alpha_max + 1e-5)
    
    def test_clip_wedge_wide_arc(self):
        """Test with wide 120° FOV."""
        # Square in first quadrant
        square = np.array([
            [1.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [1.0, 2.0],
        ], dtype=np.float32)
        
        # Wide wedge: -30° to 90° (120° FOV)
        alpha_min = np.radians(-30)
        alpha_max = np.radians(90)
        max_range = 5.0
        
        result = clip_polygon_to_wedge(square, alpha_min, alpha_max, max_range)
        
        # Square should be mostly preserved (it's within the wedge)
        assert result is not None
        assert result.shape[0] >= 3
    
    def test_clip_wedge_returns_none_if_outside(self):
        """Complete rejection when polygon is outside the wedge."""
        # Square in third quadrant (negative x and y)
        square = np.array([
            [-3.0, -3.0],
            [-2.0, -3.0],
            [-2.0, -2.0],
            [-3.0, -2.0],
        ], dtype=np.float32)
        
        # Wedge in first quadrant: 0° to 90°
        alpha_min = 0.0
        alpha_max = np.pi / 2
        max_range = 5.0
        
        result = clip_polygon_to_wedge(square, alpha_min, alpha_max, max_range)
        
        # Should be completely clipped away
        assert result is None
    
    def test_clip_wedge_validates_result(self):
        """Ensures ≥3 vertices for valid polygon."""
        # Thin sliver that might clip to < 3 vertices
        sliver = np.array([
            [0.1, 0.0],
            [0.2, 0.0],
            [0.15, 0.05],
        ], dtype=np.float32)
        
        # Wedge that mostly excludes this sliver
        alpha_min = np.pi / 4
        alpha_max = np.pi / 2
        max_range = 5.0
        
        result = clip_polygon_to_wedge(sliver, alpha_min, alpha_max, max_range)
        
        # Result should either be None or have at least 3 vertices
        if result is not None:
            assert result.shape[0] >= 3
    
    def test_clip_wedge_with_circle_clipping(self):
        """Test that circle clipping works in the pipeline."""
        # Large square that extends beyond max_range
        square = np.array([
            [1.0, 1.0],
            [10.0, 1.0],
            [10.0, 10.0],
            [1.0, 10.0],
        ], dtype=np.float32)
        
        # Wedge with small radius
        alpha_min = 0.0
        alpha_max = np.pi / 2
        max_range = 5.0
        
        result = clip_polygon_to_wedge(square, alpha_min, alpha_max, max_range)
        
        assert result is not None
        
        # All points should be within max_range
        distances = np.sqrt(result[:, 0]**2 + result[:, 1]**2)
        assert np.all(distances <= max_range + 1e-5)
    
    def test_clip_wedge_invalid_polygon_input(self):
        """Invalid input polygons should return None."""
        # Less than 3 vertices
        line = np.array([
            [1.0, 1.0],
            [2.0, 2.0],
        ], dtype=np.float32)
        
        result = clip_polygon_to_wedge(line, 0.0, np.pi/2, 5.0)
        
        assert result is None
    
    def test_clip_wedge_empty_polygon_input(self):
        """Empty input polygon should return None."""
        empty = np.array([], dtype=np.float32).reshape(0, 2)
        
        result = clip_polygon_to_wedge(empty, 0.0, np.pi/2, 5.0)
        
        assert result is None
    
    def test_clip_wedge_preserves_vertices_on_boundary(self):
        """Vertices exactly on wedge boundaries should be preserved."""
        # Triangle with vertices on wedge boundaries
        triangle = np.array([
            [1.0, 0.0],   # On alpha_min boundary (0°)
            [0.0, 1.0],   # On alpha_max boundary (90°)
            [0.5, 0.5],   # Inside wedge
        ], dtype=np.float32)
        
        alpha_min = 0.0
        alpha_max = np.pi / 2
        max_range = 5.0
        
        result = clip_polygon_to_wedge(triangle, alpha_min, alpha_max, max_range)
        
        assert result is not None
        assert result.shape[0] == 3
        
        # Check all original vertices are present
        assert any(np.allclose(v, [1.0, 0.0], atol=1e-5) for v in result)
        assert any(np.allclose(v, [0.0, 1.0], atol=1e-5) for v in result)
        assert any(np.allclose(v, [0.5, 0.5], atol=1e-5) for v in result)
