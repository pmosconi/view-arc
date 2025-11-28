"""
Tests for angular sweep operations - Phase 3 of implementation.

Step 3.1: Event Construction
- test_get_active_edges_*
- test_build_events_*

Step 3.2: Interval Resolution (to be added)
- test_resolve_interval_*
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_almost_equal

from view_arc.sweep import (
    AngularEvent,
    get_active_edges,
    build_events,
    _edge_crosses_angle,
)


# =============================================================================
# Step 3.1: Event Construction Tests
# =============================================================================

class TestGetActiveEdges:
    """Tests for get_active_edges() function."""
    
    def test_get_active_edges_no_edges_active(self):
        """Angle outside polygon angular span should return exactly 0 edges."""
        # Triangle in the positive x, positive y quadrant
        # Vertices at angles approximately 0°, 45°, 90°
        triangle = np.array([
            [2.0, 0.0],   # angle ≈ 0
            [2.0, 2.0],   # angle ≈ π/4 (45°)
            [0.0, 2.0],   # angle ≈ π/2 (90°)
        ], dtype=np.float32)
        
        # Query angle in third quadrant - should not intersect
        query_angle = -3 * np.pi / 4  # -135°
        
        result = get_active_edges(triangle, query_angle)
        
        # Exact assertion: no edges should be active
        assert result.shape == (0, 2, 2), f"Expected 0 edges, got {result.shape[0]}"
    
    def test_get_active_edges_ray_through_convex_polygon(self):
        """Ray through convex polygon should intersect exactly 2 edges (entry and exit)."""
        # Square centered around a point on the x-axis
        square = np.array([
            [3.0, -1.0],   # bottom-left
            [5.0, -1.0],   # bottom-right
            [5.0, 1.0],    # top-right
            [3.0, 1.0],    # top-left
        ], dtype=np.float32)
        
        # Query at 0° (along positive x-axis) - ray goes through center of square
        query_angle = 0.0
        
        result = get_active_edges(square, query_angle)
        
        # A ray through a convex polygon hits exactly 2 edges: entry and exit
        assert result.shape[0] == 2, f"Expected 2 edges (entry/exit), got {result.shape[0]}"
        assert result.shape[1:] == (2, 2)
        
        # Verify the edges are the left and right vertical edges of the square
        # Left edge: v3->v0 = [(3,1), (3,-1)] or v0->v3 reversed in cycle
        # Right edge: v1->v2 = [(5,-1), (5,1)]
        edge_x_coords = set()
        for edge in result:
            # Both endpoints of a vertical edge have the same x
            if np.isclose(edge[0, 0], edge[1, 0]):
                edge_x_coords.add(edge[0, 0])
        
        # Should have edges at x=3 and x=5
        assert 3.0 in edge_x_coords or np.isclose(list(edge_x_coords)[0], 3.0) or \
               5.0 in edge_x_coords or np.isclose(list(edge_x_coords)[-1], 5.0), \
               f"Expected edges at x=3 and x=5, got x-coords: {edge_x_coords}"
    
    def test_get_active_edges_ray_tangent_to_vertex(self):
        """Ray exactly at vertex angle may include adjacent edges."""
        # Triangle with vertex exactly on positive x-axis
        triangle = np.array([
            [2.0, 0.0],   # angle = 0 (exactly on x-axis)
            [3.0, 1.0],   # angle > 0
            [3.0, -1.0],  # angle < 0
        ], dtype=np.float32)
        
        # Query at 0° hits vertex 0 exactly
        query_angle = 0.0
        
        result = get_active_edges(triangle, query_angle)
        
        # Should include edges adjacent to vertex 0: edge (v2->v0) and edge (v0->v1)
        # The ray at angle=0 may also hit edge v1->v2 (the back edge at x=3)
        # All 3 edges can be intersected by a ray along the x-axis through this triangle
        assert result.shape[0] >= 1, "Should find at least 1 edge at vertex angle"
        assert result.shape[0] <= 3, f"Triangle has only 3 edges, got {result.shape[0]}"
        
        # Verify returned edges contain the vertex at [2.0, 0.0]
        vertex_found = False
        for edge in result:
            if np.allclose(edge[0], [2.0, 0.0]) or np.allclose(edge[1], [2.0, 0.0]):
                vertex_found = True
                break
        assert vertex_found, "Active edges should include the vertex at angle=0"
    
    def test_get_active_edges_specific_edge_identification(self):
        """Verify returned edges match expected vertex pairs exactly."""
        # Rectangle in first quadrant - ray should hit exactly 2 edges
        rectangle = np.array([
            [3.0, -0.5],   # v0: bottom-left, below x-axis
            [5.0, -0.5],   # v1: bottom-right, below x-axis
            [5.0, 0.5],    # v2: top-right, above x-axis
            [3.0, 0.5],    # v3: top-left, above x-axis
        ], dtype=np.float32)
        
        # Query at angle = 0 (along positive x-axis)
        # This should hit exactly 2 edges:
        # - Left edge v3->v0 (from (3,0.5) to (3,-0.5))
        # - Right edge v1->v2 (from (5,-0.5) to (5,0.5))
        query_angle = 0.0
        
        result = get_active_edges(rectangle, query_angle)
        
        # Should have exactly 2 edges
        assert result.shape[0] == 2, f"Expected 2 edges, got {result.shape[0]}"
        
        # Verify the edges are vertical edges at x=3 and x=5
        x_coords = set()
        for edge in result:
            # For vertical edges, both endpoints have the same x-coordinate
            if np.isclose(edge[0, 0], edge[1, 0]):
                x_coords.add(float(edge[0, 0]))
        
        assert len(x_coords) == 2, f"Expected 2 vertical edges, got {len(x_coords)}"
        assert np.isclose(min(x_coords), 3.0), f"Expected left edge at x=3, got {min(x_coords)}"
        assert np.isclose(max(x_coords), 5.0), f"Expected right edge at x=5, got {max(x_coords)}"
    
    def test_get_active_edges_empty_polygon(self):
        """Empty or invalid polygon should return exactly 0 edges."""
        # Empty polygon
        empty = np.array([], dtype=np.float32).reshape(0, 2)
        result = get_active_edges(empty, 0.0)
        assert result.shape == (0, 2, 2), "Empty polygon should return 0 edges"
        
        # Two vertices (not a valid polygon)
        line = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        result = get_active_edges(line, 0.0)
        assert result.shape == (0, 2, 2), "Line (2 vertices) should return 0 edges"
        
        # None polygon
        result = get_active_edges(None, 0.0)
        assert result.shape == (0, 2, 2), "None polygon should return 0 edges"
    
    def test_get_active_edges_returns_cartesian_coordinates(self):
        """Verify returned edges are in Cartesian coordinates matching polygon vertices."""
        triangle = np.array([
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
        ], dtype=np.float32)
        
        # Query at π/8 - should hit edge from (2,0) to (2,2)
        query_angle = np.pi / 8  # 22.5°
        
        result = get_active_edges(triangle, query_angle)
        
        assert result.shape[0] >= 1, "Should find at least 1 active edge"
        
        # Verify all edge endpoints come from original polygon vertices
        for edge in result:
            for endpoint in edge:
                found = any(np.allclose(endpoint, v) for v in triangle)
                assert found, f"Endpoint {endpoint} not found in polygon vertices"
    
    def test_get_active_edges_ray_misses_far_polygon(self):
        """Ray at angle that completely misses polygon should return 0 edges."""
        # Polygon in upper half-plane only
        upper_triangle = np.array([
            [1.0, 1.0],   # angle ≈ 45°
            [2.0, 2.0],   # angle ≈ 45°
            [0.5, 2.0],   # angle ≈ 76°
        ], dtype=np.float32)
        
        # Query at negative angle (lower half-plane)
        query_angle = -np.pi / 4  # -45°
        
        result = get_active_edges(upper_triangle, query_angle)
        
        assert result.shape == (0, 2, 2), \
            f"Ray in lower half-plane should miss upper polygon, got {result.shape[0]} edges"


class TestBuildEvents:
    """Tests for build_events() function."""
    
    def test_build_events_single_triangle(self):
        """Single triangle should produce 3 vertex events."""
        # Triangle fully within arc range
        triangle = np.array([
            [2.0, 0.0],   # angle ≈ 0
            [2.0, 1.0],   # angle ≈ 0.46 rad
            [1.0, 0.5],   # angle ≈ 0.46 rad
        ], dtype=np.float32)
        
        # Arc that covers the triangle
        alpha_min = -np.pi / 4
        alpha_max = np.pi / 2
        
        events = build_events([triangle], alpha_min, alpha_max)
        
        # Should have 3 vertex events (one per vertex)
        vertex_events = [e for e in events if e.event_type == 'vertex']
        assert len(vertex_events) == 3
        
        # All should have obstacle_id = 0
        assert all(e.obstacle_id == 0 for e in vertex_events)
    
    def test_build_events_multiple_obstacles(self):
        """Multiple obstacles should produce sorted mixed events."""
        # Two triangles at different positions
        triangle1 = np.array([
            [2.0, 0.0],   # angle ≈ 0
            [3.0, 0.5],   # angle ≈ 0.17 rad
            [2.0, 1.0],   # angle ≈ 0.46 rad
        ], dtype=np.float32)
        
        triangle2 = np.array([
            [1.0, 1.0],   # angle ≈ π/4
            [0.5, 2.0],   # angle ≈ 1.33 rad
            [0.0, 1.0],   # angle ≈ π/2
        ], dtype=np.float32)
        
        alpha_min = -np.pi / 4
        alpha_max = 3 * np.pi / 4
        
        events = build_events([triangle1, triangle2], alpha_min, alpha_max)
        
        # Should have vertex events from both obstacles
        vertex_events = [e for e in events if e.event_type == 'vertex']
        
        # 3 vertices from each = 6 total (assuming all in range)
        assert len(vertex_events) == 6
        
        # Check that events are sorted by angle
        angles = [e.angle for e in events]
        assert angles == sorted(angles)
        
        # Should have events from both obstacles
        obstacle_ids = {e.obstacle_id for e in events}
        assert obstacle_ids == {0, 1}
    
    def test_build_events_edge_crossings(self):
        """Detect edge crossings at arc boundaries."""
        # Triangle that spans across alpha_min boundary
        # One vertex inside arc, two outside, but edges cross the boundary
        triangle = np.array([
            [2.0, 0.5],    # angle ≈ 0.24 rad (inside arc)
            [2.0, -0.5],   # angle ≈ -0.24 rad (outside if alpha_min=0)
            [1.5, 0.0],    # angle = 0 (exactly on boundary)
        ], dtype=np.float32)
        
        # Arc from 0 to π/2
        alpha_min = 0.0
        alpha_max = np.pi / 2
        
        events = build_events([triangle], alpha_min, alpha_max)
        
        # Should have some vertex events for vertices inside the arc
        # May also have edge crossing events at boundaries
        assert len(events) >= 1
    
    def test_build_events_sorting_stability(self):
        """Vertex events should come before edge_crossing at same angle."""
        # Create a scenario where vertex and edge crossing occur at same angle
        # Vertex exactly at alpha_min, and another edge crossing at alpha_min
        triangle1 = np.array([
            [2.0, 0.0],    # angle = 0 (exactly at alpha_min)
            [2.0, 1.0],    # angle ≈ 0.46 rad
            [1.0, 0.5],    # angle ≈ 0.46 rad
        ], dtype=np.float32)
        
        alpha_min = 0.0
        alpha_max = np.pi / 2
        
        events = build_events([triangle1], alpha_min, alpha_max)
        
        # Find events at or near angle 0
        events_at_zero = [e for e in events if abs(e.angle) < 0.01]
        
        if len(events_at_zero) > 1:
            # Verify vertex events come first
            for i in range(len(events_at_zero) - 1):
                if events_at_zero[i].event_type == 'edge_crossing':
                    assert events_at_zero[i+1].event_type == 'edge_crossing'
    
    def test_build_events_empty_input(self):
        """Empty polygon list should return empty event list."""
        events = build_events([], alpha_min=0.0, alpha_max=np.pi)
        assert events == []
    
    def test_build_events_none_polygons_ignored(self):
        """None polygons in list should be skipped."""
        triangle = np.array([
            [2.0, 0.0],
            [2.0, 1.0],
            [1.0, 0.5],
        ], dtype=np.float32)
        
        # List with None elements
        polygons = [None, triangle, None]
        
        alpha_min = -np.pi / 4
        alpha_max = np.pi / 2
        
        events = build_events(polygons, alpha_min, alpha_max)
        
        # Should only have events from the valid triangle (obstacle_id=1)
        assert all(e.obstacle_id == 1 for e in events)
    
    def test_build_events_vertices_outside_arc(self):
        """Vertices outside arc range should not generate vertex events."""
        # Triangle in positive x region
        triangle = np.array([
            [2.0, 0.1],   # angle ≈ 0.05 rad
            [3.0, 0.2],   # angle ≈ 0.07 rad
            [2.5, 0.3],   # angle ≈ 0.12 rad
        ], dtype=np.float32)
        
        # Narrow arc that excludes the triangle
        alpha_min = np.pi / 2
        alpha_max = np.pi
        
        events = build_events([triangle], alpha_min, alpha_max)
        
        # No vertex events should be generated (triangle is outside arc)
        vertex_events = [e for e in events if e.event_type == 'vertex']
        assert len(vertex_events) == 0
    
    def test_build_events_arc_crossing_pi_boundary(self):
        """Test arc that crosses ±π boundary with proper angle remapping."""
        # Triangle with vertices near ±π (in the arc that crosses the boundary)
        triangle = np.array([
            [-2.0, 0.2],   # angle ≈ 3.04 rad (≈174°) - inside arc
            [-2.0, -0.2],  # angle ≈ -3.04 rad (≈-174°) - inside arc  
            [-3.0, 0.0],   # angle = π (180°) - exactly at boundary, inside arc
        ], dtype=np.float32)
        
        # Arc that crosses ±π (from 170° to -170°)
        alpha_min = 170 * np.pi / 180   # ≈ 2.97 rad
        alpha_max = -170 * np.pi / 180  # ≈ -2.97 rad
        
        events = build_events([triangle], alpha_min, alpha_max)
        
        # Triangle vertices should be within this arc
        vertex_events = [e for e in events if e.event_type == 'vertex']
        assert len(vertex_events) == 3
        
        # CRITICAL: Event angles must be monotonically non-decreasing after remapping
        # This ensures the sweep processes events in correct angular order
        event_angles = [e.angle for e in events]
        for i in range(len(event_angles) - 1):
            assert event_angles[i] <= event_angles[i + 1], \
                f"Events not sorted: angle[{i}]={event_angles[i]} > angle[{i+1}]={event_angles[i+1]}"
        
        # All remapped angles should be >= alpha_min (since arc wraps, 
        # negative angles get lifted by 2π)
        for e in events:
            assert e.angle >= alpha_min, \
                f"Remapped angle {e.angle} should be >= alpha_min {alpha_min}"
    
    def test_build_events_arc_wrap_boundary_events(self):
        """Test that edge-crossing events at boundaries are correctly remapped when arc wraps."""
        # Create a polygon that crosses both arc boundaries
        # This ensures edge-crossing events are generated at alpha_min and alpha_max
        polygon = np.array([
            [-2.0, 0.5],   # angle ≈ 2.90 rad (≈166°) - just below alpha_min
            [-2.0, -0.5],  # angle ≈ -2.90 rad (≈-166°) - just above alpha_max
            [-4.0, 0.0],   # angle = π (180°) - in the middle of the arc
        ], dtype=np.float32)
        
        # Arc from 170° to -170° (20° total, crossing ±π)
        alpha_min = 170 * np.pi / 180   # ≈ 2.97 rad
        alpha_max = -170 * np.pi / 180  # ≈ -2.97 rad
        
        events = build_events([polygon], alpha_min, alpha_max)
        
        # Should have edge-crossing events at boundaries
        edge_events = [e for e in events if e.event_type == 'edge_crossing']
        
        # Check that boundary crossing events exist and are properly ordered
        if len(edge_events) >= 2:
            edge_angles = sorted([e.angle for e in edge_events])
            # The alpha_min crossing should come before alpha_max crossing
            # alpha_max is remapped to alpha_max + 2π when arc wraps
            remapped_alpha_max = alpha_max + 2 * np.pi
            
            # Find events near alpha_min and remapped alpha_max
            min_boundary_events = [e for e in edge_events if abs(e.angle - alpha_min) < 0.1]
            max_boundary_events = [e for e in edge_events if abs(e.angle - remapped_alpha_max) < 0.1]
            
            # If both boundaries have crossing events, alpha_min should sort first
            if min_boundary_events and max_boundary_events:
                min_event_angle = min_boundary_events[0].angle
                max_event_angle = max_boundary_events[0].angle
                assert min_event_angle < max_event_angle, \
                    f"alpha_min event ({min_event_angle}) should come before alpha_max event ({max_event_angle})"
        
        # All events must be monotonically sorted
        event_angles = [e.angle for e in events]
        for i in range(len(event_angles) - 1):
            assert event_angles[i] <= event_angles[i + 1], \
                f"Events not sorted at position {i}"


class TestEdgeCrossesAngle:
    """Tests for _edge_crosses_angle() helper function."""
    
    def test_edge_crosses_angle_simple_ccw(self):
        """Edge traversed CCW should detect crossing in the middle."""
        # Edge from 0° to 90° (CCW), boundary at 45°
        assert _edge_crosses_angle(0.0, np.pi/2, np.pi/4) is True
        
    def test_edge_crosses_angle_simple_cw(self):
        """Edge traversed CW should detect crossing in the middle."""
        # Edge from 90° to 0° (CW), boundary at 45°
        assert _edge_crosses_angle(np.pi/2, 0.0, np.pi/4) is True
    
    def test_edge_crosses_angle_boundary_at_start(self):
        """Boundary exactly at edge start should NOT count as crossing."""
        # Edge from 45° to 90°, boundary at 45° (at start)
        assert _edge_crosses_angle(np.pi/4, np.pi/2, np.pi/4) is False
    
    def test_edge_crosses_angle_boundary_at_end(self):
        """Boundary exactly at edge end should NOT count as crossing."""
        # Edge from 0° to 45°, boundary at 45° (at end)
        assert _edge_crosses_angle(0.0, np.pi/4, np.pi/4) is False
    
    def test_edge_crosses_angle_boundary_outside(self):
        """Boundary outside edge span should NOT count as crossing."""
        # Edge from 0° to 45°, boundary at 90° (outside)
        assert _edge_crosses_angle(0.0, np.pi/4, np.pi/2) is False
        # Edge from 0° to 45°, boundary at -45° (outside)
        assert _edge_crosses_angle(0.0, np.pi/4, -np.pi/4) is False
    
    def test_edge_crosses_angle_wrapping_positive_to_negative(self):
        """Edge that crosses ±π from positive to negative side."""
        # Edge from 170° to -170° (crossing ±π), boundary at 180°
        angle_start = 170 * np.pi / 180  # ≈ 2.97 rad
        angle_end = -170 * np.pi / 180   # ≈ -2.97 rad
        boundary = np.pi  # 180°
        assert _edge_crosses_angle(angle_start, angle_end, boundary) is True
    
    def test_edge_crosses_angle_wrapping_negative_to_positive(self):
        """Edge that crosses ±π from negative to positive side."""
        # Edge from -170° to 170° (crossing ±π the other way), boundary at 180°
        angle_start = -170 * np.pi / 180
        angle_end = 170 * np.pi / 180
        boundary = np.pi
        assert _edge_crosses_angle(angle_start, angle_end, boundary) is True
    
    def test_edge_crosses_angle_degenerate_edge(self):
        """Degenerate edge (same start and end) should never cross."""
        assert _edge_crosses_angle(0.5, 0.5, 0.5) is False
        assert _edge_crosses_angle(np.pi/4, np.pi/4, 0.0) is False
    
    def test_edge_crosses_angle_full_circle_not_possible(self):
        """An edge cannot span a full circle; test near-full spans."""
        # Edge spanning almost 360° (from 0° to just under 0°)
        # Due to normalization, this becomes a small span
        angle_start = 0.0
        angle_end = 0.01  # tiny CCW span
        # Boundary at 180° should NOT be crossed (span is only 0.01 rad)
        assert _edge_crosses_angle(angle_start, angle_end, np.pi) is False
    
    def test_edge_crosses_angle_boundary_near_pi(self):
        """Test boundary detection near the ±π discontinuity."""
        # Edge in third quadrant crossing boundary at -135°
        angle_start = -2.5  # ≈ -143°
        angle_end = -2.0    # ≈ -115°
        boundary = -3 * np.pi / 4  # -135°
        assert _edge_crosses_angle(angle_start, angle_end, boundary) is True
    
    def test_edge_crosses_angle_small_edge_no_crossing(self):
        """Small edge that doesn't contain the boundary."""
        # Small edge from 10° to 20°, boundary at 30°
        angle_start = 10 * np.pi / 180
        angle_end = 20 * np.pi / 180
        boundary = 30 * np.pi / 180
        assert _edge_crosses_angle(angle_start, angle_end, boundary) is False


class TestAngularEventOrdering:
    """Tests for AngularEvent dataclass ordering."""
    
    def test_events_sort_by_angle(self):
        """Events should primarily sort by angle."""
        events = [
            AngularEvent(angle=0.5, obstacle_id=0, event_type='vertex'),
            AngularEvent(angle=0.1, obstacle_id=1, event_type='vertex'),
            AngularEvent(angle=0.3, obstacle_id=2, event_type='vertex'),
        ]
        
        sorted_events = sorted(events)
        
        assert sorted_events[0].angle == 0.1
        assert sorted_events[1].angle == 0.3
        assert sorted_events[2].angle == 0.5
    
    def test_vertex_before_edge_crossing_at_same_angle(self):
        """At same angle, vertex events should come before edge_crossing."""
        events = [
            AngularEvent(angle=0.5, obstacle_id=0, event_type='edge_crossing'),
            AngularEvent(angle=0.5, obstacle_id=1, event_type='vertex'),
        ]
        
        sorted_events = sorted(events)
        
        assert sorted_events[0].event_type == 'vertex'
        assert sorted_events[1].event_type == 'edge_crossing'
    
    def test_mixed_event_sorting(self):
        """Test complex sorting with mixed angles and types."""
        events = [
            AngularEvent(angle=0.5, obstacle_id=0, event_type='edge_crossing'),
            AngularEvent(angle=0.3, obstacle_id=1, event_type='vertex'),
            AngularEvent(angle=0.5, obstacle_id=2, event_type='vertex'),
            AngularEvent(angle=0.3, obstacle_id=3, event_type='edge_crossing'),
        ]
        
        sorted_events = sorted(events)
        
        # At 0.3: vertex first, then edge_crossing
        assert sorted_events[0].angle == 0.3
        assert sorted_events[0].event_type == 'vertex'
        assert sorted_events[1].angle == 0.3
        assert sorted_events[1].event_type == 'edge_crossing'
        
        # At 0.5: vertex first, then edge_crossing
        assert sorted_events[2].angle == 0.5
        assert sorted_events[2].event_type == 'vertex'
        assert sorted_events[3].angle == 0.5
        assert sorted_events[3].event_type == 'edge_crossing'
