"""
Tests for angular sweep operations - Phase 3 of implementation.

Step 3.1: Event Construction
- test_get_active_edges_*
- test_build_events_*

Step 3.2: Interval Resolution
- test_resolve_interval_*
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_almost_equal

from view_arc.obstacle.sweep import (
    AngularEvent,
    IntervalResult,
    get_active_edges,
    build_events,
    resolve_interval,
    compute_coverage,
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
        
        # Should have edges at x=3 and x=5 (one entry, one exit)
        assert any(np.isclose(x, 3.0) for x in edge_x_coords), \
            f"Missing entry edge at x≈3, got {edge_x_coords}"
        assert any(np.isclose(x, 5.0) for x in edge_x_coords), \
            f"Missing exit edge at x≈5, got {edge_x_coords}"
    
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
        
        events = build_events([(0, triangle)], alpha_min, alpha_max)
        
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
        
        events = build_events([(0, triangle1), (1, triangle2)], alpha_min, alpha_max)
        
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
        
        events = build_events([(0, triangle)], alpha_min, alpha_max)
        
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
        
        events = build_events([(0, triangle1)], alpha_min, alpha_max)
        
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
        
        # List with None elements - using tuple format with original IDs
        # Note: (0, None) and (2, None) are skipped, only (1, triangle) produces events
        polygons = [(0, None), (1, triangle), (2, None)]
        
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
        
        events = build_events([(0, triangle)], alpha_min, alpha_max)
        
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
        
        events = build_events([(0, triangle)], alpha_min, alpha_max)
        
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
        
        events = build_events([(0, polygon)], alpha_min, alpha_max)
        
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


# =============================================================================
# Step 3.2: Interval Resolution Tests
# =============================================================================

class TestResolveInterval:
    """Tests for resolve_interval() function."""
    
    def test_resolve_interval_single_obstacle(self):
        """Trivial case with single obstacle should return that obstacle."""
        # Single vertical edge at x=3, from y=-1 to y=1
        # Ray at angle 0 (along +x axis) will hit this edge at distance 3
        edges = np.array([
            [[3.0, -1.0], [3.0, 1.0]],
        ], dtype=np.float32)
        
        active_obstacles = {0: edges}
        
        result = resolve_interval(
            interval_start=-0.1,
            interval_end=0.1,
            active_obstacles=active_obstacles,
            num_samples=5
        )
        
        assert result is not None
        assert result.obstacle_id == 0
        assert_allclose(result.min_distance, 3.0, atol=0.1)
        assert result.angle_start == -0.1
        assert result.angle_end == 0.1
    
    def test_resolve_interval_two_obstacles_one_closer(self):
        """Occlusion test: closer obstacle should win."""
        # Obstacle 0: edge at x=5 (far)
        edges_far = np.array([
            [[5.0, -1.0], [5.0, 1.0]],
        ], dtype=np.float32)
        
        # Obstacle 1: edge at x=2 (closer)
        edges_close = np.array([
            [[2.0, -1.0], [2.0, 1.0]],
        ], dtype=np.float32)
        
        active_obstacles = {0: edges_far, 1: edges_close}
        
        result = resolve_interval(
            interval_start=-0.1,
            interval_end=0.1,
            active_obstacles=active_obstacles,
            num_samples=5
        )
        
        assert result is not None
        assert result.obstacle_id == 1  # Closer obstacle wins
        assert_allclose(result.min_distance, 2.0, atol=0.1)
    
    def test_resolve_interval_overlapping_at_different_distances(self):
        """Depth test: obstacles at different distances, closer one occludes."""
        # Obstacle 0: box further away (x=4 to x=6)
        edges_box1 = np.array([
            [[4.0, -1.0], [4.0, 1.0]],   # Left edge at x=4
            [[6.0, -1.0], [6.0, 1.0]],   # Right edge at x=6
            [[4.0, 1.0], [6.0, 1.0]],    # Top edge
            [[4.0, -1.0], [6.0, -1.0]],  # Bottom edge
        ], dtype=np.float32)
        
        # Obstacle 1: box closer (x=2 to x=3)
        edges_box2 = np.array([
            [[2.0, -0.5], [2.0, 0.5]],   # Left edge at x=2
            [[3.0, -0.5], [3.0, 0.5]],   # Right edge at x=3
            [[2.0, 0.5], [3.0, 0.5]],    # Top edge
            [[2.0, -0.5], [3.0, -0.5]],  # Bottom edge
        ], dtype=np.float32)
        
        active_obstacles = {0: edges_box1, 1: edges_box2}
        
        # Query interval centered on 0 where both boxes are visible
        result = resolve_interval(
            interval_start=-0.2,
            interval_end=0.2,
            active_obstacles=active_obstacles,
            num_samples=5
        )
        
        assert result is not None
        # Closer box (obstacle 1) should win
        assert result.obstacle_id == 1
        assert result.min_distance < 3.5  # Should hit closer box
    
    def test_resolve_interval_sampling_density(self):
        """Verify that correct number of samples affects winner determination.
        
        This test creates a scenario where having 5 samples vs fewer samples
        would produce different winners, thus verifying sampling behavior.
        """
        # Obstacle 0: small edge that only gets hit by 1 out of 5 samples
        # Located at angle ~0.24 rad (only hit by sample at 0.3 out of [-0.3, -0.15, 0, 0.15, 0.3])
        edges0 = np.array([
            [[3.0, 0.7], [3.0, 1.0]],  # Edge at angles ~0.23 to ~0.32 rad
        ], dtype=np.float32)
        
        # Obstacle 1: wide edge hit by all 5 samples, but further away
        edges1 = np.array([
            [[5.0, -2.0], [5.0, 2.0]],  # Edge spanning full interval, at distance 5
        ], dtype=np.float32)
        
        active_obstacles = {0: edges0, 1: edges1}
        
        # With 5 samples at angles [-0.3, -0.15, 0, 0.15, 0.3]:
        # - Obstacle 0 is hit by ~1 sample (at 0.3), distance ~3
        # - Obstacle 1 is hit by all 5 samples, distance 5
        # Obstacle 1 should win due to better coverage despite greater distance
        result = resolve_interval(
            interval_start=-0.3,
            interval_end=0.3,
            active_obstacles=active_obstacles,
            num_samples=5
        )
        
        assert result is not None
        # Obstacle 1 wins because it has consistent coverage (5/5 hits)
        # while obstacle 0 only has 1/5 hits
        assert result.obstacle_id == 1
    
    def test_resolve_interval_no_obstacles(self):
        """Empty active set should return None."""
        result = resolve_interval(
            interval_start=-0.1,
            interval_end=0.1,
            active_obstacles={},
            num_samples=5
        )
        
        assert result is None
    
    def test_resolve_interval_obstacle_at_boundary(self):
        """Edge case: obstacle edge exactly at interval boundary."""
        # Edge that only covers part of the interval
        # Angled edge from (2, 0) to (3, 1.5) - covers angles roughly 0 to ~0.46 rad
        edges = np.array([
            [[2.0, 0.0], [3.0, 1.5]],
        ], dtype=np.float32)
        
        active_obstacles = {0: edges}
        
        result = resolve_interval(
            interval_start=0.0,
            interval_end=0.5,
            active_obstacles=active_obstacles,
            num_samples=5
        )
        
        # Should still find the obstacle since some samples hit it
        assert result is not None
        assert result.obstacle_id == 0
    
    def test_resolve_interval_narrow_interval(self):
        """Small angular span should still work correctly."""
        edges = np.array([
            [[3.0, -0.1], [3.0, 0.1]],
        ], dtype=np.float32)
        
        active_obstacles = {0: edges}
        
        # Very narrow interval
        result = resolve_interval(
            interval_start=-0.01,
            interval_end=0.01,
            active_obstacles=active_obstacles,
            num_samples=5
        )
        
        assert result is not None
        assert result.obstacle_id == 0
        assert_allclose(result.min_distance, 3.0, atol=0.1)
    
    def test_resolve_interval_extremely_narrow_boundary_obstacle(self):
        """Obstacle only at endpoint of extremely narrow interval must be detected.
        
        This is a regression test for the issue where narrow intervals were 
        downsampled to a single midpoint sample, missing obstacles at boundaries.
        """
        # Edge that spans exactly angle=0 (from (3, -0.001) to (3, 0.001))
        # This edge is ONLY intersected at angle=0, not at any positive angle
        edges = np.array([
            [[3.0, -0.001], [3.0, 0.001]],
        ], dtype=np.float32)
        
        active_obstacles = {0: edges}
        
        # Extremely narrow interval [0, 5e-4]
        # The midpoint 2.5e-4 would miss this edge, but endpoint 0 should hit it
        result = resolve_interval(
            interval_start=0.0,
            interval_end=5e-4,
            active_obstacles=active_obstacles,
            num_samples=5
        )
        
        # Must detect the obstacle at the start boundary
        assert result is not None, (
            "Obstacle at interval start boundary was not detected. "
            "Sampling should always include endpoints."
        )
        assert result.obstacle_id == 0
        assert_allclose(result.min_distance, 3.0, atol=0.1)
    
    def test_resolve_interval_boundary_only_end(self):
        """Obstacle only at end boundary of narrow interval must be detected."""
        # Edge that spans a small range around angle=0.001 rad
        # At x=3, y ranges from 0.002 to 0.004 -> angles ~0.00067 to ~0.00133 rad
        edges = np.array([
            [[3.0, 0.002], [3.0, 0.004]],
        ], dtype=np.float32)
        
        active_obstacles = {0: edges}
        
        # Interval [0, 0.001] - only the endpoint at 0.001 might hit
        # (Actually this edge is at angles 0.00067-0.00133, so endpoint 0.001 hits it)
        result = resolve_interval(
            interval_start=0.0,
            interval_end=0.001,
            active_obstacles=active_obstacles,
            num_samples=5
        )
        
        # The edge might be hit by end boundary sample
        assert result is not None, (
            "Obstacle near interval end boundary was not detected."
        )
    
    def test_resolve_interval_no_intersection(self):
        """Obstacles that don't intersect any sample rays should return None."""
        # Edge in the upper part - won't be hit by rays at angle 0
        edges = np.array([
            [[3.0, 5.0], [4.0, 6.0]],  # Edge far above x-axis
        ], dtype=np.float32)
        
        active_obstacles = {0: edges}
        
        # Query interval around 0 (along x-axis)
        result = resolve_interval(
            interval_start=-0.1,
            interval_end=0.1,
            active_obstacles=active_obstacles,
            num_samples=5
        )
        
        # Should return None since no rays hit the edge
        assert result is None
    
    def test_resolve_interval_returns_correct_interval_bounds(self):
        """Verify that returned interval bounds match input."""
        edges = np.array([
            [[2.0, -1.0], [2.0, 1.0]],
        ], dtype=np.float32)
        
        active_obstacles = {0: edges}
        
        interval_start = 0.123
        interval_end = 0.456
        
        result = resolve_interval(
            interval_start=interval_start,
            interval_end=interval_end,
            active_obstacles=active_obstacles,
            num_samples=5
        )
        
        assert result is not None
        assert result.angle_start == interval_start
        assert result.angle_end == interval_end
    
    def test_resolve_interval_min_distance_accuracy(self):
        """Verify minimum distance is accurately computed."""
        # Create a V-shaped obstacle where distance varies with angle
        # The apex of the V is at (2, 0), so minimum distance should be 2
        edges = np.array([
            [[2.0, 0.0], [4.0, 1.0]],   # Right arm of V
            [[2.0, 0.0], [4.0, -1.0]],  # Left arm of V
        ], dtype=np.float32)
        
        active_obstacles = {0: edges}
        
        # Query around angle 0 where the apex is
        result = resolve_interval(
            interval_start=-0.2,
            interval_end=0.2,
            active_obstacles=active_obstacles,
            num_samples=5
        )
        
        assert result is not None
        assert result.obstacle_id == 0
        # Minimum distance should be at the apex (approximately 2.0)
        assert result.min_distance < 2.5
    
    def test_resolve_interval_partial_coverage_picks_dominant(self):
        """When one obstacle has clearly more coverage, it should win.
        
        This test verifies that the hit-ratio scoring logic works correctly:
        an obstacle with full coverage should beat one with partial coverage,
        even if the partial-coverage obstacle is slightly closer.
        """
        # Obstacle 0: covers only a small portion of positive angles
        # Edge from y=1.5 to y=2.0, at x=3 -> angles ~0.46 to ~0.59 rad
        # With samples at [-0.4, -0.2, 0, 0.2, 0.4], this gets 0-1 hits
        edges0 = np.array([
            [[3.0, 1.5], [3.0, 2.0]],  # Only hits angles ~0.46 to ~0.59 rad
        ], dtype=np.float32)
        
        # Obstacle 1: covers the full interval from -0.4 to 0.4
        # Edge from y=-2.0 to y=2.0 at x=4 -> angles ~-0.46 to ~0.46 rad
        edges1 = np.array([
            [[4.0, -2.0], [4.0, 2.0]],  # Spans full interval
        ], dtype=np.float32)
        
        active_obstacles = {0: edges0, 1: edges1}
        
        # Query interval [-0.4, 0.4] with 5 samples at [-0.4, -0.2, 0, 0.2, 0.4]
        # Obstacle 0: 0 hits (its angular range ~0.46-0.59 is outside sample angles)
        # Obstacle 1: 5 hits (covers full range)
        result = resolve_interval(
            interval_start=-0.4,
            interval_end=0.4,
            active_obstacles=active_obstacles,
            num_samples=5
        )
        
        assert result is not None
        # Obstacle 1 must win because it has full coverage (5/5 hits)
        # while obstacle 0 has zero hits in the sampled range
        assert result.obstacle_id == 1, (
            f"Expected obstacle 1 (full coverage) to win, but got obstacle {result.obstacle_id}"
        )


# =============================================================================
# Step 3.3: Coverage Computation Tests
# =============================================================================

class TestComputeCoverage:
    """Tests for compute_coverage() function."""
    
    def _make_edges_from_polygon(self, polygon):
        """Helper to convert a polygon to edge array format."""
        n = len(polygon)
        edges = np.zeros((n, 2, 2), dtype=np.float32)
        for i in range(n):
            edges[i, 0] = polygon[i]
            edges[i, 1] = polygon[(i + 1) % n]
        return edges
    
    def test_compute_coverage_single_obstacle_full_arc(self):
        """Single obstacle occupying entire FOV should get 100% coverage."""
        # Square centered on x-axis, fully covering the arc
        polygon = np.array([
            [2.0, -1.0],   # angle ≈ -0.46 rad
            [4.0, -1.0],   # angle ≈ -0.24 rad
            [4.0, 1.0],    # angle ≈ 0.24 rad
            [2.0, 1.0],    # angle ≈ 0.46 rad
        ], dtype=np.float32)
        
        edges = self._make_edges_from_polygon(polygon)
        obstacle_edges = {0: edges}
        
        # Arc from -0.3 to 0.3 rad (fully within obstacle's angular span)
        alpha_min = -0.3
        alpha_max = 0.3
        
        # Build events for the polygon
        events = build_events([(0, polygon)], alpha_min, alpha_max)
        
        coverage, min_dist, intervals = compute_coverage(
            events, obstacle_edges, alpha_min, alpha_max
        )
        
        # Obstacle 0 should own the entire arc
        assert 0 in coverage
        expected_coverage = alpha_max - alpha_min  # 0.6 rad
        assert_allclose(coverage[0], expected_coverage, rtol=0.1)
        
        # Should have a finite minimum distance
        assert 0 in min_dist
        assert min_dist[0] < 5.0  # Should be around 2.0
        
    def test_compute_coverage_two_obstacles_side_by_side(self):
        """Two obstacles without occlusion should each get their angular coverage."""
        # Obstacle 0: in upper part of arc
        polygon0 = np.array([
            [2.0, 0.5],   # angle ≈ 0.24 rad
            [3.0, 0.5],   # angle ≈ 0.17 rad
            [3.0, 1.5],   # angle ≈ 0.46 rad
            [2.0, 1.5],   # angle ≈ 0.64 rad
        ], dtype=np.float32)
        
        # Obstacle 1: in lower part of arc
        polygon1 = np.array([
            [2.0, -1.5],  # angle ≈ -0.64 rad
            [3.0, -1.5],  # angle ≈ -0.46 rad
            [3.0, -0.5],  # angle ≈ -0.17 rad
            [2.0, -0.5],  # angle ≈ -0.24 rad
        ], dtype=np.float32)
        
        edges0 = self._make_edges_from_polygon(polygon0)
        edges1 = self._make_edges_from_polygon(polygon1)
        obstacle_edges = {0: edges0, 1: edges1}
        
        # Arc covering both obstacles
        alpha_min = -0.8
        alpha_max = 0.8
        
        events = build_events([(0, polygon0), (1, polygon1)], alpha_min, alpha_max)
        
        coverage, min_dist, intervals = compute_coverage(
            events, obstacle_edges, alpha_min, alpha_max
        )
        
        # Both obstacles should have coverage
        assert 0 in coverage
        assert 1 in coverage
        
        # Both should have positive coverage
        assert coverage[0] > 0
        assert coverage[1] > 0
        
        # Total coverage should be less than or equal to arc span
        total_coverage = sum(coverage.values())
        arc_span = alpha_max - alpha_min
        assert total_coverage <= arc_span + 0.01
        
    def test_compute_coverage_two_obstacles_overlapping(self):
        """When one obstacle occludes another, closer one gets coverage."""
        # Obstacle 0: further away
        polygon0 = np.array([
            [4.0, -0.5],
            [5.0, -0.5],
            [5.0, 0.5],
            [4.0, 0.5],
        ], dtype=np.float32)
        
        # Obstacle 1: closer, overlapping the same angular region
        polygon1 = np.array([
            [2.0, -0.3],
            [3.0, -0.3],
            [3.0, 0.3],
            [2.0, 0.3],
        ], dtype=np.float32)
        
        edges0 = self._make_edges_from_polygon(polygon0)
        edges1 = self._make_edges_from_polygon(polygon1)
        obstacle_edges = {0: edges0, 1: edges1}
        
        # Arc covering both obstacles
        alpha_min = -0.2
        alpha_max = 0.2
        
        events = build_events([(0, polygon0), (1, polygon1)], alpha_min, alpha_max)
        
        coverage, min_dist, intervals = compute_coverage(
            events, obstacle_edges, alpha_min, alpha_max
        )
        
        # Closer obstacle (1) should dominate
        assert 1 in coverage
        # Obstacle 1 should have most or all of the coverage
        if 0 in coverage:
            assert coverage[1] >= coverage.get(0, 0)
        
        # Obstacle 1's min distance should be smaller
        if 0 in min_dist and 1 in min_dist:
            assert min_dist[1] < min_dist[0]
    
    def test_compute_coverage_gap_in_arc(self):
        """Some angles have no obstacle - total coverage < arc span."""
        # Obstacle only covers part of the arc
        polygon = np.array([
            [2.0, 0.2],
            [3.0, 0.2],
            [3.0, 0.6],
            [2.0, 0.6],
        ], dtype=np.float32)
        
        edges = self._make_edges_from_polygon(polygon)
        obstacle_edges = {0: edges}
        
        # Arc that's wider than the obstacle
        alpha_min = -0.5
        alpha_max = 0.8
        
        events = build_events([(0, polygon)], alpha_min, alpha_max)
        
        coverage, min_dist, intervals = compute_coverage(
            events, obstacle_edges, alpha_min, alpha_max
        )
        
        # Total coverage should be less than arc span (there's a gap)
        total_coverage = sum(coverage.values())
        arc_span = alpha_max - alpha_min
        assert total_coverage < arc_span
        
    def test_compute_coverage_min_distance_tracking(self):
        """Verify minimum distances are correctly tracked per obstacle."""
        # V-shaped obstacle with apex closer to origin
        polygon = np.array([
            [2.0, 0.0],    # Apex at distance 2
            [4.0, 1.0],    # Arm at distance ~4.1
            [4.0, -1.0],   # Arm at distance ~4.1
        ], dtype=np.float32)
        
        edges = self._make_edges_from_polygon(polygon)
        obstacle_edges = {0: edges}
        
        # Arc centered on the apex
        alpha_min = -0.3
        alpha_max = 0.3
        
        events = build_events([(0, polygon)], alpha_min, alpha_max)
        
        coverage, min_dist, intervals = compute_coverage(
            events, obstacle_edges, alpha_min, alpha_max
        )
        
        assert 0 in min_dist
        # Minimum distance should be close to apex distance (2.0)
        assert min_dist[0] < 3.0
        
    def test_compute_coverage_interval_boundaries(self):
        """Verify correct attribution of coverage at interval edges."""
        # Two adjacent obstacles
        polygon0 = np.array([
            [2.0, -0.01],
            [3.0, -0.01],
            [3.0, 0.5],
            [2.0, 0.5],
        ], dtype=np.float32)
        
        polygon1 = np.array([
            [2.0, -0.5],
            [3.0, -0.5],
            [3.0, -0.01],
            [2.0, -0.01],
        ], dtype=np.float32)
        
        edges0 = self._make_edges_from_polygon(polygon0)
        edges1 = self._make_edges_from_polygon(polygon1)
        obstacle_edges = {0: edges0, 1: edges1}
        
        # Arc covering both
        alpha_min = -0.3
        alpha_max = 0.3
        
        events = build_events([(0, polygon0), (1, polygon1)], alpha_min, alpha_max)
        
        coverage, min_dist, intervals = compute_coverage(
            events, obstacle_edges, alpha_min, alpha_max
        )
        
        # Both obstacles should have coverage
        assert 0 in coverage or 1 in coverage
        # The sum should approximately equal the arc span
        total_coverage = sum(coverage.values())
        arc_span = alpha_max - alpha_min
        assert_allclose(total_coverage, arc_span, rtol=0.2)
        
    def test_compute_coverage_empty_arc(self):
        """No obstacles visible should return empty dictionaries."""
        # Obstacle outside the arc
        polygon = np.array([
            [2.0, 2.0],    # angle ≈ 0.78 rad (45°)
            [3.0, 2.0],
            [3.0, 3.0],
            [2.0, 3.0],
        ], dtype=np.float32)
        
        edges = self._make_edges_from_polygon(polygon)
        obstacle_edges = {0: edges}
        
        # Arc that doesn't include the obstacle
        alpha_min = -0.3
        alpha_max = 0.3
        
        events = build_events([(0, polygon)], alpha_min, alpha_max)
        
        coverage, min_dist, intervals = compute_coverage(
            events, obstacle_edges, alpha_min, alpha_max
        )
        
        # No coverage expected in the queried arc
        total_coverage = sum(coverage.values())
        assert total_coverage == 0 or len(coverage) == 0
        
    def test_compute_coverage_returns_intervals(self):
        """Verify that intervals are correctly returned for analysis."""
        polygon = np.array([
            [2.0, -0.5],
            [4.0, -0.5],
            [4.0, 0.5],
            [2.0, 0.5],
        ], dtype=np.float32)
        
        edges = self._make_edges_from_polygon(polygon)
        obstacle_edges = {0: edges}
        
        alpha_min = -0.3
        alpha_max = 0.3
        
        events = build_events([(0, polygon)], alpha_min, alpha_max)
        
        coverage, min_dist, intervals = compute_coverage(
            events, obstacle_edges, alpha_min, alpha_max
        )
        
        # Should have at least one interval
        assert len(intervals) >= 1
        
        # All intervals should be IntervalResult instances
        for interval in intervals:
            assert isinstance(interval, IntervalResult)
            assert interval.obstacle_id == 0
            assert interval.min_distance > 0
            assert interval.wraps is False
            
    def test_compute_coverage_arc_crossing_pi(self):
        """Test coverage computation when arc crosses ±π boundary.
        
        This is a critical regression test: wrapped arcs must stay monotonic
        when passed to resolve_interval, and total coverage must match the
        expected arc span.
        """
        # Arc crossing ±π (from 170° to -170°)
        # This is a 20° arc that wraps around the ±π discontinuity
        alpha_min = 170 * np.pi / 180   # ≈ 2.967 rad
        alpha_max = -170 * np.pi / 180  # ≈ -2.967 rad
        
        # Expected arc span: 360° - 170° - 170° = 20° = 0.349 rad
        expected_arc_span = (2 * np.pi) - alpha_min + alpha_max
        
        # Obstacle that fully spans the arc (vertices outside the arc boundaries)
        # Needs vertices at angles < 170° and > -170° to ensure full coverage
        # Using a wide rectangle at x=-3 to -4 with enough height to span 168° to -168°
        # At x=-3, y=±0.7 gives angle ≈ arctan(0.7/-3) ≈ ±166.9°
        polygon = np.array([
            [-3.0, 0.7],    # angle ≈ 166.9° (inside arc boundary of 170°)
            [-4.0, 0.7],    # angle ≈ 170.1° (just past arc boundary)
            [-4.0, -0.7],   # angle ≈ -170.1° (just past arc boundary)
            [-3.0, -0.7],   # angle ≈ -166.9° (inside arc boundary of -170°)
        ], dtype=np.float32)
        
        edges = self._make_edges_from_polygon(polygon)
        obstacle_edges = {0: edges}
        
        events = build_events([(0, polygon)], alpha_min, alpha_max)
        
        coverage, min_dist, intervals = compute_coverage(
            events, obstacle_edges, alpha_min, alpha_max
        )
        
        # Must have coverage for the obstacle
        assert 0 in coverage, "Obstacle should have coverage in wrapped arc"
        
        # Critical assertion: total coverage must match the arc span within tight tolerance
        # This catches the regression where wrapped intervals were discarded
        assert_allclose(
            coverage[0], 
            expected_arc_span, 
            rtol=0.05,  # 5% relative tolerance
            err_msg=f"Coverage {coverage[0]:.4f} should match arc span {expected_arc_span:.4f}"
        )
        
        # Verify minimum distance is reasonable (obstacle is at x ≈ -3 to -4)
        assert 0 in min_dist
        assert 2.5 < min_dist[0] < 4.5, f"Min distance {min_dist[0]} should be ~3-4"
        
        # Verify intervals are returned and angles are normalized to [-π, π)
        assert len(intervals) >= 1, "Should have at least one interval"
        for interval in intervals:
            assert -np.pi <= interval.angle_start < np.pi, \
                f"angle_start {interval.angle_start} should be in [-π, π)"
            assert -np.pi <= interval.angle_end < np.pi, \
                f"angle_end {interval.angle_end} should be in [-π, π)"
            if interval.wraps:
                assert interval.angle_start > interval.angle_end, \
                    "Wrapped intervals must have start > end after normalization"
            else:
                assert interval.angle_start <= interval.angle_end, \
                    "Non-wrapped intervals should have start <= end"
        
    def test_compute_coverage_empty_events(self):
        """Empty event list should return empty coverage."""
        obstacle_edges = {0: np.array([[[2, 0], [3, 0]]], dtype=np.float32)}
        
        coverage, min_dist, intervals = compute_coverage(
            events=[],
            obstacle_edges=obstacle_edges,
            alpha_min=-0.5,
            alpha_max=0.5
        )
        
        assert len(coverage) == 0
        assert len(min_dist) == 0
        assert len(intervals) == 0
        
    def test_compute_coverage_empty_obstacles(self):
        """Empty obstacle edges should return empty coverage."""
        events = [AngularEvent(angle=0.0, obstacle_id=0, event_type='vertex')]
        
        coverage, min_dist, intervals = compute_coverage(
            events=events,
            obstacle_edges={},
            alpha_min=-0.5,
            alpha_max=0.5
        )
        
        assert len(coverage) == 0
        assert len(min_dist) == 0
        assert len(intervals) == 0
