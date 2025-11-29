"""
Angular sweep implementation for occlusion resolution and coverage computation.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import numpy as np
from numpy.typing import NDArray

from view_arc.geometry import normalize_angle, to_polar, intersect_ray_segment, handle_angle_discontinuity


@dataclass(order=True)
class AngularEvent:
    """
    Event at a specific angle during angular sweep.
    
    Attributes:
        angle: Angular position in radians
        obstacle_id: Index of associated obstacle
        event_type: 'vertex' or 'edge_crossing'
        vertex_idx: Index of vertex in polygon (for vertex events)
        
    Ordering: Events are sorted by (angle, event_type_priority) where
    vertex events come before edge_crossing events at the same angle.
    """
    angle: float
    # Sort priority: 0 for vertex (first), 1 for edge_crossing (second)
    _sort_priority: int = field(init=False, repr=False)
    obstacle_id: int = field(compare=False)
    event_type: str = field(compare=False)
    vertex_idx: int = field(default=-1, compare=False)
    
    def __post_init__(self):
        # Vertex events should be processed before edge_crossing events at same angle
        self._sort_priority = 0 if self.event_type == 'vertex' else 1


@dataclass
class IntervalResult:
    """
    Result of depth resolution for an angular interval.
    
    Attributes:
        obstacle_id: ID of obstacle owning this interval
        min_distance: Minimum distance found in this interval
        angle_start: Start angle of interval
        angle_end: End angle of interval
        wraps: Whether the interval crosses the ±π discontinuity
    """
    obstacle_id: int
    min_distance: float
    angle_start: float
    angle_end: float
    wraps: bool = False


def build_events(
    clipped_polygons: List[NDArray[np.float32]],
    alpha_min: float,
    alpha_max: float
) -> List[AngularEvent]:
    """
    Construct sorted event list from clipped polygons.
    
    Creates events for vertices and edge crossings of angular boundaries.
    When the arc crosses ±π (alpha_min > alpha_max), angles are remapped
    to a continuous range for proper sorting.
    
    Parameters:
        clipped_polygons: List of clipped polygons in Cartesian (x, y) form
        alpha_min: Minimum arc angle
        alpha_max: Maximum arc angle
        
    Returns:
        Sorted list of AngularEvent objects with remapped angles when arc wraps
    """
    events: List[AngularEvent] = []
    
    # Determine if arc crosses ±π boundary
    arc_wraps = alpha_min > alpha_max
    
    # Compute remapped boundary angles for edge-crossing events
    # When arc wraps, alpha_max needs to be lifted by 2π to maintain order
    remapped_alpha_min = alpha_min
    remapped_alpha_max = alpha_max + 2 * np.pi if arc_wraps else alpha_max
    
    for obstacle_id, polygon in enumerate(clipped_polygons):
        if polygon is None or len(polygon) < 3:
            continue
        
        # Convert polygon vertices to polar coordinates
        radii, angles = to_polar(polygon)
        n_vertices = len(polygon)
        
        # Remap angles if arc crosses ±π boundary
        # This ensures all vertex angles are on a continuous axis
        remapped_angles = handle_angle_discontinuity(angles, alpha_min, alpha_max)
        
        for i in range(n_vertices):
            # Use original angle for arc membership check, remapped for event
            original_angle = float(angles[i])
            event_angle = float(remapped_angles[i])
            
            # Check if vertex is within the arc range (using original angles)
            if _angle_in_arc(original_angle, alpha_min, alpha_max):
                events.append(AngularEvent(
                    angle=event_angle,
                    obstacle_id=obstacle_id,
                    event_type='vertex',
                    vertex_idx=i
                ))
            
            # Check for edge crossings at alpha_min and alpha_max
            next_i = (i + 1) % n_vertices
            angle_start = float(angles[i])
            angle_end = float(angles[next_i])
            
            # Check if edge crosses alpha_min boundary
            if _edge_crosses_angle(angle_start, angle_end, alpha_min):
                events.append(AngularEvent(
                    angle=remapped_alpha_min,
                    obstacle_id=obstacle_id,
                    event_type='edge_crossing',
                    vertex_idx=i
                ))
            
            # Check if edge crosses alpha_max boundary
            if _edge_crosses_angle(angle_start, angle_end, alpha_max):
                events.append(AngularEvent(
                    angle=remapped_alpha_max,
                    obstacle_id=obstacle_id,
                    event_type='edge_crossing',
                    vertex_idx=i
                ))
    
    # Sort events by angle, with vertex events before edge_crossing at same angle
    events.sort()
    
    return events


def _angle_in_arc(angle: float, alpha_min: float, alpha_max: float) -> bool:
    """
    Check if an angle is within the arc range [alpha_min, alpha_max].
    
    Handles the case where the arc crosses the ±π boundary.
    Also handles the special full-circle case where alpha_min = -π and alpha_max = π.
    """
    # Check for full-circle case: alpha_min = -π and alpha_max = π
    # This is a special sentinel used by the API for 360° FOV
    is_full_circle = (abs(alpha_min - (-np.pi)) < 1e-9 and 
                      abs(alpha_max - np.pi) < 1e-9)
    if is_full_circle:
        return True  # All angles are in the full circle
    
    if alpha_min <= alpha_max:
        # Normal case: arc doesn't cross ±π
        return alpha_min <= angle <= alpha_max
    else:
        # Arc crosses ±π boundary (e.g., from 170° to -170°)
        return angle >= alpha_min or angle <= alpha_max


def _edge_crosses_angle(angle_start: float, angle_end: float, boundary_angle: float) -> bool:
    """
    Check if an edge (defined by start and end angles) crosses a boundary angle.
    
    An edge crosses the boundary if the boundary is strictly between 
    the start and end angles (not at the endpoints).
    """
    # Normalize the angular span
    # We need to check if boundary_angle is strictly between angle_start and angle_end
    
    # Handle the simpler case where we don't cross ±π
    diff = angle_end - angle_start
    
    # Normalize diff to be in (-π, π] for proper direction
    while diff > np.pi:
        diff -= 2 * np.pi
    while diff <= -np.pi:
        diff += 2 * np.pi
    
    if abs(diff) < 1e-10:
        # Degenerate edge (same angle)
        return False
    
    # Calculate relative position of boundary
    rel_boundary = boundary_angle - angle_start
    while rel_boundary > np.pi:
        rel_boundary -= 2 * np.pi
    while rel_boundary <= -np.pi:
        rel_boundary += 2 * np.pi
    
    # Check if boundary is strictly between start and end
    if diff > 0:
        # CCW traversal: boundary should be in (0, diff)
        return 0 < rel_boundary < diff
    else:
        # CW traversal: boundary should be in (diff, 0)
        return diff < rel_boundary < 0


def resolve_interval(
    interval_start: float,
    interval_end: float,
    active_obstacles: Dict[int, NDArray[np.float32]],
    num_samples: int = 5
) -> Optional[IntervalResult]:
    """
    Determine which obstacle owns an angular interval via ray sampling.
    
    Samples multiple rays within the interval and selects the obstacle
    with minimum average distance. For each sample ray, finds the closest
    intersecting obstacle, then picks the winner based on the lowest
    average distance across all samples.
    
    Parameters:
        interval_start: Start angle of interval
        interval_end: End angle of interval (may be > interval_start even across ±π)
        active_obstacles: Dict mapping obstacle_id to its edges array
                         Each edges array has shape (M, 2, 2) for M segments
        num_samples: Number of rays to sample across interval
        
    Returns:
        IntervalResult indicating winner and metrics, or None if no obstacles
        intersected any samples
    """
    if not active_obstacles:
        return None
    
    # Handle narrow intervals by reducing sample count
    angular_span = interval_end - interval_start
    if angular_span <= 0:
        return None
    
    # Always use at least 2 samples to ensure both endpoints are covered.
    # This prevents missing obstacles that only appear at interval boundaries.
    # For very narrow intervals, we still sample both endpoints.
    actual_samples = max(2, min(num_samples, max(2, int(angular_span / 0.001))))
    
    # Generate sample angles within the interval (evenly distributed)
    # Always includes both endpoints (interval_start and interval_end)
    # Normalize angles to [-π, π) for ray intersection tests
    sample_angles = []
    for i in range(actual_samples):
        raw_angle = interval_start + i * angular_span / (actual_samples - 1)
        # Normalize to [-π, π) for proper ray intersection
        normalized = normalize_angle(raw_angle)
        sample_angles.append(normalized)
    
    # For each sample ray, find the closest obstacle
    # Track: obstacle_id -> list of distances where it was closest
    obstacle_hits: Dict[int, List[float]] = {oid: [] for oid in active_obstacles}
    
    for angle in sample_angles:
        closest_obstacle: Optional[int] = None
        closest_distance: float = float('inf')
        
        # Check each obstacle's edges for intersection
        for obstacle_id, edges in active_obstacles.items():
            min_dist_this_obstacle = _find_min_distance_at_angle(edges, angle)
            
            if min_dist_this_obstacle is not None and min_dist_this_obstacle < closest_distance:
                closest_distance = min_dist_this_obstacle
                closest_obstacle = obstacle_id
        
        # Record the hit for the closest obstacle
        if closest_obstacle is not None:
            obstacle_hits[closest_obstacle].append(closest_distance)
    
    # Find the obstacle that was closest most often, with lowest average distance
    best_obstacle: Optional[int] = None
    best_score: float = float('inf')
    best_min_distance: float = float('inf')
    
    for obstacle_id, distances in obstacle_hits.items():
        if not distances:
            continue
        
        # Score: average distance (lower is better)
        avg_distance = sum(distances) / len(distances)
        min_distance = min(distances)
        
        # Weight by number of hits to prefer obstacles that consistently block
        # Use hit_ratio as tie-breaker (more hits is better)
        hit_ratio = len(distances) / actual_samples
        
        # Primary criterion: average distance (weighted by hit consistency)
        score = avg_distance / hit_ratio if hit_ratio > 0 else float('inf')
        
        if score < best_score or (score == best_score and min_distance < best_min_distance):
            best_score = score
            best_obstacle = obstacle_id
            best_min_distance = min_distance
    
    if best_obstacle is None:
        return None
    
    return IntervalResult(
        obstacle_id=best_obstacle,
        min_distance=best_min_distance,
        angle_start=interval_start,
        angle_end=interval_end
    )


def _find_min_distance_at_angle(
    edges: NDArray[np.float32],
    angle: float,
    max_range: float = 1e10
) -> Optional[float]:
    """
    Find minimum distance to any edge at a given angle.
    
    Parameters:
        edges: Array of edges shape (M, 2, 2) where each edge is [[x1,y1], [x2,y2]]
        angle: Ray angle in radians
        max_range: Maximum distance to consider
        
    Returns:
        Minimum distance found, or None if no intersection
    """
    min_dist: Optional[float] = None
    
    for edge in edges:
        dist = intersect_ray_segment(angle, edge[0], edge[1], max_range)
        if dist is not None:
            if min_dist is None or dist < min_dist:
                min_dist = dist
    
    return min_dist


def compute_coverage(
    events: List[AngularEvent],
    obstacle_edges: Dict[int, NDArray[np.float32]],
    alpha_min: float,
    alpha_max: float
) -> Tuple[Dict[int, float], Dict[int, float], List[IntervalResult]]:
    """
    Perform angular sweep to compute coverage for each obstacle.
    
    Processes events to establish intervals, resolves depth for each,
    and accumulates coverage statistics.
    
    The algorithm:
    1. Determine remapped arc boundaries (for arcs crossing ±π)
    2. Create intervals from consecutive events, plus boundary intervals
    3. For each interval, determine which obstacles are active
    4. Resolve depth for each interval to find the winning obstacle
    5. Accumulate coverage and track minimum distances
    
    Parameters:
        events: Sorted list of angular events
        obstacle_edges: Dict mapping obstacle_id to edge array (M, 2, 2)
        alpha_min: Start of arc
        alpha_max: End of arc
        
    Returns:
        Tuple of (coverage_dict, min_distance_dict, intervals) where:
            coverage_dict: Maps obstacle_id to total angular coverage (radians)
            min_distance_dict: Maps obstacle_id to minimum distance encountered
            intervals: List of IntervalResult with angles normalized to [-π, π)
    """
    # Initialize result accumulators
    coverage_dict: Dict[int, float] = {}
    min_distance_dict: Dict[int, float] = {}
    intervals: List[IntervalResult] = []
    
    # Determine if arc crosses ±π boundary
    arc_wraps = alpha_min > alpha_max
    
    # Compute remapped boundary angles
    # When arc wraps, lift alpha_max by 2π so we work in a continuous space
    remapped_alpha_min = alpha_min
    remapped_alpha_max = alpha_max + 2 * np.pi if arc_wraps else alpha_max
    
    # Handle empty case
    if not events or not obstacle_edges:
        return coverage_dict, min_distance_dict, intervals
    
    # Build list of interval boundaries from events
    # Include arc boundaries and all event angles
    boundaries = [remapped_alpha_min]
    for event in events:
        if remapped_alpha_min < event.angle < remapped_alpha_max:
            boundaries.append(event.angle)
    boundaries.append(remapped_alpha_max)
    
    # Remove duplicates and sort
    boundaries = sorted(set(boundaries))
    
    # Process each interval
    for i in range(len(boundaries) - 1):
        interval_start = boundaries[i]
        interval_end = boundaries[i + 1]
        
        # Skip degenerate intervals
        if interval_end - interval_start < 1e-10:
            continue
        
        # The angular span in remapped space (always positive)
        angular_span = interval_end - interval_start
        
        # Find active obstacles at the midpoint of this interval
        # Use normalized angle for edge intersection checks
        midpoint = (interval_start + interval_end) / 2
        
        # Convert midpoint back to original angle space [-π, π) for ray intersection
        query_angle = normalize_angle(midpoint)
        
        # Determine active obstacles for this interval
        active_obstacles: Dict[int, NDArray[np.float32]] = {}
        for obstacle_id, edges in obstacle_edges.items():
            # Check if any edge is intersected at the query angle
            for edge in edges:
                dist = intersect_ray_segment(query_angle, edge[0], edge[1], 1e10)
                if dist is not None:
                    # This obstacle is active at this angle
                    active_obstacles[obstacle_id] = edges
                    break
        
        # Resolve this interval to find the winner
        if active_obstacles:
            # Pass the remapped interval to resolve_interval
            # It will handle the normalization of sample angles internally
            result = resolve_interval(
                interval_start=interval_start,
                interval_end=interval_end,
                active_obstacles=active_obstacles,
                num_samples=5
            )
            
            if result is not None:
                # Update coverage for the winning obstacle
                if result.obstacle_id not in coverage_dict:
                    coverage_dict[result.obstacle_id] = 0.0
                coverage_dict[result.obstacle_id] += angular_span
                
                # Update minimum distance
                if result.obstacle_id not in min_distance_dict:
                    min_distance_dict[result.obstacle_id] = float('inf')
                if result.min_distance < min_distance_dict[result.obstacle_id]:
                    min_distance_dict[result.obstacle_id] = result.min_distance
                
                # Store interval result with normalized angles in [-π, π)
                # Preserve whether the interval crosses the ±π discontinuity
                normalized_start = normalize_angle(interval_start)
                normalized_end = normalize_angle(interval_end)
                wraps = bool(normalized_end < normalized_start)
                
                intervals.append(IntervalResult(
                    obstacle_id=result.obstacle_id,
                    min_distance=result.min_distance,
                    angle_start=normalized_start,
                    angle_end=normalized_end,
                    wraps=wraps
                ))
    
    return coverage_dict, min_distance_dict, intervals


def get_active_edges(
    polygon: NDArray[np.float32] | None,
    angle: float,
    max_range: float = 1e10
) -> NDArray[np.float32]:
    """
    Get edges of a polygon that are active (span) at given angle.
    
    An edge is "active" at a given angle if a ray from the origin at that
    angle would intersect the edge. This is determined by actual geometric
    ray-segment intersection.
    
    Parameters:
        polygon: Polygon in Cartesian coordinates (N, 2) as (x, y)
        angle: Query angle in radians
        max_range: Maximum distance for intersection check
        
    Returns:
        Array of active edges (M, 2, 2) in Cartesian coordinates where
        each edge is represented as [[x1, y1], [x2, y2]]
        Returns empty array with shape (0, 2, 2) if no edges are active.
    """
    if polygon is None or len(polygon) < 3:
        return np.empty((0, 2, 2), dtype=np.float32)
    
    n_vertices = len(polygon)
    active_edges = []
    
    for i in range(n_vertices):
        next_i = (i + 1) % n_vertices
        
        # Use actual ray-segment intersection to check if edge is active
        intersection = intersect_ray_segment(
            angle, 
            polygon[i], 
            polygon[next_i], 
            max_range
        )
        
        if intersection is not None:
            edge = np.array([
                polygon[i],
                polygon[next_i]
            ], dtype=np.float32)
            active_edges.append(edge)
    
    if not active_edges:
        return np.empty((0, 2, 2), dtype=np.float32)
    
    return np.array(active_edges, dtype=np.float32)


def _edge_spans_angle(angle_start: float, angle_end: float, query_angle: float) -> bool:
    """
    Check if an edge (defined by start and end angles) spans a query angle.
    
    An edge spans an angle if a ray at that angle would intersect the edge.
    This includes the endpoints (unlike _edge_crosses_angle which is strict).
    """
    # Handle the case where the edge wraps around ±π
    # Calculate the angular span from start to end
    diff = angle_end - angle_start
    
    # Normalize diff to be in (-π, π] for proper direction determination
    while diff > np.pi:
        diff -= 2 * np.pi
    while diff <= -np.pi:
        diff += 2 * np.pi
    
    if abs(diff) < 1e-10:
        # Degenerate edge (same angle) - only spans if at that exact angle
        return abs(normalize_angle(query_angle - angle_start)) < 1e-10
    
    # Calculate relative position of query angle from start
    rel_query = query_angle - angle_start
    while rel_query > np.pi:
        rel_query -= 2 * np.pi
    while rel_query <= -np.pi:
        rel_query += 2 * np.pi
    
    # Check if query angle is within the arc from start to end (inclusive)
    if diff > 0:
        # CCW traversal: query should be in [0, diff]
        return 0 <= rel_query <= diff
    else:
        # CW traversal: query should be in [diff, 0]
        return diff <= rel_query <= 0
