"""
Polygon clipping operations for sector wedge (half-planes + circle).
"""

from typing import List, Optional
import numpy as np
from numpy.typing import NDArray

HALFPLANE_EPSILON = 1e-6
ARC_SAMPLE_STEP = np.deg2rad(7.5)
MIN_ARC_SAMPLES = 32


def clip_polygon_to_wedge(
    polygon: NDArray[np.float32],
    alpha_min: float,
    alpha_max: float,
    max_range: float
) -> Optional[NDArray[np.float32]]:
    """
    Clip polygon against circular sector wedge.
    
    Applies clipping in stages:
    - For narrow arcs (≤180°): intersect two half-planes
    - For wide arcs (>180°): skip half-plane clipping (sweep handles angular filtering)
    - Finally clip by circle at max_range
    
    Parameters:
        polygon: Polygon vertices (N, 2) in viewer-centric frame
        alpha_min: Minimum angle of wedge in radians (normalized to [-π, π))
        alpha_max: Maximum angle of wedge in radians (normalized to [-π, π))
        max_range: Radius of circular boundary
        
    Returns:
        Clipped polygon vertices (M, 2) or None if completely clipped away
        Returns None for degenerate results (<3 vertices)
    """
    if not is_valid_polygon(polygon):
        return None
    
    # Check for full-circle case: alpha_min = -π and alpha_max = π
    # This is a special sentinel used by the API for 360° FOV
    is_full_circle = (abs(alpha_min - (-np.pi)) < 1e-9 and 
                      abs(alpha_max - np.pi) < 1e-9)
    
    if is_full_circle:
        # Full circle: skip all angular clipping, only clip by range
        result = polygon.copy()
        result = clip_polygon_circle(result, radius=max_range)
        if result.shape[0] < 3:
            return None
        return result
    
    # Calculate arc span to determine clipping strategy
    # Arc wraps around ±π if alpha_min > alpha_max
    arc_wraps = alpha_min > alpha_max
    if arc_wraps:
        # Wrapped arc: span goes from alpha_min to +π, then -π to alpha_max
        arc_span = (2 * np.pi) - alpha_min + alpha_max
    else:
        # Non-wrapped arc: span is simply the difference
        arc_span = alpha_max - alpha_min
    
    # Half-plane intersection only works for arcs ≤ 180°
    # For wider arcs, skip half-plane clipping and let the sweep filter by angle
    is_narrow_arc = arc_span <= np.pi + 1e-6  # Small epsilon for floating point
    
    if is_narrow_arc:
        # Narrow arc case: use intersection of two half-planes
        # Stage 1: Clip by half-plane at alpha_min (keep left = CCW from alpha_min ray)
        result = clip_polygon_halfplane(polygon, plane_angle=alpha_min, keep_left=True)
        if result.shape[0] < 3:
            return None
        
        # Stage 2: Clip by half-plane at alpha_max (keep right = CW from alpha_max ray)
        result = clip_polygon_halfplane(result, plane_angle=alpha_max, keep_left=False)
        if result.shape[0] < 3:
            return None
    else:
        # Wide arc case (>180°): skip half-plane clipping
        # The sweep algorithm's _angle_in_arc correctly handles wide/wrapped arcs,
        # so we only need to clip by the circle boundary.
        # 
        # This is valid because:
        # 1. The polygon will be clipped to the max range circle
        # 2. The sweep will only count coverage for angles within the arc
        # 3. Vertices outside the arc won't contribute to coverage
        result = polygon.copy()
    
    # Final stage: Clip by circle at max_range
    result = clip_polygon_circle(result, radius=max_range)
    if result.shape[0] < 3:
        return None
    
    return result


def clip_polygon_halfplane(
    polygon: NDArray[np.float32],
    plane_angle: float,
    keep_left: bool
) -> NDArray[np.float32]:
    """
    Clip polygon against a half-plane defined by a ray from origin.
    
    Implements Sutherland-Hodgman algorithm for a single half-plane.
    The half-plane is defined by a ray from the origin at the given angle.
    
    Parameters:
        polygon: Polygon vertices (N, 2)
        plane_angle: Angle in radians defining the ray boundary
        keep_left: If True, keep points to the left of the ray (CCW side)
        
    Returns:
        Clipped polygon vertices (M, 2), may be empty array
    """
    if polygon.shape[0] == 0:
        return polygon.copy()
    
    # Direction of the ray (unit vector at plane_angle)
    ray_dir = np.array([np.cos(plane_angle), np.sin(plane_angle)], dtype=np.float32)
    
    # Normal to the ray: perpendicular pointing left (CCW)
    # For ray direction (dx, dy), left normal is (-dy, dx)
    normal = np.array([-ray_dir[1], ray_dir[0]], dtype=np.float32)
    
    if not keep_left:
        # If keeping right side, flip the normal
        normal = -normal
    
    def signed_distance(point: NDArray[np.float32]) -> float:
        """Compute signed distance from point to the half-plane boundary."""
        # Distance = dot(point, normal)
        # Positive means on the "keep" side, negative means on the "clip" side
        return float(np.dot(point, normal))
    
    def compute_intersection(p1: NDArray[np.float32], p2: NDArray[np.float32]) -> NDArray[np.float32]:
        """Compute intersection of edge p1->p2 with the half-plane boundary."""
        d1 = signed_distance(p1)
        d2 = signed_distance(p2)
        
        # Parametric intersection: t where d1 + t*(d2-d1) = 0
        t = d1 / (d1 - d2)
        
        return (p1 + t * (p2 - p1)).astype(np.float32)
    
    tolerance = float(HALFPLANE_EPSILON)
    output_vertices = []
    n = polygon.shape[0]
    
    for i in range(n):
        current = polygon[i]
        next_vertex = polygon[(i + 1) % n]
        
        d_current = signed_distance(current)
        d_next = signed_distance(next_vertex)
        
        current_inside = d_current >= -tolerance
        next_inside = d_next >= -tolerance
        
        if current_inside:
            # Current vertex is inside, add it
            output_vertices.append(current.copy())
            
            if not next_inside:
                # Edge exits the half-plane, add intersection point
                intersection = compute_intersection(current, next_vertex)
                output_vertices.append(intersection)
        else:
            # Current vertex is outside
            if next_inside:
                # Edge enters the half-plane, add intersection point
                intersection = compute_intersection(current, next_vertex)
                output_vertices.append(intersection)
    
    if len(output_vertices) == 0:
        return np.array([], dtype=np.float32).reshape(0, 2)
    
    return np.array(output_vertices, dtype=np.float32)


def clip_polygon_circle(
    polygon: NDArray[np.float32],
    radius: float
) -> NDArray[np.float32]:
    """
    Clip polygon against circle centered at origin.
    
    Uses analytical quadratic solution for edge-circle intersections.
    
    Parameters:
        polygon: Polygon vertices (N, 2)
        radius: Circle radius
        
    Returns:
        Clipped polygon vertices (M, 2), may be empty array
    """
    if polygon.shape[0] == 0:
        return polygon.copy()
    
    radius_sq = radius * radius
    output_vertices: List[NDArray[np.float32]] = []
    n = polygon.shape[0]
    pending_arc_start: Optional[float] = None
    
    def is_inside(point: NDArray[np.float32]) -> bool:
        """Check if point is inside or on the circle."""
        return float(np.dot(point, point)) <= radius_sq + HALFPLANE_EPSILON

    def append_arc_points(start_angle: float, end_angle: float) -> None:
        """Approximate circular arc between start and end angles (CCW)."""
        angle_delta = end_angle - start_angle
        if angle_delta <= 0.0:
            angle_delta += 2.0 * np.pi
        if angle_delta < 1e-6:
            return

        segments = max(1, int(np.ceil(angle_delta / ARC_SAMPLE_STEP)))
        step = angle_delta / segments
        for idx in range(1, segments):
            angle = start_angle + step * idx
            point = np.array([
                radius * np.cos(angle),
                radius * np.sin(angle),
            ], dtype=np.float32)
            output_vertices.append(point)
    
    def compute_circle_intersections(
        p1: NDArray[np.float32], 
        p2: NDArray[np.float32]
    ) -> List[tuple[float, NDArray[np.float32]]]:
        """
        Compute intersection points of line segment p1->p2 with circle.
        
        Uses parametric line equation: P(t) = p1 + t*(p2-p1), t in [0,1]
        Substituted into circle equation: |P(t)|^2 = r^2
        Results in quadratic: at^2 + bt + c = 0
        
        Returns list of (t, point) tuples for valid intersections.
        """
        d = p2 - p1  # Direction vector
        
        # Quadratic coefficients for |p1 + t*d|^2 = r^2
        # (p1 + t*d) · (p1 + t*d) = r^2
        # p1·p1 + 2t(p1·d) + t^2(d·d) = r^2
        # (d·d)t^2 + 2(p1·d)t + (p1·p1 - r^2) = 0
        a = float(np.dot(d, d))
        b = 2.0 * float(np.dot(p1, d))
        c = float(np.dot(p1, p1)) - radius_sq
        
        discriminant = b * b - 4.0 * a * c
        
        if discriminant < 0 or a < 1e-12:
            return []
        
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)
        
        intersections = []
        for t in [t1, t2]:
            if 0.0 <= t <= 1.0:
                point = (p1 + t * d).astype(np.float32)
                intersections.append((t, point))
        
        # Sort by t parameter
        intersections.sort(key=lambda x: x[0])
        return intersections
    
    for i in range(n):
        current = polygon[i]
        next_vertex = polygon[(i + 1) % n]
        
        current_inside = is_inside(current)
        next_inside = is_inside(next_vertex)
        
        if current_inside:
            # Current vertex is inside, add it
            output_vertices.append(current.copy())
        
        # Find intersections along the edge
        intersections = compute_circle_intersections(current, next_vertex)
        
        if current_inside and not next_inside:
            # Edge exits the circle - add the exit intersection
            if intersections:
                exit_point = intersections[0][1]
                if not np.allclose(exit_point, output_vertices[-1], atol=1e-7):
                    output_vertices.append(exit_point)
                pending_arc_start = float(np.arctan2(exit_point[1], exit_point[0]))
            else:
                pending_arc_start = float(np.arctan2(current[1], current[0]))
        elif not current_inside and next_inside:
            # Edge enters the circle - add the entry intersection
            if intersections:
                entry_point = intersections[-1][1]
            else:
                entry_point = next_vertex.astype(np.float32)

            if pending_arc_start is not None:
                arc_end = float(np.arctan2(entry_point[1], entry_point[0]))
                append_arc_points(pending_arc_start, arc_end)
                pending_arc_start = None

            if not output_vertices or not np.allclose(entry_point, output_vertices[-1], atol=1e-7):
                output_vertices.append(entry_point)
        elif not current_inside and not next_inside:
            # Both outside - check if edge passes through circle
            if len(intersections) == 2:
                # Edge crosses circle twice (enters and exits)
                output_vertices.append(intersections[0][1])
                output_vertices.append(intersections[1][1])
    
    if pending_arc_start is not None and len(output_vertices) > 0:
        first_angle = float(np.arctan2(output_vertices[0][1], output_vertices[0][0]))
        append_arc_points(pending_arc_start, first_angle)
        pending_arc_start = None

    if len(output_vertices) == 0:
        if _polygon_contains_point(polygon, np.array([0.0, 0.0], dtype=np.float32)):
            return _full_circle_polygon(radius)
        return np.array([], dtype=np.float32).reshape(0, 2)
    
    return np.array(output_vertices, dtype=np.float32)


def is_valid_polygon(polygon: NDArray[np.float32]) -> bool:
    """
    Check if polygon has sufficient vertices to be valid.
    
    Parameters:
        polygon: Polygon vertices (N, 2)
        
    Returns:
        True if polygon has at least 3 vertices
    """
    return polygon.shape[0] >= 3


def compute_bounding_box(
    polygon: NDArray[np.float32]
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Compute axis-aligned bounding box for polygon.
    
    Parameters:
        polygon: Polygon vertices (N, 2)
        
    Returns:
        Tuple of (min_point, max_point), each shape (2,)
    """
    if polygon.shape[0] == 0:
        raise ValueError("polygon must contain at least one vertex")
    min_point = np.min(polygon, axis=0).astype(np.float32)
    max_point = np.max(polygon, axis=0).astype(np.float32)
    return min_point, max_point


def _polygon_contains_point(
    polygon: NDArray[np.float32],
    point: NDArray[np.float32]
) -> bool:
    """Ray-casting test to check if point lies inside polygon."""
    if polygon.shape[0] < 3:
        return False

    px, py = float(point[0]), float(point[1])
    inside = False
    j = polygon.shape[0] - 1
    for i in range(polygon.shape[0]):
        xi, yi = float(polygon[i, 0]), float(polygon[i, 1])
        xj, yj = float(polygon[j, 0]), float(polygon[j, 1])
        intersects = ((yi > py) != (yj > py))
        if intersects:
            dy = yj - yi
            if abs(dy) < 1e-12:
                dy = 1e-12 if dy >= 0 else -1e-12
            slope = (xj - xi) / dy
            x_at_y = xi + slope * (py - yi)
            if x_at_y > px:
                inside = not inside
        j = i
    return inside


def _full_circle_polygon(radius: float) -> NDArray[np.float32]:
    """Return a polygonal approximation of the full circle."""
    if radius <= 0:
        return np.array([], dtype=np.float32).reshape(0, 2)

    circumference_steps = max(MIN_ARC_SAMPLES, int(np.ceil(2.0 * np.pi / ARC_SAMPLE_STEP)))
    angles = np.linspace(0.0, 2.0 * np.pi, circumference_steps, endpoint=False)
    points = np.stack([
        radius * np.cos(angles),
        radius * np.sin(angles),
    ], axis=1)
    return points.astype(np.float32)
