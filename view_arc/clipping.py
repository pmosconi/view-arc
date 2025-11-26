"""
Polygon clipping operations for sector wedge (half-planes + circle).
"""

from typing import List, Optional
import numpy as np
from numpy.typing import NDArray

HALFPLANE_EPSILON = 1e-6


def clip_polygon_to_wedge(
    polygon: NDArray[np.float32],
    alpha_min: float,
    alpha_max: float,
    max_range: float
) -> Optional[NDArray[np.float32]]:
    """
    Clip polygon against circular sector wedge.
    
    Applies three clipping stages:
    1. Half-plane at alpha_min (keep left/CCW side)
    2. Half-plane at alpha_max (keep right/CW side)
    3. Circle at radius max_range
    
    Parameters:
        polygon: Polygon vertices (N, 2) in viewer-centric frame
        alpha_min: Minimum angle of wedge in radians
        alpha_max: Maximum angle of wedge in radians
        max_range: Radius of circular boundary
        
    Returns:
        Clipped polygon vertices (M, 2) or None if completely clipped away
        Returns None for degenerate results (<3 vertices)
    """
    if not is_valid_polygon(polygon):
        return None
    
    # Stage 1: Clip by half-plane at alpha_min (keep left = CCW from alpha_min ray)
    result = clip_polygon_halfplane(polygon, plane_angle=alpha_min, keep_left=True)
    if result.shape[0] < 3:
        return None
    
    # Stage 2: Clip by half-plane at alpha_max (keep right = CW from alpha_max ray)
    result = clip_polygon_halfplane(result, plane_angle=alpha_max, keep_left=False)
    if result.shape[0] < 3:
        return None
    
    # Stage 3: Clip by circle at max_range
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
    
    def is_inside(point: NDArray[np.float32]) -> bool:
        """Check if point is inside or on the circle."""
        return float(np.dot(point, point)) <= radius_sq + HALFPLANE_EPSILON
    
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
                # Take the first intersection (closest to current)
                output_vertices.append(intersections[0][1])
        elif not current_inside and next_inside:
            # Edge enters the circle - add the entry intersection
            if intersections:
                # Take the last intersection (closest to next)
                output_vertices.append(intersections[-1][1])
        elif not current_inside and not next_inside:
            # Both outside - check if edge passes through circle
            if len(intersections) == 2:
                # Edge crosses circle twice (enters and exits)
                output_vertices.append(intersections[0][1])
                output_vertices.append(intersections[1][1])
    
    if len(output_vertices) == 0:
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
