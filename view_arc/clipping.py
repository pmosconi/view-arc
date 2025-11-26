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
    1. Half-plane at alpha_min
    2. Half-plane at alpha_max  
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
    raise NotImplementedError


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
    raise NotImplementedError


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
