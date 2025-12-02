"""
Geometry utilities for coordinate transforms, polar conversion, and ray intersection.
"""

from typing import Tuple, Optional
import numpy as np
from numpy.typing import NDArray


def normalize_angle(angle: float) -> float:
    """
    Wrap angle to [-π, π) range.
    
    Parameters:
        angle: Angle in radians
        
    Returns:
        Normalized angle in [-π, π)
    """
    # Use modulo to wrap to [0, 2π) then shift to [-π, π)
    result = np.fmod(angle + np.pi, 2 * np.pi)
    if result < 0:
        result += 2 * np.pi
    return float(result - np.pi)


def to_viewer_frame(
    points: NDArray[np.float32],
    viewer_origin: NDArray[np.float32]
) -> NDArray[np.float32]:
    """
    Translate points to viewer-centric coordinate frame.
    
    Parameters:
        points: Array of shape (N, 2) or (2,) containing (x, y) coordinates
        viewer_origin: Viewer position (2,) to become the new origin
        
    Returns:
        Translated points in same shape as input
    """
    return points - viewer_origin


def to_polar(
    points: NDArray[np.float32]
) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Convert Cartesian coordinates to polar (r, alpha).
    
    Parameters:
        points: Array of shape (N, 2) or (2,) containing (x, y) coordinates
        
    Returns:
        Tuple of (radii, angles) where:
            radii: shape (N,) or scalar - distances from origin
            angles: shape (N,) or scalar - angles in radians [-π, π)
    """
    points = np.asarray(points, dtype=np.float32)
    
    # Handle both single point (2,) and multiple points (N, 2)
    if points.ndim == 1:
        x, y = points[0], points[1]
    else:
        x, y = points[:, 0], points[:, 1]
    
    radii = np.sqrt(x**2 + y**2).astype(np.float32)
    angles = np.arctan2(y, x).astype(np.float32)
    
    return radii, angles


def validate_and_get_direction_angle(
    view_direction: NDArray[np.float32],
    tolerance: float = 1e-3
) -> float:
    """
    Validate that view direction is normalized and compute its angle.
    
    Parameters:
        view_direction: Unit vector (2,) representing view direction
        tolerance: Tolerance for normalization check (|length - 1| < tolerance)
        
    Returns:
        Angle in radians corresponding to the direction
        
    Raises:
        ValueError: If view_direction is not approximately unit length
    """
    view_direction = np.asarray(view_direction, dtype=np.float32)
    
    length = np.linalg.norm(view_direction)
    if abs(length - 1.0) >= tolerance:
        raise ValueError(
            f"view_direction must be a unit vector (length=1), "
            f"got length={length:.6f}"
        )
    
    return float(np.arctan2(view_direction[1], view_direction[0]))


def intersect_ray_segment(
    ray_angle: float,
    segment_start: NDArray[np.float32],
    segment_end: NDArray[np.float32],
    max_range: float
) -> Optional[float]:
    """
    Compute intersection of a ray from origin with a line segment.
    
    Uses parametric ray-segment intersection:
    Ray: P = r * (cos(θ), sin(θ))  for r ≥ 0
    Segment: Q = A + t * (B - A)   for t ∈ [0, 1]
    
    Parameters:
        ray_angle: Angle of ray in radians
        segment_start: Start point (2,) in viewer frame
        segment_end: End point (2,) in viewer frame
        max_range: Maximum valid distance
        
    Returns:
        Distance r along ray to intersection, or None if no valid intersection
        Valid intersections satisfy: 0 < r <= max_range and lie within segment
    """
    segment_start = np.asarray(segment_start, dtype=np.float32)
    segment_end = np.asarray(segment_end, dtype=np.float32)
    
    # Ray direction vector
    d = np.array([np.cos(ray_angle), np.sin(ray_angle)], dtype=np.float32)
    
    # Segment direction vector: v = B - A
    v = segment_end - segment_start
    
    # We want to solve: r * d = A + t * v
    # Rearranging: r * d - t * v = A
    # 
    # In matrix form: [d_x, -v_x] [r]   [A_x]
    #                 [d_y, -v_y] [t] = [A_y]
    #
    # Using Cramer's rule:
    # det = d_x * (-v_y) - d_y * (-v_x) = -d_x * v_y + d_y * v_x = d × (-v)
    # We'll use: det = d_x * v_y - d_y * v_x = -(d × v) but with opposite sign convention
    
    # Cross product 2D: a × b = a_x * b_y - a_y * b_x
    # det(M) where M = [d, -v] = d_x * (-v_y) - d_y * (-v_x) = v_x * d_y - v_y * d_x
    det = v[0] * d[1] - v[1] * d[0]
    
    # Check if ray and segment are parallel
    if abs(det) < 1e-10:
        return None
    
    # A = segment_start
    A = segment_start
    
    # Using Cramer's rule to solve for r and t:
    # r = det([A, -v]) / det([d, -v])
    # t = det([d, A]) / det([d, -v])
    #
    # det([A, -v]) = A_x * (-v_y) - A_y * (-v_x) = -A_x * v_y + A_y * v_x = A × (-v)
    # det([d, A]) = d_x * A_y - d_y * A_x
    
    r = (A[1] * v[0] - A[0] * v[1]) / det
    t = (A[1] * d[0] - A[0] * d[1]) / det
    
    # Check if intersection is within segment bounds [0, 1]
    if t < 0.0 or t > 1.0:
        return None
    
    # Check if intersection is in front of ray and within max range
    if r <= 0.0 or r > max_range:
        return None
    
    return float(r)


def handle_angle_discontinuity(
    angles: NDArray[np.float32],
    alpha_min: float,
    alpha_max: float
) -> NDArray[np.float32]:
    """
    Remap angles when field of view crosses ±π boundary.
    
    If the arc crosses the discontinuity (alpha_max > alpha_min means no crossing,
    alpha_max < alpha_min means the arc wraps around ±π), adds 2π to angles 
    below alpha_max to create a continuous range for sweep processing.
    
    Parameters:
        angles: Array of angles in [-π, π)
        alpha_min: Minimum arc angle (left edge of arc)
        alpha_max: Maximum arc angle (right edge of arc)
        
    Returns:
        Remapped angles for continuous processing
    """
    angles = np.asarray(angles, dtype=np.float32).copy()
    
    # If alpha_min <= alpha_max, the arc doesn't cross the discontinuity
    if alpha_min <= alpha_max:
        return angles
    
    # Arc crosses ±π boundary (e.g., arc from 170° to -170° = 340°)
    # We need to remap angles that are in the negative part to be > π
    # Angles <= alpha_max should have 2π added to make them continuous
    # Using <= for inclusive boundary semantics
    mask = angles <= alpha_max
    angles[mask] += 2 * np.pi
    
    return angles
