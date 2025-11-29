"""
Public API interface definitions for obstacle detection within view arc.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np
from numpy.typing import NDArray

from view_arc.geometry import (
    normalize_angle,
    validate_and_get_direction_angle,
    to_viewer_frame,
    to_polar,
)
from view_arc.clipping import clip_polygon_to_wedge, is_valid_polygon
from view_arc.sweep import build_events, compute_coverage


@dataclass
class ObstacleResult:
    """
    Result of obstacle detection within a view arc.
    
    Attributes:
        obstacle_id: Index or identifier of the winning obstacle (None if no obstacle visible)
        angular_coverage: Total visible angular span in radians
        min_distance: Minimum distance encountered for this obstacle
        intervals: Optional list of (angle_start, angle_end) tuples showing owned angular intervals
    """
    obstacle_id: Optional[int]
    angular_coverage: float
    min_distance: float
    intervals: Optional[List[Tuple[float, float]]] = None
    
    def __bool__(self) -> bool:
        """Returns True if an obstacle was found."""
        return self.obstacle_id is not None


def find_largest_obstacle(
    viewer_point: NDArray[np.float32],
    view_direction: NDArray[np.float32],
    field_of_view_deg: float,
    max_range: float,
    obstacle_contours: List[NDArray[np.float32]],
    return_intervals: bool = False
) -> ObstacleResult:
    """
    Find the obstacle with largest visible angular coverage within a view arc.
    
    Implements radial occlusion: nearer obstacles mask farther ones along the same
    angular direction. Returns the obstacle occupying the largest angular span
    after clipping and occlusion resolution.
    
    Parameters:
        viewer_point: (x, y) coordinates of the viewer position in image space, shape (2,)
        view_direction: (x, y) unit vector representing view direction, shape (2,)
                       Expected to be normalized (length 1). Follows convention:
                       x: first value (positive = RIGHT, negative = LEFT)
                       y: second value (positive = UP, negative = DOWN)
                       Example: [-0.37, 0.92] points up-left
        field_of_view_deg: Total field of view angle in degrees (symmetric around direction)
        max_range: Maximum sensing distance (radius) from viewer point
        obstacle_contours: List of obstacle polygons, each an (N, 2) array of vertices in image coordinates
        return_intervals: If True, include angular interval breakdown in result
        
    Returns:
        ObstacleResult containing winner ID, coverage, distance, and optional intervals
        
    Raises:
        ValueError: If inputs have invalid shapes or values (e.g., view_direction not normalized)
        
    Example:
        >>> viewer = np.array([100.0, 100.0], dtype=np.float32)
        >>> direction_vec = np.array([0.0, 1.0], dtype=np.float32)  # pointing UP
        >>> contours = [
        ...     np.array([[90, 150], [110, 150], [100, 170]], dtype=np.float32),
        ...     np.array([[80, 200], [120, 200], [100, 230]], dtype=np.float32)
        ... ]
        >>> result = find_largest_obstacle(viewer, direction_vec, 30.0, 150.0, contours)
        >>> print(f"Winner: obstacle {result.obstacle_id}, coverage: {result.angular_coverage:.2f} rad")
    """
    # -------------------------------------------------------------------------
    # Step 1: Input validation
    # -------------------------------------------------------------------------
    viewer_point = np.asarray(viewer_point, dtype=np.float32)
    view_direction = np.asarray(view_direction, dtype=np.float32)
    
    if viewer_point.shape != (2,):
        raise ValueError(f"viewer_point must have shape (2,), got {viewer_point.shape}")
    
    if view_direction.shape != (2,):
        raise ValueError(f"view_direction must have shape (2,), got {view_direction.shape}")
    
    # Validate view_direction is normalized and get central angle
    alpha_center = validate_and_get_direction_angle(view_direction)
    
    # Validate field of view
    if field_of_view_deg <= 0 or field_of_view_deg > 360:
        raise ValueError(f"field_of_view_deg must be in (0, 360], got {field_of_view_deg}")
    
    # Validate max_range
    if max_range <= 0:
        raise ValueError(f"max_range must be positive, got {max_range}")
    
    # Validate obstacle_contours
    if not isinstance(obstacle_contours, list):
        raise ValueError("obstacle_contours must be a list of numpy arrays")
    
    for i, contour in enumerate(obstacle_contours):
        if not isinstance(contour, np.ndarray):
            raise ValueError(f"obstacle_contours[{i}] must be a numpy array")
        if contour.ndim != 2 or contour.shape[1] != 2:
            raise ValueError(
                f"obstacle_contours[{i}] must have shape (N, 2), got {contour.shape}"
            )
    
    # -------------------------------------------------------------------------
    # Step 2: Compute arc boundaries
    # -------------------------------------------------------------------------
    # Special case: full-circle FOV (360° or very close to it)
    # Use epsilon of 1e-6 degrees to catch floating point issues
    FULL_CIRCLE_EPSILON = 1e-6
    is_full_circle = field_of_view_deg >= 360.0 - FULL_CIRCLE_EPSILON
    
    if is_full_circle:
        # For full circle, use sentinel values that span the entire angle range
        # alpha_min = -π, alpha_max = π signals "full circle" to clipping and sweep
        alpha_min = -np.pi
        alpha_max = np.pi
    else:
        half_fov_rad = np.deg2rad(field_of_view_deg) / 2.0
        alpha_min = alpha_center - half_fov_rad
        alpha_max = alpha_center + half_fov_rad
        
        # Normalize boundaries to [-π, π) so sweep helpers can correctly detect
        # wraparound via alpha_min > alpha_max
        alpha_min = normalize_angle(alpha_min)
        alpha_max = normalize_angle(alpha_max)
    
    # -------------------------------------------------------------------------
    # Step 3: Transform contours to viewer-centric frame and clip
    # -------------------------------------------------------------------------
    clipped_polygons: List[Optional[NDArray[np.float32]]] = []
    obstacle_edges: Dict[int, NDArray[np.float32]] = {}
    
    for obstacle_id, contour in enumerate(obstacle_contours):
        # Transform to viewer frame
        contour_viewer = to_viewer_frame(contour.astype(np.float32), viewer_point)
        
        # Skip invalid polygons
        if not is_valid_polygon(contour_viewer):
            clipped_polygons.append(None)
            continue
        
        # Clip to wedge (half-planes + circle)
        clipped = clip_polygon_to_wedge(
            contour_viewer,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            max_range=max_range
        )
        
        clipped_polygons.append(clipped)
        
        # Build edges array for sweep if clipped polygon is valid
        if clipped is not None and len(clipped) >= 3:
            n_verts = len(clipped)
            edges = np.zeros((n_verts, 2, 2), dtype=np.float32)
            for i in range(n_verts):
                edges[i, 0] = clipped[i]
                edges[i, 1] = clipped[(i + 1) % n_verts]
            obstacle_edges[obstacle_id] = edges
    
    # -------------------------------------------------------------------------
    # Step 4: Build events and compute coverage via angular sweep
    # -------------------------------------------------------------------------
    # Filter to only valid clipped polygons for event building
    valid_clipped = [p for p in clipped_polygons if p is not None]
    
    if not valid_clipped:
        # No obstacles visible in the arc
        return ObstacleResult(
            obstacle_id=None,
            angular_coverage=0.0,
            min_distance=float('inf'),
            intervals=[] if return_intervals else None
        )
    
    # Build events from clipped polygons
    # Need to pass with original obstacle IDs preserved
    # Create a list of tuples (obstacle_id, clipped_polygon) for valid polygons only
    valid_polygons_with_ids = [
        (i, polygon) for i, polygon in enumerate(clipped_polygons) if polygon is not None
    ]
    events = build_events([p for _, p in valid_polygons_with_ids], alpha_min, alpha_max)
    
    # Compute coverage via angular sweep
    coverage_dict, min_distance_dict, intervals = compute_coverage(
        events=events,
        obstacle_edges=obstacle_edges,
        alpha_min=alpha_min,
        alpha_max=alpha_max
    )
    
    # -------------------------------------------------------------------------
    # Step 5: Find winning obstacle (largest coverage, tie-break by distance)
    # -------------------------------------------------------------------------
    if not coverage_dict:
        # No obstacle covered any angular span
        return ObstacleResult(
            obstacle_id=None,
            angular_coverage=0.0,
            min_distance=float('inf'),
            intervals=[] if return_intervals else None
        )
    
    # Find obstacle with maximum coverage
    best_obstacle_id: Optional[int] = None
    best_coverage: float = 0.0
    best_distance: float = float('inf')
    
    for obstacle_id, coverage in coverage_dict.items():
        distance = min_distance_dict.get(obstacle_id, float('inf'))
        
        # Primary: largest coverage
        # Secondary (tie-break): smallest distance
        if coverage > best_coverage or (coverage == best_coverage and distance < best_distance):
            best_obstacle_id = obstacle_id
            best_coverage = coverage
            best_distance = distance
    
    # -------------------------------------------------------------------------
    # Step 6: Build result
    # -------------------------------------------------------------------------
    result_intervals: Optional[List[Tuple[float, float]]] = None
    
    if return_intervals and best_obstacle_id is not None:
        # Filter intervals for the winning obstacle
        result_intervals = [
            (interval.angle_start, interval.angle_end)
            for interval in intervals
            if interval.obstacle_id == best_obstacle_id
        ]
    
    return ObstacleResult(
        obstacle_id=best_obstacle_id,
        angular_coverage=best_coverage,
        min_distance=best_distance,
        intervals=result_intervals
    )
