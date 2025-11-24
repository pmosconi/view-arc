"""
Public API interface definitions for obstacle detection within view arc.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray


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
    raise NotImplementedError("Implementation pending")
