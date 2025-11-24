"""
Visualization utilities for debugging and validation.
"""

from typing import List, Optional
import numpy as np
from numpy.typing import NDArray


def draw_wedge_overlay(
    image: NDArray[np.uint8],
    viewer_point: NDArray[np.float32],
    view_direction: NDArray[np.float32],
    field_of_view_deg: float,
    max_range: float,
    color: tuple = (0, 255, 0),
    thickness: int = 2
) -> NDArray[np.uint8]:
    """
    Draw the field-of-view wedge on an image.
    
    Parameters:
        image: Input image (H, W, 3) BGR format
        viewer_point: Viewer position (2,)
        view_direction: Unit vector (2,) representing view direction
        field_of_view_deg: Field of view in degrees
        max_range: Maximum range radius
        color: BGR color tuple
        thickness: Line thickness
        
    Returns:
        Image with wedge overlay
    """
    raise NotImplementedError


def draw_obstacle_contours(
    image: NDArray[np.uint8],
    contours: List[NDArray[np.float32]],
    winner_id: Optional[int] = None,
    default_color: tuple = (255, 0, 0),
    winner_color: tuple = (0, 0, 255),
    thickness: int = 2
) -> NDArray[np.uint8]:
    """
    Draw obstacle contours, highlighting the winner.
    
    Parameters:
        image: Input image (H, W, 3) BGR format
        contours: List of obstacle contours
        winner_id: Index of winning obstacle to highlight
        default_color: BGR color for normal obstacles
        winner_color: BGR color for winning obstacle
        thickness: Line thickness
        
    Returns:
        Image with contour overlays
    """
    raise NotImplementedError


def draw_angular_intervals(
    image: NDArray[np.uint8],
    viewer_point: NDArray[np.float32],
    intervals: List[tuple],
    max_range: float,
    color: tuple = (255, 255, 0),
    thickness: int = 1
) -> NDArray[np.uint8]:
    """
    Draw angular interval rays for visualization.
    
    Parameters:
        image: Input image (H, W, 3) BGR format
        viewer_point: Viewer position (2,)
        intervals: List of (angle_start, angle_end) tuples in radians
        max_range: Ray length
        color: BGR color tuple
        thickness: Line thickness
        
    Returns:
        Image with interval rays
    """
    raise NotImplementedError
