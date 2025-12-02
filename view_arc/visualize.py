"""
Visualization utilities for debugging and validation.

This module provides functions to draw overlays on images using OpenCV,
useful for debugging the obstacle detection algorithm.
"""

from typing import List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

# Try to import cv2, set flag if not available
try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def _ensure_cv2() -> None:
    """Raise an error if cv2 is not available."""
    if not HAS_CV2:
        raise ImportError(
            "OpenCV (cv2) is required for visualization functions. "
            "Install with: pip install opencv-python"
        )


def draw_wedge_overlay(
    image: NDArray[np.uint8],
    viewer_point: NDArray[np.float32],
    view_direction: NDArray[np.float32],
    field_of_view_deg: float,
    max_range: float,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    fill_alpha: float = 0.0,
) -> NDArray[np.uint8]:
    """
    Draw the field-of-view wedge on an image.

    The wedge is drawn as two boundary rays from the viewer position and an arc
    connecting their endpoints at max_range distance. Optionally fills the wedge
    with a semi-transparent overlay.

    Parameters:
        image: Input image (H, W, 3) BGR format
        viewer_point: Viewer position (x, y), shape (2,)
        view_direction: Unit vector (x, y) representing view direction, shape (2,)
        field_of_view_deg: Field of view in degrees (symmetric around direction)
        max_range: Maximum range radius in pixels
        color: BGR color tuple for the wedge outline
        thickness: Line thickness for the outline
        fill_alpha: Alpha for semi-transparent fill (0.0 = no fill, 1.0 = solid)

    Returns:
        Image with wedge overlay (modified copy)

    Note:
        Image coordinate system: x increases rightward, y increases downward.
        The view_direction follows mathematical convention: (1,0) = right, (0,1) = down.
    """
    _ensure_cv2()

    # Make a copy to avoid modifying the original
    output = image.copy()

    # Convert to integer coordinates
    vx, vy = int(round(viewer_point[0])), int(round(viewer_point[1]))

    # Compute central angle from direction vector
    alpha_center = np.arctan2(view_direction[1], view_direction[0])

    # Compute arc boundary angles
    half_fov = np.deg2rad(field_of_view_deg) / 2
    alpha_min = alpha_center - half_fov
    alpha_max = alpha_center + half_fov

    # Compute endpoints of boundary rays
    x1_end = int(round(vx + max_range * np.cos(alpha_min)))
    y1_end = int(round(vy + max_range * np.sin(alpha_min)))
    x2_end = int(round(vx + max_range * np.cos(alpha_max)))
    y2_end = int(round(vy + max_range * np.sin(alpha_max)))

    # Draw the wedge fill if requested
    if fill_alpha > 0:
        # Create a mask with the wedge shape
        overlay = output.copy()

        # Generate arc points for the fill
        n_arc_points = max(3, int(field_of_view_deg / 2))
        arc_angles = np.linspace(alpha_min, alpha_max, n_arc_points)
        arc_points = np.array(
            [
                [
                    int(round(vx + max_range * np.cos(a))),
                    int(round(vy + max_range * np.sin(a))),
                ]
                for a in arc_angles
            ],
            dtype=np.int32,
        )

        # Create polygon points: viewer + arc points
        wedge_points = np.vstack([[[vx, vy]], arc_points])
        cv2.fillPoly(overlay, [wedge_points], color)

        # Blend with original
        cv2.addWeighted(overlay, fill_alpha, output, 1 - fill_alpha, 0, output)

    # Draw boundary rays
    cv2.line(output, (vx, vy), (x1_end, y1_end), color, thickness)
    cv2.line(output, (vx, vy), (x2_end, y2_end), color, thickness)

    # Draw the arc using ellipse
    # OpenCV ellipse uses degrees, starting from positive x-axis (3 o'clock), counter-clockwise
    # But since y increases downward in image coords, angles appear mirrored
    start_angle = np.rad2deg(alpha_min)
    end_angle = np.rad2deg(alpha_max)

    # Ensure angles are in the right order for cv2.ellipse
    if end_angle < start_angle:
        end_angle += 360

    cv2.ellipse(
        output,
        (vx, vy),
        (int(round(max_range)), int(round(max_range))),
        0,  # rotation
        start_angle,
        end_angle,
        color,
        thickness,
    )

    # Draw viewer point marker
    cv2.circle(output, (vx, vy), 5, color, -1)

    return output


def draw_obstacle_contours(
    image: NDArray[np.uint8],
    contours: List[NDArray[np.float32]],
    winner_id: Optional[int] = None,
    default_color: Tuple[int, int, int] = (255, 0, 0),
    winner_color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
    show_labels: bool = False,
) -> NDArray[np.uint8]:
    """
    Draw obstacle contours, highlighting the winner.

    Parameters:
        image: Input image (H, W, 3) BGR format
        contours: List of obstacle contours, each an (N, 2) array of vertices
        winner_id: Index of winning obstacle to highlight (None for no highlight)
        default_color: BGR color for normal obstacles
        winner_color: BGR color for winning obstacle
        thickness: Line thickness
        show_labels: If True, draw obstacle index labels at centroids

    Returns:
        Image with contour overlays (modified copy)
    """
    _ensure_cv2()

    output = image.copy()

    for idx, contour in enumerate(contours):
        if contour is None or len(contour) < 3:
            continue

        # Convert to integer coordinates for OpenCV
        pts = contour.astype(np.int32).reshape((-1, 1, 2))

        # Choose color based on winner status
        color = winner_color if idx == winner_id else default_color

        # Draw the contour
        cv2.polylines(output, [pts], isClosed=True, color=color, thickness=thickness)

        # Optionally draw labels
        if show_labels:
            # Compute centroid
            centroid = contour.mean(axis=0)
            cx, cy = int(round(centroid[0])), int(round(centroid[1]))

            # Draw label background
            label = str(idx)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            label_thickness = 2

            (text_w, text_h), _ = cv2.getTextSize(
                label, font, font_scale, label_thickness
            )
            cv2.rectangle(
                output, (cx - 2, cy - text_h - 2), (cx + text_w + 2, cy + 2), color, -1
            )
            cv2.putText(
                output,
                label,
                (cx, cy),
                font,
                font_scale,
                (255, 255, 255),
                label_thickness,
            )

    return output


def draw_angular_intervals(
    image: NDArray[np.uint8],
    viewer_point: NDArray[np.float32],
    intervals: List[Tuple[float, float]],
    max_range: float,
    color: Tuple[int, int, int] = (255, 255, 0),
    thickness: int = 1,
    draw_boundary_rays: bool = True,
    fill_alpha: float = 0.0,
) -> NDArray[np.uint8]:
    """
    Draw angular interval rays for visualization.

    Each interval is rendered as a shaded wedge or pair of boundary rays showing
    the angular span occupied by obstacles.

    Parameters:
        image: Input image (H, W, 3) BGR format
        viewer_point: Viewer position (x, y), shape (2,)
        intervals: List of (angle_start, angle_end) tuples in radians
        max_range: Ray length in pixels
        color: BGR color tuple
        thickness: Line thickness for rays
        draw_boundary_rays: If True, draw rays at interval boundaries
        fill_alpha: Alpha for interval fill (0.0 = no fill)

    Returns:
        Image with interval rays (modified copy)
    """
    _ensure_cv2()

    output = image.copy()

    vx, vy = int(round(viewer_point[0])), int(round(viewer_point[1]))

    for angle_start, angle_end in intervals:
        # Handle wrap-around for angles crossing ±π
        if angle_end < angle_start:
            # This interval crosses ±π, need to handle differently
            # For now, just ensure angles are in order
            angle_end += 2 * np.pi

        # Draw filled wedge if requested
        if fill_alpha > 0:
            overlay = output.copy()

            # Generate arc points
            arc_span = angle_end - angle_start
            n_points = max(3, int(np.rad2deg(arc_span) / 5) + 1)
            angles = np.linspace(angle_start, angle_end, n_points)

            arc_points = np.array(
                [
                    [
                        int(round(vx + max_range * np.cos(a))),
                        int(round(vy + max_range * np.sin(a))),
                    ]
                    for a in angles
                ],
                dtype=np.int32,
            )

            wedge_points = np.vstack([[[vx, vy]], arc_points])
            cv2.fillPoly(overlay, [wedge_points], color)
            cv2.addWeighted(overlay, fill_alpha, output, 1 - fill_alpha, 0, output)

        # Draw boundary rays
        if draw_boundary_rays:
            x_start = int(round(vx + max_range * np.cos(angle_start)))
            y_start = int(round(vy + max_range * np.sin(angle_start)))
            x_end = int(round(vx + max_range * np.cos(angle_end)))
            y_end = int(round(vy + max_range * np.sin(angle_end)))

            cv2.line(output, (vx, vy), (x_start, y_start), color, thickness)
            cv2.line(output, (vx, vy), (x_end, y_end), color, thickness)

            # Draw arc connecting the endpoints
            start_angle_deg = np.rad2deg(angle_start)
            end_angle_deg = np.rad2deg(angle_end)

            cv2.ellipse(
                output,
                (vx, vy),
                (int(round(max_range)), int(round(max_range))),
                0,
                start_angle_deg,
                end_angle_deg,
                color,
                thickness,
            )

    return output


def draw_complete_visualization(
    image: NDArray[np.uint8],
    viewer_point: NDArray[np.float32],
    view_direction: NDArray[np.float32],
    field_of_view_deg: float,
    max_range: float,
    obstacle_contours: List[NDArray[np.float32]],
    winner_id: Optional[int] = None,
    intervals: Optional[List[Tuple[float, float]]] = None,
    wedge_color: Tuple[int, int, int] = (0, 255, 0),
    default_obstacle_color: Tuple[int, int, int] = (255, 0, 0),
    winner_obstacle_color: Tuple[int, int, int] = (0, 0, 255),
    interval_color: Tuple[int, int, int] = (255, 255, 0),
) -> NDArray[np.uint8]:
    """
    Draw a complete visualization with wedge, obstacles, and intervals.

    This is a convenience function that combines all visualization elements
    into a single output image.

    Parameters:
        image: Input image (H, W, 3) BGR format
        viewer_point: Viewer position (x, y), shape (2,)
        view_direction: Unit vector (x, y) representing view direction
        field_of_view_deg: Field of view in degrees
        max_range: Maximum range radius in pixels
        obstacle_contours: List of obstacle contours
        winner_id: Index of winning obstacle (None for no highlight)
        intervals: Optional list of (angle_start, angle_end) tuples
        wedge_color: BGR color for the FOV wedge
        default_obstacle_color: BGR color for non-winning obstacles
        winner_obstacle_color: BGR color for winning obstacle
        interval_color: BGR color for interval rays

    Returns:
        Image with complete visualization overlay
    """
    _ensure_cv2()

    output = image.copy()

    # Draw FOV wedge first (background layer)
    output = draw_wedge_overlay(
        output,
        viewer_point,
        view_direction,
        field_of_view_deg,
        max_range,
        color=wedge_color,
        fill_alpha=0.1,
    )

    # Draw intervals if provided
    if intervals:
        output = draw_angular_intervals(
            output,
            viewer_point,
            intervals,
            max_range * 0.95,  # Slightly shorter to not overlap with wedge
            color=interval_color,
            fill_alpha=0.15,
        )

    # Draw obstacle contours on top
    output = draw_obstacle_contours(
        output,
        obstacle_contours,
        winner_id=winner_id,
        default_color=default_obstacle_color,
        winner_color=winner_obstacle_color,
        show_labels=True,
    )

    return output
