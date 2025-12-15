"""
Visualization utilities for attention tracking.

This module provides functions to draw overlays showing attention heatmaps
and labels on images, useful for analyzing viewer attention patterns.
"""

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from view_arc.tracking.dataclasses import AOI, TrackingResult

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


def _get_heatmap_color(
    normalized_value: float, colormap: str = "hot"
) -> tuple[int, int, int]:
    """Get BGR color for a normalized value [0, 1] using a colormap.

    Args:
        normalized_value: Value in range [0, 1] where 0 is cold, 1 is hot
        colormap: Color scheme - 'hot' (blue->red) or 'viridis' (purple->yellow)

    Returns:
        BGR color tuple (values 0-255)
    """
    _ensure_cv2()

    # Clamp to [0, 1]
    value = max(0.0, min(1.0, normalized_value))

    if colormap == "hot":
        # Cold (blue) to hot (red) gradient
        # Blue -> Cyan -> Green -> Yellow -> Red
        if value < 0.25:
            # Blue to cyan
            ratio = value / 0.25
            b = 255
            g = int(255 * ratio)
            r = 0
        elif value < 0.5:
            # Cyan to green
            ratio = (value - 0.25) / 0.25
            b = int(255 * (1 - ratio))
            g = 255
            r = 0
        elif value < 0.75:
            # Green to yellow
            ratio = (value - 0.5) / 0.25
            b = 0
            g = 255
            r = int(255 * ratio)
        else:
            # Yellow to red
            ratio = (value - 0.75) / 0.25
            b = 0
            g = int(255 * (1 - ratio))
            r = 255
        return (b, g, r)
    elif colormap == "viridis":
        # Purple to yellow gradient (inspired by matplotlib viridis)
        if value < 0.5:
            # Purple to teal
            ratio = value / 0.5
            b = int(255 * (0.4 + 0.3 * ratio))
            g = int(255 * 0.4 * ratio)
            r = int(255 * 0.3 * (1 - ratio))
        else:
            # Teal to yellow
            ratio = (value - 0.5) / 0.5
            b = int(255 * 0.7 * (1 - ratio))
            g = int(255 * (0.4 + 0.6 * ratio))
            r = int(255 * 0.9 * ratio)
        return (b, g, r)
    else:
        # Default to grayscale
        intensity = int(255 * value)
        return (intensity, intensity, intensity)


def draw_attention_heatmap(
    image: NDArray[np.uint8],
    aois: list[AOI],
    tracking_result: TrackingResult,
    colormap: Literal["hot", "viridis", "grayscale"] = "hot",
    fill_alpha: float = 0.5,
    draw_outlines: bool = True,
    outline_thickness: int = 2,
    background_color: tuple[int, int, int] | None = None,
) -> NDArray[np.uint8]:
    """Draw attention heatmap by coloring AOIs based on hit counts.

    Colors each AOI with a gradient from cold (low attention) to hot (high attention)
    based on the number of hits it received. AOIs with zero hits can optionally be
    drawn with a background color.

    Args:
        image: Input image (H, W, 3) BGR format
        aois: List of AOI objects to visualize
        tracking_result: TrackingResult containing hit counts for each AOI
        colormap: Color scheme - 'hot' (blue->red), 'viridis' (purple->yellow),
                  or 'grayscale'
        fill_alpha: Alpha transparency for AOI fill (0.0 = transparent, 1.0 = opaque)
        draw_outlines: If True, draw AOI outlines
        outline_thickness: Thickness of AOI outlines
        background_color: BGR color for AOIs with zero hits (None = skip drawing them)

    Returns:
        Image with heatmap overlay (modified copy)

    Example:
        >>> result = compute_attention_seconds(samples, aois)
        >>> img = cv2.imread('store.jpg')
        >>> heatmap = draw_attention_heatmap(img, aois, result)
        >>> cv2.imwrite('attention_heatmap.jpg', heatmap)
    """
    _ensure_cv2()

    # Make a copy to avoid modifying the original
    output = image.copy()

    # Find max hit count for normalization
    max_hits = max(
        (tracking_result.aoi_results[aoi.id].hit_count for aoi in aois), default=0
    )

    # Handle case where no AOI has hits
    if max_hits == 0:
        # All AOIs have zero hits - use background color if provided
        if background_color is not None:
            for aoi in aois:
                overlay = output.copy()
                pts = aoi.contour.astype(np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(overlay, [pts], background_color)
                cv2.addWeighted(overlay, fill_alpha, output, 1 - fill_alpha, 0, output)

                if draw_outlines:
                    cv2.polylines(
                        output,
                        [pts],
                        isClosed=True,
                        color=background_color,
                        thickness=outline_thickness,
                    )
        return output

    # Create overlay for transparent fill
    overlay = output.copy()

    # Draw each AOI with its heat color
    for aoi in aois:
        aoi_result = tracking_result.aoi_results.get(aoi.id)
        if aoi_result is None:
            continue

        hit_count = aoi_result.hit_count

        # Determine color based on hit count
        if hit_count == 0:
            if background_color is None:
                continue  # Skip drawing zero-hit AOIs
            color = background_color
        else:
            # Normalize hit count to [0, 1]
            normalized_value = hit_count / max_hits
            color = _get_heatmap_color(normalized_value, colormap)

        # Convert contour to integer coordinates for OpenCV
        pts = aoi.contour.astype(np.int32).reshape((-1, 1, 2))

        # Fill the AOI on overlay
        cv2.fillPoly(overlay, [pts], color)

    # Blend overlay with original image
    cv2.addWeighted(overlay, fill_alpha, output, 1 - fill_alpha, 0, output)

    # Draw outlines on top of the blended image
    if draw_outlines:
        for aoi in aois:
            aoi_result = tracking_result.aoi_results.get(aoi.id)
            if aoi_result is None:
                continue

            hit_count = aoi_result.hit_count

            # Determine outline color
            if hit_count == 0:
                if background_color is None:
                    continue
                color = background_color
            else:
                normalized_value = hit_count / max_hits
                color = _get_heatmap_color(normalized_value, colormap)

            pts = aoi.contour.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                output, [pts], isClosed=True, color=color, thickness=outline_thickness
            )

    return output


def draw_attention_labels(
    image: NDArray[np.uint8],
    aois: list[AOI],
    tracking_result: TrackingResult,
    show_hit_count: bool = True,
    show_percentage: bool = True,
    show_seconds: bool = False,
    font_scale: float = 0.6,
    font_thickness: int = 2,
    text_color: tuple[int, int, int] = (255, 255, 255),
    background_color: tuple[int, int, int] = (0, 0, 0),
    background_alpha: float = 0.7,
) -> NDArray[np.uint8]:
    """Annotate AOIs with hit counts, percentages, and/or attention seconds.

    Draws text labels at the centroid of each AOI showing attention metrics.
    The label background is semi-transparent for better readability.

    Args:
        image: Input image (H, W, 3) BGR format
        aois: List of AOI objects to annotate
        tracking_result: TrackingResult containing hit counts for each AOI
        show_hit_count: If True, show raw hit count
        show_percentage: If True, show percentage of total attention
        show_seconds: If True, show total attention seconds
        font_scale: Font size scale factor
        font_thickness: Font line thickness
        text_color: BGR color for text
        background_color: BGR color for label background
        background_alpha: Transparency of label background (0.0 = transparent, 1.0 = opaque)

    Returns:
        Image with annotation labels (modified copy)

    Example:
        >>> result = compute_attention_seconds(samples, aois)
        >>> img = cv2.imread('store.jpg')
        >>> labeled = draw_attention_labels(img, aois, result, show_percentage=True)
        >>> cv2.imwrite('attention_labels.jpg', labeled)
    """
    _ensure_cv2()

    # Make a copy to avoid modifying the original
    output = image.copy()

    # Calculate total hits for percentage calculation
    total_hits = tracking_result.samples_with_hits
    if total_hits == 0:
        # No hits to display
        return output

    font = cv2.FONT_HERSHEY_SIMPLEX

    for aoi in aois:
        aoi_result = tracking_result.aoi_results.get(aoi.id)
        if aoi_result is None:
            continue

        hit_count = aoi_result.hit_count
        if hit_count == 0:
            continue  # Skip AOIs with no attention

        # Build label text
        label_parts = []
        if show_hit_count:
            label_parts.append(f"{hit_count}")
        if show_percentage:
            percentage = (hit_count / total_hits) * 100
            label_parts.append(f"{percentage:.1f}%")
        if show_seconds:
            seconds = aoi_result.total_attention_seconds
            label_parts.append(f"{seconds:.1f}s")

        label = " | ".join(label_parts)
        if not label:
            continue

        # Compute centroid of AOI
        centroid = aoi.contour.mean(axis=0)
        cx, cy = int(round(centroid[0])), int(round(centroid[1]))

        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )

        # Calculate background rectangle position (centered on centroid)
        padding = 4
        bg_x1 = cx - text_w // 2 - padding
        bg_y1 = cy - text_h // 2 - padding
        bg_x2 = cx + text_w // 2 + padding
        bg_y2 = cy + text_h // 2 + padding + baseline

        # Draw semi-transparent background
        overlay = output.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), background_color, -1)
        cv2.addWeighted(overlay, background_alpha, output, 1 - background_alpha, 0, output)

        # Draw text centered on centroid
        text_x = cx - text_w // 2
        text_y = cy + text_h // 2
        cv2.putText(
            output,
            label,
            (text_x, text_y),
            font,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA,
        )

    return output
