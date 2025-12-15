"""
Visualization utilities for attention tracking.

This module provides functions to draw overlays showing attention heatmaps
and labels on images, useful for analyzing viewer attention patterns.
"""

from typing import Literal

import math

import numpy as np
from numpy.typing import NDArray

from view_arc.tracking.dataclasses import AOI, TrackingResult

Color = tuple[int, int, int]

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
    # Use .get() to safely handle cases where AOI list doesn't match result dict
    # (e.g., when visualizing filtered or extended AOI lists)
    max_hits = max(
        (
            tracking_result.aoi_results.get(aoi.id, None)
            for aoi in aois
            if tracking_result.aoi_results.get(aoi.id) is not None
        ),
        default=None,
        key=lambda r: r.hit_count if r is not None else 0,
    )
    max_hits_value = max_hits.hit_count if max_hits is not None else 0

    # Handle case where no AOI has hits
    if max_hits_value == 0:
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
            normalized_value = hit_count / max_hits_value
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
                normalized_value = hit_count / max_hits_value
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


def _build_sample_winners(tracking_result: TrackingResult) -> list[str | int | None]:
    """Create a per-sample winner list from TrackingResult hit timestamps."""

    winners: list[str | int | None] = [None] * tracking_result.total_samples
    for result in tracking_result.aoi_results.values():
        for sample_index in result.hit_timestamps:
            if 0 <= sample_index < tracking_result.total_samples:
                winners[sample_index] = result.aoi_id
    return winners


def _generate_color_palette(count: int) -> list[Color]:
    """Generate visually distinct colors using HSV sampling."""

    _ensure_cv2()
    if count <= 0:
        return []

    hsv = np.zeros((count, 1, 3), dtype=np.uint8)
    for idx in range(count):
        hue = int(round((180 / max(count, 1)) * idx)) % 180
        hsv[idx, 0, 0] = hue
        hsv[idx, 0, 1] = 200
        hsv[idx, 0, 2] = 255

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    colors: list[Color] = []
    for idx in range(count):
        pixel = bgr[idx, 0]
        color: Color = (int(pixel[0]), int(pixel[1]), int(pixel[2]))
        colors.append(color)
    return colors


def _resolve_aoi_colors(
    aoi_ids: list[str | int],
    preferred_colors: dict[str | int, Color] | None,
) -> dict[str | int, Color]:
    """Return a mapping of AOI IDs to BGR colors, honoring preferred colors."""

    color_map: dict[str | int, Color] = {}
    preferred_colors = preferred_colors or {}

    missing_ids: list[str | int] = []
    for aoi_id in aoi_ids:
        color = preferred_colors.get(aoi_id)
        if color is None:
            missing_ids.append(aoi_id)
            continue

        if len(color) != 3:
            raise ValueError(
                f"Preferred color for AOI '{aoi_id}' must have 3 components, got {len(color)}"
            )
        normalized: Color = (
            int(max(0, min(255, color[0]))),
            int(max(0, min(255, color[1]))),
            int(max(0, min(255, color[2]))),
        )
        color_map[aoi_id] = normalized

    auto_colors = _generate_color_palette(len(missing_ids))
    for aoi_id, color in zip(missing_ids, auto_colors, strict=False):
        color_map[aoi_id] = color

    return color_map


def _compute_sample_boundaries(total_samples: int, width: int) -> list[tuple[int, int]]:
    """Compute pixel boundaries for each sample along the timeline width."""

    if total_samples <= 0 or width <= 0:
        return []

    px_per_sample = width / total_samples
    boundaries: list[tuple[int, int]] = []
    for sample_idx in range(total_samples):
        start_x = int(round(sample_idx * px_per_sample))
        end_x = int(round((sample_idx + 1) * px_per_sample))
        if end_x <= start_x:
            end_x = start_x + 1
        start_x = max(0, min(start_x, width - 1))
        end_x = max(start_x + 1, min(end_x, width))
        boundaries.append((start_x, end_x))

    # Ensure the final segment touches the right edge exactly
    boundaries[-1] = (boundaries[-1][0], width)
    return boundaries


def _compute_legend_height(
    *, aoi_count: int, show_legend: bool, height: int, legend_columns: int, padding: int
) -> int:
    """Determine how much vertical space the legend should use."""

    if not show_legend or aoi_count == 0:
        return 0

    rows = math.ceil(aoi_count / max(1, legend_columns))
    base_height = max(40, rows * 24 + 2 * padding)
    max_available = max(0, height - 40)
    if max_available == 0:
        return 0
    return min(base_height, max_available)


def _initialize_timeline_canvas(
    *,
    tracking_result: TrackingResult,
    width: int,
    height: int,
    background_color: Color,
    gap_color: Color,
    aoi_colors: dict[str | int, Color] | None,
    show_legend: bool,
    legend_columns: int,
    timeline_padding: int,
) -> tuple[
    NDArray[np.uint8],
    list[str | int | None],
    dict[str | int, Color],
    list[tuple[int, int]],
    int,
    int,
    int,
    int,
]:
    """Prepare the base canvas and derived metadata for timeline rendering."""

    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive integers")

    winners = _build_sample_winners(tracking_result)
    color_map = _resolve_aoi_colors(list(tracking_result.aoi_results.keys()), aoi_colors)
    legend_height = _compute_legend_height(
        aoi_count=len(color_map),
        show_legend=show_legend,
        height=height,
        legend_columns=legend_columns,
        padding=timeline_padding,
    )

    timeline_area_height = max(20, height - legend_height)
    bar_height = max(8, timeline_area_height - 2 * timeline_padding)
    if bar_height > timeline_area_height:
        bar_height = timeline_area_height

    timeline_top = max(0, (timeline_area_height - bar_height) // 2)
    timeline_bottom = timeline_top + bar_height
    legend_y_start = timeline_area_height

    base_canvas = np.full((height, width, 3), background_color, dtype=np.uint8)
    sample_boundaries = _compute_sample_boundaries(len(winners), width)

    return (
        base_canvas,
        winners,
        color_map,
        sample_boundaries,
        timeline_top,
        timeline_bottom,
        legend_y_start,
        legend_height,
    )


def _fill_timeline_bar(
    canvas: NDArray[np.uint8],
    winners: list[str | int | None],
    sample_boundaries: list[tuple[int, int]],
    color_map: dict[str | int, Color],
    gap_color: Color,
    timeline_top: int,
    timeline_bottom: int,
    processed_samples: int,
    future_color: Color | None = None,
) -> None:
    """Fill the timeline bar with AOI colors for processed samples."""

    if not sample_boundaries:
        return

    bar_bottom = max(timeline_top, min(timeline_bottom - 1, canvas.shape[0] - 1))
    if bar_bottom < timeline_top:
        return

    total_samples = len(sample_boundaries)
    processed_samples = max(0, min(processed_samples, total_samples))

    for idx, (start_x, end_x) in enumerate(sample_boundaries):
        if idx >= processed_samples:
            if future_color is None:
                continue
            color = future_color
        else:
            winner = winners[idx]
            color = color_map.get(winner, gap_color) if winner is not None else gap_color

        x1 = max(0, min(start_x, canvas.shape[1] - 1))
        x2 = max(x1, min(end_x - 1, canvas.shape[1] - 1))
        cv2.rectangle(canvas, (x1, timeline_top), (x2, bar_bottom), color, -1)


def _draw_no_samples_message(
    canvas: NDArray[np.uint8],
    timeline_top: int,
    timeline_bottom: int,
    font_scale: float,
    font_thickness: int,
) -> None:
    """Draw a friendly message when there are no samples to visualize."""

    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "No samples to visualize"
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    y_center = timeline_top + max(0, (timeline_bottom - timeline_top) // 2)
    text_x = max(8, (canvas.shape[1] - text_w) // 2)
    text_y = max(text_h + 8, y_center)
    text_y = min(text_y, max(timeline_bottom, text_h + 8))
    cv2.putText(
        canvas,
        text,
        (text_x, text_y),
        font,
        font_scale,
        (80, 80, 80),
        font_thickness,
        cv2.LINE_AA,
    )


def _draw_legend(
    canvas: NDArray[np.uint8],
    color_map: dict[str | int, Color],
    legend_y_start: int,
    legend_height: int,
    legend_columns: int,
    font_scale: float,
    font_thickness: int,
) -> None:
    """Draw a color legend showing AOI labels."""

    if legend_height <= 0 or not color_map:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    column_count = max(1, legend_columns)
    col_width = canvas.shape[1] / column_count
    rows = math.ceil(len(color_map) / column_count)
    y_padding = 10
    available_height = max(1, legend_height - 2 * y_padding)
    row_spacing = max(18, available_height / max(1, rows))
    icon_size = 16

    for idx, (aoi_id, color) in enumerate(color_map.items()):
        row = idx // column_count
        col = idx % column_count
        x = int(col * col_width + 12)
        y = int(legend_y_start + y_padding + row * row_spacing)
        y = min(y, canvas.shape[0] - icon_size - 4)

        top_left = (x, y)
        bottom_right = (x + icon_size, y + icon_size)
        cv2.rectangle(canvas, top_left, bottom_right, color, -1)

        label = str(aoi_id)
        text_pos = (x + icon_size + 8, y + icon_size - 4)
        cv2.putText(
            canvas,
            label,
            text_pos,
            font,
            font_scale,
            (0, 0, 0),
            font_thickness,
            cv2.LINE_AA,
        )


def draw_viewing_timeline(
    tracking_result: TrackingResult,
    width: int = 1000,
    height: int = 240,
    *,
    aoi_colors: dict[str | int, Color] | None = None,
    gap_color: Color = (210, 210, 210),
    background_color: Color = (255, 255, 255),
    show_legend: bool = True,
    legend_columns: int = 3,
    timeline_padding: int = 16,
    font_scale: float = 0.5,
    font_thickness: int = 1,
) -> NDArray[np.uint8]:
    """Render a horizontal timeline describing which AOI was viewed per sample."""

    _ensure_cv2()

    (
        canvas,
        winners,
        color_map,
        sample_boundaries,
        timeline_top,
        timeline_bottom,
        legend_y_start,
        legend_height,
    ) = _initialize_timeline_canvas(
        tracking_result=tracking_result,
        width=width,
        height=height,
        background_color=background_color,
        gap_color=gap_color,
        aoi_colors=aoi_colors,
        show_legend=show_legend,
        legend_columns=legend_columns,
        timeline_padding=timeline_padding,
    )

    if tracking_result.total_samples > 0:
        _fill_timeline_bar(
            canvas,
            winners,
            sample_boundaries,
            color_map,
            gap_color,
            timeline_top,
            timeline_bottom,
            tracking_result.total_samples,
        )
    else:
        _draw_no_samples_message(canvas, timeline_top, timeline_bottom, font_scale, font_thickness)

    _draw_legend(
        canvas,
        color_map,
        legend_y_start,
        legend_height,
        legend_columns,
        font_scale,
        font_thickness,
    )

    return canvas


def create_tracking_animation(
    tracking_result: TrackingResult,
    width: int = 1000,
    height: int = 240,
    *,
    samples_per_frame: int = 1,
    aoi_colors: dict[str | int, Color] | None = None,
    gap_color: Color = (210, 210, 210),
    background_color: Color = (255, 255, 255),
    future_color: Color = (235, 235, 235),
    show_legend: bool = True,
    legend_columns: int = 3,
    timeline_padding: int = 16,
    font_scale: float = 0.5,
    font_thickness: int = 1,
    draw_cursor: bool = True,
    cursor_color: Color = (0, 0, 0),
    annotate_progress: bool = True,
) -> list[NDArray[np.uint8]]:
    """Create animation frames showing timeline progression across the session."""

    _ensure_cv2()
    if samples_per_frame <= 0:
        raise ValueError("samples_per_frame must be >= 1")

    (
        base_canvas,
        winners,
        color_map,
        sample_boundaries,
        timeline_top,
        timeline_bottom,
        legend_y_start,
        legend_height,
    ) = _initialize_timeline_canvas(
        tracking_result=tracking_result,
        width=width,
        height=height,
        background_color=background_color,
        gap_color=gap_color,
        aoi_colors=aoi_colors,
        show_legend=show_legend,
        legend_columns=legend_columns,
        timeline_padding=timeline_padding,
    )

    base_with_legend = base_canvas.copy()
    _draw_legend(
        base_with_legend,
        color_map,
        legend_y_start,
        legend_height,
        legend_columns,
        font_scale,
        font_thickness,
    )

    total_samples = len(winners)
    if total_samples == 0:
        _draw_no_samples_message(
            base_with_legend, timeline_top, timeline_bottom, font_scale, font_thickness
        )
        return [base_with_legend]

    frames: list[NDArray[np.uint8]] = []
    steps: list[int] = []
    step = samples_per_frame
    while step < total_samples:
        steps.append(step)
        step += samples_per_frame
    steps.append(total_samples)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for processed in steps:
        frame = base_with_legend.copy()
        _fill_timeline_bar(
            frame,
            winners,
            sample_boundaries,
            color_map,
            gap_color,
            timeline_top,
            timeline_bottom,
            processed,
            future_color=future_color,
        )

        if draw_cursor and sample_boundaries:
            if processed <= 0:
                cursor_x = sample_boundaries[0][0]
            else:
                cursor_index = max(0, min(processed - 1, total_samples - 1))
                cursor_x = sample_boundaries[cursor_index][1] - 1
            cursor_x = max(0, min(cursor_x, frame.shape[1] - 1))
            top = max(0, timeline_top - 4)
            bottom = min(frame.shape[0] - 1, timeline_bottom + 4)
            cv2.line(frame, (cursor_x, top), (cursor_x, bottom), cursor_color, 1, cv2.LINE_AA)

        if annotate_progress:
            text = f"{processed}/{total_samples} samples"
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            text_x = max(8, frame.shape[1] - text_w - 8)
            text_y = max(text_h + 8, timeline_top - 8 if timeline_top > text_h else timeline_bottom + text_h + 8)
            text_y = min(text_y, frame.shape[0] - 8)
            cv2.putText(
                frame,
                text,
                (text_x, text_y),
                font,
                font_scale,
                cursor_color,
                font_thickness,
                cv2.LINE_AA,
            )

        frames.append(frame)

    return frames
