#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization script for viewer attention tracking results.

This script generates a graphical representation of the attention tracking
results for an event, including:
1. Background store image
2. AOI polygons overlaid
3. Viewer track path
4. Attention heatmap showing which AOIs received attention
5. Timeline visualization

Usage:
    # After running the event through lambda_local.py and capturing output
    python misc/visualize_event.py --event-data output.json
    
    # Or run with live data (requires database access)
    python misc/visualize_event.py --event events/event.json --live
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Sequence, cast

from aws_pysdk import ParameterConfig
import numpy as np
from numpy.typing import NDArray

try:
    from skimage import io
    import cv2
except ImportError:
    raise SystemExit(
        "This script requires scikit-image and opencv-python.\n"
        "Install with: uv pip install -e '.[visualization]'"
    )

from view_arc.tracking import (
    AOI,
    TrackingResult,
    ViewerSample,
    compute_attention_seconds,
    draw_attention_heatmap,
    draw_attention_labels,
    draw_viewing_timeline,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "output"


def load_background_image(store_id: int) -> NDArray[np.uint8]:
    """
    Load the background image for the store.
    
    Args:
        store_id: Store ID to load image for
    
    Returns:
        RGB image as uint8 numpy array
    """
    image_path = SCRIPT_DIR / f"{store_id}.jpeg"
    if not image_path.exists():
        raise FileNotFoundError(f"Background image not found: {image_path}")
    
    logger.info(f"Loading background image: {image_path}")
    image = io.imread(str(image_path))
    
    # Ensure RGB format
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.shape[2] == 4:
        image = image[:, :, :3]
    
    return image.astype(np.uint8)


def transform_point(
    point: tuple[float, float],
    transform_matrix: dict[str, Any]
) -> tuple[float, float]:
    """
    Transform a point using the transformation matrix.
    
    Args:
        point: (x, y) coordinates in reference system
        transform_matrix: Transformation matrix with linear and translation components
    
    Returns:
        Transformed (x, y) coordinates in pixel space
    """
    linear = np.array(transform_matrix['linear'])
    translation = np.array(transform_matrix['translation'])
    
    point_arr = np.array(point)
    transformed = linear @ point_arr + translation
    
    return float(transformed[0]), float(transformed[1])


def transform_direction(
    direction: tuple[float, float],
    transform_matrix: dict[str, Any]
) -> tuple[float, float]:
    """
    Transform a direction vector (no translation).
    
    Args:
        direction: (dx, dy) direction vector in reference system
        transform_matrix: Transformation matrix with linear component
    
    Returns:
        Normalized transformed direction vector in pixel space
    """
    linear = np.array(transform_matrix['linear'])
    direction_arr = np.array(direction)
    
    transformed = linear @ direction_arr
    norm = np.linalg.norm(transformed)
    if norm > 0:
        transformed = transformed / norm
    else:
        # Keep zero direction vector as (0, 0) for missing direction
        transformed = np.array([0.0, 0.0])
    
    return float(transformed[0]), float(transformed[1])


def prepare_aois_and_samples(
    device_data: dict[str, Any],
    track_coordinates: list[dict[str, Any]]
) -> tuple[list[AOI], list[ViewerSample]]:
    """
    Transform AOIs and track coordinates to pixel space.
    
    Args:
        device_data: Device data with AOI geometries and image metadata
        track_coordinates: Track coordinates with positions and view directions
    
    Returns:
        Tuple of (AOI list, ViewerSample list)
    """
    ref_to_pixel = device_data['image_metadata']['ref_to_pixel']
    
    # Transform AOIs
    aois: list[AOI] = []
    for aoi_geom in device_data['aoi_dict']:
        contour_ref = np.array(aoi_geom['contour'], dtype=np.float64)
        
        # Transform each point
        contour_px = np.array([
            transform_point((pt[0], pt[1]), ref_to_pixel)
            for pt in contour_ref
        ], dtype=np.float64)
        
        aois.append(AOI(
            id=aoi_geom['id'],
            contour=contour_px
        ))
    
    logger.info(f"Transformed {len(aois)} AOIs to pixel space")
    
    # Transform track coordinates
    samples: list[ViewerSample] = []
    for i, coord in enumerate(track_coordinates):
        position_px = transform_point((coord['X'], coord['Y']), ref_to_pixel)
        
        # Handle missing or None direction values
        view_x = coord.get('View_X') or 0.0
        view_y = coord.get('View_Y') or 0.0
        direction_px = transform_direction(
            (view_x, view_y),
            ref_to_pixel
        )
        
        samples.append(ViewerSample(
            position=position_px,
            direction=direction_px,
            timestamp=float(i),
            allow_missing_direction=True
        ))
    
    logger.info(f"Transformed {len(samples)} track samples to pixel space")
    
    return aois, samples


def build_sample_hit_flags(tracking_result: TrackingResult) -> list[bool]:
    """Return per-sample flags indicating whether any AOI was hit."""
    timeline = tracking_result.get_viewing_timeline()
    hit_flags = [False] * tracking_result.total_samples

    for sample_idx, aoi_id in timeline:
        if 0 <= sample_idx < len(hit_flags):
            hit_flags[sample_idx] = aoi_id is not None

    return hit_flags


def draw_viewer_path(
    image: NDArray[np.uint8],
    samples: list[ViewerSample],
    color: tuple[int, int, int] = (255, 255, 0),
    thickness: int = 2,
    show_start_end: bool = True,
    show_directions: bool = True,
    direction_interval: int = 5,
    show_fov: bool = False,
    fov_deg: float = 45.0,
    fov_range: float = 50.0,
    sample_hits: Sequence[bool] | None = None,
) -> NDArray[np.uint8]:
    """
    Draw the viewer's path on the image.
    
    Args:
        image: Input image
        samples: List of viewer samples
        color: BGR color for path
        thickness: Line thickness
        show_start_end: Whether to mark start and end points
        show_directions: Whether to draw direction arrows
        direction_interval: Draw direction arrow every N samples
        show_fov: Whether to draw field of view wedges
        fov_deg: Field of view angle in degrees
        fov_range: Range/radius of FOV visualization in pixels
        sample_hits: Optional per-sample flags indicating AOI hits
    
    Returns:
        Image with path drawn
    """
    output = image.copy()
    
    # Draw FOV wedges first (if enabled) so they appear behind everything
    if show_fov:
        overlay_hit = output.copy()
        overlay_miss = output.copy()
        fov_hit_alpha = 0.30
        fov_miss_alpha = 0.25
        any_hit_fill = False
        any_miss = False
        fov_rad = np.deg2rad(fov_deg)
        
        # Draw FOV for ALL samples to ensure we see every viewing cone
        for i in range(len(samples)):
            sample = samples[i]
            pos = sample.position
            direction = sample.direction
            has_hit = True
            if sample_hits is not None and i < len(sample_hits):
                has_hit = sample_hits[i]
            
            # Calculate direction angle
            center_angle = np.arctan2(direction[1], direction[0])
            start_angle = center_angle - fov_rad / 2
            end_angle = center_angle + fov_rad / 2
            
            # Convert to degrees for OpenCV (which uses degrees, 0=right, counterclockwise)
            start_angle_deg = np.rad2deg(start_angle)
            end_angle_deg = np.rad2deg(end_angle)
            
            center = (int(pos[0]), int(pos[1]))
            axes = (int(fov_range), int(fov_range))
            
            # Draw filled wedge for transparency
            if has_hit:
                cv2.ellipse(
                    overlay_hit,
                    center,
                    axes,
                    0,
                    start_angle_deg,
                    end_angle_deg,
                    (100, 255, 255),
                    -1,
                )
                any_hit_fill = True
            else:
                cv2.ellipse(
                    overlay_miss,
                    center,
                    axes,
                    0,
                    start_angle_deg,
                    end_angle_deg,
                    (100, 255, 255),
                    -1,
                )
                any_miss = True

            # Draw thin arc outline and radius lines, adjusting transparency for misses
            if has_hit:
                cv2.ellipse(output, center, axes, 0, start_angle_deg, end_angle_deg,
                            (100, 255, 255), 1)

                end_x_start = int(pos[0] + fov_range * np.cos(start_angle))
                end_y_start = int(pos[1] + fov_range * np.sin(start_angle))
                cv2.line(output, center, (end_x_start, end_y_start), (100, 255, 255), 1)

                end_x_end = int(pos[0] + fov_range * np.cos(end_angle))
                end_y_end = int(pos[1] + fov_range * np.sin(end_angle))
                cv2.line(output, center, (end_x_end, end_y_end), (100, 255, 255), 1)
            else:
                cv2.ellipse(
                    overlay_miss,
                    center,
                    axes,
                    0,
                    start_angle_deg,
                    end_angle_deg,
                    (100, 255, 255),
                    1,
                )

                end_x_start = int(pos[0] + fov_range * np.cos(start_angle))
                end_y_start = int(pos[1] + fov_range * np.sin(start_angle))
                cv2.line(overlay_miss, center, (end_x_start, end_y_start), (100, 255, 255), 1)

                end_x_end = int(pos[0] + fov_range * np.cos(end_angle))
                end_y_end = int(pos[1] + fov_range * np.sin(end_angle))
                cv2.line(overlay_miss, center, (end_x_end, end_y_end), (100, 255, 255), 1)

        if any_hit_fill:
            cv2.addWeighted(overlay_hit, fov_hit_alpha, output, 1 - fov_hit_alpha, 0, output)
        if any_miss:
            cv2.addWeighted(overlay_miss, fov_miss_alpha, output, 1 - fov_miss_alpha, 0, output)
    
    # Draw path lines
    for i in range(len(samples) - 1):
        pt1 = (int(samples[i].position[0]), int(samples[i].position[1]))
        pt2 = (int(samples[i+1].position[0]), int(samples[i+1].position[1]))
        cv2.line(output, pt1, pt2, color, thickness, cv2.LINE_AA)
    
    # Draw yellow circles at each track point (excluding start and end)
    for i in range(len(samples)):
        if i == 0 or i == len(samples) - 1:
            continue  # Skip start and end points
        pos = (int(samples[i].position[0]), int(samples[i].position[1]))
        cv2.circle(output, pos, 2, (0, 255, 255), -1)  # Yellow circle in BGR
    
    # Draw direction arrows at intervals
    if show_directions:
        arrow_color = (255, 0, 255)  # Magenta in BGR - highly visible
        arrow_length = 25
        for i in range(0, len(samples), direction_interval):
            sample = samples[i]
            pos = sample.position
            direction = sample.direction
            
            # Calculate arrow endpoint
            end_x = int(pos[0] + direction[0] * arrow_length)
            end_y = int(pos[1] + direction[1] * arrow_length)
            start = (int(pos[0]), int(pos[1]))
            end = (end_x, end_y)
            
            # Draw arrow with thin line
            cv2.arrowedLine(output, start, end, arrow_color, 1, cv2.LINE_AA, tipLength=0.3)
    
    if show_start_end and len(samples) > 0:
        # Mark start with green circle (smaller)
        start = (int(samples[0].position[0]), int(samples[0].position[1]))
        cv2.circle(output, start, 5, (0, 255, 0), -1)
        cv2.circle(output, start, 5, (0, 0, 0), 1)
        
        # Mark end with red circle (smaller)
        end = (int(samples[-1].position[0]), int(samples[-1].position[1]))
        cv2.circle(output, end, 5, (0, 0, 255), -1)
        cv2.circle(output, end, 5, (0, 0, 0), 1)
    
    return output


def draw_legend(
    image: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Draw legend for markers and symbols."""
    output = image.copy()
    height, width = output.shape[:2]
    
    # Legend background
    legend_height = 40
    legend_y = height - legend_height
    overlay = output.copy()
    cv2.rectangle(overlay, (0, legend_y), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)
    
    # Legend items
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_thickness = 1
    y_pos = legend_y + 25
    
    # Start marker
    x_offset = 20
    cv2.circle(output, (x_offset, y_pos), 5, (0, 255, 0), -1)
    cv2.circle(output, (x_offset, y_pos), 5, (0, 0, 0), 1)
    cv2.putText(output, "Start", (x_offset + 15, y_pos + 5), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    
    # End marker
    x_offset = 100
    cv2.circle(output, (x_offset, y_pos), 5, (0, 0, 255), -1)
    cv2.circle(output, (x_offset, y_pos), 5, (0, 0, 0), 1)
    cv2.putText(output, "End", (x_offset + 15, y_pos + 5), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    
    # Direction arrow
    x_offset = 170
    cv2.arrowedLine(output, (x_offset, y_pos), (x_offset + 20, y_pos), (0, 255, 255), 1, cv2.LINE_AA, tipLength=0.3)
    cv2.putText(output, "View Direction", (x_offset + 30, y_pos + 5), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    
    return output


def draw_aoi_contours(
    image: NDArray[np.uint8],
    aois: list[AOI],
    color: tuple[int, int, int] = (255, 255, 0),
    thickness: int = 3,
    show_labels: bool = True,
    font_scale: float = 0.5,
) -> NDArray[np.uint8]:
    """
    Draw AOI contours and labels on the image.
    
    Args:
        image: Input image
        aois: List of AOIs
        color: BGR color for contours
        thickness: Line thickness
        show_labels: Whether to show AOI ID labels
        font_scale: Font size for labels
    
    Returns:
        Image with AOI contours drawn
    """
    output = image.copy()
    
    for aoi in aois:
        # Draw contour
        pts = aoi.contour.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(output, [pts], isClosed=True, color=color, thickness=thickness)
        
        # Draw label at centroid
        if show_labels:
            centroid = aoi.contour.mean(axis=0)
            cx, cy = int(centroid[0]), int(centroid[1])
            
            label = str(aoi.id)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_thickness = 1
            
            # Get text size for background
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            
            # Draw semi-transparent background for label
            padding = 3
            bg_x1 = cx - text_w // 2 - padding
            bg_y1 = cy - text_h // 2 - padding
            bg_x2 = cx + text_w // 2 + padding
            bg_y2 = cy + text_h // 2 + padding + baseline
            
            # Create overlay for transparency
            overlay = output.copy()
            cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)
            
            # Draw text
            text_x = cx - text_w // 2
            text_y = cy + text_h // 2
            cv2.putText(output, label, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    
    return output


async def load_data_live(event_file: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """
    Load data from live databases using the event message.
    
    Args:
        event_file: Path to event JSON file
    
    Returns:
        Tuple of (message, device_data, track_coordinates)
    """
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Import Lambda dependencies
    from src.lib.read_parameters import Parameters
    from src.lib.read_track import Track
    from mysql_redis_cache import MRCClient
    from aws_pysdk import ssm_load_parameters
    
    # Load event
    with open(event_file, 'r') as f:
        event = json.load(f)
    
    record = event['Records'][0]
    body = json.loads(record['body'])
    message_str = body.get('Message', '{}')
    if isinstance(message_str, str):
        message = json.loads(message_str)
    else:
        message = message_str
    
    logger.info(f"Processing Event_id: {message['Event_id']}, Store_id: {message['Store_id']}")
    
    # Load SSM parameters
    env = os.environ.get("ENV", "development")
    params_config: list[ParameterConfig] = [
        {"name": f"/{env}/db/mysql/connection", "env_var_name": "MYSQL_URL", "decrypt": True},
        {"name": f"/{env}/db/redis/connection", "env_var_name": "REDIS_URL", "decrypt": True},
        {"name": f"/{env}/db/mongo/connection", "env_var_name": "MONGO_URL", "decrypt": True},
    ]
    ssm_load_parameters(params_config)
    
    # Initialize clients
    mysql_url = f"{os.environ['MYSQL_URL']}?connectionLimit=1"
    redis_config = f"{os.environ['REDIS_URL']}?socket_timeout=30000"
    mrc = MRCClient(mysql_url, redis_config)
    
    # Get data
    parameters = Parameters(mrc)
    track = Track(message)
    
    device_data, track_coordinates = await asyncio.gather(
        parameters.get_image_data(message),
        track.get_coordinates()
    )
    
    if device_data is None:
        raise ValueError("No device data found")
    if track_coordinates is None:
        raise ValueError("No track coordinates found")
    
    # Cast to expected dict types for compatibility
    return cast(dict[str, Any], message), cast(dict[str, Any], device_data), cast(list[dict[str, Any]], track_coordinates)


def load_data_from_file(data_file: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """
    Load pre-computed data from a JSON file.
    
    Expected format:
    {
        "message": {...},
        "device_data": {...},
        "track_coordinates": [...]
    }
    
    Args:
        data_file: Path to JSON data file
    
    Returns:
        Tuple of (message, device_data, track_coordinates)
    """
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    return data['message'], data['device_data'], data['track_coordinates']


def create_visualizations(
    message: dict[str, Any],
    device_data: dict[str, Any],
    track_coordinates: list[dict[str, Any]],
    output_dir: Path,
    field_of_view_deg: float = 45.0,
    max_range_meters: float = 2.0,
) -> None:
    """
    Create all visualizations for the tracking data.
    
    Args:
        message: Event message
        device_data: Device data with AOIs
        track_coordinates: Track coordinates
        output_dir: Output directory for images
        field_of_view_deg: Field of view in degrees
        max_range_meters: Maximum range in meters
    """
    store_id = message['Store_id']
    event_id = message['Event_id']
    
    # Load background image
    background = load_background_image(store_id)
    
    # Transform data to pixel space
    aois, samples = prepare_aois_and_samples(device_data, track_coordinates)
    
    # Convert max_range to pixels (approximate)
    # Using scale from transformation matrix
    linear = np.array(device_data['image_metadata']['ref_to_pixel']['linear'])
    scale_factor = np.linalg.norm(linear[0])  # pixels per meter (approximate)
    max_range_px = float(max_range_meters * scale_factor)
    
    logger.info(f"Computing attention with FOV={field_of_view_deg}°, range={max_range_meters}m ({max_range_px:.1f}px)")
    
    # Compute attention
    result = compute_attention_seconds(
        samples=samples,
        aois=aois,
        field_of_view_deg=field_of_view_deg,
        max_range=max_range_px,
        allow_missing_direction=True # Ignore samples with zero direction vectors
    )
    
    # Print results
    logger.info(f"{'='*60}")
    logger.info("TRACKING RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Total samples: {result.total_samples}")
    logger.info(f"Samples with hits: {result.samples_with_hits}")
    logger.info(f"Samples no winner: {result.samples_no_winner}")
    logger.info(f"Coverage ratio: {result.coverage_ratio:.1%}")
    
    logger.info("\nTop AOIs by attention:")
    for rank, (aoi_id, hit_count) in enumerate(result.get_top_aois(10), 1):
        percentage = (hit_count / result.samples_with_hits * 100) if result.samples_with_hits > 0 else 0.0
        logger.info(f"  {rank}. AOI {aoi_id}: {hit_count}s ({percentage:.1f}%)")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Attention heatmap with AOI contours, viewer path, and direction arrows
    logger.info("\nGenerating attention heatmap...")
    heatmap = draw_attention_heatmap(
        background,
        aois=aois,
        tracking_result=result,
        colormap="hot",
        fill_alpha=0.6,
    )
    
    # Draw AOI contours with labels (bright yellow, thin lines)
    heatmap = draw_aoi_contours(heatmap, aois, color=(255, 255, 0), thickness=1, show_labels=True)
    
    # Determine which samples hit an AOI for adaptive FOV transparency
    sample_hits = build_sample_hit_flags(result)

    # Draw viewer path with FOV wedges, direction arrows at all points
    heatmap = draw_viewer_path(
        heatmap, 
        samples, 
        show_directions=True, 
        direction_interval=1,
        show_fov=True,
        fov_deg=field_of_view_deg,
        fov_range=max_range_px,
        sample_hits=sample_hits,
    )
    
    # Add legend
    heatmap = draw_legend(heatmap)
    
    heatmap_path = output_dir / f"event_{event_id}_heatmap.png"
    cv2.imwrite(str(heatmap_path), cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
    logger.info(f"Saved heatmap to {heatmap_path}")
    
    # 2. Viewing timeline
    logger.info("Generating viewing timeline...")
    timeline = draw_viewing_timeline(
        result,
        width=1200,
        height=300,
        show_legend=True,
        legend_columns=4,
    )
    timeline_path = output_dir / f"event_{event_id}_timeline.png"
    cv2.imwrite(str(timeline_path), timeline)
    logger.info(f"Saved timeline to {timeline_path}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"All visualizations saved to {output_dir}")
    logger.info(f"{'='*60}")


async def main_async(args: argparse.Namespace) -> None:
    """Async main function."""
    if args.live:
        if not args.event:
            raise ValueError("--event is required when using --live")
        message, device_data, track_coordinates = await load_data_live(args.event)
    elif args.event_data:
        message, device_data, track_coordinates = load_data_from_file(args.event_data)
    else:
        raise ValueError("Either --event-data or --event with --live must be specified")
    
    create_visualizations(
        message,
        device_data,
        track_coordinates,
        OUTPUT_DIR,
        field_of_view_deg=args.fov,
        max_range_meters=args.max_range,
    )


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize viewer attention tracking results"
    )
    parser.add_argument(
        "--event-data",
        type=Path,
        help="Path to pre-computed JSON data file"
    )
    parser.add_argument(
        "--event",
        type=Path,
        help="Path to event JSON file (requires --live)"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Load data from live databases (requires AWS credentials)"
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=45.0,
        help="Field of view in degrees (default: 45.0)"
    )
    parser.add_argument(
        "--max-range",
        type=float,
        default=2.0,
        help="Maximum detection range in meters (default: 2.0)"
    )
    
    args = parser.parse_args()
    
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
