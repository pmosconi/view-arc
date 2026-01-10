"""Full tracking demonstration with synthetic viewer movement.

This example demonstrates the complete attention tracking workflow:
1. Generate synthetic viewer movement through a shop floor
2. Compute attention seconds for each AOI
3. Visualize results with heatmap, path overlay, timeline, and labels

The viewer starts near the entrance and browses the store for 60 seconds
at an average speed of ~10 px/sec, looking around while moving.

Run with::

    uv run python examples/real_image_processing_tracking_full.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from skimage import color, io, util

from view_arc.tracking import (
    AOI,
    TrackingResult,
    ViewerSample,
    compute_attention_seconds,
)
from view_arc.tracking.visualize import (
    draw_attention_heatmap,
    draw_attention_labels,
    draw_viewing_timeline,
    HAS_CV2,
)

EXAMPLES_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXAMPLES_DIR.parent
IMAGE_PATH = PROJECT_ROOT / "images" / "SVB_areas.png"
POLYGON_PATH = PROJECT_ROOT / "images" / "SVB_polygon_vertices.json"
SHOP_FLOOR_PATH = PROJECT_ROOT / "images" / "SVB_shop_floor.json"
OUTPUT_DIR = EXAMPLES_DIR / "output" / "SVB"

# Simulation parameters
DURATION_SECONDS = 60
SAMPLE_RATE_HZ = 1.0
AVG_SPEED_PX_PER_SEC = 25.0
FIELD_OF_VIEW_DEG = 45.0
MAX_RANGE = 100.0
RANDOM_SEED = 42  # For reproducible results

def load_scene_image() -> NDArray[np.uint8]:
    """Load the demo background from images/background.jpeg as uint8 RGB."""
    if not IMAGE_PATH.exists():
        raise SystemExit(
            f"Sample image not found at {IMAGE_PATH}. Add the file or adjust IMAGE_PATH."
        )

    raw_image = io.imread(IMAGE_PATH)
    image = np.asarray(raw_image)
    if image.ndim == 2:
        image = color.gray2rgb(image)
    if image.dtype != np.uint8:
        image = util.img_as_ubyte(image)
    return cast(NDArray[np.uint8], image.astype(np.uint8, copy=False))


def load_polygon_from_json(json_path: Path) -> NDArray[np.float64]:
    """Load a polygon from JSON file (first entry with vertices)."""
    if not json_path.exists():
        raise SystemExit(f"File not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) == 0:
        raise SystemExit(f"Expected a list with at least one polygon in {json_path}")

    vertices = data[0].get("vertices")
    if not vertices:
        raise SystemExit(f"No 'vertices' key found in first entry of {json_path}")

    return np.array(vertices, dtype=np.float64)


def load_aois(polygon_file: Path) -> list[AOI]:
    """Load manually annotated polygons from JSON and return AOI objects."""
    if not polygon_file.exists():
        raise SystemExit(f"Polygon annotation file not found at {polygon_file}")

    with polygon_file.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    aois: list[AOI] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        vertices = entry.get("vertices")
        if not isinstance(vertices, list) or len(vertices) < 3:
            continue

        polygon = np.array(vertices, dtype=np.float64)
        if polygon.ndim != 2 or polygon.shape[1] != 2:
            continue

        # Get the annotation ID
        try:
            annotation_id = int(entry["id"])
        except (KeyError, TypeError, ValueError):
            annotation_id = len(aois)

        aois.append(AOI(id=annotation_id, contour=polygon))

    if not aois:
        raise SystemExit(f"No valid polygons found in {polygon_file}")

    return aois


def point_in_polygon(
    point: NDArray[np.floating[Any]], polygon: NDArray[np.floating[Any]]
) -> bool:
    """Check if a point is inside a polygon using ray casting algorithm."""
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def generate_browsing_path(
    shop_floor: NDArray[np.float64],
    aois: list[AOI],
    duration_sec: float,
    sample_rate: float,
    avg_speed: float,
    random_seed: int | None = None,
) -> list[tuple[float, float]]:
    """Generate a realistic browsing path within shop floor bounds.
    
    Creates a path that:
    - Starts near the entrance (between vertices 1 and 2)
    - Moves with some randomness but stays within bounds
    - Avoids walking through AOIs (shelves, displays)
    - Occasionally pauses (speed variations)
    - Turns gradually (not sharp angles)
    
    Args:
        shop_floor: Polygon defining walkable area
        aois: List of AOIs to avoid
        duration_sec: Duration of browsing session
        sample_rate: Samples per second
        avg_speed: Average walking speed in pixels per second
        random_seed: Random seed for reproducibility
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    num_samples = int(duration_sec * sample_rate)
    
    # Start position near entrance (midpoint between vertices 10 and 11)
    entrance_mid = (shop_floor[10] + shop_floor[11]) / 2
    # Move slightly inward from entrance
    direction_in = (shop_floor[0] - entrance_mid)
    direction_in = direction_in / np.linalg.norm(direction_in)
    current_pos = entrance_mid + direction_in * 20
    
    positions: list[tuple[float, float]] = [tuple(current_pos)]
    
    # Initial direction: toward center of shop floor
    center = np.mean(shop_floor, axis=0)
    current_direction = center - current_pos
    current_direction = current_direction / np.linalg.norm(current_direction)
    
    for i in range(1, num_samples):
        # Time delta
        dt = 1.0 / sample_rate
        
        # Speed variation (sometimes pause, sometimes faster)
        speed_variation = np.random.uniform(0.5, 1.3)
        if np.random.random() < 0.1:  # 10% chance to pause
            speed_variation = 0.1
        
        current_speed = avg_speed * speed_variation
        
        # Direction change: gradual turning + some randomness
        turn_angle = np.random.uniform(-0.15, 0.15)  # Radians (~8.5 degrees max)
        cos_a, sin_a = np.cos(turn_angle), np.sin(turn_angle)
        current_direction = np.array([
            current_direction[0] * cos_a - current_direction[1] * sin_a,
            current_direction[0] * sin_a + current_direction[1] * cos_a,
        ])
        current_direction = current_direction / np.linalg.norm(current_direction)
        
        # Try to move
        step = current_direction * current_speed * dt
        new_pos = current_pos + step
        
        # Keep within bounds and outside AOIs - if invalid, adjust direction gradually
        max_attempts = 20
        attempt = 0
        collision_detected = False
        while attempt < max_attempts:
            # Check if position is valid (inside shop floor and outside all AOIs)
            if not point_in_polygon(new_pos, shop_floor):
                valid = False
                collision_detected = True
            else:
                # Check if inside any AOI
                inside_aoi = False
                for aoi in aois:
                    if point_in_polygon(new_pos, aoi.contour):
                        inside_aoi = True
                        collision_detected = True
                        break
                valid = not inside_aoi
            
            if valid:
                break
            
            # Invalid position - turn more gradually, alternating left/right
            # Start with smaller turns and increase if needed
            if attempt < 5:
                turn_angle = np.random.uniform(0.3, 0.5)  # ~17-29 degrees
            elif attempt < 10:
                turn_angle = np.random.uniform(0.5, 0.8)  # ~29-46 degrees
            else:
                turn_angle = np.random.uniform(0.8, 1.2)  # ~46-69 degrees
            
            # Alternate turning direction
            if attempt % 2 == 1:
                turn_angle = -turn_angle
            
            cos_a, sin_a = np.cos(turn_angle), np.sin(turn_angle)
            current_direction = np.array([
                current_direction[0] * cos_a - current_direction[1] * sin_a,
                current_direction[0] * sin_a + current_direction[1] * cos_a,
            ])
            current_direction = current_direction / np.linalg.norm(current_direction)
            step = current_direction * current_speed * dt
            new_pos = current_pos + step
            attempt += 1
        
        if attempt < max_attempts:
            current_pos = new_pos
        else:
            # If still stuck after many attempts, reduce speed dramatically
            if collision_detected:
                current_speed *= 0.1
        
        positions.append(tuple(current_pos))
    
    return positions


def generate_view_directions(
    positions: list[tuple[float, float]],
    aois: list[AOI],
    random_seed: int | None = None,
) -> list[tuple[float, float]]:
    """Generate view directions that combine movement direction and scanning.
    
    The viewer:
    - Generally looks in the direction of movement
    - Scans left/right occasionally (shelf browsing)
    - Sometimes looks at nearby AOIs
    
    Args:
        positions: List of viewer positions
        aois: List of AOIs
        random_seed: Random seed for reproducibility
    """
    if random_seed is not None:
        np.random.seed(random_seed + 1)  # Use different seed for directions
    
    num_samples = len(positions)
    directions: list[tuple[float, float]] = []
    
    for i in range(num_samples):
        # Movement direction (if moving)
        if i > 0:
            movement = np.array(positions[i]) - np.array(positions[i-1])
            movement_norm = np.linalg.norm(movement)
            if movement_norm > 0.1:  # If actually moving
                movement_dir = movement / movement_norm
            else:
                # Not moving, use previous direction or random
                if i > 1 and len(directions) > 0:
                    movement_dir = np.array(directions[-1])
                else:
                    movement_dir = np.array([1.0, 0.0])
        else:
            # First sample: look toward center
            movement_dir = np.array([1.0, 0.0])
        
        # Add scanning behavior
        scan_mode = np.random.random()
        
        if scan_mode < 0.6:
            # 60% - Look in movement direction with small variations
            angle_offset = np.random.uniform(-0.2, 0.2)
            cos_a, sin_a = np.cos(angle_offset), np.sin(angle_offset)
            view_dir = np.array([
                movement_dir[0] * cos_a - movement_dir[1] * sin_a,
                movement_dir[0] * sin_a + movement_dir[1] * cos_a,
            ])
        elif scan_mode < 0.85:
            # 25% - Scan left or right (perpendicular to movement)
            perpendicular = np.array([-movement_dir[1], movement_dir[0]])
            if np.random.random() < 0.5:
                perpendicular = -perpendicular
            # Add more forward component to avoid backward-looking or too wide angles
            # Use 0.7 forward, 0.3 perpendicular for more natural scanning
            view_dir = 0.7 * movement_dir + 0.3 * perpendicular
        else:
            # 15% - Look at nearby AOI
            current_pos = np.array(positions[i])
            closest_aoi = None
            min_dist = float('inf')
            for aoi in aois:
                centroid = np.mean(aoi.contour, axis=0)
                dist = float(np.linalg.norm(centroid - current_pos))
                if dist < min_dist:
                    min_dist = dist
                    closest_aoi = aoi
            
            if closest_aoi is not None:
                centroid = np.mean(closest_aoi.contour, axis=0)
                view_dir = centroid - current_pos
            else:
                view_dir = movement_dir
        
        # Normalize
        view_dir = view_dir / np.linalg.norm(view_dir)
        directions.append(tuple(view_dir))
    
    return directions


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
    color: tuple[int, int, int] = (0, 0, 255),
    thickness: int = 3,
    show_start_end: bool = True,
    show_directions: bool = True,
    direction_interval: int = 5,
    show_fov: bool = False,
    fov_deg: float = 45.0,
    fov_range: float = 50.0,
    sample_hits: list[bool] | None = None,
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
    if not HAS_CV2:
        return image
    
    import cv2
    
    output = image.copy()
    
    # Draw FOV wedges first (if enabled) so they appear behind everything
    if show_fov:
        overlay_hit = output.copy()
        overlay_miss = output.copy()
        fov_hit_alpha = 0.40
        fov_miss_alpha = 0.35
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
            
            # Only draw FOV arcs when hitting an AOI
            if not has_hit:
                continue
            
            # Draw filled wedge for transparency
            cv2.ellipse(
                overlay_hit,
                center,
                axes,
                0,
                start_angle_deg,
                end_angle_deg,
                (255, 165, 0),
                -1,
            )
            any_hit_fill = True

            # Draw thin arc outline and radius lines
            cv2.ellipse(output, center, axes, 0, start_angle_deg, end_angle_deg,
                        (255, 165, 0), 2)

            end_x_start = int(pos[0] + fov_range * np.cos(start_angle))
            end_y_start = int(pos[1] + fov_range * np.sin(start_angle))
            cv2.line(output, center, (end_x_start, end_y_start), (255, 165, 0), 2)

            end_x_end = int(pos[0] + fov_range * np.cos(end_angle))
            end_y_end = int(pos[1] + fov_range * np.sin(end_angle))
            cv2.line(output, center, (end_x_end, end_y_end), (255, 165, 0), 2)

        if any_hit_fill:
            cv2.addWeighted(overlay_hit, fov_hit_alpha, output, 1 - fov_hit_alpha, 0, output)
    
    # Draw path lines
    for i in range(len(samples) - 1):
        pt1 = (int(samples[i].position[0]), int(samples[i].position[1]))
        pt2 = (int(samples[i+1].position[0]), int(samples[i+1].position[1]))
        cv2.line(output, pt1, pt2, color, thickness, cv2.LINE_AA)
    
    # Draw orange circles at each track point (excluding start and end)
    for i in range(len(samples)):
        if i == 0 or i == len(samples) - 1:
            continue  # Skip start and end points
        pos = (int(samples[i].position[0]), int(samples[i].position[1]))
        cv2.circle(output, pos, 3, (255, 165, 0), -1)  # Orange circle in RGB
    
    # Draw direction arrows at intervals
    if show_directions:
        arrow_color = (255, 0, 255)  # Magenta in RGB - highly visible
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
        cv2.circle(output, end, 5, (255, 0, 0), -1)
        cv2.circle(output, end, 5, (0, 0, 0), 1)
    
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
    if not HAS_CV2:
        return image
    
    import cv2
    
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


def draw_attention_callout(
    image: NDArray[np.uint8],
    aoi: AOI,
    attention_seconds: int,
    position: tuple[float, float] | None = None,
) -> NDArray[np.uint8]:
    """Draw a callout showing which AOI is being viewed and attention time.
    
    Args:
        image: Input image
        aoi: The AOI being viewed
        attention_seconds: Accumulated attention seconds for this AOI
        position: Optional viewer position to draw line from (if None, no line)
    
    Returns:
        Image with callout drawn
    """
    if not HAS_CV2:
        return image
    
    import cv2
    
    output = image.copy()
    
    # Calculate AOI centroid for callout position
    centroid = aoi.contour.mean(axis=0)
    cx, cy = int(centroid[0]), int(centroid[1])
    
    # Callout text
    aoi_text = f"Area {aoi.id}"
    time_text = f"{attention_seconds}s"
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    
    # Get text sizes
    (aoi_w, aoi_h), _ = cv2.getTextSize(aoi_text, font, font_scale, font_thickness)
    (time_w, time_h), baseline = cv2.getTextSize(time_text, font, font_scale, font_thickness)
    
    # Callout box dimensions
    box_w = max(aoi_w, time_w) + 20
    box_h = aoi_h + time_h + 20
    
    # Position callout above AOI centroid
    box_x = cx - box_w // 2
    box_y = cy - box_h - 30
    
    # Keep box within image bounds
    height, width = output.shape[:2]
    box_x = max(5, min(box_x, width - box_w - 5))
    box_y = max(5, min(box_y, height - box_h - 5))
    
    # Draw callout box with semi-transparent background
    overlay = output.copy()
    cv2.rectangle(
        overlay,
        (box_x, box_y),
        (box_x + box_w, box_y + box_h),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)
    
    # Draw border
    cv2.rectangle(
        output,
        (box_x, box_y),
        (box_x + box_w, box_y + box_h),
        (255, 165, 0),
        2,
    )
    
    # Draw pointer line from box to AOI centroid
    pointer_start = (box_x + box_w // 2, box_y + box_h)
    pointer_end = (cx, cy)
    cv2.line(output, pointer_start, pointer_end, (255, 165, 0), 2)
    cv2.circle(output, pointer_end, 4, (255, 165, 0), -1)
    
    # Draw text
    text_x = box_x + 10
    text_y = box_y + aoi_h + 5
    cv2.putText(output, aoi_text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    
    text_y += time_h + 5
    cv2.putText(output, time_text, (text_x, text_y), font, font_scale, (255, 165, 0), font_thickness, cv2.LINE_AA)
    
    return output


def draw_single_frame(
    image: NDArray[np.uint8],
    aois: list[AOI],
    samples: list[ViewerSample],
    current_idx: int,
    sample_hits: list[bool],
    tracking_result: TrackingResult,
    fov_deg: float,
    fov_range: float,
    callout_persistence: int = 3,
) -> NDArray[np.uint8]:
    """
    Draw a single frame showing the viewer's journey up to current_idx.
    
    Args:
        image: Background image
        aois: List of AOIs
        samples: All viewer samples
        current_idx: Current sample index to display
        sample_hits: Per-sample hit flags
        tracking_result: Tracking result with attention data
        fov_deg: Field of view angle
        fov_range: FOV visualization range
        callout_persistence: How many frames to keep callout visible after hit ends
    
    Returns:
        Frame image with path, current position, FOV, and attention callout
    """
    if not HAS_CV2:
        return image
    
    import cv2
    
    output = image.copy()
    
    # Draw AOI contours
    output = draw_aoi_contours(output, aois, color=(255, 255, 0), thickness=1, show_labels=True)
    
    # Draw path up to current position
    path_color = (0, 0, 255)  # Blue
    for i in range(min(current_idx, len(samples) - 1)):
        pt1 = (int(samples[i].position[0]), int(samples[i].position[1]))
        pt2 = (int(samples[i+1].position[0]), int(samples[i+1].position[1]))
        cv2.line(output, pt1, pt2, path_color, 3, cv2.LINE_AA)
    
    # Draw past positions as small circles
    for i in range(current_idx):
        pos = (int(samples[i].position[0]), int(samples[i].position[1]))
        cv2.circle(output, pos, 2, (255, 165, 0), -1)
    
    # Draw FOV arc for current position if hitting AOI
    if current_idx < len(samples):
        sample = samples[current_idx]
        has_hit = sample_hits[current_idx] if current_idx < len(sample_hits) else False
        
        if has_hit:
            pos = sample.position
            direction = sample.direction
            
            fov_rad = np.deg2rad(fov_deg)
            center_angle = np.arctan2(direction[1], direction[0])
            start_angle = center_angle - fov_rad / 2
            end_angle = center_angle + fov_rad / 2
            
            start_angle_deg = np.rad2deg(start_angle)
            end_angle_deg = np.rad2deg(end_angle)
            
            center = (int(pos[0]), int(pos[1]))
            axes = (int(fov_range), int(fov_range))
            
            # Draw filled wedge
            overlay = output.copy()
            cv2.ellipse(
                overlay,
                center,
                axes,
                0,
                start_angle_deg,
                end_angle_deg,
                (255, 165, 0),
                -1,
            )
            cv2.addWeighted(overlay, 0.40, output, 0.60, 0, output)
            
            # Draw arc outline
            cv2.ellipse(output, center, axes, 0, start_angle_deg, end_angle_deg,
                        (255, 165, 0), 2)
            
            # Draw radius lines
            end_x_start = int(pos[0] + fov_range * np.cos(start_angle))
            end_y_start = int(pos[1] + fov_range * np.sin(start_angle))
            cv2.line(output, center, (end_x_start, end_y_start), (255, 165, 0), 2)
            
            end_x_end = int(pos[0] + fov_range * np.cos(end_angle))
            end_y_end = int(pos[1] + fov_range * np.sin(end_angle))
            cv2.line(output, center, (end_x_end, end_y_end), (255, 165, 0), 2)
        
        # Draw current direction arrow
        arrow_color = (255, 0, 255)  # Magenta
        arrow_length = 25
        pos = sample.position
        direction = sample.direction
        
        end_x = int(pos[0] + direction[0] * arrow_length)
        end_y = int(pos[1] + direction[1] * arrow_length)
        start = (int(pos[0]), int(pos[1]))
        end = (end_x, end_y)
        cv2.arrowedLine(output, start, end, arrow_color, 2, cv2.LINE_AA, tipLength=0.3)
        
        # Draw current position as larger circle
        cv2.circle(output, (int(pos[0]), int(pos[1])), 6, (0, 255, 0), -1)
        cv2.circle(output, (int(pos[0]), int(pos[1])), 6, (0, 0, 0), 2)
    
    # Add timestamp/frame number
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Frame: {current_idx + 1}/{len(samples)}"
    cv2.putText(output, text, (10, 30), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(output, text, (10, 30), font, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    
    # Draw attention callout if currently viewing an AOI or recently viewed one
    timeline = tracking_result.get_viewing_timeline()
    
    # Find the most recent AOI hit within callout_persistence frames
    callout_aoi_id = None
    for check_idx in range(max(0, current_idx - callout_persistence), current_idx + 1):
        if check_idx < len(timeline):
            _, aoi_id = timeline[check_idx]
            if aoi_id is not None:
                callout_aoi_id = aoi_id
                break  # Use the most recent hit
    
    if callout_aoi_id is not None:
        # Find the AOI object and calculate attention up to current frame
        target_aoi = None
        for aoi in aois:
            if aoi.id == callout_aoi_id:
                target_aoi = aoi
                break
        
        if target_aoi is not None:
            # Count attention seconds up to current frame
            attention_count = sum(
                1 for idx in range(current_idx + 1)
                if idx < len(timeline) and timeline[idx][1] == callout_aoi_id
            )
            
            output = draw_attention_callout(
                output,
                target_aoi,
                attention_count,
            )
    
    return output


def draw_legend(
    image: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Draw legend for markers and symbols."""
    if not HAS_CV2:
        return image
    
    import cv2
    
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
    cv2.circle(output, (x_offset, y_pos), 5, (255, 0, 0), -1)
    cv2.circle(output, (x_offset, y_pos), 5, (0, 0, 0), 1)
    cv2.putText(output, "End", (x_offset + 15, y_pos + 5), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    
    # Direction arrow
    x_offset = 170
    cv2.arrowedLine(output, (x_offset, y_pos), (x_offset + 20, y_pos), (255, 0, 255), 1, cv2.LINE_AA, tipLength=0.3)
    cv2.putText(output, "View Direction", (x_offset + 30, y_pos + 5), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    
    return output


def main() -> None:
    """Generate synthetic tracking data and visualize results."""
    
    print("Loading scene and AOIs...")
    image = load_scene_image()
    aois = load_aois(POLYGON_PATH)
    shop_floor = load_polygon_from_json(SHOP_FLOOR_PATH)
    
    print(f"Loaded {len(aois)} AOIs")
    print(f"Shop floor has {len(shop_floor)} vertices")
    
    # Generate synthetic viewer data
    print(f"\nGenerating {DURATION_SECONDS}s browsing path at {AVG_SPEED_PX_PER_SEC} px/sec...")
    print(f"Using random seed: {RANDOM_SEED}")
    positions = generate_browsing_path(
        shop_floor,
        aois,
        DURATION_SECONDS,
        SAMPLE_RATE_HZ,
        AVG_SPEED_PX_PER_SEC,
        random_seed=RANDOM_SEED,
    )
    
    print(f"Generated {len(positions)} position samples")
    total_distance = sum(
        np.linalg.norm(np.array(positions[i+1]) - np.array(positions[i]))
        for i in range(len(positions) - 1)
    )
    print(f"Total distance traveled: {total_distance:.1f} px (avg speed: {total_distance/DURATION_SECONDS:.1f} px/sec)")
    
    print("\nGenerating view directions...")
    directions = generate_view_directions(positions, aois, random_seed=RANDOM_SEED)
    
    # Create ViewerSamples
    samples = [
        ViewerSample(
            position=pos,
            direction=dir,
            timestamp=float(i),
        )
        for i, (pos, dir) in enumerate(zip(positions, directions))
    ]
    
    # Compute attention tracking
    print("\nComputing attention seconds...")
    result = compute_attention_seconds(
        samples=samples,
        aois=aois,
        field_of_view_deg=FIELD_OF_VIEW_DEG,
        max_range=MAX_RANGE,
    )
    
    # Print results
    print(f"\n{'='*60}")
    print("TRACKING RESULTS")
    print(f"{'='*60}")
    print(f"Total samples: {result.total_samples}")
    print(f"Samples with hits: {result.samples_with_hits}")
    print(f"Samples with no winner: {result.samples_no_winner}")
    print(f"Coverage ratio: {result.coverage_ratio:.1%}")
    print(f"\nTop AOIs by attention:")
    for rank, (aoi_id, hit_count) in enumerate(result.get_top_aois(10), 1):
        percentage = (hit_count / result.samples_with_hits * 100) if result.samples_with_hits > 0 else 0.0
        print(f"  {rank}. AOI {aoi_id}: {hit_count}s ({percentage:.1f}%)")
    
    if not HAS_CV2:
        print("\nOpenCV not installed; skipping visualization output.")
        return
    
    import cv2
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Draw attention heatmap with AOI contours, viewer path, FOV, and legend
    print("\nGenerating attention heatmap...")
    heatmap = draw_attention_heatmap(
        image,
        aois=aois,
        tracking_result=result,
        colormap="hot",
        fill_alpha=0.6,
    )
    
    # Draw AOI contours with labels (bright yellow, thin lines)
    heatmap = draw_aoi_contours(heatmap, aois, color=(255, 255, 0), thickness=1, show_labels=True)
    
    # Determine which samples hit an AOI for adaptive FOV transparency
    sample_hits = build_sample_hit_flags(result)
    
    # Add viewer path with FOV wedges and direction arrows
    heatmap = draw_viewer_path(
        heatmap,
        samples,
        show_directions=True,
        direction_interval=1,
        show_fov=True,
        fov_deg=FIELD_OF_VIEW_DEG,
        fov_range=MAX_RANGE,
        sample_hits=sample_hits,
    )
    
    # Add legend
    heatmap = draw_legend(heatmap)
    
    heatmap_path = OUTPUT_DIR / "tracking_heatmap_with_path.png"
    cv2.imwrite(str(heatmap_path), cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
    print(f"Saved heatmap to {heatmap_path}")
    
    # 2. Draw attention labels
    print("Generating attention labels...")
    labeled = draw_attention_labels(
        heatmap.copy(),
        aois=aois,
        tracking_result=result,
        show_aoi_id=True,
        show_hit_count=True,
        show_percentage=True,
        font_scale=0.4,
        font_thickness=1,
    )
    
    labeled_path = OUTPUT_DIR / "tracking_labeled.png"
    cv2.imwrite(str(labeled_path), cv2.cvtColor(labeled, cv2.COLOR_RGB2BGR))
    print(f"Saved labeled image to {labeled_path}")
    
    # 3. Draw viewing timeline
    print("Generating viewing timeline...")
    timeline = draw_viewing_timeline(
        result,
        width=1200,
        height=300,
        show_legend=True,
        legend_columns=4,
    )
    
    timeline_path = OUTPUT_DIR / "tracking_timeline.png"
    cv2.imwrite(str(timeline_path), timeline)
    print(f"Saved timeline to {timeline_path}")
    
    # 4. Generate individual frames for video creation
    print("\nGenerating individual frames for video...")
    frames_dir = OUTPUT_DIR / "frames"
    frames_dir.mkdir(exist_ok=True)
    
    for idx in range(len(samples)):
        frame = draw_single_frame(
            image,
            aois,
            samples,
            idx,
            sample_hits,
            result,
            FIELD_OF_VIEW_DEG,
            MAX_RANGE,
        )
        frame_path = frames_dir / f"frame_{idx:04d}.png"
        cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        if (idx + 1) % 10 == 0 or idx == len(samples) - 1:
            print(f"  Generated {idx + 1}/{len(samples)} frames")
    
    print(f"\nSaved {len(samples)} frames to {frames_dir}")
    print("\nTo create a video, run:")
    print(f"  ffmpeg -framerate 2 -i {frames_dir}/frame_%04d.png -vf \"scale=ceil(iw/2)*2:ceil(ih/2)*2\" -c:v libx264 -pix_fmt yuv420p {OUTPUT_DIR}/tracking_video.mp4")
    
    print(f"\n{'='*60}")
    print("All visualizations saved to examples/output/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
