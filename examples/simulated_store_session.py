"""Simulated store session - generate and analyze realistic viewer trajectory.

This example demonstrates a complete synthetic attention tracking workflow:
1. Load a real store floor plan and AOI annotations
2. Generate realistic viewer movement (browsing behavior)
3. Compute attention seconds for each AOI
4. Visualize results with heatmap, path overlay, timeline, and labels

The simulation creates a realistic browsing session where a viewer:
- Enters the store near the entrance
- Browses through the space for a configurable duration
- Moves at realistic speeds with pauses
- Looks around while moving (scanning shelves)
- Stays within the shop floor boundaries

Run with::

    uv run python examples/simulated_store_session.py

Output images will be saved to examples/output/
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray
from skimage import color, io, util

from view_arc.tracking import (
    AOI,
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
IMAGE_PATH = PROJECT_ROOT / "images" / "background.jpeg"
POLYGON_PATH = PROJECT_ROOT / "images" / "polygon_vertices.json"
SHOP_FLOOR_PATH = PROJECT_ROOT / "images" / "shop-floor.json"
OUTPUT_DIR = EXAMPLES_DIR / "output"

# Simulation parameters
DURATION_SECONDS = 60
SAMPLE_RATE_HZ = 1.0
AVG_SPEED_PX_PER_SEC = 10.0
FIELD_OF_VIEW_DEG = 45.0
MAX_RANGE = 60.0


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


def point_in_polygon(point: NDArray[np.float64], polygon: NDArray[np.float64]) -> bool:
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
    duration_sec: float,
    sample_rate: float,
    avg_speed: float,
) -> list[tuple[float, float]]:
    """Generate a realistic browsing path within shop floor bounds.
    
    Creates a path that:
    - Starts near the entrance (between vertices 1 and 2)
    - Moves with some randomness but stays within bounds
    - Occasionally pauses (speed variations)
    - Turns gradually (not sharp angles)
    """
    num_samples = int(duration_sec * sample_rate)
    
    # Start position near entrance (midpoint between vertices 1 and 2)
    entrance_mid = (shop_floor[1] + shop_floor[2]) / 2
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
        turn_angle = np.random.uniform(-0.3, 0.3)  # Radians
        cos_a, sin_a = np.cos(turn_angle), np.sin(turn_angle)
        current_direction = np.array([
            current_direction[0] * cos_a - current_direction[1] * sin_a,
            current_direction[0] * sin_a + current_direction[1] * cos_a,
        ])
        current_direction = current_direction / np.linalg.norm(current_direction)
        
        # Try to move
        step = current_direction * current_speed * dt
        new_pos = current_pos + step
        
        # Keep within bounds - if we'd go outside, bounce back
        max_attempts = 10
        attempt = 0
        while not point_in_polygon(new_pos, shop_floor) and attempt < max_attempts:
            # Reflect direction
            turn_angle = np.random.uniform(np.pi/2, np.pi)
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
        # else: stay at current position (stuck)
        
        positions.append(tuple(current_pos))
    
    return positions


def generate_view_directions(
    positions: list[tuple[float, float]],
    aois: list[AOI],
) -> list[tuple[float, float]]:
    """Generate view directions that combine movement direction and scanning.
    
    The viewer:
    - Generally looks in the direction of movement
    - Scans left/right occasionally (shelf browsing)
    - Sometimes looks at nearby AOIs
    """
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
            # Add some forward component
            view_dir = 0.5 * movement_dir + 0.5 * perpendicular
        else:
            # 15% - Look at nearby AOI
            current_pos = np.array(positions[i])
            closest_aoi = None
            min_dist = float('inf')
            for aoi in aois:
                centroid = np.mean(aoi.contour, axis=0)
                dist = np.linalg.norm(centroid - current_pos)
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


def draw_viewer_path(
    image: NDArray[np.uint8],
    positions: list[tuple[float, float]],
    color: tuple[int, int, int] = (255, 255, 0),
    thickness: int = 2,
    show_start_end: bool = True,
) -> NDArray[np.uint8]:
    """Draw the viewer's path on the image."""
    if not HAS_CV2:
        return image
    
    import cv2
    
    output = image.copy()
    
    # Draw path line
    for i in range(len(positions) - 1):
        pt1 = (int(positions[i][0]), int(positions[i][1]))
        pt2 = (int(positions[i+1][0]), int(positions[i+1][1]))
        cv2.line(output, pt1, pt2, color, thickness, cv2.LINE_AA)
    
    if show_start_end and len(positions) > 0:
        # Mark start with green circle
        start = (int(positions[0][0]), int(positions[0][1]))
        cv2.circle(output, start, 8, (0, 255, 0), -1)
        cv2.circle(output, start, 8, (0, 0, 0), 2)
        
        # Mark end with red circle
        end = (int(positions[-1][0]), int(positions[-1][1]))
        cv2.circle(output, end, 8, (0, 0, 255), -1)
        cv2.circle(output, end, 8, (0, 0, 0), 2)
    
    return output


def main() -> None:
    """Generate synthetic tracking data and visualize results."""
    
    print("="*60)
    print("SIMULATED STORE SESSION")
    print("="*60)
    print()
    
    print("Loading scene and AOIs...")
    image = load_scene_image()
    aois = load_aois(POLYGON_PATH)
    shop_floor = load_polygon_from_json(SHOP_FLOOR_PATH)
    
    print(f"  Loaded {len(aois)} AOIs")
    print(f"  Shop floor has {len(shop_floor)} vertices")
    print()
    
    # Generate synthetic viewer data
    print(f"Generating {DURATION_SECONDS}s browsing path at {AVG_SPEED_PX_PER_SEC} px/sec...")
    positions = generate_browsing_path(
        shop_floor,
        DURATION_SECONDS,
        SAMPLE_RATE_HZ,
        AVG_SPEED_PX_PER_SEC,
    )
    
    print(f"  Generated {len(positions)} position samples")
    total_distance = sum(
        np.linalg.norm(np.array(positions[i+1]) - np.array(positions[i]))
        for i in range(len(positions) - 1)
    )
    print(f"  Total distance traveled: {total_distance:.1f} px")
    print(f"  Average speed: {total_distance/DURATION_SECONDS:.1f} px/sec")
    print()
    
    print("Generating view directions...")
    directions = generate_view_directions(positions, aois)
    print(f"  Generated {len(directions)} direction samples")
    print()
    
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
    print("Computing attention seconds...")
    result = compute_attention_seconds(
        samples=samples,
        aois=aois,
        field_of_view_deg=FIELD_OF_VIEW_DEG,
        max_range=MAX_RANGE,
    )
    print("  ✓ Complete")
    print()
    
    # Print results
    print("="*60)
    print("TRACKING RESULTS")
    print("="*60)
    print(f"Total samples: {result.total_samples}")
    print(f"Samples with hits: {result.samples_with_hits}")
    print(f"Samples with no winner: {result.samples_no_winner}")
    print(f"Coverage ratio: {result.coverage_ratio:.1%}")
    print()
    
    print("Top AOIs by attention:")
    for rank, (aoi_id, hit_count) in enumerate(result.get_top_aois(10), 1):
        percentage = (hit_count / result.samples_with_hits * 100) if result.samples_with_hits > 0 else 0.0
        print(f"  {rank}. AOI {aoi_id}: {hit_count}s ({percentage:.1f}%)")
    print()
    
    if not HAS_CV2:
        print("⚠ OpenCV not installed; skipping visualization output.")
        print("  Install with: uv pip install opencv-python")
        return
    
    import cv2
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Draw attention heatmap with viewer path
    print("Generating visualizations...")
    heatmap = draw_attention_heatmap(
        image,
        aois=aois,
        tracking_result=result,
        colormap="hot",
        fill_alpha=0.6,
    )
    
    # Add viewer path overlay
    heatmap = draw_viewer_path(heatmap, positions)
    
    heatmap_path = OUTPUT_DIR / "simulated_heatmap_with_path.png"
    cv2.imwrite(str(heatmap_path), heatmap)
    print(f"  • Saved heatmap to {heatmap_path.name}")
    
    # 2. Draw attention labels
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
    
    labeled_path = OUTPUT_DIR / "simulated_labeled.png"
    cv2.imwrite(str(labeled_path), labeled)
    print(f"  • Saved labeled image to {labeled_path.name}")
    
    # 3. Draw viewing timeline
    timeline = draw_viewing_timeline(
        result,
        width=1200,
        height=300,
        show_legend=True,
        legend_columns=4,
    )
    
    timeline_path = OUTPUT_DIR / "simulated_timeline.png"
    cv2.imwrite(str(timeline_path), timeline)
    print(f"  • Saved timeline to {timeline_path.name}")
    
    print()
    print("="*60)
    print(f"All visualizations saved to {OUTPUT_DIR}/")
    print("="*60)


if __name__ == "__main__":
    main()
