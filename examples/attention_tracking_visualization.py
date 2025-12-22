"""Attention tracking visualization example with heatmap output.

This example demonstrates how to visualize attention tracking results:
1. Simulate a viewing session
2. Compute attention seconds
3. Generate heatmap visualization showing attention distribution
4. Save annotated images with labels

The visualization uses color intensity to show which AOIs received more attention.

Run with::

    uv run python examples/attention_tracking_visualization.py

Output images will be saved to examples/output/
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

try:
    import cv2
except ImportError:
    raise SystemExit(
        "This example requires OpenCV. Install with: uv pip install opencv-python"
    )

from view_arc.tracking import (
    AOI,
    ViewerSample,
    compute_attention_seconds,
    draw_attention_heatmap,
    draw_attention_labels,
)


def main() -> None:
    """Generate a simulated session and create visualization outputs."""
    
    print("="*60)
    print("ATTENTION TRACKING VISUALIZATION EXAMPLE")
    print("="*60)
    print()
    
    # Define three shelf AOIs in a store (simplified rectangular shelves)
    print("Creating 3 shelf AOIs...")
    aois = [
        AOI(
            id="top_shelf",
            contour=np.array([
                [100, 50], [700, 50], [700, 150], [100, 150]
            ], dtype=np.float64),
        ),
        AOI(
            id="middle_shelf",
            contour=np.array([
                [100, 250], [700, 250], [700, 350], [100, 350]
            ], dtype=np.float64),
        ),
        AOI(
            id="bottom_shelf",
            contour=np.array([
                [100, 450], [700, 450], [700, 550], [100, 550]
            ], dtype=np.float64),
        ),
    ]
    
    # Simulate viewer samples (position and direction at 1 Hz)
    # Viewer stands at position (50, 300) and looks at different shelves
    print("Generating viewer samples (100 seconds)...")
    samples = []
    
    # First 30 seconds: looking at top shelf
    for i in range(30):
        samples.append(ViewerSample(
            position=(50.0, 300.0),
            direction=(0.8, -0.6),  # Looking up-right at top shelf
            timestamp=float(i),
        ))
    
    # Next 20 seconds: looking at middle shelf
    for i in range(30, 50):
        samples.append(ViewerSample(
            position=(50.0, 300.0),
            direction=(1.0, 0.0),  # Looking right at middle shelf
            timestamp=float(i),
        ))
    
    # Next 10 seconds: looking at bottom shelf
    for i in range(50, 60):
        samples.append(ViewerSample(
            position=(50.0, 300.0),
            direction=(0.8, 0.6),  # Looking down-right at bottom shelf
            timestamp=float(i),
        ))
    
    # Last 40 seconds: not looking at any shelf (looking away)
    for i in range(60, 100):
        samples.append(ViewerSample(
            position=(50.0, 300.0),
            direction=(-1.0, 0.0),  # Looking left (away from shelves)
            timestamp=float(i),
        ))
    
    print(f"  Generated {len(samples)} samples")
    print()
    
    # Compute attention seconds
    print("Computing attention tracking...")
    result = compute_attention_seconds(
        samples=samples,
        aois=aois,
        field_of_view_deg=90.0,
        max_range=500.0,
    )
    
    print(f"  Total samples: {result.total_samples}")
    print(f"  Samples with hits: {result.samples_with_hits}")
    print(f"  Coverage ratio: {result.coverage_ratio:.1%}")
    print()
    
    print("Per-AOI results:")
    for aoi_id, aoi_result in result.aoi_results.items():
        percentage = (aoi_result.hit_count / result.samples_with_hits * 100
                     if result.samples_with_hits > 0 else 0.0)
        print(f"  {aoi_id}: {aoi_result.hit_count}s ({percentage:.1f}%)")
    print()
    
    # Create a blank canvas for visualization
    print("Creating visualization...")
    img_height, img_width = 600, 800
    blank_image = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
    
    # Generate heatmap with 'hot' colormap (red = high attention)
    heatmap_hot = draw_attention_heatmap(
        blank_image,
        aois=aois,
        tracking_result=result,
        colormap="hot",
        fill_alpha=0.6,
        background_color=(230, 230, 230),  # Light gray for zero-hit AOIs
    )
    
    # Generate heatmap with 'viridis' colormap (yellow-green = high attention)
    heatmap_viridis = draw_attention_heatmap(
        blank_image,
        aois=aois,
        tracking_result=result,
        colormap="viridis",
        fill_alpha=0.6,
        background_color=(230, 230, 230),
    )
    
    # Add text labels to the hot heatmap
    heatmap_labeled = draw_attention_labels(
        heatmap_hot.copy(),
        aois=aois,
        tracking_result=result,
        show_aoi_id=True,
        show_hit_count=True,
        show_percentage=True,
        show_seconds=True,
    )
    
    # Save outputs
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    hot_path = output_dir / "attention_heatmap_hot.png"
    viridis_path = output_dir / "attention_heatmap_viridis.png"
    labeled_path = output_dir / "attention_heatmap_labeled.png"
    
    cv2.imwrite(str(hot_path), heatmap_hot)
    cv2.imwrite(str(viridis_path), heatmap_viridis)
    cv2.imwrite(str(labeled_path), heatmap_labeled)
    
    print("Visualization outputs saved:")
    print(f"  • {hot_path}")
    print(f"  • {viridis_path}")
    print(f"  • {labeled_path}")
    print()
    
    print("="*60)
    print("Visualization complete!")
    print("="*60)


if __name__ == "__main__":
    main()
