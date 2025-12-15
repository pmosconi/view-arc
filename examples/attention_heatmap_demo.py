"""
Example: Attention Heatmap Visualization

This example demonstrates how to visualize attention tracking results using
heatmaps and labels. The heatmap colors AOIs based on how much attention they
received, making it easy to see which areas attracted the most viewer interest.
"""

import numpy as np

try:
    import cv2
except ImportError:
    print("This example requires OpenCV. Install with: pip install opencv-python")
    exit(1)

from view_arc import compute_attention_seconds
from view_arc.tracking import AOI, ViewerSample, draw_attention_heatmap, draw_attention_labels


def main():
    """Generate a simulated attention tracking session and visualize results."""

    # Define three shelf AOIs in a store (simplified rectangular shelves)
    aois = [
        AOI(
            id="top_shelf",
            contour=np.array(
                [[100, 50], [700, 50], [700, 150], [100, 150]], dtype=np.float32
            ),
        ),
        AOI(
            id="middle_shelf",
            contour=np.array(
                [[100, 250], [700, 250], [700, 350], [100, 350]], dtype=np.float32
            ),
        ),
        AOI(
            id="bottom_shelf",
            contour=np.array(
                [[100, 450], [700, 450], [700, 550], [100, 550]], dtype=np.float32
            ),
        ),
    ]

    # Simulate viewer samples (position and direction at 1 Hz)
    # Viewer stands at position (50, 300) and looks at different shelves
    samples = []

    # First 30 seconds: looking at top shelf
    for i in range(30):
        samples.append(
            ViewerSample(
                position=(50.0, 300.0),
                direction=(0.8, -0.6),  # Looking up-right at top shelf
                timestamp=float(i),
            )
        )

    # Next 20 seconds: looking at middle shelf
    for i in range(30, 50):
        samples.append(
            ViewerSample(
                position=(50.0, 300.0),
                direction=(1.0, 0.0),  # Looking right at middle shelf
                timestamp=float(i),
            )
        )

    # Next 10 seconds: looking at bottom shelf
    for i in range(50, 60):
        samples.append(
            ViewerSample(
                position=(50.0, 300.0),
                direction=(0.8, 0.6),  # Looking down-right at bottom shelf
                timestamp=float(i),
            )
        )

    # Last 40 seconds: not looking at any shelf (looking away)
    for i in range(60, 100):
        samples.append(
            ViewerSample(
                position=(50.0, 300.0),
                direction=(-1.0, 0.0),  # Looking left (away from shelves)
                timestamp=float(i),
            )
        )

    print(f"Generated {len(samples)} viewer samples at 1 Hz")
    print(f"Total session duration: {len(samples)} seconds")

    # Compute attention seconds
    result = compute_attention_seconds(
        samples=samples,
        aois=aois,
        field_of_view_deg=90.0,
        max_range=500.0,
    )

    # Print results
    print("\n=== Attention Tracking Results ===")
    print(f"Total samples: {result.total_samples}")
    print(f"Samples with hits: {result.samples_with_hits}")
    print(f"Samples no winner: {result.samples_no_winner}")
    print(f"Coverage ratio: {result.coverage_ratio:.1%}")

    print("\n=== Per-AOI Results ===")
    for aoi_id, aoi_result in result.aoi_results.items():
        print(
            f"{aoi_id}: {aoi_result.hit_count} hits, "
            f"{aoi_result.total_attention_seconds:.1f} seconds"
        )

    # Create a blank image to visualize on
    img_height, img_width = 600, 800
    blank_image = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

    # Draw heatmap with 'hot' colormap
    heatmap_hot = draw_attention_heatmap(
        blank_image,
        aois,
        result,
        colormap="hot",
        fill_alpha=0.6,
        background_color=(230, 230, 230),  # Light gray for zero-hit AOIs
    )

    # Draw heatmap with 'viridis' colormap
    heatmap_viridis = draw_attention_heatmap(
        blank_image,
        aois,
        result,
        colormap="viridis",
        fill_alpha=0.6,
        background_color=(230, 230, 230),
    )

    # Add labels to the hot heatmap
    heatmap_with_labels = draw_attention_labels(
        heatmap_hot,
        aois,
        result,
        show_hit_count=True,
        show_percentage=True,
        show_seconds=True,
    )

    # Save outputs
    cv2.imwrite("examples/output/attention_heatmap_hot.png", heatmap_hot)
    cv2.imwrite("examples/output/attention_heatmap_viridis.png", heatmap_viridis)
    cv2.imwrite("examples/output/attention_heatmap_with_labels.png", heatmap_with_labels)

    print("\n=== Visualization Outputs ===")
    print("Saved: examples/output/attention_heatmap_hot.png")
    print("Saved: examples/output/attention_heatmap_viridis.png")
    print("Saved: examples/output/attention_heatmap_with_labels.png")

    # Get top AOIs
    top_aois = result.get_top_aois(3)
    attention_dist = result.get_attention_distribution()
    print("\n=== Top 3 AOIs by Attention ===")
    for i, (aoi_id, hit_count) in enumerate(top_aois, 1):
        percentage = attention_dist.get(aoi_id, 0.0)
        print(f"{i}. {aoi_id}: {hit_count} hits ({percentage:.1f}%)")


if __name__ == "__main__":
    main()
