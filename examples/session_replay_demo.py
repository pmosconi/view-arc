"""
Session Replay Visualization Demo
==================================

Demonstrates how to create a session replay showing the progression of viewer
attention over time. Generates individual frames showing:
- Current viewer position
- Current field-of-view wedge
- Currently viewed AOI (highlighted)
- Running hit counts on each AOI

These frames can be saved individually or combined into a video.
"""

import numpy as np

from typing import cast

try:
    import cv2
except ImportError:
    print("This demo requires opencv-python. Install with:")
    print("  pip install opencv-python")
    exit(1)

from view_arc.tracking import (
    AOI,
    ViewerSample,
    draw_session_frame,
    generate_session_replay,
    process_single_sample,
)


def main() -> None:
    """Run the session replay visualization demo."""

    # Create a blank canvas
    canvas_width = 800
    canvas_height = 600
    image = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 240

    # Define three AOIs representing store shelves
    aois = [
        AOI(
            id="shelf_top",
            contour=np.array(
                [[100, 50], [700, 50], [700, 150], [100, 150]], dtype=np.float32
            ),
        ),
        AOI(
            id="shelf_middle",
            contour=np.array(
                [[100, 250], [700, 250], [700, 350], [100, 350]], dtype=np.float32
            ),
        ),
        AOI(
            id="shelf_bottom",
            contour=np.array(
                [[100, 450], [700, 450], [700, 550], [100, 550]], dtype=np.float32
            ),
        ),
    ]

    # Draw AOI outlines on base image
    for aoi in aois:
        pts = aoi.contour.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=True, color=(200, 200, 200), thickness=1)
        # Add AOI label
        centroid = aoi.contour.mean(axis=0)
        cx, cy = int(centroid[0]), int(centroid[1])
        cv2.putText(
            image,
            str(aoi.id),
            (cx - 40, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (150, 150, 150),
            1,
        )

    # Simulate a viewer walking through the store and looking at different shelves
    samples = [
        # Start looking at top shelf
        ViewerSample(position=(400.0, 300.0), direction=(0.0, -1.0)),
        ViewerSample(position=(400.0, 300.0), direction=(0.0, -1.0)),
        ViewerSample(position=(400.0, 300.0), direction=(0.0, -1.0)),
        # Turn to middle shelf (slightly angled)
        ViewerSample(position=(400.0, 300.0), direction=(0.1, -0.995)),
        ViewerSample(position=(400.0, 300.0), direction=(0.2, -0.98)),
        # Look at middle shelf
        ViewerSample(position=(400.0, 350.0), direction=(0.0, -1.0)),
        ViewerSample(position=(400.0, 350.0), direction=(0.0, -1.0)),
        # Move down and look at bottom shelf
        ViewerSample(position=(400.0, 500.0), direction=(0.0, -1.0)),
        ViewerSample(position=(400.0, 500.0), direction=(0.0, -1.0)),
        ViewerSample(position=(400.0, 500.0), direction=(0.0, -1.0)),
    ]

    # Normalize all samples to ensure unit vectors (in case of rounding errors)
    normalized_samples = []
    for sample in samples:
        dx, dy = sample.direction
        norm = np.sqrt(dx * dx + dy * dy)
        if norm > 0.01:  # Valid direction
            dx, dy = dx / norm, dy / norm
        else:
            dx, dy = 0.0, 1.0  # Default direction if too small
        normalized_samples.append(
            ViewerSample(position=sample.position, direction=(dx, dy))
        )

    # Determine winner for each sample
    winner_ids: list[str | int | None] = []
    for sample in normalized_samples:
        winner_id = cast(
            str | int | None,
            process_single_sample(
                sample=sample,
                aois=aois,
                field_of_view_deg=90.0,
                max_range=300.0,
                return_details=False,
            ),
        )
        winner_ids.append(winner_id)

    print("Processing samples:")
    for i, (sample, winner) in enumerate(zip(normalized_samples, winner_ids)):
        print(f"  Sample {i + 1}: Looking at {winner or 'nothing'}")

    # Method 1: Generate individual frame using draw_session_frame
    print("\nMethod 1: Drawing a single frame (sample 5)...")
    sample_idx = 4
    running_counts = {}
    for aoi in aois:
        running_counts[aoi.id] = sum(
            1 for w in winner_ids[: sample_idx + 1] if w == aoi.id
        )

    single_frame = draw_session_frame(
        image=image,
        sample=normalized_samples[sample_idx],
        aois=aois,
        winner_id=winner_ids[sample_idx],
        running_hit_counts=running_counts,
        field_of_view_deg=90.0,
        max_range=300.0,
        sample_index=sample_idx,
        total_samples=len(normalized_samples),
        show_hit_counts=True,
        show_progress=True,
    )

    # Save single frame
    output_path = "examples/output/session_replay_single_frame.png"
    cv2.imwrite(output_path, single_frame)
    print(f"  Saved single frame to {output_path}")

    # Method 2: Generate complete replay sequence using generate_session_replay
    print("\nMethod 2: Generating complete replay sequence...")
    frames = generate_session_replay(
        image=image,
        samples=normalized_samples,
        aois=aois,
        winner_ids=winner_ids,
        field_of_view_deg=90.0,
        max_range=300.0,
        show_hit_counts=True,
        show_progress=True,
    )

    print(f"  Generated {len(frames)} frames")

    # Save first, middle, and last frames
    frame_indices = [0, len(frames) // 2, len(frames) - 1]
    for idx in frame_indices:
        output_path = f"examples/output/session_replay_frame_{idx:02d}.png"
        cv2.imwrite(output_path, frames[idx])
        print(f"  Saved frame {idx} to {output_path}")

    # Optional: Create an animated GIF or video
    # This requires imageio or similar library
    try:
        import imageio

        output_path = "examples/output/session_replay.gif"
        # Repeat each frame 3 times for slower animation (300ms per frame)
        repeated_frames = [frame for frame in frames for _ in range(3)]
        imageio.mimsave(output_path, cast(list, repeated_frames), fps=10, loop=0)
        print(f"\n  Created animated GIF: {output_path}")
    except ImportError:
        print(
            "\n  To create an animated GIF, install imageio: pip install imageio[ffmpeg]"
        )

    print("\nSession replay visualization demo complete!")
    print("Check examples/output/ for generated images")


if __name__ == "__main__":
    main()
