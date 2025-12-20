"""
Streaming Mode Demo
===================

This example demonstrates how to use compute_attention_seconds_streaming()
for processing very long sessions with lower memory overhead.

The streaming mode is beneficial when:
- Processing 5000+ samples (long acquisition sessions)
- Memory constraints are a concern
- Progress monitoring is needed for long-running processing

Use Case: Process a 1-hour session (3600 samples at 1 Hz) with progress updates.
"""

import numpy as np

from view_arc.tracking import AOI, ViewerSample, compute_attention_seconds_streaming


def generate_hour_long_session(num_samples: int = 3600) -> list[ViewerSample]:
    """Generate samples for a 1-hour session at 1 Hz."""
    samples = []
    rng = np.random.default_rng(42)

    # Simulate a viewer walking through a store
    for i in range(num_samples):
        # Position progresses through space
        x = 100 + (i / num_samples) * 500
        y = 200 + rng.uniform(-50, 50)

        # Direction varies as viewer looks around
        angle = rng.uniform(0, 2 * np.pi)
        dx = np.cos(angle)
        dy = np.sin(angle)

        samples.append(ViewerSample(position=(x, y), direction=(dx, dy)))

    return samples


def create_store_aois() -> list[AOI]:
    """Create AOIs representing store shelves."""
    aois = []

    for i in range(10):
        # Create rectangular shelf AOIs distributed through the store
        x_center = 100 + i * 60
        y_center = 150
        width = 40
        height = 30

        contour = np.array(
            [
                [x_center - width / 2, y_center - height / 2],
                [x_center + width / 2, y_center - height / 2],
                [x_center + width / 2, y_center + height / 2],
                [x_center - width / 2, y_center + height / 2],
            ],
            dtype=np.float32,
        )

        aois.append(AOI(id=f"shelf_{i+1}", contour=contour))

    return aois


def main() -> None:
    """Demonstrate streaming mode with progress monitoring."""
    print("=" * 60)
    print("Streaming Mode Demo: 1-Hour Session (3600 samples)")
    print("=" * 60)

    # Generate long session
    print("\nGenerating 3600 samples (1 hour at 1 Hz)...")
    samples = generate_hour_long_session(3600)
    aois = create_store_aois()

    print(f"Processing {len(samples)} samples with {len(aois)} AOIs")
    print(f"Using streaming mode with chunk_size=100")
    print()

    # Process in streaming mode with progress updates
    chunk_count = 0
    for result in compute_attention_seconds_streaming(
        samples, aois, chunk_size=100
    ):
        chunk_count += 1
        progress = (result.total_samples / len(samples)) * 100

        # Print progress update every 5 chunks (every 500 samples)
        if chunk_count % 5 == 0:
            print(f"Progress: {progress:.1f}% ({result.total_samples}/{len(samples)})")
            print(
                f"  Hits so far: {result.samples_with_hits} | "
                f"No winner: {result.samples_no_winner}"
            )

    # Final result is in 'result' from the last iteration
    print()
    print("=" * 60)
    print("Final Results")
    print("=" * 60)
    print(f"Total samples processed: {result.total_samples}")
    print(f"Samples with hits: {result.samples_with_hits}")
    print(f"Samples with no winner: {result.samples_no_winner}")
    print(f"Coverage ratio: {result.coverage_ratio:.1%}")
    print()

    # Show top 3 AOIs
    top_aois = result.get_top_aois(3)
    print("Top 3 AOIs by attention:")
    for i, (aoi_id, hit_count) in enumerate(top_aois, 1):
        total_seconds = result.aoi_results[aoi_id].total_attention_seconds
        print(
            f"  {i}. {aoi_id}: {total_seconds:.1f}s "
            f"({hit_count} hits)"
        )

    print()
    print("Memory Efficiency Notes:")
    print("  - Streaming mode processes samples in chunks")
    print("  - Peak memory is bounded by chunk_size, not total samples")
    print("  - For 3600 samples with chunk_size=100:")
    print("    • Batch mode: O(3600) samples in memory")
    print("    • Streaming mode: O(100) samples in memory at a time")
    print(
        "  - Result accumulation (hit_timestamps) still grows with O(N) "
        "across both modes"
    )


if __name__ == "__main__":
    main()
