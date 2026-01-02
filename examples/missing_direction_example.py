"""
Handling Missing Direction Data - Complete Example

Demonstrates how to handle samples with missing or invalid direction data
using the allow_missing_direction feature. This is useful when working with
real-world tracking systems that may have occasional dropout or sensor failures.

Key features demonstrated:
1. NumPy array input with batch-level flag
2. List input with per-sample flags
3. Mixed valid and missing directions
4. Result interpretation and data quality analysis
5. Streaming mode with missing directions

Run with: uv run python examples/missing_direction_example.py
"""

import numpy as np
from numpy.typing import NDArray

from view_arc.tracking import (
    AOI,
    TrackingResult,
    ViewerSample,
    compute_attention_seconds,
    compute_attention_seconds_streaming,
)


def create_store_layout() -> list[AOI]:
    """Create a simple store layout with 3 shelves."""
    return [
        AOI(
            id="shelf_A",
            contour=np.array(
                [[100, 80], [200, 80], [200, 120], [100, 120]], dtype=np.float64
            ),
        ),
        AOI(
            id="shelf_B",
            contour=np.array(
                [[300, 80], [400, 80], [400, 120], [300, 120]], dtype=np.float64
            ),
        ),
        AOI(
            id="shelf_C",
            contour=np.array(
                [[500, 80], [600, 80], [600, 120], [500, 120]], dtype=np.float64
            ),
        ),
    ]


def example_1_numpy_batch_mode() -> None:
    """Example 1: NumPy array with batch-level allow_missing_direction flag."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: NumPy Array - Batch-Level Missing Direction Handling")
    print("=" * 70)

    aois = create_store_layout()

    # Simulate tracking session with some dropout
    # Viewer walks along Y=200, looking up at shelves at Y=100
    samples = np.array(
        [
            [150.0, 200.0, 0.0, -1.0],  # t=0: Valid - looking at shelf_A
            [150.0, 200.0, 0.0, -1.0],  # t=1: Valid - still at shelf_A
            [200.0, 200.0, 0.0, 0.0],  # t=2: MISSING - tracking dropout
            [250.0, 200.0, 0.0, 0.0],  # t=3: MISSING - still no direction
            [300.0, 200.0, 0.0, -1.0],  # t=4: Valid - recovered, at shelf_B
            [350.0, 200.0, 0.0, -1.0],  # t=5: Valid - still at shelf_B
            [400.0, 200.0, 0.0, 0.0],  # t=6: MISSING - brief dropout
            [450.0, 200.0, 0.0, -1.0],  # t=7: Valid - recovered, near shelf_C
            [550.0, 200.0, 0.0, -1.0],  # t=8: Valid - at shelf_C
            [550.0, 200.0, 0.0, -1.0],  # t=9: Valid - still at shelf_C
        ]
    )

    # Process with missing direction handling enabled
    result = compute_attention_seconds(
        samples, aois, field_of_view_deg=90.0, allow_missing_direction=True
    )

    # Analyze results
    print(f"\n📊 Session Statistics:")
    print(f"  Total samples: {result.total_samples}")
    print(f"  Valid samples with hits: {result.samples_with_hits}")
    print(f"  Samples with no winner: {result.samples_no_winner}")
    print(
        f"  Data quality: {result.samples_with_hits / result.total_samples:.1%} usable"
    )

    print(f"\n🛒 Shelf Attention:")
    for aoi_id, aoi_result in result.aoi_results.items():
        print(
            f"  {aoi_id}: {aoi_result.hit_count} seconds "
            f"({aoi_result.hit_count / result.total_samples * 100:.1f}% of session)"
        )

    print(f"\n⚠️  Missing direction samples: {samples_with_zero_direction(samples)}")
    print(
        "  These samples preserved position data but contributed no attention metrics"
    )


def example_2_list_per_sample_flags() -> None:
    """Example 2: List input with per-sample allow_missing_direction flags."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: List Input - Per-Sample Missing Direction Flags")
    print("=" * 70)

    aois = create_store_layout()

    # Create samples with individual flags
    # This allows fine-grained control: some missing directions allowed, others not
    samples = [
        ViewerSample(
            position=(150.0, 200.0), direction=(0.0, -1.0)
        ),  # t=0: Valid, looking at shelf_A
        ViewerSample(
            position=(150.0, 200.0), direction=(0.0, 0.0), allow_missing_direction=True
        ),  # t=1: Missing but allowed
        ViewerSample(
            position=(350.0, 200.0), direction=(0.0, -1.0)
        ),  # t=2: Valid, looking at shelf_B
        ViewerSample(
            position=(350.0, 200.0), direction=(0.0, 0.0), allow_missing_direction=True
        ),  # t=3: Missing but allowed
        ViewerSample(
            position=(550.0, 200.0), direction=(0.0, -1.0)
        ),  # t=4: Valid, looking at shelf_C
    ]

    # Process - no global flag needed since samples have individual flags
    result = compute_attention_seconds(samples, aois, field_of_view_deg=90.0)

    print(f"\n📊 Session Statistics:")
    print(f"  Total samples: {result.total_samples}")
    print(f"  Valid samples: {result.samples_with_hits}")
    print(f"  Missing direction samples: {result.samples_no_winner}")

    print(f"\n🛒 Per-Shelf Results:")
    for aoi_id, aoi_result in result.aoi_results.items():
        if aoi_result.hit_count > 0:
            print(f"  {aoi_id}: {aoi_result.hit_count} seconds of attention")


def example_3_mixed_quality_analysis() -> None:
    """Example 3: Analyzing data quality with mixed valid/missing samples."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Data Quality Analysis")
    print("=" * 70)

    aois = create_store_layout()

    # Simulate poor tracking quality (40% dropout rate)
    np.random.seed(42)
    n_samples = 20
    positions_x = np.linspace(150, 550, n_samples)
    positions_y = np.full(n_samples, 200.0)
    directions_x = np.zeros(n_samples)
    directions_y = np.full(n_samples, -1.0)

    # Randomly set 40% to missing direction
    dropout_mask = np.random.random(n_samples) < 0.4
    directions_x[dropout_mask] = 0.0
    directions_y[dropout_mask] = 0.0

    samples = np.column_stack([positions_x, positions_y, directions_x, directions_y])

    result = compute_attention_seconds(
        samples, aois, field_of_view_deg=90.0, allow_missing_direction=True
    )

    # Quality metrics
    valid_ratio = result.samples_with_hits / result.total_samples
    missing_ratio = result.samples_no_winner / result.total_samples

    print(f"\n📈 Data Quality Metrics:")
    print(f"  Total samples collected: {result.total_samples}")
    print(f"  Valid samples (with direction): {result.samples_with_hits} ({valid_ratio:.1%})")
    print(f"  Missing direction samples: {result.samples_no_winner} ({missing_ratio:.1%})")

    print(f"\n✅ Quality Assessment:")
    if valid_ratio >= 0.8:
        print("  EXCELLENT - High quality tracking data")
    elif valid_ratio >= 0.6:
        print("  GOOD - Acceptable tracking quality")
    elif valid_ratio >= 0.4:
        print("  FAIR - Consider improving sensor setup")
    else:
        print("  POOR - High dropout rate, results may be unreliable")

    print(f"\n🛒 Attention Distribution (from valid samples only):")
    top_aois = result.get_top_aois(n=3)
    for aoi_id, hit_count in top_aois:
        # Calculate percentage relative to valid samples only
        pct = hit_count / result.samples_with_hits * 100
        print(f"  {aoi_id}: {hit_count} hits ({pct:.1f}% of valid samples)")


def example_4_streaming_mode() -> None:
    """Example 4: Streaming mode with missing directions."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Streaming Mode with Missing Directions")
    print("=" * 70)

    aois = create_store_layout()

    # Large sample set (simulate 30 seconds at 1 Hz)
    samples = np.array(
        [
            [150.0, 200.0, 0.0, -1.0],  # shelf_A area
            [150.0, 200.0, 0.0, 0.0],  # missing
            [200.0, 200.0, 0.0, -1.0],
            [250.0, 200.0, 0.0, 0.0],  # missing
            [300.0, 200.0, 0.0, -1.0],  # shelf_B area
            [350.0, 200.0, 0.0, -1.0],
            [350.0, 200.0, 0.0, 0.0],  # missing
            [400.0, 200.0, 0.0, -1.0],
            [450.0, 200.0, 0.0, -1.0],  # shelf_C area
            [500.0, 200.0, 0.0, 0.0],  # missing
            [550.0, 200.0, 0.0, -1.0],
            [550.0, 200.0, 0.0, -1.0],
        ]
    )

    print(f"\n🔄 Processing {len(samples)} samples in streaming mode...")
    print("  (chunk_size=4, allow_missing_direction=True)")

    # Process in streaming mode
    result_gen = compute_attention_seconds_streaming(
        samples, aois, field_of_view_deg=90.0, chunk_size=4, allow_missing_direction=True
    )

    # Consume generator and show progress
    chunk_num = 0
    for partial_result in result_gen:
        chunk_num += 1
        print(
            f"    Chunk {chunk_num}: {partial_result.total_samples} samples processed, "
            f"{partial_result.samples_with_hits} hits, "
            f"{partial_result.samples_no_winner} no-winner"
        )

    # Final result is the last yielded value
    final_result = partial_result

    print(f"\n✅ Streaming Complete:")
    print(f"  Total samples: {final_result.total_samples}")
    print(f"  Samples with hits: {final_result.samples_with_hits}")
    print(f"  Samples no winner: {final_result.samples_no_winner}")

    print(f"\n🛒 Final Attention Results:")
    for aoi_id, aoi_result in final_result.aoi_results.items():
        if aoi_result.hit_count > 0:
            print(f"  {aoi_id}: {aoi_result.hit_count} seconds")


def example_5_error_handling() -> None:
    """Example 5: What happens when missing directions are NOT allowed."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Error Handling (allow_missing_direction=False)")
    print("=" * 70)

    aois = create_store_layout()

    samples = np.array(
        [
            [150.0, 200.0, 0.0, -1.0],  # Valid
            [150.0, 200.0, 0.0, 0.0],  # Missing - will cause error
        ]
    )

    print("\n⚠️  Attempting to process with allow_missing_direction=False (default)...")

    try:
        result = compute_attention_seconds(
            samples,
            aois,
            field_of_view_deg=90.0,
            allow_missing_direction=False,  # Explicit default
        )
        print("  ❌ Unexpected: Should have raised an error!")
    except Exception as e:
        print(f"  ✅ Expected error caught: {type(e).__name__}")
        print(f"     Message: {str(e)}")
        print(
            "\n  💡 Solution: Set allow_missing_direction=True to handle missing directions"
        )


def samples_with_zero_direction(samples: NDArray[np.floating]) -> int:
    """Count samples with zero-magnitude direction vectors."""
    directions = samples[:, 2:4]
    magnitudes = np.sqrt(directions[:, 0] ** 2 + directions[:, 1] ** 2)
    return int(np.sum(magnitudes == 0))


def main() -> None:
    """Run all examples demonstrating missing direction handling."""
    print("\n" + "=" * 70)
    print("MISSING DIRECTION HANDLING - COMPREHENSIVE EXAMPLES")
    print("=" * 70)
    print("\nThis script demonstrates how to handle samples with missing or")
    print("invalid direction data in real-world tracking scenarios.")

    # Run all examples
    example_1_numpy_batch_mode()
    example_2_list_per_sample_flags()
    example_3_mixed_quality_analysis()
    example_4_streaming_mode()
    example_5_error_handling()

    # Summary
    print("\n" + "=" * 70)
    print("📚 KEY TAKEAWAYS")
    print("=" * 70)
    print("\n1. Missing directions (0, 0) can be handled gracefully")
    print("2. NumPy arrays use batch-level allow_missing_direction=True flag")
    print("3. List inputs support per-sample flags for fine-grained control")
    print("4. Missing direction samples count toward samples_no_winner")
    print("5. Position data is preserved for path analysis")
    print("6. Streaming mode fully supports missing directions")
    print("7. Use for handling dropout/occlusions in real-world systems")
    print("\n✅ All examples completed successfully!\n")


if __name__ == "__main__":
    main()
