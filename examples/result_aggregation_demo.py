"""
Example demonstrating Step 3.1: Result Aggregation Methods

This example shows how to use the new aggregation methods added to TrackingResult:
- get_top_aois(n) - Get top N AOIs by hit count
- get_attention_distribution() - Get percentage distribution of attention
- get_viewing_timeline() - Get chronological sequence of viewed AOIs
- to_dataframe() - Export results to pandas DataFrame
"""

import numpy as np

from view_arc.tracking import (
    AOI,
    ViewerSample,
    compute_attention_seconds,
)


def main() -> None:
    """Demonstrate result aggregation methods."""
    
    # Define some sample AOIs (store shelves)
    aois = [
        AOI(
            id="Shelf_A",
            contour=np.array([[100, 100], [200, 100], [200, 200], [100, 200]], dtype=np.float32),
        ),
        AOI(
            id="Shelf_B",
            contour=np.array([[300, 100], [400, 100], [400, 200], [300, 200]], dtype=np.float32),
        ),
        AOI(
            id="Shelf_C",
            contour=np.array([[500, 100], [600, 100], [600, 200], [500, 200]], dtype=np.float32),
        ),
    ]
    
    # Create a viewing session with 20 samples
    # Viewer looks at Shelf_A for 10s, Shelf_B for 5s, Shelf_C for 3s, and misses for 2s
    samples = []
    
    # Look at Shelf_A (samples 0-9) - viewer at position 50,150 looking right
    for i in range(10):
        samples.append(
            ViewerSample(
                position=(50.0, 150.0),
                direction=(1.0, 0.0),  # Looking right at Shelf_A (centered at x=150)
                timestamp=float(i),
            )
        )
    
    # Look at Shelf_B (samples 10-14) - viewer moves to be in front of Shelf_B
    for i in range(10, 15):
        samples.append(
            ViewerSample(
                position=(250.0, 150.0),
                direction=(1.0, 0.0),  # Looking right at Shelf_B (centered at x=350)
                timestamp=float(i),
            )
        )
    
    # Look at Shelf_C (samples 15-17) - viewer moves to be in front of Shelf_C
    for i in range(15, 18):
        samples.append(
            ViewerSample(
                position=(450.0, 150.0),
                direction=(1.0, 0.0),  # Looking right at Shelf_C (centered at x=550)
                timestamp=float(i),
            )
        )
    
    # Look away (samples 18-19)
    for i in range(18, 20):
        samples.append(
            ViewerSample(
                position=(50.0, 150.0),
                direction=(0.0, -1.0),  # Looking down (away from shelves)
                timestamp=float(i),
            )
        )
    
    # Compute attention tracking
    result = compute_attention_seconds(
        samples=samples,
        aois=aois,
        field_of_view_deg=90.0,
        max_range=1000.0,
    )
    
    print("=" * 70)
    print("TRACKING RESULT AGGREGATION METHODS DEMO")
    print("=" * 70)
    print()
    
    # 1. Get top AOIs by hit count
    print("1. TOP 3 AOIs BY HIT COUNT")
    print("-" * 70)
    top_3 = result.get_top_aois(3)
    for rank, (aoi_id, hit_count) in enumerate(top_3, start=1):
        print(f"   #{rank}: {aoi_id} - {hit_count} hits")
    print()
    
    # 2. Get attention distribution
    print("2. ATTENTION DISTRIBUTION (PERCENTAGES)")
    print("-" * 70)
    distribution = result.get_attention_distribution()
    for aoi_id, percentage in distribution.items():
        print(f"   {aoi_id}: {percentage:.2f}%")
    print()
    
    # 3. Get viewing timeline (first 10 samples)
    print("3. VIEWING TIMELINE (First 10 samples)")
    print("-" * 70)
    timeline = result.get_viewing_timeline()
    for sample_idx, aoi_id_maybe in timeline[:10]:
        if aoi_id_maybe is not None:
            aoi_str = f"'{str(aoi_id_maybe)}'"
        else:
            aoi_str = "None (no AOI)"
        print(f"   Sample {sample_idx:2d}: {aoi_str}")
    print("   ...")
    print()
    
    # 4. Export to DataFrame (if pandas is available)
    print("4. EXPORT TO PANDAS DATAFRAME")
    print("-" * 70)
    try:
        df = result.to_dataframe()
        print(df.to_string(index=False))
        print()
        print(f"   DataFrame shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
    except ImportError:
        print("   pandas is not installed - skipping DataFrame export")
        print("   Install with: pip install pandas")
    print()
    
    # 5. Summary statistics
    print("5. SESSION SUMMARY")
    print("-" * 70)
    print(f"   Total samples: {result.total_samples}")
    print(f"   Samples with hits: {result.samples_with_hits}")
    print(f"   Samples with no winner: {result.samples_no_winner}")
    print(f"   Coverage ratio: {result.coverage_ratio:.2%}")
    print()
    
    print("=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
