"""Attention tracking analysis example with result export.

This example demonstrates how to analyze attention tracking results:
1. Compute attention seconds for a session
2. Export results to pandas DataFrame
3. Compute statistics (top AOIs, attention distribution)
4. Analyze viewing timeline

This shows how to use the result aggregation API for data analysis.

Run with::

    uv run python examples/attention_tracking_analysis.py

Note: pandas is optional. If not installed, DataFrame export will be skipped.
"""

from __future__ import annotations

import numpy as np

from view_arc.tracking import (
    AOI,
    ViewerSample,
    compute_attention_seconds,
)


def main() -> None:
    """Demonstrate result analysis and aggregation methods."""
    
    print("="*70)
    print("ATTENTION TRACKING ANALYSIS EXAMPLE")
    print("="*70)
    print()
    
    # Define some sample AOIs (store shelves)
    print("Step 1: Defining AOIs...")
    aois = [
        AOI(
            id="Shelf_A",
            contour=np.array([
                [100, 100], [200, 100], [200, 200], [100, 200]
            ], dtype=np.float64),
        ),
        AOI(
            id="Shelf_B",
            contour=np.array([
                [300, 100], [400, 100], [400, 200], [300, 200]
            ], dtype=np.float64),
        ),
        AOI(
            id="Shelf_C",
            contour=np.array([
                [500, 100], [600, 100], [600, 200], [500, 200]
            ], dtype=np.float64),
        ),
    ]
    print(f"  Created {len(aois)} AOIs")
    print()
    
    # Create a viewing session with 20 samples
    # Viewer looks at Shelf_A for 10s, Shelf_B for 5s, Shelf_C for 3s, misses for 2s
    print("Step 2: Generating viewer samples...")
    samples = []
    
    # Look at Shelf_A (samples 0-9) - viewer at position 50,150 looking right
    for i in range(10):
        samples.append(ViewerSample(
            position=(50.0, 150.0),
            direction=(1.0, 0.0),  # Looking right at Shelf_A
            timestamp=float(i),
        ))
    
    # Look at Shelf_B (samples 10-14) - viewer moves to be in front of Shelf_B
    for i in range(10, 15):
        samples.append(ViewerSample(
            position=(250.0, 150.0),
            direction=(1.0, 0.0),  # Looking right at Shelf_B
            timestamp=float(i),
        ))
    
    # Look at Shelf_C (samples 15-17) - viewer moves to be in front of Shelf_C
    for i in range(15, 18):
        samples.append(ViewerSample(
            position=(450.0, 150.0),
            direction=(1.0, 0.0),  # Looking right at Shelf_C
            timestamp=float(i),
        ))
    
    # Look away (samples 18-19)
    for i in range(18, 20):
        samples.append(ViewerSample(
            position=(50.0, 150.0),
            direction=(0.0, -1.0),  # Looking down (away from shelves)
            timestamp=float(i),
        ))
    
    print(f"  Generated {len(samples)} samples")
    print()
    
    # Compute attention tracking
    print("Step 3: Computing attention tracking...")
    result = compute_attention_seconds(
        samples=samples,
        aois=aois,
        field_of_view_deg=90.0,
        max_range=1000.0,
    )
    print("  ✓ Complete")
    print()
    
    # === Analysis 1: Top AOIs by hit count ===
    print("="*70)
    print("ANALYSIS 1: TOP AOIs BY HIT COUNT")
    print("="*70)
    top_3 = result.get_top_aois(3)
    for rank, (aoi_id, hit_count) in enumerate(top_3, start=1):
        print(f"  #{rank}: {aoi_id} - {hit_count} hits")
    print()
    
    # === Analysis 2: Attention distribution (percentages) ===
    print("="*70)
    print("ANALYSIS 2: ATTENTION DISTRIBUTION")
    print("="*70)
    distribution = result.get_attention_distribution()
    for aoi_id, percentage in sorted(distribution.items(), 
                                     key=lambda x: x[1], reverse=True):
        print(f"  {aoi_id}: {percentage:.2f}%")
    
    # Verify percentages sum to 100%
    total_pct = sum(distribution.values())
    print(f"\n  Total: {total_pct:.2f}% (should be ~100%)")
    print()
    
    # === Analysis 3: Viewing timeline ===
    print("="*70)
    print("ANALYSIS 3: VIEWING TIMELINE (First 10 samples)")
    print("="*70)
    timeline = result.get_viewing_timeline()
    for sample_idx, aoi_id_maybe in timeline[:10]:
        if aoi_id_maybe is not None:
            aoi_str = f"'{str(aoi_id_maybe)}'"
        else:
            aoi_str = "None (no AOI visible)"
        print(f"  Sample {sample_idx:2d} (t={sample_idx}s): {aoi_str}")
    if len(timeline) > 10:
        print(f"  ... ({len(timeline) - 10} more samples)")
    print()
    
    # === Analysis 4: Session summary statistics ===
    print("="*70)
    print("ANALYSIS 4: SESSION SUMMARY STATISTICS")
    print("="*70)
    print(f"  Total samples: {result.total_samples}")
    print(f"  Samples with hits: {result.samples_with_hits}")
    print(f"  Samples with no winner: {result.samples_no_winner}")
    print(f"  Coverage ratio: {result.coverage_ratio:.2%}")
    print(f"  Session duration: {result.total_samples} seconds")
    print()
    
    # === Analysis 5: Export to DataFrame ===
    print("="*70)
    print("ANALYSIS 5: EXPORT TO PANDAS DATAFRAME")
    print("="*70)
    try:
        df = result.to_dataframe()
        print(df.to_string(index=False))
        print()
        print(f"  DataFrame shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print()
        
        # Show some DataFrame operations
        print("  Example DataFrame operations:")
        print(f"    • Total attention seconds: {df['hit_count'].sum()}")
        print(f"    • Average hits per AOI: {df['hit_count'].mean():.1f}")
        print(f"    • Max hits: {df['hit_count'].max()} ({df.loc[df['hit_count'].idxmax(), 'aoi_id']})")
    except ImportError:
        print("  ⚠ pandas is not installed - skipping DataFrame export")
        print("  Install with: uv pip install pandas")
    print()
    
    print("="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()
