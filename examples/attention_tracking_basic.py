"""Basic attention tracking example - minimal demonstration.

This example demonstrates the minimal usage of the attention tracking API:
1. Define a few AOIs (areas of interest)
2. Create a sequence of viewer samples
3. Compute attention seconds for each AOI
4. Print results

This is the simplest possible example showing batch attention tracking over time.

Run with::

    uv run python examples/attention_tracking_basic.py
"""

from __future__ import annotations

import numpy as np

from view_arc.tracking import (
    AOI,
    ViewerSample,
    compute_attention_seconds,
)


def main() -> None:
    """Run a minimal attention tracking demonstration."""
    
    print("="*60)
    print("BASIC ATTENTION TRACKING EXAMPLE")
    print("="*60)
    print()
    
    # Step 1: Define some AOIs (store shelves)
    print("Step 1: Defining 3 shelf AOIs...")
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
    print(f"  Created {len(aois)} AOIs: {[aoi.id for aoi in aois]}")
    print()
    
    # Step 2: Create viewer samples (position + direction at 1 Hz)
    print("Step 2: Creating viewer samples (10 seconds of data)...")
    samples = []
    
    # Viewer stands at (50, 150) and looks at different shelves
    # Look at Shelf_A for 5 seconds
    for i in range(5):
        samples.append(ViewerSample(
            position=(50.0, 150.0),
            direction=(1.0, 0.0),  # Looking right toward Shelf_A
            timestamp=float(i),
        ))
    
    # Look at Shelf_B for 3 seconds
    for i in range(5, 8):
        samples.append(ViewerSample(
            position=(50.0, 150.0),
            direction=(1.0, 0.0),  # Looking right toward Shelf_B
            timestamp=float(i),
        ))
    
    # Look away for 2 seconds (no AOI visible)
    for i in range(8, 10):
        samples.append(ViewerSample(
            position=(50.0, 150.0),
            direction=(0.0, -1.0),  # Looking down
            timestamp=float(i),
        ))
    
    print(f"  Created {len(samples)} viewer samples")
    print()
    
    # Step 3: Compute attention seconds
    print("Step 3: Computing attention seconds...")
    result = compute_attention_seconds(
        samples=samples,
        aois=aois,
        field_of_view_deg=90.0,
        max_range=500.0,
    )
    print("  ✓ Computation complete")
    print()
    
    # Step 4: Print results
    print("Step 4: Results")
    print("-"*60)
    print(f"Total samples processed: {result.total_samples}")
    print(f"Samples with AOI hits: {result.samples_with_hits}")
    print(f"Samples with no hits: {result.samples_no_winner}")
    print(f"Coverage ratio: {result.coverage_ratio:.1%}")
    print()
    
    print("Per-AOI attention:")
    for aoi_id in ["Shelf_A", "Shelf_B", "Shelf_C"]:
        aoi_result = result.aoi_results[aoi_id]
        percentage = (aoi_result.hit_count / result.samples_with_hits * 100 
                     if result.samples_with_hits > 0 else 0.0)
        print(f"  {aoi_id}: {aoi_result.hit_count} seconds ({percentage:.1f}%)")
    
    print()
    print("Top AOIs by attention:")
    for rank, (top_aoi_id, hit_count) in enumerate(result.get_top_aois(3), 1):
        print(f"  #{rank}: {top_aoi_id} - {hit_count} seconds")
    
    print()
    print("="*60)
    print("Example complete!")
    print("="*60)


if __name__ == "__main__":
    main()
