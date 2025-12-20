"""
Profile tracking performance to identify optimization opportunities.

This script profiles the compute_attention_seconds function to identify
bottlenecks and potential optimization opportunities as specified in
Step 6.2 of the tracking plan.
"""

import cProfile
import io
import pstats
from typing import Any

import numpy as np

from view_arc.tracking import AOI, ViewerSample, compute_attention_seconds


def generate_samples(n: int, seed: int = 42) -> list[ViewerSample]:
    """Generate viewer samples for profiling."""
    rng = np.random.default_rng(seed)
    samples = []
    for i in range(n):
        x = 100 + i * 2.0
        y = 100 + rng.uniform(-10, 10)
        angle = np.pi / 2 + rng.uniform(-0.1, 0.1)
        dx = np.cos(angle)
        dy = np.sin(angle)
        samples.append(ViewerSample(position=(x, y), direction=(dx, dy)))
    return samples


def generate_aois(n: int, seed: int = 42) -> list[AOI]:
    """Generate AOIs for profiling."""
    rng = np.random.default_rng(seed)
    aois = []
    for i in range(n):
        x_center = 100 + i * 100
        y_center = 200
        width = rng.uniform(30, 50)
        height = rng.uniform(20, 40)

        contour = np.array(
            [
                [x_center - width / 2, y_center - height / 2],
                [x_center + width / 2, y_center - height / 2],
                [x_center + width / 2, y_center + height / 2],
                [x_center - width / 2, y_center + height / 2],
            ],
            dtype=np.float32,
        )
        aois.append(AOI(id=f"aoi_{i}", contour=contour))
    return aois


def profile_scenario(
    n_samples: int, n_aois: int, description: str
) -> dict[str, Any]:
    """Profile a specific scenario and return statistics."""
    print(f"\n{'='*70}")
    print(f"Profiling: {description}")
    print(f"  Samples: {n_samples}, AOIs: {n_aois}")
    print(f"{'='*70}")

    samples = generate_samples(n_samples)
    aois = generate_aois(n_aois)

    # Profile the computation
    profiler = cProfile.Profile()
    profiler.enable()

    result = compute_attention_seconds(
        samples, aois, enable_profiling=False
    )

    profiler.disable()

    # Print profiling statistics
    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s)
    stats.sort_stats("cumulative")
    stats.print_stats(20)  # Top 20 functions

    print(s.getvalue())

    # Print profiling data from the result
    if result.profiling_data:
        print("\nInternal profiling data:")
        print(result.profiling_data)

    return {
        "samples": n_samples,
        "aois": n_aois,
        "profiling_data": result.profiling_data,
        "stats": stats,
    }


def main():
    """Run profiling scenarios."""
    print("Performance Profiling for Step 6.2: Batch Optimization Opportunities")
    print("="*70)

    # Scenario 1: Long session (typical use case)
    profile_scenario(
        n_samples=300,
        n_aois=20,
        description="Long session (5 min, typical AOI count)"
    )

    # Scenario 2: Many AOIs
    profile_scenario(
        n_samples=100,
        n_aois=50,
        description="Many AOIs (demanding workload)"
    )

    # Scenario 3: Very long session with moderate AOIs
    profile_scenario(
        n_samples=600,
        n_aois=10,
        description="Very long session (10 min)"
    )

    # Scenario 4: Most demanding
    profile_scenario(
        n_samples=300,
        n_aois=50,
        description="Demanding workload (300 samples × 50 AOIs)"
    )

    print("\n" + "="*70)
    print("ANALYSIS SUMMARY")
    print("="*70)
    print("""
Based on the profiling output above, we can identify:

1. **Hot spots**: Which functions consume the most time?
   - Is it the sweep algorithm?
   - Is it polygon clipping?
   - Is it the coordinate transformations?

2. **Potential optimizations**:
   - Pre-filter AOIs outside max_range from viewer position
   - Cache AOI bounding boxes (computed per call)
   - Vectorize operations where possible
   - Early exit for samples clearly outside all AOI regions

3. **Trade-offs**:
   - Optimization complexity vs. performance gain
   - Memory overhead vs. speed improvement
   - Accuracy preservation vs. approximation

Next steps:
- Review the top functions by cumulative time
- Identify if any pre-computation helps
- Consider implementing minimal optimizations if bottlenecks are clear
""")


if __name__ == "__main__":
    main()
