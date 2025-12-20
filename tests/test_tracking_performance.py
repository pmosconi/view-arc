"""
Performance tests for tracking module.

Tests verify:
1. Profiling instrumentation does not alter results (Step 6.1)
2. Performance is acceptable for long sessions (300+ samples)
3. Performance scales with many AOIs (50+)
"""

import time
from typing import List

import numpy as np
import pytest

from view_arc.tracking import AOI, ProfilingData, ViewerSample, compute_attention_seconds


def generate_simple_samples(n: int, seed: int = 42) -> list[ViewerSample]:
    """Generate simple viewer samples for testing."""
    rng = np.random.default_rng(seed)
    samples = []
    for i in range(n):
        x = 100 + i * 2.0
        y = 100 + rng.uniform(-10, 10)
        # Direction pointing up with slight variation
        angle = np.pi / 2 + rng.uniform(-0.1, 0.1)
        dx = np.cos(angle)
        dy = np.sin(angle)
        samples.append(ViewerSample(position=(x, y), direction=(dx, dy)))
    return samples


def generate_simple_aois(n: int, seed: int = 42) -> list[AOI]:
    """Generate simple rectangular AOIs for testing."""
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


class TestProfilingInstrumentation:
    """Test that profiling instrumentation does not affect results (Step 6.1)."""

    def test_profile_hook_smoke(self) -> None:
        """Ensure profiling flag does not alter results.

        This is the smoke test mentioned in Step 6.1 of the tracking plan.
        Verifies that enable_profiling=True produces identical tracking results
        to enable_profiling=False.
        """
        samples = generate_simple_samples(50)
        aois = generate_simple_aois(5)

        # Run without profiling
        result_no_profile = compute_attention_seconds(
            samples, aois, enable_profiling=False
        )

        # Run with profiling
        result_with_profile = compute_attention_seconds(
            samples, aois, enable_profiling=True
        )

        # Verify core results are identical
        assert result_no_profile.total_samples == result_with_profile.total_samples
        assert (
            result_no_profile.samples_with_hits == result_with_profile.samples_with_hits
        )
        assert (
            result_no_profile.samples_no_winner == result_with_profile.samples_no_winner
        )

        # Verify per-AOI results are identical
        for aoi_id in result_no_profile.aoi_results:
            no_prof = result_no_profile.aoi_results[aoi_id]
            with_prof = result_with_profile.aoi_results[aoi_id]

            assert no_prof.hit_count == with_prof.hit_count
            assert no_prof.total_attention_seconds == with_prof.total_attention_seconds
            assert no_prof.hit_timestamps == with_prof.hit_timestamps

        # Verify profiling data is present when enabled
        assert result_no_profile.profiling_data is None
        assert result_with_profile.profiling_data is not None
        assert isinstance(result_with_profile.profiling_data, ProfilingData)

    def test_profiling_data_structure(self) -> None:
        """Verify ProfilingData structure and derived metrics."""
        samples = generate_simple_samples(100)
        aois = generate_simple_aois(10)

        result = compute_attention_seconds(samples, aois, enable_profiling=True)

        assert result.profiling_data is not None
        prof = result.profiling_data

        # Check required fields
        assert prof.total_time_seconds > 0
        assert prof.samples_processed == 100

        # Check derived metrics are calculated
        assert prof.samples_per_second > 0
        assert prof.avg_time_per_sample_ms > 0

        # Check that throughput makes sense
        expected_throughput = 100 / prof.total_time_seconds
        assert abs(prof.samples_per_second - expected_throughput) < 0.01

    def test_profiling_data_repr(self) -> None:
        """Verify ProfilingData has human-readable repr."""
        samples = generate_simple_samples(50)
        aois = generate_simple_aois(5)

        result = compute_attention_seconds(samples, aois, enable_profiling=True)

        assert result.profiling_data is not None
        repr_str = repr(result.profiling_data)

        # Check that repr contains key information
        assert "ProfilingData:" in repr_str
        assert "Total time:" in repr_str
        assert "Samples:" in repr_str
        assert "Throughput:" in repr_str
        assert "Avg time/sample:" in repr_str


class TestPerformanceLongSession:
    """Test performance with long sessions (300+ samples)."""

    def test_performance_300_samples_20_aois(self) -> None:
        """Test with 300 samples (5 min session) and 20 AOIs.

        This represents a typical real-world session.
        Target: < 1s total runtime.
        """
        samples = generate_simple_samples(300)
        aois = generate_simple_aois(20)

        start = time.perf_counter()
        result = compute_attention_seconds(samples, aois, enable_profiling=True)
        elapsed = time.perf_counter() - start

        assert result.total_samples == 300
        assert elapsed < 1.0, f"300 samples took {elapsed:.3f}s, expected < 1s"

        # Verify profiling data matches
        assert result.profiling_data is not None
        assert result.profiling_data.samples_processed == 300

    def test_performance_600_samples_10_aois(self) -> None:
        """Test with 600 samples (10 min session) and 10 AOIs.

        Target: < 2s total runtime.
        """
        samples = generate_simple_samples(600)
        aois = generate_simple_aois(10)

        start = time.perf_counter()
        result = compute_attention_seconds(samples, aois, enable_profiling=True)
        elapsed = time.perf_counter() - start

        assert result.total_samples == 600
        assert elapsed < 2.0, f"600 samples took {elapsed:.3f}s, expected < 2s"


class TestPerformanceManyAOIs:
    """Test performance with many AOIs (50+)."""

    def test_performance_100_samples_50_aois(self) -> None:
        """Test with 100 samples and 50 AOIs.

        Target: < 1s total runtime.
        """
        samples = generate_simple_samples(100)
        aois = generate_simple_aois(50)

        start = time.perf_counter()
        result = compute_attention_seconds(samples, aois, enable_profiling=True)
        elapsed = time.perf_counter() - start

        assert result.total_samples == 100
        assert len(result.aoi_results) == 50
        assert elapsed < 1.0, f"100 samples × 50 AOIs took {elapsed:.3f}s, expected < 1s"

    def test_performance_300_samples_50_aois(self) -> None:
        """Test with 300 samples and 50 AOIs.

        This is a demanding workload.
        Target: < 2s total runtime.
        """
        samples = generate_simple_samples(300)
        aois = generate_simple_aois(50)

        start = time.perf_counter()
        result = compute_attention_seconds(samples, aois, enable_profiling=True)
        elapsed = time.perf_counter() - start

        assert result.total_samples == 300
        assert len(result.aoi_results) == 50
        assert elapsed < 2.0, f"300 samples × 50 AOIs took {elapsed:.3f}s, expected < 2s"


class TestPerformanceComplexContours:
    """Test performance with complex AOI contours (many vertices)."""

    def generate_complex_aois(
        self, n: int, vertices_per_aoi: int = 20, seed: int = 42
    ) -> list[AOI]:
        """Generate AOIs with many vertices."""
        rng = np.random.default_rng(seed)
        aois = []

        for i in range(n):
            x_center = 100 + i * 100
            y_center = 200
            radius = 30

            # Create polygon with many vertices
            angles = np.linspace(0, 2 * np.pi, vertices_per_aoi, endpoint=False)
            # Add some irregularity
            radii = radius + rng.uniform(-5, 5, vertices_per_aoi)

            x = x_center + radii * np.cos(angles)
            y = y_center + radii * np.sin(angles)
            contour = np.column_stack([x, y]).astype(np.float32)

            aois.append(AOI(id=f"aoi_{i}", contour=contour))

        return aois

    def test_performance_complex_aoi_contours(self) -> None:
        """Test with AOIs having many vertices (20 each).

        Target: < 1.5s for 200 samples × 10 complex AOIs.
        """
        samples = generate_simple_samples(200)
        aois = self.generate_complex_aois(10, vertices_per_aoi=20)

        start = time.perf_counter()
        result = compute_attention_seconds(samples, aois, enable_profiling=True)
        elapsed = time.perf_counter() - start

        assert result.total_samples == 200
        assert (
            elapsed < 1.5
        ), f"200 samples × 10 complex AOIs took {elapsed:.3f}s, expected < 1.5s"


class TestProfilingMetricsAccuracy:
    """Test that profiling metrics are accurate."""

    def test_profiling_samples_per_second_realistic(self) -> None:
        """Verify samples_per_second metric is in a realistic range."""
        samples = generate_simple_samples(100)
        aois = generate_simple_aois(10)

        result = compute_attention_seconds(samples, aois, enable_profiling=True)

        assert result.profiling_data is not None
        # Should process at least 100 samples/sec (conservative lower bound)
        assert result.profiling_data.samples_per_second > 100
        # But not impossibly fast (e.g., not > 100k samples/sec)
        assert result.profiling_data.samples_per_second < 100000

    def test_profiling_avg_time_per_sample_reasonable(self) -> None:
        """Verify avg_time_per_sample is in reasonable range."""
        samples = generate_simple_samples(100)
        aois = generate_simple_aois(10)

        result = compute_attention_seconds(samples, aois, enable_profiling=True)

        assert result.profiling_data is not None
        # Each sample should take < 10ms on average
        assert result.profiling_data.avg_time_per_sample_ms < 10
        # But not impossibly fast (> 0.001ms)
        assert result.profiling_data.avg_time_per_sample_ms > 0.001
