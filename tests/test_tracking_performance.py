"""
Performance tests for tracking module.

Tests verify:
1. Profiling instrumentation does not alter results (Step 6.1)
2. Performance is acceptable for long sessions (300+ samples)
3. Performance scales with many AOIs (50+)
4. Memory efficiency and streaming mode (Step 6.4)
"""

import time
from typing import List

import numpy as np
import pytest

from view_arc.tracking import (
    AOI,
    ProfilingData,
    ViewerSample,
    compute_attention_seconds,
    compute_attention_seconds_streaming,
)


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

        # Check peak memory tracking (Step 6.1 requirement)
        assert prof.peak_memory_bytes is not None
        assert prof.peak_memory_bytes > 0
        # Memory should be reasonable (not more than 1GB for this small workload)
        assert prof.peak_memory_bytes < 1024 * 1024 * 1024

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
        Target: < 1.0s production runtime (without profiling overhead).
        """
        samples = generate_simple_samples(300)
        aois = generate_simple_aois(20)

        start = time.perf_counter()
        result = compute_attention_seconds(samples, aois, enable_profiling=False)
        elapsed = time.perf_counter() - start

        assert result.total_samples == 300
        assert elapsed < 1.0, f"300 samples took {elapsed:.3f}s, expected < 1.0s"

        # Verify no profiling data when profiling is disabled
        assert result.profiling_data is None

    def test_performance_600_samples_10_aois(self) -> None:
        """Test with 600 samples (10 min session) and 10 AOIs.

        Target: < 2.5s production runtime (without profiling overhead).
        """
        samples = generate_simple_samples(600)
        aois = generate_simple_aois(10)

        start = time.perf_counter()
        result = compute_attention_seconds(samples, aois, enable_profiling=False)
        elapsed = time.perf_counter() - start

        assert result.total_samples == 600
        assert elapsed < 2.5, f"600 samples took {elapsed:.3f}s, expected < 2.5s"


class TestPerformanceManyAOIs:
    """Test performance with many AOIs (50+)."""

    def test_performance_100_samples_50_aois(self) -> None:
        """Test with 100 samples and 50 AOIs.

        Target: < 1.0s production runtime.
        """
        samples = generate_simple_samples(100)
        aois = generate_simple_aois(50)

        start = time.perf_counter()
        result = compute_attention_seconds(samples, aois, enable_profiling=False)
        elapsed = time.perf_counter() - start

        assert result.total_samples == 100
        assert len(result.aoi_results) == 50
        assert elapsed < 1.0, f"100 samples × 50 AOIs took {elapsed:.3f}s, expected < 1s"

    def test_performance_300_samples_50_aois(self) -> None:
        """Test with 300 samples and 50 AOIs.

        This is a demanding workload.
        Target: < 2.0s production runtime.
        """
        samples = generate_simple_samples(300)
        aois = generate_simple_aois(50)

        start = time.perf_counter()
        result = compute_attention_seconds(samples, aois, enable_profiling=False)
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

        Target: < 2.0s for 200 samples × 10 complex AOIs (production performance).
        """
        samples = generate_simple_samples(200)
        aois = self.generate_complex_aois(10, vertices_per_aoi=20)

        start = time.perf_counter()
        result = compute_attention_seconds(samples, aois, enable_profiling=False)
        elapsed = time.perf_counter() - start

        assert result.total_samples == 200
        assert (
            elapsed < 2.0
        ), f"200 samples × 10 complex AOIs took {elapsed:.3f}s, expected < 2.0s"


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


class TestMemoryEfficiency:
    """Test memory usage patterns (Step 6.4)."""

    def test_memory_usage_long_session(self) -> None:
        """Verify memory doesn't grow unbounded for long sessions.

        Tests that memory usage remains bounded even with 1000 samples,
        which represents a ~17 minute session at 1 Hz sampling.
        """
        import tracemalloc

        samples = generate_simple_samples(1000)
        aois = generate_simple_aois(20)

        # Start memory tracking
        tracemalloc.start()
        baseline_memory = tracemalloc.get_traced_memory()[0]

        # Process the batch
        result = compute_attention_seconds(samples, aois, enable_profiling=False)

        # Get peak memory usage
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Memory usage should be reasonable
        # Peak memory should not exceed 50 MB for this workload
        # (1000 samples × 20 AOIs is a demanding but realistic workload)
        memory_used_mb = (peak_memory - baseline_memory) / (1024 * 1024)
        assert (
            memory_used_mb < 50
        ), f"Memory usage too high: {memory_used_mb:.2f} MB (expected < 50 MB)"

        # Verify results are still correct
        assert result.total_samples == 1000
        assert len(result.aoi_results) == 20

    def test_memory_usage_many_aois(self) -> None:
        """Verify memory usage scales reasonably with number of AOIs."""
        import tracemalloc

        samples = generate_simple_samples(500)
        aois = generate_simple_aois(100)

        tracemalloc.start()
        baseline_memory = tracemalloc.get_traced_memory()[0]

        result = compute_attention_seconds(samples, aois, enable_profiling=False)

        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # With 100 AOIs, memory should still be reasonable (< 100 MB)
        memory_used_mb = (peak_memory - baseline_memory) / (1024 * 1024)
        assert (
            memory_used_mb < 100
        ), f"Memory usage too high: {memory_used_mb:.2f} MB (expected < 100 MB)"

        assert result.total_samples == 500
        assert len(result.aoi_results) == 100

    def test_memory_no_intermediate_accumulation(self) -> None:
        """Verify intermediate results are not retained unnecessarily.

        Process samples in batches and verify that memory doesn't grow
        excessively with the number of samples processed. Note that results
        contain hit_timestamps which grow with O(N), so we allow proportional
        growth but verify no unbounded accumulation.
        """
        import tracemalloc

        aois = generate_simple_aois(10)

        # Process first batch to establish baseline
        tracemalloc.start()
        samples_100 = generate_simple_samples(100, seed=1)
        result_100 = compute_attention_seconds(samples_100, aois, enable_profiling=False)
        mem_100 = tracemalloc.get_traced_memory()[1]

        # Process larger batch
        samples_500 = generate_simple_samples(500, seed=2)
        result_500 = compute_attention_seconds(samples_500, aois, enable_profiling=False)
        mem_500 = tracemalloc.get_traced_memory()[1]

        tracemalloc.stop()

        # Memory should grow roughly proportionally to sample count (due to hit_timestamps)
        # but not excessively. Allow up to 6x memory growth for 5x samples.
        # This accounts for hit_timestamps lists plus some overhead.
        memory_ratio = mem_500 / mem_100
        assert (
            memory_ratio < 6.0
        ), f"Memory grew too much: {memory_ratio:.2f}x (expected < 6x for 5x samples)"

        # Verify both results are correct
        assert result_100.total_samples == 100
        assert result_500.total_samples == 500


class TestStreamingMode:
    """Test streaming mode for very long sessions (Step 6.4)."""

    def test_streaming_mode_consistency(self) -> None:
        """Verify streaming mode produces same results as batch mode.

        This is the critical test for Step 6.4 - ensures that processing
        samples in chunks produces identical results to processing all at once.
        """
        samples = generate_simple_samples(500)
        aois = generate_simple_aois(20)

        # Process in batch mode
        batch_result = compute_attention_seconds(samples, aois, enable_profiling=False)

        # Process in streaming mode (consume the generator to get final result)
        streaming_results = list(
            compute_attention_seconds_streaming(samples, aois, chunk_size=50)
        )
        streaming_result = streaming_results[-1]  # Get final result

        # Verify core counters are identical
        assert streaming_result.total_samples == batch_result.total_samples
        assert streaming_result.samples_with_hits == batch_result.samples_with_hits
        assert streaming_result.samples_no_winner == batch_result.samples_no_winner

        # Verify per-AOI results are identical
        for aoi_id in batch_result.aoi_results:
            batch_aoi = batch_result.aoi_results[aoi_id]
            stream_aoi = streaming_result.aoi_results[aoi_id]

            assert batch_aoi.hit_count == stream_aoi.hit_count
            assert (
                batch_aoi.total_attention_seconds == stream_aoi.total_attention_seconds
            )
            assert batch_aoi.hit_timestamps == stream_aoi.hit_timestamps

    def test_streaming_mode_progress_tracking(self) -> None:
        """Verify streaming mode yields intermediate results."""
        samples = generate_simple_samples(300)
        aois = generate_simple_aois(10)

        chunk_size = 100
        results = []

        # Collect all intermediate results
        for result in compute_attention_seconds_streaming(
            samples, aois, chunk_size=chunk_size
        ):
            results.append(result)

        # Should have 3 results (300 samples / 100 chunk_size)
        assert len(results) == 3

        # Verify progressive accumulation
        assert results[0].total_samples == 100
        assert results[1].total_samples == 200
        assert results[2].total_samples == 300

        # Verify hit counts are monotonically increasing
        for i in range(len(results) - 1):
            assert results[i + 1].samples_with_hits >= results[i].samples_with_hits

    def test_streaming_mode_memory_efficiency(self) -> None:
        """Verify streaming mode uses comparable memory to batch mode.

        For typical sessions, both modes have similar memory usage because
        result accumulation (hit_timestamps) dominates. The benefit of streaming
        is more apparent with very large chunk sizes or when intermediate results
        are processed/discarded rather than accumulated.

        This test verifies that streaming mode doesn't use significantly MORE
        memory than batch mode.
        """
        import tracemalloc

        samples = generate_simple_samples(1000)
        aois = generate_simple_aois(20)

        # Measure batch mode memory
        tracemalloc.start()
        baseline = tracemalloc.get_traced_memory()[0]
        batch_result = compute_attention_seconds(samples, aois, enable_profiling=False)
        batch_peak = tracemalloc.get_traced_memory()[1] - baseline
        tracemalloc.stop()

        # Measure streaming mode memory
        tracemalloc.start()
        baseline = tracemalloc.get_traced_memory()[0]
        # Consume streaming results without storing them all
        for result in compute_attention_seconds_streaming(
            samples, aois, chunk_size=100
        ):
            streaming_result = result  # Keep only the latest
        streaming_peak = tracemalloc.get_traced_memory()[1] - baseline
        tracemalloc.stop()

        # Verify results are identical
        assert streaming_result.total_samples == batch_result.total_samples

        # Streaming should not use significantly more memory than batch mode
        # Allow up to 2x overhead (in practice, should be close to 1:1)
        memory_ratio = streaming_peak / batch_peak
        assert (
            memory_ratio < 2.0
        ), f"Streaming used {memory_ratio:.2f}x batch memory (expected < 2x)"

    def test_streaming_mode_empty_samples(self) -> None:
        """Verify streaming mode handles empty sample list."""
        samples: list[ViewerSample] = []
        aois = generate_simple_aois(5)

        results = list(compute_attention_seconds_streaming(samples, aois, chunk_size=10))

        # Should have no results since there are no samples
        assert len(results) == 0

    def test_streaming_mode_single_chunk(self) -> None:
        """Verify streaming mode works with samples that fit in a single chunk."""
        samples = generate_simple_samples(50)
        aois = generate_simple_aois(10)

        results = list(
            compute_attention_seconds_streaming(samples, aois, chunk_size=100)
        )

        # Should have exactly 1 result
        assert len(results) == 1
        assert results[0].total_samples == 50

    def test_streaming_mode_partial_final_chunk(self) -> None:
        """Verify streaming mode handles partial final chunk correctly."""
        samples = generate_simple_samples(250)  # Not evenly divisible by chunk_size
        aois = generate_simple_aois(10)

        results = list(
            compute_attention_seconds_streaming(samples, aois, chunk_size=100)
        )

        # Should have 3 results: 100, 100, 50
        assert len(results) == 3
        assert results[0].total_samples == 100
        assert results[1].total_samples == 200
        assert results[2].total_samples == 250  # Final partial chunk

