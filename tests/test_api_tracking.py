"""
Tests for the tracking API function compute_attention_seconds.

These tests verify that the public API is correctly exposed and behaves
consistently with the single-frame API (find_largest_obstacle).

Step 4.1: Public API Exposure
"""

import numpy as np
import pytest
from numpy.typing import NDArray

# Test that compute_attention_seconds is accessible from view_arc
import view_arc
from view_arc import (
    AOI,
    ViewerSample,
    compute_attention_seconds,
)
from view_arc.obstacle.api import find_largest_obstacle


# =============================================================================
# Helper functions for creating test fixtures
# =============================================================================


def make_triangle(center: tuple, size: float = 20.0) -> NDArray[np.float32]:
    """Create a triangle centered at given point."""
    cx, cy = center
    return np.array(
        [
            [cx, cy + size],  # top
            [cx - size, cy - size],  # bottom-left
            [cx + size, cy - size],  # bottom-right
        ],
        dtype=np.float32,
    )


def make_square(center: tuple, half_size: float = 15.0) -> NDArray[np.float32]:
    """Create a square centered at given point."""
    cx, cy = center
    return np.array(
        [
            [cx - half_size, cy - half_size],  # bottom-left
            [cx + half_size, cy - half_size],  # bottom-right
            [cx + half_size, cy + half_size],  # top-right
            [cx - half_size, cy + half_size],  # top-left
        ],
        dtype=np.float32,
    )


def make_rectangle(
    center: tuple, width: float, height: float
) -> NDArray[np.float32]:
    """Create a rectangle centered at given point."""
    cx, cy = center
    hw, hh = width / 2, height / 2
    return np.array(
        [
            [cx - hw, cy - hh],
            [cx + hw, cy - hh],
            [cx + hw, cy + hh],
            [cx - hw, cy + hh],
        ],
        dtype=np.float32,
    )


# =============================================================================
# Test: API Accessibility
# =============================================================================


class TestComputeAttentionAPIAccessible:
    """Verify that compute_attention_seconds is accessible from the public API."""

    def test_import_from_view_arc(self):
        """Test that compute_attention_seconds can be imported from view_arc."""
        # Should be able to import directly from view_arc package
        assert hasattr(view_arc, "compute_attention_seconds")
        assert callable(view_arc.compute_attention_seconds)

    def test_import_from_view_arc_direct(self):
        """Test that compute_attention_seconds can be imported directly."""
        from view_arc import compute_attention_seconds as compute_fn

        assert callable(compute_fn)

    def test_function_signature(self):
        """Test that compute_attention_seconds has expected signature."""
        import inspect

        sig = inspect.signature(compute_attention_seconds)
        params = list(sig.parameters.keys())

        # Check key parameters exist
        assert "samples" in params
        assert "aois" in params
        assert "field_of_view_deg" in params
        assert "max_range" in params
        assert "sample_interval" in params

    def test_basic_invocation(self):
        """Test that compute_attention_seconds can be invoked with basic inputs."""
        # Single sample
        samples = [
            ViewerSample(
                position=(100.0, 100.0), direction=(0.0, 1.0), timestamp=0.0
            )
        ]

        # Single AOI
        aois = [AOI(id="shelf1", contour=make_square((100, 150), half_size=20))]

        # Should not raise
        result = compute_attention_seconds(samples=samples, aois=aois)

        # Check result structure
        assert result is not None
        assert hasattr(result, "aoi_results")
        assert hasattr(result, "total_samples")


# =============================================================================
# Test: Consistency with Manual Iteration
# =============================================================================


class TestComputeAttentionMatchesManualLoop:
    """Verify that compute_attention_seconds produces same results as manual iteration."""

    def test_single_sample_matches_manual(self):
        """Single sample should match single call to find_largest_obstacle."""
        viewer_pos = (100.0, 100.0)
        viewer_dir = (0.0, 1.0)
        field_of_view_deg = 90.0
        max_range = 200.0

        # Create sample
        sample = ViewerSample(position=viewer_pos, direction=viewer_dir)

        # Create AOIs
        aoi1 = AOI(id="left", contour=make_square((50, 150), half_size=15))
        aoi2 = AOI(id="center", contour=make_square((100, 150), half_size=15))
        aoi3 = AOI(id="right", contour=make_square((150, 150), half_size=15))
        aois = [aoi1, aoi2, aoi3]

        # Manual call to find_largest_obstacle
        contours = [aoi.contour for aoi in aois]
        manual_result = find_largest_obstacle(
            viewer_point=np.array(viewer_pos, dtype=np.float32),
            view_direction=np.array(viewer_dir, dtype=np.float32),
            field_of_view_deg=field_of_view_deg,
            max_range=max_range,
            obstacle_contours=contours,
        )

        # Tracking API call
        tracking_result = compute_attention_seconds(
            samples=[sample],
            aois=aois,
            field_of_view_deg=field_of_view_deg,
            max_range=max_range,
        )

        # Results should match
        if manual_result.obstacle_id is not None:
            expected_aoi_id = aois[manual_result.obstacle_id].id
            # The winning AOI should have 1 hit
            assert expected_aoi_id in tracking_result.aoi_results
            assert tracking_result.aoi_results[expected_aoi_id].hit_count == 1
            assert tracking_result.samples_with_hits == 1
        else:
            # No winner - all AOIs should have 0 hits
            assert tracking_result.samples_with_hits == 0
            for aoi_result in tracking_result.aoi_results.values():
                assert aoi_result.hit_count == 0

    def test_multiple_samples_matches_manual_loop(self):
        """Multiple samples should match sequential calls to find_largest_obstacle."""
        field_of_view_deg = 90.0
        max_range = 200.0

        # Create samples - viewer looking at different directions (normalized)
        sqrt2_inv = 1.0 / np.sqrt(2)
        samples = [
            ViewerSample(
                position=(100.0, 100.0),
                direction=(-sqrt2_inv, sqrt2_inv),
                timestamp=0.0,
            ),  # left
            ViewerSample(
                position=(100.0, 100.0), direction=(0.0, 1.0), timestamp=1.0
            ),  # center
            ViewerSample(
                position=(100.0, 100.0),
                direction=(sqrt2_inv, sqrt2_inv),
                timestamp=2.0,
            ),  # right
        ]

        # Create AOIs
        aoi1 = AOI(id="left", contour=make_square((50, 150), half_size=15))
        aoi2 = AOI(id="center", contour=make_square((100, 150), half_size=15))
        aoi3 = AOI(id="right", contour=make_square((150, 150), half_size=15))
        aois = [aoi1, aoi2, aoi3]
        contours = [aoi.contour for aoi in aois]

        # Manual iteration
        manual_hits = {aoi.id: 0 for aoi in aois}
        for sample in samples:
            result = find_largest_obstacle(
                viewer_point=np.array(sample.position, dtype=np.float32),
                view_direction=np.array(sample.direction, dtype=np.float32),
                field_of_view_deg=field_of_view_deg,
                max_range=max_range,
                obstacle_contours=contours,
            )
            if result.obstacle_id is not None:
                winner_id = aois[result.obstacle_id].id
                manual_hits[winner_id] += 1

        # Tracking API call
        tracking_result = compute_attention_seconds(
            samples=samples,
            aois=aois,
            field_of_view_deg=field_of_view_deg,
            max_range=max_range,
        )

        # Results should match
        for aoi_id, manual_count in manual_hits.items():
            assert aoi_id in tracking_result.aoi_results
            assert (
                tracking_result.aoi_results[aoi_id].hit_count == manual_count
            ), f"AOI {aoi_id}: expected {manual_count}, got {tracking_result.aoi_results[aoi_id].hit_count}"

    def test_no_hits_matches_manual(self):
        """When no AOIs are visible, results should match manual iteration."""
        field_of_view_deg = 90.0
        max_range = 200.0

        # Sample looking away from AOIs
        sample = ViewerSample(
            position=(100.0, 100.0), direction=(0.0, -1.0)
        )  # looking DOWN

        # AOIs are above
        aoi1 = AOI(id="top", contour=make_square((100, 200), half_size=15))
        aois = [aoi1]
        contours = [aoi.contour for aoi in aois]

        # Manual call
        manual_result = find_largest_obstacle(
            viewer_point=np.array(sample.position, dtype=np.float32),
            view_direction=np.array(sample.direction, dtype=np.float32),
            field_of_view_deg=field_of_view_deg,
            max_range=max_range,
            obstacle_contours=contours,
        )

        # Tracking API call
        tracking_result = compute_attention_seconds(
            samples=[sample],
            aois=aois,
            field_of_view_deg=field_of_view_deg,
            max_range=max_range,
        )

        # Should both indicate no winner
        assert manual_result.obstacle_id is None
        assert tracking_result.samples_with_hits == 0
        assert tracking_result.aoi_results["top"].hit_count == 0


# =============================================================================
# Test: Parameter Consistency
# =============================================================================


class TestComputeAttentionParameterConsistency:
    """Verify that FOV and max_range work consistently with single-frame API."""

    def test_fov_parameter_consistency(self):
        """FOV parameter should filter AOIs same as single-frame API."""
        viewer_pos = (100.0, 100.0)
        viewer_dir = (0.0, 1.0)  # looking UP
        max_range = 200.0

        # AOI to the right at 45 degrees
        aoi_right = AOI(id="right45", contour=make_square((150, 150), half_size=10))
        aois = [aoi_right]
        sample = ViewerSample(position=viewer_pos, direction=viewer_dir)

        # With narrow FOV (30 deg), AOI should not be visible (45 > 15)
        narrow_fov = 30.0
        manual_narrow = find_largest_obstacle(
            viewer_point=np.array(viewer_pos, dtype=np.float32),
            view_direction=np.array(viewer_dir, dtype=np.float32),
            field_of_view_deg=narrow_fov,
            max_range=max_range,
            obstacle_contours=[aoi_right.contour],
        )
        tracking_narrow = compute_attention_seconds(
            samples=[sample],
            aois=aois,
            field_of_view_deg=narrow_fov,
            max_range=max_range,
        )

        # With wide FOV (90 deg), AOI should be visible
        wide_fov = 90.0
        manual_wide = find_largest_obstacle(
            viewer_point=np.array(viewer_pos, dtype=np.float32),
            view_direction=np.array(viewer_dir, dtype=np.float32),
            field_of_view_deg=wide_fov,
            max_range=max_range,
            obstacle_contours=[aoi_right.contour],
        )
        tracking_wide = compute_attention_seconds(
            samples=[sample],
            aois=aois,
            field_of_view_deg=wide_fov,
            max_range=max_range,
        )

        # Results should be consistent
        if manual_narrow.obstacle_id is None:
            assert tracking_narrow.samples_with_hits == 0
        else:
            assert tracking_narrow.samples_with_hits == 1

        if manual_wide.obstacle_id is None:
            assert tracking_wide.samples_with_hits == 0
        else:
            assert tracking_wide.samples_with_hits == 1

    def test_max_range_parameter_consistency(self):
        """max_range parameter should filter AOIs same as single-frame API."""
        viewer_pos = (100.0, 100.0)
        viewer_dir = (0.0, 1.0)  # looking UP
        field_of_view_deg = 90.0

        # AOI far away (distance ~100)
        aoi_far = AOI(id="far", contour=make_square((100, 200), half_size=10))
        aois = [aoi_far]
        sample = ViewerSample(position=viewer_pos, direction=viewer_dir)

        # With short max_range (50), AOI should not be visible
        short_range = 50.0
        manual_short = find_largest_obstacle(
            viewer_point=np.array(viewer_pos, dtype=np.float32),
            view_direction=np.array(viewer_dir, dtype=np.float32),
            field_of_view_deg=field_of_view_deg,
            max_range=short_range,
            obstacle_contours=[aoi_far.contour],
        )
        tracking_short = compute_attention_seconds(
            samples=[sample],
            aois=aois,
            field_of_view_deg=field_of_view_deg,
            max_range=short_range,
        )

        # With long max_range (150), AOI should be visible
        long_range = 150.0
        manual_long = find_largest_obstacle(
            viewer_point=np.array(viewer_pos, dtype=np.float32),
            view_direction=np.array(viewer_dir, dtype=np.float32),
            field_of_view_deg=field_of_view_deg,
            max_range=long_range,
            obstacle_contours=[aoi_far.contour],
        )
        tracking_long = compute_attention_seconds(
            samples=[sample],
            aois=aois,
            field_of_view_deg=field_of_view_deg,
            max_range=long_range,
        )

        # Results should be consistent
        if manual_short.obstacle_id is None:
            assert tracking_short.samples_with_hits == 0
        else:
            assert tracking_short.samples_with_hits == 1

        if manual_long.obstacle_id is None:
            assert tracking_long.samples_with_hits == 0
        else:
            assert tracking_long.samples_with_hits == 1

    def test_parameter_defaults_match(self):
        """Tracking API should use sensible defaults for FOV and max_range."""
        import inspect

        # Get default values from tracking API
        tracking_sig = inspect.signature(compute_attention_seconds)

        # FOV default should be reasonable (90 degrees is common)
        tracking_fov_default = tracking_sig.parameters[
            "field_of_view_deg"
        ].default
        assert tracking_fov_default == 90.0

        # max_range default should be reasonable (500 is standard)
        tracking_range_default = tracking_sig.parameters["max_range"].default
        assert tracking_range_default == 500.0

    def test_all_aois_represented_in_results(self):
        """All AOIs should appear in results even with 0 hits."""
        viewer_pos = (100.0, 100.0)
        viewer_dir = (0.0, 1.0)  # looking UP

        # Create 3 AOIs, only center visible
        aoi1 = AOI(id="left", contour=make_square((0, 100), half_size=10))  # behind
        aoi2 = AOI(
            id="center", contour=make_square((100, 150), half_size=10)
        )  # visible
        aoi3 = AOI(
            id="right", contour=make_square((200, 100), half_size=10)
        )  # behind
        aois = [aoi1, aoi2, aoi3]

        sample = ViewerSample(position=viewer_pos, direction=viewer_dir)

        result = compute_attention_seconds(samples=[sample], aois=aois)

        # All AOI IDs should be in results
        assert "left" in result.aoi_results
        assert "center" in result.aoi_results
        assert "right" in result.aoi_results

        # Check hit counts
        assert result.aoi_results["left"].hit_count == 0
        assert result.aoi_results["center"].hit_count == 1
        assert result.aoi_results["right"].hit_count == 0
