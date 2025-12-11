"""
Tests for tracking algorithm functions (Phase 2).

Step 2.1: Tests for process_single_sample()
- test_process_single_sample_one_aoi_visible() - single AOI in view
- test_process_single_sample_multiple_aoi() - returns winner
- test_process_single_sample_no_aoi_visible() - returns None
- test_process_single_sample_all_aoi_outside_range() - max_range filtering
- test_process_single_sample_preserves_aoi_id() - ID correctly mapped

Step 2.2: Tests for compute_attention_seconds()
- test_compute_attention_single_sample() - trivial case
- test_compute_attention_all_same_aoi() - viewer stares at one AOI
- test_compute_attention_alternating_aois() - viewer looks left/right
- test_compute_attention_no_hits() - viewer never looks at AOIs
- test_compute_attention_partial_hits() - some samples hit, some miss
- test_compute_attention_hit_count_accuracy() - verify counts
- test_compute_attention_all_aois_represented() - all AOIs in result
- test_compute_attention_timestamps_recorded() - hit indices tracked
"""

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from view_arc.tracking import (
    AOI,
    AOIIntervalBreakdown,
    SessionConfig,
    SingleSampleResult,
    TrackingResultWithConfig,
    ValidationError,
    ViewerSample,
    compute_attention_seconds,
    process_single_sample,
)


# =============================================================================
# Helper functions
# =============================================================================


def make_unit_vector(angle_deg: float) -> tuple[float, float]:
    """Create a unit vector from an angle in degrees.

    Args:
        angle_deg: Angle in degrees (0 = right, 90 = up, 180 = left, 270 = down)

    Returns:
        Unit vector (dx, dy)
    """
    angle_rad = math.radians(angle_deg)
    return (math.cos(angle_rad), math.sin(angle_rad))


def make_square_contour(
    center: tuple[float, float], half_size: float = 15.0
) -> NDArray[np.float64]:
    """Create a square contour centered at the given point."""
    cx, cy = center
    return np.array(
        [
            [cx - half_size, cy - half_size],
            [cx + half_size, cy - half_size],
            [cx + half_size, cy + half_size],
            [cx - half_size, cy + half_size],
        ],
        dtype=np.float64,
    )


def make_rectangle_contour(
    center: tuple[float, float],
    width: float = 30.0,
    height: float = 20.0,
) -> NDArray[np.float64]:
    """Create a rectangle contour centered at the given point."""
    cx, cy = center
    hw, hh = width / 2, height / 2
    return np.array(
        [
            [cx - hw, cy - hh],
            [cx + hw, cy - hh],
            [cx + hw, cy + hh],
            [cx - hw, cy + hh],
        ],
        dtype=np.float64,
    )


# =============================================================================
# Tests: process_single_sample() - Step 2.1
# =============================================================================


class TestProcessSingleSampleOneAOIVisible:
    """Test process_single_sample with a single AOI in view."""

    def test_single_aoi_directly_in_front(self) -> None:
        """Single AOI directly in front of viewer should be selected."""
        # Viewer at (100, 100), looking up (0, 1)
        sample = ViewerSample(
            position=(100.0, 100.0),
            direction=(0.0, 1.0),
        )

        # AOI directly above the viewer
        aoi = AOI(
            id="shelf1",
            contour=make_square_contour((100.0, 150.0), half_size=20.0),
        )

        result = process_single_sample(sample, [aoi], fov_deg=90.0, max_range=100.0)

        assert result == "shelf1"

    def test_single_aoi_within_fov(self) -> None:
        """Single AOI within FOV should be selected."""
        # Viewer at origin, looking right (1, 0)
        sample = ViewerSample(
            position=(0.0, 0.0),
            direction=(1.0, 0.0),
        )

        # AOI to the upper-right (within 90° FOV)
        aoi = AOI(
            id="display_A",
            contour=make_square_contour((50.0, 30.0), half_size=15.0),
        )

        result = process_single_sample(sample, [aoi], fov_deg=90.0, max_range=200.0)

        assert result == "display_A"

    def test_single_aoi_with_integer_id(self) -> None:
        """Single AOI with integer ID should work correctly."""
        sample = ViewerSample(
            position=(100.0, 100.0),
            direction=(0.0, 1.0),
        )

        aoi = AOI(
            id=42,
            contour=make_square_contour((100.0, 150.0), half_size=20.0),
        )

        result = process_single_sample(sample, [aoi], fov_deg=90.0, max_range=100.0)

        assert result == 42


class TestProcessSingleSampleMultipleAOI:
    """Test process_single_sample with multiple AOIs - returns winner."""

    def test_multiple_aois_larger_wins(self) -> None:
        """AOI with larger angular coverage should win.

        Geometry: Viewer looks up at two non-overlapping AOIs at same distance.
        The larger AOI (wider rectangle) has more angular coverage and should win.
        """
        # Viewer at (100, 100), looking up
        sample = ViewerSample(
            position=(100.0, 100.0),
            direction=(0.0, 1.0),
        )

        # Small AOI: 20x20 square at distance 50, centered in view
        # At distance 50, a 20-wide object subtends ~22.6° (2*atan(10/50))
        small_aoi = AOI(
            id="small_shelf",
            contour=make_square_contour((100.0, 150.0), half_size=10.0),
        )

        # Large AOI: 80x20 rectangle at same distance, to the right
        # At distance 50, an 80-wide object subtends ~77.3° (2*atan(40/50))
        # Placed so it doesn't overlap with small_aoi
        large_aoi = AOI(
            id="large_shelf",
            contour=make_rectangle_contour((160.0, 150.0), width=80.0, height=20.0),
        )

        result = process_single_sample(
            sample, [small_aoi, large_aoi], fov_deg=120.0, max_range=200.0
        )

        # The larger AOI should win due to greater angular coverage
        assert result == "large_shelf"

    def test_multiple_aois_all_visible(self) -> None:
        """When multiple equal-sized AOIs are visible, the closest one wins.

        Geometry: Three equal-sized AOIs at different distances.
        The center AOI is closest and should subtend the largest angle.
        """
        sample = ViewerSample(
            position=(100.0, 100.0),
            direction=(0.0, 1.0),
        )

        # Left AOI: at distance 80, same size as others
        aoi_left = AOI(
            id="left_shelf",
            contour=make_square_contour((60.0, 180.0), half_size=15.0),
        )

        # Center AOI: at distance 40 - CLOSEST, should win
        aoi_center = AOI(
            id="center_shelf",
            contour=make_square_contour((100.0, 140.0), half_size=15.0),
        )

        # Right AOI: at distance 80, same size as others
        aoi_right = AOI(
            id="right_shelf",
            contour=make_square_contour((140.0, 180.0), half_size=15.0),
        )

        result = process_single_sample(
            sample, [aoi_left, aoi_center, aoi_right], fov_deg=120.0, max_range=150.0
        )

        # Center shelf is closest, so it subtends the largest angle and wins
        assert result == "center_shelf"

    def test_multiple_aois_only_one_visible(self) -> None:
        """When only one AOI is in FOV, that one should win."""
        # Viewer looking right
        sample = ViewerSample(
            position=(100.0, 100.0),
            direction=(1.0, 0.0),
        )

        # AOI to the right (in FOV)
        visible_aoi = AOI(
            id="visible",
            contour=make_square_contour((150.0, 100.0), half_size=15.0),
        )

        # AOI behind the viewer (not in FOV)
        hidden_aoi = AOI(
            id="hidden",
            contour=make_square_contour((50.0, 100.0), half_size=15.0),
        )

        result = process_single_sample(
            sample, [visible_aoi, hidden_aoi], fov_deg=90.0, max_range=200.0
        )

        assert result == "visible"


class TestProcessSingleSampleNoAOIVisible:
    """Test process_single_sample when no AOI is visible."""

    def test_no_aois_returns_none(self) -> None:
        """Empty AOI list should return None."""
        sample = ViewerSample(
            position=(100.0, 100.0),
            direction=(0.0, 1.0),
        )

        result = process_single_sample(sample, [], fov_deg=90.0, max_range=100.0)

        assert result is None

    def test_all_aois_behind_viewer(self) -> None:
        """AOIs behind the viewer (outside FOV) should not be seen."""
        # Viewer looking up
        sample = ViewerSample(
            position=(100.0, 100.0),
            direction=(0.0, 1.0),
        )

        # AOI below the viewer (behind)
        behind_aoi = AOI(
            id="behind",
            contour=make_square_contour((100.0, 50.0), half_size=15.0),
        )

        result = process_single_sample(
            sample, [behind_aoi], fov_deg=90.0, max_range=100.0
        )

        assert result is None

    def test_aoi_outside_fov_angle(self) -> None:
        """AOI outside the FOV angle should not be visible."""
        # Viewer looking right with narrow FOV
        sample = ViewerSample(
            position=(100.0, 100.0),
            direction=(1.0, 0.0),
        )

        # AOI at 60 degrees above horizontal (outside 45° half-FOV)
        aoi = AOI(
            id="outside_fov",
            contour=make_square_contour((130.0, 200.0), half_size=15.0),
        )

        # With 90° FOV, half-FOV is 45°
        result = process_single_sample(
            sample, [aoi], fov_deg=60.0, max_range=200.0
        )

        # The AOI is at ~70° angle from horizontal, outside 30° half-FOV
        assert result is None


class TestProcessSingleSampleMaxRangeFiltering:
    """Test process_single_sample with max_range filtering."""

    def test_aoi_within_range(self) -> None:
        """AOI within max_range should be detected."""
        sample = ViewerSample(
            position=(100.0, 100.0),
            direction=(0.0, 1.0),
        )

        # AOI 40 pixels away
        aoi = AOI(
            id="nearby",
            contour=make_square_contour((100.0, 140.0), half_size=15.0),
        )

        result = process_single_sample(sample, [aoi], fov_deg=90.0, max_range=100.0)

        assert result == "nearby"

    def test_aoi_beyond_max_range(self) -> None:
        """AOI beyond max_range should not be detected."""
        sample = ViewerSample(
            position=(100.0, 100.0),
            direction=(0.0, 1.0),
        )

        # AOI 200 pixels away (center at y=300)
        aoi = AOI(
            id="far_away",
            contour=make_square_contour((100.0, 300.0), half_size=15.0),
        )

        # max_range of 100 should not reach AOI at distance 200
        result = process_single_sample(sample, [aoi], fov_deg=90.0, max_range=100.0)

        assert result is None

    def test_multiple_aois_some_out_of_range(self) -> None:
        """Only AOIs within range should be considered."""
        sample = ViewerSample(
            position=(100.0, 100.0),
            direction=(0.0, 1.0),
        )

        # Near AOI at distance ~50
        near_aoi = AOI(
            id="near",
            contour=make_square_contour((100.0, 150.0), half_size=15.0),
        )

        # Far AOI at distance ~200
        far_aoi = AOI(
            id="far",
            contour=make_square_contour((100.0, 300.0), half_size=15.0),
        )

        result = process_single_sample(
            sample, [near_aoi, far_aoi], fov_deg=90.0, max_range=100.0
        )

        assert result == "near"


class TestProcessSingleSamplePreservesAOIId:
    """Test that AOI IDs are correctly preserved through the pipeline."""

    def test_string_id_preserved(self) -> None:
        """String AOI IDs should be preserved exactly."""
        sample = ViewerSample(
            position=(100.0, 100.0),
            direction=(0.0, 1.0),
        )

        aoi = AOI(
            id="my-special-shelf_123",
            contour=make_square_contour((100.0, 150.0), half_size=20.0),
        )

        result = process_single_sample(sample, [aoi], fov_deg=90.0, max_range=100.0)

        assert result == "my-special-shelf_123"
        assert isinstance(result, str)

    def test_integer_id_preserved(self) -> None:
        """Integer AOI IDs should be preserved exactly."""
        sample = ViewerSample(
            position=(100.0, 100.0),
            direction=(0.0, 1.0),
        )

        aoi = AOI(
            id=999,
            contour=make_square_contour((100.0, 150.0), half_size=20.0),
        )

        result = process_single_sample(sample, [aoi], fov_deg=90.0, max_range=100.0)

        assert result == 999
        assert isinstance(result, int)

    def test_mixed_id_types_preserved(self) -> None:
        """Mixed string and integer IDs should all be preserved.

        The center AOI ("shelf_A") is closest and should win.
        """
        sample = ViewerSample(
            position=(100.0, 100.0),
            direction=(0.0, 1.0),
        )

        # Left AOI at distance 60
        aoi_left = AOI(id=1, contour=make_square_contour((70.0, 160.0), half_size=10.0))
        # Center AOI at distance 40 - CLOSEST, should win
        aoi_center = AOI(id="shelf_A", contour=make_square_contour((100.0, 140.0), half_size=10.0))
        # Right AOI at distance 60
        aoi_right = AOI(id=2, contour=make_square_contour((130.0, 160.0), half_size=10.0))

        aois = [aoi_left, aoi_center, aoi_right]

        result = process_single_sample(sample, aois, fov_deg=90.0, max_range=100.0)

        # Center shelf ("shelf_A") is closest and should win
        assert result == "shelf_A"
        assert isinstance(result, str)


class TestProcessSingleSampleReturnDetails:
    """Test process_single_sample with return_details=True."""

    def test_returns_single_sample_result(self) -> None:
        """With return_details=True, should return SingleSampleResult."""
        sample = ViewerSample(
            position=(100.0, 100.0),
            direction=(0.0, 1.0),
        )

        aoi = AOI(
            id="shelf1",
            contour=make_square_contour((100.0, 150.0), half_size=20.0),
        )

        result = process_single_sample(
            sample, [aoi], fov_deg=90.0, max_range=100.0, return_details=True
        )

        assert isinstance(result, SingleSampleResult)
        assert result.winning_aoi_id == "shelf1"
        assert result.angular_coverage > 0
        assert result.min_distance < float("inf")

    def test_details_includes_all_coverage(self) -> None:
        """Detailed result should include coverage for all visible AOIs."""
        sample = ViewerSample(
            position=(100.0, 100.0),
            direction=(0.0, 1.0),
        )

        aoi1 = AOI(id="left", contour=make_square_contour((80.0, 150.0), half_size=15.0))
        aoi2 = AOI(id="right", contour=make_square_contour((120.0, 150.0), half_size=15.0))

        result = process_single_sample(
            sample, [aoi1, aoi2], fov_deg=90.0, max_range=100.0, return_details=True
        )

        assert isinstance(result, SingleSampleResult)
        assert result.all_coverage is not None
        # Both AOIs should have some coverage
        assert len(result.all_coverage) >= 1

    def test_details_includes_all_distances(self) -> None:
        """Detailed result should include min distances for all visible AOIs."""
        sample = ViewerSample(
            position=(100.0, 100.0),
            direction=(0.0, 1.0),
        )

        # Near AOI at ~25 pixels, to the left
        aoi_near = AOI(id="near", contour=make_square_contour((70.0, 125.0), half_size=10.0))
        # Far AOI at ~65 pixels, to the right (not occluded by near AOI)
        aoi_far = AOI(id="far", contour=make_square_contour((130.0, 165.0), half_size=10.0))

        result = process_single_sample(
            sample, [aoi_near, aoi_far], fov_deg=120.0, max_range=150.0, return_details=True
        )

        assert isinstance(result, SingleSampleResult)
        assert result.all_distances is not None
        assert "near" in result.all_distances
        assert "far" in result.all_distances
        # Near AOI should have smaller distance
        assert result.all_distances["near"] < result.all_distances["far"]

    def test_details_includes_interval_details(self) -> None:
        """Detailed result should include interval breakdown for debugging."""
        sample = ViewerSample(
            position=(100.0, 100.0),
            direction=(0.0, 1.0),
        )

        aoi = AOI(
            id="shelf1",
            contour=make_square_contour((100.0, 150.0), half_size=20.0),
        )

        result = process_single_sample(
            sample, [aoi], fov_deg=90.0, max_range=100.0, return_details=True
        )

        assert isinstance(result, SingleSampleResult)
        assert result.interval_details is not None
        assert len(result.interval_details) > 0
        # All intervals should be AOIIntervalBreakdown instances
        for interval in result.interval_details:
            assert isinstance(interval, AOIIntervalBreakdown)
            assert interval.aoi_id == "shelf1"
            assert interval.angular_span > 0

    def test_details_interval_uses_aoi_ids(self) -> None:
        """Interval details should use AOI IDs, not obstacle indices."""
        sample = ViewerSample(
            position=(100.0, 100.0),
            direction=(0.0, 1.0),
        )

        aoi = AOI(
            id="my-custom-id-123",
            contour=make_square_contour((100.0, 150.0), half_size=20.0),
        )

        result = process_single_sample(
            sample, [aoi], fov_deg=90.0, max_range=100.0, return_details=True
        )

        assert isinstance(result, SingleSampleResult)
        assert result.interval_details is not None
        # The AOI ID should be the string, not the index 0
        for interval in result.interval_details:
            assert interval.aoi_id == "my-custom-id-123"

    def test_get_winner_intervals_helper(self) -> None:
        """get_winner_intervals() should return only the winner's intervals."""
        sample = ViewerSample(
            position=(100.0, 100.0),
            direction=(0.0, 1.0),
        )

        # Winner: closer AOI with larger angular coverage
        aoi_winner = AOI(
            id="winner",
            contour=make_square_contour((100.0, 130.0), half_size=20.0),
        )
        # Loser: farther AOI
        aoi_loser = AOI(
            id="loser",
            contour=make_square_contour((100.0, 180.0), half_size=15.0),
        )

        result = process_single_sample(
            sample, [aoi_winner, aoi_loser], fov_deg=90.0, max_range=150.0, return_details=True
        )

        assert isinstance(result, SingleSampleResult)
        winner_intervals = result.get_winner_intervals()
        all_intervals = result.get_all_intervals()

        # Winner intervals should only contain "winner" AOI
        assert len(winner_intervals) > 0
        for interval in winner_intervals:
            assert interval.aoi_id == "winner"

        # All intervals may contain both (or winner may occlude loser)
        assert len(all_intervals) >= len(winner_intervals)

    def test_details_no_winner(self) -> None:
        """Detailed result with no winner should have None ID."""
        sample = ViewerSample(
            position=(100.0, 100.0),
            direction=(0.0, 1.0),
        )

        # AOI behind viewer
        aoi = AOI(
            id="behind",
            contour=make_square_contour((100.0, 50.0), half_size=15.0),
        )

        result = process_single_sample(
            sample, [aoi], fov_deg=90.0, max_range=100.0, return_details=True
        )

        assert isinstance(result, SingleSampleResult)
        assert result.winning_aoi_id is None


class TestProcessSingleSampleValidation:
    """Test input validation for process_single_sample."""

    def test_invalid_sample_type(self) -> None:
        """Non-ViewerSample should raise ValidationError."""
        with pytest.raises(ValidationError, match="must be a ViewerSample"):
            process_single_sample(
                "not a sample",  # type: ignore[arg-type]
                [],
                fov_deg=90.0,
                max_range=100.0,
            )

    def test_invalid_aois_type(self) -> None:
        """Non-list aois should raise ValidationError."""
        sample = ViewerSample(position=(100.0, 100.0), direction=(1.0, 0.0))

        with pytest.raises(ValidationError, match="must be a list"):
            process_single_sample(
                sample,
                "not a list",  # type: ignore[arg-type]
                fov_deg=90.0,
                max_range=100.0,
            )

    def test_invalid_aoi_element(self) -> None:
        """Non-AOI element in list should raise ValidationError."""
        sample = ViewerSample(position=(100.0, 100.0), direction=(1.0, 0.0))

        with pytest.raises(ValidationError, match="must be an AOI"):
            process_single_sample(
                sample,
                ["not an AOI"],  # type: ignore[list-item]
                fov_deg=90.0,
                max_range=100.0,
            )

    def test_invalid_fov_deg(self) -> None:
        """Invalid fov_deg should raise ValidationError."""
        sample = ViewerSample(position=(100.0, 100.0), direction=(1.0, 0.0))

        with pytest.raises(ValidationError, match="fov_deg"):
            process_single_sample(sample, [], fov_deg=-10.0, max_range=100.0)

    def test_invalid_max_range(self) -> None:
        """Invalid max_range should raise ValidationError."""
        sample = ViewerSample(position=(100.0, 100.0), direction=(1.0, 0.0))

        with pytest.raises(ValidationError, match="max_range"):
            process_single_sample(sample, [], fov_deg=90.0, max_range=-50.0)


# =============================================================================
# Tests: compute_attention_seconds() - Step 2.2
# =============================================================================


class TestComputeAttentionSingleSample:
    """Test compute_attention_seconds with trivial single sample case."""

    def test_single_sample_hits_one_aoi(self) -> None:
        """Single sample that hits an AOI should record 1 hit."""
        sample = ViewerSample(position=(100.0, 100.0), direction=(0.0, 1.0))
        aoi = AOI(id="shelf1", contour=make_square_contour((100.0, 150.0), half_size=20.0))

        result = compute_attention_seconds([sample], [aoi])

        assert result.total_samples == 1
        assert result.samples_with_hits == 1
        assert result.samples_no_winner == 0
        assert result.get_hit_count("shelf1") == 1
        assert result.get_attention_seconds("shelf1") == 1.0

    def test_single_sample_misses_all_aois(self) -> None:
        """Single sample that misses all AOIs should record 0 hits."""
        # Viewer looking up, AOI is behind
        sample = ViewerSample(position=(100.0, 100.0), direction=(0.0, 1.0))
        aoi = AOI(id="behind", contour=make_square_contour((100.0, 50.0), half_size=15.0))

        result = compute_attention_seconds([sample], [aoi])

        assert result.total_samples == 1
        assert result.samples_with_hits == 0
        assert result.samples_no_winner == 1
        assert result.get_hit_count("behind") == 0

    def test_single_sample_empty_aoi_list(self) -> None:
        """Single sample with no AOIs should record no hits."""
        sample = ViewerSample(position=(100.0, 100.0), direction=(0.0, 1.0))

        result = compute_attention_seconds([sample], [])

        assert result.total_samples == 1
        assert result.samples_with_hits == 0
        assert result.samples_no_winner == 1


class TestComputeAttentionAllSameAOI:
    """Test compute_attention_seconds when viewer stares at one AOI."""

    def test_all_samples_hit_same_aoi(self) -> None:
        """Multiple samples all hitting same AOI should accumulate hits."""
        # Viewer stays still, looking at same AOI for 10 seconds
        samples = [
            ViewerSample(position=(100.0, 100.0), direction=(0.0, 1.0))
            for _ in range(10)
        ]
        aoi = AOI(id="target", contour=make_square_contour((100.0, 150.0), half_size=20.0))

        result = compute_attention_seconds(samples, [aoi])

        assert result.total_samples == 10
        assert result.samples_with_hits == 10
        assert result.samples_no_winner == 0
        assert result.get_hit_count("target") == 10
        assert result.get_attention_seconds("target") == 10.0

    def test_all_samples_hit_same_with_multiple_aois(self) -> None:
        """When multiple AOIs exist but viewer only sees one, only one gets hits."""
        samples = [
            ViewerSample(position=(100.0, 100.0), direction=(0.0, 1.0))
            for _ in range(5)
        ]
        
        # AOI in front
        aoi_front = AOI(id="front", contour=make_square_contour((100.0, 150.0), half_size=20.0))
        # AOI behind (invisible to viewer)
        aoi_back = AOI(id="back", contour=make_square_contour((100.0, 50.0), half_size=20.0))

        result = compute_attention_seconds(samples, [aoi_front, aoi_back])

        assert result.get_hit_count("front") == 5
        assert result.get_hit_count("back") == 0
        assert result.samples_with_hits == 5


class TestComputeAttentionAlternatingAOIs:
    """Test compute_attention_seconds when viewer looks left/right."""

    def test_alternating_between_two_aois(self) -> None:
        """Viewer alternating view direction between two AOIs."""
        # AOI on the left (viewer looks left)
        aoi_left = AOI(id="left", contour=make_square_contour((50.0, 100.0), half_size=15.0))
        # AOI on the right (viewer looks right)
        aoi_right = AOI(id="right", contour=make_square_contour((150.0, 100.0), half_size=15.0))

        # Viewer at center, alternates looking left and right
        samples = [
            ViewerSample(position=(100.0, 100.0), direction=(-1.0, 0.0)),  # left
            ViewerSample(position=(100.0, 100.0), direction=(1.0, 0.0)),   # right
            ViewerSample(position=(100.0, 100.0), direction=(-1.0, 0.0)),  # left
            ViewerSample(position=(100.0, 100.0), direction=(1.0, 0.0)),   # right
            ViewerSample(position=(100.0, 100.0), direction=(-1.0, 0.0)),  # left
        ]

        result = compute_attention_seconds(samples, [aoi_left, aoi_right])

        assert result.total_samples == 5
        assert result.samples_with_hits == 5
        assert result.get_hit_count("left") == 3
        assert result.get_hit_count("right") == 2

    def test_uneven_alternation(self) -> None:
        """Viewer spends more time looking at one AOI than another."""
        aoi_a = AOI(id="A", contour=make_square_contour((100.0, 150.0), half_size=15.0))
        aoi_b = AOI(id="B", contour=make_square_contour((150.0, 100.0), half_size=15.0))

        # Look at A for 7 samples, then B for 3 samples
        samples = [
            *[ViewerSample(position=(100.0, 100.0), direction=(0.0, 1.0)) for _ in range(7)],
            *[ViewerSample(position=(100.0, 100.0), direction=(1.0, 0.0)) for _ in range(3)],
        ]

        result = compute_attention_seconds(samples, [aoi_a, aoi_b])

        assert result.get_hit_count("A") == 7
        assert result.get_hit_count("B") == 3
        assert result.samples_with_hits == 10


class TestComputeAttentionNoHits:
    """Test compute_attention_seconds when viewer never looks at AOIs."""

    def test_viewer_always_looking_away(self) -> None:
        """Viewer always looking in direction with no AOIs."""
        # All AOIs to the right, viewer always looks left
        aoi = AOI(id="right_shelf", contour=make_square_contour((200.0, 100.0), half_size=15.0))

        samples = [
            ViewerSample(position=(100.0, 100.0), direction=(-1.0, 0.0))
            for _ in range(5)
        ]

        result = compute_attention_seconds(samples, [aoi])

        assert result.total_samples == 5
        assert result.samples_with_hits == 0
        assert result.samples_no_winner == 5
        assert result.get_hit_count("right_shelf") == 0

    def test_all_aois_out_of_range(self) -> None:
        """All AOIs beyond max_range should result in no hits."""
        # AOI far away
        aoi = AOI(id="distant", contour=make_square_contour((100.0, 500.0), half_size=15.0))

        samples = [
            ViewerSample(position=(100.0, 100.0), direction=(0.0, 1.0))
            for _ in range(3)
        ]

        result = compute_attention_seconds(samples, [aoi], max_range=100.0)

        assert result.samples_with_hits == 0
        assert result.samples_no_winner == 3


class TestComputeAttentionPartialHits:
    """Test compute_attention_seconds when some samples hit, some miss."""

    def test_mixed_hits_and_misses(self) -> None:
        """Some samples hit an AOI, some miss."""
        aoi = AOI(id="shelf", contour=make_square_contour((100.0, 150.0), half_size=15.0))

        # Alternate between looking at shelf (up) and away (down)
        samples = [
            ViewerSample(position=(100.0, 100.0), direction=(0.0, 1.0)),   # hit
            ViewerSample(position=(100.0, 100.0), direction=(0.0, -1.0)),  # miss
            ViewerSample(position=(100.0, 100.0), direction=(0.0, 1.0)),   # hit
            ViewerSample(position=(100.0, 100.0), direction=(0.0, -1.0)),  # miss
            ViewerSample(position=(100.0, 100.0), direction=(0.0, 1.0)),   # hit
        ]

        result = compute_attention_seconds(samples, [aoi])

        assert result.total_samples == 5
        assert result.samples_with_hits == 3
        assert result.samples_no_winner == 2
        assert result.get_hit_count("shelf") == 3

    def test_viewer_moves_in_and_out_of_range(self) -> None:
        """Viewer moves to positions where AOI is sometimes visible."""
        aoi = AOI(id="shelf", contour=make_square_contour((100.0, 150.0), half_size=15.0))

        samples = [
            # Close enough to see shelf
            ViewerSample(position=(100.0, 100.0), direction=(0.0, 1.0)),
            # Too far away
            ViewerSample(position=(100.0, 0.0), direction=(0.0, 1.0)),
            # Close again
            ViewerSample(position=(100.0, 100.0), direction=(0.0, 1.0)),
        ]

        result = compute_attention_seconds(samples, [aoi], max_range=80.0)

        # First and third samples are ~50 pixels away (within range)
        # Second sample is ~150 pixels away (out of range)
        assert result.samples_with_hits == 2
        assert result.samples_no_winner == 1


class TestComputeAttentionHitCountAccuracy:
    """Test that hit counts are accurately computed."""

    def test_hit_count_equals_sum_of_aoi_hits(self) -> None:
        """Total hits should equal sum of individual AOI hit counts."""
        aoi_a = AOI(id="A", contour=make_square_contour((100.0, 150.0), half_size=15.0))
        aoi_b = AOI(id="B", contour=make_square_contour((150.0, 100.0), half_size=15.0))
        aoi_c = AOI(id="C", contour=make_square_contour((50.0, 100.0), half_size=15.0))

        samples = [
            ViewerSample(position=(100.0, 100.0), direction=(0.0, 1.0)),   # A
            ViewerSample(position=(100.0, 100.0), direction=(1.0, 0.0)),   # B
            ViewerSample(position=(100.0, 100.0), direction=(-1.0, 0.0)),  # C
            ViewerSample(position=(100.0, 100.0), direction=(0.0, 1.0)),   # A
            ViewerSample(position=(100.0, 100.0), direction=(0.0, -1.0)),  # miss
        ]

        result = compute_attention_seconds(samples, [aoi_a, aoi_b, aoi_c])

        total_individual_hits = (
            result.get_hit_count("A") +
            result.get_hit_count("B") +
            result.get_hit_count("C")
        )
        assert total_individual_hits == result.samples_with_hits
        assert result.get_total_hits() == result.samples_with_hits

    def test_attention_seconds_calculation(self) -> None:
        """Attention seconds should be hit_count × sample_interval."""
        aoi = AOI(id="shelf", contour=make_square_contour((100.0, 150.0), half_size=15.0))

        samples = [
            ViewerSample(position=(100.0, 100.0), direction=(0.0, 1.0))
            for _ in range(10)
        ]

        # Test with default 1.0 second interval
        result = compute_attention_seconds(samples, [aoi], sample_interval=1.0)
        assert result.get_attention_seconds("shelf") == 10.0

        # Test with custom interval
        result_custom = compute_attention_seconds(samples, [aoi], sample_interval=0.5)
        assert result_custom.get_attention_seconds("shelf") == 5.0

    def test_invariant_total_samples_equals_hits_plus_misses(self) -> None:
        """Invariant: total_samples == samples_with_hits + samples_no_winner."""
        aoi = AOI(id="shelf", contour=make_square_contour((100.0, 150.0), half_size=15.0))

        samples = [
            ViewerSample(position=(100.0, 100.0), direction=(0.0, 1.0)),
            ViewerSample(position=(100.0, 100.0), direction=(0.0, -1.0)),
            ViewerSample(position=(100.0, 100.0), direction=(0.0, 1.0)),
        ]

        result = compute_attention_seconds(samples, [aoi])

        assert result.total_samples == result.samples_with_hits + result.samples_no_winner


class TestComputeAttentionAllAOIsRepresented:
    """Test that all AOIs are represented in results, even with 0 hits."""

    def test_aoi_with_zero_hits_in_result(self) -> None:
        """AOI that receives no attention should still appear in results."""
        aoi_visible = AOI(id="visible", contour=make_square_contour((100.0, 150.0), half_size=15.0))
        aoi_hidden = AOI(id="hidden", contour=make_square_contour((100.0, 50.0), half_size=15.0))

        samples = [
            ViewerSample(position=(100.0, 100.0), direction=(0.0, 1.0))
            for _ in range(5)
        ]

        result = compute_attention_seconds(samples, [aoi_visible, aoi_hidden])

        # Both AOIs should be in the result
        assert "visible" in result.aoi_ids
        assert "hidden" in result.aoi_ids
        
        # visible gets hits, hidden doesn't
        assert result.get_hit_count("visible") == 5
        assert result.get_hit_count("hidden") == 0
        assert result.get_attention_seconds("hidden") == 0.0

    def test_all_aois_zero_hits(self) -> None:
        """When viewer never sees any AOI, all should be in result with 0 hits."""
        aoi_a = AOI(id="A", contour=make_square_contour((200.0, 200.0), half_size=15.0))
        aoi_b = AOI(id="B", contour=make_square_contour((300.0, 300.0), half_size=15.0))

        # Viewer looking in wrong direction
        samples = [
            ViewerSample(position=(100.0, 100.0), direction=(0.0, -1.0))
            for _ in range(3)
        ]

        result = compute_attention_seconds(samples, [aoi_a, aoi_b], max_range=50.0)

        assert len(result.aoi_ids) == 2
        assert result.get_hit_count("A") == 0
        assert result.get_hit_count("B") == 0

    def test_mixed_id_types_all_represented(self) -> None:
        """String and integer AOI IDs should all be represented."""
        aoi_str = AOI(id="shelf", contour=make_square_contour((100.0, 150.0), half_size=15.0))
        aoi_int = AOI(id=42, contour=make_square_contour((150.0, 100.0), half_size=15.0))

        samples = [
            ViewerSample(position=(100.0, 100.0), direction=(0.0, 1.0))
        ]

        result = compute_attention_seconds(samples, [aoi_str, aoi_int])

        assert "shelf" in result.aoi_ids
        assert 42 in result.aoi_ids


class TestComputeAttentionTimestampsRecorded:
    """Test that hit timestamps (sample indices) are correctly tracked."""

    def test_hit_timestamps_recorded(self) -> None:
        """Sample indices where hits occurred should be recorded."""
        aoi = AOI(id="shelf", contour=make_square_contour((100.0, 150.0), half_size=15.0))

        samples = [
            ViewerSample(position=(100.0, 100.0), direction=(0.0, 1.0)),   # idx 0: hit
            ViewerSample(position=(100.0, 100.0), direction=(0.0, -1.0)),  # idx 1: miss
            ViewerSample(position=(100.0, 100.0), direction=(0.0, 1.0)),   # idx 2: hit
            ViewerSample(position=(100.0, 100.0), direction=(0.0, -1.0)),  # idx 3: miss
            ViewerSample(position=(100.0, 100.0), direction=(0.0, 1.0)),   # idx 4: hit
        ]

        result = compute_attention_seconds(samples, [aoi])

        aoi_result = result.get_aoi_result("shelf")
        assert aoi_result is not None
        assert aoi_result.hit_timestamps == [0, 2, 4]

    def test_multiple_aoi_timestamps(self) -> None:
        """Each AOI should have its own list of hit timestamps."""
        aoi_left = AOI(id="left", contour=make_square_contour((50.0, 100.0), half_size=15.0))
        aoi_right = AOI(id="right", contour=make_square_contour((150.0, 100.0), half_size=15.0))

        samples = [
            ViewerSample(position=(100.0, 100.0), direction=(-1.0, 0.0)),  # idx 0: left
            ViewerSample(position=(100.0, 100.0), direction=(1.0, 0.0)),   # idx 1: right
            ViewerSample(position=(100.0, 100.0), direction=(1.0, 0.0)),   # idx 2: right
            ViewerSample(position=(100.0, 100.0), direction=(-1.0, 0.0)),  # idx 3: left
        ]

        result = compute_attention_seconds(samples, [aoi_left, aoi_right])

        left_result = result.get_aoi_result("left")
        right_result = result.get_aoi_result("right")
        
        assert left_result is not None
        assert right_result is not None
        assert left_result.hit_timestamps == [0, 3]
        assert right_result.hit_timestamps == [1, 2]

    def test_empty_timestamps_for_zero_hits(self) -> None:
        """AOI with zero hits should have empty hit_timestamps list."""
        aoi = AOI(id="never_seen", contour=make_square_contour((100.0, 50.0), half_size=15.0))

        samples = [
            ViewerSample(position=(100.0, 100.0), direction=(0.0, 1.0))  # looking away
        ]

        result = compute_attention_seconds(samples, [aoi])

        aoi_result = result.get_aoi_result("never_seen")
        assert aoi_result is not None
        assert aoi_result.hit_timestamps == []


class TestComputeAttentionSessionConfig:
    """Test that session config is properly embedded in results."""

    def test_session_config_embedded(self) -> None:
        """SessionConfig should be embedded in result when provided."""
        config = SessionConfig(
            session_id="test-session-001",
            frame_size=(640, 480),
            viewer_id="viewer_42",
        )
        
        samples = [
            ViewerSample(position=(100.0, 100.0), direction=(0.0, 1.0))
        ]
        aoi = AOI(id="shelf", contour=make_square_contour((100.0, 150.0), half_size=15.0))

        result = compute_attention_seconds(samples, [aoi], session_config=config)

        assert isinstance(result, TrackingResultWithConfig)
        assert result.session_config is not None
        assert result.session_config.session_id == "test-session-001"
        assert result.session_config.viewer_id == "viewer_42"

    def test_no_session_config(self) -> None:
        """Result should have None session_config when not provided."""
        samples = [
            ViewerSample(position=(100.0, 100.0), direction=(0.0, 1.0))
        ]
        aoi = AOI(id="shelf", contour=make_square_contour((100.0, 150.0), half_size=15.0))

        result = compute_attention_seconds(samples, [aoi])

        assert isinstance(result, TrackingResultWithConfig)
        assert result.session_config is None


class TestComputeAttentionEmptyInputs:
    """Test compute_attention_seconds with edge case inputs."""

    def test_empty_samples_list(self) -> None:
        """Empty samples list should produce valid result with zero counts."""
        aoi = AOI(id="shelf", contour=make_square_contour((100.0, 150.0), half_size=15.0))

        result = compute_attention_seconds([], [aoi])

        assert result.total_samples == 0
        assert result.samples_with_hits == 0
        assert result.samples_no_winner == 0
        assert result.get_hit_count("shelf") == 0

    def test_empty_aois_list(self) -> None:
        """Empty AOIs list should produce result with all misses."""
        samples = [
            ViewerSample(position=(100.0, 100.0), direction=(0.0, 1.0))
            for _ in range(5)
        ]

        result = compute_attention_seconds(samples, [])

        assert result.total_samples == 5
        assert result.samples_with_hits == 0
        assert result.samples_no_winner == 5
        assert len(result.aoi_ids) == 0

    def test_both_empty(self) -> None:
        """Both empty should produce valid empty result."""
        result = compute_attention_seconds([], [])

        assert result.total_samples == 0
        assert result.samples_with_hits == 0
        assert result.samples_no_winner == 0


class TestComputeAttentionValidation:
    """Test input validation for compute_attention_seconds."""

    def test_invalid_samples_type(self) -> None:
        """Non-list samples should raise ValidationError."""
        aoi = AOI(id="shelf", contour=make_square_contour((100.0, 150.0), half_size=15.0))

        with pytest.raises(ValidationError, match="samples must be a list"):
            compute_attention_seconds("not a list", [aoi])  # type: ignore[arg-type]

    def test_invalid_aois_type(self) -> None:
        """Non-list aois should raise ValidationError."""
        sample = ViewerSample(position=(100.0, 100.0), direction=(0.0, 1.0))

        with pytest.raises(ValidationError, match="aois must be a list"):
            compute_attention_seconds([sample], "not a list")  # type: ignore[arg-type]

    def test_duplicate_aoi_ids(self) -> None:
        """Duplicate AOI IDs should raise ValidationError."""
        sample = ViewerSample(position=(100.0, 100.0), direction=(0.0, 1.0))
        aoi1 = AOI(id="same_id", contour=make_square_contour((100.0, 150.0), half_size=15.0))
        aoi2 = AOI(id="same_id", contour=make_square_contour((150.0, 100.0), half_size=15.0))

        with pytest.raises(ValidationError, match="Duplicate AOI IDs"):
            compute_attention_seconds([sample], [aoi1, aoi2])

    def test_invalid_fov_deg(self) -> None:
        """Invalid fov_deg should raise ValidationError."""
        sample = ViewerSample(position=(100.0, 100.0), direction=(0.0, 1.0))
        aoi = AOI(id="shelf", contour=make_square_contour((100.0, 150.0), half_size=15.0))

        with pytest.raises(ValidationError, match="fov_deg"):
            compute_attention_seconds([sample], [aoi], fov_deg=-10.0)

    def test_invalid_max_range(self) -> None:
        """Invalid max_range should raise ValidationError."""
        sample = ViewerSample(position=(100.0, 100.0), direction=(0.0, 1.0))
        aoi = AOI(id="shelf", contour=make_square_contour((100.0, 150.0), half_size=15.0))

        with pytest.raises(ValidationError, match="max_range"):
            compute_attention_seconds([sample], [aoi], max_range=0.0)

    def test_invalid_sample_interval(self) -> None:
        """Invalid sample_interval should raise ValidationError."""
        sample = ViewerSample(position=(100.0, 100.0), direction=(0.0, 1.0))
        aoi = AOI(id="shelf", contour=make_square_contour((100.0, 150.0), half_size=15.0))

        with pytest.raises(ValidationError, match="sample_interval"):
            compute_attention_seconds([sample], [aoi], sample_interval=-1.0)
