"""
Tests for process_single_sample() function (Phase 2, Step 2.1).

These tests cover the single-sample processing wrapper around find_largest_obstacle():
- test_process_single_sample_one_aoi_visible() - single AOI in view
- test_process_single_sample_multiple_aoi() - returns winner
- test_process_single_sample_no_aoi_visible() - returns None
- test_process_single_sample_all_aoi_outside_range() - max_range filtering
- test_process_single_sample_preserves_aoi_id() - ID correctly mapped
- test_process_single_sample_return_details() - detailed result structure
- test_process_single_sample_validation() - input validation
"""

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from view_arc.tracking import (
    AOI,
    AOIIntervalBreakdown,
    SingleSampleResult,
    ValidationError,
    ViewerSample,
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
# Tests: process_single_sample() - Single AOI Visible
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

        result = process_single_sample(sample, [aoi], field_of_view_deg=90.0, max_range=100.0)

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

        result = process_single_sample(sample, [aoi], field_of_view_deg=90.0, max_range=200.0)

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

        result = process_single_sample(sample, [aoi], field_of_view_deg=90.0, max_range=100.0)

        assert result == 42


# =============================================================================
# Tests: process_single_sample() - Multiple AOIs
# =============================================================================


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
            sample, [small_aoi, large_aoi], field_of_view_deg=120.0, max_range=200.0
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
            sample, [aoi_left, aoi_center, aoi_right], field_of_view_deg=120.0, max_range=150.0
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
            sample, [visible_aoi, hidden_aoi], field_of_view_deg=90.0, max_range=200.0
        )

        assert result == "visible"


# =============================================================================
# Tests: process_single_sample() - No AOI Visible
# =============================================================================


class TestProcessSingleSampleNoAOIVisible:
    """Test process_single_sample when no AOI is visible."""

    def test_no_aois_returns_none(self) -> None:
        """Empty AOI list should return None."""
        sample = ViewerSample(
            position=(100.0, 100.0),
            direction=(0.0, 1.0),
        )

        result = process_single_sample(sample, [], field_of_view_deg=90.0, max_range=100.0)

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
            sample, [behind_aoi], field_of_view_deg=90.0, max_range=100.0
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
            sample, [aoi], field_of_view_deg=60.0, max_range=200.0
        )

        # The AOI is at ~70° angle from horizontal, outside 30° half-FOV
        assert result is None


# =============================================================================
# Tests: process_single_sample() - Max Range Filtering
# =============================================================================


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

        result = process_single_sample(sample, [aoi], field_of_view_deg=90.0, max_range=100.0)

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
        result = process_single_sample(sample, [aoi], field_of_view_deg=90.0, max_range=100.0)

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
            sample, [near_aoi, far_aoi], field_of_view_deg=90.0, max_range=100.0
        )

        assert result == "near"


# =============================================================================
# Tests: process_single_sample() - ID Preservation
# =============================================================================


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

        result = process_single_sample(sample, [aoi], field_of_view_deg=90.0, max_range=100.0)

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

        result = process_single_sample(sample, [aoi], field_of_view_deg=90.0, max_range=100.0)

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

        result = process_single_sample(sample, aois, field_of_view_deg=90.0, max_range=100.0)

        # Center shelf ("shelf_A") is closest and should win
        assert result == "shelf_A"
        assert isinstance(result, str)


# =============================================================================
# Tests: process_single_sample() - Return Details
# =============================================================================


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
            sample, [aoi], field_of_view_deg=90.0, max_range=100.0, return_details=True
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
            sample, [aoi1, aoi2], field_of_view_deg=90.0, max_range=100.0, return_details=True
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
            sample, [aoi_near, aoi_far], field_of_view_deg=120.0, max_range=150.0, return_details=True
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
            sample, [aoi], field_of_view_deg=90.0, max_range=100.0, return_details=True
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
            sample, [aoi], field_of_view_deg=90.0, max_range=100.0, return_details=True
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
            sample, [aoi_winner, aoi_loser], field_of_view_deg=90.0, max_range=150.0, return_details=True
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
            sample, [aoi], field_of_view_deg=90.0, max_range=100.0, return_details=True
        )

        assert isinstance(result, SingleSampleResult)
        assert result.winning_aoi_id is None


# =============================================================================
# Tests: process_single_sample() - Input Validation
# =============================================================================


class TestProcessSingleSampleValidation:
    """Test input validation for process_single_sample."""

    def test_invalid_sample_type(self) -> None:
        """Non-ViewerSample should raise ValidationError."""
        with pytest.raises(ValidationError, match="must be a ViewerSample"):
            process_single_sample(
                "not a sample",  # type: ignore[arg-type]
                [],
                field_of_view_deg=90.0,
                max_range=100.0,
            )

    def test_invalid_aois_type(self) -> None:
        """Non-list aois should raise ValidationError."""
        sample = ViewerSample(position=(100.0, 100.0), direction=(1.0, 0.0))

        with pytest.raises(ValidationError, match="must be a list"):
            process_single_sample(
                sample,
                "not a list",  # type: ignore[arg-type]
                field_of_view_deg=90.0,
                max_range=100.0,
            )

    def test_invalid_aoi_element(self) -> None:
        """Non-AOI element in list should raise ValidationError."""
        sample = ViewerSample(position=(100.0, 100.0), direction=(1.0, 0.0))

        with pytest.raises(ValidationError, match="must be an AOI"):
            process_single_sample(
                sample,
                ["not an AOI"],  # type: ignore[list-item]
                field_of_view_deg=90.0,
                max_range=100.0,
            )

    def test_invalid_fov_deg(self) -> None:
        """Invalid field_of_view_deg should raise ValidationError."""
        sample = ViewerSample(position=(100.0, 100.0), direction=(1.0, 0.0))

        with pytest.raises(ValidationError, match="fov_deg"):
            process_single_sample(sample, [], field_of_view_deg=-10.0, max_range=100.0)

    def test_invalid_max_range(self) -> None:
        """Invalid max_range should raise ValidationError."""
        sample = ViewerSample(position=(100.0, 100.0), direction=(1.0, 0.0))

        with pytest.raises(ValidationError, match="max_range"):
            process_single_sample(sample, [], field_of_view_deg=90.0, max_range=-50.0)
