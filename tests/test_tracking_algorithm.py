"""
Tests for tracking algorithm functions (Phase 2).

Step 2.1: Tests for process_single_sample()
- test_process_single_sample_one_aoi_visible() - single AOI in view
- test_process_single_sample_multiple_aoi() - returns winner
- test_process_single_sample_no_aoi_visible() - returns None
- test_process_single_sample_all_aoi_outside_range() - max_range filtering
- test_process_single_sample_preserves_aoi_id() - ID correctly mapped
"""

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from view_arc.tracking import (
    AOI,
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
        """Larger AOI (more angular coverage) should win."""
        # Viewer at (100, 100), looking up
        sample = ViewerSample(
            position=(100.0, 100.0),
            direction=(0.0, 1.0),
        )

        # Small AOI closer to viewer
        small_aoi = AOI(
            id="small_shelf",
            contour=make_square_contour((100.0, 130.0), half_size=10.0),
        )

        # Large AOI farther from viewer (but not occluded by small one)
        large_aoi = AOI(
            id="large_shelf",
            contour=make_rectangle_contour((150.0, 160.0), width=60.0, height=40.0),
        )

        result = process_single_sample(
            sample, [small_aoi, large_aoi], fov_deg=90.0, max_range=200.0
        )

        # The one with larger angular coverage wins
        assert result in ["small_shelf", "large_shelf"]

    def test_multiple_aois_all_visible(self) -> None:
        """When multiple AOIs are visible, one should be selected as winner."""
        sample = ViewerSample(
            position=(100.0, 100.0),
            direction=(0.0, 1.0),
        )

        aoi_left = AOI(
            id="left_shelf",
            contour=make_square_contour((70.0, 150.0), half_size=15.0),
        )

        aoi_center = AOI(
            id="center_shelf",
            contour=make_square_contour((100.0, 150.0), half_size=15.0),
        )

        aoi_right = AOI(
            id="right_shelf",
            contour=make_square_contour((130.0, 150.0), half_size=15.0),
        )

        result = process_single_sample(
            sample, [aoi_left, aoi_center, aoi_right], fov_deg=90.0, max_range=100.0
        )

        # One of them should win
        assert result in ["left_shelf", "center_shelf", "right_shelf"]

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
        """Mixed string and integer IDs should all be preserved."""
        sample = ViewerSample(
            position=(100.0, 100.0),
            direction=(0.0, 1.0),
        )

        aois = [
            AOI(id=1, contour=make_square_contour((80.0, 150.0), half_size=10.0)),
            AOI(id="shelf_A", contour=make_square_contour((100.0, 150.0), half_size=10.0)),
            AOI(id=2, contour=make_square_contour((120.0, 150.0), half_size=10.0)),
        ]

        result = process_single_sample(sample, aois, fov_deg=90.0, max_range=100.0)

        # Result should be one of the original IDs
        assert result in [1, "shelf_A", 2]


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
