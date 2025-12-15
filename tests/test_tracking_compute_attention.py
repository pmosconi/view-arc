"""
Tests for compute_attention_seconds() function (Phase 2, Step 2.2+).

These tests cover the batch processing function:
- test_compute_attention_single_sample() - trivial case
- test_compute_attention_all_same_aoi() - viewer stares at one AOI
- test_compute_attention_alternating_aois() - viewer looks left/right
- test_compute_attention_no_hits() - viewer never looks at AOIs
- test_compute_attention_partial_hits() - some samples hit, some miss
- test_compute_attention_hit_count_accuracy() - verify counts
- test_compute_attention_all_aois_represented() - all AOIs in result
- test_compute_attention_timestamps_recorded() - hit indices tracked
- test_compute_attention_numpy_input() - numpy array input format
- test_compute_attention_frame_size_validation() - bounds checking
"""

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from view_arc.tracking import (
    AOI,
    SessionConfig,
    TrackingResultWithConfig,
    ValidationError,
    ViewerSample,
    compute_attention_seconds,
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
# Tests: Single Sample Case
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


# =============================================================================
# Tests: All Same AOI
# =============================================================================


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


# =============================================================================
# Tests: Alternating AOIs
# =============================================================================


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


# =============================================================================
# Tests: No Hits
# =============================================================================


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


# =============================================================================
# Tests: Partial Hits
# =============================================================================


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


# =============================================================================
# Tests: Hit Count Accuracy
# =============================================================================


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


# =============================================================================
# Tests: All AOIs Represented
# =============================================================================


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


# =============================================================================
# Tests: Timestamps Recorded
# =============================================================================


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


# =============================================================================
# Tests: Session Config
# =============================================================================


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


# =============================================================================
# Tests: Empty Inputs
# =============================================================================


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


# =============================================================================
# Tests: Input Validation
# =============================================================================


class TestComputeAttentionValidation:
    """Test input validation for compute_attention_seconds."""

    def test_invalid_samples_type(self) -> None:
        """Non-list/non-array samples should raise ValidationError."""
        aoi = AOI(id="shelf", contour=make_square_contour((100.0, 150.0), half_size=15.0))

        with pytest.raises(ValidationError, match="samples must be a list or numpy array"):
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
        """Invalid field_of_view_deg should raise ValidationError."""
        sample = ViewerSample(position=(100.0, 100.0), direction=(0.0, 1.0))
        aoi = AOI(id="shelf", contour=make_square_contour((100.0, 150.0), half_size=15.0))

        with pytest.raises(ValidationError, match="fov_deg"):
            compute_attention_seconds([sample], [aoi], field_of_view_deg=-10.0)

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


# =============================================================================
# Tests: NumPy Array Input (Step 2.3)
# =============================================================================


class TestComputeAttentionNumpyInput:
    """Test compute_attention_seconds with numpy array input format."""

    def test_numpy_array_shape_n4(self) -> None:
        """NumPy array of shape (N, 4) should be accepted."""
        # Array format: [x, y, dx, dy]
        samples = np.array([
            [100.0, 100.0, 0.0, 1.0],
            [100.0, 100.0, 0.0, 1.0],
            [100.0, 100.0, 1.0, 0.0],
        ])
        aoi = AOI(id="shelf", contour=make_square_contour((100.0, 150.0), half_size=20.0))

        result = compute_attention_seconds(samples, [aoi])

        assert result.total_samples == 3
        # First two samples look up (at the shelf), third looks right
        assert result.samples_with_hits == 2
        assert result.get_hit_count("shelf") == 2

    def test_numpy_array_direction_normalized(self) -> None:
        """Non-unit direction vectors in numpy input should be normalized."""
        # Direction vectors are not unit vectors - should be normalized
        samples = np.array([
            [100.0, 100.0, 0.0, 2.0],   # points up with mag 2
            [100.0, 100.0, 3.0, 0.0],   # points right with mag 3
        ])
        aoi_up = AOI(id="up", contour=make_square_contour((100.0, 150.0), half_size=15.0))
        aoi_right = AOI(id="right", contour=make_square_contour((150.0, 100.0), half_size=15.0))

        result = compute_attention_seconds(samples, [aoi_up, aoi_right])

        assert result.get_hit_count("up") == 1
        assert result.get_hit_count("right") == 1

    def test_numpy_array_empty(self) -> None:
        """Empty numpy array should produce valid empty result."""
        samples = np.array([]).reshape(0, 4)
        aoi = AOI(id="shelf", contour=make_square_contour((100.0, 150.0), half_size=15.0))

        result = compute_attention_seconds(samples, [aoi])

        assert result.total_samples == 0
        assert result.samples_with_hits == 0
        assert result.get_hit_count("shelf") == 0

    def test_numpy_array_single_sample(self) -> None:
        """Single sample in numpy array should work."""
        samples = np.array([[100.0, 100.0, 0.0, 1.0]])
        aoi = AOI(id="shelf", contour=make_square_contour((100.0, 150.0), half_size=20.0))

        result = compute_attention_seconds(samples, [aoi])

        assert result.total_samples == 1
        assert result.get_hit_count("shelf") == 1

    def test_numpy_array_wrong_dimensions(self) -> None:
        """1D numpy array should raise ValidationError."""
        samples = np.array([100.0, 100.0, 0.0, 1.0])  # 1D, not 2D

        with pytest.raises(ValidationError, match="2D array"):
            compute_attention_seconds(samples, [])

    def test_numpy_array_wrong_columns(self) -> None:
        """NumPy array with wrong number of columns should raise ValidationError."""
        samples = np.array([
            [100.0, 100.0, 0.0],  # Only 3 columns
        ])

        with pytest.raises(ValidationError, match=r"shape \(N, 4\)"):
            compute_attention_seconds(samples, [])

    def test_numpy_array_zero_direction(self) -> None:
        """NumPy array with zero-magnitude direction should raise ValidationError."""
        samples = np.array([
            [100.0, 100.0, 0.0, 0.0],  # Zero direction vector
        ])

        with pytest.raises(ValidationError, match="zero-magnitude"):
            compute_attention_seconds(samples, [])

    def test_numpy_array_matches_list_input(self) -> None:
        """NumPy array input should produce same result as list of ViewerSamples."""
        # Create equivalent inputs
        list_samples = [
            ViewerSample(position=(100.0, 100.0), direction=(0.0, 1.0)),
            ViewerSample(position=(100.0, 100.0), direction=(1.0, 0.0)),
            ViewerSample(position=(100.0, 100.0), direction=(0.0, 1.0)),
        ]
        numpy_samples = np.array([
            [100.0, 100.0, 0.0, 1.0],
            [100.0, 100.0, 1.0, 0.0],
            [100.0, 100.0, 0.0, 1.0],
        ])
        aoi = AOI(id="shelf", contour=make_square_contour((100.0, 150.0), half_size=20.0))

        result_list = compute_attention_seconds(list_samples, [aoi])
        result_numpy = compute_attention_seconds(numpy_samples, [aoi])

        assert result_list.total_samples == result_numpy.total_samples
        assert result_list.samples_with_hits == result_numpy.samples_with_hits
        assert result_list.get_hit_count("shelf") == result_numpy.get_hit_count("shelf")


# =============================================================================
# Tests: Frame Size Validation
# =============================================================================


class TestComputeAttentionFrameSizeValidation:
    """Test that frame_size from SessionConfig is used for bounds checking."""

    def test_frame_size_from_session_config_validates_positions(self) -> None:
        """SessionConfig with frame_size should validate sample positions."""
        config = SessionConfig(
            session_id="test-bounds",
            frame_size=(640, 480),
        )
        # Sample position outside frame bounds
        samples = [
            ViewerSample(position=(700.0, 100.0), direction=(0.0, 1.0)),  # x > 640
        ]
        aoi = AOI(id="shelf", contour=make_square_contour((100.0, 150.0), half_size=15.0))

        with pytest.raises(ValidationError, match="out of bounds"):
            compute_attention_seconds(samples, [aoi], session_config=config)

    def test_frame_size_y_out_of_bounds(self) -> None:
        """Sample with y position out of frame should raise ValidationError."""
        config = SessionConfig(
            session_id="test-bounds",
            frame_size=(640, 480),
        )
        samples = [
            ViewerSample(position=(100.0, 500.0), direction=(0.0, 1.0)),  # y > 480
        ]
        aoi = AOI(id="shelf", contour=make_square_contour((100.0, 150.0), half_size=15.0))

        with pytest.raises(ValidationError, match="out of bounds"):
            compute_attention_seconds(samples, [aoi], session_config=config)

    def test_frame_size_negative_position(self) -> None:
        """Sample with negative position should raise ValidationError."""
        config = SessionConfig(
            session_id="test-bounds",
            frame_size=(640, 480),
        )
        samples = [
            ViewerSample(position=(-10.0, 100.0), direction=(0.0, 1.0)),  # x < 0
        ]
        aoi = AOI(id="shelf", contour=make_square_contour((100.0, 150.0), half_size=15.0))

        with pytest.raises(ValidationError, match="out of bounds"):
            compute_attention_seconds(samples, [aoi], session_config=config)

    def test_frame_size_valid_positions_pass(self) -> None:
        """Samples within frame bounds should pass validation."""
        config = SessionConfig(
            session_id="test-bounds",
            frame_size=(640, 480),
        )
        samples = [
            ViewerSample(position=(100.0, 100.0), direction=(0.0, 1.0)),
            ViewerSample(position=(639.0, 479.0), direction=(0.0, 1.0)),  # edge of frame
            ViewerSample(position=(0.0, 0.0), direction=(0.0, 1.0)),  # corner
        ]
        aoi = AOI(id="shelf", contour=make_square_contour((100.0, 150.0), half_size=20.0))

        # Should not raise
        result = compute_attention_seconds(samples, [aoi], session_config=config)
        assert result.total_samples == 3

    def test_no_frame_size_skips_bounds_check(self) -> None:
        """Without frame_size, out-of-bounds positions are allowed."""
        # No session config
        samples = [
            ViewerSample(position=(1000.0, 1000.0), direction=(0.0, 1.0)),
        ]
        aoi = AOI(id="shelf", contour=make_square_contour((1000.0, 1050.0), half_size=20.0))

        # Should not raise - no bounds check
        result = compute_attention_seconds(samples, [aoi])
        assert result.total_samples == 1

    def test_session_config_without_frame_size_skips_bounds(self) -> None:
        """SessionConfig without frame_size should skip bounds check."""
        config = SessionConfig(
            session_id="no-frame",
            frame_size=None,  # Explicitly no frame size
        )
        samples = [
            ViewerSample(position=(1000.0, 1000.0), direction=(0.0, 1.0)),
        ]
        aoi = AOI(id="shelf", contour=make_square_contour((1000.0, 1050.0), half_size=20.0))

        # Should not raise - no bounds check
        result = compute_attention_seconds(samples, [aoi], session_config=config)
        assert result.total_samples == 1

    def test_frame_size_with_numpy_input(self) -> None:
        """Frame size validation should work with numpy array input."""
        config = SessionConfig(
            session_id="test-numpy-bounds",
            frame_size=(640, 480),
        )
        # Sample with x position out of bounds in numpy format
        samples = np.array([
            [700.0, 100.0, 0.0, 1.0],  # x > 640
        ])
        aoi = AOI(id="shelf", contour=make_square_contour((100.0, 150.0), half_size=15.0))

        with pytest.raises(ValidationError, match="out of bounds"):
            compute_attention_seconds(samples, [aoi], session_config=config)

    def test_frame_size_numpy_valid_positions(self) -> None:
        """Valid numpy samples within frame should pass."""
        config = SessionConfig(
            session_id="test-numpy-bounds",
            frame_size=(640, 480),
        )
        samples = np.array([
            [100.0, 100.0, 0.0, 1.0],
            [300.0, 200.0, 1.0, 0.0],
        ])
        aoi = AOI(id="shelf", contour=make_square_contour((100.0, 150.0), half_size=20.0))

        result = compute_attention_seconds(samples, [aoi], session_config=config)
        assert result.total_samples == 2
