"""
Tests for tracking module validation functions (Step 1.2).

Tests cover:
- validate_viewer_samples() - sample list integrity and bounds checking
- validate_aois() - AOI list validation and duplicate ID detection
- validate_tracking_params() - FOV, max_range, sample_interval validation
"""

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from view_arc.tracking import (
    AOI,
    ValidationError,
    ViewerSample,
    validate_aois,
    validate_tracking_params,
    validate_viewer_samples,
)


# =============================================================================
# Helper functions
# =============================================================================


def make_unit_vector(angle_deg: float) -> tuple[float, float]:
    """Create a unit vector from an angle in degrees."""
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


# =============================================================================
# Tests: validate_viewer_samples
# =============================================================================


class TestValidateViewerSamplesEmpty:
    """Tests for validate_viewer_samples with empty input."""

    def test_validate_samples_empty_list(self) -> None:
        """Test that empty sample list is handled gracefully."""
        # Empty list should be valid - graceful handling
        validate_viewer_samples([])  # Should not raise

    def test_validate_samples_empty_with_frame_size(self) -> None:
        """Test that empty sample list is valid even with frame_size."""
        validate_viewer_samples([], frame_size=(1920, 1080))  # Should not raise


class TestValidateViewerSamplesSingle:
    """Tests for validate_viewer_samples with single sample."""

    def test_validate_samples_single_valid(self) -> None:
        """Test that a single valid sample passes validation."""
        sample = ViewerSample(
            position=(100.0, 200.0),
            direction=(1.0, 0.0),
        )
        validate_viewer_samples([sample])  # Should not raise

    def test_validate_samples_single_with_timestamp(self) -> None:
        """Test that a single sample with timestamp is valid."""
        sample = ViewerSample(
            position=(50.0, 75.0),
            direction=(0.0, 1.0),
            timestamp=0.0,
        )
        validate_viewer_samples([sample])  # Should not raise

    def test_validate_samples_single_with_frame_size(self) -> None:
        """Test that a single sample within bounds is valid."""
        sample = ViewerSample(
            position=(100.0, 200.0),
            direction=(1.0, 0.0),
        )
        validate_viewer_samples([sample], frame_size=(1920, 1080))  # Should not raise


class TestValidateViewerSamplesBatch:
    """Tests for validate_viewer_samples with typical batch sizes."""

    def test_validate_samples_batch_60(self) -> None:
        """Test validation of 60 samples (1 minute of data)."""
        samples = [
            ViewerSample(
                position=(100.0 + i, 200.0),
                direction=make_unit_vector(i * 6),  # Rotate over time
                timestamp=float(i),
            )
            for i in range(60)
        ]
        validate_viewer_samples(samples)  # Should not raise

    def test_validate_samples_batch_large(self) -> None:
        """Test validation of large batch (300 samples = 5 minutes)."""
        samples = [
            ViewerSample(
                position=(500.0, 400.0),
                direction=(1.0, 0.0),
                timestamp=float(i),
            )
            for i in range(300)
        ]
        validate_viewer_samples(samples)  # Should not raise

    def test_validate_samples_batch_with_frame_size(self) -> None:
        """Test batch validation with frame bounds checking."""
        samples = [
            ViewerSample(
                position=(float(i * 10), float(i * 5)),
                direction=(1.0, 0.0),
            )
            for i in range(100)
        ]
        validate_viewer_samples(samples, frame_size=(1920, 1080))  # Should not raise


class TestValidateViewerSamplesInvalidPosition:
    """Tests for validate_viewer_samples with out-of-bounds positions."""

    def test_validate_samples_invalid_position_x_negative(self) -> None:
        """Test rejection of sample with negative x position."""
        sample = ViewerSample(
            position=(-10.0, 200.0),
            direction=(1.0, 0.0),
        )
        with pytest.raises(ValidationError, match="x position.*out of bounds"):
            validate_viewer_samples([sample], frame_size=(1920, 1080))

    def test_validate_samples_invalid_position_y_negative(self) -> None:
        """Test rejection of sample with negative y position."""
        sample = ViewerSample(
            position=(100.0, -50.0),
            direction=(1.0, 0.0),
        )
        with pytest.raises(ValidationError, match="y position.*out of bounds"):
            validate_viewer_samples([sample], frame_size=(1920, 1080))

    def test_validate_samples_invalid_position_x_exceeds(self) -> None:
        """Test rejection of sample with x position >= width."""
        sample = ViewerSample(
            position=(1920.0, 200.0),  # Exactly at width (invalid, should be < width)
            direction=(1.0, 0.0),
        )
        with pytest.raises(ValidationError, match="x position.*out of bounds"):
            validate_viewer_samples([sample], frame_size=(1920, 1080))

    def test_validate_samples_invalid_position_y_exceeds(self) -> None:
        """Test rejection of sample with y position >= height."""
        sample = ViewerSample(
            position=(100.0, 1080.0),  # Exactly at height (invalid)
            direction=(1.0, 0.0),
        )
        with pytest.raises(ValidationError, match="y position.*out of bounds"):
            validate_viewer_samples([sample], frame_size=(1920, 1080))

    def test_validate_samples_invalid_position_in_batch(self) -> None:
        """Test that error identifies the sample index."""
        samples = [
            ViewerSample(position=(100.0, 200.0), direction=(1.0, 0.0)),
            ViewerSample(position=(500.0, 400.0), direction=(0.0, 1.0)),
            ViewerSample(position=(2000.0, 500.0), direction=(1.0, 0.0)),  # Invalid
        ]
        with pytest.raises(ValidationError, match="index 2"):
            validate_viewer_samples(samples, frame_size=(1920, 1080))

    def test_validate_samples_no_bounds_check_without_frame_size(self) -> None:
        """Test that positions are not bounds-checked when frame_size is None."""
        # These positions would be invalid with a frame_size, but should pass without
        sample = ViewerSample(
            position=(-100.0, -200.0),
            direction=(1.0, 0.0),
        )
        validate_viewer_samples([sample])  # Should not raise


class TestValidateViewerSamplesInvalidType:
    """Tests for validate_viewer_samples with invalid input types."""

    def test_validate_samples_not_a_list(self) -> None:
        """Test rejection when samples is not a list."""
        with pytest.raises(ValidationError, match="samples must be a list"):
            validate_viewer_samples("not a list")  # type: ignore[arg-type]

    def test_validate_samples_tuple_not_list(self) -> None:
        """Test rejection when samples is a tuple instead of list."""
        sample = ViewerSample(position=(0.0, 0.0), direction=(1.0, 0.0))
        with pytest.raises(ValidationError, match="samples must be a list"):
            validate_viewer_samples((sample,))  # type: ignore[arg-type]

    def test_validate_samples_contains_non_sample(self) -> None:
        """Test rejection when list contains non-ViewerSample."""
        samples = [
            ViewerSample(position=(0.0, 0.0), direction=(1.0, 0.0)),
            {"position": (100.0, 100.0), "direction": (0.0, 1.0)},  # Dict, not ViewerSample
        ]
        with pytest.raises(ValidationError, match="index 1.*ViewerSample"):
            validate_viewer_samples(samples)  # type: ignore[arg-type]


class TestValidateViewerSamplesFrameSizeInvalid:
    """Tests for validate_viewer_samples with malformed frame_size."""

    def test_validate_samples_frame_size_not_tuple(self) -> None:
        """Test rejection when frame_size is not a tuple/list."""
        sample = ViewerSample(position=(100.0, 200.0), direction=(1.0, 0.0))
        with pytest.raises(ValidationError, match="frame_size must be a tuple"):
            validate_viewer_samples([sample], frame_size="1920x1080")  # type: ignore[arg-type]

    def test_validate_samples_frame_size_single_element(self) -> None:
        """Test rejection when frame_size has only one element."""
        sample = ViewerSample(position=(100.0, 200.0), direction=(1.0, 0.0))
        with pytest.raises(ValidationError, match="exactly 2 elements"):
            validate_viewer_samples([sample], frame_size=(1920,))  # type: ignore[arg-type]

    def test_validate_samples_frame_size_three_elements(self) -> None:
        """Test rejection when frame_size has three elements."""
        sample = ViewerSample(position=(100.0, 200.0), direction=(1.0, 0.0))
        with pytest.raises(ValidationError, match="exactly 2 elements"):
            validate_viewer_samples([sample], frame_size=(1920, 1080, 3))  # type: ignore[arg-type]

    def test_validate_samples_frame_size_string_width(self) -> None:
        """Test rejection when frame_size width is a string."""
        sample = ViewerSample(position=(100.0, 200.0), direction=(1.0, 0.0))
        with pytest.raises(ValidationError, match="width must be a number"):
            validate_viewer_samples([sample], frame_size=("1920", 1080))  # type: ignore[arg-type]

    def test_validate_samples_frame_size_string_height(self) -> None:
        """Test rejection when frame_size height is a string."""
        sample = ViewerSample(position=(100.0, 200.0), direction=(1.0, 0.0))
        with pytest.raises(ValidationError, match="height must be a number"):
            validate_viewer_samples([sample], frame_size=(1920, "1080"))  # type: ignore[arg-type]

    def test_validate_samples_frame_size_zero_width(self) -> None:
        """Test rejection when frame_size width is zero."""
        sample = ViewerSample(position=(100.0, 200.0), direction=(1.0, 0.0))
        with pytest.raises(ValidationError, match="width must be positive"):
            validate_viewer_samples([sample], frame_size=(0, 1080))

    def test_validate_samples_frame_size_zero_height(self) -> None:
        """Test rejection when frame_size height is zero."""
        sample = ViewerSample(position=(100.0, 200.0), direction=(1.0, 0.0))
        with pytest.raises(ValidationError, match="height must be positive"):
            validate_viewer_samples([sample], frame_size=(1920, 0))

    def test_validate_samples_frame_size_negative_width(self) -> None:
        """Test rejection when frame_size width is negative."""
        sample = ViewerSample(position=(100.0, 200.0), direction=(1.0, 0.0))
        with pytest.raises(ValidationError, match="width must be positive"):
            validate_viewer_samples([sample], frame_size=(-1920, 1080))

    def test_validate_samples_frame_size_negative_height(self) -> None:
        """Test rejection when frame_size height is negative."""
        sample = ViewerSample(position=(100.0, 200.0), direction=(1.0, 0.0))
        with pytest.raises(ValidationError, match="height must be positive"):
            validate_viewer_samples([sample], frame_size=(1920, -1080))

    def test_validate_samples_frame_size_nan_width(self) -> None:
        """Test rejection when frame_size width is NaN."""
        sample = ViewerSample(position=(100.0, 200.0), direction=(1.0, 0.0))
        with pytest.raises(ValidationError, match="width must be finite"):
            validate_viewer_samples([sample], frame_size=(float("nan"), 1080.0))

    def test_validate_samples_frame_size_nan_height(self) -> None:
        """Test rejection when frame_size height is NaN."""
        sample = ViewerSample(position=(100.0, 200.0), direction=(1.0, 0.0))
        with pytest.raises(ValidationError, match="height must be finite"):
            validate_viewer_samples([sample], frame_size=(1920.0, float("nan")))

    def test_validate_samples_frame_size_inf_width(self) -> None:
        """Test rejection when frame_size width is infinity."""
        sample = ViewerSample(position=(100.0, 200.0), direction=(1.0, 0.0))
        with pytest.raises(ValidationError, match="width must be finite"):
            validate_viewer_samples([sample], frame_size=(float("inf"), 1080.0))

    def test_validate_samples_frame_size_inf_height(self) -> None:
        """Test rejection when frame_size height is infinity."""
        sample = ViewerSample(position=(100.0, 200.0), direction=(1.0, 0.0))
        with pytest.raises(ValidationError, match="height must be finite"):
            validate_viewer_samples([sample], frame_size=(1920.0, float("inf")))

    def test_validate_samples_frame_size_negative_inf_width(self) -> None:
        """Test rejection when frame_size width is negative infinity."""
        sample = ViewerSample(position=(100.0, 200.0), direction=(1.0, 0.0))
        with pytest.raises(ValidationError, match="width must be finite"):
            validate_viewer_samples([sample], frame_size=(float("-inf"), 1080.0))

    def test_validate_samples_frame_size_list_accepted(self) -> None:
        """Test that frame_size as list is accepted."""
        sample = ViewerSample(position=(100.0, 200.0), direction=(1.0, 0.0))
        validate_viewer_samples([sample], frame_size=[1920, 1080])  # type: ignore[arg-type]


class TestValidateViewerSamplesFrameSizeNumpyScalars:
    """Tests for validate_viewer_samples frame_size with numpy scalar types."""

    def test_validate_samples_frame_size_numpy_int32(self) -> None:
        """Test that numpy int32 frame_size values are accepted."""
        sample = ViewerSample(position=(100.0, 200.0), direction=(1.0, 0.0))
        validate_viewer_samples(
            [sample],
            frame_size=(np.int32(1920), np.int32(1080)),  # type: ignore[arg-type]
        )  # Should not raise

    def test_validate_samples_frame_size_numpy_float64(self) -> None:
        """Test that numpy float64 frame_size values are accepted."""
        sample = ViewerSample(position=(100.0, 200.0), direction=(1.0, 0.0))
        validate_viewer_samples(
            [sample],
            frame_size=(np.float64(1920.0), np.float64(1080.0)),  # type: ignore[arg-type]
        )  # Should not raise


# =============================================================================
# Tests: validate_aois
# =============================================================================


class TestValidateAoisEmpty:
    """Tests for validate_aois with empty input."""

    def test_validate_aois_empty_list(self) -> None:
        """Test that empty AOI list is handled gracefully."""
        validate_aois([])  # Should not raise


class TestValidateAoisDuplicateIds:
    """Tests for validate_aois with duplicate IDs."""

    def test_validate_aois_duplicate_string_ids(self) -> None:
        """Test rejection of duplicate string IDs."""
        aois = [
            AOI(id="shelf_1", contour=make_square_contour((100.0, 100.0))),
            AOI(id="shelf_2", contour=make_square_contour((200.0, 100.0))),
            AOI(id="shelf_1", contour=make_square_contour((300.0, 100.0))),  # Duplicate
        ]
        with pytest.raises(ValidationError, match="Duplicate AOI IDs.*shelf_1"):
            validate_aois(aois)

    def test_validate_aois_duplicate_int_ids(self) -> None:
        """Test rejection of duplicate integer IDs."""
        aois = [
            AOI(id=1, contour=make_square_contour((100.0, 100.0))),
            AOI(id=2, contour=make_square_contour((200.0, 100.0))),
            AOI(id=1, contour=make_square_contour((300.0, 100.0))),  # Duplicate
        ]
        with pytest.raises(ValidationError, match="Duplicate AOI IDs.*1"):
            validate_aois(aois)

    def test_validate_aois_multiple_duplicates(self) -> None:
        """Test that all duplicate IDs are reported."""
        aois = [
            AOI(id="a", contour=make_square_contour((100.0, 100.0))),
            AOI(id="b", contour=make_square_contour((200.0, 100.0))),
            AOI(id="a", contour=make_square_contour((300.0, 100.0))),  # Duplicate
            AOI(id="b", contour=make_square_contour((400.0, 100.0))),  # Duplicate
        ]
        with pytest.raises(ValidationError, match="Duplicate AOI IDs"):
            validate_aois(aois)


class TestValidateAoisMixedIdTypes:
    """Tests for validate_aois with mixed string and integer IDs."""

    def test_validate_aois_mixed_id_types(self) -> None:
        """Test that mixed string and integer IDs coexist."""
        aois = [
            AOI(id="shelf_1", contour=make_square_contour((100.0, 100.0))),
            AOI(id=42, contour=make_square_contour((200.0, 100.0))),
            AOI(id="display_A", contour=make_square_contour((300.0, 100.0))),
            AOI(id=99, contour=make_square_contour((400.0, 100.0))),
        ]
        validate_aois(aois)  # Should not raise

    def test_validate_aois_string_and_int_same_value(self) -> None:
        """Test that string '1' and int 1 are considered different IDs."""
        aois = [
            AOI(id=1, contour=make_square_contour((100.0, 100.0))),
            AOI(id="1", contour=make_square_contour((200.0, 100.0))),
        ]
        validate_aois(aois)  # Should not raise - different types


class TestValidateAoisValidCases:
    """Tests for validate_aois with valid inputs."""

    def test_validate_aois_single(self) -> None:
        """Test validation of single AOI."""
        aois = [AOI(id="only_one", contour=make_square_contour((100.0, 100.0)))]
        validate_aois(aois)  # Should not raise

    def test_validate_aois_many(self) -> None:
        """Test validation of many AOIs with unique IDs."""
        aois = [
            AOI(id=f"shelf_{i}", contour=make_square_contour((float(i * 50), 100.0)))
            for i in range(100)
        ]
        validate_aois(aois)  # Should not raise


class TestValidateAoisInvalidType:
    """Tests for validate_aois with invalid input types."""

    def test_validate_aois_not_a_list(self) -> None:
        """Test rejection when aois is not a list."""
        with pytest.raises(ValidationError, match="aois must be a list"):
            validate_aois("not a list")  # type: ignore[arg-type]

    def test_validate_aois_contains_non_aoi(self) -> None:
        """Test rejection when list contains non-AOI."""
        aois = [
            AOI(id="valid", contour=make_square_contour((100.0, 100.0))),
            {"id": "invalid", "contour": [[0, 0], [1, 0], [1, 1]]},  # Dict, not AOI
        ]
        with pytest.raises(ValidationError, match="index 1.*AOI"):
            validate_aois(aois)  # type: ignore[arg-type]


# =============================================================================
# Tests: validate_tracking_params
# =============================================================================


class TestValidateTrackingParamsValid:
    """Tests for validate_tracking_params with valid inputs."""

    def test_validate_params_default_values(self) -> None:
        """Test validation with typical default values."""
        validate_tracking_params(fov_deg=90.0, max_range=500.0)  # Should not raise

    def test_validate_params_minimum_fov(self) -> None:
        """Test validation with minimum valid FOV."""
        validate_tracking_params(fov_deg=0.1, max_range=100.0)  # Should not raise

    def test_validate_params_maximum_fov(self) -> None:
        """Test validation with maximum valid FOV (360 degrees)."""
        validate_tracking_params(fov_deg=360.0, max_range=100.0)  # Should not raise

    def test_validate_params_custom_sample_interval(self) -> None:
        """Test validation with custom sample interval."""
        validate_tracking_params(
            fov_deg=90.0,
            max_range=500.0,
            sample_interval=0.5,
        )  # Should not raise

    def test_validate_params_integer_values(self) -> None:
        """Test that integer values are accepted."""
        validate_tracking_params(fov_deg=90, max_range=500, sample_interval=1)


class TestValidateTrackingParamsFovInvalid:
    """Tests for validate_tracking_params with invalid FOV."""

    def test_validate_params_fov_zero(self) -> None:
        """Test rejection of zero FOV."""
        with pytest.raises(ValidationError, match="fov_deg.*range.*0.*360"):
            validate_tracking_params(fov_deg=0.0, max_range=100.0)

    def test_validate_params_fov_negative(self) -> None:
        """Test rejection of negative FOV."""
        with pytest.raises(ValidationError, match="fov_deg.*range.*0.*360"):
            validate_tracking_params(fov_deg=-45.0, max_range=100.0)

    def test_validate_params_fov_exceeds_360(self) -> None:
        """Test rejection of FOV > 360."""
        with pytest.raises(ValidationError, match="fov_deg.*range.*0.*360"):
            validate_tracking_params(fov_deg=361.0, max_range=100.0)

    def test_validate_params_fov_invalid_type(self) -> None:
        """Test rejection of non-numeric FOV."""
        with pytest.raises(ValidationError, match="fov_deg must be a number"):
            validate_tracking_params(fov_deg="ninety", max_range=100.0)  # type: ignore[arg-type]


class TestValidateTrackingParamsMaxRangeInvalid:
    """Tests for validate_tracking_params with invalid max_range."""

    def test_validate_params_max_range_zero(self) -> None:
        """Test rejection of zero max_range."""
        with pytest.raises(ValidationError, match="max_range must be positive"):
            validate_tracking_params(fov_deg=90.0, max_range=0.0)

    def test_validate_params_max_range_negative(self) -> None:
        """Test rejection of negative max_range."""
        with pytest.raises(ValidationError, match="max_range must be positive"):
            validate_tracking_params(fov_deg=90.0, max_range=-100.0)

    def test_validate_params_max_range_invalid_type(self) -> None:
        """Test rejection of non-numeric max_range."""
        with pytest.raises(ValidationError, match="max_range must be a number"):
            validate_tracking_params(fov_deg=90.0, max_range="five hundred")  # type: ignore[arg-type]


class TestValidateTrackingParamsSampleIntervalInvalid:
    """Tests for validate_tracking_params with invalid sample_interval."""

    def test_validate_params_sample_interval_zero(self) -> None:
        """Test rejection of zero sample_interval."""
        with pytest.raises(ValidationError, match="sample_interval must be positive"):
            validate_tracking_params(fov_deg=90.0, max_range=100.0, sample_interval=0.0)

    def test_validate_params_sample_interval_negative(self) -> None:
        """Test rejection of negative sample_interval."""
        with pytest.raises(ValidationError, match="sample_interval must be positive"):
            validate_tracking_params(fov_deg=90.0, max_range=100.0, sample_interval=-1.0)

    def test_validate_params_sample_interval_invalid_type(self) -> None:
        """Test rejection of non-numeric sample_interval."""
        with pytest.raises(ValidationError, match="sample_interval must be a number"):
            validate_tracking_params(
                fov_deg=90.0,
                max_range=100.0,
                sample_interval="one",  # type: ignore[arg-type]
            )


class TestValidateTrackingParamsNonFinite:
    """Tests for validate_tracking_params rejection of NaN and infinity."""

    def test_validate_params_fov_nan(self) -> None:
        """Test rejection of NaN FOV."""
        with pytest.raises(ValidationError, match="fov_deg must be finite"):
            validate_tracking_params(fov_deg=float("nan"), max_range=100.0)

    def test_validate_params_fov_inf(self) -> None:
        """Test rejection of infinite FOV."""
        with pytest.raises(ValidationError, match="fov_deg must be finite"):
            validate_tracking_params(fov_deg=float("inf"), max_range=100.0)

    def test_validate_params_fov_negative_inf(self) -> None:
        """Test rejection of negative infinite FOV."""
        with pytest.raises(ValidationError, match="fov_deg must be finite"):
            validate_tracking_params(fov_deg=float("-inf"), max_range=100.0)

    def test_validate_params_max_range_nan(self) -> None:
        """Test rejection of NaN max_range."""
        with pytest.raises(ValidationError, match="max_range must be finite"):
            validate_tracking_params(fov_deg=90.0, max_range=float("nan"))

    def test_validate_params_max_range_inf(self) -> None:
        """Test rejection of infinite max_range."""
        with pytest.raises(ValidationError, match="max_range must be finite"):
            validate_tracking_params(fov_deg=90.0, max_range=float("inf"))

    def test_validate_params_max_range_negative_inf(self) -> None:
        """Test rejection of negative infinite max_range."""
        with pytest.raises(ValidationError, match="max_range must be finite"):
            validate_tracking_params(fov_deg=90.0, max_range=float("-inf"))

    def test_validate_params_sample_interval_nan(self) -> None:
        """Test rejection of NaN sample_interval."""
        with pytest.raises(ValidationError, match="sample_interval must be finite"):
            validate_tracking_params(
                fov_deg=90.0, max_range=100.0, sample_interval=float("nan")
            )

    def test_validate_params_sample_interval_inf(self) -> None:
        """Test rejection of infinite sample_interval."""
        with pytest.raises(ValidationError, match="sample_interval must be finite"):
            validate_tracking_params(
                fov_deg=90.0, max_range=100.0, sample_interval=float("inf")
            )

    def test_validate_params_sample_interval_negative_inf(self) -> None:
        """Test rejection of negative infinite sample_interval."""
        with pytest.raises(ValidationError, match="sample_interval must be finite"):
            validate_tracking_params(
                fov_deg=90.0, max_range=100.0, sample_interval=float("-inf")
            )


class TestValidateTrackingParamsNumpyScalars:
    """Tests for validate_tracking_params acceptance of numpy scalar types."""

    def test_validate_params_numpy_float32(self) -> None:
        """Test that numpy float32 values are accepted."""
        validate_tracking_params(
            fov_deg=np.float32(90.0),  # type: ignore[arg-type]
            max_range=np.float32(500.0),  # type: ignore[arg-type]
            sample_interval=np.float32(1.0),  # type: ignore[arg-type]
        )  # Should not raise

    def test_validate_params_numpy_float64(self) -> None:
        """Test that numpy float64 values are accepted."""
        validate_tracking_params(
            fov_deg=np.float64(90.0),  # type: ignore[arg-type]
            max_range=np.float64(500.0),  # type: ignore[arg-type]
            sample_interval=np.float64(1.0),  # type: ignore[arg-type]
        )  # Should not raise

    def test_validate_params_numpy_int32(self) -> None:
        """Test that numpy int32 values are accepted."""
        validate_tracking_params(
            fov_deg=np.int32(90),  # type: ignore[arg-type]
            max_range=np.int32(500),  # type: ignore[arg-type]
            sample_interval=np.int32(1),  # type: ignore[arg-type]
        )  # Should not raise

    def test_validate_params_numpy_int64(self) -> None:
        """Test that numpy int64 values are accepted."""
        validate_tracking_params(
            fov_deg=np.int64(90),  # type: ignore[arg-type]
            max_range=np.int64(500),  # type: ignore[arg-type]
            sample_interval=np.int64(1),  # type: ignore[arg-type]
        )  # Should not raise

    def test_validate_params_numpy_array_element(self) -> None:
        """Test that values extracted from numpy arrays are accepted."""
        config = np.array([90.0, 500.0, 1.0])
        validate_tracking_params(
            fov_deg=config[0],  # type: ignore[arg-type]
            max_range=config[1],  # type: ignore[arg-type]
            sample_interval=config[2],  # type: ignore[arg-type]
        )  # Should not raise

    def test_validate_params_mixed_numpy_and_python(self) -> None:
        """Test that mixing numpy and Python types works."""
        validate_tracking_params(
            fov_deg=np.float64(90.0),  # type: ignore[arg-type]
            max_range=500,  # Python int
            sample_interval=1.0,  # Python float
        )  # Should not raise


# =============================================================================
# Tests: normalize_sample_input() - Missing Direction Handling
# =============================================================================


class TestNormalizeSampleInputMissingDirection:
    """Test normalize_sample_input() with missing direction data."""

    def test_normalize_numpy_zero_direction_default(self) -> None:
        """Test that (0, 0) direction raises error by default."""
        from view_arc.tracking.validation import normalize_sample_input
        
        # NumPy array with zero-magnitude direction
        samples = np.array([
            [100.0, 100.0, 0.0, 0.0],
        ])
        
        with pytest.raises(ValidationError) as exc_info:
            normalize_sample_input(samples)
        
        assert "unit vector" in str(exc_info.value).lower() or "zero" in str(exc_info.value).lower()

    def test_normalize_numpy_zero_direction_allowed(self) -> None:
        """Test that (0, 0) creates sample with allow_missing_direction=True when flag is set."""
        from view_arc.tracking.validation import normalize_sample_input
        
        # NumPy array with zero-magnitude direction
        samples = np.array([
            [100.0, 100.0, 0.0, 0.0],
            [200.0, 200.0, 1.0, 0.0],
        ])
        
        result = normalize_sample_input(samples, allow_missing_direction=True)
        
        assert len(result) == 2
        
        # First sample should have missing direction flag
        assert result[0].position == (100.0, 100.0)
        assert result[0].direction == (0.0, 0.0)
        assert result[0].allow_missing_direction is True
        
        # Second sample should be normal
        assert result[1].position == (200.0, 200.0)
        assert result[1].direction == (1.0, 0.0)
        assert result[1].allow_missing_direction is False

    def test_normalize_numpy_all_zero_directions(self) -> None:
        """Test NumPy array with all zero directions and flag=True."""
        from view_arc.tracking.validation import normalize_sample_input
        
        samples = np.array([
            [100.0, 100.0, 0.0, 0.0],
            [200.0, 200.0, 0.0, 0.0],
            [300.0, 300.0, 0.0, 0.0],
        ])
        
        result = normalize_sample_input(samples, allow_missing_direction=True)
        
        assert len(result) == 3
        for sample in result:
            assert sample.direction == (0.0, 0.0)
            assert sample.allow_missing_direction is True

    def test_normalize_list_with_missing_direction_flag_true(self) -> None:
        """Test that list with ViewerSample(..., allow_missing_direction=True) passes through."""
        from view_arc.tracking.validation import normalize_sample_input
        
        samples = [
            ViewerSample(
                position=(100.0, 100.0),
                direction=(0.0, 0.0),
                allow_missing_direction=True
            ),
            ViewerSample(
                position=(200.0, 200.0),
                direction=(1.0, 0.0)
            ),
        ]
        
        result = normalize_sample_input(samples)
        
        # Should return same list
        assert result is samples
        assert len(result) == 2
        assert result[0].allow_missing_direction is True
        assert result[1].allow_missing_direction is False

    def test_normalize_list_with_missing_direction_flag_false(self) -> None:
        """Test that list with ViewerSample(direction=(0,0), allow_missing_direction=False) raises error."""
        from view_arc.tracking.validation import normalize_sample_input
        
        # This should fail during ViewerSample construction
        with pytest.raises(ValidationError) as exc_info:
            samples = [
                ViewerSample(
                    position=(100.0, 100.0),
                    direction=(0.0, 0.0),
                    allow_missing_direction=False
                ),
            ]
        
        assert "unit vector" in str(exc_info.value).lower()

    def test_normalize_numpy_mixed_valid_and_zero_directions(self) -> None:
        """Test NumPy array with mixture of valid and zero-magnitude directions."""
        from view_arc.tracking.validation import normalize_sample_input
        
        samples = np.array([
            [100.0, 100.0, 1.0, 0.0],   # Valid
            [200.0, 200.0, 0.0, 0.0],   # Missing
            [300.0, 300.0, 0.0, 1.0],   # Valid
            [400.0, 400.0, 0.0, 0.0],   # Missing
            [500.0, 500.0, -1.0, 0.0],  # Valid
        ])
        
        result = normalize_sample_input(samples, allow_missing_direction=True)
        
        assert len(result) == 5
        
        # Check each sample
        assert result[0].allow_missing_direction is False  # Valid direction
        assert result[1].allow_missing_direction is True   # Missing
        assert result[2].allow_missing_direction is False  # Valid
        assert result[3].allow_missing_direction is True   # Missing
        assert result[4].allow_missing_direction is False  # Valid
