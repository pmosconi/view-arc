"""
Tests for SessionConfig (Step 1.3).

Tests cover:
- SessionConfig default values
- SessionConfig with custom viewer metadata
- SessionConfig convenience properties
- SessionConfig input validation
- SessionConfig acceptance of various valid types
- Integration with validate_viewer_samples frame_size
"""

import numpy as np
import pytest

from view_arc.tracking import (
    SessionConfig,
    ValidationError,
    ViewerSample,
    validate_viewer_samples,
)


# =============================================================================
# Tests: SessionConfig Defaults
# =============================================================================


class TestSessionConfigDefaults:
    """Tests for SessionConfig default value behavior."""

    def test_session_config_defaults_applied(self) -> None:
        """Test that default values are correctly applied."""
        config = SessionConfig(session_id="test_session")

        assert config.session_id == "test_session"
        assert config.frame_size is None
        assert config.coordinate_space == "image"
        assert config.sample_interval_seconds == 1.0
        assert config.viewer_id is None
        assert config.notes is None

    def test_session_config_default_coordinate_space(self) -> None:
        """Test that coordinate_space defaults to 'image'."""
        config = SessionConfig(session_id="session_1")
        assert config.coordinate_space == "image"

    def test_session_config_default_sample_interval(self) -> None:
        """Test that sample_interval_seconds defaults to 1.0."""
        config = SessionConfig(session_id="session_1")
        assert config.sample_interval_seconds == 1.0

    def test_session_config_immutable(self) -> None:
        """Test that SessionConfig is frozen (immutable)."""
        config = SessionConfig(session_id="test")

        with pytest.raises(AttributeError):
            config.session_id = "modified"  # type: ignore[misc]

        with pytest.raises(AttributeError):
            config.frame_size = (1920, 1080)  # type: ignore[misc]


# =============================================================================
# Tests: SessionConfig Custom Viewer Metadata
# =============================================================================


class TestSessionConfigCustomViewerMetadata:
    """Tests for SessionConfig with custom viewer metadata."""

    def test_session_config_allows_custom_viewer_metadata(self) -> None:
        """Test that custom viewer_id is properly stored."""
        config = SessionConfig(
            session_id="store_visit_2025",
            viewer_id="customer_42",
        )

        assert config.viewer_id == "customer_42"

    def test_session_config_with_all_fields(self) -> None:
        """Test creating SessionConfig with all fields populated."""
        notes = {"store_id": "NYC_001", "timestamp": "2025-01-15T10:30:00Z"}
        config = SessionConfig(
            session_id="complete_session",
            frame_size=(1920, 1080),
            coordinate_space="image",
            sample_interval_seconds=0.5,
            viewer_id="viewer_A",
            notes=notes,
        )

        assert config.session_id == "complete_session"
        assert config.frame_size == (1920, 1080)
        assert config.coordinate_space == "image"
        assert config.sample_interval_seconds == 0.5
        assert config.viewer_id == "viewer_A"
        assert config.notes == notes

    def test_session_config_notes_dict(self) -> None:
        """Test that notes can contain arbitrary metadata."""
        notes = {
            "experiment_id": 123,
            "conditions": ["baseline", "treatment"],
            "nested": {"key": "value"},
        }
        config = SessionConfig(session_id="test", notes=notes)

        assert config.notes == notes
        assert config.notes["experiment_id"] == 123  # type: ignore[index]
        assert config.notes["conditions"] == ["baseline", "treatment"]  # type: ignore[index]


# =============================================================================
# Tests: SessionConfig Properties
# =============================================================================


class TestSessionConfigProperties:
    """Tests for SessionConfig convenience properties."""

    def test_session_config_has_frame_bounds_true(self) -> None:
        """Test has_frame_bounds returns True when frame_size is set."""
        config = SessionConfig(session_id="test", frame_size=(1920, 1080))
        assert config.has_frame_bounds is True

    def test_session_config_has_frame_bounds_false(self) -> None:
        """Test has_frame_bounds returns False when frame_size is None."""
        config = SessionConfig(session_id="test")
        assert config.has_frame_bounds is False

    def test_session_config_width_height_properties(self) -> None:
        """Test width and height properties return correct values."""
        config = SessionConfig(session_id="test", frame_size=(1920, 1080))

        assert config.width == 1920
        assert config.height == 1080

    def test_session_config_width_height_none(self) -> None:
        """Test width and height return None when frame_size is None."""
        config = SessionConfig(session_id="test")

        assert config.width is None
        assert config.height is None


# =============================================================================
# Tests: SessionConfig Validation
# =============================================================================


class TestSessionConfigValidation:
    """Tests for SessionConfig input validation."""

    def test_session_config_rejects_empty_session_id(self) -> None:
        """Test that empty session_id is rejected."""
        with pytest.raises(ValidationError, match="session_id cannot be empty"):
            SessionConfig(session_id="")

    def test_session_config_rejects_non_string_session_id(self) -> None:
        """Test that non-string session_id is rejected."""
        with pytest.raises(ValidationError, match="session_id must be a string"):
            SessionConfig(session_id=123)  # type: ignore[arg-type]

    def test_session_config_rejects_invalid_frame_size_type(self) -> None:
        """Test that non-tuple frame_size is rejected."""
        with pytest.raises(ValidationError, match="frame_size must be a tuple"):
            SessionConfig(session_id="test", frame_size="1920x1080")  # type: ignore[arg-type]

    def test_session_config_rejects_frame_size_single_element(self) -> None:
        """Test that single-element frame_size is rejected."""
        with pytest.raises(ValidationError, match="exactly 2 elements"):
            SessionConfig(session_id="test", frame_size=(1920,))  # type: ignore[arg-type]

    def test_session_config_rejects_frame_size_three_elements(self) -> None:
        """Test that three-element frame_size is rejected."""
        with pytest.raises(ValidationError, match="exactly 2 elements"):
            SessionConfig(session_id="test", frame_size=(1920, 1080, 3))  # type: ignore[arg-type]

    def test_session_config_rejects_non_integer_width(self) -> None:
        """Test that non-integer width is rejected."""
        with pytest.raises(ValidationError, match="width must be an integer"):
            SessionConfig(session_id="test", frame_size=(1920.5, 1080))  # type: ignore[arg-type]

    def test_session_config_rejects_non_integer_height(self) -> None:
        """Test that non-integer height is rejected."""
        with pytest.raises(ValidationError, match="height must be an integer"):
            SessionConfig(session_id="test", frame_size=(1920, 1080.5))  # type: ignore[arg-type]

    def test_session_config_rejects_zero_width(self) -> None:
        """Test that zero width is rejected."""
        with pytest.raises(ValidationError, match="width must be positive"):
            SessionConfig(session_id="test", frame_size=(0, 1080))

    def test_session_config_rejects_zero_height(self) -> None:
        """Test that zero height is rejected."""
        with pytest.raises(ValidationError, match="height must be positive"):
            SessionConfig(session_id="test", frame_size=(1920, 0))

    def test_session_config_rejects_negative_width(self) -> None:
        """Test that negative width is rejected."""
        with pytest.raises(ValidationError, match="width must be positive"):
            SessionConfig(session_id="test", frame_size=(-1920, 1080))

    def test_session_config_rejects_negative_height(self) -> None:
        """Test that negative height is rejected."""
        with pytest.raises(ValidationError, match="height must be positive"):
            SessionConfig(session_id="test", frame_size=(1920, -1080))

    def test_session_config_rejects_nan_width(self) -> None:
        """Test that NaN width is rejected with ValidationError."""
        with pytest.raises(ValidationError, match="width must be finite"):
            SessionConfig(session_id="test", frame_size=(float("nan"), 1080))  # type: ignore[arg-type]

    def test_session_config_rejects_nan_height(self) -> None:
        """Test that NaN height is rejected with ValidationError."""
        with pytest.raises(ValidationError, match="height must be finite"):
            SessionConfig(session_id="test", frame_size=(1920, float("nan")))  # type: ignore[arg-type]

    def test_session_config_rejects_inf_width(self) -> None:
        """Test that infinite width is rejected with ValidationError."""
        with pytest.raises(ValidationError, match="width must be finite"):
            SessionConfig(session_id="test", frame_size=(float("inf"), 1080))  # type: ignore[arg-type]

    def test_session_config_rejects_inf_height(self) -> None:
        """Test that infinite height is rejected with ValidationError."""
        with pytest.raises(ValidationError, match="height must be finite"):
            SessionConfig(session_id="test", frame_size=(1920, float("inf")))  # type: ignore[arg-type]

    def test_session_config_rejects_negative_inf_width(self) -> None:
        """Test that negative infinite width is rejected with ValidationError."""
        with pytest.raises(ValidationError, match="width must be finite"):
            SessionConfig(session_id="test", frame_size=(float("-inf"), 1080))  # type: ignore[arg-type]

    def test_session_config_rejects_negative_inf_height(self) -> None:
        """Test that negative infinite height is rejected with ValidationError."""
        with pytest.raises(ValidationError, match="height must be finite"):
            SessionConfig(session_id="test", frame_size=(1920, float("-inf")))  # type: ignore[arg-type]

    def test_session_config_rejects_zero_sample_interval(self) -> None:
        """Test that zero sample_interval_seconds is rejected."""
        with pytest.raises(ValidationError, match="sample_interval_seconds must be positive"):
            SessionConfig(session_id="test", sample_interval_seconds=0.0)

    def test_session_config_rejects_negative_sample_interval(self) -> None:
        """Test that negative sample_interval_seconds is rejected."""
        with pytest.raises(ValidationError, match="sample_interval_seconds must be positive"):
            SessionConfig(session_id="test", sample_interval_seconds=-1.0)

    def test_session_config_rejects_nan_sample_interval(self) -> None:
        """Test that NaN sample_interval_seconds is rejected."""
        with pytest.raises(ValidationError, match="sample_interval_seconds must be finite"):
            SessionConfig(session_id="test", sample_interval_seconds=float("nan"))

    def test_session_config_rejects_inf_sample_interval(self) -> None:
        """Test that infinite sample_interval_seconds is rejected."""
        with pytest.raises(ValidationError, match="sample_interval_seconds must be finite"):
            SessionConfig(session_id="test", sample_interval_seconds=float("inf"))

    def test_session_config_rejects_non_string_viewer_id(self) -> None:
        """Test that non-string viewer_id is rejected."""
        with pytest.raises(ValidationError, match="viewer_id must be a string or None"):
            SessionConfig(session_id="test", viewer_id=123)  # type: ignore[arg-type]

    def test_session_config_rejects_non_dict_notes(self) -> None:
        """Test that non-dict notes is rejected."""
        with pytest.raises(ValidationError, match="notes must be a dict or None"):
            SessionConfig(session_id="test", notes="not a dict")  # type: ignore[arg-type]


# =============================================================================
# Tests: SessionConfig Accepts Valid Types
# =============================================================================


class TestSessionConfigAcceptsValidTypes:
    """Tests for SessionConfig acceptance of various valid input types."""

    def test_session_config_accepts_integer_frame_size(self) -> None:
        """Test that integer frame_size is accepted."""
        config = SessionConfig(session_id="test", frame_size=(1920, 1080))
        assert config.frame_size == (1920, 1080)

    def test_session_config_accepts_whole_number_float_frame_size(self) -> None:
        """Test that whole-number floats are accepted and normalized to ints."""
        config = SessionConfig(session_id="test", frame_size=(1920.0, 1080.0))  # type: ignore[arg-type]
        # Should be normalized to tuple of ints
        assert config.frame_size == (1920, 1080)
        assert isinstance(config.frame_size[0], int)  # type: ignore[index]
        assert isinstance(config.frame_size[1], int)  # type: ignore[index]

    def test_session_config_accepts_numpy_int_frame_size(self) -> None:
        """Test that numpy integers are accepted and normalized to Python ints."""
        config = SessionConfig(
            session_id="test",
            frame_size=(np.int32(1920), np.int64(1080)),  # type: ignore[arg-type]
        )
        assert config.width == 1920
        assert config.height == 1080
        # Should be normalized to Python ints in a tuple
        assert isinstance(config.frame_size, tuple)
        assert isinstance(config.frame_size[0], int)
        assert isinstance(config.frame_size[1], int)

    def test_session_config_accepts_list_frame_size(self) -> None:
        """Test that list frame_size is accepted and normalized to tuple."""
        original_list = [1920, 1080]
        config = SessionConfig(session_id="test", frame_size=original_list)  # type: ignore[arg-type]
        # Should be normalized to a tuple, not the original list reference
        assert config.frame_size == (1920, 1080)
        assert isinstance(config.frame_size, tuple)
        # Mutating the original list should not affect the config
        original_list[0] = 9999
        assert config.frame_size == (1920, 1080)

    def test_session_config_accepts_float_sample_interval(self) -> None:
        """Test that float sample_interval is accepted."""
        config = SessionConfig(session_id="test", sample_interval_seconds=0.5)
        assert config.sample_interval_seconds == 0.5

    def test_session_config_accepts_integer_sample_interval(self) -> None:
        """Test that integer sample_interval is accepted."""
        config = SessionConfig(session_id="test", sample_interval_seconds=2)
        assert config.sample_interval_seconds == 2


# =============================================================================
# Tests: validate_viewer_samples Respects Frame Size
# =============================================================================


class TestValidateSamplesRespectsFrameSize:
    """Tests verifying validate_viewer_samples respects SessionConfig frame_size.

    Step 1.3: Tests that viewer sample validation integrates correctly with
    frame_size bounds checking from SessionConfig.
    """

    def test_validate_samples_respects_frame_size_from_config(self) -> None:
        """Test that validation respects frame_size for bounds checking."""
        config = SessionConfig(session_id="test", frame_size=(800, 600))

        # Sample within bounds should pass
        valid_sample = ViewerSample(position=(400.0, 300.0), direction=(1.0, 0.0))
        validate_viewer_samples([valid_sample], frame_size=config.frame_size)

        # Sample outside bounds should fail
        invalid_sample = ViewerSample(position=(900.0, 300.0), direction=(1.0, 0.0))
        with pytest.raises(ValidationError, match="x position.*out of bounds"):
            validate_viewer_samples([invalid_sample], frame_size=config.frame_size)

    def test_validate_samples_uses_config_dimensions(self) -> None:
        """Test that validation uses the exact dimensions from config."""
        config = SessionConfig(session_id="test", frame_size=(1920, 1080))

        # Position at (1919, 1079) should be valid (just inside bounds)
        edge_sample = ViewerSample(position=(1919.0, 1079.0), direction=(1.0, 0.0))
        validate_viewer_samples([edge_sample], frame_size=config.frame_size)

        # Position at (1920, 1080) should be invalid (at boundary)
        boundary_sample = ViewerSample(position=(1920.0, 1080.0), direction=(1.0, 0.0))
        with pytest.raises(ValidationError):
            validate_viewer_samples([boundary_sample], frame_size=config.frame_size)

    def test_validate_samples_no_bounds_check_without_config_frame_size(self) -> None:
        """Test that no bounds checking when config has no frame_size."""
        config = SessionConfig(session_id="test")  # No frame_size

        # Any position should pass without bounds checking
        sample = ViewerSample(position=(10000.0, 10000.0), direction=(1.0, 0.0))
        validate_viewer_samples([sample], frame_size=config.frame_size)  # Should not raise

    def test_validate_samples_batch_respects_frame_size(self) -> None:
        """Test that entire batch is validated against frame_size."""
        config = SessionConfig(session_id="test", frame_size=(640, 480))

        samples = [
            ViewerSample(position=(100.0, 100.0), direction=(1.0, 0.0)),
            ViewerSample(position=(320.0, 240.0), direction=(0.0, 1.0)),
            ViewerSample(position=(639.0, 479.0), direction=(-1.0, 0.0)),
        ]

        validate_viewer_samples(samples, frame_size=config.frame_size)  # Should not raise

        # Add one invalid sample
        samples.append(ViewerSample(position=(700.0, 200.0), direction=(1.0, 0.0)))

        with pytest.raises(ValidationError, match="index 3.*x position.*out of bounds"):
            validate_viewer_samples(samples, frame_size=config.frame_size)
