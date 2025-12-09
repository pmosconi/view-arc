"""
Tests for the tracking module data structures (Step 1.1).

Tests cover:
- ViewerSample creation and validation
- AOI creation and validation
- AOIResult tracking functionality
- TrackingResult accessors and aggregation
"""

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from view_arc.tracking import (
    AOI,
    AOIResult,
    TrackingResult,
    SessionConfig,
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


def make_square_contour(center: tuple[float, float], half_size: float = 15.0) -> NDArray[np.float64]:
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


def make_triangle_contour(center: tuple[float, float], size: float = 20.0) -> NDArray[np.float64]:
    """Create a triangle contour centered at the given point."""
    cx, cy = center
    return np.array(
        [
            [cx, cy + size],
            [cx - size, cy - size],
            [cx + size, cy - size],
        ],
        dtype=np.float64,
    )


# =============================================================================
# Tests: ViewerSample
# =============================================================================


class TestViewerSampleCreation:
    """Tests for ViewerSample dataclass creation and validation."""

    def test_viewer_sample_creation_basic(self) -> None:
        """Test creating a valid ViewerSample with minimal parameters."""
        sample = ViewerSample(
            position=(100.0, 200.0),
            direction=(1.0, 0.0),  # Looking right
        )

        assert sample.position == (100.0, 200.0)
        assert sample.direction == (1.0, 0.0)
        assert sample.timestamp is None

    def test_viewer_sample_creation_with_timestamp(self) -> None:
        """Test creating a ViewerSample with timestamp."""
        sample = ViewerSample(
            position=(50.0, 75.0),
            direction=(0.0, 1.0),  # Looking up
            timestamp=10.5,
        )

        assert sample.position == (50.0, 75.0)
        assert sample.direction == (0.0, 1.0)
        assert sample.timestamp == 10.5

    def test_viewer_sample_diagonal_direction(self) -> None:
        """Test creating a ViewerSample with diagonal unit vector direction."""
        sqrt2_inv = 1.0 / math.sqrt(2.0)
        sample = ViewerSample(
            position=(0.0, 0.0),
            direction=(sqrt2_inv, sqrt2_inv),  # 45 degrees
        )

        assert sample.direction == pytest.approx((sqrt2_inv, sqrt2_inv), rel=1e-6)

    def test_viewer_sample_various_angles(self) -> None:
        """Test ViewerSample with various unit vector directions."""
        for angle_deg in [0, 45, 90, 135, 180, 225, 270, 315]:
            direction = make_unit_vector(angle_deg)
            sample = ViewerSample(position=(0.0, 0.0), direction=direction)

            # Verify it's a unit vector
            mag = math.sqrt(sample.direction[0] ** 2 + sample.direction[1] ** 2)
            assert mag == pytest.approx(1.0, rel=1e-6)

    def test_viewer_sample_position_array(self) -> None:
        """Test position_array property returns correct numpy array."""
        sample = ViewerSample(position=(100.0, 200.0), direction=(1.0, 0.0))
        pos_array = sample.position_array

        assert isinstance(pos_array, np.ndarray)
        assert pos_array.dtype == np.float64
        assert np.array_equal(pos_array, np.array([100.0, 200.0]))

    def test_viewer_sample_direction_array(self) -> None:
        """Test direction_array property returns correct numpy array."""
        sample = ViewerSample(position=(0.0, 0.0), direction=(0.0, -1.0))
        dir_array = sample.direction_array

        assert isinstance(dir_array, np.ndarray)
        assert dir_array.dtype == np.float64
        assert np.array_equal(dir_array, np.array([0.0, -1.0]))

    def test_viewer_sample_immutable(self) -> None:
        """Test that ViewerSample is frozen (immutable)."""
        sample = ViewerSample(position=(0.0, 0.0), direction=(1.0, 0.0))

        with pytest.raises(AttributeError):
            sample.position = (10.0, 10.0)  # type: ignore[misc]

        with pytest.raises(AttributeError):
            sample.direction = (0.0, 1.0)  # type: ignore[misc]


class TestViewerSampleInvalidDirection:
    """Tests for ViewerSample direction validation."""

    def test_viewer_sample_invalid_direction_zero_vector(self) -> None:
        """Test that zero vector direction is rejected."""
        with pytest.raises(ValidationError, match="unit vector"):
            ViewerSample(position=(0.0, 0.0), direction=(0.0, 0.0))

    def test_viewer_sample_invalid_direction_non_unit(self) -> None:
        """Test that non-unit vector direction is rejected."""
        with pytest.raises(ValidationError, match="unit vector"):
            ViewerSample(position=(0.0, 0.0), direction=(2.0, 0.0))

    def test_viewer_sample_invalid_direction_too_short(self) -> None:
        """Test that vector with magnitude < 1 is rejected."""
        with pytest.raises(ValidationError, match="unit vector"):
            ViewerSample(position=(0.0, 0.0), direction=(0.5, 0.0))

    def test_viewer_sample_invalid_direction_arbitrary(self) -> None:
        """Test rejection of arbitrary non-unit vector."""
        with pytest.raises(ValidationError, match="unit vector"):
            ViewerSample(position=(100.0, 100.0), direction=(3.0, 4.0))

    def test_viewer_sample_near_unit_vector_accepted(self) -> None:
        """Test that vectors very close to unit length are accepted."""
        # 1.0 + 1e-7 should be within tolerance
        sample = ViewerSample(position=(0.0, 0.0), direction=(1.0 + 1e-7, 0.0))
        assert sample.direction[0] == pytest.approx(1.0, rel=1e-5)

    def test_viewer_sample_invalid_direction_3d_vector(self) -> None:
        """Test that 3D vector direction is rejected."""
        with pytest.raises(ValidationError, match="2 components"):
            ViewerSample(position=(0.0, 0.0), direction=(0.0, 0.0, 1.0))  # type: ignore[arg-type]

    def test_viewer_sample_invalid_direction_1d_vector(self) -> None:
        """Test that 1D vector direction is rejected."""
        with pytest.raises(ValidationError, match="2 components"):
            ViewerSample(position=(0.0, 0.0), direction=(1.0,))  # type: ignore[arg-type]

    def test_viewer_sample_invalid_direction_4d_vector(self) -> None:
        """Test that 4D vector direction is rejected."""
        with pytest.raises(ValidationError, match="2 components"):
            ViewerSample(position=(0.0, 0.0), direction=(0.5, 0.5, 0.5, 0.5))  # type: ignore[arg-type]


# =============================================================================
# Tests: AOI (Area of Interest)
# =============================================================================


class TestAOICreation:
    """Tests for AOI dataclass creation and validation."""

    def test_aoi_creation_with_string_id(self) -> None:
        """Test creating an AOI with string ID."""
        contour = make_square_contour((100.0, 100.0))
        aoi = AOI(id="shelf_1", contour=contour)

        assert aoi.id == "shelf_1"
        assert np.array_equal(aoi.contour, contour)
        assert aoi.num_vertices == 4

    def test_aoi_creation_with_int_id(self) -> None:
        """Test creating an AOI with integer ID."""
        contour = make_triangle_contour((50.0, 50.0))
        aoi = AOI(id=42, contour=contour)

        assert aoi.id == 42
        assert aoi.num_vertices == 3

    def test_aoi_creation_large_polygon(self) -> None:
        """Test creating an AOI with many vertices."""
        # Create a polygon with 10 vertices (decagon-like)
        angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
        contour = np.column_stack([np.cos(angles) * 50, np.sin(angles) * 50])
        aoi = AOI(id="complex_shape", contour=contour)

        assert aoi.num_vertices == 10

    def test_aoi_num_vertices_property(self) -> None:
        """Test num_vertices property returns correct count."""
        contour = make_square_contour((0.0, 0.0))
        aoi = AOI(id=1, contour=contour)

        assert aoi.num_vertices == 4

    def test_aoi_hash_by_id(self) -> None:
        """Test that AOI hash is based on ID."""
        contour1 = make_square_contour((0.0, 0.0))
        contour2 = make_square_contour((100.0, 100.0))

        aoi1 = AOI(id="same_id", contour=contour1)
        aoi2 = AOI(id="same_id", contour=contour2)

        # Same ID means same hash (even with different contours)
        assert hash(aoi1) == hash(aoi2)

    def test_aoi_equality(self) -> None:
        """Test AOI equality comparison."""
        contour = make_square_contour((50.0, 50.0))
        aoi1 = AOI(id="test", contour=contour.copy())
        aoi2 = AOI(id="test", contour=contour.copy())

        assert aoi1 == aoi2

    def test_aoi_inequality_different_id(self) -> None:
        """Test AOI inequality with different IDs."""
        contour = make_square_contour((50.0, 50.0))
        aoi1 = AOI(id="a", contour=contour.copy())
        aoi2 = AOI(id="b", contour=contour.copy())

        assert aoi1 != aoi2


class TestAOIInvalidContour:
    """Tests for AOI contour validation."""

    def test_aoi_invalid_contour_not_array(self) -> None:
        """Test that non-array contour is rejected."""
        with pytest.raises(ValidationError, match="numpy array"):
            AOI(id="bad", contour=[[0, 0], [1, 0], [1, 1]])  # type: ignore[arg-type]

    def test_aoi_invalid_contour_1d(self) -> None:
        """Test that 1D array is rejected."""
        with pytest.raises(ValidationError, match="2D array"):
            AOI(id="bad", contour=np.array([0, 1, 2, 3]))

    def test_aoi_invalid_contour_wrong_columns(self) -> None:
        """Test that array with wrong number of columns is rejected."""
        with pytest.raises(ValidationError, match="shape"):
            AOI(id="bad", contour=np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]]))

    def test_aoi_invalid_contour_too_few_vertices(self) -> None:
        """Test that contour with < 3 vertices is rejected."""
        with pytest.raises(ValidationError, match="at least 3 vertices"):
            AOI(id="bad", contour=np.array([[0, 0], [1, 1]]))

    def test_aoi_invalid_contour_single_point(self) -> None:
        """Test that single point is rejected."""
        with pytest.raises(ValidationError, match="at least 3 vertices"):
            AOI(id="bad", contour=np.array([[0, 0]]))

    def test_aoi_invalid_contour_empty(self) -> None:
        """Test that empty contour is rejected."""
        with pytest.raises(ValidationError, match="at least 3 vertices"):
            AOI(id="bad", contour=np.array([]).reshape(0, 2))


# =============================================================================
# Tests: AOIResult
# =============================================================================


class TestAOIResult:
    """Tests for AOIResult dataclass."""

    def test_aoi_result_creation_defaults(self) -> None:
        """Test creating AOIResult with default values."""
        result = AOIResult(aoi_id="shelf_1")

        assert result.aoi_id == "shelf_1"
        assert result.hit_count == 0
        assert result.total_attention_seconds == 0.0
        assert result.hit_timestamps == []

    def test_aoi_result_creation_with_values(self) -> None:
        """Test creating AOIResult with explicit values."""
        result = AOIResult(
            aoi_id=42,
            hit_count=5,
            total_attention_seconds=5.0,
            hit_timestamps=[0, 2, 4, 6, 8],
        )

        assert result.aoi_id == 42
        assert result.hit_count == 5
        assert result.total_attention_seconds == 5.0
        assert result.hit_timestamps == [0, 2, 4, 6, 8]

    def test_aoi_result_add_hit(self) -> None:
        """Test adding a hit to AOIResult."""
        result = AOIResult(aoi_id="test")

        result.add_hit(sample_index=5)

        assert result.hit_count == 1
        assert result.total_attention_seconds == 1.0
        assert result.hit_timestamps == [5]

    def test_aoi_result_add_multiple_hits(self) -> None:
        """Test adding multiple hits to AOIResult."""
        result = AOIResult(aoi_id="test")

        result.add_hit(0)
        result.add_hit(2)
        result.add_hit(5)

        assert result.hit_count == 3
        assert result.total_attention_seconds == 3.0
        assert result.hit_timestamps == [0, 2, 5]

    def test_aoi_result_add_hit_custom_interval(self) -> None:
        """Test adding hits with custom sample interval."""
        result = AOIResult(aoi_id="test")

        result.add_hit(0, sample_interval=0.5)
        result.add_hit(1, sample_interval=0.5)

        assert result.hit_count == 2
        assert result.total_attention_seconds == 1.0

    def test_aoi_result_hit_timestamps_mutable(self) -> None:
        """Test that hit_timestamps is properly converted to list."""
        # Even if passed as tuple, should become list
        result = AOIResult(aoi_id="test", hit_timestamps=(1, 2, 3))  # type: ignore[arg-type]

        assert isinstance(result.hit_timestamps, list)
        assert result.hit_timestamps == [1, 2, 3]


# =============================================================================
# Tests: TrackingResult
# =============================================================================


class TestTrackingResultAccessors:
    """Tests for TrackingResult data access methods."""

    def test_tracking_result_creation(self) -> None:
        """Test creating a TrackingResult."""
        aoi_results: dict[str | int, AOIResult] = {
            "a": AOIResult(aoi_id="a", hit_count=10, total_attention_seconds=10.0),
            "b": AOIResult(aoi_id="b", hit_count=5, total_attention_seconds=5.0),
        }
        result = TrackingResult(
            aoi_results=aoi_results,
            total_samples=20,
            samples_with_hits=15,
            samples_no_winner=5,
        )

        assert result.total_samples == 20
        assert result.samples_with_hits == 15
        assert result.samples_no_winner == 5

    def test_tracking_result_get_aoi_result(self) -> None:
        """Test get_aoi_result returns correct AOIResult."""
        aoi_result = AOIResult(aoi_id="shelf_1", hit_count=7)
        result = TrackingResult(
            aoi_results={"shelf_1": aoi_result},
            total_samples=10,
            samples_with_hits=7,
            samples_no_winner=3,
        )

        retrieved = result.get_aoi_result("shelf_1")
        assert retrieved is not None
        assert retrieved.hit_count == 7

    def test_tracking_result_get_aoi_result_not_found(self) -> None:
        """Test get_aoi_result returns None for missing AOI."""
        result = TrackingResult(
            aoi_results={},
            total_samples=10,
            samples_with_hits=0,
            samples_no_winner=10,
        )

        assert result.get_aoi_result("nonexistent") is None

    def test_tracking_result_get_hit_count(self) -> None:
        """Test get_hit_count returns correct count."""
        result = TrackingResult(
            aoi_results={
                "a": AOIResult(aoi_id="a", hit_count=5),
                "b": AOIResult(aoi_id="b", hit_count=3),
            },
            total_samples=10,
            samples_with_hits=8,
            samples_no_winner=2,
        )

        assert result.get_hit_count("a") == 5
        assert result.get_hit_count("b") == 3
        assert result.get_hit_count("c") == 0  # Not found

    def test_tracking_result_get_total_hits(self) -> None:
        """Test get_total_hits sums all AOI hits."""
        result = TrackingResult(
            aoi_results={
                "a": AOIResult(aoi_id="a", hit_count=10),
                "b": AOIResult(aoi_id="b", hit_count=5),
                "c": AOIResult(aoi_id="c", hit_count=0),
            },
            total_samples=20,
            samples_with_hits=15,
            samples_no_winner=5,
        )

        assert result.get_total_hits() == 15

    def test_tracking_result_get_attention_seconds(self) -> None:
        """Test get_attention_seconds returns correct value."""
        result = TrackingResult(
            aoi_results={
                "a": AOIResult(aoi_id="a", total_attention_seconds=10.5),
            },
            total_samples=15,
            samples_with_hits=10,
            samples_no_winner=5,
        )

        assert result.get_attention_seconds("a") == 10.5
        assert result.get_attention_seconds("b") == 0.0  # Not found

    def test_tracking_result_coverage_ratio(self) -> None:
        """Test coverage_ratio property."""
        result = TrackingResult(
            aoi_results={},
            total_samples=100,
            samples_with_hits=75,
            samples_no_winner=25,
        )

        assert result.coverage_ratio == 0.75

    def test_tracking_result_coverage_ratio_empty(self) -> None:
        """Test coverage_ratio with zero samples."""
        result = TrackingResult(
            aoi_results={},
            total_samples=0,
            samples_with_hits=0,
            samples_no_winner=0,
        )

        assert result.coverage_ratio == 0.0

    def test_tracking_result_aoi_ids(self) -> None:
        """Test aoi_ids property returns all IDs."""
        result = TrackingResult(
            aoi_results={
                "shelf_1": AOIResult(aoi_id="shelf_1"),
                "shelf_2": AOIResult(aoi_id="shelf_2"),
                42: AOIResult(aoi_id=42),
            },
            total_samples=10,
            samples_with_hits=5,
            samples_no_winner=5,
        )

        ids = result.aoi_ids
        assert len(ids) == 3
        assert "shelf_1" in ids
        assert "shelf_2" in ids
        assert 42 in ids

    def test_tracking_result_with_integer_and_string_ids(self) -> None:
        """Test TrackingResult handles mixed ID types correctly."""
        result = TrackingResult(
            aoi_results={
                1: AOIResult(aoi_id=1, hit_count=5),
                "two": AOIResult(aoi_id="two", hit_count=3),
            },
            total_samples=10,
            samples_with_hits=8,
            samples_no_winner=2,
        )

        assert result.get_hit_count(1) == 5
        assert result.get_hit_count("two") == 3


class TestTrackingResultValidation:
    """Tests for TrackingResult input validation."""

    def test_tracking_result_rejects_negative_total_samples(self) -> None:
        """Test that negative total_samples is rejected."""
        with pytest.raises(ValidationError, match="total_samples must be non-negative"):
            TrackingResult(
                aoi_results={},
                total_samples=-1,
                samples_with_hits=0,
                samples_no_winner=0,
            )

    def test_tracking_result_rejects_negative_samples_with_hits(self) -> None:
        """Test that negative samples_with_hits is rejected."""
        with pytest.raises(ValidationError, match="samples_with_hits must be non-negative"):
            TrackingResult(
                aoi_results={},
                total_samples=10,
                samples_with_hits=-1,
                samples_no_winner=11,
            )

    def test_tracking_result_rejects_negative_samples_no_winner(self) -> None:
        """Test that negative samples_no_winner is rejected."""
        with pytest.raises(ValidationError, match="samples_no_winner must be non-negative"):
            TrackingResult(
                aoi_results={},
                total_samples=10,
                samples_with_hits=5,
                samples_no_winner=-1,
            )

    def test_tracking_result_rejects_samples_with_hits_exceeds_total(self) -> None:
        """Test that samples_with_hits > total_samples is rejected."""
        with pytest.raises(ValidationError, match="samples_with_hits.*cannot exceed.*total_samples"):
            TrackingResult(
                aoi_results={},
                total_samples=10,
                samples_with_hits=15,
                samples_no_winner=0,
            )

    def test_tracking_result_rejects_samples_no_winner_exceeds_total(self) -> None:
        """Test that samples_no_winner > total_samples is rejected."""
        with pytest.raises(ValidationError, match="samples_no_winner.*cannot exceed.*total_samples"):
            TrackingResult(
                aoi_results={},
                total_samples=10,
                samples_with_hits=0,
                samples_no_winner=15,
            )

    def test_tracking_result_rejects_inconsistent_totals(self) -> None:
        """Test that samples_with_hits + samples_no_winner != total_samples is rejected."""
        with pytest.raises(ValidationError, match="must equal total_samples"):
            TrackingResult(
                aoi_results={},
                total_samples=100,
                samples_with_hits=60,
                samples_no_winner=30,  # Should be 40
            )

    def test_tracking_result_accepts_valid_zero_samples(self) -> None:
        """Test that zero samples is valid when all counts are zero."""
        result = TrackingResult(
            aoi_results={},
            total_samples=0,
            samples_with_hits=0,
            samples_no_winner=0,
        )
        assert result.total_samples == 0

    def test_tracking_result_accepts_all_hits(self) -> None:
        """Test that 100% hit rate is valid."""
        result = TrackingResult(
            aoi_results={"a": AOIResult(aoi_id="a", hit_count=10)},
            total_samples=10,
            samples_with_hits=10,
            samples_no_winner=0,
        )
        assert result.coverage_ratio == 1.0

    def test_tracking_result_accepts_no_hits(self) -> None:
        """Test that 0% hit rate is valid."""
        result = TrackingResult(
            aoi_results={},
            total_samples=10,
            samples_with_hits=0,
            samples_no_winner=10,
        )
        assert result.coverage_ratio == 0.0


# =============================================================================
# Tests: validate_viewer_samples (Step 1.2)
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


# =============================================================================
# Tests: validate_aois (Step 1.2)
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
# Tests: validate_tracking_params (Step 1.2)
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


# =============================================================================
# Tests: validate_viewer_samples frame_size validation (Step 1.2 Review)
# =============================================================================


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


# =============================================================================
# Tests: validate_tracking_params non-finite rejection (Step 1.2 Review)
# =============================================================================


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
            validate_tracking_params(fov_deg=90.0, max_range=100.0, sample_interval=float("nan"))

    def test_validate_params_sample_interval_inf(self) -> None:
        """Test rejection of infinite sample_interval."""
        with pytest.raises(ValidationError, match="sample_interval must be finite"):
            validate_tracking_params(fov_deg=90.0, max_range=100.0, sample_interval=float("inf"))

    def test_validate_params_sample_interval_negative_inf(self) -> None:
        """Test rejection of negative infinite sample_interval."""
        with pytest.raises(ValidationError, match="sample_interval must be finite"):
            validate_tracking_params(fov_deg=90.0, max_range=100.0, sample_interval=float("-inf"))


# =============================================================================
# Tests: validate_tracking_params numpy scalar acceptance (Step 1.2 Review)
# =============================================================================


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
# Tests: SessionConfig (Step 1.3)
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
        assert config.notes["experiment_id"] == 123 # type: ignore
        assert config.notes["conditions"] == ["baseline", "treatment"] # type: ignore


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


class TestSessionConfigAcceptsValidTypes:
    """Tests for SessionConfig acceptance of various valid input types."""

    def test_session_config_accepts_integer_frame_size(self) -> None:
        """Test that integer frame_size is accepted."""
        config = SessionConfig(session_id="test", frame_size=(1920, 1080))
        assert config.frame_size == (1920, 1080)

    def test_session_config_accepts_whole_number_float_frame_size(self) -> None:
        """Test that whole-number floats are accepted for frame_size."""
        config = SessionConfig(session_id="test", frame_size=(1920.0, 1080.0))  # type: ignore[arg-type]
        assert config.frame_size == (1920.0, 1080.0)

    def test_session_config_accepts_numpy_int_frame_size(self) -> None:
        """Test that numpy integers are accepted for frame_size."""
        config = SessionConfig(
            session_id="test",
            frame_size=(np.int32(1920), np.int64(1080)),  # type: ignore[arg-type]
        )
        assert config.width == 1920
        assert config.height == 1080

    def test_session_config_accepts_list_frame_size(self) -> None:
        """Test that list frame_size is accepted."""
        config = SessionConfig(session_id="test", frame_size=[1920, 1080])  # type: ignore[arg-type]
        assert config.frame_size == [1920, 1080]

    def test_session_config_accepts_float_sample_interval(self) -> None:
        """Test that float sample_interval is accepted."""
        config = SessionConfig(session_id="test", sample_interval_seconds=0.5)
        assert config.sample_interval_seconds == 0.5

    def test_session_config_accepts_integer_sample_interval(self) -> None:
        """Test that integer sample_interval is accepted."""
        config = SessionConfig(session_id="test", sample_interval_seconds=2)
        assert config.sample_interval_seconds == 2


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
