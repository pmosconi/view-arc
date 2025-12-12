"""
Tests for tracking module data structures (Step 1.1).

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
    ValidationError,
    ViewerSample,
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


def make_triangle_contour(
    center: tuple[float, float], size: float = 20.0
) -> NDArray[np.float64]:
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
            "a": AOIResult(aoi_id="a", hit_count=10, total_attention_seconds=10.0, hit_timestamps=list(range(10))),
            "b": AOIResult(aoi_id="b", hit_count=5, total_attention_seconds=5.0, hit_timestamps=list(range(10, 15))),
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
        aoi_result = AOIResult(aoi_id="shelf_1", hit_count=7, hit_timestamps=list(range(7)))
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
                "a": AOIResult(aoi_id="a", hit_count=5, hit_timestamps=list(range(5))),
                "b": AOIResult(aoi_id="b", hit_count=3, hit_timestamps=list(range(5, 8))),
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
                "a": AOIResult(aoi_id="a", hit_count=10, hit_timestamps=list(range(10))),
                "b": AOIResult(aoi_id="b", hit_count=5, hit_timestamps=list(range(10, 15))),
                "c": AOIResult(aoi_id="c", hit_count=0, hit_timestamps=[]),
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
                "a": AOIResult(aoi_id="a", hit_count=10, total_attention_seconds=10.5, hit_timestamps=list(range(10))),
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
            aoi_results={
                "a": AOIResult(aoi_id="a", hit_count=75, hit_timestamps=list(range(75))),
            },
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
                "shelf_1": AOIResult(aoi_id="shelf_1", hit_count=2, hit_timestamps=[0, 1]),
                "shelf_2": AOIResult(aoi_id="shelf_2", hit_count=2, hit_timestamps=[2, 3]),
                42: AOIResult(aoi_id=42, hit_count=1, hit_timestamps=[4]),
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
                1: AOIResult(aoi_id=1, hit_count=5, hit_timestamps=list(range(5))),
                "two": AOIResult(aoi_id="two", hit_count=3, hit_timestamps=list(range(5, 8))),
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
        with pytest.raises(
            ValidationError, match="samples_with_hits.*cannot exceed.*total_samples"
        ):
            TrackingResult(
                aoi_results={},
                total_samples=10,
                samples_with_hits=15,
                samples_no_winner=0,
            )

    def test_tracking_result_rejects_samples_no_winner_exceeds_total(self) -> None:
        """Test that samples_no_winner > total_samples is rejected."""
        with pytest.raises(
            ValidationError, match="samples_no_winner.*cannot exceed.*total_samples"
        ):
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
            aoi_results={"a": AOIResult(aoi_id="a", hit_count=10, hit_timestamps=list(range(10)))},
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
