"""
Temporal Attention Tracking Module
==================================

Data structures and algorithms for accumulating "attention seconds" across
multiple viewer positions and view directions over a batched acquisition period.

This module extends the view arc obstacle detection system to track which
Area of Interest (AOI) a viewer is looking at over time, counting the
total seconds of attention each AOI receives.

Assumptions:
- Samples arrive at a fixed 1 Hz cadence (one sample per second)
- Each sample represents exactly 1 second of viewing time
- Timestamps, when provided, are already sorted upstream
- AOI contours remain fixed in image coordinate space
- Each batch tracks a single viewer
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from view_arc.clipping import is_valid_polygon
from view_arc.geometry import validate_and_get_direction_angle


class ValidationError(Exception):
    """Raised when input validation fails."""

    pass


def _validate_direction(direction: tuple[float, float]) -> None:
    """Validate that direction is a unit vector with exactly 2 components.

    Args:
        direction: A 2D direction vector (dx, dy)

    Raises:
        ValidationError: If direction does not have exactly 2 components
        ValidationError: If direction is not a unit vector
    """
    # Check that direction has exactly 2 components
    if len(direction) != 2:
        raise ValidationError(
            f"Direction must have exactly 2 components (dx, dy), got {len(direction)} components"
        )
    try:
        validate_and_get_direction_angle(np.array(direction, dtype=np.float32))
    except ValueError as e:
        raise ValidationError(str(e)) from e


def _validate_contour(contour: NDArray[np.floating[Any]]) -> None:
    """Validate that contour is a valid polygon.

    Args:
        contour: Polygon vertices as numpy array

    Raises:
        ValidationError: If contour is malformed
    """
    if not isinstance(contour, np.ndarray):
        raise ValidationError(f"Contour must be a numpy array, got {type(contour).__name__}")

    if contour.ndim != 2:
        raise ValidationError(f"Contour must be 2D array, got shape {contour.shape}")

    if contour.shape[1] != 2:
        raise ValidationError(
            f"Contour must have shape (N, 2) for N vertices, got shape {contour.shape}"
        )

    if not is_valid_polygon(contour.astype(np.float32)):
        raise ValidationError(
            f"Contour must have at least 3 vertices to form a polygon, got {contour.shape[0]}"
        )


@dataclass(frozen=True)
class ViewerSample:
    """A single observation of viewer position and view direction.

    Represents one sample in a temporal sequence of viewer observations,
    typically captured at 1 Hz (one sample per second).

    Attributes:
        position: The (x, y) position of the viewer in image coordinates
        direction: A unit vector (dx, dy) indicating the view direction
        timestamp: Optional timestamp for ordering validation; if provided,
                   represents seconds from session start

    Raises:
        ValidationError: If direction is not a unit vector
    """

    position: tuple[float, float]
    direction: tuple[float, float]
    timestamp: float | None = None

    def __post_init__(self) -> None:
        """Validate that direction is a unit vector."""
        _validate_direction(self.direction)

    @property
    def position_array(self) -> NDArray[np.float64]:
        """Return position as a numpy array."""
        return np.array(self.position, dtype=np.float64)

    @property
    def direction_array(self) -> NDArray[np.float64]:
        """Return direction as a numpy array."""
        return np.array(self.direction, dtype=np.float64)


@dataclass(frozen=True)
class AOI:
    """An Area of Interest (AOI) representing a shelf or display area.

    AOIs are fixed regions in the scene that viewers may look at.
    Each AOI has a unique identifier and a polygon contour defining its boundary.

    Attributes:
        id: Unique identifier for this AOI (string or integer)
        contour: Polygon vertices as numpy array of shape (N, 2)

    Raises:
        ValidationError: If contour is malformed (not 2D, wrong shape, or < 3 vertices)
    """

    id: str | int
    contour: NDArray[np.floating[Any]]

    def __post_init__(self) -> None:
        """Validate the contour shape."""
        _validate_contour(self.contour)

    @property
    def num_vertices(self) -> int:
        """Return the number of vertices in the contour."""
        return int(self.contour.shape[0])

    def __hash__(self) -> int:
        """Hash based on ID only (contours are mutable arrays)."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Equality based on ID and contour values."""
        if not isinstance(other, AOI):
            return NotImplemented
        return self.id == other.id and np.array_equal(self.contour, other.contour)


@dataclass
class AOIResult:
    """Result for a single AOI after processing a tracking session.

    Tracks how many times this AOI was selected as the "winner" (largest
    visible obstacle) across all samples in the session.

    Attributes:
        aoi_id: The identifier of the AOI this result pertains to
        hit_count: Number of samples where this AOI was selected as winner
        total_attention_seconds: Total viewing time in seconds (= hit_count × sample_interval)
        hit_timestamps: List of sample indices where this AOI won
    """

    aoi_id: str | int
    hit_count: int = 0
    total_attention_seconds: float = 0.0
    hit_timestamps: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Ensure hit_timestamps is a mutable list."""
        if not isinstance(self.hit_timestamps, list):
            self.hit_timestamps = list(self.hit_timestamps)

    def add_hit(self, sample_index: int, sample_interval: float = 1.0) -> None:
        """Record a hit for this AOI.

        Args:
            sample_index: The index of the sample where this AOI won
            sample_interval: Time interval per sample in seconds (default 1.0)
        """
        self.hit_count += 1
        self.total_attention_seconds += sample_interval
        self.hit_timestamps.append(sample_index)

    @property
    def attention_percentage(self) -> float | None:
        """Attention percentage (requires external total to compute)."""
        # This property is a placeholder; actual percentage is computed
        # in TrackingResult based on total session samples
        return None


@dataclass
class TrackingResult:
    """Complete result of a tracking session.

    Contains aggregated results for all AOIs and session-level statistics.

    Attributes:
        aoi_results: Dictionary mapping AOI IDs to their AOIResult
        total_samples: Total number of samples processed
        samples_with_hits: Number of samples where any AOI was visible
        samples_no_winner: Number of samples where no AOI was in view
    """

    aoi_results: Mapping[str | int, AOIResult]
    total_samples: int
    samples_with_hits: int = 0
    samples_no_winner: int = 0

    def __post_init__(self) -> None:
        """Validate result consistency.

        Raises:
            ValidationError: If total_samples is negative
            ValidationError: If samples_with_hits is out of valid range [0, total_samples]
            ValidationError: If samples_no_winner is out of valid range [0, total_samples]
            ValidationError: If samples_with_hits + samples_no_winner != total_samples
        """
        # Convert to dict if needed (Mapping is read-only interface)
        if not isinstance(self.aoi_results, dict):
            object.__setattr__(self, "aoi_results", dict(self.aoi_results))

        # Validate total_samples is non-negative
        if self.total_samples < 0:
            raise ValidationError(
                f"total_samples must be non-negative, got {self.total_samples}"
            )

        # Validate samples_with_hits is in valid range [0, total_samples]
        if self.samples_with_hits < 0:
            raise ValidationError(
                f"samples_with_hits must be non-negative, got {self.samples_with_hits}"
            )
        if self.samples_with_hits > self.total_samples:
            raise ValidationError(
                f"samples_with_hits ({self.samples_with_hits}) cannot exceed "
                f"total_samples ({self.total_samples})"
            )

        # Validate samples_no_winner is in valid range [0, total_samples]
        if self.samples_no_winner < 0:
            raise ValidationError(
                f"samples_no_winner must be non-negative, got {self.samples_no_winner}"
            )
        if self.samples_no_winner > self.total_samples:
            raise ValidationError(
                f"samples_no_winner ({self.samples_no_winner}) cannot exceed "
                f"total_samples ({self.total_samples})"
            )

        # Validate that samples_with_hits + samples_no_winner == total_samples
        if self.samples_with_hits + self.samples_no_winner != self.total_samples:
            raise ValidationError(
                f"samples_with_hits ({self.samples_with_hits}) + samples_no_winner "
                f"({self.samples_no_winner}) must equal total_samples ({self.total_samples})"
            )

    def get_aoi_result(self, aoi_id: str | int) -> AOIResult | None:
        """Get the result for a specific AOI.

        Args:
            aoi_id: The identifier of the AOI

        Returns:
            The AOIResult for the AOI, or None if not found
        """
        return self.aoi_results.get(aoi_id)

    def get_hit_count(self, aoi_id: str | int) -> int:
        """Get the hit count for a specific AOI.

        Args:
            aoi_id: The identifier of the AOI

        Returns:
            The hit count, or 0 if AOI not found
        """
        result = self.aoi_results.get(aoi_id)
        return result.hit_count if result else 0

    def get_total_hits(self) -> int:
        """Get the total number of hits across all AOIs.

        Returns:
            Sum of hit counts for all AOIs
        """
        return sum(r.hit_count for r in self.aoi_results.values())

    def get_attention_seconds(self, aoi_id: str | int) -> float:
        """Get total attention seconds for a specific AOI.

        Args:
            aoi_id: The identifier of the AOI

        Returns:
            Total attention seconds, or 0.0 if AOI not found
        """
        result = self.aoi_results.get(aoi_id)
        return result.total_attention_seconds if result else 0.0

    @property
    def coverage_ratio(self) -> float:
        """Ratio of samples with hits to total samples.

        Returns:
            Float between 0.0 and 1.0 representing coverage
        """
        if self.total_samples == 0:
            return 0.0
        return self.samples_with_hits / self.total_samples

    @property
    def aoi_ids(self) -> list[str | int]:
        """List of all AOI IDs in the result.

        Returns:
            List of AOI identifiers
        """
        return list(self.aoi_results.keys())


# =============================================================================
# Session Configuration Schema (Step 1.3)
# =============================================================================


@dataclass(frozen=True)
class SessionConfig:
    """Immutable acquisition metadata for a tracking session.

    Captures session-level configuration that remains constant throughout
    a batch acquisition period. This metadata is embedded in tracking results
    for downstream analytics and reporting.

    Assumptions documented here:
    - Upstream ingestion guarantees monotonic timestamps and a strict 1 Hz cadence
    - Each sample represents exactly 1 second of viewing time
    - Coordinate space is invariant (image pixels) throughout each batch
    - AOI contours remain fixed in the same coordinate space as viewer samples

    Attributes:
        session_id: Unique identifier for this tracking session
        frame_size: Optional (width, height) of the image frame in pixels.
            When provided, enables bounds checking for viewer sample positions.
        coordinate_space: The coordinate system used for positions and contours.
            Currently only "image" (pixel coordinates) is supported.
        sample_interval_seconds: The time interval between samples in seconds.
            Records the upstream cadence without re-validating timing.
            Defaults to 1.0 (1 Hz sampling rate).
        viewer_id: Optional identifier for the viewer being tracked.
            Useful for cross-referencing batches in multi-viewer scenarios.
        notes: Optional dictionary for downstream analytics metadata.
            Can contain arbitrary key-value pairs for reporting or debugging.

    Raises:
        ValidationError: If session_id is empty or not a string
        ValidationError: If frame_size is malformed (when provided)
        ValidationError: If sample_interval_seconds is not positive
    """

    session_id: str
    frame_size: tuple[int, int] | None = None
    coordinate_space: Literal["image"] = "image"
    sample_interval_seconds: float = 1.0
    viewer_id: str | None = None
    notes: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate and normalize session configuration fields.

        Normalizes frame_size to an immutable tuple of ints to ensure
        the "immutable acquisition metadata" contract is upheld.
        """
        normalized_frame_size = _validate_session_config(self)
        # Use object.__setattr__ because the dataclass is frozen
        if normalized_frame_size is not None:
            object.__setattr__(self, "frame_size", normalized_frame_size)

    @property
    def has_frame_bounds(self) -> bool:
        """Check if frame size is available for bounds checking."""
        return self.frame_size is not None

    @property
    def width(self) -> int | None:
        """Get frame width, or None if frame_size not set."""
        return self.frame_size[0] if self.frame_size else None

    @property
    def height(self) -> int | None:
        """Get frame height, or None if frame_size not set."""
        return self.frame_size[1] if self.frame_size else None


def _validate_session_config(config: SessionConfig) -> tuple[int, int] | None:
    """Validate a SessionConfig instance and return normalized frame_size.

    Args:
        config: The SessionConfig to validate

    Returns:
        Normalized frame_size as tuple of ints, or None if not provided.
        This ensures immutability of the frame_size field.

    Raises:
        ValidationError: If any field is invalid
    """
    # Validate session_id
    if not isinstance(config.session_id, str):
        raise ValidationError(
            f"session_id must be a string, got {type(config.session_id).__name__}"
        )
    if not config.session_id:
        raise ValidationError("session_id cannot be empty")

    # Validate frame_size if provided and normalize to immutable tuple
    normalized_frame_size: tuple[int, int] | None = None
    if config.frame_size is not None:
        if not isinstance(config.frame_size, (tuple, list)):
            raise ValidationError(
                f"frame_size must be a tuple of (width, height), "
                f"got {type(config.frame_size).__name__}"
            )
        if len(config.frame_size) != 2:
            raise ValidationError(
                f"frame_size must have exactly 2 elements (width, height), "
                f"got {len(config.frame_size)} elements"
            )

        width, height = config.frame_size

        # Validate width - check for numeric type first
        if not isinstance(width, Real):
            raise ValidationError(
                f"frame_size width must be a number, got {type(width).__name__}"
            )
        width_float = float(width)
        # Check for non-finite values before attempting int conversion
        if not math.isfinite(width_float):
            raise ValidationError(
                f"frame_size width must be finite, got {width}"
            )
        # Check if it's a whole number (int or whole-number float)
        if not isinstance(width, (int, np.integer)):
            if width_float != int(width_float):
                raise ValidationError(
                    f"frame_size width must be an integer, got {type(width).__name__}"
                )
        width_int = int(width_float)
        if width_int <= 0:
            raise ValidationError(
                f"frame_size width must be positive, got {width}"
            )

        # Validate height - check for numeric type first
        if not isinstance(height, Real):
            raise ValidationError(
                f"frame_size height must be a number, got {type(height).__name__}"
            )
        height_float = float(height)
        # Check for non-finite values before attempting int conversion
        if not math.isfinite(height_float):
            raise ValidationError(
                f"frame_size height must be finite, got {height}"
            )
        # Check if it's a whole number (int or whole-number float)
        if not isinstance(height, (int, np.integer)):
            if height_float != int(height_float):
                raise ValidationError(
                    f"frame_size height must be an integer, got {type(height).__name__}"
                )
        height_int = int(height_float)
        if height_int <= 0:
            raise ValidationError(
                f"frame_size height must be positive, got {height}"
            )

        # Normalize to immutable tuple of ints
        normalized_frame_size = (width_int, height_int)

    # Validate coordinate_space (should only be "image")
    if config.coordinate_space != "image":
        raise ValidationError(
            f"coordinate_space must be 'image', got '{config.coordinate_space}'"
        )

    # Validate sample_interval_seconds
    if not isinstance(config.sample_interval_seconds, Real):
        raise ValidationError(
            f"sample_interval_seconds must be a number, "
            f"got {type(config.sample_interval_seconds).__name__}"
        )
    interval_float = float(config.sample_interval_seconds)
    if not math.isfinite(interval_float):
        raise ValidationError(
            f"sample_interval_seconds must be finite, got {config.sample_interval_seconds}"
        )
    if interval_float <= 0:
        raise ValidationError(
            f"sample_interval_seconds must be positive, got {config.sample_interval_seconds}"
        )

    # Validate viewer_id if provided
    if config.viewer_id is not None and not isinstance(config.viewer_id, str):
        raise ValidationError(
            f"viewer_id must be a string or None, got {type(config.viewer_id).__name__}"
        )

    # Validate notes if provided
    if config.notes is not None and not isinstance(config.notes, dict):
        raise ValidationError(
            f"notes must be a dict or None, got {type(config.notes).__name__}"
        )

    return normalized_frame_size


# =============================================================================
# Input Validation Functions (Step 1.2)
# =============================================================================


def _validate_frame_size(
    frame_size: tuple[float, float]
) -> tuple[float, float]:
    """Validate frame_size is a 2-tuple of finite positive numbers.

    Args:
        frame_size: Expected to be (width, height) tuple.
            Accepts int, float, or numpy scalar types.

    Returns:
        Validated (width, height) as floats.

    Raises:
        ValidationError: If frame_size is malformed.
    """
    # Check it's a tuple or list with exactly 2 elements
    if not isinstance(frame_size, (tuple, list)):
        raise ValidationError(
            f"frame_size must be a tuple of (width, height), "
            f"got {type(frame_size).__name__}"
        )

    if len(frame_size) != 2:
        raise ValidationError(
            f"frame_size must have exactly 2 elements (width, height), "
            f"got {len(frame_size)} elements"
        )

    width, height = frame_size

    # Check width is a finite positive number
    if not isinstance(width, Real):
        raise ValidationError(
            f"frame_size width must be a number, got {type(width).__name__}"
        )
    width_float = float(width)
    if not math.isfinite(width_float):
        raise ValidationError(
            f"frame_size width must be finite, got {width}"
        )
    if width_float <= 0:
        raise ValidationError(
            f"frame_size width must be positive, got {width}"
        )

    # Check height is a finite positive number
    if not isinstance(height, Real):
        raise ValidationError(
            f"frame_size height must be a number, got {type(height).__name__}"
        )
    height_float = float(height)
    if not math.isfinite(height_float):
        raise ValidationError(
            f"frame_size height must be finite, got {height}"
        )
    if height_float <= 0:
        raise ValidationError(
            f"frame_size height must be positive, got {height}"
        )

    return width_float, height_float


# Type alias for flexible sample input
SampleInput = list["ViewerSample"] | NDArray[np.floating[Any]]


def normalize_sample_input(
    samples: SampleInput,
) -> list["ViewerSample"]:
    """Normalize various input formats to a list of ViewerSample objects.

    Supports multiple input formats for ergonomic API:
    - List of ViewerSample objects (returned as-is)
    - NumPy array of shape (N, 4) for [x, y, dx, dy] per row

    The direction vectors in numpy input are normalized to unit vectors.

    Args:
        samples: Either a list of ViewerSample objects or a numpy array
            of shape (N, 4) where each row is [x, y, dx, dy].

    Returns:
        List of ViewerSample objects.

    Raises:
        ValidationError: If samples format is unrecognized
        ValidationError: If numpy array shape is not (N, 4)
        ValidationError: If any direction vector has zero magnitude
    """
    # Already a list - return as-is (validation happens later)
    if isinstance(samples, list):
        return samples

    # NumPy array of shape (N, 4)
    if isinstance(samples, np.ndarray):
        if samples.ndim != 2:
            raise ValidationError(
                f"NumPy samples must be 2D array, got shape {samples.shape}"
            )
        if samples.shape[1] != 4:
            raise ValidationError(
                f"NumPy samples must have shape (N, 4), got shape {samples.shape}"
            )
        if samples.shape[0] == 0:
            return []

        result: list[ViewerSample] = []
        for i, row in enumerate(samples):
            x, y, dx, dy = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            # Normalize direction to unit vector
            mag = math.sqrt(dx * dx + dy * dy)
            if mag == 0:
                raise ValidationError(
                    f"Sample at index {i} has zero-magnitude direction vector"
                )
            dx_norm, dy_norm = dx / mag, dy / mag
            result.append(
                ViewerSample(
                    position=(x, y),
                    direction=(dx_norm, dy_norm),
                )
            )
        return result

    raise ValidationError(
        f"samples must be a list or numpy array, got {type(samples).__name__}"
    )


def validate_viewer_samples(
    samples: list[ViewerSample],
    frame_size: tuple[float, float] | None = None,
) -> None:
    """Validate a list of ViewerSample objects.

    Checks that the sample list is valid and all samples have valid positions.
    Empty sample lists are allowed (graceful handling).

    Args:
        samples: List of ViewerSample objects to validate
        frame_size: Optional (width, height) tuple for bounds checking.
            If provided, positions must be within [0, width) x [0, height).
            Must be a 2-tuple of finite positive numbers.
            Accepts int, float, or numpy scalar types at runtime.

    Raises:
        ValidationError: If samples is not a list
        ValidationError: If any sample is not a ViewerSample instance
        ValidationError: If frame_size is malformed (not a 2-tuple of finite positive numbers)
        ValidationError: If any position is out of bounds (when frame_size provided)
    """
    if not isinstance(samples, list):
        raise ValidationError(
            f"samples must be a list, got {type(samples).__name__}"
        )

    # Validate frame_size upfront if provided
    validated_frame_size: tuple[float, float] | None = None
    if frame_size is not None:
        validated_frame_size = _validate_frame_size(frame_size)

    for i, sample in enumerate(samples):
        if not isinstance(sample, ViewerSample):
            raise ValidationError(
                f"Sample at index {i} must be a ViewerSample, "
                f"got {type(sample).__name__}"
            )

        # Position bounds checking when frame_size is provided
        if validated_frame_size is not None:
            width, height = validated_frame_size
            x, y = sample.position

            if not (0 <= x < width):
                raise ValidationError(
                    f"Sample at index {i} has x position {x} out of bounds "
                    f"[0, {width})"
                )
            if not (0 <= y < height):
                raise ValidationError(
                    f"Sample at index {i} has y position {y} out of bounds "
                    f"[0, {height})"
                )


def validate_aois(aois: list[AOI]) -> None:
    """Validate a list of AOI objects.

    Checks that the AOI list is valid and all AOIs have unique IDs.
    Empty AOI lists are allowed (graceful handling).

    Args:
        aois: List of AOI objects to validate

    Raises:
        ValidationError: If aois is not a list
        ValidationError: If any element is not an AOI instance
        ValidationError: If duplicate AOI IDs are found
    """
    if not isinstance(aois, list):
        raise ValidationError(
            f"aois must be a list, got {type(aois).__name__}"
        )

    seen_ids: set[str | int] = set()
    duplicate_ids: list[str | int] = []

    for i, aoi in enumerate(aois):
        if not isinstance(aoi, AOI):
            raise ValidationError(
                f"AOI at index {i} must be an AOI instance, "
                f"got {type(aoi).__name__}"
            )

        if aoi.id in seen_ids:
            duplicate_ids.append(aoi.id)
        seen_ids.add(aoi.id)

    if duplicate_ids:
        raise ValidationError(
            f"Duplicate AOI IDs found: {duplicate_ids}"
        )


def validate_tracking_params(
    fov_deg: float,
    max_range: float,
    sample_interval: float = 1.0,
) -> None:
    """Validate tracking parameters.

    Checks that FOV, max_range, and sample_interval are valid finite values.

    Args:
        fov_deg: Field of view in degrees. Must be a finite number in (0, 360].
            Accepts int, float, or numpy scalar types at runtime.
        max_range: Maximum detection range in pixels. Must be a finite positive number.
            Accepts int, float, or numpy scalar types at runtime.
        sample_interval: Time interval between samples in seconds. Must be a finite positive number.
            Accepts int, float, or numpy scalar types at runtime.

    Raises:
        ValidationError: If fov_deg is not a finite number in (0, 360]
        ValidationError: If max_range is not a finite positive number
        ValidationError: If sample_interval is not a finite positive number
    """
    # Validate fov_deg
    if not isinstance(fov_deg, Real):
        raise ValidationError(
            f"fov_deg must be a number, got {type(fov_deg).__name__}"
        )
    fov_deg_float = float(fov_deg)
    if not math.isfinite(fov_deg_float):
        raise ValidationError(
            f"fov_deg must be finite, got {fov_deg}"
        )
    if not (0 < fov_deg_float <= 360):
        raise ValidationError(
            f"fov_deg must be in range (0, 360], got {fov_deg}"
        )

    # Validate max_range
    if not isinstance(max_range, Real):
        raise ValidationError(
            f"max_range must be a number, got {type(max_range).__name__}"
        )
    max_range_float = float(max_range)
    if not math.isfinite(max_range_float):
        raise ValidationError(
            f"max_range must be finite, got {max_range}"
        )
    if max_range_float <= 0:
        raise ValidationError(
            f"max_range must be positive, got {max_range}"
        )

    # Validate sample_interval
    if not isinstance(sample_interval, Real):
        raise ValidationError(
            f"sample_interval must be a number, got {type(sample_interval).__name__}"
        )
    sample_interval_float = float(sample_interval)
    if not math.isfinite(sample_interval_float):
        raise ValidationError(
            f"sample_interval must be finite, got {sample_interval}"
        )
    if sample_interval_float <= 0:
        raise ValidationError(
            f"sample_interval must be positive, got {sample_interval}"
        )

# =============================================================================
# Core Tracking Algorithm (Phase 2)
# =============================================================================


@dataclass
class AOIIntervalBreakdown:
    """Detailed breakdown of a single angular interval for an AOI.

    This mirrors the IntervalBreakdown from the core API but uses AOI IDs
    instead of obstacle indices.

    Attributes:
        angle_start: Start angle in radians
        angle_end: End angle in radians
        angular_span: Angular span in radians
        aoi_id: ID of the AOI owning this interval
        min_distance: Minimum distance within this interval
        wraps: Whether the interval crosses the ±π discontinuity
    """

    angle_start: float
    angle_end: float
    angular_span: float
    aoi_id: str | int
    min_distance: float
    wraps: bool = False

    @property
    def angular_span_deg(self) -> float:
        """Angular span in degrees."""
        return float(np.rad2deg(self.angular_span))

    @property
    def angle_start_deg(self) -> float:
        """Start angle in degrees."""
        return float(np.rad2deg(self.angle_start))

    @property
    def angle_end_deg(self) -> float:
        """End angle in degrees."""
        return float(np.rad2deg(self.angle_end))


@dataclass
class SingleSampleResult:
    """Result from processing a single viewer sample.

    Contains detailed information about which AOI (if any) was selected
    as the winner for a single sample observation.

    Attributes:
        winning_aoi_id: The ID of the AOI with largest angular coverage,
            or None if no AOI was visible in the view arc
        angular_coverage: Angular coverage of the winning AOI in radians
        min_distance: Minimum distance to the winning AOI
        all_coverage: Optional dict mapping all visible AOI IDs to their coverage
        all_distances: Optional dict mapping all visible AOI IDs to their min distances
        interval_details: Optional list of AOIIntervalBreakdown objects showing
            all angular intervals and which AOI owns each one
    """

    winning_aoi_id: str | int | None
    angular_coverage: float = 0.0
    min_distance: float = float("inf")
    all_coverage: dict[str | int, float] | None = None
    all_distances: dict[str | int, float] | None = None
    interval_details: list[AOIIntervalBreakdown] | None = None

    def get_winner_intervals(self) -> list[AOIIntervalBreakdown]:
        """Get intervals owned by the winning AOI only.

        Returns:
            List of AOIIntervalBreakdown objects for the winner, or empty list
        """
        if self.interval_details is None or self.winning_aoi_id is None:
            return []
        return [iv for iv in self.interval_details if iv.aoi_id == self.winning_aoi_id]

    def get_all_intervals(self) -> list[AOIIntervalBreakdown]:
        """Get all intervals (for all AOIs, not just the winner).

        Returns:
            List of AOIIntervalBreakdown objects, or empty list if not available
        """
        if self.interval_details is not None:
            return self.interval_details
        return []


def process_single_sample(
    sample: ViewerSample,
    aois: list[AOI],
    fov_deg: float = 90.0,
    max_range: float = 500.0,
    return_details: bool = False,
) -> str | int | None | SingleSampleResult:
    """Process a single viewer sample to find which AOI is being viewed.

    This is a wrapper around `find_largest_obstacle()` that:
    - Accepts a ViewerSample and list of AOIs
    - Converts AOI contours to the format expected by find_largest_obstacle
    - Returns the winning AOI ID (or None if no winner)
    - Optionally returns detailed result for debugging

    Args:
        sample: A ViewerSample containing position and direction
        aois: List of AOI objects to check against
        fov_deg: Field of view in degrees (default 90.0)
        max_range: Maximum detection range in pixels (default 500.0)
        return_details: If True, return SingleSampleResult with full details;
            if False, return just the winning AOI ID or None

    Returns:
        If return_details is False: The winning AOI ID (str or int), or None
            if no AOI was visible in the view arc
        If return_details is True: A SingleSampleResult with full information

    Raises:
        ValidationError: If sample is not a ViewerSample
        ValidationError: If aois is not a list of AOI objects
        ValidationError: If fov_deg or max_range are invalid

    Example:
        >>> sample = ViewerSample(position=(100.0, 100.0), direction=(0.0, 1.0))
        >>> aoi = AOI(id="shelf1", contour=np.array([[90, 150], [110, 150], [100, 170]]))
        >>> winner_id = process_single_sample(sample, [aoi])
        >>> print(f"Viewer is looking at: {winner_id}")
    """
    # Import here to avoid circular imports
    from view_arc.api import find_largest_obstacle

    # Validate inputs
    if not isinstance(sample, ViewerSample):
        raise ValidationError(
            f"sample must be a ViewerSample, got {type(sample).__name__}"
        )

    if not isinstance(aois, list):
        raise ValidationError(f"aois must be a list, got {type(aois).__name__}")

    for i, aoi in enumerate(aois):
        if not isinstance(aoi, AOI):
            raise ValidationError(
                f"aois[{i}] must be an AOI, got {type(aoi).__name__}"
            )

    # Validate tracking parameters
    validate_tracking_params(fov_deg, max_range)

    # Handle empty AOI list
    if len(aois) == 0:
        if return_details:
            return SingleSampleResult(winning_aoi_id=None)
        return None

    # Build mapping from obstacle index to AOI ID
    aoi_id_by_index: dict[int, str | int] = {}
    obstacle_contours: list[NDArray[np.float32]] = []

    for idx, aoi in enumerate(aois):
        aoi_id_by_index[idx] = aoi.id
        obstacle_contours.append(aoi.contour.astype(np.float32))

    # Convert sample to numpy arrays for find_largest_obstacle
    viewer_point = np.array(sample.position, dtype=np.float32)
    view_direction = np.array(sample.direction, dtype=np.float32)

    # Call the core obstacle detection API
    # When details are requested, also get intervals for debugging
    result = find_largest_obstacle(
        viewer_point=viewer_point,
        view_direction=view_direction,
        field_of_view_deg=fov_deg,
        max_range=max_range,
        obstacle_contours=obstacle_contours,
        return_intervals=return_details,
        return_all_coverage=return_details,
    )

    # Map obstacle index back to AOI ID
    winning_aoi_id: str | int | None = None
    if result.obstacle_id is not None:
        winning_aoi_id = aoi_id_by_index.get(result.obstacle_id)

    if not return_details:
        return winning_aoi_id

    # Build detailed result
    all_coverage: dict[str | int, float] | None = None
    if result.all_coverage is not None:
        all_coverage = {}
        for obs_idx, coverage in result.all_coverage.items():
            aoi_id = aoi_id_by_index.get(obs_idx)
            if aoi_id is not None:
                all_coverage[aoi_id] = coverage

    # Build all_distances mapping
    all_distances: dict[str | int, float] | None = None
    if result.all_distances is not None:
        all_distances = {}
        for obs_idx, distance in result.all_distances.items():
            aoi_id = aoi_id_by_index.get(obs_idx)
            if aoi_id is not None:
                all_distances[aoi_id] = distance

    # Build interval_details with AOI IDs instead of obstacle indices
    interval_details: list[AOIIntervalBreakdown] | None = None
    if result.interval_details is not None:
        interval_details = []
        for iv in result.interval_details:
            aoi_id = aoi_id_by_index.get(iv.obstacle_id)
            if aoi_id is not None:
                interval_details.append(
                    AOIIntervalBreakdown(
                        angle_start=iv.angle_start,
                        angle_end=iv.angle_end,
                        angular_span=iv.angular_span,
                        aoi_id=aoi_id,
                        min_distance=iv.min_distance,
                        wraps=iv.wraps,
                    )
                )

    return SingleSampleResult(
        winning_aoi_id=winning_aoi_id,
        angular_coverage=result.angular_coverage,
        min_distance=result.min_distance,
        all_coverage=all_coverage,
        all_distances=all_distances,
        interval_details=interval_details,
    )


# =============================================================================
# Batch Processing Function (Step 2.2)
# =============================================================================


@dataclass
class TrackingResultWithConfig(TrackingResult):
    """TrackingResult extended with embedded SessionConfig.

    Contains all the fields from TrackingResult plus the session configuration
    that was used for this tracking run.

    Attributes:
        session_config: The SessionConfig used for this tracking session,
            or None if not provided
    """

    session_config: SessionConfig | None = None


def compute_attention_seconds(
    samples: SampleInput,
    aois: list[AOI],
    fov_deg: float = 90.0,
    max_range: float = 500.0,
    sample_interval: float = 1.0,
    session_config: SessionConfig | None = None,
) -> TrackingResultWithConfig:
    """Compute accumulated attention seconds for each AOI from a batch of samples.

    This is the main entry point for batch processing. It iterates through all
    viewer samples, determines which AOI (if any) is being viewed at each sample,
    and accumulates hit counts per AOI.

    Each processed sample is assumed to represent exactly `sample_interval` seconds
    of viewing time (default 1 second at 1 Hz sampling rate).

    Args:
        samples: Viewer observations in one of the following formats:
            - List of ViewerSample objects
            - NumPy array of shape (N, 4) where each row is [x, y, dx, dy]
            Direction vectors in numpy input are automatically normalized.
        aois: List of AOI objects defining the areas of interest to track.
        fov_deg: Field of view in degrees (default 90.0)
        max_range: Maximum detection range in pixels (default 500.0)
        sample_interval: Time interval per sample in seconds (default 1.0).
            Each hit adds this many seconds to the AOI's total_attention_seconds.
        session_config: Optional session configuration for metadata tracking.
            If provided, frame_size is used for bounds checking on sample positions,
            and the config is embedded in the result for downstream analytics.

    Returns:
        TrackingResultWithConfig containing:
        - aoi_results: Dict mapping AOI IDs to AOIResult objects with hit counts
        - total_samples: Total number of samples processed
        - samples_with_hits: Number of samples where any AOI was visible
        - samples_no_winner: Number of samples where no AOI was in view
        - session_config: The SessionConfig used (or None if not provided)

    Raises:
        ValidationError: If samples is not a valid list of ViewerSamples
        ValidationError: If aois contains invalid or duplicate IDs
        ValidationError: If fov_deg, max_range, or sample_interval are invalid

    Invariants:
        - total_samples == len(samples)
        - samples_with_hits + samples_no_winner == total_samples
        - sum(aoi_result.hit_count for all AOIs) == samples_with_hits
        - All AOI IDs present in result (even with hit_count=0)

    Example:
        >>> samples = [
        ...     ViewerSample(position=(100.0, 100.0), direction=(0.0, 1.0)),
        ...     ViewerSample(position=(100.0, 100.0), direction=(0.0, 1.0)),
        ...     ViewerSample(position=(100.0, 100.0), direction=(1.0, 0.0)),
        ... ]
        >>> aois = [
        ...     AOI(id="shelf_A", contour=np.array([[90, 150], [110, 150], [100, 170]])),
        ...     AOI(id="shelf_B", contour=np.array([[150, 90], [170, 90], [160, 110]])),
        ... ]
        >>> result = compute_attention_seconds(samples, aois)
        >>> print(f"shelf_A received {result.get_hit_count('shelf_A')} seconds of attention")
    """
    # Normalize input to list of ViewerSample objects
    normalized_samples = normalize_sample_input(samples)

    # Determine frame_size for bounds checking
    frame_size: tuple[float, float] | None = None
    if session_config is not None and session_config.frame_size is not None:
        frame_size = (
            float(session_config.frame_size[0]),
            float(session_config.frame_size[1]),
        )

    # Validate inputs
    validate_viewer_samples(normalized_samples, frame_size=frame_size)
    validate_aois(aois)
    validate_tracking_params(fov_deg, max_range, sample_interval)

    # Initialize AOI results for all AOIs (even those with 0 hits)
    aoi_results: dict[str | int, AOIResult] = {}
    for aoi in aois:
        aoi_results[aoi.id] = AOIResult(aoi_id=aoi.id)

    # Track counters
    total_samples = len(normalized_samples)
    samples_with_hits = 0
    samples_no_winner = 0

    # Process each sample
    for sample_index, sample in enumerate(normalized_samples):
        # Get the winning AOI ID for this sample
        winning_id = process_single_sample(
            sample=sample,
            aois=aois,
            fov_deg=fov_deg,
            max_range=max_range,
            return_details=False,
        )

        if winning_id is not None:
            # Record hit for this AOI
            samples_with_hits += 1
            # winning_id is guaranteed to be str | int when not None
            assert isinstance(winning_id, (str, int))  # for type checker
            aoi_results[winning_id].add_hit(sample_index, sample_interval)
        else:
            samples_no_winner += 1

    return TrackingResultWithConfig(
        aoi_results=aoi_results,
        total_samples=total_samples,
        samples_with_hits=samples_with_hits,
        samples_no_winner=samples_no_winner,
        session_config=session_config,
    )