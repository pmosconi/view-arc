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
from typing import Any

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
