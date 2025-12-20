"""
Tracking Data Structures
========================

Core data structures for temporal attention tracking:
- ViewerSample: Single observation of viewer position and direction
- AOI: Area of Interest with ID and contour
- AOIResult: Per-AOI result with hit count
- TrackingResult: Complete session results
- SessionConfig: Immutable acquisition metadata
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from view_arc.obstacle.clipping import is_valid_polygon
from view_arc.obstacle.geometry import validate_and_get_direction_angle

if TYPE_CHECKING:
    import pandas as pd


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

    def copy(self) -> "AOIResult":
        """Create a deep copy of this AOIResult.

        Returns:
            A new AOIResult with the same data but independent hit_timestamps list.
        """
        return AOIResult(
            aoi_id=self.aoi_id,
            hit_count=self.hit_count,
            total_attention_seconds=self.total_attention_seconds,
            hit_timestamps=list(self.hit_timestamps),
        )

    @property
    def attention_percentage(self) -> float | None:
        """Attention percentage (requires external total to compute)."""
        # This property is a placeholder; actual percentage is computed
        # in TrackingResult based on total session samples
        return None


# =============================================================================
# Sampling Assumptions (Step 2.4)
# =============================================================================

# These assumptions are documented in the module docstring, README, and SessionConfig.
# Exposing them as a tuple allows downstream analytics to know the data quality contract.
SAMPLING_ASSUMPTIONS: tuple[str, ...] = (
    "Samples arrive at a fixed 1 Hz cadence (one sample per second)",
    "Each sample represents exactly 1 second of viewing time",
    "Timestamps, when provided, are already sorted upstream",
    "AOI contours remain fixed in image coordinate space",
    "Each batch tracks a single viewer",
)
"""Tuple of sampling assumptions that apply to all tracking sessions.

These invariants are guaranteed by upstream ingestion and are NOT re-validated
at runtime. Downstream analytics can inspect this tuple to understand the
data quality contract without scanning logs.
"""


@dataclass
class TrackingResult:
    """Complete result of a tracking session.

    Contains aggregated results for all AOIs and session-level statistics.

    The `assumptions` property exposes the sampling invariants that were applied
    during tracking, allowing downstream consumers to know the data quality
    contract without scanning logs.

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

        # Validate that per-AOI tallies agree with session totals
        total_hit_count = sum(r.hit_count for r in self.aoi_results.values())
        if total_hit_count != self.samples_with_hits:
            raise ValidationError(
                f"Sum of AOI hit counts ({total_hit_count}) must equal "
                f"samples_with_hits ({self.samples_with_hits})"
            )

        total_timestamp_count = sum(len(r.hit_timestamps) for r in self.aoi_results.values())
        if total_timestamp_count != self.samples_with_hits:
            raise ValidationError(
                f"Sum of AOI hit_timestamps lengths ({total_timestamp_count}) must equal "
                f"samples_with_hits ({self.samples_with_hits})"
            )

        # Validate that each AOI's hit_count matches its hit_timestamps length
        for aoi_id, result in self.aoi_results.items():
            if len(result.hit_timestamps) != result.hit_count:
                raise ValidationError(
                    f"AOI '{aoi_id}' hit_timestamps length ({len(result.hit_timestamps)}) "
                    f"does not match hit_count ({result.hit_count})"
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
    def dominant_aoi(self) -> str | int | None:
        """AOI with the most hits, or None if no hits or tie.

        When multiple AOIs have the same (maximum) hit count, returns None
        to indicate no clear winner. When no AOI has any hits, returns None.

        Returns:
            The ID of the dominant AOI, or None if there's a tie or no hits
        """
        if self.samples_with_hits == 0:
            return None
        
        # Find maximum hit count
        max_hits = max(result.hit_count for result in self.aoi_results.values())
        
        if max_hits == 0:
            return None
        
        # Count how many AOIs have the maximum hit count
        aois_with_max = [
            aoi_id for aoi_id, result in self.aoi_results.items()
            if result.hit_count == max_hits
        ]
        
        # Return None if there's a tie
        if len(aois_with_max) > 1:
            return None
        
        return aois_with_max[0]

    @property
    def engagement_score(self) -> float:
        """Weighted engagement score based on attention distribution.

        Calculates a score that reflects how concentrated or distributed the
        viewer's attention was across AOIs. The score is based on the
        Herfindahl-Hirschman Index (HHI), normalized to [0, 1]:
        - 1.0 = all attention on a single AOI (maximum concentration)
        - Close to 0 = attention evenly distributed across many AOIs
        - 0.0 = no attention recorded

        Returns:
            Float between 0.0 and 1.0 representing engagement concentration

        Note:
            For a single AOI, the score is 1.0. For N AOIs with equal attention,
            the score is 1/N. The score is 0.0 when no hits were recorded.
        """
        if self.samples_with_hits == 0:
            return 0.0
        
        # Calculate HHI: sum of squared market shares
        hhi = sum(
            (result.hit_count / self.samples_with_hits) ** 2
            for result in self.aoi_results.values()
        )
        
        return float(hhi)

    @property
    def session_duration(self) -> float:
        """Total session duration in seconds.

        Calculated as total_samples × sample_interval (default 1.0 second).
        This represents the total time span covered by the tracking session.

        Returns:
            Total session duration in seconds

        Note:
            This assumes the standard 1.0 second sample interval. If a different
            interval was used, this calculation would need to account for that
            (future enhancement with SessionConfig integration).
        """
        # Default sample interval is 1.0 second
        # TODO: When TrackingResultWithConfig is used, get interval from SessionConfig
        sample_interval = 1.0
        return self.total_samples * sample_interval

    @property
    def aoi_ids(self) -> list[str | int]:
        """List of all AOI IDs in the result.

        Returns:
            List of AOI identifiers
        """
        return list(self.aoi_results.keys())

    @property
    def assumptions(self) -> tuple[str, ...]:
        """Sampling assumptions applied during tracking.

        These invariants were guaranteed by upstream ingestion and were NOT
        re-validated at runtime. Downstream analytics can inspect this tuple
        to understand the data quality contract.

        Returns:
            Tuple of strings describing the sampling assumptions.
        """
        return SAMPLING_ASSUMPTIONS

    def get_top_aois(self, n: int) -> list[tuple[str | int, int]]:
        """Return the top N AOIs by hit count.

        AOIs are sorted in descending order of hit count. If multiple AOIs have
        the same hit count, they are sorted by their ID (lexicographically).
        If n exceeds the number of AOIs, all AOIs are returned.

        Args:
            n: Number of top AOIs to return

        Returns:
            List of (aoi_id, hit_count) tuples, sorted by descending hit count

        Raises:
            ValidationError: If n is negative
        """
        if n < 0:
            raise ValidationError(f"n must be non-negative, got {n}")

        # Sort AOIs by hit count (descending), then by ID (ascending) for ties
        sorted_aois = sorted(
            self.aoi_results.items(),
            key=lambda item: (-item[1].hit_count, str(item[0])),
        )

        # Return top n (or all if n exceeds available)
        return [(aoi_id, result.hit_count) for aoi_id, result in sorted_aois[:n]]

    def get_attention_distribution(
        self, include_no_hits: bool = False
    ) -> dict[str | int, float]:
        """Return the attention distribution as percentages across AOIs.

        Calculates what percentage of time was spent viewing each AOI.
        By default, excludes AOIs with zero hits unless include_no_hits is True.

        Args:
            include_no_hits: If True, include AOIs with zero hits in the result

        Returns:
            Dictionary mapping AOI IDs to their attention percentage (0-100).
            Percentages sum to 100.0 (excluding samples with no winner).

        Note:
            If no AOI was ever viewed (samples_with_hits == 0), returns an
            empty dictionary (or all zeros if include_no_hits=True).
        """
        distribution: dict[str | int, float] = {}

        # If no samples had hits, return empty or zeros
        if self.samples_with_hits == 0:
            if include_no_hits:
                return {aoi_id: 0.0 for aoi_id in self.aoi_results.keys()}
            return {}

        # Calculate percentage based on samples with hits
        for aoi_id, result in self.aoi_results.items():
            percentage = (result.hit_count / self.samples_with_hits) * 100.0
            if include_no_hits or percentage > 0:
                distribution[aoi_id] = percentage

        return distribution

    def get_viewing_timeline(self) -> list[tuple[int, str | int | None]]:
        """Return the viewing timeline as a chronological sequence.

        Returns a list of (sample_index, aoi_id) tuples representing which AOI
        was being viewed at each sample. Samples with no winner have aoi_id=None.

        Returns:
            List of (sample_index, aoi_id) tuples in chronological order.
            aoi_id is None for samples where no AOI was visible.

        Raises:
            ValidationError: If any hit_timestamps contain invalid indices
                (negative or >= total_samples)

        Note:
            This method reconstructs the timeline from hit_timestamps recorded
            in each AOIResult. The timeline length equals total_samples.
        """
        # Build a mapping from sample index to AOI ID
        timeline: list[str | int | None] = [None] * self.total_samples

        for aoi_id, result in self.aoi_results.items():
            for sample_idx in result.hit_timestamps:
                # Validate index is within valid range
                if not isinstance(sample_idx, int):
                    raise ValidationError(
                        f"AOI '{aoi_id}' has non-integer hit_timestamp: {sample_idx}"
                    )
                if sample_idx < 0:
                    raise ValidationError(
                        f"AOI '{aoi_id}' has negative hit_timestamp: {sample_idx}"
                    )
                if sample_idx >= self.total_samples:
                    raise ValidationError(
                        f"AOI '{aoi_id}' has hit_timestamp ({sample_idx}) >= "
                        f"total_samples ({self.total_samples})"
                    )
                timeline[sample_idx] = aoi_id

        # Convert to list of tuples with sample indices
        return [(idx, aoi_id) for idx, aoi_id in enumerate(timeline)]

    def to_dataframe(self) -> pd.DataFrame:
        """Export tracking results to a pandas DataFrame.

        Creates a DataFrame with one row per AOI containing:
        - aoi_id: The AOI identifier
        - hit_count: Number of times this AOI won
        - total_attention_seconds: Total viewing time in seconds
        - attention_percentage: Percentage of viewing time (0-100)

        Returns:
            pandas.DataFrame with AOI results

        Raises:
            ImportError: If pandas is not installed

        Note:
            This method has pandas as an optional dependency. Install with:
            `pip install pandas` or `uv pip install pandas`
        """
        try:
            import pandas as pd  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install with: pip install pandas"
            ) from e

        # Get attention distribution
        distribution = self.get_attention_distribution(include_no_hits=True)

        # Build rows for DataFrame
        rows = []
        for aoi_id, result in self.aoi_results.items():
            rows.append(
                {
                    "aoi_id": aoi_id,
                    "hit_count": result.hit_count,
                    "total_attention_seconds": result.total_attention_seconds,
                    "attention_percentage": distribution.get(aoi_id, 0.0),
                }
            )

        # Create DataFrame with explicit columns to ensure empty DataFrame has correct structure
        columns = ["aoi_id", "hit_count", "total_attention_seconds", "attention_percentage"]
        df = pd.DataFrame(rows, columns=columns)
        
        # Sort by hit count descending if not empty
        if not df.empty:
            df = df.sort_values("hit_count", ascending=False).reset_index(drop=True)

        return df


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
# Profiling Data (Step 6.1)
# =============================================================================


@dataclass
class ProfilingData:
    """Performance profiling data for a tracking run.

    This dataclass captures timing and sample processing metrics when
    profiling is enabled in compute_attention_seconds().

    Attributes:
        total_time_seconds: Total wall-clock time for processing all samples
        samples_processed: Number of samples successfully processed
        samples_per_second: Throughput (samples/second)
        avg_time_per_sample_ms: Average time per sample in milliseconds
        peak_memory_bytes: Peak memory usage if available (None otherwise)
        cache_hit_ratio: Ratio of cache hits if caching is enabled (None otherwise)
    """

    total_time_seconds: float
    samples_processed: int
    samples_per_second: float = 0.0
    avg_time_per_sample_ms: float = 0.0
    peak_memory_bytes: int | None = None
    cache_hit_ratio: float | None = None

    def __post_init__(self) -> None:
        """Calculate derived metrics."""
        if self.total_time_seconds > 0 and self.samples_processed > 0:
            object.__setattr__(
                self, "samples_per_second", self.samples_processed / self.total_time_seconds
            )
            object.__setattr__(
                self,
                "avg_time_per_sample_ms",
                (self.total_time_seconds / self.samples_processed) * 1000,
            )

    def __repr__(self) -> str:
        """Human-readable representation."""
        lines = [
            f"ProfilingData:",
            f"  Total time: {self.total_time_seconds:.3f}s",
            f"  Samples: {self.samples_processed}",
            f"  Throughput: {self.samples_per_second:.1f} samples/s",
            f"  Avg time/sample: {self.avg_time_per_sample_ms:.2f}ms",
        ]
        if self.peak_memory_bytes is not None:
            lines.append(f"  Peak memory: {self.peak_memory_bytes / 1024 / 1024:.1f} MB")
        if self.cache_hit_ratio is not None:
            lines.append(f"  Cache hit ratio: {self.cache_hit_ratio:.1%}")
        return "\n".join(lines)
