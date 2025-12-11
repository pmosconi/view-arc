"""
Tracking Algorithm
==================

Core tracking algorithm functions:
- process_single_sample: Process one viewer sample to find winning AOI
- compute_attention_seconds: Batch process samples and accumulate attention
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from view_arc.tracking.dataclasses import (
    AOI,
    AOIResult,
    SessionConfig,
    TrackingResult,
    ValidationError,
    ViewerSample,
)
from view_arc.tracking.validation import (
    SampleInput,
    normalize_sample_input,
    validate_aois,
    validate_tracking_params,
    validate_viewer_samples,
)


# =============================================================================
# Single Sample Result Dataclasses
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


# =============================================================================
# Single-Sample Processing (Step 2.1)
# =============================================================================


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
    from view_arc.obstacle.api import find_largest_obstacle

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
