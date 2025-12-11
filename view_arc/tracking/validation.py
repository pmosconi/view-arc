"""
Tracking Input Validation
=========================

Validation functions for tracking inputs:
- validate_viewer_samples: Validate sample list and positions
- validate_aois: Validate AOI list and unique IDs
- validate_tracking_params: Validate FOV, max_range, sample_interval
- normalize_sample_input: Convert various input formats to ViewerSample list
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

import numpy as np
from numpy.typing import NDArray

from view_arc.tracking.dataclasses import (
    AOI,
    ValidationError,
    ViewerSample,
)


# Type alias for flexible sample input
SampleInput = list["ViewerSample"] | NDArray[np.floating[Any]]


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


def normalize_sample_input(
    samples: SampleInput,
) -> list[ViewerSample]:
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
