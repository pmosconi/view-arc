"""
View Arc - Attention Tracking
=============================

Public API for tracking viewer attention on Areas of Interest (AOIs).

The main entry point is `compute_attention_seconds()` which processes
batches of viewer samples and accumulates attention time per AOI.

For lower-level access to obstacle detection, use `view_arc.obstacle`.
For visualization utilities, use `view_arc.obstacle.visualize`.
"""

# Primary tracking API
from view_arc.tracking import (
    # Main function
    compute_attention_seconds,
    # Data structures
    ViewerSample,
    AOI,
    AOIResult,
    TrackingResult,
    TrackingResultWithConfig,
    SessionConfig,
    SingleSampleResult,
    AOIIntervalBreakdown,
    # Validation
    ValidationError,
    SampleInput,
    # Lower-level processing
    process_single_sample,
)

# Re-export obstacle detection API for convenience
from view_arc.obstacle import (
    find_largest_obstacle,
    ObstacleResult,
    IntervalBreakdown,
)

__all__ = [
    # Primary tracking API
    'compute_attention_seconds',
    # Data structures
    'ViewerSample',
    'AOI',
    'AOIResult',
    'TrackingResult',
    'TrackingResultWithConfig',
    'SessionConfig',
    'SingleSampleResult',
    'AOIIntervalBreakdown',
    # Validation
    'ValidationError',
    'SampleInput',
    # Lower-level processing
    'process_single_sample',
    # Obstacle detection (for advanced use)
    'find_largest_obstacle',
    'ObstacleResult',
    'IntervalBreakdown',
]

__version__ = '1.0.0'
