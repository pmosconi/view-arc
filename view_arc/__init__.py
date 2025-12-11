"""
View Arc Obstacle Detection
============================

Public API for finding the obstacle with largest visible angular coverage
within a field-of-view arc from a viewer point.
"""

from view_arc.api import find_largest_obstacle, ObstacleResult, IntervalBreakdown
from view_arc.debug import (
    DebugResult,
    ClipResult,
    IntervalDebugInfo,
    log_clipping_stage,
    log_events,
    log_interval_resolution,
    log_coverage_summary,
    log_result,
    format_angle,
    format_point,
    format_polygon,
    setup_debug_logging,
    disable_debug_logging,
)
from view_arc.tracking import (
    ViewerSample,
    AOI,
    AOIResult,
    TrackingResult,
    TrackingResultWithConfig,
    SessionConfig,
    ValidationError,
    SingleSampleResult,
    AOIIntervalBreakdown,
    process_single_sample,
    compute_attention_seconds,
)

__all__ = [
    # Main API
    'find_largest_obstacle',
    'ObstacleResult',
    'IntervalBreakdown',
    # Tracking API
    'ViewerSample',
    'AOI',
    'AOIResult',
    'TrackingResult',
    'TrackingResultWithConfig',
    'SessionConfig',
    'ValidationError',
    'SingleSampleResult',
    'AOIIntervalBreakdown',
    'process_single_sample',
    'compute_attention_seconds',
    # Debug utilities
    'DebugResult',
    'ClipResult',
    'IntervalDebugInfo',
    'log_clipping_stage',
    'log_events',
    'log_interval_resolution',
    'log_coverage_summary',
    'log_result',
    'format_angle',
    'format_point',
    'format_polygon',
    'setup_debug_logging',
    'disable_debug_logging',
]
__version__ = '0.1.0'
