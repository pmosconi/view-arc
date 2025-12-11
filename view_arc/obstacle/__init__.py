"""
Obstacle Detection Module
=========================

Low-level geometry, clipping, angular sweep, and visualization utilities
for finding the obstacle with largest visible angular coverage within a
field-of-view arc from a viewer point.

This module is used internally by the tracking API and can be accessed
directly for advanced use cases.
"""

from view_arc.obstacle.api import find_largest_obstacle, ObstacleResult, IntervalBreakdown
from view_arc.obstacle.debug import (
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

__all__ = [
    # Main API
    'find_largest_obstacle',
    'ObstacleResult',
    'IntervalBreakdown',
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
