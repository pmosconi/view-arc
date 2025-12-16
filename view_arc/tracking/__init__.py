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

Module Structure:
- dataclasses: Core data structures (ViewerSample, AOI, SessionConfig, etc.)
- validation: Input validation functions
- algorithm: Core tracking algorithm (process_single_sample, compute_attention_seconds)
"""

# Data structures
from view_arc.tracking.dataclasses import (
    AOI,
    AOIResult,
    SAMPLING_ASSUMPTIONS,
    SessionConfig,
    TrackingResult,
    ValidationError,
    ViewerSample,
)

# Validation functions
from view_arc.tracking.validation import (
    SampleInput,
    normalize_sample_input,
    validate_aois,
    validate_tracking_params,
    validate_viewer_samples,
)

# Algorithm functions and result types
from view_arc.tracking.algorithm import (
    AOIIntervalBreakdown,
    SingleSampleResult,
    TrackingResultWithConfig,
    compute_attention_seconds,
    process_single_sample,
)

# Visualization functions
from view_arc.tracking.visualize import (
    create_tracking_animation,
    draw_attention_heatmap,
    draw_attention_labels,
    draw_session_frame,
    draw_viewing_timeline,
    generate_session_replay,
)

__all__ = [
    # Data structures
    "AOI",
    "AOIResult",
    "SAMPLING_ASSUMPTIONS",
    "SessionConfig",
    "TrackingResult",
    "ValidationError",
    "ViewerSample",
    # Validation
    "SampleInput",
    "normalize_sample_input",
    "validate_aois",
    "validate_tracking_params",
    "validate_viewer_samples",
    # Algorithm
    "AOIIntervalBreakdown",
    "SingleSampleResult",
    "TrackingResultWithConfig",
    "compute_attention_seconds",
    "process_single_sample",
    # Visualization
    "create_tracking_animation",
    "draw_attention_heatmap",
    "draw_attention_labels",
    "draw_session_frame",
    "draw_viewing_timeline",
    "generate_session_replay",
]
