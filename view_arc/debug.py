"""
Debug utilities for logging intermediate results and detailed analysis.

This module provides helper functions for inspecting the obstacle detection
algorithm's internal state during development and troubleshooting.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import logging
import numpy as np
from numpy.typing import NDArray

from view_arc.sweep import AngularEvent, IntervalResult


# Configure module-level logger
logger = logging.getLogger("view_arc.debug")


@dataclass
class ClipResult:
    """
    Debug information about clipping stage for a single obstacle.
    
    Attributes:
        obstacle_id: Index of the obstacle
        original_vertices: Number of vertices before clipping
        clipped_vertices: Number of vertices after clipping (0 if rejected)
        rejected: Whether the obstacle was completely clipped away
        rejection_reason: If rejected, the reason (e.g., 'outside_wedge', 'outside_range')
    """
    obstacle_id: int
    original_vertices: int
    clipped_vertices: int
    rejected: bool
    rejection_reason: Optional[str] = None


@dataclass
class IntervalDebugInfo:
    """
    Detailed debug information for a single angular interval.
    
    Attributes:
        angle_start: Start angle in radians
        angle_end: End angle in radians
        angular_span_deg: Angular span in degrees
        active_obstacle_ids: List of obstacle IDs active in this interval
        winner_id: ID of the obstacle that won this interval
        winner_distance: Minimum distance to the winning obstacle
        sample_distances: Dict mapping obstacle_id to list of sample distances
    """
    angle_start: float
    angle_end: float
    angular_span_deg: float
    active_obstacle_ids: List[int]
    winner_id: Optional[int]
    winner_distance: float
    sample_distances: Dict[int, List[float]] = field(default_factory=dict)


@dataclass
class DebugResult:
    """
    Comprehensive debug output from the obstacle detection algorithm.
    
    This class provides detailed information about all stages of the
    algorithm for debugging and validation purposes.
    
    Attributes:
        viewer_point: The viewer position
        view_direction: The normalized view direction vector
        alpha_center: Central angle of the view arc (radians)
        alpha_min: Start angle of the view arc (radians)
        alpha_max: End angle of the view arc (radians)
        fov_deg: Field of view in degrees
        max_range: Maximum sensing range
        clip_results: List of clipping results for each obstacle
        events: List of angular events generated
        intervals: List of detailed interval debug info
        coverage_summary: Dict mapping obstacle_id to coverage in radians
        distance_summary: Dict mapping obstacle_id to minimum distance
        winner_id: Final winning obstacle ID (None if no obstacle visible)
        winner_coverage: Angular coverage of the winner
        winner_distance: Minimum distance to the winner
        processing_time_ms: Time taken for processing (optional)
    """
    viewer_point: Tuple[float, float]
    view_direction: Tuple[float, float]
    alpha_center: float
    alpha_min: float
    alpha_max: float
    fov_deg: float
    max_range: float
    clip_results: List[ClipResult] = field(default_factory=list)
    events: List[AngularEvent] = field(default_factory=list)
    intervals: List[IntervalDebugInfo] = field(default_factory=list)
    coverage_summary: Dict[int, float] = field(default_factory=dict)
    distance_summary: Dict[int, float] = field(default_factory=dict)
    winner_id: Optional[int] = None
    winner_coverage: float = 0.0
    winner_distance: float = float('inf')
    processing_time_ms: Optional[float] = None

    def summary(self) -> str:
        """Generate a human-readable summary of the debug result."""
        lines = [
            "=" * 60,
            "VIEW ARC OBSTACLE DETECTION - DEBUG SUMMARY",
            "=" * 60,
            "",
            f"Viewer Position: ({self.viewer_point[0]:.2f}, {self.viewer_point[1]:.2f})",
            f"View Direction: ({self.view_direction[0]:.3f}, {self.view_direction[1]:.3f})",
            f"Central Angle: {np.rad2deg(self.alpha_center):.1f}°",
            f"Arc Range: [{np.rad2deg(self.alpha_min):.1f}°, {np.rad2deg(self.alpha_max):.1f}°]",
            f"FOV: {self.fov_deg:.1f}°, Max Range: {self.max_range:.1f}",
            "",
            "-" * 40,
            "CLIPPING RESULTS",
            "-" * 40,
        ]
        
        for cr in self.clip_results:
            status = "REJECTED" if cr.rejected else "VISIBLE"
            reason = f" ({cr.rejection_reason})" if cr.rejection_reason else ""
            lines.append(
                f"  Obstacle {cr.obstacle_id}: {cr.original_vertices} → "
                f"{cr.clipped_vertices} vertices [{status}{reason}]"
            )
        
        visible_count = sum(1 for cr in self.clip_results if not cr.rejected)
        lines.extend([
            f"  Total: {len(self.clip_results)} obstacles, {visible_count} visible",
            "",
            "-" * 40,
            f"ANGULAR EVENTS ({len(self.events)} total)",
            "-" * 40,
        ])
        
        for event in self.events[:10]:  # Limit output
            lines.append(
                f"  {np.rad2deg(event.angle):7.2f}° | Obstacle {event.obstacle_id} | {event.event_type}"
            )
        if len(self.events) > 10:
            lines.append(f"  ... and {len(self.events) - 10} more events")
        
        lines.extend([
            "",
            "-" * 40,
            f"INTERVAL BREAKDOWN ({len(self.intervals)} intervals)",
            "-" * 40,
        ])
        
        for iv in self.intervals[:10]:  # Limit output
            active_str = ", ".join(str(x) for x in iv.active_obstacle_ids)
            winner_str = f"Winner: {iv.winner_id}" if iv.winner_id is not None else "No winner"
            lines.append(
                f"  [{np.rad2deg(iv.angle_start):7.2f}° → {np.rad2deg(iv.angle_end):7.2f}°] "
                f"({iv.angular_span_deg:.2f}°) Active: [{active_str}] | {winner_str}"
            )
        if len(self.intervals) > 10:
            lines.append(f"  ... and {len(self.intervals) - 10} more intervals")
        
        lines.extend([
            "",
            "-" * 40,
            "COVERAGE SUMMARY",
            "-" * 40,
        ])
        
        for obstacle_id in sorted(self.coverage_summary.keys()):
            coverage_deg = np.rad2deg(self.coverage_summary[obstacle_id])
            distance = self.distance_summary.get(obstacle_id, float('inf'))
            lines.append(
                f"  Obstacle {obstacle_id}: {coverage_deg:.2f}° coverage, "
                f"min distance = {distance:.2f}"
            )
        
        lines.extend([
            "",
            "=" * 60,
            "RESULT",
            "=" * 60,
        ])
        
        if self.winner_id is not None:
            lines.extend([
                f"  Winner: Obstacle {self.winner_id}",
                f"  Coverage: {np.rad2deg(self.winner_coverage):.2f}°",
                f"  Min Distance: {self.winner_distance:.2f}",
            ])
        else:
            lines.append("  No obstacle visible in the view arc")
        
        if self.processing_time_ms is not None:
            lines.append(f"  Processing Time: {self.processing_time_ms:.2f} ms")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary for JSON serialization or logging."""
        return {
            "viewer_point": self.viewer_point,
            "view_direction": self.view_direction,
            "alpha_center_deg": np.rad2deg(self.alpha_center),
            "alpha_min_deg": np.rad2deg(self.alpha_min),
            "alpha_max_deg": np.rad2deg(self.alpha_max),
            "fov_deg": self.fov_deg,
            "max_range": self.max_range,
            "clip_results": [
                {
                    "obstacle_id": cr.obstacle_id,
                    "original_vertices": cr.original_vertices,
                    "clipped_vertices": cr.clipped_vertices,
                    "rejected": cr.rejected,
                    "rejection_reason": cr.rejection_reason,
                }
                for cr in self.clip_results
            ],
            "num_events": len(self.events),
            "num_intervals": len(self.intervals),
            "coverage_summary_deg": {
                k: np.rad2deg(v) for k, v in self.coverage_summary.items()
            },
            "distance_summary": dict(self.distance_summary),
            "winner_id": self.winner_id,
            "winner_coverage_deg": np.rad2deg(self.winner_coverage),
            "winner_distance": self.winner_distance,
            "processing_time_ms": self.processing_time_ms,
        }


def log_clipping_stage(
    obstacle_id: int,
    original_contour: NDArray[np.float32],
    clipped_polygon: Optional[NDArray[np.float32]],
    rejection_reason: Optional[str] = None,
    level: int = logging.DEBUG,
) -> ClipResult:
    """
    Log clipping stage results for a single obstacle.
    
    Parameters:
        obstacle_id: Index of the obstacle
        original_contour: Original contour before clipping
        clipped_polygon: Clipped polygon (None if completely rejected)
        rejection_reason: Reason for rejection (if applicable)
        level: Logging level
        
    Returns:
        ClipResult containing the clipping information
    """
    original_vertices = len(original_contour) if original_contour is not None else 0
    clipped_vertices = len(clipped_polygon) if clipped_polygon is not None else 0
    rejected = clipped_polygon is None or clipped_vertices < 3
    
    result = ClipResult(
        obstacle_id=obstacle_id,
        original_vertices=original_vertices,
        clipped_vertices=clipped_vertices,
        rejected=rejected,
        rejection_reason=rejection_reason if rejected else None,
    )
    
    if rejected:
        logger.log(
            level,
            f"Obstacle {obstacle_id}: REJECTED - {original_vertices} vertices, "
            f"reason: {rejection_reason or 'outside_wedge'}",
        )
    else:
        logger.log(
            level,
            f"Obstacle {obstacle_id}: CLIPPED - {original_vertices} → {clipped_vertices} vertices",
        )
    
    return result


def log_events(events: List[AngularEvent], level: int = logging.DEBUG) -> None:
    """
    Log all angular events.
    
    Parameters:
        events: List of angular events
        level: Logging level
    """
    logger.log(level, f"Generated {len(events)} angular events:")
    for event in events:
        logger.log(
            level,
            f"  Angle: {np.rad2deg(event.angle):.2f}°, "
            f"Obstacle: {event.obstacle_id}, Type: {event.event_type}",
        )


def log_interval_resolution(
    interval_start: float,
    interval_end: float,
    active_obstacles: Dict[int, Any],
    winner_id: Optional[int],
    winner_distance: float,
    sample_distances: Optional[Dict[int, List[float]]] = None,
    level: int = logging.DEBUG,
) -> IntervalDebugInfo:
    """
    Log interval resolution results.
    
    Parameters:
        interval_start: Start angle of interval
        interval_end: End angle of interval
        active_obstacles: Dict of active obstacle IDs to their edge arrays
        winner_id: ID of the winning obstacle
        winner_distance: Minimum distance to the winner
        sample_distances: Optional dict of sample distances per obstacle
        level: Logging level
        
    Returns:
        IntervalDebugInfo containing the resolution information
    """
    angular_span = interval_end - interval_start
    angular_span_deg = np.rad2deg(angular_span)
    active_ids = list(active_obstacles.keys())
    
    result = IntervalDebugInfo(
        angle_start=interval_start,
        angle_end=interval_end,
        angular_span_deg=angular_span_deg,
        active_obstacle_ids=active_ids,
        winner_id=winner_id,
        winner_distance=winner_distance,
        sample_distances=sample_distances or {},
    )
    
    active_str = ", ".join(str(x) for x in active_ids)
    winner_str = f"Winner: {winner_id} @ {winner_distance:.2f}" if winner_id is not None else "No winner"
    
    logger.log(
        level,
        f"Interval [{np.rad2deg(interval_start):.2f}°, {np.rad2deg(interval_end):.2f}°] "
        f"({angular_span_deg:.2f}°): Active=[{active_str}], {winner_str}",
    )
    
    return result


def log_coverage_summary(
    coverage_dict: Dict[int, float],
    min_distance_dict: Dict[int, float],
    level: int = logging.DEBUG,
) -> None:
    """
    Log coverage summary for all obstacles.
    
    Parameters:
        coverage_dict: Dict mapping obstacle_id to coverage in radians
        min_distance_dict: Dict mapping obstacle_id to minimum distance
        level: Logging level
    """
    logger.log(level, "Coverage Summary:")
    for obstacle_id in sorted(coverage_dict.keys()):
        coverage_deg = np.rad2deg(coverage_dict[obstacle_id])
        distance = min_distance_dict.get(obstacle_id, float('inf'))
        logger.log(
            level,
            f"  Obstacle {obstacle_id}: {coverage_deg:.2f}° coverage, "
            f"min_distance={distance:.2f}",
        )


def log_result(
    winner_id: Optional[int],
    winner_coverage: float,
    winner_distance: float,
    level: int = logging.INFO,
) -> None:
    """
    Log the final result.
    
    Parameters:
        winner_id: Winning obstacle ID (None if no obstacle visible)
        winner_coverage: Angular coverage of the winner
        winner_distance: Minimum distance to the winner
        level: Logging level
    """
    if winner_id is not None:
        logger.log(
            level,
            f"RESULT: Obstacle {winner_id} wins with {np.rad2deg(winner_coverage):.2f}° "
            f"coverage, min_distance={winner_distance:.2f}",
        )
    else:
        logger.log(level, "RESULT: No obstacle visible in the view arc")


def format_angle(radians: float, precision: int = 2) -> str:
    """
    Format an angle in radians as a human-readable string.
    
    Parameters:
        radians: Angle in radians
        precision: Decimal precision for degrees
        
    Returns:
        Formatted string like "45.00° (0.785 rad)"
    """
    degrees = np.rad2deg(radians)
    return f"{degrees:.{precision}f}° ({radians:.3f} rad)"


def format_point(point: NDArray[np.float32] | Tuple[float, float], precision: int = 2) -> str:
    """
    Format a 2D point as a human-readable string.
    
    Parameters:
        point: (x, y) coordinates
        precision: Decimal precision
        
    Returns:
        Formatted string like "(100.00, 200.00)"
    """
    if isinstance(point, np.ndarray):
        return f"({point[0]:.{precision}f}, {point[1]:.{precision}f})"
    return f"({point[0]:.{precision}f}, {point[1]:.{precision}f})"


def format_polygon(polygon: Optional[NDArray[np.float32]], max_vertices: int = 5) -> str:
    """
    Format a polygon as a human-readable string.
    
    Parameters:
        polygon: Array of (N, 2) vertices
        max_vertices: Maximum vertices to show before truncating
        
    Returns:
        Formatted string representation
    """
    if polygon is None:
        return "None"
    
    n_vertices = len(polygon)
    if n_vertices <= max_vertices:
        points = [format_point(p) for p in polygon]
        return f"[{', '.join(points)}]"
    else:
        points = [format_point(p) for p in polygon[:max_vertices]]
        return f"[{', '.join(points)}, ... ({n_vertices} vertices total)]"


def setup_debug_logging(level: int = logging.DEBUG) -> None:
    """
    Set up debug logging for the view_arc module.
    
    This configures the view_arc.debug logger with console output
    at the specified level.
    
    Parameters:
        level: Logging level (default: DEBUG)
    """
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False


def disable_debug_logging() -> None:
    """Disable debug logging for the view_arc module."""
    logger.setLevel(logging.CRITICAL)
    logger.handlers.clear()
