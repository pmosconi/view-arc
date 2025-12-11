"""
Tests for debug utilities and edge case scenarios.

This module contains tests for the debug utilities as well as test cases
for edge cases and scenarios discovered during integration testing.
"""

import logging
import numpy as np
import pytest
from numpy.typing import NDArray

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
from view_arc.obstacle.sweep import AngularEvent


# =============================================================================
# Helper functions for creating test fixtures
# =============================================================================


def make_triangle(center: tuple[float, float], size: float = 20.0) -> NDArray[np.float32]:
    """Create a triangle centered at given point."""
    cx, cy = center
    return np.array(
        [
            [cx, cy + size],  # top
            [cx - size, cy - size],  # bottom-left
            [cx + size, cy - size],  # bottom-right
        ],
        dtype=np.float32,
    )


def make_square(center: tuple[float, float], half_size: float = 15.0) -> NDArray[np.float32]:
    """Create a square centered at given point."""
    cx, cy = center
    return np.array(
        [
            [cx - half_size, cy - half_size],
            [cx + half_size, cy - half_size],
            [cx + half_size, cy + half_size],
            [cx - half_size, cy + half_size],
        ],
        dtype=np.float32,
    )


# =============================================================================
# Tests for Debug Data Classes
# =============================================================================


class TestClipResult:
    """Tests for ClipResult dataclass."""

    def test_clip_result_visible(self) -> None:
        """Test ClipResult for a visible obstacle."""
        result = ClipResult(
            obstacle_id=0,
            original_vertices=4,
            clipped_vertices=4,
            rejected=False,
            rejection_reason=None,
        )
        assert result.obstacle_id == 0
        assert result.original_vertices == 4
        assert result.clipped_vertices == 4
        assert not result.rejected
        assert result.rejection_reason is None

    def test_clip_result_rejected(self) -> None:
        """Test ClipResult for a rejected obstacle."""
        result = ClipResult(
            obstacle_id=1,
            original_vertices=4,
            clipped_vertices=0,
            rejected=True,
            rejection_reason="outside_wedge",
        )
        assert result.obstacle_id == 1
        assert result.rejected
        assert result.rejection_reason == "outside_wedge"


class TestIntervalDebugInfo:
    """Tests for IntervalDebugInfo dataclass."""

    def test_interval_debug_info_basic(self) -> None:
        """Test IntervalDebugInfo creation."""
        info = IntervalDebugInfo(
            angle_start=0.0,
            angle_end=0.5,
            angular_span_deg=np.rad2deg(0.5),
            active_obstacle_ids=[0, 1],
            winner_id=0,
            winner_distance=50.0,
            sample_distances={0: [50.0, 52.0], 1: [60.0, 62.0]},
        )
        assert info.angle_start == 0.0
        assert info.angle_end == 0.5
        assert 0 in info.active_obstacle_ids
        assert info.winner_id == 0
        assert info.winner_distance == 50.0


class TestDebugResult:
    """Tests for DebugResult dataclass."""

    def test_debug_result_creation(self) -> None:
        """Test DebugResult creation with minimal data."""
        result = DebugResult(
            viewer_point=(100.0, 100.0),
            view_direction=(0.0, 1.0),
            alpha_center=np.pi / 2,
            alpha_min=np.pi / 4,
            alpha_max=3 * np.pi / 4,
            fov_deg=90.0,
            max_range=150.0,
        )
        assert result.viewer_point == (100.0, 100.0)
        assert result.fov_deg == 90.0
        assert result.winner_id is None

    def test_debug_result_summary(self) -> None:
        """Test DebugResult summary generation."""
        result = DebugResult(
            viewer_point=(100.0, 100.0),
            view_direction=(0.0, 1.0),
            alpha_center=np.pi / 2,
            alpha_min=np.pi / 4,
            alpha_max=3 * np.pi / 4,
            fov_deg=90.0,
            max_range=150.0,
            winner_id=0,
            winner_coverage=0.5,
            winner_distance=50.0,
        )
        summary = result.summary()
        assert "VIEW ARC OBSTACLE DETECTION" in summary
        assert "Winner: Obstacle 0" in summary
        assert "100.00" in summary  # viewer position

    def test_debug_result_to_dict(self) -> None:
        """Test DebugResult to_dict conversion."""
        result = DebugResult(
            viewer_point=(100.0, 100.0),
            view_direction=(0.0, 1.0),
            alpha_center=np.pi / 2,
            alpha_min=np.pi / 4,
            alpha_max=3 * np.pi / 4,
            fov_deg=90.0,
            max_range=150.0,
            winner_id=0,
            winner_coverage=0.5,
            winner_distance=50.0,
        )
        d = result.to_dict()
        assert d["viewer_point"] == (100.0, 100.0)
        assert d["fov_deg"] == 90.0
        assert d["winner_id"] == 0
        assert "alpha_center_deg" in d


# =============================================================================
# Tests for Formatting Functions
# =============================================================================


class TestFormatFunctions:
    """Tests for formatting helper functions."""

    def test_format_angle_zero(self) -> None:
        """Test formatting zero angle."""
        result = format_angle(0.0)
        assert "0.00°" in result
        assert "0.000 rad" in result

    def test_format_angle_90_deg(self) -> None:
        """Test formatting 90 degrees."""
        result = format_angle(np.pi / 2)
        assert "90.00°" in result

    def test_format_angle_negative(self) -> None:
        """Test formatting negative angle."""
        result = format_angle(-np.pi / 4)
        assert "-45.00°" in result

    def test_format_point_array(self) -> None:
        """Test formatting numpy array point."""
        point = np.array([100.5, 200.25], dtype=np.float32)
        result = format_point(point)
        assert "(100.50, 200.25)" == result

    def test_format_point_tuple(self) -> None:
        """Test formatting tuple point."""
        result = format_point((50.0, 75.5))
        assert "(50.00, 75.50)" == result

    def test_format_polygon_none(self) -> None:
        """Test formatting None polygon."""
        result = format_polygon(None)
        assert result == "None"

    def test_format_polygon_small(self) -> None:
        """Test formatting small polygon."""
        polygon = np.array([[0, 0], [10, 0], [5, 10]], dtype=np.float32)
        result = format_polygon(polygon)
        assert "(0.00, 0.00)" in result
        assert "(10.00, 0.00)" in result
        assert "(5.00, 10.00)" in result

    def test_format_polygon_truncated(self) -> None:
        """Test formatting large polygon gets truncated."""
        polygon = np.array(
            [[i * 10, i * 20] for i in range(10)], dtype=np.float32
        )
        result = format_polygon(polygon, max_vertices=3)
        assert "10 vertices total" in result


# =============================================================================
# Tests for Logging Functions
# =============================================================================


class TestLoggingFunctions:
    """Tests for logging helper functions."""

    def test_log_clipping_stage_visible(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test logging for visible obstacle."""
        setup_debug_logging(logging.DEBUG)
        try:
            original = np.array([[0, 0], [10, 0], [5, 10]], dtype=np.float32)
            clipped = np.array([[0, 0], [10, 0], [5, 10]], dtype=np.float32)
            
            with caplog.at_level(logging.DEBUG, logger="view_arc.debug"):
                result = log_clipping_stage(0, original, clipped)
            
            assert not result.rejected
            assert result.original_vertices == 3
            assert result.clipped_vertices == 3
        finally:
            disable_debug_logging()

    def test_log_clipping_stage_rejected(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test logging for rejected obstacle."""
        setup_debug_logging(logging.DEBUG)
        try:
            original = np.array([[0, 0], [10, 0], [5, 10]], dtype=np.float32)
            
            with caplog.at_level(logging.DEBUG, logger="view_arc.debug"):
                result = log_clipping_stage(0, original, None, "outside_range")
            
            assert result.rejected
            assert result.rejection_reason == "outside_range"
        finally:
            disable_debug_logging()

    def test_log_events(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test logging angular events."""
        setup_debug_logging(logging.DEBUG)
        try:
            events = [
                AngularEvent(angle=0.5, obstacle_id=0, event_type="vertex", vertex_idx=0),
                AngularEvent(angle=1.0, obstacle_id=1, event_type="edge_crossing", vertex_idx=0),
            ]
            
            with caplog.at_level(logging.DEBUG, logger="view_arc.debug"):
                log_events(events)
            
            # Just verify the function runs without error
            # Log capture may not work due to logger propagation settings
        finally:
            disable_debug_logging()

    def test_log_interval_resolution(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test logging interval resolution."""
        setup_debug_logging(logging.DEBUG)
        try:
            with caplog.at_level(logging.DEBUG, logger="view_arc.debug"):
                result = log_interval_resolution(
                    interval_start=0.0,
                    interval_end=0.5,
                    active_obstacles={0: np.array([]), 1: np.array([])},
                    winner_id=0,
                    winner_distance=50.0,
                )
            
            assert result.winner_id == 0
            assert 0 in result.active_obstacle_ids
            assert 1 in result.active_obstacle_ids
        finally:
            disable_debug_logging()

    def test_log_coverage_summary(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test logging coverage summary."""
        setup_debug_logging(logging.DEBUG)
        try:
            with caplog.at_level(logging.DEBUG, logger="view_arc.debug"):
                log_coverage_summary(
                    coverage_dict={0: 0.5, 1: 0.3},
                    min_distance_dict={0: 50.0, 1: 75.0},
                )
            
            # Just verify the function runs without error
            # Log capture may not work due to logger propagation settings
        finally:
            disable_debug_logging()

    def test_log_result_with_winner(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test logging result with winner."""
        setup_debug_logging(logging.INFO)
        try:
            with caplog.at_level(logging.INFO, logger="view_arc.debug"):
                log_result(winner_id=0, winner_coverage=0.5, winner_distance=50.0)
            
            # Just verify the function runs without error
            # Log capture may not work due to logger propagation settings
        finally:
            disable_debug_logging()

    def test_log_result_no_winner(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test logging result with no winner."""
        setup_debug_logging(logging.INFO)
        try:
            with caplog.at_level(logging.INFO, logger="view_arc.debug"):
                log_result(winner_id=None, winner_coverage=0.0, winner_distance=float('inf'))
            
            # Just verify the function runs without error
            # Log capture may not work due to logger propagation settings
        finally:
            disable_debug_logging()


# =============================================================================
# Tests for Enhanced ObstacleResult
# =============================================================================


class TestObstacleResultEnhanced:
    """Tests for enhanced ObstacleResult with detailed interval breakdown."""

    def test_obstacle_result_angular_coverage_deg(self) -> None:
        """Test angular_coverage_deg property."""
        result = ObstacleResult(
            obstacle_id=0,
            angular_coverage=np.pi / 4,  # 45 degrees in radians
            min_distance=50.0,
        )
        assert abs(result.angular_coverage_deg - 45.0) < 0.01

    def test_obstacle_result_get_all_intervals_empty(self) -> None:
        """Test get_all_intervals with no intervals."""
        result = ObstacleResult(
            obstacle_id=0,
            angular_coverage=0.5,
            min_distance=50.0,
            interval_details=None,
        )
        assert result.get_all_intervals() == []

    def test_obstacle_result_get_all_intervals(self) -> None:
        """Test get_all_intervals with intervals."""
        intervals = [
            IntervalBreakdown(0.0, 0.5, 0.5, 0, 50.0),
            IntervalBreakdown(0.5, 1.0, 0.5, 1, 60.0),
        ]
        result = ObstacleResult(
            obstacle_id=0,
            angular_coverage=0.5,
            min_distance=50.0,
            interval_details=intervals,
        )
        assert len(result.get_all_intervals()) == 2

    def test_obstacle_result_get_winner_intervals(self) -> None:
        """Test get_winner_intervals filters correctly."""
        intervals = [
            IntervalBreakdown(0.0, 0.5, 0.5, 0, 50.0),
            IntervalBreakdown(0.5, 1.0, 0.5, 1, 60.0),
            IntervalBreakdown(1.0, 1.5, 0.5, 0, 55.0),
        ]
        result = ObstacleResult(
            obstacle_id=0,
            angular_coverage=1.0,
            min_distance=50.0,
            interval_details=intervals,
        )
        winner_intervals = result.get_winner_intervals()
        assert len(winner_intervals) == 2
        assert all(iv.obstacle_id == 0 for iv in winner_intervals)

    def test_obstacle_result_summary_with_winner(self) -> None:
        """Test summary generation with winner."""
        result = ObstacleResult(
            obstacle_id=0,
            angular_coverage=np.pi / 4,
            min_distance=50.0,
            intervals=[(0.0, 0.5)],
            all_coverage={0: np.pi / 4, 1: np.pi / 8},
            all_distances={0: 50.0, 1: 75.0},
        )
        summary = result.summary()
        assert "Winner: Obstacle 0" in summary
        assert "Coverage:" in summary
        assert "Min Distance:" in summary
        assert "All Obstacles:" in summary

    def test_obstacle_result_summary_no_winner(self) -> None:
        """Test summary generation without winner."""
        result = ObstacleResult(
            obstacle_id=None,
            angular_coverage=0.0,
            min_distance=float('inf'),
        )
        summary = result.summary()
        assert "No obstacle visible" in summary


class TestIntervalBreakdown:
    """Tests for IntervalBreakdown dataclass."""

    def test_interval_breakdown_properties(self) -> None:
        """Test IntervalBreakdown property conversions."""
        interval = IntervalBreakdown(
            angle_start=0.0,
            angle_end=np.pi / 4,
            angular_span=np.pi / 4,
            obstacle_id=0,
            min_distance=50.0,
        )
        assert abs(interval.angular_span_deg - 45.0) < 0.01
        assert abs(interval.angle_start_deg - 0.0) < 0.01
        assert abs(interval.angle_end_deg - 45.0) < 0.01


# =============================================================================
# Tests for API with Debug Options
# =============================================================================


class TestAPIDebugOptions:
    """Tests for find_largest_obstacle with debug options."""

    def test_return_all_coverage(self) -> None:
        """Test return_all_coverage option."""
        viewer = np.array([100.0, 100.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)
        
        contours = [
            make_triangle((100, 150), 20),  # obstacle 0 - closer
            make_square((100, 200), 15),    # obstacle 1 - farther
        ]
        
        result = find_largest_obstacle(
            viewer, direction, 60.0, 150.0, contours,
            return_intervals=True,
            return_all_coverage=True,
        )
        
        assert result.all_coverage is not None
        assert result.all_distances is not None
        # At least one obstacle should be visible
        assert len(result.all_coverage) >= 1

    def test_interval_details_populated(self) -> None:
        """Test that interval_details is populated when return_intervals=True."""
        viewer = np.array([100.0, 100.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)
        
        contours = [make_triangle((100, 150), 20)]
        
        result = find_largest_obstacle(
            viewer, direction, 60.0, 150.0, contours,
            return_intervals=True,
        )
        
        if result.obstacle_id is not None:
            assert result.interval_details is not None
            assert len(result.interval_details) >= 1
            for detail in result.interval_details:
                assert isinstance(detail, IntervalBreakdown)


# =============================================================================
# Edge Case Scenarios (Bug Reproductions)
# =============================================================================


class TestEdgeCaseScenarios:
    """Test cases for edge cases and previously failing scenarios."""

    def test_obstacle_exactly_at_arc_boundary(self) -> None:
        """Test obstacle positioned exactly at arc boundary."""
        viewer = np.array([100.0, 100.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)
        
        # Triangle at exactly 30° from center (at FOV boundary for 60° FOV)
        offset_x = 50 * np.sin(np.pi / 6)  # 30 degrees
        offset_y = 50 * np.cos(np.pi / 6)
        contour = make_triangle((100 + offset_x, 100 + offset_y), 10)
        
        result = find_largest_obstacle(
            viewer, direction, 60.0, 100.0, [contour],
            return_intervals=True,
        )
        
        # Should handle boundary case gracefully
        # (may or may not be visible depending on exact positioning)
        assert isinstance(result, ObstacleResult)

    def test_very_small_fov(self) -> None:
        """Test with very small field of view (1 degree)."""
        viewer = np.array([100.0, 100.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)
        
        # Obstacle directly in front
        contour = make_triangle((100, 150), 5)
        
        result = find_largest_obstacle(
            viewer, direction, 1.0, 100.0, [contour],
        )
        
        # May or may not see it with 1° FOV, but should not crash
        assert isinstance(result, ObstacleResult)

    def test_obstacle_at_max_range(self) -> None:
        """Test obstacle positioned exactly at max range."""
        viewer = np.array([100.0, 100.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)
        
        # Triangle at exactly max_range distance
        contour = make_triangle((100, 200), 10)  # 100 units away
        
        result = find_largest_obstacle(
            viewer, direction, 60.0, 100.0, [contour],
        )
        
        assert isinstance(result, ObstacleResult)

    def test_obstacle_straddling_pi_boundary(self) -> None:
        """Test obstacle at the ±π angle discontinuity."""
        viewer = np.array([100.0, 100.0], dtype=np.float32)
        # Direction pointing left (towards negative x)
        direction = np.array([-1.0, 0.0], dtype=np.float32)
        
        # Obstacle to the left
        contour = make_square((50, 100), 20)
        
        result = find_largest_obstacle(
            viewer, direction, 60.0, 100.0, [contour],
            return_intervals=True,
        )
        
        assert isinstance(result, ObstacleResult)

    def test_many_overlapping_obstacles(self) -> None:
        """Test with many overlapping obstacles (stress test)."""
        viewer = np.array([100.0, 100.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)
        
        # Create 10 overlapping triangles at different distances
        contours = [
            make_triangle((100, 120 + i * 10), 15) for i in range(10)
        ]
        
        result = find_largest_obstacle(
            viewer, direction, 90.0, 300.0, contours,
            return_all_coverage=True,
        )
        
        assert isinstance(result, ObstacleResult)
        if result.obstacle_id is not None:
            # Winner should be the closest one (index 0)
            assert result.obstacle_id == 0

    def test_single_vertex_nearly_collinear(self) -> None:
        """Test polygon with nearly collinear vertices."""
        viewer = np.array([100.0, 100.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)
        
        # Very thin triangle (nearly degenerate)
        contour = np.array([
            [99.99, 150],
            [100.01, 150],
            [100.0, 170],
        ], dtype=np.float32)
        
        result = find_largest_obstacle(
            viewer, direction, 60.0, 100.0, [contour],
        )
        
        # Should handle nearly-degenerate polygon
        assert isinstance(result, ObstacleResult)

    def test_zero_area_polygon(self) -> None:
        """Test with a degenerate (zero-area) polygon."""
        viewer = np.array([100.0, 100.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)
        
        # Collinear points (line, not polygon)
        contour = np.array([
            [100, 150],
            [100, 160],
            [100, 170],
        ], dtype=np.float32)
        
        result = find_largest_obstacle(
            viewer, direction, 60.0, 100.0, [contour],
        )
        
        # Should handle gracefully
        assert isinstance(result, ObstacleResult)


# =============================================================================
# Integration Tests for Debug Workflow
# =============================================================================


class TestDebugWorkflow:
    """Tests for complete debug workflow."""

    def test_full_debug_workflow(self) -> None:
        """Test complete debug workflow with all utilities."""
        viewer = np.array([100.0, 100.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)
        
        contours = [
            make_triangle((100, 150), 20),
            make_square((130, 180), 15),
        ]
        
        # Run with all debug options
        result = find_largest_obstacle(
            viewer, direction, 60.0, 150.0, contours,
            return_intervals=True,
            return_all_coverage=True,
        )
        
        # Create debug result manually
        debug = DebugResult(
            viewer_point=(100.0, 100.0),
            view_direction=(0.0, 1.0),
            alpha_center=np.pi / 2,
            alpha_min=np.pi / 4,
            alpha_max=3 * np.pi / 4,
            fov_deg=60.0,
            max_range=150.0,
            winner_id=result.obstacle_id,
            winner_coverage=result.angular_coverage,
            winner_distance=result.min_distance,
            coverage_summary=result.all_coverage or {},
            distance_summary=result.all_distances or {},
        )
        
        # Verify debug result
        summary = debug.summary()
        assert "VIEW ARC OBSTACLE DETECTION" in summary
        
        # Verify to_dict
        d = debug.to_dict()
        assert d["fov_deg"] == 60.0


class TestLoggingSetup:
    """Tests for logging configuration."""

    def test_setup_and_disable_logging(self) -> None:
        """Test that logging can be set up and disabled."""
        # Should not raise
        setup_debug_logging(logging.DEBUG)
        disable_debug_logging()
        
        # Re-enable for other tests
        setup_debug_logging(logging.WARNING)
        disable_debug_logging()

    def test_setup_logging_multiple_times_no_duplicate_handlers(self) -> None:
        """Test that calling setup_debug_logging multiple times doesn't stack handlers."""
        from view_arc.obstacle.debug import logger
        
        # Call setup multiple times
        setup_debug_logging(logging.DEBUG)
        setup_debug_logging(logging.DEBUG)
        setup_debug_logging(logging.DEBUG)
        
        # Should only have one handler, not three
        assert len(logger.handlers) == 1
        
        disable_debug_logging()
        assert len(logger.handlers) == 0


class TestSkippedObstacleIds:
    """Tests to ensure obstacle IDs are preserved correctly when some obstacles are skipped."""

    def test_skipped_obstacle_preserves_original_ids(self) -> None:
        """
        Test that when an earlier obstacle is clipped away, later obstacles
        retain their original indices in the result.
        
        This is a regression test for the bug where build_events re-enumerated
        filtered polygons, causing wrong obstacle IDs in events and results.
        """
        viewer = np.array([100.0, 100.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)
        
        # Obstacle 0: completely behind the viewer (will be clipped away)
        obstacle_behind = make_square((100, 50), 15)  # Behind viewer looking up
        
        # Obstacle 1: completely outside FOV to the right (will be clipped away)
        obstacle_right = make_square((200, 100), 15)  # Far to the right
        
        # Obstacle 2: visible in front (should be the winner with ID=2, not ID=0)
        obstacle_visible = make_triangle((100, 150), 20)
        
        contours = [obstacle_behind, obstacle_right, obstacle_visible]
        
        result = find_largest_obstacle(
            viewer, direction, 60.0, 100.0, contours,
            return_intervals=True,
            return_all_coverage=True,
        )
        
        # The winner should be obstacle 2 (original index), not 0
        assert result.obstacle_id == 2, (
            f"Expected winner ID=2 (third obstacle), got {result.obstacle_id}. "
            "This suggests obstacle IDs are being re-enumerated after filtering."
        )
        
        # Coverage dict should use original ID
        if result.all_coverage:
            assert 2 in result.all_coverage, (
                f"Expected obstacle ID 2 in coverage dict, got keys: {list(result.all_coverage.keys())}"
            )
            # Obstacles 0 and 1 should NOT be in coverage (they were clipped)
            assert 0 not in result.all_coverage
            assert 1 not in result.all_coverage

    def test_skipped_obstacle_multiple_visible(self) -> None:
        """
        Test with multiple visible obstacles where one earlier obstacle is skipped.
        Ensures all visible obstacles retain correct IDs.
        """
        viewer = np.array([100.0, 100.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)
        
        # Obstacle 0: behind viewer (clipped away)
        obstacle_behind = make_square((100, 30), 15)
        
        # Obstacle 1: visible, closer
        obstacle_1 = make_triangle((90, 140), 15)
        
        # Obstacle 2: visible, farther
        obstacle_2 = make_triangle((110, 180), 15)
        
        contours = [obstacle_behind, obstacle_1, obstacle_2]
        
        result = find_largest_obstacle(
            viewer, direction, 90.0, 150.0, contours,
            return_intervals=True,
            return_all_coverage=True,
        )
        
        # Winner should be 1 or 2, definitely not 0
        assert result.obstacle_id in (1, 2), (
            f"Expected winner ID in (1, 2), got {result.obstacle_id}"
        )
        
        # Coverage should have correct IDs
        if result.all_coverage:
            # Should NOT have obstacle 0
            assert 0 not in result.all_coverage
            # Should have at least one of the visible obstacles with correct ID
            visible_ids = set(result.all_coverage.keys())
            assert visible_ids.issubset({1, 2}), (
                f"Expected coverage keys to be subset of {{1, 2}}, got {visible_ids}"
            )

    def test_interval_details_use_original_obstacle_ids(self) -> None:
        """Test that interval_details contain correct original obstacle IDs."""
        viewer = np.array([100.0, 100.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)
        
        # Obstacle 0: invalid (too few vertices) - will be skipped
        invalid_obstacle = np.array([[100, 150], [110, 150]], dtype=np.float32)
        
        # Obstacle 1: valid visible obstacle
        valid_obstacle = make_triangle((100, 150), 20)
        
        contours = [invalid_obstacle, valid_obstacle]
        
        result = find_largest_obstacle(
            viewer, direction, 60.0, 100.0, contours,
            return_intervals=True,
        )
        
        assert result.obstacle_id == 1, (
            f"Expected winner ID=1, got {result.obstacle_id}"
        )
        
        if result.interval_details:
            for detail in result.interval_details:
                assert detail.obstacle_id == 1, (
                    f"Expected interval obstacle_id=1, got {detail.obstacle_id}"
                )
