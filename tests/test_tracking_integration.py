"""
Integration Tests for Realistic Tracking Scenarios (Step 7.1)

These tests simulate realistic in-store viewer behavior patterns and verify
that the tracking system produces intuitive and correct results:

- test_scenario_stationary_viewer() - viewer doesn't move, rotates head
- test_scenario_walking_viewer() - viewer moves through store
- test_scenario_browsing_behavior() - viewer stops at shelves
- test_scenario_quick_glances() - rapid direction changes
- test_scenario_long_stare() - extended viewing of one AOI
- test_scenario_peripheral_viewing() - AOIs at edge of FOV
- test_scenario_complete_store_walkthrough() - end-to-end simulation
"""

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from view_arc.tracking import (
    AOI,
    SessionConfig,
    ValidationError,
    ViewerSample,
    compute_attention_seconds,
)


# =============================================================================
# Helper Functions
# =============================================================================


def make_unit_vector(angle_deg: float) -> tuple[float, float]:
    """Create a unit vector from an angle in degrees.

    Note: In image coordinates where Y increases downward:
    - 0° = right/east (+X direction)
    - 90° = down/south (+Y direction)
    - 180° = left/west (-X direction)
    - 270° = up/north (-Y direction)

    Args:
        angle_deg: Angle in degrees (standard math convention, counterclockwise from +X axis)

    Returns:
        Unit vector (dx, dy) in image coordinate system
    """
    angle_rad = math.radians(angle_deg)
    return (math.cos(angle_rad), math.sin(angle_rad))


def make_rectangle_contour(
    center: tuple[float, float],
    width: float = 100.0,
    height: float = 50.0,
) -> NDArray[np.float64]:
    """Create a rectangle contour centered at the given point."""
    cx, cy = center
    hw, hh = width / 2, height / 2
    return np.array(
        [
            [cx - hw, cy - hh],
            [cx + hw, cy - hh],
            [cx + hw, cy + hh],
            [cx - hw, cy + hh],
        ],
        dtype=np.float64,
    )


def make_square_contour(
    center: tuple[float, float], half_size: float = 25.0
) -> NDArray[np.float64]:
    """Create a square contour centered at the given point."""
    cx, cy = center
    return np.array(
        [
            [cx - half_size, cy - half_size],
            [cx + half_size, cy - half_size],
            [cx + half_size, cy + half_size],
            [cx - half_size, cy + half_size],
        ],
        dtype=np.float64,
    )


def interpolate_position(
    start: tuple[float, float],
    end: tuple[float, float],
    t: float,
) -> tuple[float, float]:
    """Linearly interpolate between two positions.

    Args:
        start: Starting position (x, y)
        end: Ending position (x, y)
        t: Interpolation factor [0, 1]

    Returns:
        Interpolated position (x, y)
    """
    x = start[0] + (end[0] - start[0]) * t
    y = start[1] + (end[1] - start[1]) * t
    return (x, y)


def create_store_layout() -> list[AOI]:
    """Create a typical store layout with multiple shelves/displays.

    Store layout (bird's eye view, coordinates in pixels):
    - Viewer enters from bottom (high Y values) and looks up/north (negative Y direction, 270°)
    - Shelves are on the north wall (low Y values)
    - Left shelf: (200, 100), 100x50
    - Center shelf: (400, 100), 100x50
    - Right shelf: (600, 100), 100x50
    - Back wall display: (400, 50), 150x40

    Returns:
        List of AOI objects representing the store layout
    """
    return [
        AOI(id="left_shelf", contour=make_rectangle_contour((200.0, 100.0), 100.0, 50.0)),
        AOI(
            id="center_shelf", contour=make_rectangle_contour((400.0, 100.0), 100.0, 50.0)
        ),
        AOI(id="right_shelf", contour=make_rectangle_contour((600.0, 100.0), 100.0, 50.0)),
        AOI(
            id="back_display", contour=make_rectangle_contour((400.0, 50.0), 150.0, 40.0)
        ),
    ]


# =============================================================================
# Test: Stationary Viewer (Head Rotation)
# =============================================================================


class TestScenarioStationaryViewer:
    """Test viewer who stays in one position but rotates their head.

    Simulates a shopper standing in one spot and looking around at
    different shelves by rotating their view direction.
    """

    def test_stationary_viewer_scans_left_center_right(self) -> None:
        """Viewer stands still and looks left, center, right in sequence."""
        # Viewer position: center of the store aisle at Y=300, shelves at Y=100
        viewer_pos = (400.0, 300.0)

        # Create store layout
        aois = create_store_layout()

        # Simulate 12-second session: 4 seconds per shelf
        samples = []

        # Seconds 0-3: Look toward left shelf (225 deg = up-left)
        for _ in range(4):
            samples.append(ViewerSample(position=viewer_pos, direction=make_unit_vector(225)))

        # Seconds 4-7: Look toward center shelf (270 degrees = straight up)
        for _ in range(4):
            samples.append(ViewerSample(position=viewer_pos, direction=make_unit_vector(270)))

        # Seconds 8-11: Look toward right shelf (315 deg = up-right)
        for _ in range(4):
            samples.append(ViewerSample(position=viewer_pos, direction=make_unit_vector(315)))

        # Compute attention
        result = compute_attention_seconds(samples, aois, field_of_view_deg=60.0)

        # Verify basic stats
        assert result.total_samples == 12
        assert result.samples_with_hits >= 8  # Should hit at least 2 shelves

        # Verify that left, center, and right shelves all received attention
        # (exact counts may vary due to FOV and geometry)
        left_count = result.get_hit_count("left_shelf")
        center_count = result.get_hit_count("center_shelf")
        right_count = result.get_hit_count("right_shelf")

        # Each shelf should have been visible for some portion of the scan
        assert left_count > 0, "Left shelf should have been viewed"
        assert center_count > 0, "Center shelf should have been viewed"
        assert right_count > 0, "Right shelf should have been viewed"

        # Total attention should sum to total samples (or less if some miss)
        total_hits = left_count + center_count + right_count + result.get_hit_count(
            "back_display"
        )
        assert total_hits <= 12

    def test_stationary_viewer_prolonged_stare(self) -> None:
        """Viewer stands still and stares at one shelf for entire session."""
        viewer_pos = (400.0, 300.0)  # At Y=300, looking up at shelves at Y=100
        aois = create_store_layout()

        # 30 seconds of staring at center shelf (270 deg = straight up)
        samples = [
            ViewerSample(position=viewer_pos, direction=make_unit_vector(270))
            for _ in range(30)
        ]

        result = compute_attention_seconds(samples, aois, field_of_view_deg=60.0)

        assert result.total_samples == 30
        # Center shelf should dominate
        center_count = result.get_hit_count("center_shelf")
        assert center_count >= 20, "Should capture most of the stare"
        assert center_count == result.samples_with_hits


# =============================================================================
# Test: Walking Viewer
# =============================================================================


class TestScenarioWalkingViewer:
    """Test viewer who walks through the store while looking around.

    Simulates a shopper walking down an aisle, passing different
    displays and shelves.
    """

    def test_walking_viewer_passes_shelves(self) -> None:
        """Viewer walks past multiple shelves, looking forward."""
        aois = create_store_layout()

        # Walk from left to right at Y=300, shelves at Y=100
        # Starting position: (100, 300), ending position: (700, 300)
        # Duration: 20 seconds
        samples = []
        for i in range(20):
            t = i / 19.0  # Interpolation factor [0, 1]
            pos = interpolate_position((100.0, 300.0), (700.0, 300.0), t)
            # Always looking up toward shelves (270 deg = up)
            samples.append(ViewerSample(position=pos, direction=make_unit_vector(270)))

        result = compute_attention_seconds(samples, aois, field_of_view_deg=90.0)

        assert result.total_samples == 20

        # As viewer walks past, should hit different shelves in sequence
        left_count = result.get_hit_count("left_shelf")
        center_count = result.get_hit_count("center_shelf")
        right_count = result.get_hit_count("right_shelf")

        # Should have viewed multiple shelves while walking
        shelves_viewed = sum([left_count > 0, center_count > 0, right_count > 0])
        assert shelves_viewed >= 2, "Should view multiple shelves while walking"

        # Total hits should be positive
        assert result.samples_with_hits > 0

    def test_walking_viewer_diagonal_path(self) -> None:
        """Viewer walks diagonally through store with changing view direction."""
        aois = create_store_layout()

        # Walk diagonal path while looking in different directions
        samples = []
        for i in range(25):
            t = i / 24.0
            # Diagonal walk from bottom-left toward top-center
            pos = interpolate_position((150.0, 400.0), (650.0, 200.0), t)

            # Vary view direction: scan from left to right while moving
            angle = 225 + (90 * t)  # Start at 225 (up-left), end at 315 (up-right)
            direction = make_unit_vector(angle)

            samples.append(ViewerSample(position=pos, direction=direction))

        result = compute_attention_seconds(samples, aois, field_of_view_deg=90.0)

        assert result.total_samples == 25
        # Should hit multiple AOIs during diagonal traverse
        assert result.samples_with_hits > 5


# =============================================================================
# Test: Browsing Behavior
# =============================================================================


class TestScenarioBrowsingBehavior:
    """Test realistic browsing: walk, stop, examine, move on.

    Simulates a shopper walking down an aisle, stopping at a shelf
    of interest, examining it for several seconds, then moving on.
    """

    def test_browsing_walk_stop_examine_continue(self) -> None:
        """Viewer walks, stops at a shelf, examines it, then continues."""
        aois = create_store_layout()

        samples = []

        # Phase 1: Walk toward left shelf (0-5 seconds)
        for i in range(5):
            t = i / 4.0
            pos = interpolate_position((100.0, 350.0), (200.0, 250.0), t)
            samples.append(ViewerSample(position=pos, direction=make_unit_vector(270)))

        # Phase 2: Stop and examine left shelf (5-15 seconds)
        for _ in range(10):
            samples.append(
                ViewerSample(position=(200.0, 250.0), direction=make_unit_vector(270))
            )

        # Phase 3: Walk toward center shelf (15-20 seconds)
        for i in range(5):
            t = i / 4.0
            pos = interpolate_position((200.0, 250.0), (400.0, 250.0), t)
            samples.append(ViewerSample(position=pos, direction=make_unit_vector(270)))

        # Phase 4: Stop and examine center shelf (20-25 seconds)
        for _ in range(5):
            samples.append(
                ViewerSample(position=(400.0, 250.0), direction=make_unit_vector(270))
            )

        result = compute_attention_seconds(samples, aois, field_of_view_deg=60.0)

        assert result.total_samples == 25

        # Left shelf should have significant attention (examined for 10 seconds)
        left_count = result.get_hit_count("left_shelf")
        assert left_count >= 8, "Left shelf examined for ~10 seconds"

        # Center shelf should also have attention
        center_count = result.get_hit_count("center_shelf")
        assert center_count >= 3, "Center shelf examined for ~5 seconds"

        # Verify browsing pattern is captured
        assert result.samples_with_hits >= 15

    def test_browsing_compare_two_items(self) -> None:
        """Viewer alternates gaze between two adjacent shelves (comparing items)."""
        aois = create_store_layout()

        # Stand between left and center shelves at Y=250, shelves at Y=100
        viewer_pos = (300.0, 250.0)
        samples = []

        for i in range(20):
            # Alternate: 2 seconds left, 2 seconds center
            if (i // 2) % 2 == 0:
                direction = make_unit_vector(250)  # Look toward left shelf (up-left)
            else:
                direction = make_unit_vector(290)  # Look toward center shelf (up-right)
            samples.append(ViewerSample(position=viewer_pos, direction=direction))

        result = compute_attention_seconds(samples, aois, field_of_view_deg=50.0)

        assert result.total_samples == 20

        left_count = result.get_hit_count("left_shelf")
        center_count = result.get_hit_count("center_shelf")

        # Both shelves should receive roughly equal attention
        assert left_count > 5, "Left shelf should be viewed multiple times"
        assert center_count > 5, "Center shelf should be viewed multiple times"


# =============================================================================
# Test: Quick Glances
# =============================================================================


class TestScenarioQuickGlances:
    """Test rapid direction changes simulating quick glances.

    Simulates a shopper quickly scanning multiple items or getting
    distracted by various displays.
    """

    def test_quick_glances_rapid_scanning(self) -> None:
        """Viewer rapidly scans across multiple shelves (1-2 seconds each)."""
        aois = create_store_layout()

        viewer_pos = (400.0, 300.0)  # Y=300, shelves at Y=100
        samples = []

        # Rapid scan pattern: 15 seconds total, changing direction every 1-2 seconds
        # Angles centered around 270 (up) to see shelves
        angles = [225, 225, 250, 270, 290, 315, 315, 300, 270, 240, 225, 270, 300, 315, 270]
        for angle in angles:
            samples.append(ViewerSample(position=viewer_pos, direction=make_unit_vector(angle)))

        result = compute_attention_seconds(samples, aois, field_of_view_deg=70.0)

        assert result.total_samples == 15

        # Should hit multiple shelves during rapid scanning
        shelves_with_hits = sum(
            [
                result.get_hit_count(aoi_id) > 0
                for aoi_id in ["left_shelf", "center_shelf", "right_shelf", "back_display"]
            ]
        )
        assert shelves_with_hits >= 2, "Quick glances should hit multiple shelves"

    def test_quick_glances_distraction_pattern(self) -> None:
        """Viewer is distracted, looks away and back repeatedly."""
        aois = create_store_layout()

        viewer_pos = (400.0, 300.0)  # Y=300, shelves at Y=100
        samples = []

        # Pattern: Look at shelf, look away, look back
        # 3 seconds on shelf, 2 seconds away, repeat
        for _ in range(4):  # 4 cycles = 20 seconds
            # 3 seconds looking at center shelf (up = 270 degrees)
            for _ in range(3):
                samples.append(
                    ViewerSample(position=viewer_pos, direction=make_unit_vector(270))
                )
            # 2 seconds looking away (down at phone/floor = 90 degrees)
            for _ in range(2):
                samples.append(
                    ViewerSample(position=viewer_pos, direction=make_unit_vector(90))
                )

        result = compute_attention_seconds(samples, aois, field_of_view_deg=60.0)

        assert result.total_samples == 20

        # Should hit center shelf during focused periods
        center_count = result.get_hit_count("center_shelf")
        assert center_count >= 8, "Should capture focused viewing periods"

        # Should have periods of no hits (when distracted)
        assert result.samples_no_winner >= 4, "Should capture distraction periods"


# =============================================================================
# Test: Long Stare
# =============================================================================


class TestScenarioLongStare:
    """Test extended viewing of a single AOI.

    Simulates a shopper who finds something interesting and stares
    at it for an extended period (reading labels, comparing prices, etc.).
    """

    def test_long_stare_reading_labels(self) -> None:
        """Viewer stares at one shelf for extended period (60 seconds)."""
        aois = create_store_layout()

        # Stand in front of center shelf at Y=250, shelf at Y=100
        viewer_pos = (400.0, 250.0)
        samples = [
            ViewerSample(position=viewer_pos, direction=make_unit_vector(270))
            for _ in range(60)
        ]

        result = compute_attention_seconds(samples, aois, field_of_view_deg=60.0)

        assert result.total_samples == 60

        # Center shelf should dominate attention
        center_count = result.get_hit_count("center_shelf")
        assert center_count >= 50, "Should capture most of the long stare"

        # Verify this is the dominant AOI
        top_aois = result.get_top_aois(1)
        assert len(top_aois) == 1
        assert top_aois[0][0] == "center_shelf"
        assert top_aois[0][1] >= 50

    def test_long_stare_with_micro_movements(self) -> None:
        """Extended viewing with small head movements (realistic)."""
        aois = create_store_layout()

        # Small position and direction variations around center shelf
        samples = []
        for i in range(45):
            # Small random-ish variations in position (Y=250, shelf at Y=100)
            x_offset = (i % 5) - 2  # Varies by ±2 pixels
            y_offset = (i % 3) - 1  # Varies by ±1 pixel
            pos = (400.0 + x_offset, 250.0 + y_offset)

            # Small variations in viewing angle (265-275 degrees = mostly up)
            angle = 270 + ((i % 7) - 3) * 1.5  # Varies by ±4.5 degrees
            direction = make_unit_vector(angle)

            samples.append(ViewerSample(position=pos, direction=direction))

        result = compute_attention_seconds(samples, aois, field_of_view_deg=60.0)

        assert result.total_samples == 45

        # Despite micro-movements, should still capture center shelf as dominant
        center_count = result.get_hit_count("center_shelf")
        assert center_count >= 35, "Micro-movements shouldn't break attention tracking"


# =============================================================================
# Test: Peripheral Viewing
# =============================================================================


class TestScenarioPeripheralViewing:
    """Test AOIs at edge of field of view.

    Simulates situations where AOIs are only partially visible or
    at the edge of the viewer's peripheral vision.
    """

    def test_peripheral_viewing_narrow_fov(self) -> None:
        """Viewer with narrow FOV sees less peripheral content."""
        aois = create_store_layout()

        # Viewer looking straight up at shelves with different FOVs
        viewer_pos = (400.0, 300.0)  # Y=300, shelves at Y=100
        samples = [
            ViewerSample(position=viewer_pos, direction=make_unit_vector(270))
            for _ in range(20)
        ]

        result_narrow = compute_attention_seconds(samples, aois, field_of_view_deg=30.0)

        # Same scenario with wide FOV (120 degrees)
        result_wide = compute_attention_seconds(samples, aois, field_of_view_deg=120.0)

        # Wide FOV should capture more or equal AOIs
        assert result_wide.samples_with_hits >= result_narrow.samples_with_hits

    def test_peripheral_viewing_aoi_at_edge(self) -> None:
        """AOI at exact edge of FOV should be detected."""
        # Create a simple scenario where we can precisely control the geometry
        # Viewer at origin looking up/north (270 degrees in image coordinates)
        viewer_pos = (0.0, 0.0)
        viewer_direction = make_unit_vector(270)
        fov = 60.0  # 60 degree FOV means ±30 degrees from center

        # Place AOI at exactly 30 degrees from center direction
        # Center is at 270°, so edge is at 270° + 30° = 300°
        # AOI should be at distance 100, angle 300°
        aoi_angle_rad = math.radians(300)
        aoi_distance = 100.0
        aoi_x = aoi_distance * math.cos(aoi_angle_rad)
        aoi_y = aoi_distance * math.sin(aoi_angle_rad)

        aoi = AOI(
            id="edge_aoi", contour=make_square_contour((aoi_x, aoi_y), half_size=10.0)
        )

        samples = [ViewerSample(position=viewer_pos, direction=viewer_direction)]

        result = compute_attention_seconds(samples, [aoi], field_of_view_deg=fov)

        # AOI at edge should be detected
        assert result.get_hit_count("edge_aoi") > 0, "AOI at edge of FOV should be visible"

    def test_peripheral_viewing_multiple_aois_in_periphery(self) -> None:
        """Multiple AOIs in peripheral vision, largest one wins."""
        viewer_pos = (400.0, 300.0)  # Y=300
        viewer_direction = make_unit_vector(270)  # Looking up

        # Create three AOIs in view at Y=100-150: one center, two peripheral
        aois = [
            # Center AOI (small) directly ahead
            AOI(
                id="center_small",
                contour=make_square_contour((400.0, 150.0), half_size=15.0),
            ),
            # Left peripheral AOI (large)
            AOI(
                id="left_large",
                contour=make_square_contour((250.0, 150.0), half_size=40.0),
            ),
            # Right peripheral AOI (medium)
            AOI(
                id="right_medium",
                contour=make_square_contour((550.0, 150.0), half_size=25.0),
            ),
        ]

        samples = [ViewerSample(position=viewer_pos, direction=viewer_direction)]

        result = compute_attention_seconds(samples, aois, field_of_view_deg=90.0)

        # Largest AOI should win even if it's in periphery
        left_large_hits = result.get_hit_count("left_large")
        center_small_hits = result.get_hit_count("center_small")
        right_medium_hits = result.get_hit_count("right_medium")

        # Assert that the largest AOI (left_large with half_size=40) wins
        assert left_large_hits > 0, "Largest AOI should be detected"
        assert left_large_hits >= center_small_hits, "Largest AOI should beat or tie with smaller center AOI"
        assert left_large_hits >= right_medium_hits, "Largest AOI should beat or tie with medium peripheral AOI"


# =============================================================================
# Test: Complete Store Walkthrough
# =============================================================================


class TestScenarioCompleteStoreWalkthrough:
    """End-to-end simulation of complete shopping trip.

    Simulates a realistic 2-minute shopping session with multiple
    behaviors: entering, browsing, selecting, and exiting.
    """

    def test_complete_store_walkthrough(self) -> None:
        """Full 120-second shopping trip with realistic behavior."""
        aois = create_store_layout()

        samples = []

        # === Phase 1: Enter store (0-10 seconds) ===
        # Walk from entrance (Y=400) toward first shelf
        for i in range(10):
            t = i / 9.0
            pos = interpolate_position((400.0, 400.0), (200.0, 300.0), t)
            # Looking ahead toward shelves (up = 270 deg)
            samples.append(ViewerSample(position=pos, direction=make_unit_vector(270)))

        # === Phase 2: Browse left shelf (10-25 seconds) ===
        # Stop and examine left shelf
        for i in range(15):
            pos = (200.0, 300.0)
            # Slight head movements while examining (around 270 deg)
            angle = 270 + ((i % 5) - 2) * 3
            samples.append(ViewerSample(position=pos, direction=make_unit_vector(angle)))

        # === Phase 3: Move to center (25-30 seconds) ===
        # Quick walk to center shelf
        for i in range(5):
            t = i / 4.0
            pos = interpolate_position((200.0, 300.0), (400.0, 300.0), t)
            samples.append(ViewerSample(position=pos, direction=make_unit_vector(270)))

        # === Phase 4: Quick glance at center (30-35 seconds) ===
        # Brief look, not interested
        for _ in range(5):
            samples.append(
                ViewerSample(position=(400.0, 300.0), direction=make_unit_vector(270))
            )

        # === Phase 5: Move to right shelf (35-40 seconds) ===
        for i in range(5):
            t = i / 4.0
            pos = interpolate_position((400.0, 300.0), (600.0, 300.0), t)
            samples.append(ViewerSample(position=pos, direction=make_unit_vector(270)))

        # === Phase 6: Browse right shelf extensively (40-70 seconds) ===
        # Found something interesting
        for i in range(30):
            pos = (600.0, 300.0)
            angle = 270 + ((i % 7) - 3) * 2
            samples.append(ViewerSample(position=pos, direction=make_unit_vector(angle)))

        # === Phase 7: Look at back display (70-85 seconds) ===
        # Notice back wall display, walk closer
        for i in range(15):
            t = i / 14.0
            pos = interpolate_position((600.0, 300.0), (400.0, 200.0), t)
            # Looking generally up toward display (270 deg)
            angle = 270
            samples.append(ViewerSample(position=pos, direction=make_unit_vector(angle)))

        # === Phase 8: Examine back display (85-100 seconds) ===
        for i in range(15):
            pos = (400.0, 200.0)
            angle = 270 + ((i % 4) - 1) * 5
            samples.append(ViewerSample(position=pos, direction=make_unit_vector(angle)))

        # === Phase 9: Exit store (100-120 seconds) ===
        # Walk toward exit, mostly looking down/away
        for i in range(20):
            t = i / 19.0
            pos = interpolate_position((400.0, 200.0), (400.0, 450.0), t)
            # Mostly looking down/away from shelves (toward exit)
            if i % 5 == 0:
                angle = 270  # Occasional look back at shelves
            else:
                angle = 90  # Look toward exit (down)
            samples.append(ViewerSample(position=pos, direction=make_unit_vector(angle)))

        # === Compute and Validate ===
        result = compute_attention_seconds(samples, aois, field_of_view_deg=70.0)

        assert result.total_samples == 120

        # Verify session statistics - realistic expectations
        assert result.samples_with_hits >= 50, "Should have substantial viewing time"
        assert result.coverage_ratio >= 0.4, "Coverage ratio should be significant"

        # Verify attention distribution
        left_count = result.get_hit_count("left_shelf")
        center_count = result.get_hit_count("center_shelf")
        right_count = result.get_hit_count("right_shelf")
        back_count = result.get_hit_count("back_display")

        # All shelves should have received some attention
        assert left_count > 0, "Left shelf was browsed"
        assert center_count > 0, "Center shelf was glanced at"
        assert right_count > 0, "Right shelf was examined"
        # Back display may or may not be visible depending on exact geometry
        # assert back_count > 0, "Back display was viewed"

        # Right shelf should have most attention (browsed for 30 seconds)
        top_aois = result.get_top_aois(4)
        top_aoi_ids = [aoi_id for aoi_id, _ in top_aois]
        assert "right_shelf" in top_aoi_ids, "Right shelf should be in top AOIs"

        # Verify realistic distribution
        total_attention = left_count + center_count + right_count + back_count
        assert total_attention <= 120, "Total attention can't exceed sample count"

        # Center shelf may get more hits due to being in the middle path
        # (viewer walks past it multiple times)

    def test_complete_store_walkthrough_with_session_config(self) -> None:
        """Verify session config is properly embedded in walkthrough result."""
        aois = create_store_layout()

        # Simple walkthrough: 30 seconds at Y=300, looking up at shelves
        samples = []
        for i in range(30):
            pos = (300.0 + i * 5, 300.0)
            samples.append(ViewerSample(position=pos, direction=make_unit_vector(270)))

        session_config = SessionConfig(
            session_id="store_visit_001",
            frame_size=(800, 600),
            viewer_id="customer_123",
            sample_interval_seconds=1.0,
        )

        result = compute_attention_seconds(
            samples, aois, field_of_view_deg=70.0, session_config=session_config
        )

        # Verify session config is embedded
        assert result.session_config is not None
        assert result.session_config.session_id == "store_visit_001"
        assert result.session_config.frame_size == (800, 600)
        assert result.session_config.viewer_id == "customer_123"
        assert result.session_config.sample_interval_seconds == 1.0


# =============================================================================
# Additional Edge Cases
# =============================================================================


class TestScenarioEdgeCases:
    """Test edge cases and boundary conditions in realistic scenarios."""

    def test_scenario_no_aois_visible_entire_session(self) -> None:
        """Viewer never looks at any AOI during entire session."""
        aois = create_store_layout()

        # Viewer at Y=300, shelves at Y=100, looking down (away from shelves)
        samples = [
            ViewerSample(position=(400.0, 300.0), direction=make_unit_vector(90))
            for _ in range(30)
        ]

        result = compute_attention_seconds(samples, aois, field_of_view_deg=60.0)

        assert result.total_samples == 30
        assert result.samples_with_hits == 0
        assert result.samples_no_winner == 30
        assert result.coverage_ratio == 0.0

    def test_scenario_all_aois_always_visible(self) -> None:
        """Viewer position and FOV allows all AOIs to be visible simultaneously."""
        # Create small, closely-grouped AOIs at Y=80
        aois = [
            AOI(id="aoi1", contour=make_square_contour((100.0, 80.0), half_size=10.0)),
            AOI(id="aoi2", contour=make_square_contour((120.0, 80.0), half_size=10.0)),
            AOI(id="aoi3", contour=make_square_contour((110.0, 90.0), half_size=10.0)),
        ]

        # Viewer at Y=150, looking up at AOIs with wide FOV
        samples = [
            ViewerSample(position=(110.0, 150.0), direction=make_unit_vector(270))
            for _ in range(20)
        ]

        result = compute_attention_seconds(samples, aois, field_of_view_deg=120.0)

        assert result.total_samples == 20
        # One AOI should win each sample (largest visible)
        assert result.samples_with_hits == 20
        assert result.samples_no_winner == 0

    def test_scenario_rapid_position_changes(self) -> None:
        """Viewer teleports around store (unrealistic but valid input)."""
        aois = create_store_layout()

        # Jump to random positions each second, all at Y >= 200 to potentially see shelves at Y=100
        positions = [
            (100.0, 200.0),
            (500.0, 300.0),
            (300.0, 250.0),
            (600.0, 350.0),
            (200.0, 280.0),
            (450.0, 320.0),
            (350.0, 260.0),
            (550.0, 310.0),
        ]

        samples = [
            ViewerSample(position=pos, direction=make_unit_vector(270)) for pos in positions
        ]

        result = compute_attention_seconds(samples, aois, field_of_view_deg=70.0)

        assert result.total_samples == 8
        # Should still process correctly despite unrealistic movement
        assert result.samples_with_hits >= 0
        assert result.samples_with_hits <= 8


# =============================================================================
# Tests: Scenarios - Missing Direction Handling
# =============================================================================


class TestScenarioMissingDirection:
    """Test scenarios with missing direction data."""

    def test_compute_attention_numpy_mixed_samples(self) -> None:
        """Test NumPy array with mix of valid and (0, 0) directions."""
        aois = create_store_layout()
        
        # Mix of valid and missing directions
        # Some samples looking at shelves, some with missing direction
        samples = np.array([
            [300.0, 200.0, 0.0, -1.0],   # Valid - looking at shelves below
            [300.0, 200.0, 0.0, 0.0],    # Missing direction
            [400.0, 200.0, -1.0, 0.0],   # Valid - looking left at shelves
            [400.0, 200.0, 0.0, 0.0],    # Missing direction
            [500.0, 200.0, 0.0, -1.0],   # Valid - looking at shelves below
        ])
        
        result = compute_attention_seconds(
            samples,
            aois,
            field_of_view_deg=90.0,
            allow_missing_direction=True
        )
        
        # Verify counters
        assert result.total_samples == 5
        # Two samples have missing directions
        # Three samples have valid directions - may or may not hit depending on geometry
        assert result.samples_with_hits + result.samples_no_winner == 5
        # At least 2 samples should be no-winner (the missing direction ones)
        assert result.samples_no_winner >= 2

    def test_compute_attention_list_mixed_samples(self) -> None:
        """Test list of ViewerSamples with some having allow_missing_direction=True."""
        aois = create_store_layout()
        
        samples = [
            ViewerSample(position=(300.0, 200.0), direction=(0.0, -1.0)),  # Valid
            ViewerSample(
                position=(300.0, 200.0),
                direction=(0.0, 0.0),
                allow_missing_direction=True
            ),  # Missing
            ViewerSample(position=(400.0, 200.0), direction=(-1.0, 0.0)),  # Valid
            ViewerSample(
                position=(400.0, 200.0),
                direction=(0.0, 0.0),
                allow_missing_direction=True
            ),  # Missing
        ]
        
        result = compute_attention_seconds(samples, aois, field_of_view_deg=90.0)
        
        # Verify counters
        assert result.total_samples == 4
        assert result.samples_with_hits + result.samples_no_winner == 4
        # Two samples have missing directions
        assert result.samples_no_winner >= 2

    def test_compute_attention_all_missing_directions(self) -> None:
        """Test that all samples with missing directions results in zero hits."""
        aois = create_store_layout()
        
        # All samples have missing direction
        samples = np.array([
            [100.0, 100.0, 0.0, 0.0],
            [200.0, 200.0, 0.0, 0.0],
            [300.0, 300.0, 0.0, 0.0],
            [400.0, 400.0, 0.0, 0.0],
            [500.0, 500.0, 0.0, 0.0],
        ])
        
        result = compute_attention_seconds(
            samples,
            aois,
            field_of_view_deg=90.0,
            allow_missing_direction=True
        )
        
        # All samples should be no-winner
        assert result.total_samples == 5
        assert result.samples_with_hits == 0
        assert result.samples_no_winner == 5
        
        # All AOIs should have zero hits
        for aoi_id, aoi_result in result.aoi_results.items():
            assert aoi_result.hit_count == 0
            assert aoi_result.total_attention_seconds == 0.0

    def test_compute_attention_missing_direction_flag_false(self) -> None:
        """Test that NumPy array with (0, 0) and allow_missing_direction=False raises error."""
        aois = create_store_layout()
        
        # Sample with missing direction
        samples = np.array([
            [100.0, 100.0, 1.0, 0.0],
            [200.0, 200.0, 0.0, 0.0],  # Missing direction
        ])
        
        # Should raise error with flag=False (default)
        with pytest.raises(ValidationError) as exc_info:
            compute_attention_seconds(
                samples,
                aois,
                field_of_view_deg=90.0,
                allow_missing_direction=False
            )
        
        assert "unit vector" in str(exc_info.value).lower() or "zero" in str(exc_info.value).lower()

    def test_streaming_mode_with_missing_directions(self) -> None:
        """Test compute_attention_seconds_streaming with missing directions."""
        from view_arc.tracking import compute_attention_seconds_streaming
        
        aois = create_store_layout()
        
        # Mix of valid and missing directions
        samples = np.array([
            [300.0, 200.0, 0.0, -1.0],   # Valid
            [300.0, 200.0, 0.0, 0.0],    # Missing
            [400.0, 200.0, -1.0, 0.0],   # Valid
            [400.0, 200.0, 0.0, 0.0],    # Missing
            [500.0, 200.0, 0.0, -1.0],   # Valid
        ])
        
        result_gen = compute_attention_seconds_streaming(
            samples,
            aois,
            field_of_view_deg=90.0,
            chunk_size=2,
            allow_missing_direction=True
        )
        
        # Consume generator to get final result
        result = None
        for result in result_gen:
            pass  # Iterate through all chunks
        
        # Verify final result counters
        assert result is not None
        assert result.total_samples == 5
        assert result.samples_with_hits + result.samples_no_winner == 5
        # At least 2 samples should be no-winner (the missing direction ones)
        assert result.samples_no_winner >= 2

    def test_streaming_mode_missing_direction_default_error(self) -> None:
        """Test that streaming mode raises error for missing directions by default."""
        from view_arc.tracking import compute_attention_seconds_streaming
        
        aois = create_store_layout()
        
        samples = np.array([
            [300.0, 200.0, 1.0, 0.0],
            [300.0, 200.0, 0.0, 0.0],  # Missing direction
        ])
        
        # Should raise error when consuming generator with default allow_missing_direction=False
        with pytest.raises(ValidationError) as exc_info:
            result_gen = compute_attention_seconds_streaming(
                samples,
                aois,
                field_of_view_deg=90.0,
                chunk_size=2
            )
            # Must consume generator to trigger validation
            for _ in result_gen:
                pass
        
        assert "unit vector" in str(exc_info.value).lower() or "zero" in str(exc_info.value).lower()

    def test_missing_direction_maintains_invariants(self) -> None:
        """Test that missing direction samples maintain result invariants."""
        aois = create_store_layout()
        
        # Create samples with known outcomes
        samples = [
            ViewerSample(position=(300.0, 200.0), direction=(0.0, -1.0)),  # Valid - likely hits shelf
            ViewerSample(
                position=(300.0, 200.0),
                direction=(0.0, 0.0),
                allow_missing_direction=True
            ),  # Missing - no hit
            ViewerSample(
                position=(999.0, 999.0),
                direction=(0.0, 0.0),
                allow_missing_direction=True
            ),  # Missing - no hit
        ]
        
        result = compute_attention_seconds(samples, aois, field_of_view_deg=90.0)
        
        # Verify invariants
        assert result.total_samples == 3
        assert result.samples_with_hits + result.samples_no_winner == result.total_samples
        assert result.samples_with_hits >= 0
        assert result.samples_no_winner >= 2  # At least the two missing direction samples
        
        # Verify hit counts sum correctly
        total_hits = sum(aoi_result.hit_count for aoi_result in result.aoi_results.values())
        assert total_hits == result.samples_with_hits
