"""
Visual validation tests for visualization utilities (Step 5.1).

These tests verify that the OpenCV-based visualization functions work correctly
by generating test images and validating their modification.

Both programmatic tests (verifying image modification) and visual tests
(generating output images for inspection) are included.

Run with: pytest tests/visual/test_visualize.py -v

Output figures are saved to: tests/visual/output/
"""

from __future__ import annotations

import pytest
import numpy as np
from pathlib import Path
from typing import List, Tuple, TYPE_CHECKING

# Try to import cv2, skip tests if not available
try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Try to import matplotlib for visual comparison tests
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Wedge
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from view_arc.obstacle.visualize import (
    draw_wedge_overlay,
    draw_obstacle_contours,
    draw_angular_intervals,
    draw_complete_visualization,
)
from view_arc.obstacle.api import find_largest_obstacle


# Output directory for visual test results
OUTPUT_DIR = Path(__file__).parent / "output"


@pytest.fixture(scope="module", autouse=True)
def setup_output_dir() -> None:
    """Create output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_image(image: np.ndarray, name: str) -> None:
    """Save image to output directory using OpenCV."""
    if HAS_CV2:
        filepath = OUTPUT_DIR / f"{name}.png"
        cv2.imwrite(str(filepath), image)
        print(f"Saved: {filepath}")


def save_figure(fig: Figure, name: str) -> None:
    """Save matplotlib figure to output directory."""
    filepath = OUTPUT_DIR / f"{name}.png"
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filepath}")


# =============================================================================
# Helper functions for creating test fixtures
# =============================================================================


def create_blank_image(
    width: int = 400, height: int = 400, color: Tuple[int, int, int] = (240, 240, 240)
) -> np.ndarray:
    """Create a blank BGR image with the specified background color."""
    image = np.full((height, width, 3), color, dtype=np.uint8)
    return image


def make_triangle(center: Tuple[float, float], size: float = 20.0) -> np.ndarray:
    """Create a triangle centered at given point."""
    cx, cy = center
    return np.array(
        [
            [cx, cy - size],  # top vertex (up in image coords)
            [cx - size, cy + size],  # bottom-left
            [cx + size, cy + size],  # bottom-right
        ],
        dtype=np.float32,
    )


def make_square(center: Tuple[float, float], half_size: float = 15.0) -> np.ndarray:
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


def make_rectangle(
    center: Tuple[float, float], width: float, height: float
) -> np.ndarray:
    """Create a rectangle centered at given point."""
    cx, cy = center
    hw, hh = width / 2, height / 2
    return np.array(
        [
            [cx - hw, cy - hh],
            [cx + hw, cy - hh],
            [cx + hw, cy + hh],
            [cx - hw, cy + hh],
        ],
        dtype=np.float32,
    )


# =============================================================================
# Programmatic Tests - Verify functions work correctly
# =============================================================================


@pytest.mark.skipif(not HAS_CV2, reason="OpenCV not installed")
class TestDrawWedgeOverlay:
    """Programmatic tests for draw_wedge_overlay function."""

    def test_draw_wedge_overlay_basic(self) -> None:
        """Verify wedge overlay modifies the image."""
        image = create_blank_image()
        viewer = np.array([200.0, 200.0], dtype=np.float32)
        direction = np.array([1.0, 0.0], dtype=np.float32)  # Looking right

        result = draw_wedge_overlay(
            image, viewer, direction, field_of_view_deg=60.0, max_range=100.0
        )

        # Image should be modified (not identical to blank)
        assert not np.array_equal(result, create_blank_image())
        # Result should have same shape as input
        assert result.shape == image.shape

    def test_draw_wedge_overlay_various_fovs(self) -> None:
        """Test wedge overlay with different field of view angles."""
        viewer = np.array([200.0, 200.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)  # Looking down

        for fov in [30.0, 60.0, 90.0, 120.0, 180.0]:
            image = create_blank_image()
            result = draw_wedge_overlay(
                image, viewer, direction, field_of_view_deg=fov, max_range=100.0
            )
            # Should modify image
            assert not np.array_equal(result, image)

    def test_draw_wedge_overlay_with_fill(self) -> None:
        """Test wedge overlay with fill alpha."""
        image = create_blank_image()
        viewer = np.array([200.0, 200.0], dtype=np.float32)
        direction = np.array([1.0, 0.0], dtype=np.float32)

        result = draw_wedge_overlay(
            image,
            viewer,
            direction,
            field_of_view_deg=60.0,
            max_range=100.0,
            fill_alpha=0.3,
        )

        # Image should be modified
        assert not np.array_equal(result, create_blank_image())

    def test_draw_wedge_overlay_returns_copy(self) -> None:
        """Verify that the function returns a copy, not modifying original."""
        image = create_blank_image()
        original = image.copy()
        viewer = np.array([200.0, 200.0], dtype=np.float32)
        direction = np.array([1.0, 0.0], dtype=np.float32)

        _ = draw_wedge_overlay(
            image, viewer, direction, field_of_view_deg=60.0, max_range=100.0
        )

        # Original should not be modified
        assert np.array_equal(image, original)


@pytest.mark.skipif(not HAS_CV2, reason="OpenCV not installed")
class TestDrawObstacleContours:
    """Programmatic tests for draw_obstacle_contours function."""

    def test_draw_obstacle_contours_no_winner(self) -> None:
        """Test drawing contours without a winner highlighted."""
        image = create_blank_image()
        contours = [
            make_square((100, 100)),
            make_triangle((300, 300)),
        ]

        result = draw_obstacle_contours(image, contours)

        # Image should be modified
        assert not np.array_equal(result, create_blank_image())

    def test_draw_obstacle_contours_with_winner(self) -> None:
        """Test drawing contours with a winner highlighted."""
        image = create_blank_image()
        contours = [
            make_square((100, 100)),
            make_triangle((300, 100)),
            make_rectangle((200, 300), 40, 20),
        ]

        result = draw_obstacle_contours(image, contours, winner_id=1)

        # Image should be modified
        assert not np.array_equal(result, create_blank_image())

    def test_draw_obstacle_contours_with_labels(self) -> None:
        """Test drawing contours with index labels."""
        image = create_blank_image()
        contours = [
            make_square((100, 100)),
            make_triangle((300, 300)),
        ]

        result = draw_obstacle_contours(image, contours, show_labels=True)

        # Image should be modified
        assert not np.array_equal(result, create_blank_image())

    def test_draw_obstacle_contours_empty_list(self) -> None:
        """Test with empty contour list."""
        image = create_blank_image()
        result = draw_obstacle_contours(image, [])

        # Should return copy of original (no contours to draw)
        assert np.array_equal(result, image)

    def test_draw_obstacle_contours_invalid_contour_skipped(self) -> None:
        """Test that invalid contours (< 3 vertices) are skipped."""
        image = create_blank_image()
        contours = [
            np.array([[100, 100], [200, 100]], dtype=np.float32),  # Only 2 points
            make_square((300, 300)),
        ]

        # Should not raise, just skip the invalid contour
        result = draw_obstacle_contours(image, contours)
        assert not np.array_equal(result, create_blank_image())


@pytest.mark.skipif(not HAS_CV2, reason="OpenCV not installed")
class TestDrawAngularIntervals:
    """Programmatic tests for draw_angular_intervals function."""

    def test_draw_angular_intervals_single(self) -> None:
        """Test drawing a single angular interval."""
        image = create_blank_image()
        viewer = np.array([200.0, 200.0], dtype=np.float32)
        intervals = [(0.0, np.pi / 4)]  # 0 to 45 degrees

        result = draw_angular_intervals(image, viewer, intervals, max_range=100.0)

        assert not np.array_equal(result, create_blank_image())

    def test_draw_angular_intervals_multiple(self) -> None:
        """Test drawing multiple angular intervals."""
        image = create_blank_image()
        viewer = np.array([200.0, 200.0], dtype=np.float32)
        intervals = [
            (0.0, np.pi / 6),
            (np.pi / 3, np.pi / 2),
            (np.pi, np.pi * 1.25),
        ]

        result = draw_angular_intervals(image, viewer, intervals, max_range=100.0)

        assert not np.array_equal(result, create_blank_image())

    def test_draw_angular_intervals_with_fill(self) -> None:
        """Test drawing intervals with fill."""
        image = create_blank_image()
        viewer = np.array([200.0, 200.0], dtype=np.float32)
        intervals = [(0.0, np.pi / 4)]

        result = draw_angular_intervals(
            image, viewer, intervals, max_range=100.0, fill_alpha=0.3
        )

        assert not np.array_equal(result, create_blank_image())

    def test_draw_angular_intervals_empty(self) -> None:
        """Test with empty interval list."""
        image = create_blank_image()
        viewer = np.array([200.0, 200.0], dtype=np.float32)

        result = draw_angular_intervals(image, viewer, [], max_range=100.0)

        # Should return copy of original
        assert np.array_equal(result, image)


@pytest.mark.skipif(not HAS_CV2, reason="OpenCV not installed")
class TestDrawCompleteVisualization:
    """Programmatic tests for draw_complete_visualization function."""

    def test_complete_visualization_basic(self) -> None:
        """Test complete visualization with all elements."""
        image = create_blank_image()
        viewer = np.array([200.0, 200.0], dtype=np.float32)
        direction = np.array([1.0, 0.0], dtype=np.float32)
        contours = [make_square((280, 200)), make_triangle((320, 180))]

        result = draw_complete_visualization(
            image,
            viewer,
            direction,
            field_of_view_deg=60.0,
            max_range=100.0,
            obstacle_contours=contours,
            winner_id=0,
        )

        assert not np.array_equal(result, create_blank_image())

    def test_complete_visualization_with_intervals(self) -> None:
        """Test complete visualization including intervals."""
        image = create_blank_image()
        viewer = np.array([200.0, 200.0], dtype=np.float32)
        direction = np.array([1.0, 0.0], dtype=np.float32)
        contours = [make_square((280, 200))]
        intervals = [(-0.2, 0.2)]

        result = draw_complete_visualization(
            image,
            viewer,
            direction,
            field_of_view_deg=60.0,
            max_range=100.0,
            obstacle_contours=contours,
            winner_id=0,
            intervals=intervals,
        )

        assert not np.array_equal(result, create_blank_image())


# =============================================================================
# Visual Tests - Generate images for manual inspection
# =============================================================================


@pytest.mark.skipif(not HAS_CV2, reason="OpenCV not installed")
class TestVisualizeVisual:
    """Visual tests that generate output images for inspection."""

    def test_visual_wedge_overlay_directions(self) -> None:
        """Generate wedge overlays for different view directions."""
        directions = [
            ("right", np.array([1.0, 0.0], dtype=np.float32)),
            ("down", np.array([0.0, 1.0], dtype=np.float32)),
            ("left", np.array([-1.0, 0.0], dtype=np.float32)),
            ("up", np.array([0.0, -1.0], dtype=np.float32)),
            ("diagonal_dr", np.array([0.707, 0.707], dtype=np.float32)),
            ("diagonal_ul", np.array([-0.707, -0.707], dtype=np.float32)),
        ]

        for name, direction in directions:
            image = create_blank_image(500, 500)
            viewer = np.array([250.0, 250.0], dtype=np.float32)

            result = draw_wedge_overlay(
                image,
                viewer,
                direction,
                field_of_view_deg=60.0,
                max_range=150.0,
                color=(0, 255, 0),
                fill_alpha=0.2,
            )

            save_image(result, f"viz_wedge_direction_{name}")

    def test_visual_wedge_overlay_fov_comparison(self) -> None:
        """Generate wedge overlays with different FOV angles for comparison."""
        fovs = [30.0, 60.0, 90.0, 120.0, 180.0]
        image_width = 300
        combined = create_blank_image(image_width * len(fovs), 300)

        for i, fov in enumerate(fovs):
            segment = create_blank_image(image_width, 300)
            viewer = np.array([150.0, 200.0], dtype=np.float32)
            direction = np.array([0.0, -1.0], dtype=np.float32)  # Looking up

            result = draw_wedge_overlay(
                segment,
                viewer,
                direction,
                field_of_view_deg=fov,
                max_range=120.0,
                color=(0, 200, 0),
                fill_alpha=0.15,
            )

            # Add label
            cv2.putText(
                result,
                f"FOV: {int(fov)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )

            combined[:, i * image_width : (i + 1) * image_width] = result

        save_image(combined, "viz_wedge_fov_comparison")

    def test_visual_obstacle_contours_winner_highlight(self) -> None:
        """Generate image showing winner highlighting."""
        image = create_blank_image(500, 500)

        contours = [
            make_square((100, 150), 30),
            make_triangle((250, 120), 35),
            make_rectangle((400, 180), 60, 40),
            make_square((150, 350), 25),
            make_triangle((350, 380), 40),
        ]

        # Draw without winner
        no_winner = draw_obstacle_contours(
            image.copy(), contours, show_labels=True, default_color=(200, 100, 50)
        )
        cv2.putText(
            no_winner,
            "No winner",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
        )
        save_image(no_winner, "viz_contours_no_winner")

        # Draw with winner = 1
        with_winner = draw_obstacle_contours(
            image.copy(),
            contours,
            winner_id=1,
            show_labels=True,
            default_color=(200, 100, 50),
            winner_color=(0, 0, 255),
        )
        cv2.putText(
            with_winner,
            "Winner: 1 (red)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
        )
        save_image(with_winner, "viz_contours_with_winner")

    def test_visual_angular_intervals_multiple(self) -> None:
        """Generate image showing multiple angular intervals."""
        image = create_blank_image(500, 500)
        viewer = np.array([250.0, 250.0], dtype=np.float32)

        intervals = [
            (-np.pi / 6, np.pi / 6),  # Front (right)
            (np.pi / 3, 2 * np.pi / 3),  # Front-down
            (-2 * np.pi / 3, -np.pi / 3),  # Front-up
        ]

        result = draw_angular_intervals(
            image,
            viewer,
            intervals,
            max_range=180.0,
            color=(0, 200, 200),
            fill_alpha=0.25,
            thickness=2,
        )

        # Add labels
        cv2.putText(
            result,
            "Angular intervals",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
        )
        cv2.circle(result, (250, 250), 5, (0, 0, 255), -1)  # Mark viewer

        save_image(result, "viz_angular_intervals")

    def test_visual_complete_scenario(self) -> None:
        """Generate a complete visualization scenario."""
        image = create_blank_image(600, 600, (250, 250, 250))
        viewer = np.array([300.0, 400.0], dtype=np.float32)
        direction = np.array([0.0, -1.0], dtype=np.float32)  # Looking up

        # Create obstacles
        contours = [
            make_square((300, 250), 40),  # Directly ahead
            make_triangle((180, 200), 30),  # Left side
            make_rectangle((420, 180), 50, 30),  # Right side
            make_square((280, 80), 25),  # Far ahead, partially visible
        ]

        # Run the actual algorithm to get winner
        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=90.0,
            max_range=250.0,
            obstacle_contours=contours,
            return_intervals=True,
        )

        winner_id = result.obstacle_id
        intervals = result.intervals if result.intervals else []

        # Draw complete visualization
        output = draw_complete_visualization(
            image,
            viewer,
            direction,
            field_of_view_deg=90.0,
            max_range=250.0,
            obstacle_contours=contours,
            winner_id=winner_id,
            intervals=intervals,
        )

        # Add info text
        if winner_id is not None:
            info = f"Winner: {winner_id}, Coverage: {np.rad2deg(result.angular_coverage):.1f} deg"
        else:
            info = "No obstacle visible"
        cv2.putText(
            output, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2
        )

        save_image(output, "viz_complete_scenario")

    def test_visual_occlusion_scenario(self) -> None:
        """Generate visualization showing occlusion between obstacles."""
        image = create_blank_image(600, 600, (250, 250, 250))
        viewer = np.array([300.0, 500.0], dtype=np.float32)
        direction = np.array([0.0, -1.0], dtype=np.float32)  # Looking up

        # Create overlapping obstacles at different distances
        contours = [
            make_square((300, 350), 50),  # Near, large
            make_rectangle((300, 150), 80, 40),  # Far, should be partially occluded
        ]

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=60.0,
            max_range=400.0,
            obstacle_contours=contours,
            return_intervals=True,
        )

        output = draw_complete_visualization(
            image,
            viewer,
            direction,
            field_of_view_deg=60.0,
            max_range=400.0,
            obstacle_contours=contours,
            winner_id=result.obstacle_id,
            intervals=result.intervals if result.intervals else [],
        )

        cv2.putText(
            output,
            f"Occlusion test - Winner: {result.obstacle_id}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )

        save_image(output, "viz_occlusion_scenario")


# =============================================================================
# Comparison tests with matplotlib (if available)
# =============================================================================


@pytest.mark.skipif(
    not (HAS_CV2 and HAS_MATPLOTLIB), reason="OpenCV and matplotlib required"
)
class TestVisualizationComparison:
    """Compare OpenCV visualization with matplotlib reference."""

    def _draw_matplotlib_wedge(
        self,
        ax: Axes,
        viewer: np.ndarray,
        direction: np.ndarray,
        fov_deg: float,
        max_range: float,
    ) -> None:
        """Draw a wedge using matplotlib for reference."""
        alpha_center = np.arctan2(direction[1], direction[0])
        half_fov = np.deg2rad(fov_deg) / 2
        theta1 = np.rad2deg(alpha_center - half_fov)
        theta2 = np.rad2deg(alpha_center + half_fov)

        wedge = Wedge(
            (viewer[0], viewer[1]),
            max_range,
            theta1,
            theta2,
            facecolor="green",
            alpha=0.2,
            edgecolor="green",
            linewidth=2,
        )
        ax.add_patch(wedge)
        ax.plot(viewer[0], viewer[1], "ko", markersize=8)

    def test_visual_side_by_side_comparison(self) -> None:
        """Generate side-by-side comparison of OpenCV and matplotlib outputs."""
        viewer = np.array([200.0, 200.0], dtype=np.float32)
        direction = np.array([1.0, 0.0], dtype=np.float32)
        fov_deg = 60.0
        max_range = 120.0

        contours = [
            make_square((280, 180), 25),
            make_triangle((300, 240), 30),
        ]

        # Create OpenCV visualization
        cv_image = create_blank_image(400, 400)
        cv_result = draw_complete_visualization(
            cv_image,
            viewer,
            direction,
            fov_deg,
            max_range,
            contours,
            winner_id=0,
        )

        # Convert BGR to RGB for matplotlib
        cv_rgb = cv2.cvtColor(cv_result, cv2.COLOR_BGR2RGB)

        # Create matplotlib figure with side-by-side comparison
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Left: OpenCV output
        axes[0].imshow(cv_rgb)
        axes[0].set_title("OpenCV Visualization")
        axes[0].axis("off")

        # Right: Matplotlib reference
        axes[1].set_xlim(0, 400)
        axes[1].set_ylim(400, 0)  # Flip Y to match image coordinates
        axes[1].set_aspect("equal")
        axes[1].set_facecolor((0.94, 0.94, 0.94))

        self._draw_matplotlib_wedge(axes[1], viewer, direction, fov_deg, max_range)

        for idx, contour in enumerate(contours):
            color = "red" if idx == 0 else "blue"
            patch = mpatches.Polygon(
                contour,
                closed=True,
                facecolor=color,
                alpha=0.3,
                edgecolor=color,
                linewidth=2,
            )
            axes[1].add_patch(patch)

        axes[1].set_title("Matplotlib Reference")
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(
            "Visualization Comparison: OpenCV vs Matplotlib", fontsize=14, y=1.02
        )
        plt.tight_layout()
        save_figure(fig, "viz_comparison_cv_matplotlib")
