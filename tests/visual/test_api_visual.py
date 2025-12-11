"""
Visual validation tests for the main API (Step 4.1).

These tests create matplotlib figures showing:
- Complete obstacle detection scenarios
- Viewer position and field of view
- Obstacle contours with winner highlighting
- Angular coverage visualization
- Occlusion scenarios

Run with: pytest tests/visual/test_api_visual.py -v

Output figures are saved to: tests/visual/output/
"""

import pytest
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

# Try to import matplotlib, skip tests if not available
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Wedge, FancyArrowPatch, Circle, Arc
    from matplotlib.collections import LineCollection
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from view_arc.obstacle.api import find_largest_obstacle, ObstacleResult
from view_arc.obstacle.geometry import to_viewer_frame, to_polar, validate_and_get_direction_angle
from view_arc.obstacle.clipping import clip_polygon_to_wedge


# Output directory for visual test results
OUTPUT_DIR = Path(__file__).parent / "output"


@pytest.fixture(scope="module", autouse=True)
def setup_output_dir():
    """Create output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig, name: str):
    """Save figure to output directory."""
    filepath = OUTPUT_DIR / f"{name}.png"
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filepath}")


# =============================================================================
# Helper functions for creating test fixtures
# =============================================================================

def make_triangle(center: Tuple[float, float], size: float = 20.0) -> np.ndarray:
    """Create a triangle centered at given point."""
    cx, cy = center
    return np.array([
        [cx, cy + size],
        [cx - size, cy - size],
        [cx + size, cy - size],
    ], dtype=np.float32)


def make_square(center: Tuple[float, float], half_size: float = 15.0) -> np.ndarray:
    """Create a square centered at given point."""
    cx, cy = center
    return np.array([
        [cx - half_size, cy - half_size],
        [cx + half_size, cy - half_size],
        [cx + half_size, cy + half_size],
        [cx - half_size, cy + half_size],
    ], dtype=np.float32)


def make_rectangle(
    center: Tuple[float, float], width: float, height: float
) -> np.ndarray:
    """Create a rectangle centered at given point."""
    cx, cy = center
    hw, hh = width / 2, height / 2
    return np.array([
        [cx - hw, cy - hh],
        [cx + hw, cy - hh],
        [cx + hw, cy + hh],
        [cx - hw, cy + hh],
    ], dtype=np.float32)


def make_pentagon(center: Tuple[float, float], radius: float = 20.0) -> np.ndarray:
    """Create a regular pentagon."""
    cx, cy = center
    angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, 6)[:-1]
    return np.array([
        [cx + radius * np.cos(a), cy + radius * np.sin(a)]
        for a in angles
    ], dtype=np.float32)


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
class TestAPIVisual:
    """Visual validation tests for the main API."""

    def _draw_polygon(
        self,
        ax,
        polygon: np.ndarray,
        color: str = "blue",
        alpha: float = 0.3,
        label: Optional[str] = None,
        edgecolor: Optional[str] = None,
        linewidth: float = 2,
        show_vertices: bool = False,
    ):
        """Draw a polygon on the axes."""
        if polygon is None or len(polygon) < 3:
            return
        if edgecolor is None:
            edgecolor = color
        patch = mpatches.Polygon(
            polygon,
            closed=True,
            facecolor=color,
            alpha=alpha,
            edgecolor=edgecolor,
            linewidth=linewidth,
            label=label,
        )
        ax.add_patch(patch)

        if show_vertices:
            ax.scatter(polygon[:, 0], polygon[:, 1], color=edgecolor, s=30, zorder=5)

    def _draw_viewer(
        self,
        ax,
        viewer: np.ndarray,
        direction: np.ndarray,
        fov_deg: float,
        max_range: float,
        alpha: float = 0.15,
    ):
        """Draw viewer position and field of view wedge."""
        # Draw viewer point
        ax.plot(
            viewer[0], viewer[1], "ko", markersize=10, zorder=20, label="Viewer"
        )

        # Compute arc angles
        alpha_center = np.arctan2(direction[1], direction[0])
        half_fov = np.deg2rad(fov_deg) / 2
        alpha_min = alpha_center - half_fov
        alpha_max = alpha_center + half_fov

        # Draw FOV wedge
        theta1 = np.rad2deg(alpha_min)
        theta2 = np.rad2deg(alpha_max)

        wedge = Wedge(
            (viewer[0], viewer[1]),
            max_range,
            theta1,
            theta2,
            facecolor="yellow",
            alpha=alpha,
            edgecolor="orange",
            linewidth=1.5,
            label=f"FOV ({fov_deg}°)",
        )
        ax.add_patch(wedge)

        # Draw direction arrow
        arrow_len = max_range * 0.3
        ax.arrow(
            viewer[0],
            viewer[1],
            direction[0] * arrow_len,
            direction[1] * arrow_len,
            head_width=max_range * 0.05,
            head_length=max_range * 0.03,
            fc="red",
            ec="red",
            linewidth=2,
            zorder=15,
            label="View direction",
        )

        # Draw arc boundary rays
        for angle, style in [(alpha_min, "--"), (alpha_max, "--")]:
            x_end = viewer[0] + max_range * np.cos(angle)
            y_end = viewer[1] + max_range * np.sin(angle)
            ax.plot(
                [viewer[0], x_end],
                [viewer[1], y_end],
                "orange",
                linestyle=style,
                linewidth=1,
                alpha=0.7,
            )

    def _draw_result_annotation(
        self, ax, result: ObstacleResult, position: Tuple[float, float]
    ):
        """Draw result annotation box."""
        if result.obstacle_id is not None:
            text = (
                f"Winner: Obstacle {result.obstacle_id}\n"
                f"Coverage: {np.rad2deg(result.angular_coverage):.1f}°\n"
                f"Distance: {result.min_distance:.1f}"
            )
            color = "green"
        else:
            text = "No obstacle visible"
            color = "red"

        ax.annotate(
            text,
            position,
            xycoords="axes fraction",
            fontsize=10,
            fontweight="bold",
            color=color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            verticalalignment="top",
        )

    def _setup_axes(
        self,
        ax,
        title: str,
        xlim: Tuple[float, float] = (-50, 150),
        ylim: Tuple[float, float] = (-50, 200),
    ):
        """Setup axes with grid and labels."""
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(title)

    # =========================================================================
    # Test: Single Obstacle Scenarios
    # =========================================================================

    def test_visual_single_obstacle_centered(self):
        """Visual: Single obstacle centered in field of view."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        viewer = np.array([100.0, 100.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)  # Looking UP

        # Three scenarios with different obstacle sizes
        scenarios = [
            ("Small Triangle", [make_triangle((100, 150), size=15)]),
            ("Medium Square", [make_square((100, 160), half_size=20)]),
            ("Large Pentagon", [make_pentagon((100, 170), radius=30)]),
        ]

        for ax, (title, contours) in zip(axes, scenarios):
            self._setup_axes(ax, title, xlim=(40, 160), ylim=(60, 220))

            # Draw viewer and FOV
            self._draw_viewer(ax, viewer, direction, fov_deg=60.0, max_range=100.0)

            # Draw obstacle
            self._draw_polygon(
                ax,
                contours[0],
                color="lightblue",
                edgecolor="blue",
                alpha=0.5,
                label="Obstacle 0",
            )

            # Run algorithm
            result = find_largest_obstacle(
                viewer_point=viewer,
                view_direction=direction,
                field_of_view_deg=60.0,
                max_range=100.0,
                obstacle_contours=contours,
            )

            # Annotate result
            self._draw_result_annotation(ax, result, (0.02, 0.98))
            ax.legend(loc="lower right", fontsize=8)

        fig.suptitle(
            "Single Obstacle Detection: Centered in FOV",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        save_figure(fig, "api_single_obstacle_centered")

    def test_visual_single_obstacle_partial_visibility(self):
        """Visual: Single obstacle partially visible (at FOV edge or beyond range)."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        viewer = np.array([100.0, 100.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        scenarios = [
            ("At FOV Edge (left)", [make_square((60, 150), half_size=20)]),
            ("At FOV Edge (right)", [make_square((140, 150), half_size=20)]),
            ("Partially Beyond Range", [make_rectangle((100, 180), 40, 60)]),
        ]

        for ax, (title, contours) in zip(axes, scenarios):
            self._setup_axes(ax, title, xlim=(20, 180), ylim=(60, 250))

            self._draw_viewer(ax, viewer, direction, fov_deg=60.0, max_range=100.0)

            self._draw_polygon(
                ax,
                contours[0],
                color="lightgreen",
                edgecolor="darkgreen",
                alpha=0.5,
                label="Obstacle 0",
            )

            result = find_largest_obstacle(
                viewer_point=viewer,
                view_direction=direction,
                field_of_view_deg=60.0,
                max_range=100.0,
                obstacle_contours=contours,
            )

            self._draw_result_annotation(ax, result, (0.02, 0.98))
            ax.legend(loc="lower right", fontsize=8)

        fig.suptitle(
            "Single Obstacle: Partial Visibility Scenarios",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        save_figure(fig, "api_single_obstacle_partial")

    # =========================================================================
    # Test: Multiple Obstacles - No Occlusion
    # =========================================================================

    def test_visual_two_obstacles_side_by_side(self):
        """Visual: Two obstacles side by side without occlusion."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        viewer = np.array([100.0, 100.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        scenarios = [
            (
                "Equal Size",
                [
                    make_square((60, 150), half_size=15),
                    make_square((140, 150), half_size=15),
                ],
            ),
            (
                "Left Larger",
                [
                    make_square((55, 150), half_size=25),
                    make_square((140, 150), half_size=12),
                ],
            ),
            (
                "Right Larger",
                [
                    make_square((60, 150), half_size=12),
                    make_square((145, 150), half_size=28),
                ],
            ),
        ]

        colors = [
            ("lightblue", "blue"),
            ("lightcoral", "darkred"),
        ]

        for ax, (title, contours) in zip(axes, scenarios):
            self._setup_axes(ax, title, xlim=(0, 200), ylim=(60, 220))

            self._draw_viewer(ax, viewer, direction, fov_deg=90.0, max_range=100.0)

            for i, contour in enumerate(contours):
                fc, ec = colors[i]
                self._draw_polygon(
                    ax,
                    contour,
                    color=fc,
                    edgecolor=ec,
                    alpha=0.5,
                    label=f"Obstacle {i}",
                )

            result = find_largest_obstacle(
                viewer_point=viewer,
                view_direction=direction,
                field_of_view_deg=90.0,
                max_range=100.0,
                obstacle_contours=contours,
            )

            # Highlight winner
            if result.obstacle_id is not None:
                self._draw_polygon(
                    ax,
                    contours[result.obstacle_id],
                    color="none",
                    edgecolor="gold",
                    linewidth=4,
                    alpha=1.0,
                )

            self._draw_result_annotation(ax, result, (0.02, 0.98))
            ax.legend(loc="lower right", fontsize=8)

        fig.suptitle(
            "Two Obstacles Side by Side (No Occlusion)",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        save_figure(fig, "api_two_obstacles_side_by_side")

    def test_visual_three_obstacles_scattered(self):
        """Visual: Three obstacles scattered in the FOV."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        viewer = np.array([100.0, 100.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        contours = [
            make_triangle((60, 140), size=18),  # Left, close
            make_square((100, 180), half_size=22),  # Center, far
            make_pentagon((150, 155), radius=16),  # Right, medium
        ]

        colors = [
            ("lightblue", "blue"),
            ("lightgreen", "darkgreen"),
            ("lightsalmon", "darkorange"),
        ]

        self._setup_axes(ax, "Three Scattered Obstacles", xlim=(20, 180), ylim=(60, 230))
        self._draw_viewer(ax, viewer, direction, fov_deg=100.0, max_range=120.0)

        for i, (contour, (fc, ec)) in enumerate(zip(contours, colors)):
            self._draw_polygon(
                ax, contour, color=fc, edgecolor=ec, alpha=0.5, label=f"Obstacle {i}"
            )

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=100.0,
            max_range=120.0,
            obstacle_contours=contours,
        )

        if result.obstacle_id is not None:
            self._draw_polygon(
                ax,
                contours[result.obstacle_id],
                color="none",
                edgecolor="gold",
                linewidth=5,
                alpha=1.0,
            )

        self._draw_result_annotation(ax, result, (0.02, 0.98))
        ax.legend(loc="lower right", fontsize=9)

        plt.tight_layout()
        save_figure(fig, "api_three_obstacles_scattered")

    # =========================================================================
    # Test: Occlusion Scenarios
    # =========================================================================

    def test_visual_occlusion_near_blocks_far(self):
        """Visual: Near obstacle occludes far obstacle."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        viewer = np.array([100.0, 100.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        scenarios = [
            (
                "Small Near vs Large Far",
                [
                    make_square((100, 130), half_size=12),  # Near, small
                    make_rectangle((100, 180), 80, 30),  # Far, large
                ],
            ),
            (
                "Partial Occlusion",
                [
                    make_square((80, 135), half_size=15),  # Near, left-offset
                    make_rectangle((100, 175), 70, 25),  # Far, centered
                ],
            ),
            (
                "Multiple Near Blocking",
                [
                    make_square((70, 130), half_size=10),
                    make_square((130, 130), half_size=10),
                    make_rectangle((100, 180), 100, 30),
                ],
            ),
        ]

        for ax, (title, contours) in zip(axes, scenarios):
            self._setup_axes(ax, title, xlim=(20, 180), ylim=(60, 230))
            self._draw_viewer(ax, viewer, direction, fov_deg=90.0, max_range=120.0)

            colors = [
                ("lightcoral", "darkred"),
                ("lightblue", "blue"),
                ("lightgreen", "darkgreen"),
            ]

            for i, contour in enumerate(contours):
                fc, ec = colors[i % len(colors)]
                self._draw_polygon(
                    ax, contour, color=fc, edgecolor=ec, alpha=0.5, label=f"Obstacle {i}"
                )

            result = find_largest_obstacle(
                viewer_point=viewer,
                view_direction=direction,
                field_of_view_deg=90.0,
                max_range=120.0,
                obstacle_contours=contours,
            )

            if result.obstacle_id is not None:
                self._draw_polygon(
                    ax,
                    contours[result.obstacle_id],
                    color="none",
                    edgecolor="gold",
                    linewidth=4,
                    alpha=1.0,
                )

            self._draw_result_annotation(ax, result, (0.02, 0.98))
            ax.legend(loc="lower right", fontsize=8)

        fig.suptitle(
            "Occlusion Scenarios: Near Obstacle Blocks Far",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        save_figure(fig, "api_occlusion_scenarios")

    def test_visual_complex_occlusion(self):
        """Visual: Complex occlusion with multiple overlapping obstacles."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        viewer = np.array([100.0, 50.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        # Create a complex scene with overlapping obstacles at different depths
        contours = [
            make_square((100, 100), half_size=15),  # Very close, center
            make_triangle((60, 130), size=20),  # Medium, left
            make_rectangle((140, 140), 30, 40),  # Medium, right
            make_pentagon((100, 180), radius=35),  # Far, center (large)
            make_square((50, 160), half_size=18),  # Medium-far, far left
        ]

        colors = [
            ("lightcoral", "darkred"),
            ("lightblue", "blue"),
            ("lightgreen", "darkgreen"),
            ("lightyellow", "orange"),
            ("plum", "purple"),
        ]

        self._setup_axes(
            ax, "Complex Occlusion Scene", xlim=(-20, 220), ylim=(20, 250)
        )
        self._draw_viewer(ax, viewer, direction, fov_deg=120.0, max_range=180.0)

        for i, (contour, (fc, ec)) in enumerate(zip(contours, colors)):
            self._draw_polygon(
                ax, contour, color=fc, edgecolor=ec, alpha=0.5, label=f"Obstacle {i}"
            )

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=120.0,
            max_range=180.0,
            obstacle_contours=contours,
        )

        if result.obstacle_id is not None:
            self._draw_polygon(
                ax,
                contours[result.obstacle_id],
                color="none",
                edgecolor="gold",
                linewidth=5,
                alpha=1.0,
            )

        self._draw_result_annotation(ax, result, (0.02, 0.98))
        ax.legend(loc="lower right", fontsize=9)

        plt.tight_layout()
        save_figure(fig, "api_complex_occlusion")

    # =========================================================================
    # Test: Different View Directions
    # =========================================================================

    def test_visual_different_view_directions(self):
        """Visual: Same scene viewed from different directions."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 14))
        axes = axes.flatten()

        viewer = np.array([100.0, 100.0], dtype=np.float32)

        # Same obstacles, different viewing directions
        contours = [
            make_square((100, 160), half_size=20),  # North
            make_square((160, 100), half_size=18),  # East
            make_square((100, 40), half_size=15),  # South
            make_square((40, 100), half_size=22),  # West
        ]

        directions = [
            (np.array([0.0, 1.0], dtype=np.float32), "Looking UP (North)"),
            (np.array([1.0, 0.0], dtype=np.float32), "Looking RIGHT (East)"),
            (np.array([0.0, -1.0], dtype=np.float32), "Looking DOWN (South)"),
            (np.array([-1.0, 0.0], dtype=np.float32), "Looking LEFT (West)"),
        ]

        colors = [
            ("lightblue", "blue"),
            ("lightgreen", "darkgreen"),
            ("lightsalmon", "darkorange"),
            ("plum", "purple"),
        ]

        for ax, (direction, title) in zip(axes, directions):
            self._setup_axes(ax, title, xlim=(0, 200), ylim=(0, 200))
            self._draw_viewer(ax, viewer, direction, fov_deg=90.0, max_range=80.0)

            for i, (contour, (fc, ec)) in enumerate(zip(contours, colors)):
                self._draw_polygon(
                    ax, contour, color=fc, edgecolor=ec, alpha=0.4, label=f"Obstacle {i}"
                )

            result = find_largest_obstacle(
                viewer_point=viewer,
                view_direction=direction,
                field_of_view_deg=90.0,
                max_range=80.0,
                obstacle_contours=contours,
            )

            if result.obstacle_id is not None:
                self._draw_polygon(
                    ax,
                    contours[result.obstacle_id],
                    color="none",
                    edgecolor="gold",
                    linewidth=4,
                    alpha=1.0,
                )

            self._draw_result_annotation(ax, result, (0.02, 0.98))
            ax.legend(loc="lower right", fontsize=8)

        fig.suptitle(
            "Same Scene, Different View Directions",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        save_figure(fig, "api_different_directions")

    def test_visual_diagonal_directions(self):
        """Visual: Diagonal viewing directions."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 14))
        axes = axes.flatten()

        viewer = np.array([100.0, 100.0], dtype=np.float32)

        contours = [
            make_triangle((150, 150), size=18),  # NE
            make_square((150, 50), half_size=16),  # SE
            make_pentagon((50, 50), radius=15),  # SW
            make_rectangle((50, 150), 25, 35),  # NW
        ]

        sqrt2 = np.sqrt(2) / 2
        directions = [
            (np.array([sqrt2, sqrt2], dtype=np.float32), "Looking NE (45°)"),
            (np.array([sqrt2, -sqrt2], dtype=np.float32), "Looking SE (-45°)"),
            (np.array([-sqrt2, -sqrt2], dtype=np.float32), "Looking SW (-135°)"),
            (np.array([-sqrt2, sqrt2], dtype=np.float32), "Looking NW (135°)"),
        ]

        colors = [
            ("lightcoral", "darkred"),
            ("lightblue", "blue"),
            ("lightgreen", "darkgreen"),
            ("lightyellow", "orange"),
        ]

        for ax, (direction, title) in zip(axes, directions):
            self._setup_axes(ax, title, xlim=(0, 200), ylim=(0, 200))
            self._draw_viewer(ax, viewer, direction, fov_deg=60.0, max_range=90.0)

            for i, (contour, (fc, ec)) in enumerate(zip(contours, colors)):
                self._draw_polygon(
                    ax, contour, color=fc, edgecolor=ec, alpha=0.4, label=f"Obstacle {i}"
                )

            result = find_largest_obstacle(
                viewer_point=viewer,
                view_direction=direction,
                field_of_view_deg=60.0,
                max_range=90.0,
                obstacle_contours=contours,
            )

            if result.obstacle_id is not None:
                self._draw_polygon(
                    ax,
                    contours[result.obstacle_id],
                    color="none",
                    edgecolor="gold",
                    linewidth=4,
                    alpha=1.0,
                )

            self._draw_result_annotation(ax, result, (0.02, 0.98))
            ax.legend(loc="lower right", fontsize=8)

        fig.suptitle(
            "Diagonal View Directions", fontsize=14, fontweight="bold"
        )
        plt.tight_layout()
        save_figure(fig, "api_diagonal_directions")

    # =========================================================================
    # Test: FOV Width Variations
    # =========================================================================

    def test_visual_fov_width_comparison(self):
        """Visual: Compare results with different FOV widths."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        viewer = np.array([100.0, 100.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        contours = [
            make_square((60, 150), half_size=18),  # Left
            make_square((100, 140), half_size=12),  # Center (closer)
            make_square((145, 160), half_size=20),  # Right
        ]

        fov_values = [30.0, 60.0, 90.0, 120.0, 150.0, 180.0]

        colors = [
            ("lightcoral", "darkred"),
            ("lightblue", "blue"),
            ("lightgreen", "darkgreen"),
        ]

        for ax, fov in zip(axes, fov_values):
            self._setup_axes(ax, f"FOV = {fov}°", xlim=(20, 180), ylim=(60, 220))
            self._draw_viewer(ax, viewer, direction, fov_deg=fov, max_range=100.0)

            for i, (contour, (fc, ec)) in enumerate(zip(contours, colors)):
                self._draw_polygon(
                    ax, contour, color=fc, edgecolor=ec, alpha=0.4, label=f"Obstacle {i}"
                )

            result = find_largest_obstacle(
                viewer_point=viewer,
                view_direction=direction,
                field_of_view_deg=fov,
                max_range=100.0,
                obstacle_contours=contours,
            )

            if result.obstacle_id is not None:
                self._draw_polygon(
                    ax,
                    contours[result.obstacle_id],
                    color="none",
                    edgecolor="gold",
                    linewidth=4,
                    alpha=1.0,
                )

            self._draw_result_annotation(ax, result, (0.02, 0.98))
            ax.legend(loc="lower right", fontsize=7)

        fig.suptitle(
            "Effect of Field of View Width on Winner Selection",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        save_figure(fig, "api_fov_width_comparison")

    # =========================================================================
    # Test: Range Limit Variations
    # =========================================================================

    def test_visual_range_limit_comparison(self):
        """Visual: Compare results with different max range limits."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        viewer = np.array([100.0, 100.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        contours = [
            make_square((100, 130), half_size=15),  # Close (30 units away)
            make_square((100, 170), half_size=20),  # Medium (70 units away)
            make_square((100, 220), half_size=30),  # Far (120 units away)
        ]

        range_values = [40.0, 60.0, 90.0, 120.0, 150.0, 200.0]

        colors = [
            ("lightcoral", "darkred"),
            ("lightblue", "blue"),
            ("lightgreen", "darkgreen"),
        ]

        for ax, max_range in zip(axes, range_values):
            self._setup_axes(
                ax, f"Max Range = {max_range}", xlim=(40, 160), ylim=(60, 280)
            )
            self._draw_viewer(ax, viewer, direction, fov_deg=60.0, max_range=max_range)

            for i, (contour, (fc, ec)) in enumerate(zip(contours, colors)):
                self._draw_polygon(
                    ax, contour, color=fc, edgecolor=ec, alpha=0.4, label=f"Obstacle {i}"
                )

            result = find_largest_obstacle(
                viewer_point=viewer,
                view_direction=direction,
                field_of_view_deg=60.0,
                max_range=max_range,
                obstacle_contours=contours,
            )

            if result.obstacle_id is not None:
                self._draw_polygon(
                    ax,
                    contours[result.obstacle_id],
                    color="none",
                    edgecolor="gold",
                    linewidth=4,
                    alpha=1.0,
                )

            self._draw_result_annotation(ax, result, (0.02, 0.98))
            ax.legend(loc="lower right", fontsize=7)

        fig.suptitle(
            "Effect of Max Range on Winner Selection",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        save_figure(fig, "api_range_limit_comparison")

    # =========================================================================
    # Test: Edge Cases
    # =========================================================================

    def test_visual_no_obstacles_visible(self):
        """Visual: Scenarios where no obstacles are visible."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        viewer = np.array([100.0, 100.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        scenarios = [
            (
                "All Behind Viewer",
                [
                    make_square((100, 50), half_size=20),
                    make_square((60, 40), half_size=15),
                ],
            ),
            (
                "All Beyond Range",
                [
                    make_square((100, 250), half_size=25),
                    make_square((80, 280), half_size=20),
                ],
            ),
            (
                "All Outside FOV",
                [
                    make_square((20, 150), half_size=15),
                    make_square((180, 160), half_size=18),
                ],
            ),
        ]

        for ax, (title, contours) in zip(axes, scenarios):
            self._setup_axes(ax, title, xlim=(-20, 220), ylim=(0, 320))
            self._draw_viewer(ax, viewer, direction, fov_deg=60.0, max_range=100.0)

            colors = [("lightgray", "gray"), ("lightgray", "gray")]
            for i, contour in enumerate(contours):
                fc, ec = colors[i]
                self._draw_polygon(
                    ax, contour, color=fc, edgecolor=ec, alpha=0.5, label=f"Obstacle {i}"
                )

            result = find_largest_obstacle(
                viewer_point=viewer,
                view_direction=direction,
                field_of_view_deg=60.0,
                max_range=100.0,
                obstacle_contours=contours,
            )

            self._draw_result_annotation(ax, result, (0.02, 0.98))
            ax.legend(loc="lower right", fontsize=8)

        fig.suptitle(
            "Edge Cases: No Obstacles Visible", fontsize=14, fontweight="bold"
        )
        plt.tight_layout()
        save_figure(fig, "api_no_obstacles_visible")

    def test_visual_empty_scene(self):
        """Visual: Empty scene with no obstacles."""
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        viewer = np.array([100.0, 100.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        self._setup_axes(ax, "Empty Scene", xlim=(20, 180), ylim=(20, 220))
        self._draw_viewer(ax, viewer, direction, fov_deg=90.0, max_range=100.0)

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=90.0,
            max_range=100.0,
            obstacle_contours=[],
        )

        self._draw_result_annotation(ax, result, (0.02, 0.98))

        plt.tight_layout()
        save_figure(fig, "api_empty_scene")

    # =========================================================================
    # Test: With Clipped Polygon Visualization
    # =========================================================================

    def test_visual_clipping_effect(self):
        """Visual: Show original vs clipped polygons."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        viewer = np.array([100.0, 100.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)
        fov_deg = 60.0
        max_range = 80.0

        # Large obstacle extending beyond FOV and range
        contour = make_rectangle((100, 170), 150, 80)
        contours = [contour]

        # Compute clipped polygon
        alpha_center = validate_and_get_direction_angle(direction)
        half_fov = np.deg2rad(fov_deg) / 2
        alpha_min = alpha_center - half_fov
        alpha_max = alpha_center + half_fov

        contour_viewer = to_viewer_frame(contour, viewer)
        clipped = clip_polygon_to_wedge(contour_viewer, alpha_min, alpha_max, max_range)

        # Left: Original polygon
        ax = axes[0]
        self._setup_axes(ax, "Original Polygon", xlim=(0, 200), ylim=(60, 260))
        self._draw_viewer(ax, viewer, direction, fov_deg=fov_deg, max_range=max_range)
        self._draw_polygon(
            ax,
            contour,
            color="lightblue",
            edgecolor="blue",
            alpha=0.5,
            label="Original",
            show_vertices=True,
        )
        ax.legend(loc="lower right")

        # Right: Clipped polygon (transformed back to world coords)
        ax = axes[1]
        self._setup_axes(ax, "Clipped Polygon", xlim=(0, 200), ylim=(60, 260))
        self._draw_viewer(ax, viewer, direction, fov_deg=fov_deg, max_range=max_range)

        # Draw original faded
        self._draw_polygon(
            ax, contour, color="lightgray", edgecolor="gray", alpha=0.2, label="Original"
        )

        # Draw clipped (transform back to world coords)
        if clipped is not None and len(clipped) >= 3:
            clipped_world = clipped + viewer
            self._draw_polygon(
                ax,
                clipped_world,
                color="lightgreen",
                edgecolor="darkgreen",
                alpha=0.6,
                label="Clipped",
                show_vertices=True,
            )

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=fov_deg,
            max_range=max_range,
            obstacle_contours=contours,
        )

        self._draw_result_annotation(ax, result, (0.02, 0.98))
        ax.legend(loc="lower right")

        fig.suptitle(
            "Clipping Effect: Original vs Visible Polygon",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        save_figure(fig, "api_clipping_effect")

    # =========================================================================
    # Test: Realistic Scenario
    # =========================================================================

    def test_visual_realistic_retail_scenario(self):
        """Visual: Simulated retail scenario with person looking at products."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))

        # Person in a store aisle
        viewer = np.array([200.0, 100.0], dtype=np.float32)

        # Looking slightly up-left (like looking at a shelf)
        direction = np.array([-0.3, 0.95], dtype=np.float32)
        direction = (direction / np.linalg.norm(direction)).astype(np.float32)

        # Product/obstacle polygons at different positions
        contours = [
            # Close small product
            make_square((170, 130), half_size=12),
            # Medium product on shelf
            make_rectangle((140, 170), 35, 25),
            # Large display farther away
            make_rectangle((100, 200), 60, 40),
            # Product to the side
            make_square((220, 180), half_size=18),
            # Distant signage
            make_rectangle((150, 260), 80, 20),
        ]

        colors = [
            ("lightsalmon", "chocolate"),
            ("lightblue", "steelblue"),
            ("lightgreen", "forestgreen"),
            ("plum", "purple"),
            ("lightyellow", "goldenrod"),
        ]

        self._setup_axes(
            ax, "Retail Scenario: Person Looking at Products", xlim=(50, 350), ylim=(50, 320)
        )
        self._draw_viewer(ax, viewer, direction, fov_deg=45.0, max_range=150.0)

        for i, (contour, (fc, ec)) in enumerate(zip(contours, colors)):
            self._draw_polygon(
                ax, contour, color=fc, edgecolor=ec, alpha=0.5, label=f"Product {i}"
            )

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=45.0,
            max_range=150.0,
            obstacle_contours=contours,
        )

        if result.obstacle_id is not None:
            self._draw_polygon(
                ax,
                contours[result.obstacle_id],
                color="none",
                edgecolor="gold",
                linewidth=5,
                alpha=1.0,
            )

        self._draw_result_annotation(ax, result, (0.02, 0.98))
        ax.legend(loc="lower right", fontsize=9)

        plt.tight_layout()
        save_figure(fig, "api_realistic_retail_scenario")
