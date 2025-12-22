"""
Visual validation tests for real-world integration scenarios (Step 4.2).

These tests create matplotlib figures showing:
- Person looking in various directions (up, left, diagonal)
- Close vs far obstacle comparisons
- Large vs narrow obstacle comparisons
- Obstacles at arc boundaries
- Max range limiting scenarios
- Complex polygon contours

These tests are marked as 'visual' and are NOT run by default.
Run with: pytest -m visual tests/visual/test_api_integration_visual.py -v

Output figures are saved to: tests/visual/output/
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.visual
import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

# Try to import matplotlib, skip tests if not available
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Wedge

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

from view_arc.obstacle.api import find_largest_obstacle, ObstacleResult


# Output directory for visual test results
OUTPUT_DIR = Path(__file__).parent / "output"


@pytest.fixture(scope="module", autouse=True)
def setup_output_dir() -> None:
    """Create output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig: Figure, name: str) -> None:
    """Save figure to output directory."""
    filepath = OUTPUT_DIR / f"{name}.png"
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filepath}")


# =============================================================================
# Helper functions for creating test fixtures
# =============================================================================


def make_triangle(center: Tuple[float, float], size: float = 20.0) -> np.ndarray:
    """Create a triangle centered at given point."""
    cx, cy = center
    return np.array(
        [
            [cx, cy + size],
            [cx - size, cy - size],
            [cx + size, cy - size],
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


def make_polygon(
    center: Tuple[float, float], n_sides: int, radius: float
) -> np.ndarray:
    """Create a regular polygon with n sides centered at given point."""
    cx, cy = center
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    vertices = np.column_stack(
        [cx + radius * np.cos(angles), cy + radius * np.sin(angles)]
    )
    return vertices.astype(np.float32)


def normalize_direction(x: float, y: float) -> np.ndarray:
    """Normalize a direction vector to unit length."""
    vec = np.array([x, y], dtype=np.float32)
    return (vec / np.linalg.norm(vec)).astype(np.float32)


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
class TestAPIIntegrationVisual:
    """Visual validation tests for real-world integration scenarios."""

    def _draw_polygon(
        self,
        ax: Axes,
        polygon: np.ndarray,
        color: str = "blue",
        alpha: float = 0.3,
        label: Optional[str] = None,
        edgecolor: Optional[str] = None,
        linewidth: float = 2,
        show_vertices: bool = False,
    ) -> None:
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
        ax: Axes,
        viewer: np.ndarray,
        direction: np.ndarray,
        fov_deg: float,
        max_range: float,
        alpha: float = 0.15,
    ) -> None:
        """Draw viewer position and field of view wedge."""
        # Draw viewer point
        ax.plot(viewer[0], viewer[1], "ko", markersize=10, zorder=20, label="Viewer")

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
        self, ax: Axes, result: ObstacleResult, position: Tuple[float, float]
    ) -> None:
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
        ax: Axes,
        title: str,
        xlim: Tuple[float, float] = (-50, 150),
        ylim: Tuple[float, float] = (-50, 200),
    ) -> None:
        """Setup axes with grid and labels."""
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(title)

    def _highlight_winner(
        self,
        ax: Axes,
        contours: List[np.ndarray],
        result: ObstacleResult,
    ) -> None:
        """Highlight the winning obstacle with a gold border."""
        if result.obstacle_id is not None:
            self._draw_polygon(
                ax,
                contours[result.obstacle_id],
                color="none",
                edgecolor="gold",
                linewidth=4,
                alpha=1.0,
            )

    # =========================================================================
    # Test: Person Looking in Different Directions
    # =========================================================================

    def test_visual_person_looking_up(self) -> None:
        """Visual: Person looking up (view_direction=[0, 1])."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        viewer = np.array([200.0, 200.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)  # Looking UP

        scenarios = [
            (
                "Single Obstacle Ahead",
                [make_square((200, 280), half_size=25)],
            ),
            (
                "Multiple at Different Distances",
                [
                    make_square((200, 240), half_size=10),
                    make_square((200, 320), half_size=40),
                ],
            ),
            (
                "Obstacles on Either Side",
                [
                    make_square((140, 280), half_size=20),
                    make_square((260, 280), half_size=20),
                ],
            ),
        ]

        colors = [
            ("lightblue", "blue"),
            ("lightcoral", "darkred"),
            ("lightgreen", "darkgreen"),
        ]

        for ax, (title, contours) in zip(axes, scenarios):
            self._setup_axes(ax, title, xlim=(80, 320), ylim=(150, 380))
            self._draw_viewer(ax, viewer, direction, fov_deg=90.0, max_range=150.0)

            for i, contour in enumerate(contours):
                fc, ec = colors[i % len(colors)]
                self._draw_polygon(
                    ax, contour, color=fc, edgecolor=ec, alpha=0.5, label=f"Obstacle {i}"
                )

            result = find_largest_obstacle(
                viewer_point=viewer,
                view_direction=direction,
                field_of_view_deg=90.0,
                max_range=150.0,
                obstacle_contours=contours,
            )

            self._highlight_winner(ax, contours, result)
            self._draw_result_annotation(ax, result, (0.02, 0.98))
            ax.legend(loc="lower right", fontsize=8)

        fig.suptitle(
            "Person Looking UP (direction=[0, 1])",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        save_figure(fig, "integration_person_looking_up")

    def test_visual_person_looking_left(self) -> None:
        """Visual: Person looking left (view_direction=[-1, 0])."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        viewer = np.array([300.0, 200.0], dtype=np.float32)
        direction = np.array([-1.0, 0.0], dtype=np.float32)  # Looking LEFT

        scenarios = [
            (
                "Obstacle to the Left",
                [make_square((200, 200), half_size=25)],
            ),
            (
                "Obstacle Behind (Not Visible)",
                [make_square((400, 200), half_size=25)],
            ),
            (
                "Peripheral Vision Edge",
                [make_square((220, 260), half_size=20)],
            ),
        ]

        colors = [
            ("lightblue", "blue"),
            ("lightcoral", "darkred"),
            ("lightgreen", "darkgreen"),
        ]

        for ax, (title, contours) in zip(axes, scenarios):
            self._setup_axes(ax, title, xlim=(100, 450), ylim=(100, 300))
            self._draw_viewer(ax, viewer, direction, fov_deg=60.0, max_range=150.0)

            for i, contour in enumerate(contours):
                fc, ec = colors[i % len(colors)]
                self._draw_polygon(
                    ax, contour, color=fc, edgecolor=ec, alpha=0.5, label=f"Obstacle {i}"
                )

            result = find_largest_obstacle(
                viewer_point=viewer,
                view_direction=direction,
                field_of_view_deg=60.0,
                max_range=150.0,
                obstacle_contours=contours,
            )

            self._highlight_winner(ax, contours, result)
            self._draw_result_annotation(ax, result, (0.02, 0.98))
            ax.legend(loc="lower right", fontsize=8)

        fig.suptitle(
            "Person Looking LEFT (direction=[-1, 0])",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        save_figure(fig, "integration_person_looking_left")

    def test_visual_person_looking_diagonal(self) -> None:
        """Visual: Person looking diagonal (view_direction=[-0.37, 0.92] normalized)."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        viewer = np.array([200.0, 200.0], dtype=np.float32)
        direction = normalize_direction(-0.37, 0.92)  # Up-left diagonal

        # Calculate position along the direction for target placement
        target_dist = 80.0
        target_x = 200.0 + direction[0] * target_dist
        target_y = 200.0 + direction[1] * target_dist

        scenarios = [
            (
                "Obstacle in Diagonal Direction",
                [make_square((140, 300), half_size=25)],
            ),
            (
                "Off-Axis Obstacles",
                [
                    make_square((120, 280), half_size=15),
                    make_square((250, 320), half_size=20),
                    make_square((100, 350), half_size=18),
                ],
            ),
            (
                "Precisely Along Direction",
                [make_square((target_x, target_y), half_size=20)],
            ),
        ]

        colors = [
            ("lightblue", "blue"),
            ("lightcoral", "darkred"),
            ("lightgreen", "darkgreen"),
        ]

        for ax, (title, contours) in zip(axes, scenarios):
            self._setup_axes(ax, title, xlim=(50, 350), ylim=(150, 400))
            self._draw_viewer(ax, viewer, direction, fov_deg=60.0, max_range=150.0)

            for i, contour in enumerate(contours):
                fc, ec = colors[i % len(colors)]
                self._draw_polygon(
                    ax, contour, color=fc, edgecolor=ec, alpha=0.5, label=f"Obstacle {i}"
                )

            result = find_largest_obstacle(
                viewer_point=viewer,
                view_direction=direction,
                field_of_view_deg=60.0,
                max_range=150.0,
                obstacle_contours=contours,
            )

            self._highlight_winner(ax, contours, result)
            self._draw_result_annotation(ax, result, (0.02, 0.98))
            ax.legend(loc="lower right", fontsize=8)

        fig.suptitle(
            "Person Looking DIAGONAL (direction≈[-0.37, 0.92])",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        save_figure(fig, "integration_person_looking_diagonal")

    # =========================================================================
    # Test: Close vs Far Obstacles
    # =========================================================================

    def test_visual_close_vs_far_obstacles(self) -> None:
        """Visual: Distance-based obstacle selection scenarios."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        viewer = np.array([100.0, 100.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        scenarios = [
            (
                "Closer Occludes Farther",
                [
                    make_square((100, 130), half_size=8),
                    make_square((100, 200), half_size=30),
                ],
            ),
            (
                "Tie-Breaking: Closer Wins",
                [
                    make_square((-10, 140), half_size=10),  # Close left
                    make_square((50, 180), half_size=20),  # Far right
                ],
            ),
            (
                "Very Close = Large Coverage",
                [make_square((100, 115), half_size=10)],
            ),
        ]

        colors = [
            ("lightcoral", "darkred"),
            ("lightblue", "blue"),
        ]

        for ax, (title, contours) in zip(axes, scenarios):
            self._setup_axes(ax, title, xlim=(-50, 200), ylim=(50, 260))
            self._draw_viewer(ax, viewer, direction, fov_deg=90.0, max_range=150.0)

            for i, contour in enumerate(contours):
                fc, ec = colors[i % len(colors)]
                self._draw_polygon(
                    ax, contour, color=fc, edgecolor=ec, alpha=0.5, label=f"Obstacle {i}"
                )

            result = find_largest_obstacle(
                viewer_point=viewer,
                view_direction=direction,
                field_of_view_deg=90.0,
                max_range=150.0,
                obstacle_contours=contours,
            )

            self._highlight_winner(ax, contours, result)
            self._draw_result_annotation(ax, result, (0.02, 0.98))
            ax.legend(loc="lower right", fontsize=8)

        fig.suptitle(
            "Close vs Far Obstacles (Distance-Based Selection)",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        save_figure(fig, "integration_close_vs_far")

    # =========================================================================
    # Test: Large vs Narrow Obstacles
    # =========================================================================

    def test_visual_large_vs_narrow_obstacles(self) -> None:
        """Visual: Angular coverage comparison between obstacle shapes."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        scenarios = [
            (
                "Wide vs Narrow at Same Distance",
                [
                    make_rectangle((-40, 60), width=60, height=10),  # Wide
                    make_rectangle((40, 60), width=10, height=40),  # Narrow
                ],
            ),
            (
                "Tall vs Wide Obstacle",
                [
                    make_rectangle((0, 70), width=10, height=50),  # Tall narrow
                    make_rectangle((0, 40), width=50, height=10),  # Wide short
                ],
            ),
            (
                "Small Filling Narrow FOV",
                [make_square((0, 30), half_size=8)],
            ),
        ]

        colors = [
            ("lightblue", "blue"),
            ("lightcoral", "darkred"),
        ]

        fov_settings = [90.0, 90.0, 20.0]  # Third scenario uses narrow FOV

        for ax, (title, contours), fov in zip(axes, scenarios, fov_settings):
            self._setup_axes(ax, title, xlim=(-100, 100), ylim=(-20, 140))
            self._draw_viewer(ax, viewer, direction, fov_deg=fov, max_range=100.0)

            for i, contour in enumerate(contours):
                fc, ec = colors[i % len(colors)]
                self._draw_polygon(
                    ax, contour, color=fc, edgecolor=ec, alpha=0.5, label=f"Obstacle {i}"
                )

            result = find_largest_obstacle(
                viewer_point=viewer,
                view_direction=direction,
                field_of_view_deg=fov,
                max_range=100.0,
                obstacle_contours=contours,
            )

            self._highlight_winner(ax, contours, result)
            self._draw_result_annotation(ax, result, (0.02, 0.98))
            ax.legend(loc="lower right", fontsize=8)

        fig.suptitle(
            "Large vs Narrow Obstacles (Angular Coverage Comparison)",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        save_figure(fig, "integration_large_vs_narrow")

    # =========================================================================
    # Test: Obstacles at Arc Boundary
    # =========================================================================

    def test_visual_obstacle_at_arc_boundary(self) -> None:
        """Visual: Obstacles at the edge of the field of view."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 14))

        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        scenarios = [
            (
                "Partially in FOV (Wide)",
                [make_rectangle((0, 50), width=200, height=20)],
                60.0,
            ),
            (
                "At Left Edge",
                [make_square((-25, 55), half_size=15)],
                60.0,
            ),
            (
                "At Right Edge",
                [make_square((25, 55), half_size=15)],
                60.0,
            ),
            (
                "Just Outside FOV",
                [make_square((100, 20), half_size=10)],
                30.0,
            ),
        ]

        colors = [("lightgreen", "darkgreen")]

        for ax, (title, contours, fov) in zip(axes.flat, scenarios):
            self._setup_axes(ax, title, xlim=(-120, 120), ylim=(-20, 100))
            self._draw_viewer(ax, viewer, direction, fov_deg=fov, max_range=100.0)

            for i, contour in enumerate(contours):
                fc, ec = colors[i % len(colors)]
                self._draw_polygon(
                    ax, contour, color=fc, edgecolor=ec, alpha=0.5, label=f"Obstacle {i}"
                )

            result = find_largest_obstacle(
                viewer_point=viewer,
                view_direction=direction,
                field_of_view_deg=fov,
                max_range=100.0,
                obstacle_contours=contours,
            )

            self._highlight_winner(ax, contours, result)
            self._draw_result_annotation(ax, result, (0.02, 0.98))
            ax.legend(loc="lower right", fontsize=8)

        fig.suptitle(
            "Obstacles at Arc Boundary (Partial Visibility)",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        save_figure(fig, "integration_arc_boundary")

    # =========================================================================
    # Test: Max Range Limiting
    # =========================================================================

    def test_visual_max_range_limit(self) -> None:
        """Visual: Obstacles rejected due to max range."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 14))

        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        scenarios = [
            (
                "Beyond Max Range",
                [make_square((0, 100), half_size=20)],
                50.0,
            ),
            (
                "Partially Beyond Range",
                [make_rectangle((0, 60), width=30, height=40)],
                70.0,
            ),
            (
                "Exactly at Max Range",
                [make_square((0, 50), half_size=10)],
                45.0,
            ),
            (
                "Mix: Some Beyond Range",
                [
                    make_square((0, 40), half_size=10),  # In range
                    make_square((30, 150), half_size=20),  # Beyond
                    make_square((-30, 60), half_size=12),  # In range
                ],
                80.0,
            ),
        ]

        colors = [
            ("lightblue", "blue"),
            ("lightcoral", "darkred"),
            ("lightgreen", "darkgreen"),
        ]

        for ax, (title, contours, max_range) in zip(axes.flat, scenarios):
            self._setup_axes(ax, title, xlim=(-80, 80), ylim=(-20, 200))
            self._draw_viewer(ax, viewer, direction, fov_deg=90.0, max_range=max_range)

            for i, contour in enumerate(contours):
                fc, ec = colors[i % len(colors)]
                self._draw_polygon(
                    ax, contour, color=fc, edgecolor=ec, alpha=0.5, label=f"Obstacle {i}"
                )

            result = find_largest_obstacle(
                viewer_point=viewer,
                view_direction=direction,
                field_of_view_deg=90.0,
                max_range=max_range,
                obstacle_contours=contours,
            )

            self._highlight_winner(ax, contours, result)
            self._draw_result_annotation(ax, result, (0.02, 0.98))
            ax.legend(loc="lower right", fontsize=8)

        fig.suptitle(
            "Max Range Limiting (Distant Obstacles Rejected)",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        save_figure(fig, "integration_max_range")

    # =========================================================================
    # Test: Complex Polygon Contours
    # =========================================================================

    def test_visual_complex_contours(self) -> None:
        """Visual: Obstacles with complex polygon shapes."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 14))

        viewer = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)

        scenarios = [
            (
                "Octagon",
                [make_polygon((0, 60), n_sides=8, radius=20)],
            ),
            (
                "Hexagon",
                [make_polygon((0, 60), n_sides=6, radius=25)],
            ),
            (
                "Circle Approximation (32-gon)",
                [make_polygon((0, 50), n_sides=32, radius=15)],
            ),
            (
                "Multiple Complex Polygons",
                [
                    make_polygon((-40, 60), n_sides=5, radius=15),  # Pentagon
                    make_polygon((0, 80), n_sides=12, radius=20),  # Dodecagon
                    make_polygon((40, 55), n_sides=7, radius=18),  # Heptagon
                ],
            ),
        ]

        colors = [
            ("lightblue", "blue"),
            ("lightcoral", "darkred"),
            ("lightgreen", "darkgreen"),
        ]

        for ax, (title, contours) in zip(axes.flat, scenarios):
            self._setup_axes(ax, title, xlim=(-80, 80), ylim=(-20, 120))
            self._draw_viewer(ax, viewer, direction, fov_deg=90.0, max_range=100.0)

            for i, contour in enumerate(contours):
                fc, ec = colors[i % len(colors)]
                self._draw_polygon(
                    ax,
                    contour,
                    color=fc,
                    edgecolor=ec,
                    alpha=0.5,
                    label=f"Obstacle {i}",
                    show_vertices=True,
                )

            result = find_largest_obstacle(
                viewer_point=viewer,
                view_direction=direction,
                field_of_view_deg=90.0,
                max_range=100.0,
                obstacle_contours=contours,
            )

            self._highlight_winner(ax, contours, result)
            self._draw_result_annotation(ax, result, (0.02, 0.98))
            ax.legend(loc="lower right", fontsize=8)

        fig.suptitle(
            "Complex Polygon Contours (Many Vertices)",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        save_figure(fig, "integration_complex_contours")

    # =========================================================================
    # Test: Edge Cases
    # =========================================================================

    def test_visual_edge_cases(self) -> None:
        """Visual: Edge cases and boundary conditions."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 14))

        scenarios = [
            (
                "Very Small Obstacle",
                np.array([0.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0], dtype=np.float32),
                [make_square((0, 50), half_size=1)],
                90.0,
                100.0,
            ),
            (
                "Very Large Obstacle",
                np.array([0.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0], dtype=np.float32),
                [make_rectangle((0, 50), width=500, height=100)],
                90.0,
                100.0,
            ),
            (
                "Minimal Triangle",
                np.array([0.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0], dtype=np.float32),
                [
                    np.array(
                        [[0, 40], [10, 60], [-10, 55]],
                        dtype=np.float32,
                    )
                ],
                90.0,
                100.0,
            ),
            (
                "Many Obstacles (Grid)",
                np.array([0.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0], dtype=np.float32),
                [
                    make_square((-80 + i * 40, 30 + j * 35), half_size=8)
                    for i in range(5)
                    for j in range(3)
                ],
                180.0,
                150.0,
            ),
        ]

        colors = [
            ("lightblue", "blue"),
            ("lightcoral", "darkred"),
            ("lightgreen", "darkgreen"),
            ("plum", "purple"),
            ("lightyellow", "orange"),
        ]

        for ax, (title, viewer, direction, contours, fov, max_range) in zip(
            axes.flat, scenarios
        ):
            # Adjust view limits based on scenario
            if "Large" in title:
                xlim, ylim = (-300, 300), (-50, 200)
            elif "Grid" in title:
                xlim, ylim = (-120, 120), (-20, 180)
            else:
                xlim, ylim = (-80, 80), (-20, 100)

            self._setup_axes(ax, title, xlim=xlim, ylim=ylim)
            self._draw_viewer(ax, viewer, direction, fov_deg=fov, max_range=max_range)

            for i, contour in enumerate(contours):
                fc, ec = colors[i % len(colors)]
                self._draw_polygon(
                    ax,
                    contour,
                    color=fc,
                    edgecolor=ec,
                    alpha=0.5,
                    label=f"Obs {i}" if i < 3 else None,
                )

            result = find_largest_obstacle(
                viewer_point=viewer,
                view_direction=direction,
                field_of_view_deg=fov,
                max_range=max_range,
                obstacle_contours=contours,
            )

            self._highlight_winner(ax, contours, result)
            self._draw_result_annotation(ax, result, (0.02, 0.98))
            if len(contours) <= 5:
                ax.legend(loc="lower right", fontsize=8)

        fig.suptitle(
            "Edge Cases and Boundary Conditions",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        save_figure(fig, "integration_edge_cases")

    # =========================================================================
    # Test: Comprehensive Summary Scene
    # =========================================================================

    def test_visual_comprehensive_scene(self) -> None:
        """Visual: A comprehensive scene combining multiple real-world aspects."""
        fig, ax = plt.subplots(1, 1, figsize=(14, 12))

        # Person in the middle of a room looking up-right
        viewer = np.array([150.0, 150.0], dtype=np.float32)
        direction = normalize_direction(0.5, 0.866)  # 60° (up-right)

        # Various obstacles representing a realistic scene
        contours = [
            # Nearby small object
            make_square((180, 180), half_size=8),
            # Medium object in the direction
            make_rectangle((220, 250), width=40, height=20),
            # Large object to the side
            make_polygon((120, 240), n_sides=6, radius=25),
            # Far object (might be partially in range)
            make_square((280, 300), half_size=30),
            # Object at FOV edge
            make_triangle((250, 200), size=15),
        ]

        colors = [
            ("lightcoral", "darkred"),
            ("lightblue", "blue"),
            ("lightgreen", "darkgreen"),
            ("lightyellow", "orange"),
            ("plum", "purple"),
        ]

        self._setup_axes(
            ax, "Comprehensive Real-World Scene", xlim=(50, 350), ylim=(100, 350)
        )
        self._draw_viewer(ax, viewer, direction, fov_deg=90.0, max_range=150.0)

        for i, (contour, (fc, ec)) in enumerate(zip(contours, colors)):
            self._draw_polygon(
                ax,
                contour,
                color=fc,
                edgecolor=ec,
                alpha=0.5,
                label=f"Obstacle {i}",
                show_vertices=True,
            )

        result = find_largest_obstacle(
            viewer_point=viewer,
            view_direction=direction,
            field_of_view_deg=90.0,
            max_range=150.0,
            obstacle_contours=contours,
            return_intervals=True,
        )

        self._highlight_winner(ax, contours, result)
        self._draw_result_annotation(ax, result, (0.02, 0.98))
        ax.legend(loc="lower right", fontsize=9)

        # Add additional info about intervals if available
        if result.intervals:
            interval_text = "Intervals:\n"
            for i, (start, end) in enumerate(result.intervals):
                interval_text += f"  {i}: [{np.rad2deg(start):.1f}°, {np.rad2deg(end):.1f}°]\n"
            ax.annotate(
                interval_text,
                (0.02, 0.75),
                xycoords="axes fraction",
                fontsize=9,
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment="top",
            )

        plt.tight_layout()
        save_figure(fig, "integration_comprehensive_scene")
