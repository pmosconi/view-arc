"""
Visual validation tests for angular sweep operations (Step 3.1).

These tests create matplotlib figures showing:
- Polygon obstacles with vertices and edges
- Query rays and active edges
- Angular events along the arc
- Event ordering visualization

These tests are marked as 'visual' and are NOT run by default.
Run with: pytest -m visual tests/visual/test_sweep_visual.py -v

Output figures are saved to: tests/visual/output/
"""

import pytest

pytestmark = pytest.mark.visual
import numpy as np
from pathlib import Path

# Try to import matplotlib, skip tests if not available
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Wedge, FancyArrowPatch
    from matplotlib.collections import LineCollection
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from view_arc.obstacle.sweep import (
    AngularEvent,
    IntervalResult,
    get_active_edges,
    build_events,
    resolve_interval,
    compute_coverage,
)
from view_arc.obstacle.geometry import to_polar, intersect_ray_segment


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


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
class TestSweepVisual:
    """Visual validation tests for sweep operations."""

    def _draw_polygon(self, ax, polygon, color='blue', alpha=0.3, label=None, 
                      edgecolor=None, show_vertices=True, show_vertex_labels=True):
        """Draw a polygon on the axes with optional vertex markers and labels."""
        if polygon is None or len(polygon) < 3:
            return
        if edgecolor is None:
            edgecolor = color
        patch = mpatches.Polygon(polygon, closed=True, 
                                  facecolor=color, alpha=alpha, 
                                  edgecolor=edgecolor, linewidth=2,
                                  label=label)
        ax.add_patch(patch)
        
        if show_vertices:
            ax.scatter(polygon[:, 0], polygon[:, 1], color=edgecolor, 
                      s=50, zorder=5)
            
        if show_vertex_labels:
            for i, (x, y) in enumerate(polygon):
                ax.annotate(f'v{i}', (x, y), textcoords="offset points", 
                           xytext=(5, 5), fontsize=8, color=edgecolor)

    def _draw_ray(self, ax, angle, length=4.0, color='red', label=None, linewidth=2):
        """Draw a ray from origin at given angle."""
        x_end = length * np.cos(angle)
        y_end = length * np.sin(angle)
        ax.arrow(0, 0, x_end * 0.95, y_end * 0.95, 
                 head_width=0.1, head_length=0.08,
                 fc=color, ec=color, linewidth=linewidth, label=label)

    def _draw_arc_sector(self, ax, alpha_min, alpha_max, radius=3, 
                         color='yellow', alpha=0.2, label=None):
        """Draw an arc sector (wedge) showing the field of view."""
        # Convert to degrees for matplotlib
        theta1 = np.rad2deg(alpha_min)
        theta2 = np.rad2deg(alpha_max)
        
        # Handle wrap-around
        if alpha_min > alpha_max:
            # Draw two wedges
            wedge1 = Wedge((0, 0), radius, theta1, 180, 
                          facecolor=color, alpha=alpha, edgecolor='orange', linewidth=1)
            wedge2 = Wedge((0, 0), radius, -180, theta2, 
                          facecolor=color, alpha=alpha, edgecolor='orange', linewidth=1)
            ax.add_patch(wedge1)
            ax.add_patch(wedge2)
        else:
            wedge = Wedge((0, 0), radius, theta1, theta2, 
                         facecolor=color, alpha=alpha, edgecolor='orange', 
                         linewidth=1, label=label)
            ax.add_patch(wedge)

    def _draw_active_edges(self, ax, edges, color='red', linewidth=3, label=None):
        """Draw active edges with highlighting."""
        if edges.shape[0] == 0:
            return
        
        for i, edge in enumerate(edges):
            lbl = label if i == 0 else None
            ax.plot([edge[0, 0], edge[1, 0]], [edge[0, 1], edge[1, 1]], 
                   color=color, linewidth=linewidth, label=lbl, zorder=10)

    def _setup_axes(self, ax, title, xlim=(-4, 5), ylim=(-4, 5)):
        """Setup axes with grid and labels."""
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(title)
        ax.plot(0, 0, 'ko', markersize=8, zorder=20)  # Origin marker

    # =========================================================================
    # Test: get_active_edges()
    # =========================================================================

    def test_visual_active_edges_single_polygon(self):
        """Visual: Show active edges for a single polygon at various query angles."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Square obstacle
        square = np.array([
            [2.0, -1.0],
            [4.0, -1.0],
            [4.0, 1.0],
            [2.0, 1.0],
        ], dtype=np.float32)
        
        # Test at different query angles
        query_angles = [0.0, np.pi/6, np.pi/4, np.pi/2, np.pi, -np.pi/4]
        angle_names = ['0°', '30°', '45°', '90°', '180°', '-45°']
        
        for ax, angle, name in zip(axes, query_angles, angle_names):
            self._setup_axes(ax, f"Query angle: {name}")
            
            # Draw polygon
            self._draw_polygon(ax, square, color='lightblue', alpha=0.4, 
                             edgecolor='blue', label='Polygon')
            
            # Draw query ray
            self._draw_ray(ax, angle, length=5, color='green', label='Query ray')
            
            # Get and draw active edges
            active = get_active_edges(square, angle)
            self._draw_active_edges(ax, active, color='red', label=f'Active ({len(active)} edges)')
            
            ax.legend(loc='upper left', fontsize=8)
        
        fig.suptitle("Active Edges: Ray-Polygon Intersection", fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_figure(fig, "sweep_active_edges_single")

    def test_visual_active_edges_triangle(self):
        """Visual: Active edges for a triangle at various angles."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Triangle obstacle
        triangle = np.array([
            [3.0, 0.0],
            [5.0, 2.0],
            [5.0, -2.0],
        ], dtype=np.float32)
        
        # Test at different query angles
        query_angles = [0.0, np.pi/8, np.pi/4, -np.pi/8, np.pi/2, np.pi*3/4]
        angle_names = ['0°', '22.5°', '45°', '-22.5°', '90°', '135°']
        
        for ax, angle, name in zip(axes, query_angles, angle_names):
            self._setup_axes(ax, f"Query angle: {name}", xlim=(-2, 7), ylim=(-4, 4))
            
            # Draw polygon
            self._draw_polygon(ax, triangle, color='lightgreen', alpha=0.4, 
                             edgecolor='darkgreen', label='Triangle')
            
            # Draw query ray
            self._draw_ray(ax, angle, length=6, color='purple', label='Query ray')
            
            # Get and draw active edges
            active = get_active_edges(triangle, angle)
            self._draw_active_edges(ax, active, color='red', linewidth=4, 
                                   label=f'Active ({len(active)} edges)')
            
            ax.legend(loc='upper left', fontsize=8)
        
        fig.suptitle("Active Edges: Triangle Obstacle", fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_figure(fig, "sweep_active_edges_triangle")

    def test_visual_active_edges_no_intersection(self):
        """Visual: Verify no active edges when ray misses polygon."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Polygon in first quadrant
        polygon = np.array([
            [2.0, 1.0],
            [4.0, 1.0],
            [3.0, 3.0],
        ], dtype=np.float32)
        
        # Rays that miss the polygon
        miss_angles = [-np.pi/2, np.pi, -np.pi/4]
        angle_names = ['-90° (down)', '180° (left)', '-45° (down-right)']
        
        for ax, angle, name in zip(axes, miss_angles, angle_names):
            self._setup_axes(ax, f"Query: {name}")
            
            # Draw polygon
            self._draw_polygon(ax, polygon, color='lightyellow', alpha=0.5, 
                             edgecolor='orange', label='Polygon')
            
            # Draw query ray
            self._draw_ray(ax, angle, length=5, color='red', label='Query ray (miss)')
            
            # Get active edges
            active = get_active_edges(polygon, angle)
            
            # Annotate result
            ax.annotate(f'Active edges: {len(active)}', (0.5, 0.95), 
                       xycoords='axes fraction', fontsize=12, 
                       color='red' if len(active) == 0 else 'green',
                       fontweight='bold')
            
            ax.legend(loc='upper left', fontsize=8)
        
        fig.suptitle("Active Edges: Ray Misses Polygon", fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_figure(fig, "sweep_active_edges_miss")

    # =========================================================================
    # Test: build_events()
    # =========================================================================

    def test_visual_build_events_single_triangle(self):
        """Visual: Show events generated for a single triangle."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Triangle in the arc
        triangle = np.array([
            [2.0, 0.0],
            [3.0, 1.5],
            [1.5, 1.0],
        ], dtype=np.float32)
        
        alpha_min = -np.pi/4
        alpha_max = np.pi/2
        
        # Left plot: Spatial view
        ax = axes[0]
        self._setup_axes(ax, "Spatial View: Triangle in Arc", xlim=(-1, 5), ylim=(-2, 3))
        
        # Draw arc sector
        self._draw_arc_sector(ax, alpha_min, alpha_max, radius=4, label='FOV Arc')
        
        # Draw arc boundaries
        self._draw_ray(ax, alpha_min, length=4, color='orange', linewidth=1, label='α_min')
        self._draw_ray(ax, alpha_max, length=4, color='orange', linewidth=1, label='α_max')
        
        # Draw polygon
        self._draw_polygon(ax, triangle, color='lightblue', alpha=0.5, 
                          edgecolor='blue', label='Triangle')
        
        # Build events
        events = build_events([(0, triangle)], alpha_min, alpha_max)
        
        # Draw rays to vertices (for vertex events)
        vertex_events = [e for e in events if e.event_type == 'vertex']
        for e in vertex_events:
            self._draw_ray(ax, e.angle, length=4, color='green', linewidth=1)
        
        ax.legend(loc='upper left', fontsize=8)
        
        # Right plot: Angular view (events on angle axis)
        ax = axes[1]
        ax.set_xlim(-1, 2.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_xlabel('Angle (radians)')
        ax.set_title('Events on Angular Axis')
        
        # Draw angle axis
        ax.axhline(y=0, color='black', linewidth=2)
        ax.axhline(y=1, color='black', linewidth=2, linestyle='--', alpha=0.3)
        
        # Mark alpha_min and alpha_max
        ax.axvline(x=alpha_min, color='orange', linewidth=2, linestyle='--', label='α_min')
        ax.axvline(x=alpha_max, color='orange', linewidth=2, linestyle='--', label='α_max')
        
        # Plot events
        vertex_angles = [e.angle for e in events if e.event_type == 'vertex']
        edge_angles = [e.angle for e in events if e.event_type == 'edge_crossing']
        
        ax.scatter(vertex_angles, [0]*len(vertex_angles), s=100, color='blue', 
                  marker='o', label=f'Vertex events ({len(vertex_angles)})', zorder=10)
        ax.scatter(edge_angles, [0]*len(edge_angles), s=100, color='red', 
                  marker='x', label=f'Edge events ({len(edge_angles)})', zorder=10)
        
        # Annotate vertex indices
        for e in events:
            if e.event_type == 'vertex':
                ax.annotate(f'v{e.vertex_idx}', (e.angle, 0), 
                           textcoords="offset points", xytext=(0, 15),
                           fontsize=10, ha='center')
        
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        fig.suptitle(f"build_events(): {len(events)} events from triangle", 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_figure(fig, "sweep_build_events_triangle")

    def test_visual_build_events_multiple_obstacles(self):
        """Visual: Show events from multiple obstacles with different colors."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Two obstacles
        triangle = np.array([
            [2.0, 0.5],
            [3.0, 1.5],
            [2.0, 1.5],
        ], dtype=np.float32)
        
        square = np.array([
            [1.0, -1.5],
            [2.5, -1.5],
            [2.5, -0.5],
            [1.0, -0.5],
        ], dtype=np.float32)
        
        alpha_min = -np.pi/2
        alpha_max = np.pi/2
        
        # Left plot: Spatial view
        ax = axes[0]
        self._setup_axes(ax, "Two Obstacles in Arc", xlim=(-1, 4), ylim=(-3, 3))
        
        # Draw arc sector
        self._draw_arc_sector(ax, alpha_min, alpha_max, radius=4)
        
        # Draw arc boundaries
        self._draw_ray(ax, alpha_min, length=4, color='orange', linewidth=1)
        self._draw_ray(ax, alpha_max, length=4, color='orange', linewidth=1)
        
        # Draw obstacles
        self._draw_polygon(ax, triangle, color='lightblue', alpha=0.5, 
                          edgecolor='blue', label='Obstacle 0')
        self._draw_polygon(ax, square, color='lightgreen', alpha=0.5, 
                          edgecolor='green', label='Obstacle 1')
        
        ax.legend(loc='upper left', fontsize=8)
        
        # Build events
        events = build_events([(0, triangle), (1, square)], alpha_min, alpha_max)
        
        # Right plot: Events by obstacle
        ax = axes[1]
        ax.set_xlim(-2, 2)
        ax.set_ylim(-0.5, 2.5)
        ax.set_xlabel('Angle (radians)')
        ax.set_ylabel('Obstacle ID')
        ax.set_title('Events by Obstacle')
        
        # Mark boundaries
        ax.axvline(x=alpha_min, color='orange', linewidth=2, linestyle='--')
        ax.axvline(x=alpha_max, color='orange', linewidth=2, linestyle='--')
        
        # Plot events with y-position by obstacle_id
        colors = ['blue', 'green']
        for e in events:
            y = e.obstacle_id
            color = colors[e.obstacle_id % len(colors)]
            marker = 'o' if e.event_type == 'vertex' else 'x'
            ax.scatter(e.angle, y, s=120, color=color, marker=marker, zorder=10)
            if e.event_type == 'vertex':
                ax.annotate(f'v{e.vertex_idx}', (e.angle, y), 
                           textcoords="offset points", xytext=(0, 10),
                           fontsize=9, ha='center', color=color)
        
        # Add legend
        ax.scatter([], [], s=100, color='gray', marker='o', label='Vertex')
        ax.scatter([], [], s=100, color='gray', marker='x', label='Edge crossing')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Obstacle 0', 'Obstacle 1'])
        
        fig.suptitle(f"build_events(): {len(events)} events from 2 obstacles", 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_figure(fig, "sweep_build_events_multiple")

    def test_visual_event_sorting(self):
        """Visual: Demonstrate event sorting - vertex before edge_crossing at same angle."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Create events manually to show sorting
        events = [
            AngularEvent(angle=0.5, obstacle_id=0, event_type='edge_crossing'),
            AngularEvent(angle=0.3, obstacle_id=1, event_type='vertex', vertex_idx=0),
            AngularEvent(angle=0.5, obstacle_id=2, event_type='vertex', vertex_idx=1),
            AngularEvent(angle=0.3, obstacle_id=3, event_type='edge_crossing'),
            AngularEvent(angle=0.7, obstacle_id=0, event_type='vertex', vertex_idx=2),
        ]
        
        # Sort events
        sorted_events = sorted(events)
        
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-1, len(sorted_events) + 1)
        ax.set_xlabel('Angle (radians)', fontsize=12)
        ax.set_ylabel('Processing Order', fontsize=12)
        ax.set_title('Event Sorting: Vertex before Edge-Crossing at Same Angle', fontsize=12)
        
        # Plot sorted events
        for i, e in enumerate(sorted_events):
            color = 'blue' if e.event_type == 'vertex' else 'red'
            marker = 'o' if e.event_type == 'vertex' else 'x'
            ax.scatter(e.angle, i, s=200, color=color, marker=marker, zorder=10)
            ax.annotate(f'{e.event_type} (obs {e.obstacle_id})', (e.angle, i),
                       textcoords="offset points", xytext=(10, 0),
                       fontsize=10, va='center')
            
            # Draw line connecting processing order
            if i > 0:
                prev = sorted_events[i-1]
                ax.plot([prev.angle, e.angle], [i-1, i], 
                       color='gray', linewidth=1, linestyle='--', alpha=0.5)
        
        # Highlight same-angle groups
        ax.axvline(x=0.3, color='green', alpha=0.3, linewidth=20)
        ax.axvline(x=0.5, color='green', alpha=0.3, linewidth=20)
        ax.annotate('Same angle\nvertex first', (0.3, -0.5), ha='center', fontsize=9, color='green')
        ax.annotate('Same angle\nvertex first', (0.5, -0.5), ha='center', fontsize=9, color='green')
        
        # Legend
        ax.scatter([], [], s=150, color='blue', marker='o', label='Vertex event')
        ax.scatter([], [], s=150, color='red', marker='x', label='Edge-crossing event')
        ax.legend(loc='upper right', fontsize=10)
        
        ax.grid(True, alpha=0.3)
        ax.set_yticks(range(len(sorted_events)))
        
        plt.tight_layout()
        save_figure(fig, "sweep_event_sorting")

    def test_visual_arc_crossing_boundary(self):
        """Visual: Events when arc crosses ±π boundary."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Triangle near ±π boundary
        triangle = np.array([
            [-2.0, 0.2],
            [-2.0, -0.2],
            [-3.0, 0.0],
        ], dtype=np.float32)
        
        # Arc that crosses ±π (from 150° to -150°)
        alpha_min = 150 * np.pi / 180   # 2.618 rad
        alpha_max = -150 * np.pi / 180  # -2.618 rad
        
        # Left plot: Spatial view
        ax = axes[0]
        self._setup_axes(ax, "Arc Crossing ±π Boundary", xlim=(-4, 2), ylim=(-3, 3))
        
        # Draw arc sector (wraps around)
        self._draw_arc_sector(ax, alpha_min, alpha_max, radius=4, label='FOV Arc')
        
        # Draw arc boundaries
        self._draw_ray(ax, alpha_min, length=4, color='orange', linewidth=2, label='α_min (150°)')
        self._draw_ray(ax, alpha_max, length=4, color='red', linewidth=2, label='α_max (-150°)')
        self._draw_ray(ax, np.pi, length=4, color='purple', linewidth=1, label='±π line')
        
        # Draw polygon
        self._draw_polygon(ax, triangle, color='lightblue', alpha=0.5, 
                          edgecolor='blue', label='Triangle')
        
        ax.legend(loc='upper right', fontsize=8)
        
        # Build events
        events = build_events([(0, triangle)], alpha_min, alpha_max)
        
        # Right plot: Angular distribution
        ax = axes[1]
        ax.set_xlim(-4, 4)
        ax.set_ylim(-0.5, 1.5)
        ax.set_xlabel('Angle (radians)')
        ax.set_title(f'Events: {len(events)} total')
        
        # Draw full angle range
        ax.axhline(y=0, color='black', linewidth=2)
        
        # Mark key angles
        ax.axvline(x=-np.pi, color='purple', linewidth=2, linestyle=':', label='-π')
        ax.axvline(x=np.pi, color='purple', linewidth=2, linestyle=':', label='+π')
        ax.axvline(x=alpha_min, color='orange', linewidth=2, linestyle='--', label='α_min')
        ax.axvline(x=alpha_max, color='red', linewidth=2, linestyle='--', label='α_max')
        
        # Shade the arc region
        ax.axvspan(alpha_min, np.pi, color='yellow', alpha=0.2)
        ax.axvspan(-np.pi, alpha_max, color='yellow', alpha=0.2)
        
        # Plot events
        for e in events:
            marker = 'o' if e.event_type == 'vertex' else 'x'
            ax.scatter(e.angle, 0, s=150, color='blue', marker=marker, zorder=10)
            ax.annotate(f'v{e.vertex_idx}' if e.event_type == 'vertex' else 'edge',
                       (e.angle, 0), textcoords="offset points", xytext=(0, 15),
                       fontsize=10, ha='center')
        
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        fig.suptitle("build_events(): Arc Crossing ±π Discontinuity", 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_figure(fig, "sweep_arc_crossing_pi")

    def test_visual_vertex_angles(self):
        """Visual: Show polar angles of polygon vertices."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Create a polygon
        polygon = np.array([
            [2.0, 0.0],    # 0°
            [1.5, 1.5],    # 45°
            [0.0, 2.0],    # 90°
            [-1.0, 1.0],   # 135°
            [-1.5, -0.5],  # ~160°
        ], dtype=np.float32)
        
        # Left plot: Spatial view with angle annotations
        ax = axes[0]
        self._setup_axes(ax, "Polygon Vertices with Polar Angles", xlim=(-3, 4), ylim=(-2, 4))
        
        # Draw polygon
        self._draw_polygon(ax, polygon, color='lightblue', alpha=0.4, 
                          edgecolor='blue', show_vertex_labels=False)
        
        # Get polar coordinates
        radii, angles = to_polar(polygon)
        
        # Draw rays to each vertex with angle annotation
        cmap = plt.get_cmap('rainbow')
        colors = [cmap(x) for x in np.linspace(0, 1, len(polygon))]
        for i, (r, a, c) in enumerate(zip(radii, angles, colors)):
            self._draw_ray(ax, a, length=r, color=c, linewidth=1)
            ax.annotate(f'v{i}: {np.rad2deg(a):.1f}°', 
                       (polygon[i, 0], polygon[i, 1]),
                       textcoords="offset points", xytext=(8, 8),
                       fontsize=10, color=c, fontweight='bold')
        
        ax.legend(['Origin'], loc='upper left')
        
        # Right plot: Polar diagram
        ax = axes[1]
        polar_ax = fig.add_subplot(122, projection='polar')
        polar_ax.set_title("Polar View of Vertices")
        
        # Plot vertices in polar coordinates
        for i, (r, a, c) in enumerate(zip(radii, angles, colors)):
            polar_ax.scatter(a, r, s=100, color=c, zorder=10)
            polar_ax.annotate(f'v{i}', (a, r), textcoords="offset points", 
                       xytext=(10, 5), fontsize=10, color=c)
        
        # Connect vertices
        angles_closed = np.append(angles, angles[0])
        radii_closed = np.append(radii, radii[0])
        polar_ax.plot(angles_closed, radii_closed, 'b-', linewidth=2, alpha=0.5)
        
        polar_ax.set_rlim(0, 3)  # type: ignore[attr-defined]
        polar_ax.grid(True)
        
        fig.suptitle("Vertex Polar Coordinates", fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_figure(fig, "sweep_vertex_angles")

    # =========================================================================
    # Test: resolve_interval() - Step 3.2
    # =========================================================================

    def _polygon_to_edges(self, polygon: np.ndarray) -> np.ndarray:
        """Convert a polygon to an array of edges (M, 2, 2)."""
        if polygon is None or len(polygon) < 3:
            return np.empty((0, 2, 2), dtype=np.float32)
        n = len(polygon)
        edges = []
        for i in range(n):
            edge = np.array([polygon[i], polygon[(i + 1) % n]], dtype=np.float32)
            edges.append(edge)
        return np.array(edges, dtype=np.float32)

    def test_visual_resolve_interval_single_obstacle(self):
        """Visual: Interval resolution with a single obstacle."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Create a simple rectangular obstacle
        obstacle = np.array([
            [2.0, -1.0],
            [4.0, -1.0],
            [4.0, 1.0],
            [2.0, 1.0],
        ], dtype=np.float32)
        
        edges = self._polygon_to_edges(obstacle)
        
        interval_start = -0.3
        interval_end = 0.3
        
        # Left plot: Spatial view
        ax = axes[0]
        self._setup_axes(ax, "Single Obstacle - Interval Resolution", xlim=(-1, 6), ylim=(-3, 3))
        
        # Draw arc sector for the interval
        self._draw_arc_sector(ax, interval_start, interval_end, radius=5, 
                             color='lightgreen', alpha=0.3, label='Interval')
        
        # Draw polygon
        self._draw_polygon(ax, obstacle, color='lightblue', alpha=0.5, 
                          edgecolor='blue', label='Obstacle 0')
        
        # Draw sample rays (5 samples)
        num_samples = 5
        sample_angles = np.linspace(interval_start, interval_end, num_samples)
        for i, angle in enumerate(sample_angles):
            # Find intersection distance
            min_dist = None
            for edge in edges:
                dist = intersect_ray_segment(angle, edge[0], edge[1], 10.0)
                if dist is not None and (min_dist is None or dist < min_dist):
                    min_dist = dist
            
            ray_length = min_dist if min_dist else 5
            color = 'green' if min_dist else 'gray'
            ax.arrow(0, 0, ray_length * 0.95 * np.cos(angle), 
                    ray_length * 0.95 * np.sin(angle),
                    head_width=0.1, head_length=0.08, fc=color, ec=color, linewidth=1)
            
            if min_dist:
                hit_x = min_dist * np.cos(angle)
                hit_y = min_dist * np.sin(angle)
                ax.scatter([hit_x], [hit_y], color='red', s=80, zorder=15, 
                          marker='x', linewidths=2)
                ax.annotate(f'd={min_dist:.2f}', (hit_x, hit_y), 
                           textcoords="offset points", xytext=(5, 5), fontsize=8)
        
        # Resolve interval
        result = resolve_interval(interval_start, interval_end, {0: edges}, num_samples)
        
        # Annotate result
        result_text = (f"Winner: Obstacle {result.obstacle_id}\n"
                      f"Min Distance: {result.min_distance:.2f}") if result else "No intersection"
        ax.annotate(result_text, (0.02, 0.98), xycoords='axes fraction',
                   fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.legend(loc='upper right', fontsize=8)
        
        # Right plot: Polar view
        ax = axes[1]
        polar_ax = fig.add_subplot(122, projection='polar')
        polar_ax.set_title("Polar View with Sample Rays")
        
        # Draw obstacle in polar coordinates
        radii, angles = to_polar(obstacle)
        angles_closed = np.append(angles, angles[0])
        radii_closed = np.append(radii, radii[0])
        polar_ax.fill(angles_closed, radii_closed, alpha=0.4, color='blue', label='Obstacle')
        polar_ax.plot(angles_closed, radii_closed, 'b-', linewidth=2)
        
        # Draw sample rays
        for angle in sample_angles:
            polar_ax.plot([angle, angle], [0, 5], 'g-', linewidth=1, alpha=0.7)
        
        # Highlight interval boundaries
        polar_ax.axvline(x=interval_start, color='orange', linewidth=2, linestyle='--')
        polar_ax.axvline(x=interval_end, color='orange', linewidth=2, linestyle='--')
        
        polar_ax.set_rlim(0, 6)  # type: ignore[attr-defined]
        polar_ax.legend(loc='upper right')
        
        fig.suptitle("Interval Resolution: Single Obstacle", fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_figure(fig, "sweep_resolve_interval_single")

    def test_visual_resolve_interval_occlusion(self):
        """Visual: Interval resolution with occlusion - closer obstacle wins."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Far obstacle (obstacle 0)
        obstacle_far = np.array([
            [5.0, -1.5],
            [7.0, -1.5],
            [7.0, 1.5],
            [5.0, 1.5],
        ], dtype=np.float32)
        
        # Close obstacle (obstacle 1)
        obstacle_close = np.array([
            [2.0, -0.8],
            [3.5, -0.8],
            [3.5, 0.8],
            [2.0, 0.8],
        ], dtype=np.float32)
        
        edges_far = self._polygon_to_edges(obstacle_far)
        edges_close = self._polygon_to_edges(obstacle_close)
        
        interval_start = -0.25
        interval_end = 0.25
        
        # Left plot: Spatial view
        ax = axes[0]
        self._setup_axes(ax, "Occlusion Test - Closer Obstacle Wins", xlim=(-1, 9), ylim=(-3, 3))
        
        # Draw arc sector
        self._draw_arc_sector(ax, interval_start, interval_end, radius=8, 
                             color='lightyellow', alpha=0.3, label='Interval')
        
        # Draw obstacles
        self._draw_polygon(ax, obstacle_far, color='lightcoral', alpha=0.5, 
                          edgecolor='red', label='Obstacle 0 (far)')
        self._draw_polygon(ax, obstacle_close, color='lightgreen', alpha=0.5, 
                          edgecolor='green', label='Obstacle 1 (close)')
        
        # Draw sample rays
        num_samples = 5
        sample_angles = np.linspace(interval_start, interval_end, num_samples)
        
        for angle in sample_angles:
            # Find closest intersection
            min_dist = None
            winner = None
            for oid, edges in [(0, edges_far), (1, edges_close)]:
                for edge in edges:
                    dist = intersect_ray_segment(angle, edge[0], edge[1], 10.0)
                    if dist is not None and (min_dist is None or dist < min_dist):
                        min_dist = dist
                        winner = oid
            
            ray_length = min_dist if min_dist else 8
            color = 'green' if winner == 1 else ('red' if winner == 0 else 'gray')
            ax.arrow(0, 0, ray_length * 0.95 * np.cos(angle), 
                    ray_length * 0.95 * np.sin(angle),
                    head_width=0.15, head_length=0.1, fc=color, ec=color, linewidth=1.5)
            
            if min_dist:
                hit_x = min_dist * np.cos(angle)
                hit_y = min_dist * np.sin(angle)
                ax.scatter([hit_x], [hit_y], color=color, s=100, zorder=15, 
                          marker='o', edgecolors='black', linewidths=1)
        
        # Resolve interval
        active_obstacles = {0: edges_far, 1: edges_close}
        result = resolve_interval(interval_start, interval_end, active_obstacles, num_samples)
        
        # Annotate result
        if result:
            winner_name = "Close (green)" if result.obstacle_id == 1 else "Far (red)"
            result_text = (f"Winner: Obstacle {result.obstacle_id} ({winner_name})\n"
                          f"Min Distance: {result.min_distance:.2f}")
        else:
            result_text = "No intersection"
        ax.annotate(result_text, (0.02, 0.98), xycoords='axes fraction',
                   fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.legend(loc='upper right', fontsize=8)
        
        # Right plot: Side-by-side distance comparison
        ax = axes[1]
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(0, 10)
        ax.set_xlabel('Obstacle ID')
        ax.set_ylabel('Distance')
        ax.set_title('Distance Comparison per Sample Ray')
        
        # Compute distances for each obstacle at each sample
        for i, angle in enumerate(sample_angles):
            for oid, edges, color in [(0, edges_far, 'red'), (1, edges_close, 'green')]:
                min_dist = None
                for edge in edges:
                    dist = intersect_ray_segment(angle, edge[0], edge[1], 10.0)
                    if dist is not None and (min_dist is None or dist < min_dist):
                        min_dist = dist
                if min_dist:
                    ax.bar(oid + i * 0.1 - 0.2, min_dist, width=0.08, 
                          color=color, alpha=0.7, edgecolor='black')
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Obstacle 0\n(far)', 'Obstacle 1\n(close)'])
        ax.axhline(y=2.0, color='green', linestyle='--', alpha=0.5, label='Close dist')
        ax.axhline(y=5.0, color='red', linestyle='--', alpha=0.5, label='Far dist')
        ax.legend()
        
        fig.suptitle("Interval Resolution: Occlusion (Closer Wins)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_figure(fig, "sweep_resolve_interval_occlusion")

    def test_visual_resolve_interval_sampling(self):
        """Visual: Show how multi-ray sampling works within an interval."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Create a V-shaped obstacle
        v_obstacle = np.array([
            [2.0, 0.0],    # apex
            [5.0, 2.0],    # top right
            [5.0, -2.0],   # bottom right
        ], dtype=np.float32)
        
        edges = self._polygon_to_edges(v_obstacle)
        
        interval_start = -0.3
        interval_end = 0.3
        
        # Test with different sample counts
        sample_counts = [1, 3, 5, 9]
        
        for ax, num_samples in zip(axes.flatten(), sample_counts):
            self._setup_axes(ax, f"Sampling with {num_samples} ray(s)", xlim=(-1, 7), ylim=(-3, 3))
            
            # Draw arc sector
            self._draw_arc_sector(ax, interval_start, interval_end, radius=6, 
                                 color='lightyellow', alpha=0.3)
            
            # Draw polygon
            self._draw_polygon(ax, v_obstacle, color='lightblue', alpha=0.5, 
                              edgecolor='blue', label='V-shaped obstacle')
            
            # Draw sample rays
            if num_samples == 1:
                sample_angles = [(interval_start + interval_end) / 2]
            else:
                sample_angles = np.linspace(interval_start, interval_end, num_samples)
            
            distances = []
            for angle in sample_angles:
                min_dist = None
                for edge in edges:
                    dist = intersect_ray_segment(angle, edge[0], edge[1], 10.0)
                    if dist is not None and (min_dist is None or dist < min_dist):
                        min_dist = dist
                
                if min_dist:
                    distances.append(min_dist)
                    hit_x = min_dist * np.cos(angle)
                    hit_y = min_dist * np.sin(angle)
                    ax.arrow(0, 0, hit_x * 0.95, hit_y * 0.95,
                            head_width=0.1, head_length=0.08, fc='green', ec='green', linewidth=1)
                    ax.scatter([hit_x], [hit_y], color='red', s=60, zorder=15, marker='x')
                    ax.annotate(f'{min_dist:.2f}', (hit_x, hit_y), 
                               textcoords="offset points", xytext=(5, 5), fontsize=8)
            
            # Resolve interval
            result = resolve_interval(interval_start, interval_end, {0: edges}, num_samples)
            
            # Statistics
            if distances:
                avg_dist = sum(distances) / len(distances)
                min_dist_found = min(distances)
                stats_text = (f"Samples: {num_samples}\n"
                             f"Hits: {len(distances)}\n"
                             f"Min dist: {min_dist_found:.2f}\n"
                             f"Avg dist: {avg_dist:.2f}")
            else:
                stats_text = f"Samples: {num_samples}\nNo hits"
            
            ax.annotate(stats_text, (0.02, 0.98), xycoords='axes fraction',
                       fontsize=10, verticalalignment='top', family='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax.legend(loc='upper right', fontsize=8)
        
        fig.suptitle("Interval Resolution: Effect of Sample Count", fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_figure(fig, "sweep_resolve_interval_sampling")

    def test_visual_resolve_interval_partial_coverage(self):
        """Visual: Obstacle that only covers part of the interval."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Obstacle that only covers positive angles
        obstacle_partial = np.array([
            [2.5, 0.5],
            [4.0, 0.5],
            [4.0, 2.0],
            [2.5, 2.0],
        ], dtype=np.float32)
        
        # Obstacle that covers full interval
        obstacle_full = np.array([
            [3.5, -1.5],
            [5.5, -1.5],
            [5.5, 1.5],
            [3.5, 1.5],
        ], dtype=np.float32)
        
        edges_partial = self._polygon_to_edges(obstacle_partial)
        edges_full = self._polygon_to_edges(obstacle_full)
        
        interval_start = -0.4
        interval_end = 0.6
        
        # Left plot: Spatial view
        ax = axes[0]
        self._setup_axes(ax, "Partial vs Full Coverage", xlim=(-1, 7), ylim=(-3, 4))
        
        # Draw arc sector
        self._draw_arc_sector(ax, interval_start, interval_end, radius=6, 
                             color='lightyellow', alpha=0.3, label='Interval')
        
        # Draw obstacles
        self._draw_polygon(ax, obstacle_partial, color='lightcoral', alpha=0.5, 
                          edgecolor='red', label='Obstacle 0 (partial)')
        self._draw_polygon(ax, obstacle_full, color='lightgreen', alpha=0.5, 
                          edgecolor='green', label='Obstacle 1 (full)')
        
        # Draw sample rays
        num_samples = 7
        sample_angles = np.linspace(interval_start, interval_end, num_samples)
        
        hits_per_obstacle = {0: 0, 1: 0}
        
        for angle in sample_angles:
            min_dist = None
            winner = None
            
            for oid, edges in [(0, edges_partial), (1, edges_full)]:
                for edge in edges:
                    dist = intersect_ray_segment(angle, edge[0], edge[1], 10.0)
                    if dist is not None and (min_dist is None or dist < min_dist):
                        min_dist = dist
                        winner = oid
            
            if winner is not None:
                hits_per_obstacle[winner] += 1
            
            ray_length = min_dist if min_dist else 6
            if winner == 0:
                color = 'red'
            elif winner == 1:
                color = 'green'
            else:
                color = 'gray'
            
            ax.arrow(0, 0, ray_length * 0.95 * np.cos(angle), 
                    ray_length * 0.95 * np.sin(angle),
                    head_width=0.12, head_length=0.08, fc=color, ec=color, linewidth=1)
            
            if min_dist:
                hit_x = min_dist * np.cos(angle)
                hit_y = min_dist * np.sin(angle)
                ax.scatter([hit_x], [hit_y], color=color, s=80, zorder=15, 
                          marker='o', edgecolors='black', linewidths=1)
        
        # Resolve interval
        active_obstacles = {0: edges_partial, 1: edges_full}
        result = resolve_interval(interval_start, interval_end, active_obstacles, num_samples)
        
        # Annotate hits
        ax.annotate(f"Obstacle 0 hits: {hits_per_obstacle[0]}\n"
                   f"Obstacle 1 hits: {hits_per_obstacle[1]}", 
                   (0.02, 0.85), xycoords='axes fraction',
                   fontsize=10, family='monospace',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if result:
            winner_name = "Partial (red)" if result.obstacle_id == 0 else "Full (green)"
            result_text = f"Winner: Obstacle {result.obstacle_id} ({winner_name})"
        else:
            result_text = "No intersection"
        ax.annotate(result_text, (0.02, 0.98), xycoords='axes fraction',
                   fontsize=11, verticalalignment='top', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.legend(loc='upper right', fontsize=8)
        
        # Right plot: Hit ratio visualization
        ax = axes[1]
        ax.set_title('Coverage Analysis')
        
        # Bar chart of hits
        bars = ax.bar([0, 1], [hits_per_obstacle[0], hits_per_obstacle[1]], 
                     color=['red', 'green'], alpha=0.7, edgecolor='black')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Obstacle 0\n(partial)', 'Obstacle 1\n(full)'])
        ax.set_ylabel('Number of Ray Hits')
        ax.set_ylim(0, num_samples + 1)
        
        # Add hit ratio text
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ratio = height / num_samples * 100
            ax.annotate(f'{int(height)} hits\n({ratio:.0f}%)', 
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.axhline(y=num_samples, color='black', linestyle='--', alpha=0.3, 
                  label=f'Total samples: {num_samples}')
        ax.legend()
        
        fig.suptitle("Interval Resolution: Partial vs Full Coverage", fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_figure(fig, "sweep_resolve_interval_partial")

    def test_visual_resolve_interval_no_intersection(self):
        """Visual: Interval where obstacles don't intersect sample rays."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Obstacle far above the x-axis
        obstacle = np.array([
            [2.0, 3.0],
            [4.0, 3.0],
            [4.0, 5.0],
            [2.0, 5.0],
        ], dtype=np.float32)
        
        edges = self._polygon_to_edges(obstacle)
        
        interval_start = -0.2
        interval_end = 0.2
        
        self._setup_axes(ax, "No Intersection Scenario", xlim=(-1, 6), ylim=(-2, 6))
        
        # Draw arc sector
        self._draw_arc_sector(ax, interval_start, interval_end, radius=5, 
                             color='lightgreen', alpha=0.3, label='Interval')
        
        # Draw polygon
        self._draw_polygon(ax, obstacle, color='lightblue', alpha=0.5, 
                          edgecolor='blue', label='Obstacle (above rays)')
        
        # Draw sample rays
        num_samples = 5
        sample_angles = np.linspace(interval_start, interval_end, num_samples)
        
        for angle in sample_angles:
            ax.arrow(0, 0, 5 * np.cos(angle), 5 * np.sin(angle),
                    head_width=0.1, head_length=0.08, fc='gray', ec='gray', linewidth=1)
        
        # Resolve interval
        result = resolve_interval(interval_start, interval_end, {0: edges}, num_samples)
        
        # Annotate
        result_text = "Result: None (no intersections)" if result is None else f"Result: {result}"
        ax.annotate(result_text, (0.02, 0.98), xycoords='axes fraction',
                   fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        ax.annotate("Rays miss the obstacle\n(obstacle is above the sampled angular range)", 
                   (3, 1), fontsize=10, ha='center',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.legend(loc='upper right', fontsize=9)
        
        fig.suptitle("Interval Resolution: No Intersection", fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_figure(fig, "sweep_resolve_interval_no_hit")

    # =========================================================================
    # Test: compute_coverage() - Step 3.3
    # =========================================================================

    def test_visual_compute_coverage_single_obstacle(self):
        """Visual: Coverage computation for a single obstacle occupying the arc."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Square obstacle centered on x-axis
        polygon = np.array([
            [2.0, -1.0],
            [4.0, -1.0],
            [4.0, 1.0],
            [2.0, 1.0],
        ], dtype=np.float32)
        
        edges = self._polygon_to_edges(polygon)
        obstacle_edges = {0: edges}
        
        alpha_min = -0.4
        alpha_max = 0.4
        
        # Build events
        events = build_events([(0, polygon)], alpha_min, alpha_max)
        
        # Compute coverage
        coverage, min_dist, intervals = compute_coverage(
            events, obstacle_edges, alpha_min, alpha_max
        )
        
        # Left plot: Spatial view
        ax = axes[0]
        self._setup_axes(ax, "Spatial View", xlim=(-1, 6), ylim=(-3, 3))
        
        # Draw arc sector
        self._draw_arc_sector(ax, alpha_min, alpha_max, radius=5, 
                             color='lightyellow', alpha=0.3, label='FOV Arc')
        
        # Draw arc boundaries
        self._draw_ray(ax, alpha_min, length=5, color='orange', linewidth=1, label='α_min')
        self._draw_ray(ax, alpha_max, length=5, color='orange', linewidth=1, label='α_max')
        
        # Draw polygon
        self._draw_polygon(ax, polygon, color='lightblue', alpha=0.5, 
                          edgecolor='blue', label='Obstacle 0')
        
        # Draw interval rays
        for interval in intervals:
            mid_angle = (interval.angle_start + interval.angle_end) / 2
            self._draw_ray(ax, mid_angle, length=interval.min_distance, 
                          color='green', linewidth=1)
        
        ax.legend(loc='upper right', fontsize=8)
        
        # Middle plot: Angular coverage visualization
        ax = axes[1]
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.5, 1.5)
        ax.set_xlabel('Angle (radians)')
        ax.set_title('Angular Coverage by Obstacle')
        
        # Draw arc boundaries
        ax.axvline(x=alpha_min, color='orange', linewidth=2, linestyle='--', label='α_min')
        ax.axvline(x=alpha_max, color='orange', linewidth=2, linestyle='--', label='α_max')
        
        # Draw intervals as horizontal bars
        colors = [plt.get_cmap('tab10')(i) for i in range(10)]
        for interval in intervals:
            color = colors[interval.obstacle_id % len(colors)]
            ax.barh(y=interval.obstacle_id, 
                   width=interval.angle_end - interval.angle_start,
                   left=interval.angle_start, 
                   height=0.3, 
                   color=color, alpha=0.7, edgecolor='black')
        
        ax.set_yticks([0])
        ax.set_yticklabels(['Obstacle 0'])
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Right plot: Coverage statistics
        ax = axes[2]
        ax.axis('off')
        ax.set_title('Coverage Statistics', fontsize=12, fontweight='bold')
        
        arc_span = alpha_max - alpha_min
        stats_text = f"""
Arc Range: [{np.rad2deg(alpha_min):.1f}°, {np.rad2deg(alpha_max):.1f}°]
Arc Span: {arc_span:.3f} rad ({np.rad2deg(arc_span):.1f}°)

Coverage Results:
"""
        for oid, cov in coverage.items():
            pct = (cov / arc_span) * 100 if arc_span > 0 else 0
            dist = min_dist.get(oid, float('inf'))
            stats_text += f"  Obstacle {oid}: {cov:.3f} rad ({pct:.1f}%), min_dist={dist:.2f}\n"
        
        stats_text += f"\nTotal Intervals: {len(intervals)}"
        
        ax.text(0.1, 0.8, stats_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        fig.suptitle("compute_coverage(): Single Obstacle Full Coverage", 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_figure(fig, "sweep_coverage_single_obstacle")

    def test_visual_compute_coverage_two_obstacles_no_occlusion(self):
        """Visual: Coverage with two obstacles side by side (no occlusion)."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Upper obstacle
        polygon0 = np.array([
            [2.0, 0.5],
            [3.5, 0.5],
            [3.5, 1.8],
            [2.0, 1.8],
        ], dtype=np.float32)
        
        # Lower obstacle
        polygon1 = np.array([
            [2.0, -1.8],
            [3.5, -1.8],
            [3.5, -0.5],
            [2.0, -0.5],
        ], dtype=np.float32)
        
        edges0 = self._polygon_to_edges(polygon0)
        edges1 = self._polygon_to_edges(polygon1)
        obstacle_edges = {0: edges0, 1: edges1}
        
        alpha_min = -0.8
        alpha_max = 0.8
        
        # Build events
        events = build_events([(0, polygon0), (1, polygon1)], alpha_min, alpha_max)
        
        # Compute coverage
        coverage, min_dist, intervals = compute_coverage(
            events, obstacle_edges, alpha_min, alpha_max
        )
        
        # Left plot: Spatial view
        ax = axes[0]
        self._setup_axes(ax, "Spatial View: Side-by-Side Obstacles", xlim=(-1, 5), ylim=(-3, 3))
        
        # Draw arc sector
        self._draw_arc_sector(ax, alpha_min, alpha_max, radius=4, 
                             color='lightyellow', alpha=0.2, label='FOV Arc')
        
        # Draw obstacles with colors matching coverage chart
        colors = ['#1f77b4', '#ff7f0e']  # tab10 blue and orange
        self._draw_polygon(ax, polygon0, color=colors[0], alpha=0.5, 
                          edgecolor=colors[0], label='Obstacle 0 (upper)')
        self._draw_polygon(ax, polygon1, color=colors[1], alpha=0.5, 
                          edgecolor=colors[1], label='Obstacle 1 (lower)')
        
        # Draw arc boundaries
        self._draw_ray(ax, alpha_min, length=4.5, color='orange', linewidth=1)
        self._draw_ray(ax, alpha_max, length=4.5, color='orange', linewidth=1)
        
        ax.legend(loc='upper right', fontsize=8)
        
        # Middle plot: Angular coverage visualization
        ax = axes[1]
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-0.5, 2.5)
        ax.set_xlabel('Angle (radians)')
        ax.set_ylabel('Obstacle ID')
        ax.set_title('Angular Coverage by Obstacle')
        
        # Draw arc boundaries
        ax.axvline(x=alpha_min, color='orange', linewidth=2, linestyle='--')
        ax.axvline(x=alpha_max, color='orange', linewidth=2, linestyle='--')
        
        # Draw intervals
        for interval in intervals:
            color = colors[interval.obstacle_id]
            ax.barh(y=interval.obstacle_id, 
                   width=interval.angle_end - interval.angle_start,
                   left=interval.angle_start, 
                   height=0.4, 
                   color=color, alpha=0.7, edgecolor='black')
        
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Obstacle 0\n(upper)', 'Obstacle 1\n(lower)'])
        ax.grid(True, alpha=0.3)
        
        # Right plot: Pie chart of coverage
        ax = axes[2]
        ax.set_title('Coverage Distribution', fontsize=12, fontweight='bold')
        
        arc_span = alpha_max - alpha_min
        coverage_values = [coverage.get(0, 0), coverage.get(1, 0)]
        uncovered = arc_span - sum(coverage_values)
        if uncovered < 0:
            uncovered = 0
        
        labels = [f'Obstacle 0\n({coverage_values[0]:.2f} rad)',
                 f'Obstacle 1\n({coverage_values[1]:.2f} rad)']
        pie_colors = colors
        
        if uncovered > 0.01:
            coverage_values.append(uncovered)
            labels.append(f'Uncovered\n({uncovered:.2f} rad)')
            pie_colors = colors + ['lightgray']
        
        wedges, texts, autotexts = ax.pie(coverage_values, labels=labels, 
                                          colors=pie_colors, autopct='%1.1f%%',
                                          startangle=90)
        ax.axis('equal')
        
        fig.suptitle("compute_coverage(): Two Obstacles Side-by-Side (No Occlusion)", 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_figure(fig, "sweep_coverage_two_obstacles_no_occlusion")

    def test_visual_compute_coverage_occlusion(self):
        """Visual: Coverage with occlusion - closer obstacle wins."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Far obstacle (will be occluded)
        polygon_far = np.array([
            [4.0, -1.2],
            [6.0, -1.2],
            [6.0, 1.2],
            [4.0, 1.2],
        ], dtype=np.float32)
        
        # Close obstacle (occludes the far one in overlapping region)
        polygon_close = np.array([
            [2.0, -0.6],
            [3.5, -0.6],
            [3.5, 0.6],
            [2.0, 0.6],
        ], dtype=np.float32)
        
        edges_far = self._polygon_to_edges(polygon_far)
        edges_close = self._polygon_to_edges(polygon_close)
        obstacle_edges = {0: edges_far, 1: edges_close}
        
        alpha_min = -0.35
        alpha_max = 0.35
        
        # Build events
        events = build_events([(0, polygon_far), (1, polygon_close)], alpha_min, alpha_max)
        
        # Compute coverage
        coverage, min_dist, intervals = compute_coverage(
            events, obstacle_edges, alpha_min, alpha_max
        )
        
        # Top-left: Spatial view
        ax = axes[0, 0]
        self._setup_axes(ax, "Spatial View: Occlusion Scenario", xlim=(-1, 8), ylim=(-2.5, 2.5))
        
        # Draw arc sector
        self._draw_arc_sector(ax, alpha_min, alpha_max, radius=7, 
                             color='lightyellow', alpha=0.2, label='FOV Arc')
        
        # Draw obstacles
        colors = ['#d62728', '#2ca02c']  # red for far, green for close
        self._draw_polygon(ax, polygon_far, color=colors[0], alpha=0.4, 
                          edgecolor=colors[0], label='Obstacle 0 (far)')
        self._draw_polygon(ax, polygon_close, color=colors[1], alpha=0.6, 
                          edgecolor=colors[1], label='Obstacle 1 (close)')
        
        # Draw sample rays showing occlusion
        sample_angles = np.linspace(alpha_min, alpha_max, 7)
        for angle in sample_angles:
            # Find closest hit
            min_hit = None
            for oid, edges in obstacle_edges.items():
                for edge in edges:
                    dist = intersect_ray_segment(angle, edge[0], edge[1], 10.0)
                    if dist is not None and (min_hit is None or dist < min_hit):
                        min_hit = dist
            
            if min_hit:
                hit_x = min_hit * np.cos(angle)
                hit_y = min_hit * np.sin(angle)
                ax.plot([0, hit_x], [0, hit_y], 'g-', linewidth=1, alpha=0.6)
                ax.scatter([hit_x], [hit_y], color='green', s=40, zorder=15, marker='o')
        
        ax.legend(loc='upper right', fontsize=8)
        
        # Top-right: Angular coverage
        ax = axes[0, 1]
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 2.5)
        ax.set_xlabel('Angle (radians)')
        ax.set_title('Angular Coverage (Occlusion Resolved)')
        
        ax.axvline(x=alpha_min, color='orange', linewidth=2, linestyle='--')
        ax.axvline(x=alpha_max, color='orange', linewidth=2, linestyle='--')
        
        for interval in intervals:
            color = colors[interval.obstacle_id]
            ax.barh(y=interval.obstacle_id, 
                   width=interval.angle_end - interval.angle_start,
                   left=interval.angle_start, 
                   height=0.4, 
                   color=color, alpha=0.7, edgecolor='black')
        
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Obstacle 0\n(far)', 'Obstacle 1\n(close, winner)'])
        ax.grid(True, alpha=0.3)
        
        # Bottom-left: Distance comparison
        ax = axes[1, 0]
        ax.set_title('Distance Comparison', fontsize=12)
        
        bar_width = 0.35
        x = np.array([0, 1])
        
        min_distances = [min_dist.get(0, 0), min_dist.get(1, 0)]
        bars = ax.bar(x, min_distances, bar_width, color=colors, alpha=0.7, edgecolor='black')
        
        ax.set_xticks(x)
        ax.set_xticklabels(['Obstacle 0\n(far)', 'Obstacle 1\n(close)'])
        ax.set_ylabel('Minimum Distance')
        
        for bar, dist in zip(bars, min_distances):
            if dist > 0:
                ax.annotate(f'{dist:.2f}', xy=(bar.get_x() + bar.get_width()/2, dist),
                           ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Bottom-right: Coverage summary
        ax = axes[1, 1]
        ax.axis('off')
        ax.set_title('Coverage Summary', fontsize=12, fontweight='bold')
        
        arc_span = alpha_max - alpha_min
        summary_text = f"""
Arc Span: {arc_span:.3f} rad ({np.rad2deg(arc_span):.1f}°)

Obstacle 0 (far, red):
  Coverage: {coverage.get(0, 0):.4f} rad ({coverage.get(0, 0)/arc_span*100:.1f}%)
  Min Distance: {min_dist.get(0, float('inf')):.2f}

Obstacle 1 (close, green):
  Coverage: {coverage.get(1, 0):.4f} rad ({coverage.get(1, 0)/arc_span*100:.1f}%)
  Min Distance: {min_dist.get(1, float('inf')):.2f}

Total Intervals: {len(intervals)}

Note: The closer obstacle (1) dominates the
overlapping angular region due to occlusion.
"""
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        fig.suptitle("compute_coverage(): Occlusion - Closer Obstacle Wins", 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_figure(fig, "sweep_coverage_occlusion")

    def test_visual_compute_coverage_gap_in_arc(self):
        """Visual: Coverage with a gap - some angles have no obstacle."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Obstacle that only covers part of the arc (upper region)
        polygon = np.array([
            [2.0, 0.5],
            [4.0, 0.5],
            [4.0, 2.0],
            [2.0, 2.0],
        ], dtype=np.float32)
        
        edges = self._polygon_to_edges(polygon)
        obstacle_edges = {0: edges}
        
        # Wide arc that extends beyond the obstacle
        alpha_min = -0.5
        alpha_max = 0.9
        
        # Build events
        events = build_events([(0, polygon)], alpha_min, alpha_max)
        
        # Compute coverage
        coverage, min_dist, intervals = compute_coverage(
            events, obstacle_edges, alpha_min, alpha_max
        )
        
        # Left plot: Spatial view
        ax = axes[0]
        self._setup_axes(ax, "Spatial View: Partial Coverage", xlim=(-1, 6), ylim=(-2, 4))
        
        # Draw arc sector
        self._draw_arc_sector(ax, alpha_min, alpha_max, radius=5, 
                             color='lightyellow', alpha=0.3, label='FOV Arc')
        
        # Draw obstacle
        self._draw_polygon(ax, polygon, color='lightblue', alpha=0.5, 
                          edgecolor='blue', label='Obstacle 0')
        
        # Draw arc boundaries
        self._draw_ray(ax, alpha_min, length=5, color='orange', linewidth=1, label='α_min')
        self._draw_ray(ax, alpha_max, length=5, color='orange', linewidth=1, label='α_max')
        
        # Draw rays showing coverage and gaps
        sample_angles = np.linspace(alpha_min, alpha_max, 11)
        for angle in sample_angles:
            min_hit = None
            for edge in edges:
                dist = intersect_ray_segment(angle, edge[0], edge[1], 10.0)
                if dist is not None and (min_hit is None or dist < min_hit):
                    min_hit = dist
            
            if min_hit:
                hit_x = min_hit * np.cos(angle)
                hit_y = min_hit * np.sin(angle)
                ax.plot([0, hit_x], [0, hit_y], 'g-', linewidth=1.5, alpha=0.7)
                ax.scatter([hit_x], [hit_y], color='green', s=50, zorder=15)
            else:
                # Draw miss ray (gray, extends to arc edge)
                ax.plot([0, 5*np.cos(angle)], [0, 5*np.sin(angle)], 
                       'gray', linewidth=1, alpha=0.4, linestyle='--')
        
        ax.legend(loc='upper right', fontsize=8)
        
        # Middle plot: Angular coverage
        ax = axes[1]
        arc_span = alpha_max - alpha_min
        
        ax.set_xlim(-0.7, 1.1)
        ax.set_ylim(-0.3, 1.0)
        ax.set_xlabel('Angle (radians)')
        ax.set_title('Coverage vs Gap Analysis')
        
        # Draw arc boundaries
        ax.axvline(x=alpha_min, color='orange', linewidth=2, linestyle='--', label='Arc bounds')
        ax.axvline(x=alpha_max, color='orange', linewidth=2, linestyle='--')
        
        # Draw full arc span as background
        ax.barh(y=0.5, width=arc_span, left=alpha_min, height=0.15, 
               color='lightgray', alpha=0.5, edgecolor='black', label='Full arc')
        
        # Draw covered intervals
        for interval in intervals:
            ax.barh(y=0.3, 
                   width=interval.angle_end - interval.angle_start,
                   left=interval.angle_start, 
                   height=0.15, 
                   color='blue', alpha=0.7, edgecolor='black', label='Covered')
        
        ax.set_yticks([0.3, 0.5])
        ax.set_yticklabels(['Covered', 'Full Arc'])
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Right plot: Coverage statistics
        ax = axes[2]
        
        total_coverage = sum(coverage.values())
        gap = arc_span - total_coverage
        
        ax.set_title('Coverage Breakdown', fontsize=12, fontweight='bold')
        
        # Stacked bar showing covered vs gap
        ax.barh(y=0, width=total_coverage, left=0, height=0.5, 
               color='blue', alpha=0.7, edgecolor='black', label='Covered')
        ax.barh(y=0, width=gap, left=total_coverage, height=0.5, 
               color='lightgray', alpha=0.7, edgecolor='black', label='Gap')
        
        ax.set_xlim(0, arc_span * 1.1)
        ax.set_ylim(-0.5, 0.8)
        ax.set_xlabel('Angular Span (radians)')
        ax.set_yticks([])
        
        # Annotate values
        ax.annotate(f'Covered: {total_coverage:.3f} rad\n({total_coverage/arc_span*100:.1f}%)', 
                   (total_coverage/2, 0.35), ha='center', fontsize=10, fontweight='bold')
        ax.annotate(f'Gap: {gap:.3f} rad\n({gap/arc_span*100:.1f}%)', 
                   (total_coverage + gap/2, 0.35), ha='center', fontsize=10, color='gray')
        
        ax.legend(loc='upper right', fontsize=9)
        
        fig.suptitle("compute_coverage(): Partial Coverage with Gap", 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_figure(fig, "sweep_coverage_gap")

    def test_visual_compute_coverage_arc_crossing_pi(self):
        """Visual: Coverage computation when arc crosses ±π boundary."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Obstacle near ±π boundary
        polygon = np.array([
            [-2.5, 0.4],
            [-4.0, 0.4],
            [-4.0, -0.4],
            [-2.5, -0.4],
        ], dtype=np.float32)
        
        edges = self._polygon_to_edges(polygon)
        obstacle_edges = {0: edges}
        
        # Arc crossing ±π (from 160° to -160°)
        alpha_min = 160 * np.pi / 180   # ≈ 2.79 rad
        alpha_max = -160 * np.pi / 180  # ≈ -2.79 rad
        
        # Build events
        events = build_events([(0, polygon)], alpha_min, alpha_max)
        
        # Compute coverage
        coverage, min_dist, intervals = compute_coverage(
            events, obstacle_edges, alpha_min, alpha_max
        )
        
        # Left plot: Spatial view
        ax = axes[0]
        self._setup_axes(ax, "Arc Crossing ±π Boundary", xlim=(-5, 2), ylim=(-3, 3))
        
        # Draw arc sector (wraps around)
        self._draw_arc_sector(ax, alpha_min, alpha_max, radius=4, 
                             color='lightyellow', alpha=0.3, label='FOV Arc')
        
        # Draw obstacle
        self._draw_polygon(ax, polygon, color='lightblue', alpha=0.5, 
                          edgecolor='blue', label='Obstacle 0')
        
        # Draw arc boundaries and π line
        self._draw_ray(ax, alpha_min, length=4.5, color='orange', linewidth=2, label='α_min (160°)')
        self._draw_ray(ax, alpha_max, length=4.5, color='red', linewidth=2, label='α_max (-160°)')
        self._draw_ray(ax, np.pi, length=4.5, color='purple', linewidth=1, label='±π line')
        self._draw_ray(ax, -np.pi, length=4.5, color='purple', linewidth=1)
        
        ax.legend(loc='upper right', fontsize=8)
        
        # Middle plot: Angular view (unwrapped)
        ax = axes[1]
        ax.set_xlim(-4, 5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_xlabel('Angle (radians)')
        ax.set_title('Angular View (Remapped for Continuity)')
        
        # Mark key angles
        ax.axvline(x=-np.pi, color='purple', linewidth=2, linestyle=':', label='-π')
        ax.axvline(x=np.pi, color='purple', linewidth=2, linestyle=':', label='+π')
        ax.axvline(x=alpha_min, color='orange', linewidth=2, linestyle='--', label='α_min')
        
        # Remapped alpha_max
        remapped_alpha_max = alpha_max + 2 * np.pi
        ax.axvline(x=remapped_alpha_max, color='red', linewidth=2, linestyle='--', 
                  label=f'α_max (remapped)')
        
        # Shade arc region
        ax.axvspan(alpha_min, np.pi, color='yellow', alpha=0.2)
        ax.axvspan(-np.pi + 2*np.pi, remapped_alpha_max, color='yellow', alpha=0.2)
        
        # Draw intervals
        for interval in intervals:
            ax.barh(y=0, 
                   width=interval.angle_end - interval.angle_start,
                   left=interval.angle_start, 
                   height=0.3, 
                   color='blue', alpha=0.7, edgecolor='black')
        
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Right plot: Coverage statistics
        ax = axes[2]
        ax.axis('off')
        ax.set_title('Coverage Statistics', fontsize=12, fontweight='bold')
        
        # Compute arc span (wrapping case)
        arc_span = (2 * np.pi) - alpha_min + alpha_max
        if arc_span < 0:
            arc_span += 2 * np.pi
        
        stats_text = f"""
Arc Configuration:
  α_min: {alpha_min:.3f} rad ({np.rad2deg(alpha_min):.1f}°)
  α_max: {alpha_max:.3f} rad ({np.rad2deg(alpha_max):.1f}°)
  Arc crosses ±π boundary: Yes
  Arc Span: {arc_span:.3f} rad ({np.rad2deg(arc_span):.1f}°)

Coverage Results:
  Obstacle 0: {coverage.get(0, 0):.4f} rad
  Coverage %: {coverage.get(0, 0)/arc_span*100:.1f}%
  Min Distance: {min_dist.get(0, float('inf')):.2f}

Events: {len(events)}
Intervals: {len(intervals)}
"""
        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        fig.suptitle("compute_coverage(): Arc Crossing ±π Discontinuity", 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_figure(fig, "sweep_coverage_arc_crossing_pi")

    def test_visual_compute_coverage_multiple_intervals(self):
        """Visual: Coverage with multiple intervals from complex obstacle arrangement."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Three obstacles at different positions
        polygon0 = np.array([
            [2.0, 1.0],
            [3.0, 1.0],
            [3.0, 2.5],
            [2.0, 2.5],
        ], dtype=np.float32)
        
        polygon1 = np.array([
            [3.0, -0.5],
            [5.0, -0.5],
            [5.0, 0.5],
            [3.0, 0.5],
        ], dtype=np.float32)
        
        polygon2 = np.array([
            [2.0, -2.5],
            [3.5, -2.5],
            [3.5, -1.0],
            [2.0, -1.0],
        ], dtype=np.float32)
        
        edges0 = self._polygon_to_edges(polygon0)
        edges1 = self._polygon_to_edges(polygon1)
        edges2 = self._polygon_to_edges(polygon2)
        obstacle_edges = {0: edges0, 1: edges1, 2: edges2}
        
        alpha_min = -1.0
        alpha_max = 1.0
        
        # Build events
        events = build_events([(0, polygon0), (1, polygon1), (2, polygon2)], alpha_min, alpha_max)
        
        # Compute coverage
        coverage, min_dist, intervals = compute_coverage(
            events, obstacle_edges, alpha_min, alpha_max
        )
        
        # Top-left: Spatial view
        ax = axes[0, 0]
        self._setup_axes(ax, "Three Obstacles in Arc", xlim=(-1, 6), ylim=(-4, 4))
        
        # Draw arc sector
        self._draw_arc_sector(ax, alpha_min, alpha_max, radius=5, 
                             color='lightyellow', alpha=0.2)
        
        # Draw obstacles with distinct colors
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        self._draw_polygon(ax, polygon0, color=colors[0], alpha=0.5, 
                          edgecolor=colors[0], label='Obstacle 0')
        self._draw_polygon(ax, polygon1, color=colors[1], alpha=0.5, 
                          edgecolor=colors[1], label='Obstacle 1')
        self._draw_polygon(ax, polygon2, color=colors[2], alpha=0.5, 
                          edgecolor=colors[2], label='Obstacle 2')
        
        # Draw arc boundaries
        self._draw_ray(ax, alpha_min, length=5.5, color='orange', linewidth=1)
        self._draw_ray(ax, alpha_max, length=5.5, color='orange', linewidth=1)
        
        ax.legend(loc='upper right', fontsize=8)
        
        # Top-right: Angular intervals
        ax = axes[0, 1]
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.5, 3.5)
        ax.set_xlabel('Angle (radians)')
        ax.set_ylabel('Obstacle ID')
        ax.set_title('Angular Intervals by Obstacle')
        
        ax.axvline(x=alpha_min, color='orange', linewidth=2, linestyle='--')
        ax.axvline(x=alpha_max, color='orange', linewidth=2, linestyle='--')
        
        for interval in intervals:
            color = colors[interval.obstacle_id]
            ax.barh(y=interval.obstacle_id, 
                   width=interval.angle_end - interval.angle_start,
                   left=interval.angle_start, 
                   height=0.4, 
                   color=color, alpha=0.7, edgecolor='black')
        
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['Obstacle 0', 'Obstacle 1', 'Obstacle 2'])
        ax.grid(True, alpha=0.3)
        
        # Bottom-left: Coverage bar chart
        ax = axes[1, 0]
        ax.set_title('Coverage by Obstacle', fontsize=12)
        
        obstacle_ids = sorted(coverage.keys())
        coverage_values = [coverage[oid] for oid in obstacle_ids]
        bar_colors = [colors[oid] for oid in obstacle_ids]
        
        bars = ax.bar(obstacle_ids, coverage_values, color=bar_colors, 
                     alpha=0.7, edgecolor='black')
        
        ax.set_xticks(obstacle_ids)
        ax.set_xticklabels([f'Obstacle {oid}' for oid in obstacle_ids])
        ax.set_ylabel('Coverage (radians)')
        
        for bar, val in zip(bars, coverage_values):
            ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, val),
                       ha='center', va='bottom', fontsize=10)
        
        # Bottom-right: Statistics table
        ax = axes[1, 1]
        ax.axis('off')
        ax.set_title('Coverage Summary', fontsize=12, fontweight='bold')
        
        arc_span = alpha_max - alpha_min
        total_coverage = sum(coverage.values())
        
        table_text = f"""
Arc Span: {arc_span:.3f} rad ({np.rad2deg(arc_span):.1f}°)
Total Covered: {total_coverage:.3f} rad ({total_coverage/arc_span*100:.1f}%)
Total Intervals: {len(intervals)}

Obstacle Details:
"""
        for oid in sorted(coverage.keys()):
            cov = coverage[oid]
            dist = min_dist.get(oid, float('inf'))
            pct = cov / arc_span * 100
            table_text += f"  [{oid}] Coverage: {cov:.3f} rad ({pct:.1f}%), Min dist: {dist:.2f}\n"
        
        ax.text(0.1, 0.9, table_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        fig.suptitle("compute_coverage(): Multiple Obstacles with Multiple Intervals", 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_figure(fig, "sweep_coverage_multiple_intervals")
