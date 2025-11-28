"""
Visual validation tests for angular sweep operations (Step 3.1).

These tests create matplotlib figures showing:
- Polygon obstacles with vertices and edges
- Query rays and active edges
- Angular events along the arc
- Event ordering visualization

Run with: pytest tests/visual/test_sweep_visual.py -v

Output figures are saved to: tests/visual/output/
"""

import pytest
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

from view_arc.sweep import (
    AngularEvent,
    get_active_edges,
    build_events,
)
from view_arc.geometry import to_polar


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

    def _draw_ray(self, ax, angle, length=4, color='red', label=None, linewidth=2):
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
        events = build_events([triangle], alpha_min, alpha_max)
        
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
        events = build_events([triangle, square], alpha_min, alpha_max)
        
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
        events = build_events([triangle], alpha_min, alpha_max)
        
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
