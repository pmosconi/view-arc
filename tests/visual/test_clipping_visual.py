"""
Visual validation tests for polygon clipping operations (Step 2.1).

These tests create matplotlib figures showing:
- Original polygons
- Clipping boundaries (half-planes)
- Clipped results

Run with: pytest tests/visual/test_clipping_visual.py -v

Output figures are saved to: tests/visual/output/
"""

import pytest
import numpy as np
from pathlib import Path

# Try to import matplotlib, skip tests if not available
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.collections import PatchCollection
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from view_arc.obstacle.clipping import (
    is_valid_polygon,
    compute_bounding_box,
    clip_polygon_halfplane,
    clip_polygon_circle,
    clip_polygon_to_wedge,
)


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
class TestClippingVisual:
    """Visual validation tests for clipping operations."""

    def _draw_polygon(self, ax, polygon, color='blue', alpha=0.3, label=None, edgecolor=None):
        """Draw a polygon on the axes."""
        if polygon.shape[0] < 3:
            return
        if edgecolor is None:
            edgecolor = color
        patch = mpatches.Polygon(polygon, closed=True, 
                                  facecolor=color, alpha=alpha, 
                                  edgecolor=edgecolor, linewidth=2,
                                  label=label)
        ax.add_patch(patch)

    def _draw_halfplane_boundary(self, ax, angle, extent=5, color='red', label=None):
        """Draw the half-plane boundary ray."""
        # Draw ray from origin
        x_end = extent * np.cos(angle)
        y_end = extent * np.sin(angle)
        ax.arrow(0, 0, x_end, y_end, head_width=0.15, head_length=0.1, 
                 fc=color, ec=color, linewidth=2, label=label)
        
        # Draw normal indicator (small perpendicular arrow)
        normal_x = -np.sin(angle) * 0.5
        normal_y = np.cos(angle) * 0.5
        mid_x = x_end * 0.5
        mid_y = y_end * 0.5
        ax.arrow(mid_x, mid_y, normal_x, normal_y, 
                 head_width=0.1, head_length=0.05, 
                 fc='green', ec='green', linewidth=1)

    def _setup_axes(self, ax, title, xlim=(-3, 4), ylim=(-3, 4)):
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
        ax.plot(0, 0, 'ko', markersize=8)  # Origin marker

    def test_visual_halfplane_clip_partial(self):
        """Visual: Partial clipping of square by horizontal half-plane."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Square centered at origin
        square = np.array([
            [-1.5, -1.5],
            [1.5, -1.5],
            [1.5, 1.5],
            [-1.5, 1.5],
        ], dtype=np.float32)
        
        # Test case 1: Clip with x-axis, keep upper half
        ax = axes[0]
        self._setup_axes(ax, "Clip: keep y ≥ 0 (left of x-axis ray)")
        self._draw_polygon(ax, square, color='lightblue', alpha=0.3, label='Original')
        self._draw_halfplane_boundary(ax, 0.0, label='Boundary')
        result = clip_polygon_halfplane(square, plane_angle=0.0, keep_left=True)
        self._draw_polygon(ax, result, color='blue', alpha=0.5, edgecolor='darkblue', label='Clipped')
        ax.legend(loc='upper right')
        
        # Test case 2: Clip with x-axis, keep lower half
        ax = axes[1]
        self._setup_axes(ax, "Clip: keep y ≤ 0 (right of x-axis ray)")
        self._draw_polygon(ax, square, color='lightblue', alpha=0.3, label='Original')
        self._draw_halfplane_boundary(ax, 0.0)
        result = clip_polygon_halfplane(square, plane_angle=0.0, keep_left=False)
        self._draw_polygon(ax, result, color='orange', alpha=0.5, edgecolor='darkorange', label='Clipped')
        ax.legend(loc='upper right')
        
        # Test case 3: Clip with y-axis, keep right half
        ax = axes[2]
        self._setup_axes(ax, "Clip: keep x ≥ 0 (right of y-axis ray)")
        self._draw_polygon(ax, square, color='lightblue', alpha=0.3, label='Original')
        self._draw_halfplane_boundary(ax, np.pi/2)
        result = clip_polygon_halfplane(square, plane_angle=np.pi/2, keep_left=False)
        self._draw_polygon(ax, result, color='green', alpha=0.5, edgecolor='darkgreen', label='Clipped')
        ax.legend(loc='upper right')
        
        fig.suptitle("Half-Plane Clipping: Partial Clips", fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_figure(fig, "halfplane_partial_clip")

    def test_visual_halfplane_clip_diagonal(self):
        """Visual: Clipping with diagonal half-planes."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Square from (0,0) to (2,2)
        square = np.array([
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
        ], dtype=np.float32)
        
        # Test case 1: 45° angle, keep left (y > x region)
        ax = axes[0]
        self._setup_axes(ax, "Clip: 45° ray, keep left (y ≥ x)", xlim=(-1, 3), ylim=(-1, 3))
        self._draw_polygon(ax, square, color='lightblue', alpha=0.3, label='Original')
        self._draw_halfplane_boundary(ax, np.pi/4, extent=3)
        result = clip_polygon_halfplane(square, plane_angle=np.pi/4, keep_left=True)
        self._draw_polygon(ax, result, color='blue', alpha=0.5, edgecolor='darkblue', label='Clipped')
        ax.legend(loc='upper right')
        
        # Test case 2: 45° angle, keep right (y < x region)
        ax = axes[1]
        self._setup_axes(ax, "Clip: 45° ray, keep right (y ≤ x)", xlim=(-1, 3), ylim=(-1, 3))
        self._draw_polygon(ax, square, color='lightblue', alpha=0.3, label='Original')
        self._draw_halfplane_boundary(ax, np.pi/4, extent=3)
        result = clip_polygon_halfplane(square, plane_angle=np.pi/4, keep_left=False)
        self._draw_polygon(ax, result, color='orange', alpha=0.5, edgecolor='darkorange', label='Clipped')
        ax.legend(loc='upper right')
        
        # Test case 3: -45° angle (135°), keep left
        ax = axes[2]
        self._setup_axes(ax, "Clip: 135° ray, keep left", xlim=(-1, 3), ylim=(-1, 3))
        self._draw_polygon(ax, square, color='lightblue', alpha=0.3, label='Original')
        self._draw_halfplane_boundary(ax, 3*np.pi/4, extent=3)
        result = clip_polygon_halfplane(square, plane_angle=3*np.pi/4, keep_left=True)
        self._draw_polygon(ax, result, color='purple', alpha=0.5, edgecolor='darkviolet', label='Clipped')
        ax.legend(loc='upper right')
        
        fig.suptitle("Half-Plane Clipping: Diagonal Boundaries", fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_figure(fig, "halfplane_diagonal_clip")

    def test_visual_halfplane_fully_inside_outside(self):
        """Visual: Polygons fully inside or outside clipping region."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Test case 1: Polygon fully inside (should be unchanged)
        ax = axes[0]
        self._setup_axes(ax, "Fully Inside: No clipping")
        triangle = np.array([
            [0.5, 0.5],
            [2.0, 0.5],
            [1.25, 2.0],
        ], dtype=np.float32)
        self._draw_polygon(ax, triangle, color='lightblue', alpha=0.3, label='Original')
        self._draw_halfplane_boundary(ax, 0.0)
        result = clip_polygon_halfplane(triangle, plane_angle=0.0, keep_left=True)
        self._draw_polygon(ax, result, color='blue', alpha=0.5, edgecolor='darkblue', label='Clipped')
        ax.legend(loc='upper right')
        ax.annotate('All vertices preserved', xy=(1.25, 1.0), fontsize=10, ha='center')
        
        # Test case 2: Polygon fully outside (should be empty)
        ax = axes[1]
        self._setup_axes(ax, "Fully Outside: Complete removal")
        triangle = np.array([
            [0.5, -2.0],
            [2.0, -2.0],
            [1.25, -0.5],
        ], dtype=np.float32)
        self._draw_polygon(ax, triangle, color='lightblue', alpha=0.3, label='Original')
        self._draw_halfplane_boundary(ax, 0.0)
        result = clip_polygon_halfplane(triangle, plane_angle=0.0, keep_left=True)
        if result.shape[0] == 0:
            ax.annotate('Result: Empty polygon', xy=(1.25, 0.5), fontsize=10, 
                       ha='center', color='red', fontweight='bold')
        ax.legend(loc='upper right')
        
        # Test case 3: Touching the boundary
        ax = axes[2]
        self._setup_axes(ax, "Vertex on Boundary: Included")
        triangle = np.array([
            [1.0, 0.0],   # On boundary
            [2.0, 1.0],
            [0.5, 1.5],
        ], dtype=np.float32)
        self._draw_polygon(ax, triangle, color='lightblue', alpha=0.3, label='Original')
        self._draw_halfplane_boundary(ax, 0.0)
        result = clip_polygon_halfplane(triangle, plane_angle=0.0, keep_left=True)
        self._draw_polygon(ax, result, color='green', alpha=0.5, edgecolor='darkgreen', label='Clipped')
        ax.plot(1.0, 0.0, 'ro', markersize=10, label='On boundary')
        ax.legend(loc='upper right')
        
        fig.suptitle("Half-Plane Clipping: Boundary Cases", fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_figure(fig, "halfplane_boundary_cases")

    def test_visual_bounding_box(self):
        """Visual: Bounding box computation for various shapes."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Test case 1: Axis-aligned square
        ax = axes[0]
        self._setup_axes(ax, "Bounding Box: Square", xlim=(-1, 4), ylim=(-1, 4))
        square = np.array([
            [0.5, 0.5],
            [2.5, 0.5],
            [2.5, 2.5],
            [0.5, 2.5],
        ], dtype=np.float32)
        self._draw_polygon(ax, square, color='blue', alpha=0.3, label='Polygon')
        min_pt, max_pt = compute_bounding_box(square)
        bbox_rect = mpatches.Rectangle(tuple(min_pt), max_pt[0]-min_pt[0], max_pt[1]-min_pt[1],
                                        fill=False, edgecolor='red', linewidth=2, 
                                        linestyle='--', label='AABB')
        ax.add_patch(bbox_rect)
        ax.plot(*min_pt, 'r^', markersize=10, label=f'Min: {min_pt}')
        ax.plot(*max_pt, 'rv', markersize=10, label=f'Max: {max_pt}')
        ax.legend(loc='upper right')
        
        # Test case 2: Rotated triangle
        ax = axes[1]
        self._setup_axes(ax, "Bounding Box: Triangle", xlim=(-1, 5), ylim=(-1, 5))
        triangle = np.array([
            [1.0, 0.5],
            [4.0, 1.5],
            [2.0, 3.5],
        ], dtype=np.float32)
        self._draw_polygon(ax, triangle, color='green', alpha=0.3, label='Polygon')
        min_pt, max_pt = compute_bounding_box(triangle)
        bbox_rect = mpatches.Rectangle(tuple(min_pt), max_pt[0]-min_pt[0], max_pt[1]-min_pt[1],
                                        fill=False, edgecolor='red', linewidth=2, 
                                        linestyle='--', label='AABB')
        ax.add_patch(bbox_rect)
        ax.plot(*min_pt, 'r^', markersize=10, label=f'Min: ({min_pt[0]:.1f}, {min_pt[1]:.1f})')
        ax.plot(*max_pt, 'rv', markersize=10, label=f'Max: ({max_pt[0]:.1f}, {max_pt[1]:.1f})')
        ax.legend(loc='upper right')
        
        # Test case 3: Complex polygon spanning negative coords
        ax = axes[2]
        self._setup_axes(ax, "Bounding Box: Complex Shape", xlim=(-3, 4), ylim=(-3, 4))
        polygon = np.array([
            [-2.0, -1.0],
            [1.0, -2.0],
            [3.0, 0.0],
            [2.0, 3.0],
            [-1.0, 2.0],
        ], dtype=np.float32)
        self._draw_polygon(ax, polygon, color='purple', alpha=0.3, label='Polygon')
        min_pt, max_pt = compute_bounding_box(polygon)
        bbox_rect = mpatches.Rectangle(tuple(min_pt), max_pt[0]-min_pt[0], max_pt[1]-min_pt[1],
                                        fill=False, edgecolor='red', linewidth=2, 
                                        linestyle='--', label='AABB')
        ax.add_patch(bbox_rect)
        ax.plot(*min_pt, 'r^', markersize=10, label=f'Min: ({min_pt[0]:.1f}, {min_pt[1]:.1f})')
        ax.plot(*max_pt, 'rv', markersize=10, label=f'Max: ({max_pt[0]:.1f}, {max_pt[1]:.1f})')
        ax.legend(loc='upper right')
        
        fig.suptitle("Axis-Aligned Bounding Box (AABB) Computation", fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_figure(fig, "bounding_box_computation")

    def test_visual_complex_clipping_sequence(self):
        """Visual: Multiple successive half-plane clips (wedge preview)."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Large polygon to clip into a wedge
        polygon = np.array([
            [-2.0, -2.0],
            [3.0, -2.0],
            [3.0, 3.0],
            [-2.0, 3.0],
        ], dtype=np.float32)
        
        # Define a wedge: angles from 30° to 90°
        alpha_min = np.radians(30)
        alpha_max = np.radians(90)
        
        # Row 1: Step by step clipping
        ax = axes[0, 0]
        self._setup_axes(ax, "Step 0: Original Polygon", xlim=(-3, 4), ylim=(-3, 4))
        self._draw_polygon(ax, polygon, color='lightblue', alpha=0.5, label='Original')
        ax.legend(loc='upper right')
        
        ax = axes[0, 1]
        self._setup_axes(ax, f"Step 1: Clip at α_min={np.degrees(alpha_min):.0f}°", xlim=(-3, 4), ylim=(-3, 4))
        self._draw_polygon(ax, polygon, color='lightblue', alpha=0.2)
        self._draw_halfplane_boundary(ax, alpha_min, extent=4, color='red')
        clipped1 = clip_polygon_halfplane(polygon, plane_angle=alpha_min, keep_left=True)
        self._draw_polygon(ax, clipped1, color='blue', alpha=0.5, edgecolor='darkblue', label='After clip 1')
        ax.legend(loc='upper right')
        
        ax = axes[0, 2]
        self._setup_axes(ax, f"Step 2: Clip at α_max={np.degrees(alpha_max):.0f}°", xlim=(-3, 4), ylim=(-3, 4))
        self._draw_polygon(ax, clipped1, color='lightblue', alpha=0.2)
        self._draw_halfplane_boundary(ax, alpha_max, extent=4, color='green')
        clipped2 = clip_polygon_halfplane(clipped1, plane_angle=alpha_max, keep_left=False)
        self._draw_polygon(ax, clipped2, color='orange', alpha=0.5, edgecolor='darkorange', label='After clip 2')
        ax.legend(loc='upper right')
        
        # Row 2: Different wedge angles
        angles_sets = [
            (np.radians(-30), np.radians(30), "Wedge: -30° to 30°"),
            (np.radians(45), np.radians(135), "Wedge: 45° to 135°"),
            (np.radians(150), np.radians(-150), "Wedge: 150° to -150° (wrap)"),
        ]
        
        for i, (a_min, a_max, title) in enumerate(angles_sets):
            ax = axes[1, i]
            self._setup_axes(ax, title, xlim=(-3, 4), ylim=(-3, 4))
            
            # Draw origin rays for the wedge
            self._draw_halfplane_boundary(ax, a_min, extent=4, color='red')
            self._draw_halfplane_boundary(ax, a_max, extent=4, color='green')
            
            # Apply clipping
            temp = clip_polygon_halfplane(polygon, plane_angle=a_min, keep_left=True)
            if temp.shape[0] >= 3:
                result = clip_polygon_halfplane(temp, plane_angle=a_max, keep_left=False)
                if result.shape[0] >= 3:
                    self._draw_polygon(ax, result, color='purple', alpha=0.5, 
                                       edgecolor='darkviolet', label='Wedge result')
            
            ax.legend(loc='upper right')
        
        fig.suptitle("Wedge Clipping Preview: Two Half-Planes", fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_figure(fig, "wedge_clipping_preview")

    def test_visual_edge_intersection_accuracy(self):
        """Visual: Verify intersection point accuracy."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Test case 1: Triangle with apex above, base below x-axis
        ax = axes[0]
        self._setup_axes(ax, "Intersection Accuracy: Triangle", xlim=(-3, 3), ylim=(-2, 3))
        
        triangle = np.array([
            [0.0, 2.0],    # Apex above
            [-2.0, -1.0],  # Below
            [2.0, -1.0],   # Below
        ], dtype=np.float32)
        
        self._draw_polygon(ax, triangle, color='lightblue', alpha=0.3, label='Original')
        self._draw_halfplane_boundary(ax, 0.0)
        
        result = clip_polygon_halfplane(triangle, plane_angle=0.0, keep_left=True)
        self._draw_polygon(ax, result, color='blue', alpha=0.5, edgecolor='darkblue', label='Clipped')
        
        # Mark intersection points (should be on x-axis)
        for v in result:
            if abs(v[1]) < 0.01:  # On x-axis
                ax.plot(v[0], v[1], 'go', markersize=12, zorder=5)
                ax.annotate(f'({v[0]:.2f}, {v[1]:.2f})', xy=(v[0], v[1]+0.2), 
                           ha='center', fontsize=9)
        
        # Expected intersections: parametric solution
        # Line from (0,2) to (-2,-1): y = 2 - 1.5*t, x = -2*t
        # When y=0: t = 4/3, x = -8/3 ≈ -0.67 (wait, let me recalc)
        # Actually: from (0,2) to (-2,-1), direction is (-2, -3)
        # Point: (0,2) + t*(-2,-3) = (-2t, 2-3t)
        # When y=0: 2-3t=0 -> t=2/3 -> x = -4/3 ≈ -1.33
        ax.annotate('Expected: x ≈ ±1.33', xy=(0, -0.5), ha='center', fontsize=10, color='red')
        ax.legend(loc='upper right')
        
        # Test case 2: Polygon with edge parallel to boundary
        ax = axes[1]
        self._setup_axes(ax, "Edge Cases: Various Orientations", xlim=(-3, 3), ylim=(-3, 3))
        
        # Pentagon with various edge orientations
        pentagon = np.array([
            [0.0, 2.0],
            [2.0, 0.5],
            [1.5, -1.5],
            [-1.5, -1.5],
            [-2.0, 0.5],
        ], dtype=np.float32)
        
        self._draw_polygon(ax, pentagon, color='lightblue', alpha=0.3, label='Original')
        self._draw_halfplane_boundary(ax, 0.0)
        
        result = clip_polygon_halfplane(pentagon, plane_angle=0.0, keep_left=True)
        self._draw_polygon(ax, result, color='green', alpha=0.5, edgecolor='darkgreen', label='Clipped')
        
        # Mark all intersection points
        for v in result:
            ax.plot(v[0], v[1], 'mo', markersize=6, zorder=5)
        
        ax.legend(loc='upper right')
        
        fig.suptitle("Edge-Boundary Intersection Accuracy", fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_figure(fig, "intersection_accuracy")


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
class TestKnownGeometricConstructions:
    """Compare clipping results against known geometric constructions."""

    def _draw_polygon(self, ax, polygon, color='blue', alpha=0.3, label=None, edgecolor=None):
        """Draw a polygon on the axes."""
        if polygon.shape[0] < 3:
            return
        if edgecolor is None:
            edgecolor = color
        patch = mpatches.Polygon(polygon, closed=True, 
                                  facecolor=color, alpha=alpha, 
                                  edgecolor=edgecolor, linewidth=2,
                                  label=label)
        ax.add_patch(patch)

    def _draw_circle(self, ax, radius, color='red', linestyle='--', label=None):
        """Draw a circle centered at origin."""
        circle = mpatches.Circle((0, 0), radius, fill=False,
                                edgecolor=color, linewidth=2, linestyle=linestyle,
                                label=label)
        ax.add_patch(circle)

    def _draw_wedge(self, ax, alpha_min, alpha_max, radius, color='red', alpha=0.1, label=None):
        """Draw a wedge (circular sector) from origin."""
        # Convert to degrees for matplotlib
        theta1 = np.degrees(alpha_min)
        theta2 = np.degrees(alpha_max)
        wedge = mpatches.Wedge((0, 0), radius, theta1, theta2,
                               facecolor=color, alpha=alpha,
                               edgecolor=color, linewidth=2, linestyle='--',
                               label=label)
        ax.add_patch(wedge)

    def _setup_axes(self, ax, title, xlim=(-3, 4), ylim=(-3, 4)):
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
        ax.plot(0, 0, 'ko', markersize=8)  # Origin marker

    def test_visual_compare_to_analytical_solution(self):
        """Compare algorithm output to analytically computed result."""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Setup
        ax.set_xlim(-1, 4)
        ax.set_ylim(-1, 4)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.set_title("Analytical Verification: Unit Square Clipped at 45°")
        
        # Unit square at origin
        square = np.array([
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
        ], dtype=np.float32)
        
        # Analytical result: clipping by y=x (45° line), keeping y >= x
        # Should result in triangle: (0,0), (2,2), (0,2)
        analytical_result = np.array([
            [0.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
        ], dtype=np.float32)
        
        # Algorithm result
        algorithm_result = clip_polygon_halfplane(square, plane_angle=np.pi/4, keep_left=True)
        
        # Draw original
        patch = mpatches.Polygon(square, closed=True, 
                                  facecolor='lightgray', alpha=0.3,
                                  edgecolor='gray', linewidth=2, linestyle='--',
                                  label='Original square')
        ax.add_patch(patch)
        
        # Draw analytical (expected)
        patch = mpatches.Polygon(analytical_result, closed=True, 
                                  facecolor='none', alpha=1,
                                  edgecolor='green', linewidth=3,
                                  label='Analytical (expected)')
        ax.add_patch(patch)
        
        # Draw algorithm result
        # Remove duplicates for cleaner visualization
        unique_verts = []
        for v in algorithm_result:
            is_dup = False
            for uv in unique_verts:
                if np.allclose(v, uv, atol=1e-5):
                    is_dup = True
                    break
            if not is_dup:
                unique_verts.append(v)
        unique_verts = np.array(unique_verts, dtype=np.float32)
        
        if len(unique_verts) >= 3:
            patch = mpatches.Polygon(unique_verts, closed=True, 
                                      facecolor='blue', alpha=0.3,
                                      edgecolor='blue', linewidth=2,
                                      label='Algorithm result')
            ax.add_patch(patch)
        
        # Draw the 45° line
        ax.plot([0, 3], [0, 3], 'r-', linewidth=2, label='y = x (boundary)')
        ax.arrow(0, 0, 2*np.cos(np.pi/4), 2*np.sin(np.pi/4), 
                head_width=0.1, head_length=0.05, fc='red', ec='red')
        
        # Mark vertices
        for i, v in enumerate(analytical_result):
            ax.plot(v[0], v[1], 'go', markersize=12)
            ax.annotate(f'A{i}: ({v[0]:.1f}, {v[1]:.1f})', 
                       xy=(v[0]-0.3, v[1]+0.15), fontsize=9, color='green')
        
        ax.legend(loc='upper right')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        plt.tight_layout()
        save_figure(fig, "analytical_comparison")
        
        # Verify the algorithm produced correct unique vertices
        for av in analytical_result:
            found = False
            for uv in unique_verts:
                if np.allclose(av, uv, atol=1e-5):
                    found = True
                    break
            assert found, f"Expected vertex {av} not found in algorithm result"


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
class TestCircleClippingVisual:
    """Visual validation tests for circle clipping operations (Step 2.2)."""

    def _draw_polygon(self, ax, polygon, color='blue', alpha=0.3, label=None, edgecolor=None):
        """Draw a polygon on the axes."""
        if polygon.shape[0] < 3:
            return
        if edgecolor is None:
            edgecolor = color
        patch = mpatches.Polygon(polygon, closed=True, 
                                  facecolor=color, alpha=alpha, 
                                  edgecolor=edgecolor, linewidth=2,
                                  label=label)
        ax.add_patch(patch)

    def _draw_circle(self, ax, radius, color='red', linestyle='--', label=None):
        """Draw a circle centered at origin."""
        circle = mpatches.Circle((0, 0), radius, fill=False,
                                 edgecolor=color, linewidth=2, linestyle=linestyle,
                                 label=label)
        ax.add_patch(circle)

    def _setup_axes(self, ax, title, xlim=(-5, 5), ylim=(-5, 5)):
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
        ax.plot(0, 0, 'ko', markersize=8)  # Origin marker

    def test_visual_circle_clip_basic(self):
        """Visual: Basic circle clipping scenarios."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        radius = 3.0
        
        # Test case 1: Polygon fully inside circle
        ax = axes[0]
        self._setup_axes(ax, "Fully Inside: No clipping")
        square = np.array([
            [0.5, 0.5],
            [1.5, 0.5],
            [1.5, 1.5],
            [0.5, 1.5],
        ], dtype=np.float32)
        self._draw_circle(ax, radius, label=f'Circle r={radius}')
        self._draw_polygon(ax, square, color='lightblue', alpha=0.3, label='Original')
        result = clip_polygon_circle(square, radius)
        self._draw_polygon(ax, result, color='blue', alpha=0.5, edgecolor='darkblue', label='Clipped')
        ax.legend(loc='upper right')
        
        # Test case 2: Polygon fully outside circle
        ax = axes[1]
        self._setup_axes(ax, "Fully Outside: Complete removal")
        square = np.array([
            [4.0, 4.0],
            [5.0, 4.0],
            [5.0, 5.0],
            [4.0, 5.0],
        ], dtype=np.float32)
        self._draw_circle(ax, radius, label=f'Circle r={radius}')
        self._draw_polygon(ax, square, color='lightblue', alpha=0.3, label='Original')
        result = clip_polygon_circle(square, radius)
        if result.shape[0] == 0:
            ax.annotate('Result: Empty polygon', xy=(0, 0), fontsize=10, 
                       ha='center', color='red', fontweight='bold')
        ax.legend(loc='upper right')
        
        # Test case 3: Polygon partially inside
        ax = axes[2]
        self._setup_axes(ax, "Partial: Some vertices clipped")
        square = np.array([
            [1.0, 1.0],
            [4.0, 1.0],
            [4.0, 4.0],
            [1.0, 4.0],
        ], dtype=np.float32)
        self._draw_circle(ax, radius, label=f'Circle r={radius}')
        self._draw_polygon(ax, square, color='lightblue', alpha=0.3, label='Original')
        result = clip_polygon_circle(square, radius)
        self._draw_polygon(ax, result, color='green', alpha=0.5, edgecolor='darkgreen', label='Clipped')
        # Mark intersection points on circle
        if result.shape[0] > 0:
            distances = np.sqrt(result[:, 0]**2 + result[:, 1]**2)
            for i, (v, d) in enumerate(zip(result, distances)):
                if abs(d - radius) < 0.1:
                    ax.plot(v[0], v[1], 'ro', markersize=8, zorder=5)
        ax.legend(loc='upper right')
        
        fig.suptitle("Circle Clipping: Basic Scenarios", fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_figure(fig, "circle_clip_basic")

    def test_visual_circle_clip_edge_through(self):
        """Visual: Edge passes through circle (both endpoints outside)."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        radius = 2.0
        
        # Test case 1: Horizontal rectangle through origin
        ax = axes[0]
        self._setup_axes(ax, "Edge Through Circle: Horizontal")
        rectangle = np.array([
            [-4.0, -0.5],
            [4.0, -0.5],
            [4.0, 0.5],
            [-4.0, 0.5],
        ], dtype=np.float32)
        self._draw_circle(ax, radius, label=f'Circle r={radius}')
        self._draw_polygon(ax, rectangle, color='lightblue', alpha=0.3, label='Original')
        result = clip_polygon_circle(rectangle, radius)
        self._draw_polygon(ax, result, color='blue', alpha=0.5, edgecolor='darkblue', label='Clipped')
        ax.legend(loc='upper right')
        
        # Test case 2: Diagonal rectangle
        ax = axes[1]
        self._setup_axes(ax, "Edge Through Circle: Diagonal")
        # Rectangle from bottom-left to top-right
        rectangle = np.array([
            [-4.0, -4.0],
            [-3.0, -4.0],
            [4.0, 3.0],
            [3.0, 3.0],
        ], dtype=np.float32)
        self._draw_circle(ax, radius, label=f'Circle r={radius}')
        self._draw_polygon(ax, rectangle, color='lightblue', alpha=0.3, label='Original')
        result = clip_polygon_circle(rectangle, radius)
        if result.shape[0] >= 3:
            self._draw_polygon(ax, result, color='orange', alpha=0.5, edgecolor='darkorange', label='Clipped')
        ax.legend(loc='upper right')
        
        fig.suptitle("Circle Clipping: Edges Passing Through", fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_figure(fig, "circle_clip_edge_through")

    def test_visual_circle_clip_various_shapes(self):
        """Visual: Circle clipping with various polygon shapes."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        radius = 3.0
        
        # Triangle extending beyond circle
        ax = axes[0, 0]
        self._setup_axes(ax, "Triangle Clipped")
        triangle = np.array([
            [0.0, 0.0],
            [5.0, 0.0],
            [2.5, 5.0],
        ], dtype=np.float32)
        self._draw_circle(ax, radius, label=f'Circle r={radius}')
        self._draw_polygon(ax, triangle, color='lightblue', alpha=0.3, label='Original')
        result = clip_polygon_circle(triangle, radius)
        self._draw_polygon(ax, result, color='blue', alpha=0.5, edgecolor='darkblue', label='Clipped')
        ax.legend(loc='upper right')
        
        # Pentagon
        ax = axes[0, 1]
        self._setup_axes(ax, "Pentagon Clipped")
        angles = np.linspace(0, 2*np.pi, 6)[:-1] + np.pi/2
        pentagon = np.array([[4*np.cos(a), 4*np.sin(a)] for a in angles], dtype=np.float32)
        self._draw_circle(ax, radius, label=f'Circle r={radius}')
        self._draw_polygon(ax, pentagon, color='lightblue', alpha=0.3, label='Original')
        result = clip_polygon_circle(pentagon, radius)
        self._draw_polygon(ax, result, color='green', alpha=0.5, edgecolor='darkgreen', label='Clipped')
        ax.legend(loc='upper right')
        
        # Star shape (non-convex)
        ax = axes[0, 2]
        self._setup_axes(ax, "Star Shape Clipped")
        star_angles = np.linspace(0, 2*np.pi, 11)[:-1]
        star_radii = [4 if i % 2 == 0 else 2 for i in range(10)]
        star = np.array([[r*np.cos(a), r*np.sin(a)] for a, r in zip(star_angles, star_radii)], dtype=np.float32)
        self._draw_circle(ax, radius, label=f'Circle r={radius}')
        self._draw_polygon(ax, star, color='lightblue', alpha=0.3, label='Original')
        result = clip_polygon_circle(star, radius)
        self._draw_polygon(ax, result, color='purple', alpha=0.5, edgecolor='darkviolet', label='Clipped')
        ax.legend(loc='upper right')

    def test_visual_circle_clip_arc_sampling(self):
        """Visual: Highlight sampled arcs when circle boundary dominates."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Case 1: Polygon fully encloses the circle -> expect circular result
        ax = axes[0]
        self._setup_axes(ax, "Polygon Encloses Circle")
        square = np.array([
            [-4.0, -4.0],
            [4.0, -4.0],
            [4.0, 4.0],
            [-4.0, 4.0],
        ], dtype=np.float32)
        radius = 1.5
        self._draw_circle(ax, radius, label=f'Circle r={radius}')
        self._draw_polygon(ax, square, color='lightblue', alpha=0.2, label='Original')
        result = clip_polygon_circle(square, radius)
        self._draw_polygon(ax, result, color='blue', alpha=0.5, edgecolor='darkblue', label='Clipped (Arc Sampled)')
        ax.annotate(f"{result.shape[0]} samples", xy=(0.05, 0.9), xycoords='axes fraction',
                    fontsize=10, color='blue', fontweight='bold')
        ax.legend(loc='upper right')

        # Case 2: Rectangle above circle -> clipped area defined by arc boundary
        ax = axes[1]
        self._setup_axes(ax, "Arc Boundary Clipping", ylim=(-0.5, 4))
        rectangle = np.array([
            [-1.2, 0.0],
            [1.2, 0.0],
            [1.2, 3.5],
            [-1.2, 3.5],
        ], dtype=np.float32)
        radius = 1.0
        self._draw_circle(ax, radius, label=f'Circle r={radius}')
        self._draw_polygon(ax, rectangle, color='lightblue', alpha=0.2, label='Original')
        result = clip_polygon_circle(rectangle, radius)
        self._draw_polygon(ax, result, color='green', alpha=0.5, edgecolor='darkgreen', label='Clipped (Arc Preserved)')
        # Highlight arc points
        arc_points = result[result[:, 1] > 0.1]
        ax.scatter(arc_points[:, 0], arc_points[:, 1], color='red', s=25, zorder=5, label='Arc Samples')
        ax.legend(loc='upper right')

        fig.suptitle("Circle Clipping: Sampled Arc Visualization", fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_figure(fig, "circle_clip_arc_sampling")


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
class TestWedgeClippingVisual:
    """Visual validation tests for wedge clipping operations (Step 2.2)."""

    def _draw_polygon(self, ax, polygon, color='blue', alpha=0.3, label=None, edgecolor=None):
        """Draw a polygon on the axes."""
        if polygon.shape[0] < 3:
            return
        if edgecolor is None:
            edgecolor = color
        patch = mpatches.Polygon(polygon, closed=True, 
                                  facecolor=color, alpha=alpha, 
                                  edgecolor=edgecolor, linewidth=2,
                                  label=label)
        ax.add_patch(patch)

    def _draw_wedge(self, ax, alpha_min, alpha_max, radius, color='red', alpha=0.1, label=None):
        """Draw a wedge (circular sector) from origin."""
        # Convert to degrees for matplotlib
        theta1 = np.degrees(alpha_min)
        theta2 = np.degrees(alpha_max)
        wedge = mpatches.Wedge((0, 0), radius, theta1, theta2,
                               facecolor=color, alpha=alpha,
                               edgecolor=color, linewidth=2, linestyle='--',
                               label=label)
        ax.add_patch(wedge)

    def _draw_wedge_boundaries(self, ax, alpha_min, alpha_max, radius, color='red'):
        """Draw wedge boundary rays and arc."""
        # Draw rays from origin
        for angle, lbl in [(alpha_min, 'α_min'), (alpha_max, 'α_max')]:
            x_end = radius * np.cos(angle)
            y_end = radius * np.sin(angle)
            ax.plot([0, x_end], [0, y_end], color=color, linewidth=2, linestyle='--')
            ax.annotate(lbl, xy=(x_end*0.7, y_end*0.7), fontsize=9, color=color)
        
        # Draw arc
        arc_angles = np.linspace(alpha_min, alpha_max, 50)
        arc_x = radius * np.cos(arc_angles)
        arc_y = radius * np.sin(arc_angles)
        ax.plot(arc_x, arc_y, color=color, linewidth=2, linestyle='--')

    def _setup_axes(self, ax, title, xlim=(-5, 5), ylim=(-5, 5)):
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
        ax.plot(0, 0, 'ko', markersize=8)  # Origin marker

    def test_visual_wedge_clip_basic(self):
        """Visual: Basic wedge clipping scenarios."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Test case 1: Polygon fully inside wedge
        ax = axes[0]
        self._setup_axes(ax, "Fully Inside Wedge")
        alpha_min, alpha_max, radius = np.radians(20), np.radians(70), 4.0
        square = np.array([
            [1.0, 1.0],
            [2.0, 1.0],
            [2.0, 2.0],
            [1.0, 2.0],
        ], dtype=np.float32)
        self._draw_wedge(ax, alpha_min, alpha_max, radius, label='Wedge')
        self._draw_polygon(ax, square, color='lightblue', alpha=0.3, label='Original')
        result = clip_polygon_to_wedge(square, alpha_min, alpha_max, radius)
        if result is not None:
            self._draw_polygon(ax, result, color='blue', alpha=0.5, edgecolor='darkblue', label='Clipped')
        ax.legend(loc='upper right')
        
        # Test case 2: Polygon fully outside wedge
        ax = axes[1]
        self._setup_axes(ax, "Fully Outside Wedge")
        square = np.array([
            [-3.0, -3.0],
            [-2.0, -3.0],
            [-2.0, -2.0],
            [-3.0, -2.0],
        ], dtype=np.float32)
        self._draw_wedge(ax, alpha_min, alpha_max, radius, label='Wedge')
        self._draw_polygon(ax, square, color='lightblue', alpha=0.3, label='Original')
        result = clip_polygon_to_wedge(square, alpha_min, alpha_max, radius)
        if result is None:
            ax.annotate('Result: None (outside)', xy=(0, -1), fontsize=10, 
                       ha='center', color='red', fontweight='bold')
        ax.legend(loc='upper right')
        
        # Test case 3: Partial clipping
        ax = axes[2]
        self._setup_axes(ax, "Partial Wedge Clipping")
        square = np.array([
            [0.5, 0.5],
            [3.5, 0.5],
            [3.5, 3.5],
            [0.5, 3.5],
        ], dtype=np.float32)
        self._draw_wedge(ax, alpha_min, alpha_max, radius, label='Wedge')
        self._draw_polygon(ax, square, color='lightblue', alpha=0.3, label='Original')
        result = clip_polygon_to_wedge(square, alpha_min, alpha_max, radius)
        if result is not None:
            self._draw_polygon(ax, result, color='green', alpha=0.5, edgecolor='darkgreen', label='Clipped')
        ax.legend(loc='upper right')
        
        fig.suptitle("Wedge Clipping: Basic Scenarios", fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_figure(fig, "wedge_clip_basic")

    def test_visual_wedge_clip_fov_angles(self):
        """Visual: Wedge clipping with different FOV angles."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        radius = 4.0
        # Large polygon to clip
        polygon = np.array([
            [-3.0, -3.0],
            [4.0, -3.0],
            [4.0, 4.0],
            [-3.0, 4.0],
        ], dtype=np.float32)
        
        fov_configs = [
            (np.radians(-15), np.radians(15), "30° FOV (narrow)"),
            (np.radians(-30), np.radians(30), "60° FOV"),
            (np.radians(-45), np.radians(45), "90° FOV"),
            (np.radians(-60), np.radians(60), "120° FOV (wide)"),
            (np.radians(0), np.radians(90), "90° FOV (first quadrant)"),
            (np.radians(45), np.radians(135), "90° FOV (upper quadrant)"),
        ]
        
        for ax, (alpha_min, alpha_max, title) in zip(axes.flat, fov_configs):
            self._setup_axes(ax, title)
            self._draw_wedge(ax, alpha_min, alpha_max, radius, label='Wedge')
            self._draw_polygon(ax, polygon, color='lightblue', alpha=0.2, label='Original')
            result = clip_polygon_to_wedge(polygon, alpha_min, alpha_max, radius)
            if result is not None:
                self._draw_polygon(ax, result, color='blue', alpha=0.5, edgecolor='darkblue', label='Clipped')
            else:
                ax.annotate('Result: None', xy=(0, 0), fontsize=10, 
                           ha='center', color='red', fontweight='bold')
            ax.legend(loc='upper right', fontsize=8)
        
        fig.suptitle("Wedge Clipping: Various Field of View Angles", fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_figure(fig, "wedge_clip_fov_angles")

    def test_visual_wedge_clip_pipeline_steps(self):
        """Visual: Step-by-step wedge clipping pipeline."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        alpha_min = np.radians(30)
        alpha_max = np.radians(75)
        radius = 3.5
        
        # Large polygon
        polygon = np.array([
            [-2.0, -2.0],
            [5.0, -2.0],
            [5.0, 5.0],
            [-2.0, 5.0],
        ], dtype=np.float32)
        
        # Step 0: Original
        ax = axes[0, 0]
        self._setup_axes(ax, "Step 0: Original Polygon")
        self._draw_wedge(ax, alpha_min, alpha_max, radius, alpha=0.05, label='Target wedge')
        self._draw_polygon(ax, polygon, color='lightblue', alpha=0.5, label='Original')
        ax.legend(loc='upper right')
        
        # Step 1: After alpha_min half-plane clip
        ax = axes[0, 1]
        self._setup_axes(ax, f"Step 1: Clip at α_min = {np.degrees(alpha_min):.0f}°")
        step1 = clip_polygon_halfplane(polygon, plane_angle=alpha_min, keep_left=True)
        # Draw half-plane boundary
        x_end = 5 * np.cos(alpha_min)
        y_end = 5 * np.sin(alpha_min)
        ax.plot([0, x_end], [0, y_end], 'r-', linewidth=2, label='α_min boundary')
        ax.fill([0, x_end, x_end, 0], [0, y_end, -5, -5], alpha=0.1, color='red')
        self._draw_polygon(ax, polygon, color='lightblue', alpha=0.2)
        self._draw_polygon(ax, step1, color='blue', alpha=0.5, edgecolor='darkblue', label='After clip 1')
        ax.legend(loc='upper right')
        
        # Step 2: After alpha_max half-plane clip
        ax = axes[1, 0]
        self._setup_axes(ax, f"Step 2: Clip at α_max = {np.degrees(alpha_max):.0f}°")
        step2 = clip_polygon_halfplane(step1, plane_angle=alpha_max, keep_left=False)
        # Draw half-plane boundary
        x_end = 5 * np.cos(alpha_max)
        y_end = 5 * np.sin(alpha_max)
        ax.plot([0, x_end], [0, y_end], 'g-', linewidth=2, label='α_max boundary')
        ax.fill([0, x_end, -5, -5], [0, y_end, 5, 0], alpha=0.1, color='green')
        self._draw_polygon(ax, step1, color='lightblue', alpha=0.2)
        self._draw_polygon(ax, step2, color='orange', alpha=0.5, edgecolor='darkorange', label='After clip 2')
        ax.legend(loc='upper right')
        
        # Step 3: After circle clip (final result)
        ax = axes[1, 1]
        self._setup_axes(ax, f"Step 3: Circle clip at r = {radius}")
        from view_arc.obstacle.clipping import clip_polygon_circle
        step3 = clip_polygon_circle(step2, radius)
        circle = mpatches.Circle((0, 0), radius, fill=False,
                     edgecolor='purple', linewidth=2, linestyle='--',
                     label=f'Circle r={radius}')
        ax.add_patch(circle)
        self._draw_polygon(ax, step2, color='lightblue', alpha=0.2)
        self._draw_polygon(ax, step3, color='purple', alpha=0.5, edgecolor='darkviolet', label='Final result')
        ax.legend(loc='upper right')
        
        fig.suptitle("Wedge Clipping Pipeline: Step by Step", fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_figure(fig, "wedge_clip_pipeline_steps")

    def test_visual_wedge_clip_complex_polygons(self):
        """Visual: Wedge clipping with complex polygon shapes."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        alpha_min = np.radians(15)
        alpha_max = np.radians(75)
        radius = 4.0
        
        # Triangle
        ax = axes[0, 0]
        self._setup_axes(ax, "Triangle in Wedge")
        triangle = np.array([
            [0.5, 0.5],
            [4.0, 0.5],
            [2.0, 4.0],
        ], dtype=np.float32)
        self._draw_wedge(ax, alpha_min, alpha_max, radius, label='Wedge')
        self._draw_polygon(ax, triangle, color='lightblue', alpha=0.3, label='Original')
        result = clip_polygon_to_wedge(triangle, alpha_min, alpha_max, radius)
        if result is not None:
            self._draw_polygon(ax, result, color='blue', alpha=0.5, edgecolor='darkblue', label='Clipped')
        ax.legend(loc='upper right')
        
        # Pentagon
        ax = axes[0, 1]
        self._setup_axes(ax, "Pentagon in Wedge")
        angles = np.linspace(0, 2*np.pi, 6)[:-1] + np.pi/10
        pentagon = np.array([[3*np.cos(a)+1, 3*np.sin(a)+1] for a in angles], dtype=np.float32)
        self._draw_wedge(ax, alpha_min, alpha_max, radius, label='Wedge')
        self._draw_polygon(ax, pentagon, color='lightblue', alpha=0.3, label='Original')
        result = clip_polygon_to_wedge(pentagon, alpha_min, alpha_max, radius)
        if result is not None:
            self._draw_polygon(ax, result, color='green', alpha=0.5, edgecolor='darkgreen', label='Clipped')
        ax.legend(loc='upper right')
        
        # L-shaped polygon
        ax = axes[0, 2]
        self._setup_axes(ax, "L-shape in Wedge")
        l_shape = np.array([
            [0.5, 0.5],
            [3.0, 0.5],
            [3.0, 1.5],
            [1.5, 1.5],
            [1.5, 3.5],
            [0.5, 3.5],
        ], dtype=np.float32)
        self._draw_wedge(ax, alpha_min, alpha_max, radius, label='Wedge')
        self._draw_polygon(ax, l_shape, color='lightblue', alpha=0.3, label='Original')
        result = clip_polygon_to_wedge(l_shape, alpha_min, alpha_max, radius)
        if result is not None:
            self._draw_polygon(ax, result, color='orange', alpha=0.5, edgecolor='darkorange', label='Clipped')
        ax.legend(loc='upper right')
        
        # Cross-shaped polygon spanning wedge
        ax = axes[1, 0]
        self._setup_axes(ax, "Cross in Wedge")
        cross = np.array([
            [1.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [3.0, 1.0],
            [3.0, 2.0],
            [2.0, 2.0],
            [2.0, 3.0],
            [1.0, 3.0],
            [1.0, 2.0],
            [0.0, 2.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ], dtype=np.float32)
        self._draw_wedge(ax, alpha_min, alpha_max, radius, label='Wedge')
        self._draw_polygon(ax, cross, color='lightblue', alpha=0.3, label='Original')
        result = clip_polygon_to_wedge(cross, alpha_min, alpha_max, radius)
        if result is not None:
            self._draw_polygon(ax, result, color='purple', alpha=0.5, edgecolor='darkviolet', label='Clipped')
        ax.legend(loc='upper right')
        
        # Obstacle extending far beyond radius
        ax = axes[1, 1]
        self._setup_axes(ax, "Large Obstacle Clipped", xlim=(-2, 8), ylim=(-2, 8))
        large = np.array([
            [1.0, 0.5],
            [7.0, 0.5],
            [7.0, 7.0],
            [1.0, 7.0],
        ], dtype=np.float32)
        self._draw_wedge(ax, alpha_min, alpha_max, radius, label='Wedge')
        self._draw_polygon(ax, large, color='lightblue', alpha=0.3, label='Original')
        result = clip_polygon_to_wedge(large, alpha_min, alpha_max, radius)
        if result is not None:
            self._draw_polygon(ax, result, color='teal', alpha=0.5, edgecolor='darkcyan', label='Clipped')
        ax.legend(loc='upper right')
        
        # Multiple small obstacles
        ax = axes[1, 2]
        self._setup_axes(ax, "Narrow Wedge Clipping")
        narrow_min = np.radians(40)
        narrow_max = np.radians(50)
        polygon = np.array([
            [1.0, 0.5],
            [3.0, 0.5],
            [3.0, 2.5],
            [1.0, 2.5],
        ], dtype=np.float32)
        self._draw_wedge(ax, narrow_min, narrow_max, radius, label='Narrow wedge')
        self._draw_polygon(ax, polygon, color='lightblue', alpha=0.3, label='Original')
        result = clip_polygon_to_wedge(polygon, narrow_min, narrow_max, radius)
        if result is not None:
            self._draw_polygon(ax, result, color='coral', alpha=0.5, edgecolor='orangered', label='Clipped')
        else:
            ax.annotate('Result: None', xy=(2, 1.5), fontsize=10, ha='center', color='red')
        ax.legend(loc='upper right')
        
        fig.suptitle("Wedge Clipping: Complex Polygon Shapes", fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_figure(fig, "wedge_clip_complex_polygons")

    def test_visual_wedge_realistic_scenario(self):
        """Visual: Realistic obstacle detection scenario."""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        self._setup_axes(ax, "Realistic Scenario: Viewer with Obstacles", xlim=(-6, 6), ylim=(-6, 6))
        
        # Viewer FOV: looking roughly at +X direction with 90° FOV
        alpha_min = np.radians(-45)
        alpha_max = np.radians(45)
        radius = 5.0
        
        # Draw the wedge (viewer's field of view)
        self._draw_wedge(ax, alpha_min, alpha_max, radius, color='green', alpha=0.1, label='Field of View')
        self._draw_wedge_boundaries(ax, alpha_min, alpha_max, radius, color='green')
        
        # Various obstacles
        obstacles = [
            # Obstacle 1: Fully in FOV
            np.array([[2.0, -0.5], [3.0, -0.5], [3.0, 0.5], [2.0, 0.5]], dtype=np.float32),
            # Obstacle 2: Partially in FOV (extends beyond radius)
            np.array([[3.0, 1.0], [6.0, 1.0], [6.0, 2.0], [3.0, 2.0]], dtype=np.float32),
            # Obstacle 3: Partially in FOV (extends beyond angle)
            np.array([[1.0, 2.0], [2.0, 2.0], [2.0, 4.0], [1.0, 4.0]], dtype=np.float32),
            # Obstacle 4: Outside FOV
            np.array([[-3.0, 1.0], [-2.0, 1.0], [-2.0, 2.0], [-3.0, 2.0]], dtype=np.float32),
            # Obstacle 5: At edge of FOV
            np.array([[1.5, -2.0], [2.5, -2.0], [2.5, -1.0], [1.5, -1.0]], dtype=np.float32),
        ]
        
        colors = ['blue', 'orange', 'purple', 'gray', 'teal']
        
        for i, (obs, color) in enumerate(zip(obstacles, colors)):
            # Draw original
            self._draw_polygon(ax, obs, color=color, alpha=0.2, edgecolor=color)
            
            # Clip and draw result
            result = clip_polygon_to_wedge(obs, alpha_min, alpha_max, radius)
            if result is not None:
                self._draw_polygon(ax, result, color=color, alpha=0.6, 
                                  edgecolor='black', label=f'Obstacle {i+1} (visible)')
            else:
                # Mark center of original as not visible
                center = obs.mean(axis=0)
                ax.plot(center[0], center[1], 'x', color=color, markersize=10)
        
        # Draw viewer position and direction
        ax.annotate('Viewer', xy=(0, -0.5), fontsize=12, ha='center', fontweight='bold')
        ax.arrow(0, 0, 1.5, 0, head_width=0.2, head_length=0.1, fc='black', ec='black')
        
        ax.legend(loc='upper left', fontsize=9)
        
        plt.tight_layout()
        save_figure(fig, "wedge_realistic_scenario")
