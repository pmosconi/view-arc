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

from view_arc.clipping import (
    is_valid_polygon,
    compute_bounding_box,
    clip_polygon_halfplane,
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
        bbox_rect = mpatches.Rectangle(min_pt, max_pt[0]-min_pt[0], max_pt[1]-min_pt[1],
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
        bbox_rect = mpatches.Rectangle(min_pt, max_pt[0]-min_pt[0], max_pt[1]-min_pt[1],
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
        bbox_rect = mpatches.Rectangle(min_pt, max_pt[0]-min_pt[0], max_pt[1]-min_pt[1],
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
