# Implementation Plan: View Arc Obstacle Detection

## Overview
Implement the obstacle detection algorithm following a test-driven development approach, building from low-level geometry utilities up to the complete API.

## Phase 1: Core Geometry Foundation (Days 1-2)

### Step 1.1: Geometry Utilities - Basic Operations
**Implementation:**
- `normalize_angle()` - wrap angles to [-π, π)
- `to_viewer_frame()` - coordinate translation
- `to_polar()` - Cartesian to polar conversion
- `validate_and_get_direction_angle()` - validate unit vector and extract angle

**Tests to Create:**
- `tests/test_geometry.py`:
  - `test_normalize_angle_various_ranges()` - test wrapping from different ranges
  - `test_normalize_angle_boundary_cases()` - exactly at ±π
  - `test_to_viewer_frame_single_point()` - single coordinate translation
  - `test_to_viewer_frame_multiple_points()` - batch translation with broadcasting
  - `test_to_polar_basic()` - convert known Cartesian points
  - `test_to_polar_angles()` - verify angle ranges and quadrants
  - `test_validate_direction_normalized()` - accept valid unit vectors
  - `test_validate_direction_not_normalized()` - reject non-unit vectors
  - `test_validate_direction_angle_extraction()` - verify atan2 computation

**Validation:**
- All geometry tests pass
- Manual verification with known coordinates (e.g., [1,0] → 0°, [0,1] → 90°)

---

### Step 1.2: Geometry Utilities - Ray Intersection
**Implementation:**
- `intersect_ray_segment()` - ray-segment intersection with parametric solution
- `handle_angle_discontinuity()` - remap angles across ±π boundary

**Tests to Create:**
- `tests/test_geometry.py` (continued):
  - `test_intersect_ray_horizontal_segment()` - ray hits horizontal edge
  - `test_intersect_ray_vertical_segment()` - ray hits vertical edge
  - `test_intersect_ray_diagonal_segment()` - general case
  - `test_intersect_ray_no_intersection()` - ray misses segment
  - `test_intersect_ray_behind_origin()` - intersection at negative r
  - `test_intersect_ray_beyond_max_range()` - intersection too far
  - `test_intersect_ray_parallel()` - ray parallel to segment
  - `test_handle_angle_discontinuity_no_wrap()` - arc doesn't cross ±π
  - `test_handle_angle_discontinuity_with_wrap()` - arc crosses boundary
  - `test_handle_angle_discontinuity_remapping()` - verify 2π addition

**Validation:**
- Ray intersection matches analytical solutions
- Discontinuity handling creates continuous angle ranges

---

## Phase 2: Polygon Clipping (Days 3-4)

### Step 2.1: Half-Plane Clipping
**Implementation:**
- `is_valid_polygon()` - check vertex count
- `compute_bounding_box()` - AABB calculation
- `clip_polygon_halfplane()` - Sutherland-Hodgman for single half-plane

**Tests to Create:**
- `tests/test_clipping.py`:
  - `test_is_valid_polygon_sufficient_vertices()` - 3+ vertices
  - `test_is_valid_polygon_insufficient_vertices()` - <3 vertices
  - `test_compute_bounding_box_square()` - axis-aligned square
  - `test_compute_bounding_box_triangle()` - non-axis-aligned shape
  - `test_clip_halfplane_fully_inside()` - no clipping needed
  - `test_clip_halfplane_fully_outside()` - complete removal
  - `test_clip_halfplane_partial()` - some vertices clipped
  - `test_clip_halfplane_edge_intersection()` - verify intersection points
  - `test_clip_halfplane_ccw_preservation()` - maintain winding order

**Validation:**
- Visual inspection of clipped polygons using matplotlib
- Compare to known geometric constructions

---

### Step 2.2: Circle and Wedge Clipping
**Implementation:**
- `clip_polygon_circle()` - analytical circle clipping with quadratic solver
- `clip_polygon_to_wedge()` - complete 3-stage pipeline

**Tests to Create:**
- `tests/test_clipping.py` (continued):
  - `test_clip_circle_fully_inside()` - polygon within radius
  - `test_clip_circle_fully_outside()` - polygon beyond radius
  - `test_clip_circle_partial()` - some vertices beyond radius
  - `test_clip_circle_edge_intersections()` - verify intersection accuracy
  - `test_clip_circle_degenerate_to_point()` - handle edge case
  - `test_clip_wedge_full_pipeline()` - integration of 3 stages
  - `test_clip_wedge_narrow_arc()` - 30° FOV example
  - `test_clip_wedge_wide_arc()` - 120° FOV example
  - `test_clip_wedge_returns_none_if_outside()` - complete rejection
  - `test_clip_wedge_validates_result()` - ensures ≥3 vertices

**Validation:**
- Visualization overlay on sample images
- Verify clipped area is geometrically correct subset

---

## Phase 3: Angular Sweep (Days 5-6)

### Step 3.1: Event Construction
**Implementation:**
- `AngularEvent` dataclass (already defined)
- `get_active_edges()` - determine which edges span an angle
- `build_events()` - construct sorted event list from polygons

**Tests to Create:**
- `tests/test_sweep.py`:
  - `test_get_active_edges_no_edges_active()` - angle outside polygon
  - `test_get_active_edges_single_edge()` - angle intersects one edge
  - `test_get_active_edges_multiple_edges()` - non-convex case
  - `test_build_events_single_triangle()` - 3 vertex events
  - `test_build_events_multiple_obstacles()` - sorted mixed events
  - `test_build_events_edge_crossings()` - detect angular boundaries
  - `test_build_events_sorting_stability()` - vertex before edge at same angle
  - `test_build_events_empty_input()` - handle no polygons

**Validation:**
- Event list length matches expected vertices + crossings
- Events are properly sorted by angle

---

### Step 3.2: Interval Resolution
**Implementation:**
- `resolve_interval()` - multi-ray sampling and depth resolution
- Helper functions for ray sampling within interval

**Tests to Create:**
- `tests/test_sweep.py` (continued):
  - `test_resolve_interval_single_obstacle()` - trivial case
  - `test_resolve_interval_two_obstacles_one_closer()` - occlusion
  - `test_resolve_interval_overlapping_at_different_distances()` - depth test
  - `test_resolve_interval_sampling_density()` - verify 5 samples used
  - `test_resolve_interval_no_obstacles()` - empty active set
  - `test_resolve_interval_obstacle_at_boundary()` - edge case
  - `test_resolve_interval_narrow_interval()` - small angular span

**Validation:**
- Closest obstacle wins in occlusion scenarios
- Distance measurements are accurate

---

### Step 3.3: Coverage Computation
**Implementation:**
- `compute_coverage()` - complete sweep with interval processing
- Accumulation logic for angular coverage and min distances

**Tests to Create:**
- `tests/test_sweep.py` (continued):
  - `test_compute_coverage_single_obstacle_full_arc()` - occupies entire FOV
  - `test_compute_coverage_two_obstacles_side_by_side()` - no occlusion
  - `test_compute_coverage_two_obstacles_overlapping()` - occlusion test
  - `test_compute_coverage_gap_in_arc()` - some angles have no obstacle
  - `test_compute_coverage_min_distance_tracking()` - verify distance records
  - `test_compute_coverage_interval_boundaries()` - correct attribution at edges
  - `test_compute_coverage_empty_arc()` - no obstacles visible

**Validation:**
- Coverage sums match expected angular spans
- Min distances are correctly tracked per obstacle

---

## Phase 4: API Integration (Days 7-8)

### Step 4.1: Main Algorithm Implementation
**Implementation:**
- `find_largest_obstacle()` in `api.py`
- Integration of all modules: validation → transform → clipping → sweep → result

**Tests to Create:**
- `tests/test_api.py`:
  - `test_find_largest_obstacle_single_centered()` - one obstacle in center
  - `test_find_largest_obstacle_two_side_by_side()` - no occlusion
  - `test_find_largest_obstacle_one_occludes_other()` - depth ordering
  - `test_find_largest_obstacle_empty_scene()` - no obstacles
  - `test_find_largest_obstacle_all_outside_arc()` - all rejected by clipping
  - `test_find_largest_obstacle_narrow_fov()` - 30° field of view
  - `test_find_largest_obstacle_wide_fov()` - 120° field of view
  - `test_find_largest_obstacle_invalid_direction()` - non-unit vector
  - `test_find_largest_obstacle_invalid_contours()` - malformed input
  - `test_find_largest_obstacle_with_intervals()` - return_intervals=True
  - `test_find_largest_obstacle_tie_breaking()` - equal coverage, closest wins

**Validation:**
- Integration tests pass
- End-to-end scenarios produce expected winners

---

### Step 4.2: Real-World Scenario Tests
**Implementation:**
- No new implementation, only comprehensive testing

**Tests to Create:**
- `tests/test_api_integration.py`:
  - `test_scenario_person_looking_up()` - view_direction=[0, 1]
  - `test_scenario_person_looking_left()` - view_direction=[-1, 0]
  - `test_scenario_person_looking_diagonal()` - view_direction=[-0.37, 0.92]
  - `test_scenario_close_vs_far_obstacles()` - distance-based selection
  - `test_scenario_large_vs_narrow_obstacles()` - angular coverage comparison
  - `test_scenario_obstacle_at_arc_boundary()` - partial visibility
  - `test_scenario_max_range_limit()` - distant obstacles rejected
  - `test_scenario_complex_contours()` - polygons with many vertices

**Validation:**
- Realistic scenarios match intuitive expectations
- Performance is acceptable (<100ms per frame)

---

## Phase 5: Visualization & Debugging (Day 9)

### Step 5.1: Visualization Tools
**Implementation:**
- `draw_wedge_overlay()` - FOV arc rendering
- `draw_obstacle_contours()` - contour rendering with winner highlight
- `draw_angular_intervals()` - interval ray visualization

**Tests to Create:**
- `tests/test_visualize.py`:
  - `test_draw_wedge_overlay_basic()` - verify image modified
  - `test_draw_wedge_overlay_various_fovs()` - different angles
  - `test_draw_obstacle_contours_no_winner()` - default color for all
  - `test_draw_obstacle_contours_with_winner()` - highlight winner
  - `test_draw_angular_intervals_multiple()` - render several intervals
  - Manual visual tests (saved to `tests/output/` directory)

**Validation:**
- Visual inspection of generated images
- Overlays correctly positioned in image space

---

### Step 5.2: Debug Utilities
**Implementation:**
- Helper functions for logging intermediate results
- Optional detailed interval breakdown in ObstacleResult

**Tests to Create:**
- `tests/test_debug_scenarios.py`:
  - Create test cases that previously failed and verify fixes
  - Edge cases discovered during integration testing

---

## Phase 6: Performance & Polish (Day 10)

### Step 6.1: Performance Optimization
**Implementation:**
- Profile with `cProfile` on typical workloads
- Vectorize any remaining loops in clipping/sweep
- Add early-exit optimizations for candidate culling

**Tests to Create:**
- `tests/test_performance.py`:
  - `test_performance_many_obstacles()` - 50+ contours
  - `test_performance_complex_polygons()` - contours with 100+ vertices
  - `test_performance_wide_fov()` - 180° arc
  - Benchmark tests with timing assertions (<100ms target)

**Validation:**
- Profile results show no obvious bottlenecks
- Performance targets met for realistic scenarios

---

### Step 6.2: Documentation & Examples
**Implementation:**
- Complete docstrings for all functions
- Add type hints validation with mypy
- Create example notebooks

**Deliverables:**
- `examples/basic_usage.py` - simple API demonstration
- `examples/visualization_demo.py` - show debug overlays
- `examples/real_image_processing.py` - process actual images
- Update README.md with complete usage guide

---

## Phase 7: Final Validation (Day 11)

### Final Test Suite Run
- Execute full test suite with coverage report (target: >90%)
- Fix any remaining edge cases
- Verify all tests pass on Python 3.13

### Integration Testing
- Test with actual images from your camera system
- Verify coordinate convention matches transmitted data
- Validate results against manual annotations

### Code Quality
- Run `black` formatter
- Run `ruff` linter
- Run `mypy` type checker
- Address all issues

---

## Summary of Test Files to Create

1. **`tests/test_geometry.py`** (~20 tests) - coordinate transforms, polar conversion, ray intersection
2. **`tests/test_clipping.py`** (~20 tests) - polygon clipping against half-planes, circle, wedge
3. **`tests/test_sweep.py`** (~20 tests) - event construction, interval resolution, coverage computation
4. **`tests/test_api.py`** (~15 tests) - main API function with various inputs
5. **`tests/test_api_integration.py`** (~10 tests) - real-world scenarios
6. **`tests/test_visualize.py`** (~8 tests) - visualization functions
7. **`tests/test_performance.py`** (~5 tests) - performance benchmarks
8. **`tests/test_debug_scenarios.py`** (~5 tests) - edge cases and bug reproductions

**Total: ~100+ tests covering all functionality**

---

## Development Approach

- **Test-First:** Write tests before implementation for each step
- **Incremental:** Complete each phase before moving to next
- **Validation:** Visual and numerical validation at each step
- **Iteration:** Refactor as needed based on test results

## Success Criteria

✅ All tests pass with >90% coverage  
✅ Performance <100ms per frame  
✅ Handles all edge cases gracefully  
✅ Produces correct results on real images  
✅ Clean code passing all linters  
✅ Complete documentation
