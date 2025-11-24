# Obstacle Selection Within a View Arc

## Goal
Determine which obstacle contour has the largest **visible** angular coverage from a viewer point within a limited field-of-view arc and distance range. "Visible" means the portion of the obstacle that remains after clipping to the arc wedge and accounting for occlusion by nearer obstacles. "Largest" refers to the angular span (in radians or degrees) of the visible region, **not** the physical size or area of the original obstacle contour. Visibility is radial: nearer obstacles occlude farther ones along the same angular direction. The procedure must avoid code but describe an implementable, efficient algorithm.

## Coordinate and Input Model
- Image coordinates are 2D Cartesian with origin at pixel `(0,0)` unless specified otherwise.
- `P0 = (x0, y0)` is the viewer point.
- `P1 = (x1, y1)` defines the viewing direction; the unit vector is `v = normalize(P1 - P0)`.
- Field of view is characterized by a symmetric half-angle `θ_fov/2` centered on `v` and a maximum sensing radius `R_max`.
- Obstacles are provided as simple polygonal contours `Oi`, each a vertex list ordered consistently (clockwise or counter-clockwise) without self-intersections.
- Output is the identifier of the obstacle with maximal visible angular coverage inside the arc, plus its aggregated coverage value (radians or degrees) and representative distance metrics.

## Derived Geometry
1. Translate every point by subtracting `P0` to work in a viewer-centric reference frame (`P0` becomes origin).
2. Precompute `α_center = atan2(v_y, v_x)` and bounds `[α_min, α_max] = [α_center - θ_fov/2, α_center + θ_fov/2]`.
3. Define the circular-sector wedge `W` as the intersection of:
   - Two half-planes bounded by rays at `α_min` and `α_max` issuing from the origin.
   - A circle of radius `R_max`.

## Supporting Diagrams

```
             obstacle B (within arc)
                    /\
                   /  \
                  /    \
                 /      \
                /        \
 α_max ray ->  /          \
              /            \
             /              \
            * P0 viewer origin ----> v (toward P1)
             \              /
              \            /
               \          /
                \        /
 α_min ray ->     \      /
                   \    /
                    \  /
                     \/

θ_fov = 30° ⇒ 15° on each side of v

Legend: both α_min and α_max rays originate at `P0`, forming a narrow wedge (30° total) capped at radius `R_max`. Obstacle B lies entirely inside and is therefore a strong candidate for selection.
```

```
Angular sweep (top view)

 α_min ----|==== interval 1 ====|==== interval 2 ====|---- α_max
             ^ near obstacle A     ^ farther obstacle B

For each interval demarcated by vertex events, a ray test finds the nearest intersecting polygon. Interval 1 stays with obstacle A because it is closer; interval 2 belongs to obstacle B once A exits the active set. Coverage for each obstacle equals the sum of the intervals it controls.
```

## Algorithm Overview
1. **Candidate culling**
   - For each obstacle, check whether its bounding box intersects the circle of radius `R_max` and whether any of its vertices’ angles fall within `[α_min, α_max]`. Reject otherwise to avoid expensive processing.

2. **Sector clipping**
   - Clip each remaining polygon against wedge `W` in three stages:
     a. Clip against the two half-planes at `α_min` and `α_max` using Sutherland–Hodgman algorithm.
     b. Clip against the circular boundary at radius `R_max`. For each edge, solve the quadratic equation to find intersection points with the circle analytically (more accurate than polygon approximation). Edges entirely beyond `R_max` are discarded; edges crossing the circle are split at intersection points.
   - **Important**: After clipping, validate that the result is a proper polygon (≥3 vertices). Degenerate cases (line segments, single points) should be either discarded or handled by converting to minimal-area triangles for the sweep stage.
   - The resulting polygon represents the visible portion of the obstacle within the arc.

3. **Polar conversion**
   - Convert vertices of the clipped polygon into polar coordinates `(r, α)`. Normalize every angle into `[α_min, α_max]` and clamp radii to `[0, R_max]`.

4. **Angular event construction**
   - For each clipped polygon vertex at polar coordinates `(r, α)`, create an event at angle `α` containing `obstacle_id` and the vertex index.
   - Additionally, for each polygon edge, determine if it crosses radial lines within `[α_min, α_max]` by checking whether adjacent vertices span different angular regions. Mark these as edge-crossing events.
   - Accumulate all vertex and edge-crossing events into a list and sort by angle (ascending). Ties are broken by processing vertex events before edge events at the same angle to ensure correct topology.
   - **Critical**: This approach handles arbitrary polygon shapes including non-convex contours where a single obstacle may enter/exit the active set multiple times across the arc.

5. **Angular sweep with depth resolution**
   - Sweep from `α_min` to `α_max`, processing each event to establish contiguous angular intervals.
   - For each interval `[α_i, α_{i+1}]`, determine which obstacles have edges that intersect rays in this range. Rather than testing only the midpoint, sample at 3-5 rays per interval or test against all polygon edges analytically to avoid missing thin obstacles.
   - For each sampled ray direction `α_sample`, compute intersections with all potentially visible obstacles by solving line-segment intersection equations. Select the obstacle with minimum positive `r` at that ray.
   - Aggregate results across samples: the obstacle appearing closest most frequently (or with minimal average `r`) owns the interval. Add interval width to that obstacle's `coverage`; update `min_distance[id]`.
   - **Robustness**: Handle degenerate cases where no obstacle intersects (gap in coverage) and where obstacles touch exactly at interval boundaries (assign to closer obstacle).

6. **Winner selection**
   - After covering the whole arc, choose the obstacle with the largest `coverage`. Resolve ties by picking the obstacle with the smallest recorded `r` (closest winner requirement). If all `coverage` values are zero, report that no obstacle is visible.

## Data Structures
- **Event list:** array of size `~2M` where `M` is total number of vertices after clipping; sorting cost `O(M log M)`.
- **Active edge registry:** for each angular interval, maintain a list of obstacle edges (line segments) that geometrically span that interval. Update this as the sweep progresses by adding edges when their angular span begins and removing when it ends.
- **Obstacle ledger:** dictionary keyed by obstacle identifier storing cumulative coverage (in radians), minimum distance encountered, and optionally a list of owned intervals for debugging.

## Complexity Considerations
- Clipping and polar conversion: `O(total_vertices)`.
- Event sorting: `O(M log M)`.
- Angular sweep: `O(I log K)` where `I` is number of angular intervals (≤ 2M) and `K` is average active set size. For typical retail-scene camera feeds, this remains near-linear.

## Numerical Notes
- Use a small angular epsilon (`~1e-6` radians) when comparing against `α_min`/`α_max` to cope with floating-point jitter.
- **Angle discontinuity handling**: If the field of view crosses the ±π boundary (i.e., `α_max - α_min > 2π` after wrapping), remap all angles to a continuous range by adding 2π to angles below `α_min`. This ensures the sweep progresses monotonically without wrapping artifacts.
- Normalize angles consistently (wrap to `[-π, π)` after computing `arctan2`) before any interval comparisons.
- When intersecting rays with polygons, use robust segment–ray intersection with denominator checks to avoid division by near-zero values. Discard hits with `r ≤ 0` (behind origin) or `r > R_max`.
- For vertices exactly at `r = R_max`, treat as inclusive to avoid boundary oscillation.

## Optional Optimizations
1. Spatial indexing (quadtree/k-d tree) around `P0` to limit the obstacles entering the clipping stage when many contours exist.
2. Adaptive angular sampling: if exact sweep is too costly, subdivide the arc into fine discrete bins, intersect rays at bin centers, and approximate coverage as `bin_count × bin_width`.
3. Caching: if the same viewer point is queried repeatedly with minor orientation changes, store clipped silhouettes and reuse until a significant motion threshold is crossed.

## Python Implementation Plan (No Code Yet)

### Libraries and Modules
- **NumPy** for vectorized geometry (coordinate translation, dot/cross computations, polar transforms).
- **OpenCV (cv2)** to ingest obstacle contours (`findContours` outputs) and perform polygon clipping via `cv2.clipLine` for circle edges plus custom half-plane clipping.
- **scikit-image** utilities (`measure.regionprops`, `draw.polygon_perimeter`) for contour management if needed; also provides `skimage.draw.line` for ray rasterization when validating.
- Optional: **Shapely** for robust polygon clipping if third-party geometry is permitted; otherwise stick to OpenCV + NumPy.

### Proposed Package Layout
- `view_arc/__init__.py` exports the public API.
- `view_arc/geometry.py` handles normalization, polar conversion, and ray–segment intersections.
- `view_arc/clipping.py` contains wedge clipping routines built around NumPy arrays and OpenCV helpers.
- `view_arc/sweep.py` implements event construction, sorting, and angular sweep logic.
- `view_arc/visualize.py` (optional) overlays arcs and winning obstacles on top of the original frame using OpenCV drawing functions for debugging.
- `tests/` includes unit tests using `pytest` plus synthetic fixtures with hand-authored polygons.

### Processing Steps in Python Terms
1. **Input hydration**
   - Accept `P0`, `P1`, `theta_fov_deg`, `r_max`, and a list of NumPy arrays shaped `(n_i, 2)` representing obstacle contours.
   - Convert degrees to radians once; compute viewing unit vector via `cv2.normalize` or NumPy `np.linalg.norm`.

2. **Viewer-centric transform**
   - Subtract `P0` from all contour points using NumPy broadcasting; store per-obstacle bounding boxes (`np.min`, `np.max`).

3. **Candidate culling**
   - Estimate min distance via `np.linalg.norm` on bbox corners; reject obstacles whose bbox circle test exceeds `r_max` and whose angles (computed with `np.arctan2`) fall outside `[α_min, α_max]`. Use vectorized comparisons to keep it efficient.

4. **Wedge clipping implementation**
   - Clip against half-planes by iterating contour edges and inserting intersection points computed in `geometry.py` (Sutherland–Hodgman style using NumPy arrays).
   - Clip against the circle boundary: detect segments with endpoints beyond `r_max`; solve for intersection using quadratic form on parametric edge equations; rely on NumPy broadcasting to handle multiple edges quickly.

5. **Polar conversion & event building**
   - Transform clipped vertices into `(r, α)` arrays via `np.hypot` and `np.arctan2`.
   - For each vertex, create an event tuple `(α, obstacle_id, vertex_idx)`. For edges spanning different angular regions, add edge-crossing events.
   - Sort all events using `np.argsort` (or `sorted()` with stable ordering); ensure vertex events precede edge events at identical angles.

6. **Angular sweep data structures**
   - Maintain an **active edge registry**: a list or dict mapping each obstacle to its currently relevant edges (those whose angular span includes the current sweep position).
   - For each interval between consecutive events, sample at 3-5 ray angles (or use adaptive density). For each ray, intersect with all active edges using vectorized NumPy operations: represent edges as `(start, end)` arrays, compute parametric intersections, filter for `0 < r < R_max`.
   - Aggregate ray results: assign interval to the obstacle with minimal median or mean `r` across samples.

7. **Coverage accumulation**
   - Store coverage in a dictionary keyed by id or in a NumPy vector aligned with obstacle order. When an obstacle owns an interval of width `Δα`, add `Δα` and update `min_distance[id] = min(current_min, r_min)`.

8. **Result packaging**
   - After the sweep, identify the obstacle with `np.argmax(coverage)`; break ties by checking `min_distance`. Return a dataclass-like structure (e.g., `typing.NamedTuple`) for clarity.

### Testing & Validation Hooks
- Unit tests for `geometry.intersect_ray_segment`, `clipping.clip_to_wedge`, and `sweep.compute_coverage` using synthetic NumPy arrays.
- Integration tests reading sample frames, extracting contours via OpenCV, and comparing outputs to expected winners.
- Visualization helper that draws the wedge (`cv2.ellipse`), overlays angular intervals, and labels the selected obstacle for manual inspection.

### Performance Notes
- Use float32 arrays where possible to align with OpenCV defaults and reduce memory.
- Batch operations (especially clipping and ray solves) to avoid Python loops; rely on NumPy vectorization.
- Profile with `cProfile` or `line_profiler` focusing on the sweep stage; consider caching intermediate per-obstacle polar data when camera geometry is static across frames.

## Validation Checklist
- **Unit scenes:** single obstacle in center, obstacle touching arc boundary, multiple nested obstacles, disjoint obstacles partially within `R_max`.
- **Occlusion tests:** verify that a nearer, narrower obstacle masks a wider but farther obstacle along overlapping angular spans.
- **Visualization:** overlay computed coverage intervals on the original image to visually inspect correctness.
- **Performance logging:** track per-frame obstacle counts, clipped vertex totals, and sweeping time to confirm the algorithm satisfies real-time constraints.

## Deliverables
- Implementation must expose:
  - API accepting `P0`, `P1`, `θ_fov`, `R_max`, and obstacle contours.
  - Result structure containing winning obstacle ID, angular coverage value, minimal distance, and optionally the list of angular intervals it occupies.
- Unit and integration tests scripted according to the validation checklist.
