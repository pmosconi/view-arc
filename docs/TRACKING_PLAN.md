# Implementation Plan: Temporal Eyeball Tracking

## Overview
Extend the view arc obstacle detection system to accumulate "eyeballs" (viewing time in seconds) across multiple viewer positions and view directions over a batched acquisition period. This leverages the existing `find_largest_obstacle()` API to determine which area of interest (AOI) is being viewed at each timestamp.

## Context
- **Use Case**: Track viewer attention on store shelves/displays over time
- **Input**: Batch of viewer positions and view directions sampled at 1/second
- **Output**: Per-AOI hit counts representing seconds of viewing time
- **Processing**: Batch processing after acquisition (not real-time)

---

## Terminology
- **AOI (Area of Interest)**: Represents shelves or display areas in a store (previously called "obstacle")
- **Eyeball**: A single second of viewing time attributed to an AOI
- **Hit**: When an AOI is selected as the largest visible object at a given timestamp
- **Session**: A complete acquisition period (potentially minutes) with 1 sample/second

---

## Phase 1: Data Structures & Input Validation (Day 1)

### Step 1.1: Core Data Structures
**Implementation in `view_arc/tracking.py`:**
- `ViewerSample` dataclass - single observation (position, direction, timestamp)
- `AOI` dataclass - area of interest with ID and contour
- `AOIResult` dataclass - per-AOI result with hit count and observation details
- `TrackingResult` dataclass - complete session results

**Data Structures:**
```python
@dataclass
class ViewerSample:
    position: tuple[float, float]  # (x, y) in image coordinates
    direction: tuple[float, float]  # unit vector
    timestamp: float | None = None  # optional, for ordering validation

@dataclass
class AOI:
    id: str | int  # unique identifier
    contour: np.ndarray  # polygon vertices, shape (N, 2)

@dataclass
class AOIResult:
    aoi_id: str | int
    hit_count: int  # number of times selected as winner
    total_eyeball_seconds: float  # = hit_count × sample_interval
    hit_timestamps: list[int]  # indices of samples where this AOI won

@dataclass
class TrackingResult:
    aoi_results: dict[str | int, AOIResult]  # keyed by AOI ID
    total_samples: int
    samples_with_hits: int  # samples where any AOI was visible
    samples_no_winner: int  # samples where no AOI was in view
```

**Tests to Create:**
- `tests/test_tracking.py`:
  - `test_viewer_sample_creation()` - valid sample construction
  - `test_viewer_sample_invalid_direction()` - non-unit vector rejected
  - `test_aoi_creation()` - valid AOI with ID and contour
  - `test_aoi_invalid_contour()` - reject malformed contours
  - `test_tracking_result_accessors()` - verify result data access

**Validation:**
- All dataclass fields properly typed
- Validation logic prevents invalid inputs

---

### Step 1.2: Input Validation Functions
**Implementation in `view_arc/tracking.py`:**
- `validate_viewer_samples()` - check sample array integrity
- `validate_aois()` - check AOI list, ensure unique IDs
- `validate_tracking_params()` - validate FOV, max_range parameters

**Tests to Create:**
- `tests/test_tracking.py` (continued):
  - `test_validate_samples_empty()` - handle empty input gracefully
  - `test_validate_samples_single()` - single sample valid
  - `test_validate_samples_batch()` - typical batch (60+ samples)
  - `test_validate_samples_invalid_position()` - reject out-of-bounds
  - `test_validate_aois_empty()` - handle no AOIs
  - `test_validate_aois_duplicate_ids()` - reject duplicate IDs
  - `test_validate_aois_mixed_id_types()` - str and int IDs coexist

**Validation:**
- Clear error messages for invalid inputs
- Edge cases handled gracefully

---

## Phase 2: Core Tracking Algorithm (Days 2-3)

### Step 2.1: Single-Sample Processing Wrapper
**Implementation in `view_arc/tracking.py`:**
- `process_single_sample()` - wrapper around `find_largest_obstacle()` that:
  - Accepts a ViewerSample and list of AOIs
  - Returns the winning AOI ID (or None if no winner)
  - Optionally returns detailed result for debugging

**Tests to Create:**
- `tests/test_tracking.py` (continued):
  - `test_process_single_sample_one_aoi_visible()` - single AOI in view
  - `test_process_single_sample_multiple_aoi()` - returns winner
  - `test_process_single_sample_no_aoi_visible()` - returns None
  - `test_process_single_sample_all_aoi_outside_range()` - max_range filtering
  - `test_process_single_sample_preserves_aoi_id()` - ID correctly mapped

**Validation:**
- Wrapper correctly delegates to existing API
- AOI IDs properly tracked through the pipeline

---

### Step 2.2: Batch Processing Function
**Implementation in `view_arc/tracking.py`:**
- `compute_eyeballs()` - main entry point that:
  - Accepts batch of ViewerSamples and list of AOIs
  - Iterates through samples, calling `find_largest_obstacle()` for each
  - Accumulates hit counts per AOI
  - Returns TrackingResult with complete statistics

**Function Signature:**
```python
def compute_eyeballs(
    samples: list[ViewerSample] | np.ndarray,
    aois: list[AOI],
    fov_deg: float = 90.0,
    max_range: float = 500.0,
    sample_interval: float = 1.0,  # seconds between samples
) -> TrackingResult:
```

**Tests to Create:**
- `tests/test_tracking.py` (continued):
  - `test_compute_eyeballs_single_sample()` - trivial case
  - `test_compute_eyeballs_all_same_aoi()` - viewer stares at one AOI
  - `test_compute_eyeballs_alternating_aois()` - viewer looks left/right
  - `test_compute_eyeballs_no_hits()` - viewer never looks at AOIs
  - `test_compute_eyeballs_partial_hits()` - some samples hit, some miss
  - `test_compute_eyeballs_hit_count_accuracy()` - verify counts
  - `test_compute_eyeballs_all_aois_represented()` - all AOIs in result
  - `test_compute_eyeballs_timestamps_recorded()` - hit indices tracked

**Validation:**
- Total hits across AOIs ≤ total samples
- Hit counts sum correctly
- All AOI IDs present in result (even with 0 hits)

---

### Step 2.3: Convenience Input Formats
**Implementation in `view_arc/tracking.py`:**
- Support multiple input formats for ergonomic API:
  - List of ViewerSample objects
  - Tuple of (positions_array, directions_array)
  - Single numpy array of shape (N, 4) for [x, y, dx, dy]
- `normalize_sample_input()` - convert any format to internal representation

**Tests to Create:**
- `tests/test_tracking.py` (continued):
  - `test_input_format_sample_list()` - list of ViewerSample
  - `test_input_format_tuple_arrays()` - (positions, directions) tuple
  - `test_input_format_single_array()` - shape (N, 4) array
  - `test_input_format_mixed_types()` - numpy vs list positions

**Validation:**
- All formats produce identical results
- Clear errors for malformed inputs

---

## Phase 3: Result Analysis & Reporting (Day 4)

### Step 3.1: Result Aggregation Methods
**Implementation in `view_arc/tracking.py`:**
- `TrackingResult` methods:
  - `get_top_aois(n: int)` - return top N AOIs by hit count
  - `get_attention_distribution()` - percentage of time per AOI
  - `get_viewing_timeline()` - sequence of (timestamp, aoi_id) tuples
  - `to_dataframe()` - export to pandas DataFrame (optional dependency)

**Tests to Create:**
- `tests/test_tracking_results.py`:
  - `test_get_top_aois_basic()` - correct ordering
  - `test_get_top_aois_ties()` - handle equal hit counts
  - `test_get_top_aois_more_than_available()` - n > num AOIs
  - `test_attention_distribution_sums_to_100()` - percentages valid
  - `test_attention_distribution_excludes_no_hits()` - handles misses
  - `test_viewing_timeline_order()` - chronological sequence
  - `test_viewing_timeline_includes_none()` - gaps recorded
  - `test_to_dataframe_columns()` - correct structure

**Validation:**
- Aggregations are mathematically correct
- Edge cases (ties, zeros) handled gracefully

---

### Step 3.2: Session Statistics
**Implementation in `view_arc/tracking.py`:**
- `TrackingResult` computed properties:
  - `coverage_ratio` - fraction of samples with a hit
  - `dominant_aoi` - AOI with most hits (or None)
  - `engagement_score` - weighted score based on distribution
  - `session_duration` - total time covered

**Tests to Create:**
- `tests/test_tracking_results.py` (continued):
  - `test_coverage_ratio_full_coverage()` - every sample has hit
  - `test_coverage_ratio_no_coverage()` - no hits
  - `test_coverage_ratio_partial()` - typical case
  - `test_dominant_aoi_clear_winner()` - one AOI dominates
  - `test_dominant_aoi_tie()` - multiple equal winners
  - `test_session_duration_calculation()` - samples × interval

**Validation:**
- Statistics match manual calculations
- Properties are read-only/cached appropriately

---

## Phase 4: Integration with Existing API (Day 5)

### Step 4.1: API Module Extension
**Implementation in `view_arc/api.py`:**
- Add `compute_eyeballs()` as public API function (re-export from tracking)
- Ensure consistent parameter naming with `find_largest_obstacle()`
- Add re-export in `view_arc/__init__.py`

**Tests to Create:**
- `tests/test_api_tracking.py`:
  - `test_compute_eyeballs_api_accessible()` - import from view_arc
  - `test_compute_eyeballs_matches_manual_loop()` - same results as manual iteration
  - `test_compute_eyeballs_parameter_consistency()` - FOV, max_range work same as single-frame

**Validation:**
- Public API is clean and documented
- Parameters behave consistently with existing API

---

### Step 4.2: AOI ID Mapping
**Implementation in `view_arc/tracking.py`:**
- Internal mapping from contour index to AOI ID
- Ensure `find_largest_obstacle()` result maps back to correct AOI

**Tests to Create:**
- `tests/test_api_tracking.py` (continued):
  - `test_aoi_id_mapping_integer_ids()` - numeric IDs preserved
  - `test_aoi_id_mapping_string_ids()` - string IDs preserved
  - `test_aoi_id_mapping_mixed_ids()` - heterogeneous IDs
  - `test_aoi_id_stable_across_calls()` - consistent mapping

**Validation:**
- IDs never get confused or swapped
- Mapping is deterministic

---

## Phase 5: Visualization Extensions (Day 6)

### Step 5.1: Heatmap Visualization
**Implementation in `view_arc/visualize.py`:**
- `draw_attention_heatmap()` - color AOIs by hit count
  - Gradient from cold (low attention) to hot (high attention)
  - Optional: alpha blending over background image
- `draw_attention_labels()` - annotate AOIs with hit counts/percentages

**Tests to Create:**
- `tests/visual/test_tracking_visualize.py`:
  - `test_draw_attention_heatmap_basic()` - image modified
  - `test_draw_attention_heatmap_color_scale()` - colors vary with hits
  - `test_draw_attention_heatmap_zero_hits()` - handle AOIs with no hits
  - `test_draw_attention_labels_positioning()` - labels visible
  - Manual visual tests saved to `tests/visual/output/`

**Validation:**
- Visual inspection confirms correct coloring
- Heatmap accurately represents hit distribution

---

### Step 5.2: Timeline Visualization
**Implementation in `view_arc/visualize.py`:**
- `draw_viewing_timeline()` - horizontal timeline showing which AOI was viewed
  - Color-coded segments per AOI
  - Gaps shown for no-hit samples
- `create_tracking_animation()` - optional animated GIF/video of session

**Tests to Create:**
- `tests/visual/test_tracking_visualize.py` (continued):
  - `test_draw_viewing_timeline_basic()` - timeline rendered
  - `test_draw_viewing_timeline_gaps()` - gaps visible
  - `test_draw_viewing_timeline_legend()` - AOI colors labeled

**Validation:**
- Timeline correctly represents viewing sequence
- Gaps are clearly distinguishable

---

### Step 5.3: Session Replay Visualization
**Implementation in `view_arc/visualize.py`:**
- `draw_session_frame()` - single frame of a session replay showing:
  - Current viewer position
  - Current view arc
  - Current winner highlighted
  - Running hit counts
- `generate_session_replay()` - produce sequence of frames for video export

**Tests to Create:**
- `tests/visual/test_tracking_visualize.py` (continued):
  - `test_draw_session_frame_components()` - all elements present
  - `test_generate_session_replay_frame_count()` - correct number of frames

**Validation:**
- Replay frames are self-consistent
- Viewer position matches sample data

---

## Phase 6: Performance Considerations (Day 7)

### Step 6.1: Batch Optimization Opportunities
**Analysis and minimal optimization:**
- Profile `compute_eyeballs()` on large sessions (300+ samples)
- Identify if any pre-computation helps (e.g., pre-clip AOIs to max_range circle)
- Consider caching AOI bounding boxes (already computed per call)

**Potential Optimizations (implement only if needed):**
- Pre-filter AOIs unlikely to be visible from any sample position
- Vectorize sample iteration where possible
- Early exit for samples clearly outside all AOI regions

### Step 6.2: Result Caching for Similar Samples (Optional - Investigate)
**Concept:**
When consecutive samples have nearly identical viewer position and view direction, the `find_largest_obstacle()` result will be the same. We could potentially:
- Cache the winning AOI ID along with the (position, direction) that produced it
- For new samples, check if they are "close enough" to a cached result to reuse it
- Skip the full clipping/sweep computation when a cache hit occurs

**Similarity Criteria (tentative):**
- Position distance < threshold (e.g., 5 pixels)
- Direction angle difference < threshold (e.g., 2°)
- Same FOV and max_range parameters

**Trade-offs to Evaluate:**
| Pros | Cons |
|------|------|
| Could significantly reduce computation for stationary viewers | Added complexity in cache management |
| Natural fit for "staring" behavior (common in stores) | Cache invalidation logic needed |
| Memory overhead is minimal (store position, direction, winner ID) | Threshold tuning required |
| | May introduce subtle inaccuracies at edge cases |

**Recommendation:**
Defer this optimization until after the basic implementation is complete. Profile first to determine if the simple loop is already fast enough (<1s target). If profiling shows `find_largest_obstacle()` is the bottleneck and many consecutive samples are similar, then implement caching.

**If Implemented - Tests to Create:**
- `test_cache_hit_identical_samples()` - exact same position/direction reuses result
- `test_cache_hit_near_samples()` - similar position/direction reuses result
- `test_cache_miss_different_position()` - position change invalidates cache
- `test_cache_miss_different_direction()` - direction change invalidates cache
- `test_cache_accuracy_vs_full_computation()` - verify cached results match full computation within tolerance

**Tests to Create:**
- `tests/test_tracking_performance.py`:
  - `test_performance_long_session()` - 300 samples (5 min session)
  - `test_performance_many_aois()` - 50+ areas of interest
  - `test_performance_complex_aoi_contours()` - AOIs with many vertices
  - Benchmark: target <1s for 300 samples × 20 AOIs

**Validation:**
- Performance acceptable for expected use cases
- No regression in accuracy from optimizations

---

### Step 6.3: Memory Efficiency
**Implementation:**
- Ensure intermediate results are not retained unnecessarily
- Use generators where appropriate for large datasets
- Optional: streaming mode for very long sessions

**Tests to Create:**
- `tests/test_tracking_performance.py` (continued):
  - `test_memory_usage_long_session()` - memory doesn't grow unbounded
  - `test_streaming_mode_consistency()` - same results as batch

**Validation:**
- Memory usage is bounded
- Large sessions don't cause OOM

---

## Phase 7: Integration Testing & Examples (Day 8)

### Step 7.1: Realistic Scenario Tests
**Tests to Create:**
- `tests/test_tracking_integration.py`:
  - `test_scenario_stationary_viewer()` - viewer doesn't move, rotates head
  - `test_scenario_walking_viewer()` - viewer moves through store
  - `test_scenario_browsing_behavior()` - viewer stops at shelves
  - `test_scenario_quick_glances()` - rapid direction changes
  - `test_scenario_long_stare()` - extended viewing of one AOI
  - `test_scenario_peripheral_viewing()` - AOIs at edge of FOV
  - `test_scenario_complete_store_walkthrough()` - end-to-end simulation

**Validation:**
- Results match intuitive expectations
- Edge cases from real usage are covered

---

### Step 7.2: Example Scripts
**Implementation:**
- `examples/eyeball_tracking_basic.py` - minimal example
- `examples/eyeball_tracking_visualization.py` - with heatmap output
- `examples/eyeball_tracking_analysis.py` - with result analysis
- `examples/simulated_store_session.py` - generate and analyze synthetic data

**Content for each example:**
1. **Basic**: Load AOIs, simulate viewer samples, compute eyeballs, print results
2. **Visualization**: Add heatmap overlay, save annotated image
3. **Analysis**: Export to DataFrame, compute statistics, identify top AOIs
4. **Simulation**: Generate realistic viewer trajectory, analyze attention patterns

**Validation:**
- Examples run without errors
- Output is informative and correct

---

## Phase 8: Documentation & Polish (Day 9)

### Step 8.1: API Documentation
**Implementation:**
- Complete docstrings for all new functions and classes
- Type hints for all parameters and returns
- Add to README.md:
  - New feature description
  - Usage examples
  - API reference

### Step 8.2: Type Checking & Linting
**Implementation:**
- Run mypy on new code
- Fix all type errors
- Run ruff/black for code formatting

**Validation:**
- `mypy view_arc/tracking.py` passes
- All linters pass

---

## Summary of New Test Files

| Test File | Test Count | Description |
|-----------|------------|-------------|
| `tests/test_tracking.py` | ~25 | Data structures, validation, core algorithm |
| `tests/test_tracking_results.py` | ~15 | Result aggregation and statistics |
| `tests/test_api_tracking.py` | ~10 | API integration and ID mapping |
| `tests/visual/test_tracking_visualize.py` | ~10 | Visualization functions |
| `tests/test_tracking_performance.py` | ~6 | Performance benchmarks |
| `tests/test_tracking_integration.py` | ~8 | Realistic scenarios |

**Total: ~75 new tests**

---

## New Files to Create

```
view_arc/
    tracking.py              # Core tracking logic and data structures
    
tests/
    test_tracking.py         # Core tracking tests
    test_tracking_results.py # Result analysis tests
    test_api_tracking.py     # API integration tests
    test_tracking_performance.py  # Performance benchmarks
    test_tracking_integration.py  # Integration scenarios
    visual/
        test_tracking_visualize.py  # Visualization tests
        
examples/
    eyeball_tracking_basic.py          # Minimal usage example
    eyeball_tracking_visualization.py  # Heatmap visualization
    eyeball_tracking_analysis.py       # Result analysis
    simulated_store_session.py         # Synthetic data simulation
```

---

## Files to Modify

```
view_arc/
    __init__.py      # Export new functions: compute_eyeballs, AOI, TrackingResult
    api.py           # Add compute_eyeballs() re-export (optional)
    visualize.py     # Add heatmap, timeline, replay functions
    
README.md            # Add tracking feature documentation
```

---

## API Summary

### Primary Function
```python
from view_arc import compute_eyeballs, AOI, ViewerSample

# Define areas of interest
aois = [
    AOI(id="shelf_1", contour=np.array([[100, 100], [200, 100], [200, 200], [100, 200]])),
    AOI(id="shelf_2", contour=np.array([[300, 100], [400, 100], [400, 200], [300, 200]])),
]

# Define viewer samples (position, direction pairs sampled at 1/sec)
samples = [
    ViewerSample(position=(150, 300), direction=(0.0, -1.0)),  # looking up at shelf_1
    ViewerSample(position=(150, 300), direction=(0.0, -1.0)),  # still looking
    ViewerSample(position=(350, 300), direction=(0.0, -1.0)),  # moved, looking at shelf_2
    # ... more samples
]

# Compute eyeballs
result = compute_eyeballs(
    samples=samples,
    aois=aois,
    fov_deg=90.0,
    max_range=500.0,
    sample_interval=1.0,
)

# Access results
print(result.aoi_results["shelf_1"].hit_count)  # e.g., 2
print(result.aoi_results["shelf_2"].hit_count)  # e.g., 1
print(result.get_top_aois(5))  # Top 5 AOIs by attention
print(result.coverage_ratio)  # e.g., 1.0 (100% of time looking at some AOI)
```

### Alternative Input Formats
```python
# Using numpy arrays directly
positions = np.array([[150, 300], [150, 300], [350, 300]])
directions = np.array([[0.0, -1.0], [0.0, -1.0], [0.0, -1.0]])

result = compute_eyeballs(
    samples=(positions, directions),  # tuple of arrays
    aois=aois,
)

# Using single array of shape (N, 4)
data = np.array([
    [150, 300, 0.0, -1.0],
    [150, 300, 0.0, -1.0],
    [350, 300, 0.0, -1.0],
])

result = compute_eyeballs(samples=data, aois=aois)
```

### Visualization
```python
from view_arc import draw_attention_heatmap

annotated_image = draw_attention_heatmap(
    image=background_image,
    aois=aois,
    result=result,
    colormap="hot",
)
```

---

## Success Criteria

- [ ] `compute_eyeballs()` correctly accumulates hits per AOI
- [ ] All AOI IDs correctly mapped through the pipeline
- [ ] Results match manual iteration over `find_largest_obstacle()`
- [ ] Performance <1s for typical sessions (300 samples, 20 AOIs)
- [ ] Heatmap visualization accurately represents attention distribution
- [ ] All tests pass with >90% coverage on new code
- [ ] Type hints pass mypy validation
- [ ] Documentation complete with examples

---

## Dependencies

No new external dependencies required. Uses existing:
- `numpy` - array operations
- `opencv-python` - visualization (already used)
- `matplotlib` - optional visualization (already used)

Optional for result export:
- `pandas` - DataFrame export (soft dependency, graceful fallback)

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Performance bottleneck in batch loop | Profile early, optimize only if needed |
| AOI ID confusion with contour indices | Explicit mapping, comprehensive tests |
| Memory growth on long sessions | Stream results, don't store intermediate states |
| API inconsistency with existing functions | Reuse parameter names, validate identically |
