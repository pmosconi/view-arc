# Implementation Plan: Temporal Attention Tracking

## Overview
Extend the view arc obstacle detection system to accumulate "attention seconds" across multiple viewer positions and view directions over a batched acquisition period. This leverages the existing `find_largest_obstacle()` API to determine which area of interest (AOI) is being viewed at each timestamp.

## Context
- **Use Case**: Track viewer attention on store shelves/displays over time
- **Input**: Batch of viewer positions and view directions sampled at 1/second
- **Output**: Per-AOI hit counts representing seconds of viewing time
- **Processing**: Batch processing after acquisition (not real-time)

---

## Terminology
- **AOI (Area of Interest)**: Represents shelves or display areas in a store (previously called "obstacle")
- **Attention Second**: A single second of viewing time attributed to an AOI (legacy references may still call this an "eyeball")
- **Hit**: When an AOI is selected as the largest visible object at a given timestamp
- **Session**: A complete acquisition period (potentially minutes) with 1 sample/second


### Confirmed Constraints & Scope
- Samples arrive strictly at 1 Hz; each hit counts for exactly one second and no interpolation is performed.
- Sample timestamps, when provided, are already sorted upstream; we consume them as-is without enforcing ordering because doing so would not change the outcome under fixed-cadence sampling.
- AOI contours stay fixed in the same coordinate space as the viewer samples; no runtime transforms required.
- Each batch tracks a single viewer; multi-viewer aggregation happens outside this API.

---

## Phase 1: Data Structures & Input Validation (Day 1)

> **Implementation Note**: Before implementing new validation or utility functions, 
> check for existing helpers in `view_arc.obstacle.geometry`, `view_arc.obstacle.clipping`, and 
> `view_arc.obstacle.visualize`. Reuse these where possible to avoid duplicated logic:
> - `validate_and_get_direction_angle()` in `obstacle/geometry.py` - validates unit vectors
> - `is_valid_polygon()` in `obstacle/clipping.py` - validates polygon vertex counts
> - Coordinate transforms and polar conversions in `obstacle/geometry.py`

### Package Structure

The `view_arc` package is organized with tracking as the primary API:

```
view_arc/
    __init__.py              # Main public API (compute_attention_seconds, etc.)
    obstacle/                # Low-level obstacle detection
        __init__.py          # Re-exports obstacle detection API
        api.py               # find_largest_obstacle, ObstacleResult
        geometry.py          # Coordinate transforms, polar conversion
        clipping.py          # Polygon clipping operations
        sweep.py             # Angular sweep algorithm
        debug.py             # Debug utilities
        visualize.py         # OpenCV visualization
    tracking/                # Temporal attention tracking
        __init__.py          # Re-exports tracking API
        dataclasses.py       # ViewerSample, AOI, TrackingResult
        validation.py        # Input validation
        algorithm.py         # compute_attention_seconds, process_single_sample
```

### Module Structure (tracking subpackage)

The tracking functionality is organized into a subpackage `view_arc/tracking/`:
- `view_arc/tracking/dataclasses.py` - Core data structures (ViewerSample, AOI, SessionConfig, etc.)
- `view_arc/tracking/validation.py` - Input validation functions
- `view_arc/tracking/algorithm.py` - Core tracking algorithm (process_single_sample, compute_attention_seconds)
- `view_arc/tracking/__init__.py` - Re-exports all public symbols

### Step 1.1: Core Data Structures
**Implementation in `view_arc/tracking/dataclasses.py`:**
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
  total_attention_seconds: float  # = hit_count × sample_interval
    hit_timestamps: list[int]  # indices of samples where this AOI won

@dataclass
class TrackingResult:
    aoi_results: dict[str | int, AOIResult]  # keyed by AOI ID
    total_samples: int
    samples_with_hits: int  # samples where any AOI was visible
    samples_no_winner: int  # samples where no AOI was in view
```

**Tests to Create:**
- `tests/test_tracking_dataclasses.py`:
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
**Implementation in `view_arc/tracking/validation.py`:**
- `validate_viewer_samples()` - check sample array integrity
- `validate_aois()` - check AOI list, ensure unique IDs
- `validate_tracking_params()` - validate FOV, max_range parameters
- `normalize_sample_input()` - convert any input format to ViewerSample list

**Tests to Create:**
- `tests/test_tracking_validation.py`:
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

### Step 1.3: Session Configuration Schema 
**Implementation in `view_arc/tracking/dataclasses.py`:**
- `SessionConfig` dataclass gathers immutable acquisition metadata:
  - `session_id: str`
  - `frame_size: tuple[int, int] | None` (image width/height for bounds checks)
  - `coordinate_space: Literal["image"] = "image"` (documented constant)
  - `sample_interval_seconds: float = 1.0` (record the upstream cadence without re-validating it)
  - `viewer_id: str | None` (in case the batch needs cross-referencing upstream)
  - `notes: dict[str, Any] | None` for downstream analytics
**Tests to Create (in `tests/test_tracking_session_config.py`):**
- `test_session_config_defaults_applied()`
- `test_session_config_allows_custom_viewer_metadata()`
- `test_validate_samples_respects_frame_size()` (moved from Step 1.2 rationale).

**Validation:**
- Every tracking run emits a `SessionConfig` that downstream consumers embed in reports/logs.
- Viewer sample validation gains explicit knowledge of coordinate bounds when available.
- Coordinate space is explicitly documented as invariant (image pixels) throughout each batch.

---

## Phase 2: Core Tracking Algorithm (Days 2-3)

### Step 2.1: Single-Sample Processing Wrapper
**Implementation in `view_arc/tracking/algorithm.py`:**
- `process_single_sample()` - wrapper around `find_largest_obstacle()` that:
  - Accepts a ViewerSample and list of AOIs
  - Returns the winning AOI ID (or None if no winner)
  - Optionally returns detailed result for debugging
- `SingleSampleResult` dataclass - detailed result for debugging
- `AOIIntervalBreakdown` dataclass - interval details with AOI IDs

**Tests to Create:**
- `tests/test_tracking_process_single_sample.py`:
  - `test_process_single_sample_one_aoi_visible()` - single AOI in view
  - `test_process_single_sample_multiple_aoi()` - returns winner
  - `test_process_single_sample_no_aoi_visible()` - returns None
  - `test_process_single_sample_all_aoi_outside_range()` - max_range filtering
  - `test_process_single_sample_preserves_aoi_id()` - ID correctly mapped
  - `test_process_single_sample_return_details()` - detailed result structure
  - `test_process_single_sample_validation()` - input validation

**Validation:**
- Wrapper correctly delegates to existing API
- AOI IDs properly tracked through the pipeline

---

### Step 2.2: Batch Processing Function
**Implementation in `view_arc/tracking/algorithm.py`:**
- `compute_attention_seconds()` - main entry point that:
  - Accepts batch of ViewerSamples and list of AOIs
  - Iterates through samples, calling `find_largest_obstacle()` for each
  - Accumulates hit counts per AOI
  - Assumes each processed sample represents exactly 1 second of attention
  - Returns `TrackingResultWithConfig` with complete statistics and embedded `SessionConfig`
- `TrackingResultWithConfig` dataclass - extends TrackingResult with session config

**Function Signature:**
```python
def compute_attention_seconds(
    samples: list[ViewerSample] | np.ndarray,
    aois: list[AOI],
    field_of_view_deg: float = 90.0,
    max_range: float = 500.0,
    sample_interval: float = 1.0,
    session_config: SessionConfig | None = None,
) -> TrackingResultWithConfig:
```

**Tests to Create:**
- `tests/test_tracking_compute_attention.py`:
  - `test_compute_attention_single_sample()` - trivial case
  - `test_compute_attention_all_same_aoi()` - viewer stares at one AOI
  - `test_compute_attention_alternating_aois()` - viewer looks left/right
  - `test_compute_attention_no_hits()` - viewer never looks at AOIs
  - `test_compute_attention_partial_hits()` - some samples hit, some miss
  - `test_compute_attention_hit_count_accuracy()` - verify counts
  - `test_compute_attention_all_aois_represented()` - all AOIs in result
  - `test_compute_attention_timestamps_recorded()` - hit indices tracked
  - `test_compute_attention_session_config()` - session config embedding
  - `test_compute_attention_numpy_input()` - numpy array input format
  - `test_compute_attention_frame_size_validation()` - bounds checking

**Validation:**
- Total hits across AOIs ≤ total samples
- Hit counts sum correctly
- All AOI IDs present in result (even with 0 hits)
- Total attention seconds equals `hit_count × 1s`

---

### Step 2.3: Convenience Input Formats
**Implementation in `view_arc/tracking/validation.py`:**
- Support multiple input formats for ergonomic API:
  - List of ViewerSample objects
  - Single numpy array of shape (N, 4) for [x, y, dx, dy]
- `normalize_sample_input()` - convert any format to internal representation

**Tests to Create:**
- `tests/test_tracking_compute_attention.py` (TestComputeAttentionNumpyInput class):
  - `test_numpy_array_shape_n4()` - shape (N, 4) array accepted
  - `test_numpy_array_direction_normalized()` - directions auto-normalized
  - `test_numpy_array_matches_list_input()` - equivalent results

**Validation:**
- All formats produce identical results
- Clear errors for malformed inputs

---

### Step 2.4: Sampling Assumptions Documentation (**UPDATED**)
**Implementation in `view_arc/tracking/__init__.py` + docs:**
- Clearly document (module docstrings, README snippet, and `SessionConfig`) that upstream ingestion guarantees monotonic timestamps and a strict 1 Hz cadence. Our tracking loop therefore treats every accepted sample as 1 second without extra checks.
- Ensure `TrackingResult` exposes `assumptions: tuple[str, ...]` or similar metadata so downstream analytics always know the data quality contract without per-sample flags.

**Tests to Create:**
- Documentation lint/check to ensure the invariant text stays in README (simple `pytest` doc test or `pytest --doctest-glob` entry).

**Validation:**
- Sampling/time-ordering invariants remain clearly communicated though not re-validated, which keeps the implementation lightweight while preventing misuse.
- Consumers inspecting a `TrackingResult` can see which assumptions were applied without scanning logs.

---

## Phase 3: Result Analysis & Reporting (Day 4)

### Step 3.1: Result Aggregation Methods ✅ COMPLETED
**Implementation in `view_arc/tracking/dataclasses.py`:**
- `TrackingResult` methods:
  - `get_top_aois(n: int)` - return top N AOIs by hit count
  - `get_attention_distribution()` - percentage of time per AOI
  - `get_viewing_timeline()` - sequence of (timestamp, aoi_id) tuples
  - `to_dataframe()` - export to pandas DataFrame (optional dependency)

**Enhanced Validation (Security & Data Integrity):**
- `TrackingResult.__post_init__()` validates tally consistency:
  - `sum(r.hit_count) == samples_with_hits` - prevents count drift
  - `sum(len(r.hit_timestamps)) == samples_with_hits` - ensures timestamp completeness
  - `len(r.hit_timestamps) == r.hit_count` per AOI - validates per-AOI consistency
- `get_viewing_timeline()` validates hit_timestamps indices:
  - Rejects negative indices (prevents Python's negative indexing bugs)
  - Rejects indices >= total_samples (prevents IndexError)
  - Rejects non-integer indices
  - Clear ValidationError messages identify which AOI has invalid data

**Tests Created:**
- `tests/test_tracking_results.py` (30 tests):
  - `test_get_top_aois_*()` - correct ordering, ties, edge cases
  - `test_attention_distribution_*()` - percentages sum to 100, zero handling
  - `test_viewing_timeline_*()` - chronological sequence, gaps, invalid indices
  - `test_to_dataframe_*()` - correct structure, pandas optional dependency
  - `test_tracking_result_rejects_*()` - tally validation (HIGH priority fixes)
  - `test_viewing_timeline_rejects_*()` - index validation (MEDIUM priority fixes)

**Validation:**
- ✅ All aggregations are mathematically correct
- ✅ Edge cases (ties, zeros, empty results) handled gracefully
- ✅ Data corruption detected immediately at construction time
- ✅ Invalid hit_timestamps raise clear ValidationError messages
- ✅ All existing tests updated to provide valid hit_timestamps
- ✅ 294 tracking tests pass, 97% coverage on dataclasses.py

---

### Step 3.2: Session Statistics
**Implementation in `view_arc/tracking.py`:**
- `TrackingResult` computed properties:
  - `coverage_ratio` - fraction of samples with a hit
  - `dominant_aoi` - AOI with most hits (or None)
  - `engagement_score` - weighted score based on distribution
  - `session_duration` - total time covered (= total samples × 1 s)

**Tests to Create:**
- `tests/test_tracking_results.py` (continued):
  - `test_coverage_ratio_full_coverage()` - every sample has hit
  - `test_coverage_ratio_no_coverage()` - no hits
  - `test_coverage_ratio_partial()` - typical case
  - `test_dominant_aoi_clear_winner()` - one AOI dominates
  - `test_dominant_aoi_tie()` - multiple equal winners
  - `test_session_duration_calculation()` - total samples × 1 s

**Validation:**
- Statistics match manual calculations
- Properties are read-only/cached appropriately

---

## Phase 4: Integration with Existing API (Day 5)

### Step 4.1: Public API Exposure

- Add `compute_attention_seconds()` as public API function (re-export from tracking)
- Ensure consistent parameter naming with `find_largest_obstacle()`
- Add re-export in `view_arc/__init__.py`

**Tests to Create:**
- `tests/test_api_tracking.py`:
  - `test_compute_attention_api_accessible()` - import from view_arc
  - `test_compute_attention_matches_manual_loop()` - same results as manual iteration
  - `test_compute_attention_parameter_consistency()` - FOV, max_range work same as single-frame

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

### Step 6.1: Instrumentation & Regression Guardrails 
**Implementation:**
- Add lightweight profiling hooks (timing + sample counters) inside `compute_attention_seconds()` gated by a debug flag.
- Extend `profile_workload.py` to compare new tracking runs against a golden baseline (Runtime + accuracy on canned fixture) and emit alerts when drift exceeds thresholds.
- Capture peak memory and cache-hit ratios during performance tests and persist them under `examples/output/profile_runs.csv` for trend tracking.

**Tests/Automation:**
- `tests/test_tracking_performance.py::test_profile_hook_smoke()` ensures the instrumentation flag does not alter results.
- CI workflow step to run `python profile_workload.py --scenario tracking_baseline` weekly (documented in README).

**Validation:**
- Performance regressions are caught early, and instrumentation can be toggled without code changes.

---

### Step 6.2: Batch Optimization Opportunities
**Analysis and minimal optimization:**
- Profile `compute_attention_seconds()` on large sessions (300+ samples)
- Identify if any pre-computation helps (e.g., pre-clip AOIs to max_range circle)
- Consider caching AOI bounding boxes (already computed per call)

**Potential Optimizations (implement only if needed):**
- Pre-filter AOIs unlikely to be visible from any sample position
- Vectorize sample iteration where possible
- Early exit for samples clearly outside all AOI regions

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

### Step 6.3: Result Caching for Similar Samples (Optional - Investigate)
**Concept:**
When consecutive samples have nearly identical viewer position and view direction, the `find_largest_obstacle()` result will be the same. A per-batch, in-memory cache can:
- Store the winning AOI ID along with the (position, direction) that produced it
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

- **Recommendation:**
- Keep the cache lifetime scoped strictly to a single `compute_attention_seconds()` call; do not persist across sessions yet.
- Document a future enhancement idea for persistent caching (e.g., hashed viewpoints) but defer implementation until after baseline performance goals are met.

**If Implemented - Tests to Create:**
- `test_cache_hit_identical_samples()` - exact same position/direction reuses result
- `test_cache_hit_near_samples()` - similar position/direction reuses result
- `test_cache_miss_different_position()` - position change invalidates cache
- `test_cache_miss_different_direction()` - direction change invalidates cache
- `test_cache_accuracy_vs_full_computation()` - verify cached results match full computation within tolerance

**Validation:**
- Performance benefits are realized without risking stale data between sessions.

---

### Step 6.4: Memory Efficiency
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
- `examples/attention_tracking_basic.py` - minimal example
- `examples/attention_tracking_visualization.py` - with heatmap output
- `examples/attention_tracking_analysis.py` - with result analysis
- `examples/simulated_store_session.py` - generate and analyze synthetic data

**Content for each example:**
1. **Basic**: Load AOIs, simulate viewer samples, compute attention seconds, print results
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

### Step 8.3: Operational Notes & Dev Ergonomics 
**Implementation:**
- Document the `uv`-based virtual environment workflow (create, sync, run mypy) directly in README and `docs/IMPLEMENTATION_PLAN.md` so new contributors follow the same tooling.
- Provide a `make tracking-check` (or `uv run`) recipe that chains: mypy → targeted pytest suites → profile smoke test.
- Capture any required environment variables (e.g., data paths) in `.env.example` referenced by the docs.

**Validation:**
- Onboarding a new engineer only requires running the documented `uv` commands to reproduce tracking results and type checks.

---

## Summary of New Test Files

| Test File | Test Count | Description |
|-----------|------------|-------------|
| `tests/test_tracking_dataclasses.py` | ~50 | Data structures (ViewerSample, AOI, AOIResult, TrackingResult) - Step 1.1 |
| `tests/test_tracking_validation.py` | ~60 | Validation functions (validate_viewer_samples, validate_aois, validate_tracking_params) - Step 1.2 |
| `tests/test_tracking_session_config.py` | ~45 | SessionConfig validation and integration - Step 1.3 |
| `tests/test_tracking_algorithm.py` | ~25 | Core algorithm (process_single_sample, compute_attention_seconds) - Steps 2.1-2.4 |
| `tests/test_tracking_results.py` | ~15 | Result aggregation and statistics - Phase 3 |
| `tests/test_api_tracking.py` | ~10 | API integration and ID mapping - Phase 4 |
| `tests/visual/test_tracking_visualize.py` | ~10 | Visualization functions - Phase 5 |
| `tests/test_tracking_performance.py` | ~6 | Performance benchmarks - Phase 6 |
| `tests/test_tracking_integration.py` | ~8 | Realistic scenarios - Phase 7 |

**Total: ~230+ tests** (Phase 1 complete: 171 tests across 3 files)

---

## New Files to Create

```
view_arc/
    tracking.py                   # Core tracking logic and data structures
    
tests/
    test_tracking_dataclasses.py  # Step 1.1: ViewerSample, AOI, AOIResult, TrackingResult tests
    test_tracking_validation.py   # Step 1.2: Validation function tests
    test_tracking_session_config.py # Step 1.3: SessionConfig tests
    test_tracking_algorithm.py    # Steps 2.1-2.4: Core algorithm tests
    test_tracking_results.py      # Phase 3: Result analysis tests
    test_api_tracking.py          # Phase 4: API integration tests
    test_tracking_performance.py  # Phase 6: Performance benchmarks
    test_tracking_integration.py  # Phase 7: Integration scenarios
    visual/
        test_tracking_visualize.py  # Phase 5: Visualization tests
        
examples/
  attention_tracking_basic.py          # Minimal usage example
  attention_tracking_visualization.py  # Heatmap visualization
  attention_tracking_analysis.py       # Result analysis
    simulated_store_session.py         # Synthetic data simulation
```

---

## Files to Modify

```
view_arc/
  __init__.py          # Export new functions: compute_attention_seconds, AOI, TrackingResult
  obstacle/
    visualize.py       # Add heatmap, timeline, replay functions (optional)
    
README.md              # Add tracking feature documentation
```

---

## API Summary

### Primary Function
```python
from view_arc import compute_attention_seconds, AOI, ViewerSample

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

# Compute attention seconds
result = compute_attention_seconds(
    samples=samples,
    aois=aois,
    field_of_view_deg=90.0,
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

result = compute_attention_seconds(
    samples=(positions, directions),  # tuple of arrays
    aois=aois,
)

# Using single array of shape (N, 4)
data = np.array([
    [150, 300, 0.0, -1.0],
    [150, 300, 0.0, -1.0],
    [350, 300, 0.0, -1.0],
])

result = compute_attention_seconds(samples=data, aois=aois)
```

### Visualization
```python
from view_arc.obstacle.visualize import draw_attention_heatmap

annotated_image = draw_attention_heatmap(
    image=background_image,
    aois=aois,
    result=result,
    colormap="hot",
)
```

---

## Success Criteria

- [ ] `compute_attention_seconds()` correctly accumulates hits per AOI
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
