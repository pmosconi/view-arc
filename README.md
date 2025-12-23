# Temporal Attention Tracking

Track viewer attention on Areas of Interest (AOIs) over time by analyzing where viewers are looking during acquisition sessions. This system accumulates "attention seconds" for each AOI (such as store shelves or displays) by processing batches of viewer position and view direction samples.

**Primary Use Case:** Analyze shopping behavior in retail environments by tracking which shelves or displays attract viewer attention and for how long.

**Core Technology:** Uses view arc obstacle detection to determine which AOI has the largest visible angular coverage from each viewer position, then aggregates these detections across time to compute total attention metrics.

## Installation

```bash
# Basic installation (numpy only)
uv pip install actvalue.view-arc

# With all optional features (visualization + examples)
uv pip install "actvalue.view-arc[all]"

# With visualization support only (opencv + matplotlib)
uv pip install "actvalue.view-arc[viz]"

# For development
uv pip install "actvalue.view-arc[dev]"
```

For local development:
```bash
uv venv --python 3.13
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
```

## Quick Start

The primary use case is tracking attention over time. Here's a minimal example:

```python
import numpy as np
from view_arc import compute_attention_seconds, AOI, ViewerSample

# Define areas of interest (e.g., store shelves)
aois = [
    AOI(id="shelf_A", contour=np.array([[100, 100], [200, 100], [200, 200], [100, 200]])),
    AOI(id="shelf_B", contour=np.array([[300, 100], [400, 100], [400, 200], [300, 200]])),
]

# Viewer samples (position + direction) captured at 1 Hz
samples = [
    ViewerSample(position=(150, 300), direction=(0.0, -1.0)),  # looking at shelf_A
    ViewerSample(position=(150, 300), direction=(0.0, -1.0)),  # still looking at shelf_A
    ViewerSample(position=(350, 300), direction=(0.0, -1.0)),  # moved, now at shelf_B
]

# Compute attention seconds
result = compute_attention_seconds(samples=samples, aois=aois)

# Analyze results
print(f"Shelf A received {result.aoi_results['shelf_A'].hit_count} seconds of attention")
print(f"Top AOIs: {result.get_top_aois(n=3)}")
print(f"Coverage: {result.coverage_ratio:.1%} of time spent viewing AOIs")
```

For single-frame obstacle detection (the underlying algorithm), see the [API Reference](#single-frame-obstacle-detection) section below.

## Usage

### Complete Examples

### Single-Frame Obstacle Detection

- `examples/basic_usage.py` – minimal, self-contained invocation with console output.
- `examples/visualization_demo.py` – renders the wedge, obstacles, and resolved intervals to `examples/output/visualization_demo.png` (requires OpenCV).
- `examples/real_image_processing.py` – extracts contours from the `skimage` astronaut image, runs detection, and saves an annotated overlay.

### Temporal Attention Tracking

- `examples/attention_tracking_basic.py` – minimal batch tracking example (10 seconds of viewer samples)
  - Shows basic usage of `compute_attention_seconds()`
  - Prints per-AOI hit counts and top AOIs
  - **Start here** for attention tracking

- `examples/attention_tracking_visualization.py` – heatmap visualization of attention distribution
  - Simulates 100-second viewing session
  - Generates colored heatmaps (hot/viridis colormaps)
  - Adds labels with hit counts and percentages
  - Saves outputs to `examples/output/`

- `examples/attention_tracking_analysis.py` – result analysis and export
  - Demonstrates all aggregation methods (`get_top_aois`, `get_attention_distribution`, etc.)
  - Exports results to pandas DataFrame
  - Shows viewing timeline and session statistics
  - Includes DataFrame operations examples

- `examples/simulated_store_session.py` – complete realistic simulation
  - Loads real store layout and AOI annotations from JSON
  - Generates 60 seconds of realistic browsing behavior
  - Simulates natural walking patterns and view scanning
  - Produces comprehensive visualizations (heatmap + timeline + path overlay)
  - **Most complete example** showing full workflow

Run every example with `uv run python <script>` so that dependencies resolve inside the project environment.

**For single-frame obstacle detection examples:**
- `examples/basic_usage.py` – minimal obstacle detection invocation with console output
- `examples/visualization_demo.py` – renders the view wedge, obstacles, and angular intervals
- `examples/real_image_processing.py` – extracts contours from images and runs detection

## How It Works

### Temporal Attention Tracking

The system processes batches of viewer samples (typically captured at 1 Hz) and determines which AOI the viewer is looking at for each sample:

1. **Input:** Batch of viewer positions and view directions sampled over time (e.g., 60 samples = 1 minute at 1 Hz)
2. **Processing:** For each sample, run view arc obstacle detection to find which AOI has the largest visible angular coverage
3. **Accumulation:** Count hits per AOI - each hit represents 1 second of attention time
4. **Output:** Per-AOI metrics (hit counts, total attention seconds, viewing timeline, etc.)

### View Arc Obstacle Detection (Foundation)

The underlying algorithm determines which polygon (AOI) has the largest visible angular coverage within a viewer's field of view:

1. **Transform** obstacle contours to viewer-relative polar coordinates
2. **Clip** polygons to the circular view range and angular field-of-view wedge
3. **Compute** angular intervals where each obstacle is the closest visible object (angular sweep algorithm)
4. **Select** the obstacle with the largest total angular coverage

This single-frame detection is then applied repeatedly across time to build attention metrics.

## Tracking Assumptions

Temporal attention tracking (see `docs/TRACKING_PLAN.md`) relies on the following upstream guarantees that we do **not** re-validate at runtime:

1. **1 Hz cadence**: Viewer samples arrive exactly at 1 Hz, so each accepted hit represents one second of attention.
2. **One second per sample**: Each sample represents exactly one second of viewing time—no interpolation is performed.
3. **Monotonic timestamps**: Timestamps, when provided, are already sorted upstream; we consume them as-is.
4. **Fixed coordinate space**: Viewer positions, directions, and AOI contours all share the same immutable image-coordinate space for the entire batch.
5. **Single viewer**: Each batch tracks a single viewer; multi-viewer aggregation happens outside this API.

Any pipeline feeding `compute_attention_seconds()` must uphold these invariants to keep the reported metrics meaningful. You can also inspect `TrackingResult.assumptions` or the `SAMPLING_ASSUMPTIONS` constant to see these contracts programmatically.

## Type Checking

Mypy is configured with ``disallow_untyped_defs`` and must remain clean:

```bash
uv run mypy .
```

Address any type errors before committing changes.

## Performance Profiling

The project includes performance profiling tools to monitor runtime and accuracy regression:

### Running Profiling Scenarios

Run performance baselines with:

```bash
# Run tracking baseline scenarios only
uv run python profile_workload.py --scenario tracking_baseline

# Run all scenarios (obstacle detection + tracking)
uv run python profile_workload.py --scenario all

# Save results to CSV for trend tracking
uv run python profile_workload.py --scenario tracking_baseline --save-csv
```

Results are saved to `examples/output/profile_runs.csv` with timestamp, runtime, throughput, and accuracy metrics.

### Profiling in Code

Enable lightweight profiling in tracking runs:

```python
from view_arc.tracking import compute_attention_seconds

result = compute_attention_seconds(
    samples=samples,
    aois=aois,
    enable_profiling=True  # Enable performance metrics
)

# Access profiling data
if result.profiling_data:
    print(result.profiling_data)
    # Output includes:
    # - Total time
    # - Samples processed
    # - Throughput (samples/s)
    # - Average time per sample (ms)
```

**Note**: Profiling has negligible overhead (<1%) and does not alter results.

### CI Integration

For weekly regression detection, run:

```bash
uv run python profile_workload.py --scenario tracking_baseline --save-csv
```

Compare results against previous runs in `examples/output/profile_runs.csv`. Threshold alerts are emitted when:
- Runtime exceeds 120% of golden baseline
- Hit counts differ from expected values

## API Reference

### Temporal Attention Tracking

#### `compute_attention_seconds()`

Main entry point for batch processing of viewer samples to compute attention seconds per AOI.

```python
from view_arc import compute_attention_seconds, AOI, ViewerSample
import numpy as np

# Define areas of interest
aois = [
    AOI(id="shelf_1", contour=np.array([[100, 100], [200, 100], [200, 200], [100, 200]])),
    AOI(id="shelf_2", contour=np.array([[300, 100], [400, 100], [400, 200], [300, 200]])),
]

# Define viewer samples (position, direction pairs sampled at 1 Hz)
samples = [
    ViewerSample(position=(150, 300), direction=(0.0, -1.0)),  # looking up at shelf_1
    ViewerSample(position=(150, 300), direction=(0.0, -1.0)),  # still looking
    ViewerSample(position=(350, 300), direction=(0.0, -1.0)),  # moved, looking at shelf_2
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
print(result.get_top_aois(n=5))  # Top 5 AOIs by attention
print(result.coverage_ratio)  # Fraction of samples with a hit
```

**Parameters:**
- `samples`: List of `ViewerSample` objects or numpy array of shape (N, 4) where each row is [x, y, dx, dy]
- `aois`: List of `AOI` objects defining the areas to track
- `field_of_view_deg`: Field of view in degrees (default 90.0)
- `max_range`: Maximum detection range in pixels (default 500.0)
- `sample_interval`: Time per sample in seconds (default 1.0)
- `session_config`: Optional `SessionConfig` for metadata tracking
- `enable_profiling`: Enable performance metrics (default False)

**Returns:** `TrackingResultWithConfig` containing:
- `aoi_results`: Dict mapping AOI IDs to `AOIResult` objects with hit counts
- `total_samples`: Total number of samples processed
- `samples_with_hits`: Samples where any AOI was visible
- `samples_no_winner`: Samples where no AOI was in view
- `session_config`: Embedded session configuration
- `profiling_data`: Performance metrics if enabled

#### Result Analysis Methods

```python
# Get top N AOIs by hit count
top_aois = result.get_top_aois(n=3)
# Returns: [(aoi_id, hit_count), ...]

# Get attention distribution as percentages
distribution = result.get_attention_distribution()
# Returns: {aoi_id: percentage, ...}

# Get viewing timeline (chronological sequence)
timeline = result.get_viewing_timeline()
# Returns: [(sample_index, aoi_id), ...]

# Export to pandas DataFrame (optional dependency)
df = result.to_dataframe()
# Returns: DataFrame with columns [aoi_id, hit_count, total_attention_seconds, ...]
```

#### Alternative Input Formats

```python
# Using numpy arrays directly (directions are auto-normalized)
data = np.array([
    [150, 300, 0.0, -1.0],  # x, y, dx, dy
    [150, 300, 0.0, -1.0],
    [350, 300, 0.0, -1.0],
])

result = compute_attention_seconds(samples=data, aois=aois)
```

### Single-Frame Obstacle Detection

#### `find_largest_obstacle()`

The foundational algorithm that finds which obstacle has the largest visible angular coverage. This is called internally by `compute_attention_seconds()` for each viewer sample.

```python
from view_arc import find_largest_obstacle
import numpy as np

viewer = np.array([100.0, 100.0], dtype=np.float32)
view_direction = np.array([0.0, 1.0], dtype=np.float32)  # unit vector
contours = [
    np.array([[90.0, 150.0], [110.0, 150.0], [100.0, 190.0]], dtype=np.float32),
    np.array([[70.0, 130.0], [130.0, 130.0], [130.0, 180.0], [70.0, 180.0]], dtype=np.float32),
]

result = find_largest_obstacle(
    viewer_point=viewer,
    view_direction=view_direction,
    field_of_view_deg=60.0,
    max_range=150.0,
    obstacle_contours=contours,
    return_intervals=True,
)

print(result.summary())
```

**Parameters:**
- `viewer_point`: (x, y) position in image coordinates
- `view_direction`: Unit vector (dx, dy) indicating viewing direction
- `field_of_view_deg`: Field of view in degrees
- `max_range`: Maximum detection range in pixels
- `obstacle_contours`: List of numpy arrays, each shape (N, 2) representing polygon vertices
- `return_intervals`: Return detailed angular interval breakdown (default False)
- `return_all_coverage`: Return coverage for all obstacles (default False)

**Returns:** `ObstacleResult` containing:
- `obstacle_id`: Index of winning obstacle or None
- `angular_coverage`: Angular coverage in radians
- `min_distance`: Minimum distance to the winner
- `interval_details`: Detailed angular intervals (if requested)
- `all_coverage`: Coverage for all obstacles (if requested)

### Visualization Functions

```python
from view_arc.tracking import draw_attention_heatmap, draw_viewing_timeline, generate_session_replay

# Draw attention heatmap
heatmap_image = draw_attention_heatmap(
    image=background_image,
    aois=aois,
    result=tracking_result,
    colormap="hot",  # or "viridis", "plasma", etc.
    alpha=0.5,
)

# Draw viewing timeline
timeline_image = draw_viewing_timeline(
    tracking_result=tracking_result,
    aois=aois,
    width=800,
    height=200,
)

# Generate session replay frames
frames = generate_session_replay(
    samples=samples,
    aois=aois,
    winners=winner_ids,
    result=tracking_result,
    image_size=(800, 600),
    field_of_view_deg=90.0,
)
```

### Data Structures

#### `ViewerSample`
```python
@dataclass(frozen=True)
class ViewerSample:
    position: tuple[float, float]  # (x, y) in image coordinates
    direction: tuple[float, float]  # unit vector (dx, dy)
    timestamp: float | None = None  # optional timestamp
```

#### `AOI`
```python
@dataclass(frozen=True)
class AOI:
    id: str | int  # unique identifier
    contour: NDArray[np.floating[Any]]  # polygon vertices, shape (N, 2)
```

#### `AOIResult`
```python
@dataclass
class AOIResult:
    aoi_id: str | int
    hit_count: int  # number of times selected as winner
    total_attention_seconds: float  # = hit_count × sample_interval
    hit_timestamps: list[int]  # indices of samples where this AOI won
```

#### `SessionConfig`
```python
@dataclass
class SessionConfig:
    session_id: str
    frame_size: tuple[int, int] | None = None  # (width, height) for bounds checking
    coordinate_space: Literal["image"] = "image"
    sample_interval_seconds: float = 1.0
    viewer_id: str | None = None
    notes: dict[str, Any] | None = None
```

For complete API documentation with type hints and detailed descriptions, use:
```python
help(compute_attention_seconds)  # Primary API for attention tracking
help(find_largest_obstacle)       # Underlying obstacle detection
```

## Architecture

The package is organized with temporal tracking as the primary interface:

```
view_arc/
    __init__.py              # Main public API (compute_attention_seconds, etc.)
    tracking/                # Temporal attention tracking
        algorithm.py         # Batch processing and accumulation
        dataclasses.py       # ViewerSample, AOI, TrackingResult
        validation.py        # Input validation
        visualize.py         # Heatmaps, timelines, session replay
    obstacle/                # Low-level obstacle detection (foundation)
        api.py               # find_largest_obstacle
        geometry.py          # Coordinate transforms
        clipping.py          # Polygon clipping
        sweep.py             # Angular sweep algorithm
        visualize.py         # Single-frame visualization
```

**Design Philosophy:** Temporal tracking is the main use case. Obstacle detection is exposed as a public API for advanced users who need single-frame analysis or want to build custom tracking logic.

## Development Workflow

### Environment Setup

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python environment management.

```bash
# Create virtual environment with Python 3.13
uv venv --python 3.13

# Activate the environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install project with dev dependencies
uv pip install -e ".[dev]"
```

**Optional Environment Variables:**

The project includes a `.env.example` file for reference. Most configuration is handled through function parameters, but environment variables can be used for custom workflows:

```bash
# Copy and customize if needed
cp .env.example .env
```

Variables documented in `.env.example`:
- Data paths for custom data loading
- Output paths for results and visualizations
- Profiling configuration thresholds
- Testing configuration options

### Running Tests

```bash
# Run all automated tests (excludes visual tests)
uv run pytest

# Run with coverage report
uv run pytest --cov=view_arc --cov-report=html

# Run specific test module
uv run pytest tests/test_tracking_compute_attention.py

# Run only tracking tests
uv run pytest tests/test_tracking_*.py
```

**Visual Tests (Optional):**

Visual tests generate matplotlib/OpenCV figures for manual inspection but do not run assertions. They are marked with `@pytest.mark.visual` and excluded from the default test suite.

```bash
# Run all visual tests (generates output images)
uv run pytest -m visual

# Run specific visual test file
uv run pytest -m visual tests/visual/test_api_visual.py

# Combine with other markers
uv run pytest -m "visual and not slow"
```

Visual test outputs are saved to `tests/visual/output/`. These tests are useful for:
- Debugging clipping and sweep algorithms
- Verifying obstacle detection visualization
- Creating documentation figures

**Note:** The main test suite (`pytest` without `-m visual`) provides complete behavioral coverage. Visual tests serve as supplementary validation tools, not correctness checks.

### Type Checking

The project enforces strict type checking with mypy:

```bash
# Check all files
uv run mypy .

# Check specific module
uv run mypy view_arc/tracking/

# Fix all type errors before committing
```

Configuration is in `pyproject.toml` with `disallow_untyped_defs = true`.

### Performance Testing

```bash
# Run tracking performance baseline
uv run python profile_workload.py --scenario tracking_baseline

# Run all performance scenarios
uv run python profile_workload.py --scenario all

# Save results to CSV for trend tracking
uv run python profile_workload.py --scenario tracking_baseline --save-csv
```

### Quick Commands

**Using Make (recommended):**
```bash
# Show all available commands
make help

# Full validation pipeline (type check + test + profile)
make check

# Run tests
make test

# Run tests with coverage
make test-cov

# Type checking
make mypy

# Format and lint code
make format
make lint

# Run performance profiling
make profile

# Run examples
make example-basic
make example-simulation
```

**Using uv directly:**
```bash
# Full validation pipeline (type check + test + profile)
uv run mypy . && uv run pytest && uv run python profile_workload.py --scenario tracking_baseline

# Run examples to verify functionality
uv run python examples/attention_tracking_basic.py
uv run python examples/simulated_store_session.py
```

### Code Formatting

```bash
# Format code with ruff
uv run ruff format .

# Check linting
uv run ruff check .

# Fix auto-fixable linting issues
uv run ruff check --fix .
```

## References

- **Algorithm Details:** See `docs/obstacle_arc_spec.md` for the view arc obstacle detection algorithm specification
- **Implementation Plan:** See `docs/TRACKING_PLAN.md` for the complete temporal tracking implementation plan
- **Performance Analysis:** See `docs/PERFORMANCE_ANALYSIS.md` for performance characteristics and optimization opportunities
