# Step 7.2 Implementation Summary

## Overview
Implemented all four example scripts specified in the tracking plan Phase 7, Step 7.2.

## Files Created

### 1. `examples/attention_tracking_basic.py`
**Purpose**: Minimal batch attention tracking demonstration

**Features**:
- Defines 3 simple rectangular shelf AOIs
- Creates 10 viewer samples (5s looking at Shelf_A, 3s at Shelf_B, 2s looking away)
- Computes attention seconds using `compute_attention_seconds()`
- Prints summary statistics and top AOIs
- **Ideal starting point** for users learning the API

**Output**:
```
Total samples processed: 10
Samples with AOI hits: 8
Coverage ratio: 80.0%
Per-AOI attention and rankings
```

### 2. `examples/attention_tracking_visualization.py`
**Purpose**: Demonstrate visualization capabilities with heatmaps

**Features**:
- Simulates 100-second viewing session with 3 shelves
- Generates attention heatmaps with multiple colormaps (hot, viridis)
- Adds text labels showing hit counts, percentages, and seconds
- Saves visualization outputs to `examples/output/`

**Output Files**:
- `attention_heatmap_hot.png` - Red-scale heatmap
- `attention_heatmap_viridis.png` - Yellow-green heatmap
- `attention_heatmap_labeled.png` - Heatmap with text annotations

**Requirements**: OpenCV (`opencv-python`)

### 3. `examples/attention_tracking_analysis.py`
**Purpose**: Demonstrate result aggregation and data export

**Features**:
- Shows all result aggregation methods:
  - `get_top_aois()` - Top N AOIs by hit count
  - `get_attention_distribution()` - Percentage breakdown
  - `get_viewing_timeline()` - Chronological sequence
  - `to_dataframe()` - pandas export
- Computes session summary statistics
- Demonstrates DataFrame operations (sum, mean, max)
- Validates percentages sum to 100%

**Output**:
- Formatted tables with rankings and distributions
- pandas DataFrame with columns: `aoi_id`, `hit_count`, `total_attention_seconds`, `attention_percentage`
- Example DataFrame operations

**Optional Dependency**: pandas (gracefully skips if not installed)

### 4. `examples/simulated_store_session.py`
**Purpose**: Complete end-to-end realistic simulation

**Features**:
- Loads actual store floor plan from JSON (`images/shop-floor.json`)
- Loads AOI annotations from JSON (`images/polygon_vertices.json`)
- Generates realistic viewer trajectory:
  - Starts at store entrance
  - Browses for 60 seconds at ~10 px/sec
  - Includes speed variations and pauses (10% chance)
  - Stays within shop floor boundaries
  - Turns gradually (no sharp angles)
- Generates natural view directions:
  - 60% following movement direction
  - 25% scanning left/right (perpendicular)
  - 15% looking at nearby AOIs
- Produces comprehensive visualizations:
  - Heatmap with viewer path overlay
  - Labeled heatmap with hit counts
  - Viewing timeline graphic
- Prints detailed statistics including top 10 AOIs

**Output Files**:
- `simulated_heatmap_with_path.png` - Shows attention + walking path
- `simulated_labeled.png` - Annotated with AOI IDs and stats
- `simulated_timeline.png` - Chronological viewing sequence

**Requirements**: OpenCV, scikit-image

## Testing

All examples were tested and verified working:

```bash
# Basic example
uv run python examples/attention_tracking_basic.py
✓ Runs successfully, prints results

# Visualization example  
uv run python examples/attention_tracking_visualization.py
✓ Creates 3 PNG files in examples/output/

# Analysis example
uv run python examples/attention_tracking_analysis.py
✓ Shows all aggregation methods, exports DataFrame

# Simulation example
uv run python examples/simulated_store_session.py
✓ Generates realistic session with visualizations
```

## Documentation Updates

### Updated Files

1. **`docs/TRACKING_PLAN.md`**
   - Marked Step 7.2 as ✅ COMPLETED
   - Added detailed descriptions of each example
   - Documented validation results

2. **`README.md`**
   - Added new "Temporal Attention Tracking" examples section
   - Organized examples by category (Single-Frame vs Tracking)
   - Added descriptions and "start here" guidance
   - Listed output files for each example

## Relationship to Existing Examples

**Note**: Several similar examples already existed:
- `basic_usage_tracking.py` - Uses `process_single_sample()` (single-frame wrapper)
- `attention_heatmap_demo.py` - Similar to new visualization example
- `result_aggregation_demo.py` - Similar to new analysis example
- `real_image_processing_tracking_full.py` - Similar to new simulation example

**Decision**: Created new examples with standardized naming per the plan:
- Consistent naming scheme (`attention_tracking_*.py`)
- Clear documentation headers
- Uniform structure and output formatting
- Better aligned with plan specifications

## API Coverage

The four examples collectively demonstrate:

✓ `ViewerSample` dataclass usage  
✓ `AOI` dataclass usage  
✓ `compute_attention_seconds()` function  
✓ `TrackingResult` properties and methods  
✓ `get_top_aois()` aggregation  
✓ `get_attention_distribution()` aggregation  
✓ `get_viewing_timeline()` aggregation  
✓ `to_dataframe()` export  
✓ `draw_attention_heatmap()` visualization  
✓ `draw_attention_labels()` visualization  
✓ `draw_viewing_timeline()` visualization  
✓ Path overlay visualization  
✓ Multiple colormap options  
✓ Session statistics (coverage_ratio, etc.)  

## Validation Checklist

Per TRACKING_PLAN.md Step 7.2:

- ✅ Examples run without errors
- ✅ Output is informative and correct
- ✅ All four specified examples created
- ✅ Content matches plan specifications:
  - ✅ Basic: Load AOIs, simulate samples, compute, print
  - ✅ Visualization: Heatmap overlay, save images
  - ✅ Analysis: DataFrame export, statistics, top AOIs
  - ✅ Simulation: Realistic trajectory, attention patterns

## Usage Recommendations

**For new users:**
1. Start with `attention_tracking_basic.py` to understand core API
2. Try `attention_tracking_visualization.py` to see heatmaps
3. Use `attention_tracking_analysis.py` for data export/analysis
4. Study `simulated_store_session.py` for complete workflow

**For production:**
- Use simulation example as template for real viewer data
- Adapt visualization examples for custom analysis dashboards
- Export to DataFrame for integration with analytics pipelines

## Step 7.2 Status

✅ **COMPLETED** - All requirements implemented and tested.
