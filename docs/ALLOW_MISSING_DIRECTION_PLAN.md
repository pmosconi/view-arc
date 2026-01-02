# Implementation Plan: Allow Missing Direction Feature

## Overview
Add support for samples with missing/invalid direction vectors under a new flag `allow_missing_direction`. When enabled, samples with missing directions will result in "no hit" rather than throwing an error.

## Motivation
Current behavior rejects samples with zero-magnitude direction vectors (`(0, 0)`) or `None` values. In real-world tracking scenarios, direction data may occasionally be unavailable (e.g., sensor failures, stationary viewers, data gaps). This feature provides graceful handling of such cases without discarding the entire batch.

## Design Decisions

### 1. Missing Direction Representation
**Decision**: Use `(0.0, 0.0)` as the canonical representation
- Maintains type consistency (direction is always `tuple[float, float]`)
- Works seamlessly with NumPy array input format (can't use `None` in numeric arrays)
- Recognizable sentinel value that won't accidentally occur in valid data

### 2. ViewerSample Changes Required
**Decision**: Add optional `allow_missing_direction: bool = False` field to `ViewerSample` dataclass
- **Critical**: Without this, `ViewerSample.__post_init__()` validation rejects `(0.0, 0.0)` before any processing occurs
- Allows users creating `list[ViewerSample]` to opt-in per sample: `ViewerSample(position=(...), direction=(0.0, 0.0), allow_missing_direction=True)`
- For NumPy array inputs, the flag is controlled by the `compute_attention_seconds()` parameter
- Backwards compatible: default `False` maintains strict validation
- Frozen dataclass remains immutable

### 3. Implementation Level
**Decision**: Implement at the **tracking level only**
- The obstacle API (`find_largest_obstacle`) expects valid normalized directions - this is a core contract
- The tracking layer is already responsible for sample validation and normalization
- Keeps the obstacle API clean and focused on geometric computations
- Missing directions are a data quality concern, not a geometric computation concern

### 4. Configuration Interface
**Decision**: Two-level configuration
- **Per-sample level**: `ViewerSample(..., allow_missing_direction=True)` for explicit list inputs
- **Batch level**: `compute_attention_seconds(..., allow_missing_direction=True)` for NumPy array inputs
- Both levels must agree: batch-level flag enables missing directions for all samples from NumPy arrays
- `SessionConfig` remains unchanged - it's for immutable acquisition metadata, not data processing flags

## Implementation Details

### Phase 1: Core Changes

#### 1.1 Update `ViewerSample` Dataclass
**File**: `view_arc/tracking/dataclasses.py`

**Critical Changes**:
1. Add `allow_missing_direction: bool = False` field to `ViewerSample` dataclass
   - Place after `timestamp` field
   - Include in `__post_init__()` validation signature

2. Modify `_validate_direction()` signature:
   ```python
   def _validate_direction(direction: tuple[float, float], allow_missing: bool = False) -> None:
   ```
   - When `allow_missing=True` and `direction == (0.0, 0.0)`, return early (valid)
   - When `allow_missing=False` or direction is non-zero, apply current unit vector validation

3. Update `ViewerSample.__post_init__()`:
   ```python
   def __post_init__(self) -> None:
       """Validate that direction is a unit vector or missing if allowed."""
       _validate_direction(self.direction, allow_missing=self.allow_missing_direction)
   ```

4. Add helper function:
   ```python
   def is_missing_direction(direction: tuple[float, float]) -> bool:
       """Check if direction represents missing data."""
       return direction == (0.0, 0.0)
   ```

**Backwards Compatibility**: Default `allow_missing_direction=False` maintains current strict behavior

#### 1.2 Update Sample Normalization
**File**: `view_arc/tracking/validation.py`

**Changes**:
- Add `allow_missing_direction: bool = False` parameter to `normalize_sample_input()`
- When normalizing NumPy arrays (lines ~136-147):
  - If `mag == 0` and `allow_missing_direction=True`:
    ```python
    # Create sample with missing direction sentinel
    result.append(
        ViewerSample(
            position=(x, y),
            direction=(0.0, 0.0),
            allow_missing_direction=True  # Critical: enables validation bypass
        )
    )
    continue  # Skip normalization
    ```
  - If `mag == 0` and `allow_missing_direction=False`: raise `ValidationError` (current behavior)
- For list inputs: samples are returned as-is, their individual `allow_missing_direction` flags are respected

#### 1.3 Update `process_single_sample()`
**File**: `view_arc/tracking/algorithm.py`

**Changes**:
- No parameter needed - check the sample's own `allow_missing_direction` field
- Before calling `find_largest_obstacle()` (after initial validation):
  ```python
  # Handle missing directions (skip obstacle API entirely)
  if is_missing_direction(sample.direction):
      if not sample.allow_missing_direction:
          raise ValidationError(
              "Sample has missing direction (0.0, 0.0) but allow_missing_direction=False"
          )
      # Return no-hit result
      if return_details:
          return SingleSampleResult(winning_aoi_id=None)
      return None
  ```
- This check happens after validation but before any obstacle API calls
- Provides clear error if direction is `(0.0, 0.0)` but flag is `False`

#### 1.4 Update `compute_attention_seconds()`
**File**: `view_arc/tracking/algorithm.py`

**Changes**:
- Add `allow_missing_direction: bool = False` parameter
- Pass it to `normalize_sample_input()`
- When processing samples:
  - Check for missing direction before calling `process_single_sample()`
  - Count samples with missing directions in `samples_no_winner`
- Document behavior in docstring

#### 1.5 Update `_iterate_samples_chunked()`
**File**: `view_arc/tracking/algorithm.py`

**Critical Changes** (addresses streaming mode validation):
- Add `allow_missing_direction: bool = False` parameter to function signature
- In NumPy array normalization section (lines ~571-579):
  ```python
  # Normalize direction to unit vector
  mag = math.sqrt(dx * dx + dy * dy)
  if mag == 0:
      if allow_missing_direction:
          # Create sample with missing direction
          chunk.append(
              ViewerSample(
                  position=(x, y),
                  direction=(0.0, 0.0),
                  allow_missing_direction=True
              )
          )
          continue  # Skip to next sample
      else:
          raise ValidationError(
              f"Sample at index {i} has zero-magnitude direction vector"
          )
  ```
- This prevents streaming mode from crashing before samples reach the main loop

#### 1.6 Update `compute_attention_seconds_streaming()`
**File**: `view_arc/tracking/algorithm.py`

**Changes**:
- Add `allow_missing_direction: bool = False` parameter
- Pass through to `_iterate_samples_chunked(samples, chunk_size, allow_missing_direction)`
- Consistent behavior with batch mode

### Phase 2: Testing

#### 2.1 Unit Tests
**File**: `tests/test_tracking_dataclasses.py`

Add new test class `TestViewerSampleMissingDirection`:
- `test_missing_direction_rejected_by_default()`: Verify `ViewerSample(direction=(0, 0))` raises error
- `test_missing_direction_accepted_with_flag()`: Verify `ViewerSample(direction=(0, 0), allow_missing_direction=True)` succeeds
- `test_missing_direction_false_with_zero_direction()`: Verify `ViewerSample(direction=(0, 0), allow_missing_direction=False)` raises error
- `test_is_missing_direction_utility()`: Test the helper function
- `test_missing_direction_field_in_frozen_dataclass()`: Verify field works with frozen dataclass

**File**: `tests/test_tracking_validation.py`

Add tests for `normalize_sample_input()`:
- `test_normalize_numpy_zero_direction_default()`: Verify error on `(0, 0)` by default
- `test_normalize_numpy_zero_direction_allowed()`: Verify `(0, 0)` creates sample with `allow_missing_direction=True`
- `test_normalize_list_with_missing_direction_flag_true()`: Test list with `ViewerSample(..., allow_missing_direction=True)` passes through
- `test_normalize_list_with_missing_direction_flag_false()`: Test list with `ViewerSample(direction=(0,0))` raises error during construction

**File**: `tests/test_tracking_process_single_sample.py`

Add tests:
- `test_process_single_sample_missing_direction_flag_false()`: Verify error when sample has `allow_missing_direction=False` and direction is `(0, 0)`
- `test_process_single_sample_missing_direction_flag_true()`: Verify None returned when sample has `allow_missing_direction=True`
- `test_process_single_sample_missing_direction_with_details()`: Verify `SingleSampleResult(winning_aoi_id=None)` with flag
- `test_missing_direction_does_not_call_obstacle_api()`: Verify obstacle API never called (mock test)

#### 2.2 Integration Tests
**File**: `tests/test_tracking_integration.py`

Add tests:
- `test_compute_attention_numpy_mixed_samples()`: NumPy array with mix of valid and `(0, 0)` directions
  - Use `allow_missing_direction=True` parameter
  - Verify hit counts only from valid samples
  - Verify `samples_no_winner` includes missing direction samples
  - Verify invariants maintained
- `test_compute_attention_list_mixed_samples()`: List of ViewerSamples with some having `allow_missing_direction=True`
  - Verify those samples treated as no-hit
  - Verify samples without flag work normally
- `test_compute_attention_all_missing_directions()`: All NumPy samples have `(0, 0)` with flag
  - Verify all AOIs have 0 hits
  - Verify `samples_no_winner == total_samples`
- `test_compute_attention_missing_direction_flag_false()`: Verify error when NumPy has `(0, 0)` but `allow_missing_direction=False`
- `test_streaming_mode_with_missing_directions()`: Test `_iterate_samples_chunked()` and streaming mode handle missing directions correctly

### Phase 3: Documentation

#### 3.1 API Documentation
**File**: `view_arc/tracking/algorithm.py`

Update docstrings:
- `compute_attention_seconds()`: Document new parameter and behavior
- `compute_attention_seconds_streaming()`: Document new parameter
- `process_single_sample()`: Document new parameter

#### 3.2 Main README
**File**: `README.md`

Add section under "Advanced Features":
```markdown
### Handling Missing Direction Data

In real-world scenarios, direction data may occasionally be unavailable. The tracking API can gracefully handle missing directions:

```python
import numpy as np

# Sample with missing direction (0, 0)
samples = np.array([
    [100.0, 100.0, 1.0, 0.0],    # Valid direction
    [100.0, 100.0, 0.0, 0.0],    # Missing direction
    [100.0, 100.0, 0.0, 1.0],    # Valid direction
])

# Enable missing direction handling
result = compute_attention_seconds(
    samples, 
    aois,
    allow_missing_direction=True  # Default: False
)

# Samples with missing directions count as "no hit"
print(result.samples_no_winner)  # Includes missing direction samples
```

**Behavior**:
- Samples with direction `(0.0, 0.0)` are treated as "no hit"
- They increment `samples_no_winner` counter
- No AOI receives attention for these samples
- All other validation and invariants maintained
```

#### 3.3 Tracking Plan Document
**File**: `docs/TRACKING_PLAN.md`

Add section on missing directions explaining:
- When to use the feature
- Performance characteristics (faster, skips obstacle API)
- Data quality considerations

#### 3.4 Example Script
**File**: `examples/missing_direction_handling.py`

Create new example demonstrating:
- Creating samples with missing directions
- Enabling the flag
- Interpreting results
- Best practices for data quality

### Phase 4: Type Checking & Quality

#### 4.1 Type Annotations
- Ensure all new parameters have proper type hints
- Run `mypy` to verify no type errors introduced

#### 4.2 Code Review Checklist
- [ ] All parameters have default values for backwards compatibility
- [ ] All new code has docstrings
- [ ] All test invariants verified
- [ ] No breaking changes to existing API
- [ ] Documentation updated in all relevant places

## Edge Cases & Considerations

### 1. Timestamp Handling
- Samples with missing directions should NOT appear in `hit_timestamps`
- They should still be included in `total_samples` count

### 2. Validation Order
- Missing direction check should happen AFTER normalization
- But BEFORE calling obstacle API

### 3. NumPy Array Input
- Zero magnitude direction in NumPy array creates `ViewerSample` with `allow_missing_direction=True` when batch flag enabled
- Samples created with the flag set appropriately to match batch-level parameter

### 4. List[ViewerSample] Input
- Users must explicitly set `allow_missing_direction=True` when constructing samples with `(0, 0)` direction
- Example: `ViewerSample(position=(100, 100), direction=(0.0, 0.0), allow_missing_direction=True)`
- Attempting to construct `ViewerSample(direction=(0, 0))` without the flag raises `ValidationError` at construction time

### 5. Error Messages
Multiple validation points with clear messages:
- At construction: "view_direction must be a unit vector" if `direction=(0, 0)` and `allow_missing_direction=False`
- In `process_single_sample()`: "Sample has missing direction (0.0, 0.0) but allow_missing_direction=False"
- In normalization: "Sample at index {i} has zero-magnitude direction vector" if flag is `False`

### 6. Performance
- Skipping obstacle API call for missing directions improves performance
- Early return in `process_single_sample()` avoids all geometric computations

## Migration Path

### For Existing Users
No action required - default behavior unchanged

### For New Users Wanting Feature
- **NumPy arrays**: Add `allow_missing_direction=True` to `compute_attention_seconds()` call
- **List inputs**: Set `allow_missing_direction=True` on individual samples during construction

## Testing Strategy Summary

1. **Unit Tests**: Test each component in isolation
2. **Integration Tests**: Test end-to-end behavior with mixed data
3. **Edge Case Tests**: Empty samples, all missing, none missing

## Success Criteria

- [ ] All tests pass (existing + new)
- [ ] No type errors with `mypy`
- [ ] Documentation complete and accurate
- [ ] Backwards compatible (no breaking changes)
- [ ] Code review approved

## Estimated Effort

- **Phase 1 (Implementation)**: 2-3 hours
- **Phase 2 (Testing)**: 2 hours  
- **Phase 3 (Documentation)**: 1-2 hours
- **Phase 4 (Quality/Review)**: 1 hour

**Total**: ~6-8 hours

## Critical Implementation Notes

### Addressing Review Findings

1. **ViewerSample Construction Issue** (CRITICAL):
   - **Problem**: `__post_init__()` validates unconditionally, preventing `(0, 0)` construction
   - **Solution**: Add `allow_missing_direction` field to dataclass, pass to `_validate_direction()`
   - **Impact**: Requires signature change but maintains backwards compatibility via default `False`

2. **List[ViewerSample] Input Issue** (MAJOR):
   - **Problem**: `normalize_sample_input()` returns list inputs as-is, no way to apply flag
   - **Solution**: Users must set `allow_missing_direction=True` on individual samples during construction
   - **Impact**: Provides explicit opt-in mechanism for list inputs
   - **Example**: `ViewerSample(position=(...), direction=(0.0, 0.0), allow_missing_direction=True)`

3. **Streaming Mode Issue** (MAJOR):
   - **Problem**: `_iterate_samples_chunked()` raises on zero magnitude before main loop sees samples
   - **Solution**: Add `allow_missing_direction` parameter to function, handle zero magnitude by creating flagged samples
   - **Impact**: Must update both `_iterate_samples_chunked()` and `compute_attention_seconds_streaming()`

### Additional Notes

1. **Logging/Warnings**: Keep silent - users can check `samples_no_winner` count if concerned about data quality.

2. **Separate Tracking**: Missing direction count not tracked separately from other no-winner cases in initial implementation. Can add later if needed.

3. **Thresholds**: No automatic threshold (e.g., error if >50% missing) - let users decide data quality thresholds.

4. **Type Safety**: The `allow_missing_direction` field is fully type-safe and works with frozen dataclasses.

## Implementation Order

1. **Phase 1.1**: Update `ViewerSample` dataclass and `_validate_direction()` - CRITICAL FIRST
2. **Phase 2.1**: Add dataclass unit tests immediately to verify construction works
3. **Phase 1.2**: Update `normalize_sample_input()` for NumPy arrays
4. **Phase 1.3**: Update `process_single_sample()` with missing direction check
5. **Phase 1.4**: Update `compute_attention_seconds()` with parameter
6. **Phase 1.5-1.6**: Update `_iterate_samples_chunked()` and streaming mode - CRITICAL for streaming
7. **Phase 2.1 (continued)**: Add remaining unit tests for validation and processing
8. **Phase 2.2**: Integration tests for both NumPy and list inputs
9. **Phase 3**: Documentation (docstrings, README, examples)
10. **Phase 4**: Final quality checks (mypy, review)

**Critical Path**: Steps 1-2 must succeed before anything else works. Step 6 is critical for streaming mode support.

This order addresses the validation-at-construction issue first, tests it immediately, then builds out the rest of the feature.
