"""
Tests for TrackingResult aggregation methods (Step 3.1).

Tests cover:
- get_top_aois(): ordering, ties, edge cases
- get_attention_distribution(): percentages, zero hits
- get_viewing_timeline(): chronological sequence, gaps
- to_dataframe(): pandas export, column structure
"""

import numpy as np
import pytest

from view_arc.tracking.dataclasses import (
    AOI,
    AOIResult,
    TrackingResult,
    ValidationError,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_aois() -> list[AOI]:
    """Create sample AOIs for testing."""
    return [
        AOI(id="shelf_A", contour=np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)),
        AOI(id="shelf_B", contour=np.array([[200, 0], [300, 0], [300, 100], [200, 100]], dtype=np.float32)),
        AOI(id="shelf_C", contour=np.array([[400, 0], [500, 0], [500, 100], [400, 100]], dtype=np.float32)),
    ]


@pytest.fixture
def tracking_result_basic() -> TrackingResult:
    """Create a basic tracking result with varied hit counts."""
    aoi_results: dict[str | int, AOIResult] = {
        "shelf_A": AOIResult(aoi_id="shelf_A", hit_count=10, total_attention_seconds=10.0, hit_timestamps=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        "shelf_B": AOIResult(aoi_id="shelf_B", hit_count=5, total_attention_seconds=5.0, hit_timestamps=[10, 11, 12, 13, 14]),
        "shelf_C": AOIResult(aoi_id="shelf_C", hit_count=2, total_attention_seconds=2.0, hit_timestamps=[15, 16]),
    }
    return TrackingResult(
        aoi_results=aoi_results,
        total_samples=20,
        samples_with_hits=17,
        samples_no_winner=3,
    )


@pytest.fixture
def tracking_result_with_ties() -> TrackingResult:
    """Create a tracking result with tied hit counts."""
    aoi_results: dict[str | int, AOIResult] = {
        "shelf_A": AOIResult(aoi_id="shelf_A", hit_count=5, total_attention_seconds=5.0, hit_timestamps=[0, 1, 2, 3, 4]),
        "shelf_B": AOIResult(aoi_id="shelf_B", hit_count=5, total_attention_seconds=5.0, hit_timestamps=[5, 6, 7, 8, 9]),
        "shelf_C": AOIResult(aoi_id="shelf_C", hit_count=3, total_attention_seconds=3.0, hit_timestamps=[10, 11, 12]),
    }
    return TrackingResult(
        aoi_results=aoi_results,
        total_samples=15,
        samples_with_hits=13,
        samples_no_winner=2,
    )


@pytest.fixture
def tracking_result_no_hits() -> TrackingResult:
    """Create a tracking result with no hits."""
    aoi_results: dict[str | int, AOIResult] = {
        "shelf_A": AOIResult(aoi_id="shelf_A", hit_count=0, total_attention_seconds=0.0, hit_timestamps=[]),
        "shelf_B": AOIResult(aoi_id="shelf_B", hit_count=0, total_attention_seconds=0.0, hit_timestamps=[]),
    }
    return TrackingResult(
        aoi_results=aoi_results,
        total_samples=10,
        samples_with_hits=0,
        samples_no_winner=10,
    )


# =============================================================================
# Tests: get_top_aois()
# =============================================================================


def test_get_top_aois_basic(tracking_result_basic: TrackingResult) -> None:
    """Test basic top AOIs retrieval with correct ordering."""
    top_3 = tracking_result_basic.get_top_aois(3)
    
    assert len(top_3) == 3
    assert top_3[0] == ("shelf_A", 10)
    assert top_3[1] == ("shelf_B", 5)
    assert top_3[2] == ("shelf_C", 2)


def test_get_top_aois_ties(tracking_result_with_ties: TrackingResult) -> None:
    """Test that ties are handled with lexicographic ordering."""
    top_2 = tracking_result_with_ties.get_top_aois(2)
    
    assert len(top_2) == 2
    # shelf_A and shelf_B both have 5 hits, so alphabetical order applies
    assert top_2[0] == ("shelf_A", 5)
    assert top_2[1] == ("shelf_B", 5)


def test_get_top_aois_more_than_available(tracking_result_basic: TrackingResult) -> None:
    """Test requesting more AOIs than exist returns all."""
    top_10 = tracking_result_basic.get_top_aois(10)
    
    assert len(top_10) == 3  # Only 3 AOIs exist
    assert top_10[0] == ("shelf_A", 10)
    assert top_10[1] == ("shelf_B", 5)
    assert top_10[2] == ("shelf_C", 2)


def test_get_top_aois_zero(tracking_result_basic: TrackingResult) -> None:
    """Test requesting 0 AOIs returns empty list."""
    top_0 = tracking_result_basic.get_top_aois(0)
    
    assert top_0 == []


def test_get_top_aois_negative_raises_error(tracking_result_basic: TrackingResult) -> None:
    """Test that negative n raises ValidationError."""
    with pytest.raises(ValidationError, match="n must be non-negative"):
        tracking_result_basic.get_top_aois(-1)


def test_get_top_aois_with_zero_hits(tracking_result_no_hits: TrackingResult) -> None:
    """Test top AOIs when all have zero hits (alphabetical order)."""
    top_2 = tracking_result_no_hits.get_top_aois(2)
    
    assert len(top_2) == 2
    assert top_2[0] == ("shelf_A", 0)
    assert top_2[1] == ("shelf_B", 0)


# =============================================================================
# Tests: get_attention_distribution()
# =============================================================================


def test_attention_distribution_sums_to_100(tracking_result_basic: TrackingResult) -> None:
    """Test that attention percentages sum to 100."""
    distribution = tracking_result_basic.get_attention_distribution()
    
    total_percentage = sum(distribution.values())
    assert abs(total_percentage - 100.0) < 0.01  # Allow small floating point error


def test_attention_distribution_excludes_no_hits_by_default(tracking_result_basic: TrackingResult) -> None:
    """Test that AOIs with zero hits are excluded by default."""
    # Create a result with one zero-hit AOI
    aoi_results: dict[str | int, AOIResult] = {
        "shelf_A": AOIResult(aoi_id="shelf_A", hit_count=10, total_attention_seconds=10.0, hit_timestamps=list(range(10))),
        "shelf_B": AOIResult(aoi_id="shelf_B", hit_count=0, total_attention_seconds=0.0, hit_timestamps=[]),
    }
    result = TrackingResult(
        aoi_results=aoi_results,
        total_samples=15,
        samples_with_hits=10,
        samples_no_winner=5,
    )
    
    distribution = result.get_attention_distribution()
    
    assert "shelf_A" in distribution
    assert "shelf_B" not in distribution
    assert distribution["shelf_A"] == 100.0


def test_attention_distribution_includes_no_hits_when_requested(tracking_result_basic: TrackingResult) -> None:
    """Test that zero-hit AOIs are included when include_no_hits=True."""
    # Create a result with one zero-hit AOI
    aoi_results: dict[str | int, AOIResult] = {
        "shelf_A": AOIResult(aoi_id="shelf_A", hit_count=10, total_attention_seconds=10.0, hit_timestamps=list(range(10))),
        "shelf_B": AOIResult(aoi_id="shelf_B", hit_count=0, total_attention_seconds=0.0, hit_timestamps=[]),
    }
    result = TrackingResult(
        aoi_results=aoi_results,
        total_samples=15,
        samples_with_hits=10,
        samples_no_winner=5,
    )
    
    distribution = result.get_attention_distribution(include_no_hits=True)
    
    assert "shelf_A" in distribution
    assert "shelf_B" in distribution
    assert distribution["shelf_A"] == 100.0
    assert distribution["shelf_B"] == 0.0


def test_attention_distribution_correct_percentages(tracking_result_basic: TrackingResult) -> None:
    """Test that percentages are correctly calculated."""
    distribution = tracking_result_basic.get_attention_distribution()
    
    # Total hits = 17 (10 + 5 + 2)
    # shelf_A: 10/17 * 100 ≈ 58.82%
    # shelf_B: 5/17 * 100 ≈ 29.41%
    # shelf_C: 2/17 * 100 ≈ 11.76%
    assert abs(distribution["shelf_A"] - 58.82352941176471) < 0.01
    assert abs(distribution["shelf_B"] - 29.411764705882355) < 0.01
    assert abs(distribution["shelf_C"] - 11.764705882352942) < 0.01


def test_attention_distribution_no_hits_returns_empty(tracking_result_no_hits: TrackingResult) -> None:
    """Test that empty distribution is returned when no hits."""
    distribution = tracking_result_no_hits.get_attention_distribution()
    
    assert distribution == {}


def test_attention_distribution_no_hits_with_include_flag(tracking_result_no_hits: TrackingResult) -> None:
    """Test that all zeros returned when no hits and include_no_hits=True."""
    distribution = tracking_result_no_hits.get_attention_distribution(include_no_hits=True)
    
    assert len(distribution) == 2
    assert distribution["shelf_A"] == 0.0
    assert distribution["shelf_B"] == 0.0


# =============================================================================
# Tests: get_viewing_timeline()
# =============================================================================


def test_viewing_timeline_order(tracking_result_basic: TrackingResult) -> None:
    """Test that timeline is in chronological order."""
    timeline = tracking_result_basic.get_viewing_timeline()
    
    # Check length matches total samples
    assert len(timeline) == 20
    
    # Check that indices are sequential
    for i, (idx, _) in enumerate(timeline):
        assert idx == i


def test_viewing_timeline_includes_none(tracking_result_basic: TrackingResult) -> None:
    """Test that gaps (no winner) are recorded as None."""
    timeline = tracking_result_basic.get_viewing_timeline()
    
    # Check that we have 3 None entries (samples_no_winner = 3)
    none_count = sum(1 for _, aoi_id in timeline if aoi_id is None)
    assert none_count == 3


def test_viewing_timeline_correct_mapping(tracking_result_basic: TrackingResult) -> None:
    """Test that AOI IDs are correctly mapped to sample indices."""
    timeline = tracking_result_basic.get_viewing_timeline()
    
    # shelf_A hits at indices 0-9
    for i in range(10):
        assert timeline[i][1] == "shelf_A"
    
    # shelf_B hits at indices 10-14
    for i in range(10, 15):
        assert timeline[i][1] == "shelf_B"
    
    # shelf_C hits at indices 15-16
    assert timeline[15][1] == "shelf_C"
    assert timeline[16][1] == "shelf_C"
    
    # Indices 17-19 should be None (no winner)
    for i in range(17, 20):
        assert timeline[i][1] is None


def test_viewing_timeline_empty_result() -> None:
    """Test timeline with zero samples."""
    result = TrackingResult(
        aoi_results={},
        total_samples=0,
        samples_with_hits=0,
        samples_no_winner=0,
    )
    
    timeline = result.get_viewing_timeline()
    assert timeline == []


# =============================================================================
# Tests: to_dataframe()
# =============================================================================


def test_to_dataframe_columns(tracking_result_basic: TrackingResult) -> None:
    """Test that DataFrame has correct columns."""
    pytest.importorskip("pandas")  # Skip if pandas not installed
    
    df = tracking_result_basic.to_dataframe()
    
    expected_columns = ["aoi_id", "hit_count", "total_attention_seconds", "attention_percentage"]
    assert list(df.columns) == expected_columns


def test_to_dataframe_sorted_by_hit_count(tracking_result_basic: TrackingResult) -> None:
    """Test that DataFrame is sorted by hit count descending."""
    pytest.importorskip("pandas")
    
    df = tracking_result_basic.to_dataframe()
    
    assert len(df) == 3
    assert df.iloc[0]["aoi_id"] == "shelf_A"
    assert df.iloc[0]["hit_count"] == 10
    assert df.iloc[1]["aoi_id"] == "shelf_B"
    assert df.iloc[1]["hit_count"] == 5
    assert df.iloc[2]["aoi_id"] == "shelf_C"
    assert df.iloc[2]["hit_count"] == 2


def test_to_dataframe_percentages_match(tracking_result_basic: TrackingResult) -> None:
    """Test that DataFrame percentages match get_attention_distribution()."""
    pytest.importorskip("pandas")
    
    df = tracking_result_basic.to_dataframe()
    distribution = tracking_result_basic.get_attention_distribution(include_no_hits=True)
    
    for _, row in df.iterrows():
        aoi_id = row["aoi_id"]
        assert abs(row["attention_percentage"] - distribution[aoi_id]) < 0.01


def test_to_dataframe_no_pandas_raises_import_error(tracking_result_basic: TrackingResult, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that ImportError is raised if pandas is not available."""
    # Mock pandas import to fail
    import sys
    monkeypatch.setitem(sys.modules, "pandas", None)
    
    with pytest.raises(ImportError, match="pandas is required"):
        tracking_result_basic.to_dataframe()


def test_to_dataframe_empty_result() -> None:
    """Test DataFrame creation with empty result."""
    pytest.importorskip("pandas")
    
    result = TrackingResult(
        aoi_results={},
        total_samples=0,
        samples_with_hits=0,
        samples_no_winner=0,
    )
    
    df = result.to_dataframe()
    assert len(df) == 0
    assert list(df.columns) == ["aoi_id", "hit_count", "total_attention_seconds", "attention_percentage"]


def test_to_dataframe_with_zero_hits(tracking_result_no_hits: TrackingResult) -> None:
    """Test DataFrame includes AOIs with zero hits."""
    pytest.importorskip("pandas")
    
    df = tracking_result_no_hits.to_dataframe()
    
    assert len(df) == 2
    assert all(df["hit_count"] == 0)
    assert all(df["attention_percentage"] == 0.0)


# =============================================================================
# Tests: Tally Validation (High Priority)
# =============================================================================


def test_tracking_result_rejects_mismatched_hit_count_sum() -> None:
    """Test that ValidationError is raised when sum of hit_counts != samples_with_hits."""
    aoi_results: dict[str | int, AOIResult] = {
        "shelf_A": AOIResult(aoi_id="shelf_A", hit_count=10, total_attention_seconds=10.0, hit_timestamps=list(range(10))),
        "shelf_B": AOIResult(aoi_id="shelf_B", hit_count=5, total_attention_seconds=5.0, hit_timestamps=list(range(10, 15))),
    }
    
    # Sum of hit_counts = 15, but samples_with_hits = 14 (mismatch)
    with pytest.raises(ValidationError, match="Sum of AOI hit counts .* must equal samples_with_hits"):
        TrackingResult(
            aoi_results=aoi_results,
            total_samples=20,
            samples_with_hits=14,  # Wrong! Should be 15
            samples_no_winner=6,   # Wrong! Should be 5
        )


def test_tracking_result_rejects_mismatched_timestamps_count() -> None:
    """Test that ValidationError is raised when sum of hit_timestamps lengths != samples_with_hits."""
    aoi_results: dict[str | int, AOIResult] = {
        "shelf_A": AOIResult(aoi_id="shelf_A", hit_count=10, total_attention_seconds=10.0, hit_timestamps=list(range(10))),
        # This result has 6 timestamps but hit_count=5 - inconsistent!
        "shelf_B": AOIResult(aoi_id="shelf_B", hit_count=5, total_attention_seconds=5.0, hit_timestamps=list(range(10, 16))),
    }
    
    with pytest.raises(ValidationError, match="Sum of AOI hit_timestamps lengths .* must equal samples_with_hits"):
        TrackingResult(
            aoi_results=aoi_results,
            total_samples=20,
            samples_with_hits=15,
            samples_no_winner=5,
        )


def test_tracking_result_rejects_aoi_timestamp_count_mismatch() -> None:
    """Test that ValidationError is raised when an AOI's hit_count != len(hit_timestamps)."""
    aoi_results: dict[str | int, AOIResult] = {
        "shelf_A": AOIResult(aoi_id="shelf_A", hit_count=10, total_attention_seconds=10.0, hit_timestamps=list(range(10))),
        # This AOI says hit_count=6 but has only 5 timestamps! (hit_count = 10+6=16, timestamps = 10+5=15)
        "shelf_B": AOIResult(aoi_id="shelf_B", hit_count=6, total_attention_seconds=6.0, hit_timestamps=list(range(10, 15))),
    }
    
    # The validation will catch this either at the sum level or the per-AOI level
    with pytest.raises(ValidationError, match="(Sum of AOI hit_timestamps lengths|AOI 'shelf_B' hit_timestamps length)"):
        TrackingResult(
            aoi_results=aoi_results,
            total_samples=20,
            samples_with_hits=16,  # Matches sum of hit_counts (10+6), but not sum of timestamps (10+5)
            samples_no_winner=4,
        )


def test_tracking_result_accepts_consistent_tallies() -> None:
    """Test that TrackingResult is created successfully when all tallies are consistent."""
    aoi_results: dict[str | int, AOIResult] = {
        "shelf_A": AOIResult(aoi_id="shelf_A", hit_count=10, total_attention_seconds=10.0, hit_timestamps=list(range(10))),
        "shelf_B": AOIResult(aoi_id="shelf_B", hit_count=5, total_attention_seconds=5.0, hit_timestamps=list(range(10, 15))),
    }
    
    # All tallies are consistent: 10 + 5 = 15 samples_with_hits, 15 + 5 = 20 total
    result = TrackingResult(
        aoi_results=aoi_results,
        total_samples=20,
        samples_with_hits=15,
        samples_no_winner=5,
    )
    
    assert result.samples_with_hits == 15
    assert result.get_total_hits() == 15


# =============================================================================
# Tests: Timeline Index Validation (Medium Priority)
# =============================================================================


def test_viewing_timeline_rejects_negative_index() -> None:
    """Test that get_viewing_timeline() raises ValidationError for negative hit_timestamps."""
    aoi_results: dict[str | int, AOIResult] = {
        "shelf_A": AOIResult(aoi_id="shelf_A", hit_count=2, total_attention_seconds=2.0, hit_timestamps=[0, -1]),  # -1 is invalid
    }
    
    result = TrackingResult(
        aoi_results=aoi_results,
        total_samples=10,
        samples_with_hits=2,
        samples_no_winner=8,
    )
    
    with pytest.raises(ValidationError, match="AOI 'shelf_A' has negative hit_timestamp: -1"):
        result.get_viewing_timeline()


def test_viewing_timeline_rejects_out_of_bounds_index() -> None:
    """Test that get_viewing_timeline() raises ValidationError for hit_timestamps >= total_samples."""
    aoi_results: dict[str | int, AOIResult] = {
        "shelf_A": AOIResult(aoi_id="shelf_A", hit_count=2, total_attention_seconds=2.0, hit_timestamps=[0, 10]),  # 10 >= total_samples
    }
    
    result = TrackingResult(
        aoi_results=aoi_results,
        total_samples=10,
        samples_with_hits=2,
        samples_no_winner=8,
    )
    
    with pytest.raises(ValidationError, match=r"AOI 'shelf_A' has hit_timestamp \(10\) >= total_samples \(10\)"):
        result.get_viewing_timeline()


def test_viewing_timeline_rejects_non_integer_index() -> None:
    """Test that get_viewing_timeline() raises ValidationError for non-integer hit_timestamps."""
    aoi_results: dict[str | int, AOIResult] = {
        "shelf_A": AOIResult(aoi_id="shelf_A", hit_count=2, total_attention_seconds=2.0, hit_timestamps=[0, 1.5]),  # 1.5 is not an int # type: ignore
    }
    
    result = TrackingResult(
        aoi_results=aoi_results,
        total_samples=10,
        samples_with_hits=2,
        samples_no_winner=8,
    )
    
    with pytest.raises(ValidationError, match="AOI 'shelf_A' has non-integer hit_timestamp: 1.5"):
        result.get_viewing_timeline()


def test_viewing_timeline_accepts_valid_indices() -> None:
    """Test that get_viewing_timeline() works correctly with valid indices."""
    aoi_results: dict[str | int, AOIResult] = {
        "shelf_A": AOIResult(aoi_id="shelf_A", hit_count=3, total_attention_seconds=3.0, hit_timestamps=[0, 5, 9]),
    }
    
    result = TrackingResult(
        aoi_results=aoi_results,
        total_samples=10,
        samples_with_hits=3,
        samples_no_winner=7,
    )
    
    timeline = result.get_viewing_timeline()
    
    assert len(timeline) == 10
    assert timeline[0][1] == "shelf_A"
    assert timeline[5][1] == "shelf_A"
    assert timeline[9][1] == "shelf_A"
    assert timeline[1][1] is None
    assert timeline[4][1] is None


# =============================================================================
# Tests: Session Statistics (Step 3.2)
# =============================================================================


def test_coverage_ratio_full_coverage() -> None:
    """Test coverage_ratio when every sample has a hit."""
    aoi_results: dict[str | int, AOIResult] = {
        "shelf_A": AOIResult(
            aoi_id="shelf_A",
            hit_count=10,
            total_attention_seconds=10.0,
            hit_timestamps=list(range(10)),
        ),
    }
    
    result = TrackingResult(
        aoi_results=aoi_results,
        total_samples=10,
        samples_with_hits=10,
        samples_no_winner=0,
    )
    
    assert result.coverage_ratio == 1.0


def test_coverage_ratio_no_coverage() -> None:
    """Test coverage_ratio when no AOI has any hits."""
    aoi_results: dict[str | int, AOIResult] = {
        "shelf_A": AOIResult(aoi_id="shelf_A", hit_count=0, total_attention_seconds=0.0, hit_timestamps=[]),
    }
    
    result = TrackingResult(
        aoi_results=aoi_results,
        total_samples=10,
        samples_with_hits=0,
        samples_no_winner=10,
    )
    
    assert result.coverage_ratio == 0.0


def test_coverage_ratio_partial() -> None:
    """Test coverage_ratio for typical partial coverage case."""
    aoi_results: dict[str | int, AOIResult] = {
        "shelf_A": AOIResult(
            aoi_id="shelf_A",
            hit_count=7,
            total_attention_seconds=7.0,
            hit_timestamps=[0, 1, 2, 3, 4, 5, 6],
        ),
    }
    
    result = TrackingResult(
        aoi_results=aoi_results,
        total_samples=20,
        samples_with_hits=7,
        samples_no_winner=13,
    )
    
    assert result.coverage_ratio == 0.35


def test_coverage_ratio_zero_samples() -> None:
    """Test coverage_ratio handles zero total_samples gracefully."""
    aoi_results: dict[str | int, AOIResult] = {}
    
    result = TrackingResult(
        aoi_results=aoi_results,
        total_samples=0,
        samples_with_hits=0,
        samples_no_winner=0,
    )
    
    assert result.coverage_ratio == 0.0


def test_dominant_aoi_clear_winner() -> None:
    """Test dominant_aoi when one AOI has clearly the most hits."""
    aoi_results: dict[str | int, AOIResult] = {
        "shelf_A": AOIResult(
            aoi_id="shelf_A",
            hit_count=10,
            total_attention_seconds=10.0,
            hit_timestamps=list(range(10)),
        ),
        "shelf_B": AOIResult(
            aoi_id="shelf_B",
            hit_count=5,
            total_attention_seconds=5.0,
            hit_timestamps=list(range(10, 15)),
        ),
        "shelf_C": AOIResult(
            aoi_id="shelf_C",
            hit_count=2,
            total_attention_seconds=2.0,
            hit_timestamps=[15, 16],
        ),
    }
    
    result = TrackingResult(
        aoi_results=aoi_results,
        total_samples=20,
        samples_with_hits=17,
        samples_no_winner=3,
    )
    
    assert result.dominant_aoi == "shelf_A"


def test_dominant_aoi_tie() -> None:
    """Test dominant_aoi returns None when multiple AOIs tie for most hits."""
    aoi_results: dict[str | int, AOIResult] = {
        "shelf_A": AOIResult(
            aoi_id="shelf_A",
            hit_count=5,
            total_attention_seconds=5.0,
            hit_timestamps=[0, 1, 2, 3, 4],
        ),
        "shelf_B": AOIResult(
            aoi_id="shelf_B",
            hit_count=5,
            total_attention_seconds=5.0,
            hit_timestamps=[5, 6, 7, 8, 9],
        ),
        "shelf_C": AOIResult(
            aoi_id="shelf_C",
            hit_count=3,
            total_attention_seconds=3.0,
            hit_timestamps=[10, 11, 12],
        ),
    }
    
    result = TrackingResult(
        aoi_results=aoi_results,
        total_samples=15,
        samples_with_hits=13,
        samples_no_winner=2,
    )
    
    assert result.dominant_aoi is None


def test_dominant_aoi_no_hits() -> None:
    """Test dominant_aoi returns None when no AOI has any hits."""
    aoi_results: dict[str | int, AOIResult] = {
        "shelf_A": AOIResult(aoi_id="shelf_A", hit_count=0, total_attention_seconds=0.0, hit_timestamps=[]),
        "shelf_B": AOIResult(aoi_id="shelf_B", hit_count=0, total_attention_seconds=0.0, hit_timestamps=[]),
    }
    
    result = TrackingResult(
        aoi_results=aoi_results,
        total_samples=10,
        samples_with_hits=0,
        samples_no_winner=10,
    )
    
    assert result.dominant_aoi is None


def test_dominant_aoi_single_aoi_with_hits() -> None:
    """Test dominant_aoi returns the single AOI when only one has hits."""
    aoi_results: dict[str | int, AOIResult] = {
        "shelf_A": AOIResult(
            aoi_id="shelf_A",
            hit_count=7,
            total_attention_seconds=7.0,
            hit_timestamps=[0, 1, 2, 3, 4, 5, 6],
        ),
        "shelf_B": AOIResult(aoi_id="shelf_B", hit_count=0, total_attention_seconds=0.0, hit_timestamps=[]),
    }
    
    result = TrackingResult(
        aoi_results=aoi_results,
        total_samples=10,
        samples_with_hits=7,
        samples_no_winner=3,
    )
    
    assert result.dominant_aoi == "shelf_A"


def test_engagement_score_full_concentration() -> None:
    """Test engagement_score when all attention is on a single AOI (score = 1.0)."""
    aoi_results: dict[str | int, AOIResult] = {
        "shelf_A": AOIResult(
            aoi_id="shelf_A",
            hit_count=10,
            total_attention_seconds=10.0,
            hit_timestamps=list(range(10)),
        ),
        "shelf_B": AOIResult(aoi_id="shelf_B", hit_count=0, total_attention_seconds=0.0, hit_timestamps=[]),
    }
    
    result = TrackingResult(
        aoi_results=aoi_results,
        total_samples=15,
        samples_with_hits=10,
        samples_no_winner=5,
    )
    
    assert result.engagement_score == 1.0


def test_engagement_score_even_distribution() -> None:
    """Test engagement_score when attention is evenly distributed across AOIs."""
    # With 3 AOIs each getting 1/3 of attention, HHI = 3 * (1/3)^2 = 3/9 = 1/3
    aoi_results: dict[str | int, AOIResult] = {
        "shelf_A": AOIResult(
            aoi_id="shelf_A",
            hit_count=3,
            total_attention_seconds=3.0,
            hit_timestamps=[0, 1, 2],
        ),
        "shelf_B": AOIResult(
            aoi_id="shelf_B",
            hit_count=3,
            total_attention_seconds=3.0,
            hit_timestamps=[3, 4, 5],
        ),
        "shelf_C": AOIResult(
            aoi_id="shelf_C",
            hit_count=3,
            total_attention_seconds=3.0,
            hit_timestamps=[6, 7, 8],
        ),
    }
    
    result = TrackingResult(
        aoi_results=aoi_results,
        total_samples=12,
        samples_with_hits=9,
        samples_no_winner=3,
    )
    
    expected_score = 1.0 / 3.0  # Each AOI has 1/3 share, HHI = 3 * (1/3)^2
    assert abs(result.engagement_score - expected_score) < 1e-10


def test_engagement_score_no_hits() -> None:
    """Test engagement_score returns 0.0 when no hits are recorded."""
    aoi_results: dict[str | int, AOIResult] = {
        "shelf_A": AOIResult(aoi_id="shelf_A", hit_count=0, total_attention_seconds=0.0, hit_timestamps=[]),
        "shelf_B": AOIResult(aoi_id="shelf_B", hit_count=0, total_attention_seconds=0.0, hit_timestamps=[]),
    }
    
    result = TrackingResult(
        aoi_results=aoi_results,
        total_samples=10,
        samples_with_hits=0,
        samples_no_winner=10,
    )
    
    assert result.engagement_score == 0.0


def test_engagement_score_moderate_concentration() -> None:
    """Test engagement_score for moderate concentration (one AOI dominates but not completely)."""
    # shelf_A: 7/10 = 0.7, shelf_B: 3/10 = 0.3
    # HHI = 0.7^2 + 0.3^2 = 0.49 + 0.09 = 0.58
    aoi_results: dict[str | int, AOIResult] = {
        "shelf_A": AOIResult(
            aoi_id="shelf_A",
            hit_count=7,
            total_attention_seconds=7.0,
            hit_timestamps=[0, 1, 2, 3, 4, 5, 6],
        ),
        "shelf_B": AOIResult(
            aoi_id="shelf_B",
            hit_count=3,
            total_attention_seconds=3.0,
            hit_timestamps=[7, 8, 9],
        ),
    }
    
    result = TrackingResult(
        aoi_results=aoi_results,
        total_samples=15,
        samples_with_hits=10,
        samples_no_winner=5,
    )
    
    expected_score = 0.7**2 + 0.3**2  # 0.49 + 0.09 = 0.58
    assert abs(result.engagement_score - expected_score) < 1e-10


def test_session_duration_calculation() -> None:
    """Test session_duration calculates total time correctly (total_samples × 1s)."""
    aoi_results: dict[str | int, AOIResult] = {
        "shelf_A": AOIResult(
            aoi_id="shelf_A",
            hit_count=10,
            total_attention_seconds=10.0,
            hit_timestamps=list(range(10)),
        ),
    }
    
    result = TrackingResult(
        aoi_results=aoi_results,
        total_samples=60,
        samples_with_hits=10,
        samples_no_winner=50,
    )
    
    # 60 samples × 1 second = 60 seconds
    assert result.session_duration == 60.0


def test_session_duration_zero_samples() -> None:
    """Test session_duration returns 0.0 for zero samples."""
    aoi_results: dict[str | int, AOIResult] = {}
    
    result = TrackingResult(
        aoi_results=aoi_results,
        total_samples=0,
        samples_with_hits=0,
        samples_no_winner=0,
    )
    
    assert result.session_duration == 0.0


def test_session_duration_single_sample() -> None:
    """Test session_duration for single sample is 1 second."""
    aoi_results: dict[str | int, AOIResult] = {
        "shelf_A": AOIResult(
            aoi_id="shelf_A",
            hit_count=1,
            total_attention_seconds=1.0,
            hit_timestamps=[0],
        ),
    }
    
    result = TrackingResult(
        aoi_results=aoi_results,
        total_samples=1,
        samples_with_hits=1,
        samples_no_winner=0,
    )
    
    assert result.session_duration == 1.0
