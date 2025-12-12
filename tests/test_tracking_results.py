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
