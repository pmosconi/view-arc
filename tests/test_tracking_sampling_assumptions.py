"""
Tests for tracking sampling assumptions documentation (Step 2.4).

Tests cover:
- SAMPLING_ASSUMPTIONS constant is accessible and populated
- TrackingResult exposes assumptions property
- README.md contains the required invariant text
- Module docstring contains the required assumptions
"""

from pathlib import Path

import numpy as np
import pytest

from view_arc.tracking import (
    AOIResult,
    SAMPLING_ASSUMPTIONS,
    TrackingResult,
)


# =============================================================================
# Tests: SAMPLING_ASSUMPTIONS constant
# =============================================================================


class TestSamplingAssumptionsConstant:
    """Tests for the SAMPLING_ASSUMPTIONS constant."""

    def test_sampling_assumptions_is_tuple(self) -> None:
        """SAMPLING_ASSUMPTIONS should be a tuple."""
        assert isinstance(SAMPLING_ASSUMPTIONS, tuple)

    def test_sampling_assumptions_not_empty(self) -> None:
        """SAMPLING_ASSUMPTIONS should contain at least one assumption."""
        assert len(SAMPLING_ASSUMPTIONS) > 0

    def test_sampling_assumptions_all_strings(self) -> None:
        """All elements in SAMPLING_ASSUMPTIONS should be strings."""
        for assumption in SAMPLING_ASSUMPTIONS:
            assert isinstance(assumption, str), f"Expected string, got {type(assumption)}"

    def test_sampling_assumptions_contains_1hz_cadence(self) -> None:
        """SAMPLING_ASSUMPTIONS should document the 1 Hz cadence invariant."""
        assumptions_text = " ".join(SAMPLING_ASSUMPTIONS).lower()
        assert "1 hz" in assumptions_text or "1hz" in assumptions_text or "one sample per second" in assumptions_text

    def test_sampling_assumptions_contains_sorted_timestamps(self) -> None:
        """SAMPLING_ASSUMPTIONS should document the sorted timestamps invariant."""
        assumptions_text = " ".join(SAMPLING_ASSUMPTIONS).lower()
        assert "sorted" in assumptions_text or "monotonic" in assumptions_text

    def test_sampling_assumptions_contains_coordinate_space(self) -> None:
        """SAMPLING_ASSUMPTIONS should document the coordinate space invariant."""
        assumptions_text = " ".join(SAMPLING_ASSUMPTIONS).lower()
        assert "image" in assumptions_text or "coordinate" in assumptions_text

    def test_sampling_assumptions_contains_single_viewer(self) -> None:
        """SAMPLING_ASSUMPTIONS should document the single viewer invariant."""
        assumptions_text = " ".join(SAMPLING_ASSUMPTIONS).lower()
        assert "single viewer" in assumptions_text

    def test_sampling_assumptions_immutable(self) -> None:
        """SAMPLING_ASSUMPTIONS should be immutable (tuple, not list)."""
        # Tuples are immutable, so this test verifies the type
        assert type(SAMPLING_ASSUMPTIONS) is tuple  # noqa: E721
        # Attempting to modify should raise TypeError
        with pytest.raises(TypeError):
            SAMPLING_ASSUMPTIONS[0] = "modified"  # type: ignore[index]


# =============================================================================
# Tests: TrackingResult.assumptions property
# =============================================================================


class TestTrackingResultAssumptions:
    """Tests for the TrackingResult.assumptions property."""

    def test_tracking_result_has_assumptions_property(self) -> None:
        """TrackingResult should expose an assumptions property."""
        result = TrackingResult(
            aoi_results={},
            total_samples=0,
            samples_with_hits=0,
            samples_no_winner=0,
        )
        assert hasattr(result, "assumptions")

    def test_tracking_result_assumptions_returns_tuple(self) -> None:
        """TrackingResult.assumptions should return a tuple."""
        result = TrackingResult(
            aoi_results={},
            total_samples=0,
            samples_with_hits=0,
            samples_no_winner=0,
        )
        assert isinstance(result.assumptions, tuple)

    def test_tracking_result_assumptions_matches_constant(self) -> None:
        """TrackingResult.assumptions should return the SAMPLING_ASSUMPTIONS constant."""
        result = TrackingResult(
            aoi_results={},
            total_samples=0,
            samples_with_hits=0,
            samples_no_winner=0,
        )
        assert result.assumptions == SAMPLING_ASSUMPTIONS

    def test_tracking_result_assumptions_with_data(self) -> None:
        """TrackingResult.assumptions should work with populated result."""
        aoi_results: dict[str | int, AOIResult] = {
            "shelf_1": AOIResult(aoi_id="shelf_1", hit_count=10, total_attention_seconds=10.0),
            "shelf_2": AOIResult(aoi_id="shelf_2", hit_count=5, total_attention_seconds=5.0),
        }
        result = TrackingResult(
            aoi_results=aoi_results,
            total_samples=20,
            samples_with_hits=15,
            samples_no_winner=5,
        )
        assert result.assumptions == SAMPLING_ASSUMPTIONS
        # Verify assumptions doesn't interfere with other properties
        assert result.total_samples == 20
        assert result.coverage_ratio == 0.75


# =============================================================================
# Tests: Documentation invariant text (doc-test style checks)
# =============================================================================


class TestDocumentationInvariants:
    """Tests to ensure sampling assumptions are documented in README and module docstrings."""

    @pytest.fixture
    def project_root(self) -> Path:
        """Get the project root directory."""
        # tests/ is one level below project root
        return Path(__file__).parent.parent

    @pytest.fixture
    def readme_content(self, project_root: Path) -> str:
        """Read the README.md content."""
        readme_path = project_root / "README.md"
        assert readme_path.exists(), f"README.md not found at {readme_path}"
        return readme_path.read_text()

    @pytest.fixture
    def tracking_init_content(self, project_root: Path) -> str:
        """Read the tracking module __init__.py content."""
        init_path = project_root / "view_arc" / "tracking" / "__init__.py"
        assert init_path.exists(), f"__init__.py not found at {init_path}"
        return init_path.read_text()

    def test_readme_contains_tracking_assumptions_section(self, readme_content: str) -> None:
        """README should contain a 'Tracking Assumptions' section."""
        assert "## Tracking Assumptions" in readme_content, (
            "README.md must contain a '## Tracking Assumptions' section "
            "documenting the sampling invariants"
        )

    def test_readme_documents_1hz_cadence(self, readme_content: str) -> None:
        """README should document the 1 Hz sampling cadence invariant."""
        readme_lower = readme_content.lower()
        assert "1 hz" in readme_lower or "one second" in readme_lower, (
            "README.md must document the 1 Hz sampling cadence invariant"
        )

    def test_readme_documents_coordinate_space(self, readme_content: str) -> None:
        """README should document the coordinate space invariant."""
        readme_lower = readme_content.lower()
        assert "image-coordinate" in readme_lower or "coordinate space" in readme_lower, (
            "README.md must document the coordinate space invariant"
        )

    def test_readme_documents_no_runtime_validation(self, readme_content: str) -> None:
        """README should clarify that invariants are not re-validated at runtime."""
        readme_lower = readme_content.lower()
        # The README uses "do **not** re-validate" with markdown bold formatting
        assert (
            "not re-validate" in readme_lower
            or "do not re-validate" in readme_lower
            or "not re-validated" in readme_lower
            or "**not** re-validate" in readme_lower  # markdown bold format
        ), (
            "README.md must clarify that sampling invariants are NOT re-validated at runtime"
        )

    def test_tracking_init_documents_assumptions(self, tracking_init_content: str) -> None:
        """Tracking module __init__.py should document sampling assumptions."""
        assert "Assumptions:" in tracking_init_content or "assumptions" in tracking_init_content.lower(), (
            "view_arc/tracking/__init__.py must document sampling assumptions in its docstring"
        )

    def test_tracking_init_documents_1hz(self, tracking_init_content: str) -> None:
        """Tracking module __init__.py should document 1 Hz cadence."""
        content_lower = tracking_init_content.lower()
        assert "1 hz" in content_lower or "one sample per second" in content_lower, (
            "view_arc/tracking/__init__.py must document the 1 Hz sampling cadence"
        )


# =============================================================================
# Tests: Assumptions consistency across locations
# =============================================================================


class TestAssumptionsConsistency:
    """Tests to ensure assumptions are consistent across all documented locations."""

    @pytest.fixture
    def project_root(self) -> Path:
        """Get the project root directory."""
        return Path(__file__).parent.parent

    def test_all_sampling_assumptions_described_somewhere(self, project_root: Path) -> None:
        """Each SAMPLING_ASSUMPTION should appear in at least one doc location."""
        # Gather all documentation sources
        readme_path = project_root / "README.md"
        init_path = project_root / "view_arc" / "tracking" / "__init__.py"
        dataclasses_path = project_root / "view_arc" / "tracking" / "dataclasses.py"

        all_docs = ""
        for path in [readme_path, init_path, dataclasses_path]:
            if path.exists():
                all_docs += path.read_text().lower()

        # Check each assumption has key terms present in documentation
        key_terms_per_assumption = [
            ["1 hz", "1hz", "one sample per second", "fixed"],  # First assumption
            ["1 second", "one second", "viewing time"],  # Second assumption
            ["sorted", "monotonic", "timestamp"],  # Third assumption
            ["image", "coordinate", "fixed", "contour"],  # Fourth assumption
            ["single viewer", "one viewer", "single"],  # Fifth assumption
        ]

        for i, key_terms in enumerate(key_terms_per_assumption):
            found = any(term in all_docs for term in key_terms)
            assert found, (
                f"SAMPLING_ASSUMPTIONS[{i}] ('{SAMPLING_ASSUMPTIONS[i]}') "
                f"should have key terms {key_terms} documented somewhere"
            )
