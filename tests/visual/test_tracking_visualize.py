"""
Visual tests for attention tracking visualization functions.

These tests verify that the heatmap and label drawing functions produce
correct visual output. Output images are saved to tests/visual/output/
for manual inspection.

**REQUIREMENTS**: These tests require OpenCV (cv2) to run. The dependency
is included in the [dev] extras (opencv-python-headless) and must be
installed for CI/test environments. If cv2 is missing, tests will FAIL
rather than skip to prevent regressions from slipping through unnoticed.
"""

import os
from pathlib import Path

import numpy as np
import pytest

# Import cv2 directly - fail loudly if missing rather than silently skipping
try:
    import cv2
except ImportError as e:
    pytest.fail(
        f"OpenCV (cv2) is required for visualization tests but is not installed.\n"
        f"Install with: pip install opencv-python-headless\n"
        f"Or install all dev dependencies: pip install -e .[dev]\n"
        f"Original error: {e}",
        pytrace=False,
    )

from view_arc.tracking.dataclasses import AOI, AOIResult, TrackingResult
from view_arc.tracking.visualize import draw_attention_heatmap, draw_attention_labels

# Output directory for visual test results
OUTPUT_DIR = Path(__file__).parent / "output"


@pytest.fixture(scope="module", autouse=True)
def setup_output_dir() -> None:
    """Create output directory for test images."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture
def blank_image() -> np.ndarray:
    """Create a blank white image for testing."""
    return np.ones((600, 800, 3), dtype=np.uint8) * 255


@pytest.fixture
def sample_aois() -> list[AOI]:
    """Create sample AOIs representing store shelves."""
    # Three rectangular shelves arranged vertically
    aois = [
        AOI(
            id="shelf_top",
            contour=np.array(
                [[100, 50], [700, 50], [700, 150], [100, 150]], dtype=np.float32
            ),
        ),
        AOI(
            id="shelf_middle",
            contour=np.array(
                [[100, 250], [700, 250], [700, 350], [100, 350]], dtype=np.float32
            ),
        ),
        AOI(
            id="shelf_bottom",
            contour=np.array(
                [[100, 450], [700, 450], [700, 550], [100, 550]], dtype=np.float32
            ),
        ),
    ]
    return aois


@pytest.fixture
def tracking_result_varied() -> TrackingResult:
    """Create a tracking result with varied hit counts."""
    return TrackingResult(
        aoi_results={
            "shelf_top": AOIResult(
                aoi_id="shelf_top",
                hit_count=45,
                total_attention_seconds=45.0,
                hit_timestamps=list(range(45)),
            ),
            "shelf_middle": AOIResult(
                aoi_id="shelf_middle",
                hit_count=15,
                total_attention_seconds=15.0,
                hit_timestamps=list(range(45, 60)),
            ),
            "shelf_bottom": AOIResult(
                aoi_id="shelf_bottom",
                hit_count=0,
                total_attention_seconds=0.0,
                hit_timestamps=[],
            ),
        },
        total_samples=100,
        samples_with_hits=60,
        samples_no_winner=40,
    )


@pytest.fixture
def tracking_result_all_zero() -> TrackingResult:
    """Create a tracking result where no AOI has hits."""
    return TrackingResult(
        aoi_results={
            "shelf_top": AOIResult(
                aoi_id="shelf_top",
                hit_count=0,
                total_attention_seconds=0.0,
                hit_timestamps=[],
            ),
            "shelf_middle": AOIResult(
                aoi_id="shelf_middle",
                hit_count=0,
                total_attention_seconds=0.0,
                hit_timestamps=[],
            ),
            "shelf_bottom": AOIResult(
                aoi_id="shelf_bottom",
                hit_count=0,
                total_attention_seconds=0.0,
                hit_timestamps=[],
            ),
        },
        total_samples=100,
        samples_with_hits=0,
        samples_no_winner=100,
    )


class TestDrawAttentionHeatmap:
    """Test suite for draw_attention_heatmap function."""

    def test_draw_attention_heatmap_basic(
        self, blank_image: np.ndarray, sample_aois: list[AOI], tracking_result_varied: TrackingResult
    ) -> None:
        """Test basic heatmap drawing with varied hit counts."""
        # Draw heatmap
        result_img = draw_attention_heatmap(blank_image, sample_aois, tracking_result_varied)

        # Verify image was modified
        assert not np.array_equal(result_img, blank_image)

        # Verify output shape is same as input
        assert result_img.shape == blank_image.shape

        # Save output for visual inspection
        output_path = OUTPUT_DIR / "test_heatmap_basic.png"
        cv2.imwrite(str(output_path), result_img)
        assert output_path.exists()

    def test_draw_attention_heatmap_color_scale(
        self, blank_image: np.ndarray, sample_aois: list[AOI], tracking_result_varied: TrackingResult
    ) -> None:
        """Test that colors vary with hit counts (hot colormap)."""
        # Draw heatmap
        result_img = draw_attention_heatmap(
            blank_image, sample_aois, tracking_result_varied, colormap="hot"
        )

        # Extract average color from each AOI region
        colors = []
        for aoi in sample_aois:
            # Create mask for this AOI
            mask = np.zeros(blank_image.shape[:2], dtype=np.uint8)
            pts = aoi.contour.astype(np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255)  # type: ignore[arg-type, call-overload]

            # Get average color in masked region
            avg_color = cv2.mean(result_img, mask=mask)[:3]  # BGR
            colors.append(avg_color)

        # Verify that shelf_top (45 hits) has different color than shelf_middle (15 hits)
        # which should be different from shelf_bottom (0 hits)
        color_top = np.array(colors[0])
        color_middle = np.array(colors[1])
        color_bottom = np.array(colors[2])

        # Colors should be distinct
        assert not np.allclose(color_top, color_middle, atol=10)
        assert not np.allclose(color_middle, color_bottom, atol=10)

        # Save output for visual inspection
        output_path = OUTPUT_DIR / "test_heatmap_color_scale.png"
        cv2.imwrite(str(output_path), result_img)

    def test_draw_attention_heatmap_zero_hits(
        self, blank_image: np.ndarray, sample_aois: list[AOI], tracking_result_all_zero: TrackingResult
    ) -> None:
        """Test heatmap when all AOIs have zero hits."""
        # Draw heatmap without background color - should return mostly unchanged image
        result_img_no_bg = draw_attention_heatmap(
            blank_image, sample_aois, tracking_result_all_zero, background_color=None
        )

        # Image should be mostly unchanged (original is white)
        # There might be slight differences due to blending, but should be close
        assert np.allclose(result_img_no_bg, blank_image, atol=5)

        # Draw heatmap with background color - AOIs should be visible
        result_img_with_bg = draw_attention_heatmap(
            blank_image,
            sample_aois,
            tracking_result_all_zero,
            background_color=(200, 200, 200),  # Light gray
        )

        # Image should be modified
        assert not np.array_equal(result_img_with_bg, blank_image)

        # Save outputs for visual inspection
        output_path_no_bg = OUTPUT_DIR / "test_heatmap_zero_hits_no_background.png"
        cv2.imwrite(str(output_path_no_bg), result_img_no_bg)

        output_path_with_bg = OUTPUT_DIR / "test_heatmap_zero_hits_with_background.png"
        cv2.imwrite(str(output_path_with_bg), result_img_with_bg)

    def test_draw_attention_heatmap_viridis_colormap(
        self, blank_image: np.ndarray, sample_aois: list[AOI], tracking_result_varied: TrackingResult
    ) -> None:
        """Test heatmap with viridis colormap."""
        result_img = draw_attention_heatmap(
            blank_image, sample_aois, tracking_result_varied, colormap="viridis"
        )

        # Verify image was modified
        assert not np.array_equal(result_img, blank_image)

        # Save output for visual inspection
        output_path = OUTPUT_DIR / "test_heatmap_viridis.png"
        cv2.imwrite(str(output_path), result_img)

    def test_draw_attention_heatmap_no_outlines(
        self, blank_image: np.ndarray, sample_aois: list[AOI], tracking_result_varied: TrackingResult
    ) -> None:
        """Test heatmap without AOI outlines."""
        result_img = draw_attention_heatmap(
            blank_image, sample_aois, tracking_result_varied, draw_outlines=False
        )

        # Verify image was modified
        assert not np.array_equal(result_img, blank_image)

        # Save output for visual inspection
        output_path = OUTPUT_DIR / "test_heatmap_no_outlines.png"
        cv2.imwrite(str(output_path), result_img)

    def test_draw_attention_heatmap_alpha_variations(
        self, blank_image: np.ndarray, sample_aois: list[AOI], tracking_result_varied: TrackingResult
    ) -> None:
        """Test heatmap with different alpha transparency levels."""
        alphas = [0.3, 0.5, 0.8]

        for alpha in alphas:
            result_img = draw_attention_heatmap(
                blank_image, sample_aois, tracking_result_varied, fill_alpha=alpha
            )

            # Verify image was modified
            assert not np.array_equal(result_img, blank_image)

            # Save output for visual inspection
            output_path = OUTPUT_DIR / f"test_heatmap_alpha_{alpha:.1f}.png"
            cv2.imwrite(str(output_path), result_img)

    def test_draw_attention_heatmap_mismatched_aois(
        self, blank_image: np.ndarray
    ) -> None:
        """Test heatmap when AOI list doesn't match tracking result (HIGH priority fix).
        
        This test verifies the fix for the KeyError bug when visualizing filtered
        or extended AOI lists that don't match the tracking result dictionary.
        """
        # Create AOIs
        aois = [
            AOI(
                id="shelf_a",
                contour=np.array([[100, 50], [700, 50], [700, 150], [100, 150]], dtype=np.float32),
            ),
            AOI(
                id="shelf_b",
                contour=np.array([[100, 250], [700, 250], [700, 350], [100, 350]], dtype=np.float32),
            ),
            AOI(
                id="shelf_c",  # This AOI is NOT in the tracking result
                contour=np.array([[100, 450], [700, 450], [700, 550], [100, 550]], dtype=np.float32),
            ),
        ]
        
        # Create result that only has data for shelf_a and shelf_b (shelf_c is missing)
        result = TrackingResult(
            aoi_results={
                "shelf_a": AOIResult(
                    aoi_id="shelf_a",
                    hit_count=30,
                    total_attention_seconds=30.0,
                    hit_timestamps=list(range(30)),
                ),
                "shelf_b": AOIResult(
                    aoi_id="shelf_b",
                    hit_count=10,
                    total_attention_seconds=10.0,
                    hit_timestamps=list(range(30, 40)),
                ),
                # shelf_c is intentionally missing
            },
            total_samples=60,
            samples_with_hits=40,
            samples_no_winner=20,
        )
        
        # This should NOT raise KeyError even though shelf_c is not in the result
        result_img = draw_attention_heatmap(
            blank_image,
            aois,
            result,
            colormap="hot",
            background_color=(230, 230, 230),
        )
        
        # Verify image was modified
        assert not np.array_equal(result_img, blank_image)
        
        # Save output for visual inspection
        output_path = OUTPUT_DIR / "test_heatmap_mismatched_aois.png"
        cv2.imwrite(str(output_path), result_img)


class TestDrawAttentionLabels:
    """Test suite for draw_attention_labels function."""

    def test_draw_attention_labels_positioning(
        self, blank_image: np.ndarray, sample_aois: list[AOI], tracking_result_varied: TrackingResult
    ) -> None:
        """Test that labels are drawn at correct positions."""
        result_img = draw_attention_labels(
            blank_image, sample_aois, tracking_result_varied, show_hit_count=True, show_percentage=True
        )

        # Verify image was modified
        assert not np.array_equal(result_img, blank_image)

        # Save output for visual inspection
        output_path = OUTPUT_DIR / "test_labels_positioning.png"
        cv2.imwrite(str(output_path), result_img)

    def test_draw_attention_labels_hit_count_only(
        self, blank_image: np.ndarray, sample_aois: list[AOI], tracking_result_varied: TrackingResult
    ) -> None:
        """Test labels showing only hit count."""
        result_img = draw_attention_labels(
            blank_image,
            sample_aois,
            tracking_result_varied,
            show_hit_count=True,
            show_percentage=False,
            show_seconds=False,
        )

        # Verify image was modified
        assert not np.array_equal(result_img, blank_image)

        # Save output for visual inspection
        output_path = OUTPUT_DIR / "test_labels_hit_count_only.png"
        cv2.imwrite(str(output_path), result_img)

    def test_draw_attention_labels_percentage_only(
        self, blank_image: np.ndarray, sample_aois: list[AOI], tracking_result_varied: TrackingResult
    ) -> None:
        """Test labels showing only percentage."""
        result_img = draw_attention_labels(
            blank_image,
            sample_aois,
            tracking_result_varied,
            show_hit_count=False,
            show_percentage=True,
            show_seconds=False,
        )

        # Verify image was modified
        assert not np.array_equal(result_img, blank_image)

        # Save output for visual inspection
        output_path = OUTPUT_DIR / "test_labels_percentage_only.png"
        cv2.imwrite(str(output_path), result_img)

    def test_draw_attention_labels_all_metrics(
        self, blank_image: np.ndarray, sample_aois: list[AOI], tracking_result_varied: TrackingResult
    ) -> None:
        """Test labels showing all metrics (hit count, percentage, seconds)."""
        result_img = draw_attention_labels(
            blank_image,
            sample_aois,
            tracking_result_varied,
            show_hit_count=True,
            show_percentage=True,
            show_seconds=True,
        )

        # Verify image was modified
        assert not np.array_equal(result_img, blank_image)

        # Save output for visual inspection
        output_path = OUTPUT_DIR / "test_labels_all_metrics.png"
        cv2.imwrite(str(output_path), result_img)

    def test_draw_attention_labels_zero_hits(
        self, blank_image: np.ndarray, sample_aois: list[AOI], tracking_result_all_zero: TrackingResult
    ) -> None:
        """Test labels when all AOIs have zero hits."""
        result_img = draw_attention_labels(blank_image, sample_aois, tracking_result_all_zero)

        # Image should be unchanged since no labels should be drawn
        assert np.array_equal(result_img, blank_image)

    def test_draw_attention_labels_skip_zero_aoi(
        self, blank_image: np.ndarray, sample_aois: list[AOI], tracking_result_varied: TrackingResult
    ) -> None:
        """Test that labels are not drawn for AOIs with zero hits."""
        result_img = draw_attention_labels(blank_image, sample_aois, tracking_result_varied)

        # shelf_bottom has 0 hits, so it should not have a label
        # Check that the bottom region is mostly unchanged
        bottom_region = result_img[450:550, 100:700]
        original_bottom = blank_image[450:550, 100:700]

        # Should be very similar (allowing for minor antialiasing differences)
        assert np.allclose(bottom_region, original_bottom, atol=5)

        # Save output for visual inspection
        output_path = OUTPUT_DIR / "test_labels_skip_zero.png"
        cv2.imwrite(str(output_path), result_img)


class TestCombinedVisualization:
    """Test suite for combined heatmap and label visualization."""

    def test_heatmap_with_labels(
        self, blank_image: np.ndarray, sample_aois: list[AOI], tracking_result_varied: TrackingResult
    ) -> None:
        """Test combining heatmap and labels in one visualization."""
        # First draw heatmap
        img_with_heatmap = draw_attention_heatmap(blank_image, sample_aois, tracking_result_varied)

        # Then add labels on top
        img_complete = draw_attention_labels(
            img_with_heatmap,
            sample_aois,
            tracking_result_varied,
            show_hit_count=True,
            show_percentage=True,
        )

        # Verify image was modified from original
        assert not np.array_equal(img_complete, blank_image)

        # Save output for visual inspection
        output_path = OUTPUT_DIR / "test_combined_heatmap_labels.png"
        cv2.imwrite(str(output_path), img_complete)
