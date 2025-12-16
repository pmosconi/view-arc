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
import math
from pathlib import Path
from typing import cast

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
from view_arc.tracking.visualize import (
    create_tracking_animation,
    draw_attention_heatmap,
    draw_attention_labels,
    draw_viewing_timeline,
)

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


@pytest.fixture
def tracking_result_timeline() -> TrackingResult:
    """Create a compact tracking result for timeline visualizations."""
    return TrackingResult(
        aoi_results={
            "a": AOIResult(
                aoi_id="a",
                hit_count=2,
                total_attention_seconds=2.0,
                hit_timestamps=[1, 2],
            ),
            "b": AOIResult(
                aoi_id="b",
                hit_count=2,
                total_attention_seconds=2.0,
                hit_timestamps=[4, 5],
            ),
        },
        total_samples=6,
        samples_with_hits=4,
        samples_no_winner=2,
    )


def _expected_cursor_column(width: int, total_samples: int, processed_samples: int) -> int:
    """Replicate timeline boundary rounding to predict cursor placement."""

    if width <= 0 or total_samples <= 0:
        raise ValueError("width and total_samples must be positive")

    processed = max(1, min(processed_samples, total_samples))
    px_per_sample = width / total_samples
    start = int(round((processed - 1) * px_per_sample))
    end = int(round(processed * px_per_sample))
    if end <= start:
        end = start + 1
    start = max(0, min(start, width - 1))
    end = max(start + 1, min(end, width))
    return min(end - 1, width - 1)


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


class TestDrawViewingTimeline:
    """Tests for the timeline visualization helper."""

    def test_draw_viewing_timeline_basic(
        self, tracking_result_timeline: TrackingResult
    ) -> None:
        """Timeline renders with AOI colors and expected dimensions."""
        image = draw_viewing_timeline(
            tracking_result_timeline,
            width=240,
            height=140,
            show_legend=False,
        )

        assert image.shape == (140, 240, 3)
        unique_colors = np.unique(image.reshape(-1, 3), axis=0)
        assert unique_colors.shape[0] > 1  # timeline should not be monochrome

        output_path = OUTPUT_DIR / "test_timeline_basic.png"
        cv2.imwrite(str(output_path), image)

    def test_draw_viewing_timeline_gaps(
        self, tracking_result_timeline: TrackingResult
    ) -> None:
        """Gap color appears for samples with no AOI winner."""
        gap_color = (5, 5, 5)
        image = draw_viewing_timeline(
            tracking_result_timeline,
            width=240,
            height=140,
            show_legend=False,
            gap_color=gap_color,
        )

        gap_mask = np.all(image == np.array(gap_color, dtype=np.uint8), axis=-1)
        assert gap_mask.any(), "Expected at least one gap segment colored with gap_color"

        output_path = OUTPUT_DIR / "test_timeline_gaps.png"
        cv2.imwrite(str(output_path), image)

    def test_draw_viewing_timeline_legend(
        self, tracking_result_all_zero: TrackingResult
    ) -> None:
        """Legend shows AOI colors even when the timeline has no hits."""
        custom_colors = cast(
            dict[str | int, tuple[int, int, int]],
            {
                "shelf_top": (10, 60, 200),
                "shelf_middle": (40, 200, 60),
                "shelf_bottom": (200, 80, 80),
            },
        )
        image = draw_viewing_timeline(
            tracking_result_all_zero,
            width=260,
            height=200,
            aoi_colors=custom_colors,
            show_legend=True,
        )

        for color in custom_colors.values():
            color_mask = np.all(image == np.array(color, dtype=np.uint8), axis=-1)
            assert color_mask.any(), f"Legend is missing color {color}"

        output_path = OUTPUT_DIR / "test_timeline_legend.png"
        cv2.imwrite(str(output_path), image)


class TestCreateTrackingAnimation:
    """Tests for animation frame generation."""

    def test_create_tracking_animation_frames(
        self, tracking_result_timeline: TrackingResult
    ) -> None:
        """Animation frames progress from partial to full timeline."""
        frames = create_tracking_animation(
            tracking_result_timeline,
            width=240,
            height=150,
            samples_per_frame=2,
            show_legend=False,
        )

        expected_frames = math.ceil(tracking_result_timeline.total_samples / 2)
        assert len(frames) == expected_frames
        assert frames[0].shape == (150, 240, 3)
        assert not np.array_equal(frames[0], frames[-1])

        first_path = OUTPUT_DIR / "test_animation_frame_0.png"
        last_path = OUTPUT_DIR / "test_animation_frame_last.png"
        cv2.imwrite(str(first_path), frames[0])
        cv2.imwrite(str(last_path), frames[-1])

    def test_create_tracking_animation_cursor_alignment(
        self, tracking_result_timeline: TrackingResult
    ) -> None:
        """Cursor column aligns with the last processed sample boundary."""
        width = 120
        cursor_color = (0, 0, 0)
        frames_with_cursor = create_tracking_animation(
            tracking_result_timeline,
            width=width,
            height=120,
            samples_per_frame=1,
            show_legend=False,
            annotate_progress=False,
            cursor_color=cursor_color,
        )
        frames_without_cursor = create_tracking_animation(
            tracking_result_timeline,
            width=width,
            height=120,
            samples_per_frame=1,
            show_legend=False,
            annotate_progress=False,
            cursor_color=cursor_color,
            draw_cursor=False,
        )

        first_frame = frames_with_cursor[0].astype(np.int16)
        baseline = frames_without_cursor[0].astype(np.int16)
        diff = np.abs(first_frame - baseline)
        diff_mask = np.any(diff > 5, axis=2)
        cursor_columns = np.where(diff_mask.any(axis=0))[0]
        assert cursor_columns.size > 0, "Cursor line should appear in the first frame"

        expected_column = _expected_cursor_column(width, tracking_result_timeline.total_samples, 1)
        assert np.any(cursor_columns == expected_column)

    def test_create_tracking_animation_future_color_segments(
        self, tracking_result_timeline: TrackingResult
    ) -> None:
        """Unprocessed samples are tinted with the future_color shade."""
        width = 120
        future_color = (12, 34, 56)
        frames = create_tracking_animation(
            tracking_result_timeline,
            width=width,
            height=120,
            samples_per_frame=2,
            show_legend=False,
            annotate_progress=False,
            future_color=future_color,
        )

        first_frame = frames[0]
        future_mask = np.all(first_frame == np.array(future_color, dtype=np.uint8), axis=2)
        future_columns = np.where(future_mask.any(axis=0))[0]
        assert future_columns.size > 0, "Expected future_color shading for unprocessed samples"

        processed_boundary = _expected_cursor_column(width, tracking_result_timeline.total_samples, 2)
        assert future_columns.min() > processed_boundary


class TestSessionReplay:
    """Tests for session replay visualization (Step 5.3)."""

    def test_draw_session_frame_components(
        self, blank_image: np.ndarray, sample_aois: list[AOI]
    ) -> None:
        """Verify all components are present in a session frame.

        This test checks that draw_session_frame includes:
        - Current viewer position
        - Current view arc
        - Current winner highlighted
        - Running hit counts
        """
        from view_arc.tracking.dataclasses import ViewerSample
        from view_arc.tracking.visualize import draw_session_frame

        # Create a viewer sample
        sample = ViewerSample(position=(400.0, 300.0), direction=(0.0, -1.0))

        # Simulate running hit counts
        running_hit_counts: dict[str | int, int] = {"shelf_top": 5, "shelf_middle": 3, "shelf_bottom": 0}

        # Draw frame with shelf_top as winner
        frame = draw_session_frame(
            image=blank_image,
            sample=sample,
            aois=sample_aois,
            winner_id="shelf_top",
            running_hit_counts=running_hit_counts,
            field_of_view_deg=90.0,
            max_range=200.0,
            sample_index=7,
            total_samples=10,
            show_hit_counts=True,
            show_progress=True,
        )

        # Verify frame was created
        assert frame.shape == blank_image.shape
        assert frame.dtype == np.uint8

        # Verify frame is not blank (has been modified)
        assert not np.array_equal(frame, blank_image)

        # Verify winner color is present (shelf_top should be highlighted in blue)
        # Winner color is (0, 0, 255) BGR = blue
        winner_color = np.array([0, 0, 255], dtype=np.uint8)
        # Check for winner color in the shelf_top region (y: 50-150, x: 100-700)
        shelf_top_region = frame[50:150, 100:700, :]
        winner_pixels = np.all(shelf_top_region == winner_color, axis=2)
        assert np.any(winner_pixels), "Winner color not found in shelf_top region"

        # Verify progress text is present
        # Progress should be "Sample 8/10" (sample_index + 1)
        # Check top-left area for dark (text background) pixels
        progress_region = frame[0:40, 0:200, :]
        dark_pixels = np.all(progress_region < 50, axis=2)
        assert np.any(dark_pixels), "Progress text background not found"

        # Save for visual inspection
        output_path = OUTPUT_DIR / "test_session_frame_components.png"
        cv2.imwrite(str(output_path), frame)

    def test_draw_session_frame_no_winner(
        self, blank_image: np.ndarray, sample_aois: list[AOI]
    ) -> None:
        """Verify frame handles case with no winner correctly."""
        from view_arc.tracking.dataclasses import ViewerSample
        from view_arc.tracking.visualize import draw_session_frame

        sample = ViewerSample(position=(50.0, 50.0), direction=(1.0, 0.0))
        running_hit_counts: dict[str | int, int] = {"shelf_top": 0, "shelf_middle": 0, "shelf_bottom": 0}

        # Draw frame with no winner
        frame = draw_session_frame(
            image=blank_image,
            sample=sample,
            aois=sample_aois,
            winner_id=None,
            running_hit_counts=running_hit_counts,
            field_of_view_deg=90.0,
            max_range=200.0,
        )

        assert frame.shape == blank_image.shape
        assert not np.array_equal(frame, blank_image)

        output_path = OUTPUT_DIR / "test_session_frame_no_winner.png"
        cv2.imwrite(str(output_path), frame)

    def test_draw_session_frame_minimal_options(
        self, blank_image: np.ndarray, sample_aois: list[AOI]
    ) -> None:
        """Verify frame can be drawn with minimal options."""
        from view_arc.tracking.dataclasses import ViewerSample
        from view_arc.tracking.visualize import draw_session_frame

        sample = ViewerSample(position=(400.0, 300.0), direction=(0.0, -1.0))
        running_hit_counts: dict[str | int, int] = {"shelf_top": 10, "shelf_middle": 5, "shelf_bottom": 2}

        # Draw frame without progress and hit counts
        frame = draw_session_frame(
            image=blank_image,
            sample=sample,
            aois=sample_aois,
            winner_id="shelf_middle",
            running_hit_counts=running_hit_counts,
            show_hit_counts=False,
            show_progress=False,
        )

        assert frame.shape == blank_image.shape
        assert not np.array_equal(frame, blank_image)

        output_path = OUTPUT_DIR / "test_session_frame_minimal.png"
        cv2.imwrite(str(output_path), frame)

    def test_generate_session_replay_frame_count(
        self, blank_image: np.ndarray, sample_aois: list[AOI]
    ) -> None:
        """Verify correct number of frames are generated."""
        from view_arc.tracking.dataclasses import ViewerSample
        from view_arc.tracking.visualize import generate_session_replay

        # Create a sequence of samples
        samples = [
            ViewerSample(position=(400.0, 300.0), direction=(0.0, -1.0)),
            ViewerSample(position=(400.0, 300.0), direction=(0.0, -1.0)),
            ViewerSample(position=(400.0, 300.0), direction=(0.0, -1.0)),
            ViewerSample(position=(350.0, 300.0), direction=(-0.707, -0.707)),
            ViewerSample(position=(350.0, 300.0), direction=(-0.707, -0.707)),
        ]

        # Corresponding winners
        winner_ids: list[str | int | None] = [
            "shelf_top",
            "shelf_top",
            "shelf_top",
            "shelf_middle",
            "shelf_middle",
        ]

        # Generate replay
        frames = generate_session_replay(
            image=blank_image,
            samples=samples,
            aois=sample_aois,
            winner_ids=winner_ids,
            field_of_view_deg=90.0,
            max_range=200.0,
        )

        # Verify frame count matches sample count
        assert len(frames) == len(samples)
        assert len(frames) == 5

        # Verify all frames have correct shape
        for frame in frames:
            assert frame.shape == blank_image.shape
            assert frame.dtype == np.uint8

        # Verify frames are different (progression)
        assert not np.array_equal(frames[0], frames[-1])

        # Save first and last frames for visual inspection
        first_path = OUTPUT_DIR / "test_session_replay_frame_0.png"
        last_path = OUTPUT_DIR / "test_session_replay_frame_4.png"
        cv2.imwrite(str(first_path), frames[0])
        cv2.imwrite(str(last_path), frames[-1])

    def test_generate_session_replay_empty_samples(
        self, blank_image: np.ndarray, sample_aois: list[AOI]
    ) -> None:
        """Verify empty sample list returns empty frame list."""
        from view_arc.tracking.visualize import generate_session_replay

        frames = generate_session_replay(
            image=blank_image, samples=[], aois=sample_aois, winner_ids=[]
        )

        assert frames == []

    def test_generate_session_replay_length_mismatch(
        self, blank_image: np.ndarray, sample_aois: list[AOI]
    ) -> None:
        """Verify error when samples and winner_ids have different lengths."""
        from view_arc.tracking.dataclasses import ViewerSample
        from view_arc.tracking.visualize import generate_session_replay

        samples = [
            ViewerSample(position=(400.0, 300.0), direction=(0.0, -1.0)),
            ViewerSample(position=(400.0, 300.0), direction=(0.0, -1.0)),
        ]
        winner_ids: list[str | int | None] = ["shelf_top"]  # Mismatched length

        with pytest.raises(ValueError, match="must have the same length"):
            generate_session_replay(
                image=blank_image, samples=samples, aois=sample_aois, winner_ids=winner_ids
            )

    def test_generate_session_replay_running_counts_accuracy(
        self, blank_image: np.ndarray, sample_aois: list[AOI]
    ) -> None:
        """Verify running hit counts are correctly accumulated across frames."""
        from view_arc.tracking.dataclasses import ViewerSample
        from view_arc.tracking.visualize import generate_session_replay

        # Create samples where shelf_top gets 3 hits, shelf_middle gets 2
        samples = [
            ViewerSample(position=(400.0, 200.0), direction=(0.0, -1.0)),  # shelf_top
            ViewerSample(position=(400.0, 200.0), direction=(0.0, -1.0)),  # shelf_top
            ViewerSample(position=(400.0, 400.0), direction=(0.0, -1.0)),  # shelf_middle
            ViewerSample(position=(400.0, 200.0), direction=(0.0, -1.0)),  # shelf_top
            ViewerSample(position=(400.0, 400.0), direction=(0.0, -1.0)),  # shelf_middle
        ]

        winner_ids: list[str | int | None] = [
            "shelf_top",
            "shelf_top",
            "shelf_middle",
            "shelf_top",
            "shelf_middle",
        ]

        frames = generate_session_replay(
            image=blank_image,
            samples=samples,
            aois=sample_aois,
            winner_ids=winner_ids,
        )

        # We can't directly inspect the hit counts in the frames without
        # more sophisticated image analysis, but we can verify:
        # 1. All frames were generated
        assert len(frames) == 5

        # 2. Frames show progression (not all identical)
        for i in range(len(frames) - 1):
            assert not np.array_equal(frames[i], frames[i + 1])

        # Save middle and last frames for manual verification
        middle_path = OUTPUT_DIR / "test_session_replay_counts_frame_2.png"
        last_path = OUTPUT_DIR / "test_session_replay_counts_frame_4.png"
        cv2.imwrite(str(middle_path), frames[2])
        cv2.imwrite(str(last_path), frames[4])

    def test_generate_session_replay_mixed_id_types(
        self, blank_image: np.ndarray
    ) -> None:
        """Verify session replay works with mixed string and integer AOI IDs.
        
        Regression test for sorted() TypeError when AOI IDs mix strings and ints.
        """
        from view_arc.tracking.dataclasses import ViewerSample, AOI
        from view_arc.tracking.visualize import generate_session_replay

        # Create AOIs with mixed ID types (string and int)
        mixed_aois = [
            AOI(id="shelf_a", contour=np.array([[100, 100], [200, 100], [200, 200], [100, 200]], dtype=np.float32)),
            AOI(id=1, contour=np.array([[300, 100], [400, 100], [400, 200], [300, 200]], dtype=np.float32)),
            AOI(id="shelf_b", contour=np.array([[100, 300], [200, 300], [200, 400], [100, 400]], dtype=np.float32)),
            AOI(id=2, contour=np.array([[300, 300], [400, 300], [400, 400], [300, 400]], dtype=np.float32)),
        ]

        samples = [
            ViewerSample(position=(150.0, 50.0), direction=(0.0, 1.0)),
            ViewerSample(position=(350.0, 50.0), direction=(0.0, 1.0)),
            ViewerSample(position=(150.0, 250.0), direction=(0.0, 1.0)),
            ViewerSample(position=(350.0, 250.0), direction=(0.0, 1.0)),
        ]

        # Mix string and int IDs in winner_ids
        winner_ids: list[str | int | None] = ["shelf_a", 1, "shelf_b", 2]

        # Should not raise TypeError when sorting for error messages
        frames = generate_session_replay(
            image=blank_image,
            samples=samples,
            aois=mixed_aois,
            winner_ids=winner_ids,
        )

        assert len(frames) == 4
        for frame in frames:
            assert frame.shape == blank_image.shape

    def test_draw_session_frame_mixed_id_validation_error(
        self, blank_image: np.ndarray
    ) -> None:
        """Verify validation error message works with mixed ID types.
        
        Regression test: ensure sorted() doesn't crash when displaying
        error message for mixed string/int AOI IDs.
        """
        from view_arc.tracking.dataclasses import ViewerSample, AOI
        from view_arc.tracking.visualize import draw_session_frame

        # Create AOIs with mixed ID types
        mixed_aois = [
            AOI(id="shelf_a", contour=np.array([[100, 100], [200, 100], [200, 200], [100, 200]], dtype=np.float32)),
            AOI(id=42, contour=np.array([[300, 100], [400, 100], [400, 200], [300, 200]], dtype=np.float32)),
        ]

        sample = ViewerSample(position=(250.0, 150.0), direction=(0.0, 1.0))
        running_hit_counts: dict[str | int, int] = {"shelf_a": 5, 42: 3}

        # Try to reference a missing AOI - should get clear error, not TypeError
        with pytest.raises(ValueError, match="winner_id 'missing' not found in aois"):
            draw_session_frame(
                image=blank_image,
                sample=sample,
                aois=mixed_aois,
                winner_id="missing",
                running_hit_counts=running_hit_counts,
            )

    def test_generate_session_replay_mixed_id_validation_error(
        self, blank_image: np.ndarray
    ) -> None:
        """Verify validation error message works with mixed ID types in batch.
        
        Regression test: ensure sorted() doesn't crash when displaying
        error message for mixed string/int AOI IDs in generate_session_replay.
        """
        from view_arc.tracking.dataclasses import ViewerSample, AOI
        from view_arc.tracking.visualize import generate_session_replay

        # Create AOIs with mixed ID types
        mixed_aois = [
            AOI(id="shelf_a", contour=np.array([[100, 100], [200, 100], [200, 200], [100, 200]], dtype=np.float32)),
            AOI(id=99, contour=np.array([[300, 100], [400, 100], [400, 200], [300, 200]], dtype=np.float32)),
        ]

        samples = [
            ViewerSample(position=(150.0, 50.0), direction=(0.0, 1.0)),
            ViewerSample(position=(350.0, 50.0), direction=(0.0, 1.0)),
        ]

        # Include missing IDs of both types
        winner_ids: list[str | int | None] = ["missing_str", 404]

        # Should get clear ValueError, not TypeError from sorting
        with pytest.raises(ValueError, match="Found 2 winner_id\\(s\\) not in aois"):
            generate_session_replay(
                image=blank_image,
                samples=samples,
                aois=mixed_aois,
                winner_ids=winner_ids,
            )

    def test_draw_session_frame_invalid_winner(
        self, blank_image: np.ndarray, sample_aois: list[AOI]
    ) -> None:
        """Verify error when winner_id doesn't exist in aois."""
        from view_arc.tracking.dataclasses import ViewerSample
        from view_arc.tracking.visualize import draw_session_frame

        sample = ViewerSample(position=(400.0, 300.0), direction=(0.0, -1.0))
        running_hit_counts: dict[str | int, int] = {"shelf_top": 5}

        # Try to draw frame with winner_id not in aois
        with pytest.raises(ValueError, match="winner_id 'unknown_shelf' not found in aois"):
            draw_session_frame(
                image=blank_image,
                sample=sample,
                aois=sample_aois,
                winner_id="unknown_shelf",
                running_hit_counts=running_hit_counts,
            )

    def test_generate_session_replay_invalid_winners(
        self, blank_image: np.ndarray, sample_aois: list[AOI]
    ) -> None:
        """Verify error when winner_ids contain IDs not in aois."""
        from view_arc.tracking.dataclasses import ViewerSample
        from view_arc.tracking.visualize import generate_session_replay

        samples = [
            ViewerSample(position=(400.0, 300.0), direction=(0.0, -1.0)),
            ViewerSample(position=(400.0, 300.0), direction=(0.0, -1.0)),
        ]

        # Include an invalid winner ID
        winner_ids: list[str | int | None] = ["shelf_top", "invalid_aoi"]

        with pytest.raises(
            ValueError, match="Found 1 winner_id\\(s\\) not in aois: \\[invalid_aoi\\]"
        ):
            generate_session_replay(
                image=blank_image, samples=samples, aois=sample_aois, winner_ids=winner_ids
            )

    def test_generate_session_replay_filtered_aois(
        self, blank_image: np.ndarray, sample_aois: list[AOI]
    ) -> None:
        """Verify error when AOI list is filtered but winner_ids reference filtered AOIs."""
        from view_arc.tracking.dataclasses import ViewerSample
        from view_arc.tracking.visualize import generate_session_replay

        samples = [
            ViewerSample(position=(400.0, 300.0), direction=(0.0, -1.0)),
            ViewerSample(position=(400.0, 300.0), direction=(0.0, -1.0)),
        ]

        # Winner IDs reference all shelves
        winner_ids: list[str | int | None] = ["shelf_top", "shelf_bottom"]

        # But only provide subset of AOIs (filtered list)
        filtered_aois = [sample_aois[0]]  # Only shelf_top

        # Should fail because shelf_bottom is not in filtered list
        with pytest.raises(
            ValueError, match="winner_id\\(s\\) not in aois.*shelf_bottom"
        ):
            generate_session_replay(
                image=blank_image,
                samples=samples,
                aois=filtered_aois,
                winner_ids=winner_ids,
            )

    def test_generate_session_replay_progress_indicators(
        self, blank_image: np.ndarray, sample_aois: list[AOI]
    ) -> None:
        """Verify progress indicators increment correctly across frames."""
        from view_arc.tracking.dataclasses import ViewerSample
        from view_arc.tracking.visualize import generate_session_replay

        samples = [
            ViewerSample(position=(400.0, 300.0), direction=(0.0, -1.0)),
            ViewerSample(position=(400.0, 300.0), direction=(0.0, -1.0)),
            ViewerSample(position=(400.0, 300.0), direction=(0.0, -1.0)),
        ]

        winner_ids: list[str | int | None] = ["shelf_top", "shelf_top", "shelf_top"]

        frames = generate_session_replay(
            image=blank_image,
            samples=samples,
            aois=sample_aois,
            winner_ids=winner_ids,
            show_progress=True,
        )

        # Check that frames differ in progress region (top-left)
        # Extract progress regions
        progress_region_1 = frames[0][0:40, 0:200, :]
        progress_region_2 = frames[1][0:40, 0:200, :]
        progress_region_3 = frames[2][0:40, 0:200, :]

        # Progress regions should be different across frames
        assert not np.array_equal(progress_region_1, progress_region_2)
        assert not np.array_equal(progress_region_2, progress_region_3)

