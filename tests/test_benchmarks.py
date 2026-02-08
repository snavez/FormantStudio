"""
Performance benchmarks for SpectrogramCanvas rendering pipeline.

Run with: py -3 -m pytest tests/test_benchmarks.py -v --tb=short
Each test reports its timing via print() and asserts a generous upper bound
to catch severe regressions. The bounds are intentionally loose — the goal
is to track trends, not enforce tight SLAs.

To see timing output: py -3 -m pytest tests/test_benchmarks.py -v -s
"""

import numpy as np
import time
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from formant_editor import (
    SpectrogramCanvas, MainWindow, FormantData, TextGrid, Tier, Interval,
    Point, extract_formants_from_praat,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def qapp():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


@pytest.fixture
def wav_path():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test.wav")
    if not os.path.exists(path):
        pytest.skip("test.wav not found")
    return path


@pytest.fixture
def large_textgrid():
    """TextGrid with many tiers and intervals for stress testing."""
    tiers = []
    for t in range(5):
        intervals = []
        n_intervals = 200
        for i in range(n_intervals):
            xmin = i * 0.05
            xmax = (i + 1) * 0.05
            intervals.append(Interval(xmin, xmax, f"seg{i}"))
        tiers.append(Tier(f"tier{t}", "IntervalTier", 0, 10.0,
                          intervals=intervals))
    return TextGrid(0, 10.0, tiers)


def _measure(fn, label, n_iterations=1):
    """Run fn n_iterations times, return (mean_ms, total_ms)."""
    times = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    mean_ms = np.mean(times)
    total_ms = sum(times)
    print(f"  {label}: {mean_ms:.1f} ms (mean of {n_iterations})")
    return mean_ms, total_ms


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

class TestRenderBenchmarks:
    """Benchmark the full render() pipeline and its components."""

    def test_bench_full_render(self, qapp, wav_path):
        """Full render() call — the main metric users feel."""
        c = SpectrogramCanvas()
        c.load_sound(wav_path)
        import parselmouth
        c.formant_data = extract_formants_from_praat(parselmouth.Sound(wav_path))

        # Warm up
        c.render()

        mean_ms, _ = _measure(c.render, "full render (no TextGrid)", n_iterations=10)
        assert mean_ms < 500, f"Full render took {mean_ms:.0f}ms, expected <500ms"

    def test_bench_render_with_textgrid(self, qapp, wav_path):
        """Full render with a TextGrid loaded."""
        c = SpectrogramCanvas()
        c.load_sound(wav_path)
        import parselmouth
        c.formant_data = extract_formants_from_praat(parselmouth.Sound(wav_path))
        tg = TextGrid(0, c.total_duration, [
            Tier("words", "IntervalTier", 0, c.total_duration,
                 intervals=[Interval(0, c.total_duration / 2, "a"),
                            Interval(c.total_duration / 2, c.total_duration, "b")]),
            Tier("phones", "IntervalTier", 0, c.total_duration,
                 intervals=[Interval(i * 0.1, (i + 1) * 0.1, f"p{i}")
                            for i in range(int(c.total_duration / 0.1))]),
        ])
        c.textgrid_data = tg
        c._setup_axes()
        c.render()

        mean_ms, _ = _measure(c.render, "full render (2-tier TextGrid)", n_iterations=10)
        assert mean_ms < 500, f"Render with TextGrid took {mean_ms:.0f}ms, expected <500ms"

    def test_bench_render_large_textgrid(self, qapp, wav_path, large_textgrid):
        """Render with a stress-test TextGrid (5 tiers, 200 intervals each)."""
        c = SpectrogramCanvas()
        c.load_sound(wav_path)
        c.textgrid_data = large_textgrid
        c._setup_axes()
        c.render()

        mean_ms, _ = _measure(c.render, "full render (5-tier, 200 intervals)", n_iterations=5)
        assert mean_ms < 1000, f"Large TextGrid render took {mean_ms:.0f}ms, expected <1000ms"

    def test_bench_spectrogram_only(self, qapp, wav_path):
        """Benchmark just the spectrogram drawing."""
        c = SpectrogramCanvas()
        c.load_sound(wav_path)
        c._setup_axes()
        c._spec_plot.setXRange(c.view_start, c.view_end, padding=0)
        c._spec_plot.setYRange(0, c.max_freq, padding=0)

        mean_ms, _ = _measure(c._draw_spectrogram, "spectrogram draw", n_iterations=20)
        assert mean_ms < 200, f"Spectrogram draw took {mean_ms:.0f}ms, expected <200ms"

    def test_bench_waveform_only(self, qapp, wav_path):
        """Benchmark just the waveform drawing."""
        c = SpectrogramCanvas()
        c.load_sound(wav_path)
        c._setup_axes()
        c._spec_plot.setXRange(c.view_start, c.view_end, padding=0)

        mean_ms, _ = _measure(c._draw_waveform, "waveform draw", n_iterations=20)
        assert mean_ms < 100, f"Waveform draw took {mean_ms:.0f}ms, expected <100ms"

    def test_bench_formant_draw(self, qapp, wav_path):
        """Benchmark formant overlay drawing."""
        c = SpectrogramCanvas()
        c.load_sound(wav_path)
        import parselmouth
        c.formant_data = extract_formants_from_praat(parselmouth.Sound(wav_path))
        c._setup_axes()
        c._spec_plot.setXRange(c.view_start, c.view_end, padding=0)
        c._spec_plot.setYRange(0, c.max_freq, padding=0)

        mean_ms, _ = _measure(c._draw_formants, "formant draw", n_iterations=20)
        assert mean_ms < 200, f"Formant draw took {mean_ms:.0f}ms, expected <200ms"


class TestZoomBenchmarks:
    """Benchmark zoom + render cycles (what users feel during scroll)."""

    def test_bench_zoom_cycle(self, qapp, wav_path):
        """Zoom in then render — simulates one scroll wheel notch."""
        c = SpectrogramCanvas()
        c.load_sound(wav_path)
        import parselmouth
        c.formant_data = extract_formants_from_praat(parselmouth.Sound(wav_path))
        c.render()

        def zoom_cycle():
            c.zoom(1.0 / 1.3)
            c.render()

        mean_ms, _ = _measure(zoom_cycle, "zoom-in + render", n_iterations=10)
        assert mean_ms < 500, f"Zoom cycle took {mean_ms:.0f}ms, expected <500ms"

    def test_bench_pan_cycle(self, qapp, wav_path):
        """Pan right then render — simulates scrollbar drag."""
        c = SpectrogramCanvas()
        c.load_sound(wav_path)
        import parselmouth
        c.formant_data = extract_formants_from_praat(parselmouth.Sound(wav_path))
        c.zoom(0.3)
        c.render()

        shift = c.view_width * 0.1

        def pan_cycle():
            c.set_view(c.view_start + shift, c.view_end + shift)
            c.render()

        mean_ms, _ = _measure(pan_cycle, "pan + render", n_iterations=10)
        assert mean_ms < 500, f"Pan cycle took {mean_ms:.0f}ms, expected <500ms"


class TestSetupBenchmarks:
    """Benchmark setup operations."""

    def test_bench_load_sound(self, qapp, wav_path):
        """Benchmark load_sound (spectrogram computation)."""
        c = SpectrogramCanvas()
        mean_ms, _ = _measure(lambda: c.load_sound(wav_path), "load_sound", n_iterations=3)
        assert mean_ms < 5000, f"load_sound took {mean_ms:.0f}ms, expected <5000ms"

    def test_bench_setup_axes(self, qapp, wav_path):
        """Benchmark _setup_axes layout creation."""
        c = SpectrogramCanvas()
        c.load_sound(wav_path)
        tg = TextGrid(0, c.total_duration, [
            Tier("t1", "IntervalTier", 0, c.total_duration,
                 intervals=[Interval(0, c.total_duration, "")]),
            Tier("t2", "IntervalTier", 0, c.total_duration,
                 intervals=[Interval(0, c.total_duration, "")]),
        ])
        c.textgrid_data = tg

        mean_ms, _ = _measure(c._setup_axes, "setup_axes (2 tiers)", n_iterations=5)
        assert mean_ms < 500, f"_setup_axes took {mean_ms:.0f}ms, expected <500ms"

    def test_bench_formant_analysis(self, qapp, wav_path):
        """Benchmark Praat formant analysis."""
        import parselmouth
        sound = parselmouth.Sound(wav_path)

        def analyse():
            extract_formants_from_praat(sound)

        mean_ms, _ = _measure(analyse, "formant analysis", n_iterations=3)
        assert mean_ms < 10000, f"Formant analysis took {mean_ms:.0f}ms, expected <10000ms"


class TestOverlayBenchmarks:
    """Benchmark selection overlay and crosshair updates."""

    def test_bench_selection_overlay(self, qapp, wav_path):
        """Benchmark _draw_selection_overlay with selections active."""
        c = SpectrogramCanvas()
        c.load_sound(wav_path)
        tg = TextGrid(0, c.total_duration, [
            Tier("t1", "IntervalTier", 0, c.total_duration,
                 intervals=[Interval(0, 0.5, "a"), Interval(0.5, c.total_duration, "b")]),
        ])
        c.textgrid_data = tg
        c._setup_axes()
        c.render()
        c._selection_start = 0.1
        c._selection_end = 0.3
        c._selected_boundary = (0, 0.5)

        mean_ms, _ = _measure(c._draw_selection_overlay, "selection overlay", n_iterations=20)
        assert mean_ms < 50, f"Selection overlay took {mean_ms:.0f}ms, expected <50ms"

    def test_bench_crosshair_update(self, qapp, wav_path):
        """Benchmark crosshair position update (should be near-instant)."""
        c = SpectrogramCanvas()
        c.load_sound(wav_path)
        c.render()

        x_positions = np.linspace(c.view_start, c.view_end, 100)
        idx = [0]

        def update_crosshair():
            x = x_positions[idx[0] % len(x_positions)]
            c._update_crosshair(x, 1000.0, on_spectrogram=True)
            idx[0] += 1

        mean_ms, _ = _measure(update_crosshair, "crosshair update", n_iterations=100)
        assert mean_ms < 5, f"Crosshair update took {mean_ms:.1f}ms, expected <5ms"
