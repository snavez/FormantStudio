"""
Tests for reuse of the plot layout.

Rebuilding the axes tears down every plot and builds fresh ones, which costs
more than everything else a TextGrid load does put together. The render items
all update in place, so a layout that already matches the data can simply be
drawn into. These pin down when that reuse is allowed and, more importantly,
when it is not: a layout that no longer matches the data must be rebuilt.
"""

import os
import sys

import numpy as np
import parselmouth
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication

from formant_editor import Interval, SpectrogramCanvas, TextGrid, Tier


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


def _grid(names, dur=2.0):
    return TextGrid(0, dur, [
        Tier(n, "IntervalTier", 0, dur,
             intervals=[Interval(0, dur / 2, "a"), Interval(dur / 2, dur, "b")])
        for n in names])


@pytest.fixture
def canvas(qapp):
    c = SpectrogramCanvas()
    sr = 16000
    c.sound = parselmouth.Sound(
        np.random.default_rng(0).standard_normal(sr * 2) * 0.05,
        sampling_frequency=sr)
    c.total_duration = c.sound.duration
    c.view_start, c.view_end = 0.0, c.sound.duration
    c._compute_spectrogram()
    c.textgrid_data = _grid(["words", "phones"])
    c._setup_axes()
    yield c


class TestLayoutIsReused:
    """The same layout must not be torn down and rebuilt."""

    def test_a_grid_with_the_same_tiers_keeps_the_plots(self, canvas):
        """Walking a corpus: every file asks for the same layout."""
        before = list(canvas._tier_plots)
        canvas.textgrid_data = _grid(["words", "phones"])
        canvas._setup_axes()
        assert canvas._tier_plots == before

    def test_calling_it_twice_over_changes_nothing(self, canvas):
        before = list(canvas._tier_plots)
        canvas._setup_axes()
        canvas._setup_axes()
        assert canvas._tier_plots == before

    def test_force_rebuilds_anyway(self, canvas):
        before = list(canvas._tier_plots)
        canvas._setup_axes(force=True)
        assert canvas._tier_plots != before


class TestLayoutIsRebuiltWhenItMustBe:
    """Every way the layout can stop matching the data."""

    def test_a_different_tier_count(self, canvas):
        before = list(canvas._tier_plots)
        canvas.textgrid_data = _grid(["words", "phones", "syllables"])
        canvas._setup_axes()
        assert canvas._tier_plots != before
        assert len(canvas._tier_plots) == 3

    def test_a_renamed_tier(self, canvas):
        """The left axis is sized to the widest name."""
        before = list(canvas._tier_plots)
        canvas.textgrid_data = _grid(["words", "a_much_longer_tier_name"])
        canvas._setup_axes()
        assert canvas._tier_plots != before

    def test_reordered_tiers(self, canvas):
        before = list(canvas._tier_plots)
        canvas.textgrid_data = _grid(["phones", "words"])
        canvas._setup_axes()
        assert canvas._tier_plots != before

    def test_hiding_a_tier(self, canvas):
        before = list(canvas._tier_plots)
        canvas.hidden_tiers = {0}
        canvas._setup_axes()
        assert canvas._tier_plots != before
        assert len(canvas._tier_plots) == 1

    def test_losing_the_textgrid(self, canvas):
        canvas.textgrid_data = None
        canvas._setup_axes()
        assert canvas._tier_plots == []

    def test_gaining_audio(self, qapp):
        c = SpectrogramCanvas()
        c.textgrid_data = _grid(["words"])
        c._setup_axes()
        assert c._wave_plot is None
        sr = 16000
        c.sound = parselmouth.Sound(
            np.random.default_rng(1).standard_normal(sr) * 0.05,
            sampling_frequency=sr)
        c.total_duration = c.sound.duration
        c.view_start, c.view_end = 0.0, c.sound.duration
        c._compute_spectrogram()
        c._setup_axes()
        assert c._wave_plot is not None


class TestReusedPlotsStillDrawTheNewData:
    """Reuse is only safe because the render items update in place."""

    def test_the_new_labels_are_what_gets_drawn(self, canvas):
        canvas.render()
        tg = _grid(["words", "phones"])
        tg.tiers[0].intervals[0].text = "CHANGED"
        canvas.textgrid_data = tg
        canvas._setup_axes()
        canvas.render()
        pool = canvas._tier_text_pools[canvas._tier_plots[0]]
        shown = [t._fs_text for t in pool if t.isVisible()]
        assert "CHANGED" in shown

    def test_fewer_labels_hides_the_leftovers(self, canvas):
        """A pooled TextItem from the previous file must not linger."""
        canvas.render()
        dur = canvas.total_duration
        sparse = TextGrid(0, dur, [
            Tier("words", "IntervalTier", 0, dur,
                 intervals=[Interval(0, dur, "only")]),
            Tier("phones", "IntervalTier", 0, dur,
                 intervals=[Interval(0, dur, "one")])])
        canvas.textgrid_data = sparse
        canvas._setup_axes()
        canvas.render()
        pool = canvas._tier_text_pools[canvas._tier_plots[0]]
        shown = [t._fs_text for t in pool if t.isVisible()]
        assert shown == ["only"]
