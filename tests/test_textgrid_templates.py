"""
Tests for TextGrid duration fitting and template instantiation, plus the
save-path guard that stops one file's annotation being written over another's.
"""

import os
import sys

import pytest

# Ensure the project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from formant_editor import (
    Interval, MainWindow, Point, Tier, TextGrid, TIME_EPS, _new_tier,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _grid(duration=10.0):
    """A two-tier grid: three labelled words plus two marked points."""
    words = Tier("words", "IntervalTier", 0, duration, intervals=[
        Interval(0.0, 2.0, ""),
        Interval(2.0, 5.0, "tena"),
        Interval(5.0, 9.0, "koutou"),
        Interval(9.0, duration, ""),
    ])
    marks = Tier("marks", "TextTier", 0, duration, points=[
        Point(3.0, "a"),
        Point(9.5, "b"),
    ])
    return TextGrid(0, duration, [words, marks])


def _boundaries(tier):
    return [(iv.xmin, iv.xmax, iv.text) for iv in tier.intervals]


# ---------------------------------------------------------------------------
# _new_tier
# ---------------------------------------------------------------------------

def test_new_interval_tier_spans_whole_file():
    tier = _new_tier("phones", "IntervalTier", 4.0)
    assert tier.xmin == 0 and tier.xmax == 4.0
    assert _boundaries(tier) == [(0, 4.0, "")]


def test_new_point_tier_starts_empty():
    tier = _new_tier("events", "TextTier", 4.0)
    assert tier.points == []
    assert tier.xmax == 4.0


# ---------------------------------------------------------------------------
# items_beyond
# ---------------------------------------------------------------------------

def test_items_beyond_counts_only_fully_out_of_range():
    tg = _grid(10.0)
    # At 8 s: the [9, 10] interval and the point at 9.5 are lost; the
    # [5, 9] "koutou" interval straddles and is merely truncated.
    assert tg.items_beyond(8.0) == 2


def test_items_beyond_zero_when_grid_fits():
    assert _grid(10.0).items_beyond(10.0) == 0


def test_items_beyond_ignores_straddling_interval():
    tg = TextGrid(0, 10.0, [
        Tier("t", "IntervalTier", 0, 10.0,
             intervals=[Interval(0, 4.0, "a"), Interval(4.0, 10.0, "b")]),
    ])
    assert tg.items_beyond(8.0) == 0


# ---------------------------------------------------------------------------
# fit_to_duration
# ---------------------------------------------------------------------------

def test_fit_trims_to_shorter_audio():
    tg = _grid(10.0)
    tg.fit_to_duration(8.0)

    assert tg.xmax == 8.0
    words, marks = tg.tiers
    assert words.xmax == 8.0 and marks.xmax == 8.0
    # [9, 10] dropped, "koutou" truncated to the new end.
    assert _boundaries(words) == [
        (0.0, 2.0, ""), (2.0, 5.0, "tena"), (5.0, 8.0, "koutou"),
    ]
    assert [p.mark for p in marks.points] == ["a"]


def test_fit_extends_to_longer_audio():
    tg = _grid(10.0)
    tg.fit_to_duration(12.0)

    assert tg.xmax == 12.0
    words = tg.tiers[0]
    assert words.xmax == 12.0
    assert words.intervals[-1].xmax == 12.0
    # Nothing is lost when growing.
    assert [iv.text for iv in words.intervals] == ["", "tena", "koutou", ""]
    assert len(tg.tiers[1].points) == 2


def test_fit_keeps_interval_tier_non_empty():
    tg = TextGrid(0, 10.0, [
        Tier("t", "IntervalTier", 0, 10.0,
             intervals=[Interval(0, 5.0, "a"), Interval(5.0, 10.0, "b")]),
    ])
    tg.fit_to_duration(0.5)
    tier = tg.tiers[0]
    assert len(tier.intervals) == 1
    assert tier.intervals[0].xmin == 0 and tier.intervals[0].xmax == 0.5


def test_fit_drops_every_interval_but_still_leaves_one():
    tg = TextGrid(0, 10.0, [
        Tier("t", "IntervalTier", 0, 10.0,
             intervals=[Interval(4.0, 7.0, "a"), Interval(7.0, 10.0, "b")]),
    ])
    tg.fit_to_duration(2.0)
    assert _boundaries(tg.tiers[0]) == [(0, 2.0, "")]


def test_fit_is_idempotent():
    tg = _grid(10.0)
    tg.fit_to_duration(8.0)
    snapshot = TextGrid(tg.xmin, tg.xmax, [t.copy() for t in tg.tiers])
    tg.fit_to_duration(8.0)
    assert tg == snapshot


# ---------------------------------------------------------------------------
# from_template
# ---------------------------------------------------------------------------

def test_template_scales_boundaries_proportionally():
    tg = TextGrid.from_template(_grid(10.0), 5.0)

    assert tg.xmin == 0 and tg.xmax == 5.0
    words = tg.tiers[0]
    assert _boundaries(words) == [
        (0.0, 1.0, ""), (1.0, 2.5, "tena"), (2.5, 4.5, "koutou"),
        (4.5, 5.0, ""),
    ]
    assert [(p.time, p.mark) for p in tg.tiers[1].points] == [
        (1.5, "a"), (4.75, "b"),
    ]


def test_template_snaps_endpoints_exactly():
    # A scale factor that does not divide cleanly must still land on 0 and
    # the exact duration, so the grid's domain matches the sound's.
    tg = TextGrid.from_template(_grid(10.0), 7.3)
    words = tg.tiers[0]
    assert words.intervals[0].xmin == 0
    assert words.intervals[-1].xmax == 7.3
    assert tg.xmax == 7.3


def test_template_preserves_labels_and_tier_structure():
    tg = TextGrid.from_template(_grid(10.0), 5.0)
    assert [(t.name, t.tier_class) for t in tg.tiers] == [
        ("words", "IntervalTier"), ("marks", "TextTier"),
    ]
    assert [iv.text for iv in tg.tiers[0].intervals] == [
        "", "tena", "koutou", "",
    ]


def test_template_tiers_only_discards_labels():
    tg = TextGrid.from_template(_grid(10.0), 5.0, copy_labels=False)
    assert [(t.name, t.tier_class) for t in tg.tiers] == [
        ("words", "IntervalTier"), ("marks", "TextTier"),
    ]
    assert _boundaries(tg.tiers[0]) == [(0, 5.0, "")]
    assert tg.tiers[1].points == []


def test_template_handles_non_zero_xmin():
    src = TextGrid(2.0, 6.0, [
        Tier("t", "IntervalTier", 2.0, 6.0,
             intervals=[Interval(2.0, 4.0, "a"), Interval(4.0, 6.0, "b")]),
    ])
    tg = TextGrid.from_template(src, 8.0)
    assert _boundaries(tg.tiers[0]) == [(0.0, 4.0, "a"), (4.0, 8.0, "b")]


def test_template_same_duration_is_a_faithful_copy():
    src = _grid(10.0)
    tg = TextGrid.from_template(src, 10.0)
    assert tg == src


def test_template_copy_is_independent_of_source():
    src = _grid(10.0)
    tg = TextGrid.from_template(src, 10.0)
    tg.tiers[0].intervals[1].text = "changed"
    assert src.tiers[0].intervals[1].text == "tena"


def test_template_rejects_zero_length_domain():
    src = TextGrid(0, 0, [_new_tier("t", "IntervalTier", 0)])
    with pytest.raises(ValueError):
        TextGrid.from_template(src, 5.0)


def test_template_survives_save_and_reload(tmp_path):
    path = tmp_path / "project1_template.TextGrid"
    _grid(10.0).save(str(path))

    reloaded = TextGrid.from_file(str(path))
    tg = TextGrid.from_template(reloaded, 5.0)
    assert [iv.text for iv in tg.tiers[0].intervals] == [
        "", "tena", "koutou", "",
    ]
    assert tg.tiers[0].intervals[-1].xmax == pytest.approx(5.0, abs=TIME_EPS)


# ---------------------------------------------------------------------------
# Save-path guard (pure logic, exercised without a Qt window)
# ---------------------------------------------------------------------------

class _PathHolder:
    """Stand-in exposing the path logic without building a Qt window."""

    _textgrid_path_matches_audio = MainWindow._textgrid_path_matches_audio
    _default_textgrid_path = MainWindow._default_textgrid_path

    def __init__(self, audio, grid):
        self._filepath = audio
        self._textgrid_path = grid


@pytest.mark.parametrize("audio,grid,expected", [
    (r"C:\corpus\file2.wav", r"C:\corpus\file2.TextGrid", True),
    (r"C:\corpus\file2.wav", r"C:\corpus\file2.textgrid", True),
    (r"C:\corpus\file2.wav", r"C:\other\file2.TextGrid", True),
    (r"C:\corpus\file2.wav", r"C:\corpus\file1.TextGrid", False),
    (r"C:\corpus\file2.wav", r"C:\corpus\FILE2.TextGrid", True),
    (r"C:\corpus\file2.wav", None, True),
    (None, r"C:\corpus\file1.TextGrid", True),
])
def test_textgrid_path_match_detection(audio, grid, expected):
    assert _PathHolder(audio, grid)._textgrid_path_matches_audio() is expected


def test_default_save_path_follows_the_audio_file():
    holder = _PathHolder(os.path.join("c", "file2.wav"),
                         os.path.join("c", "file1.TextGrid"))
    assert holder._default_textgrid_path() == os.path.join("c", "file2.TextGrid")


def test_default_save_path_empty_without_audio():
    assert _PathHolder(None, None)._default_textgrid_path() == ""
