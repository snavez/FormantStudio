"""
Tests for the row-anchor CSV model: one row per row-tier token, wide
numbered target columns, row_start/row_end anchors, and time-based label
matching in both directions (encapsulating parent, contained children,
points within the token).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from formant_editor import (
    FormantData,
    Interval,
    Point,
    TextGrid,
    Tier,
    _build_csv_data,
)


DUR = 0.5


def _linear_formants(duration_s=DUR, time_step=0.005):
    """F1 = 300 + 1000*t so the sampled time is recoverable from the value."""
    n = int(duration_s / time_step) + 1
    times = np.linspace(0.0, duration_s, n)
    values = np.full((5, n), np.nan)
    values[0] = 300.0 + 1000.0 * times
    values[1] = 1200.0 + 1000.0 * times
    values[2] = 2500.0 + 1000.0 * times
    return FormantData(times=times, values=values, n_formants=5,
                       time_step=time_step)


@pytest.fixture
def corpus(tmp_path):
    """Word 'tata' [0.05-0.45] > phones 't' [0.05-0.2], 'a' [0.2-0.45];
    allophone subdivides 't' into 't0'/'tH'; Target points: one in 't'
    (release), two in 'a' (diphthong-style); one stress point in 'a'."""
    audio_dir = tmp_path / "audio"
    tg_dir = tmp_path / "tg"
    fmt_dir = tmp_path / "fmt"
    for d in (audio_dir, tg_dir, fmt_dir):
        d.mkdir()
    (audio_dir / "test.wav").write_bytes(b"\x00")

    words = Tier("words", "IntervalTier", 0.0, DUR,
                 intervals=[Interval(0.05, 0.45, "tata")])
    phones = Tier("phones", "IntervalTier", 0.0, DUR,
                  intervals=[Interval(0.05, 0.2, "t"),
                             Interval(0.2, 0.45, "a")])
    allo = Tier("allophone", "IntervalTier", 0.0, DUR,
                intervals=[Interval(0.0, 0.05, ""),
                           Interval(0.05, 0.12, "t0"),
                           Interval(0.12, 0.2, "tH"),
                           Interval(0.2, DUR, "")])
    target = Tier("Target", "TextTier", 0.0, DUR,
                  points=[Point(0.10, "rel"),
                          Point(0.25, "T1"), Point(0.40, "T2")])
    stress = Tier("Stress", "TextTier", 0.0, DUR, points=[Point(0.22, "1")])

    TextGrid(0.0, DUR, [words, phones, allo, target, stress]).save(
        str(tg_dir / "test.TextGrid"))
    _linear_formants().save(str(fmt_dir / "test.formants"))
    tiers = TextGrid.from_file(str(tg_dir / "test.TextGrid")).tiers
    return {
        "audio": str(audio_dir), "tg": str(tg_dir), "fmt": str(fmt_dir),
        "tiers": {t.name: t for t in tiers},
    }


def _run(corpus, order, **over):
    base = dict(
        audio_dir=corpus["audio"], textgrid_dir=corpus["tg"],
        formants_dir=corpus["fmt"],
        selected_tiers=[corpus["tiers"][n] for n in order],
        extract_formants=True, formant_mode="at_points",
        point_tier_name="Target", segment_tier_name=None,
        percentage_markers=[], extract_durations=False,
        duration_tier_names=[], include_point_times=True,
    )
    base.update(over)
    return _build_csv_data(**base)


class TestWideTargetColumns:
    def test_headers_sized_by_corpus_max(self, corpus):
        headers, _ = _run(corpus, ["words", "phones", "Target"])
        # 'a' has two targets → two numbered sets, none more
        for i in (1, 2):
            assert f"Target{i}_time" in headers
            for f in ("F1", "F2", "F3"):
                assert f"{f}_Target{i}" in headers
        assert "F1_Target3" not in headers
        assert "F1_Target" not in headers   # old un-numbered form is gone

    def test_one_row_per_token_with_time_ordered_values(self, corpus):
        headers, rows = _run(corpus, ["words", "phones", "Target"])
        assert len(rows) == 2               # 't' and 'a' — never row-per-point
        r_t, r_a = rows
        f1_1 = headers.index("F1_Target1")
        f1_2 = headers.index("F1_Target2")
        # 't': one target at 0.10 → F1 = 300 + 1000*0.10 = 400; slot 2 blank
        assert float(r_t[f1_1]) == pytest.approx(400.0, abs=6)
        assert r_t[f1_2] == ""
        assert r_t[headers.index("Target2_time")] == ""
        # 'a': targets at 0.25 and 0.40, in time order
        assert float(r_a[f1_1]) == pytest.approx(550.0, abs=6)
        assert float(r_a[f1_2]) == pytest.approx(700.0, abs=6)
        assert float(r_a[headers.index("Target1_time")]) == pytest.approx(0.25)

    def test_row_bounds_columns(self, corpus):
        headers, rows = _run(corpus, ["words", "phones", "Target"])
        s = headers.index("row_start")
        e = headers.index("row_end")
        assert [float(rows[0][s]), float(rows[0][e])] == [0.05, 0.2]
        assert [float(rows[1][s]), float(rows[1][e])] == [0.2, 0.45]


class TestTimeBasedLabelMatching:
    def test_child_tier_labels_join(self, corpus):
        headers, rows = _run(
            corpus, ["words", "allophone", "phones", "Target"])
        allo_col = headers.index("allophone")
        assert rows[0][allo_col] == "t0; tH"    # 't' subdivides
        assert rows[1][allo_col] == ""          # 'a' has no allophone labels

    def test_parent_tier_label(self, corpus):
        headers, rows = _run(corpus, ["words", "phones", "Target"])
        w = headers.index("words")
        assert rows[0][w] == "tata" and rows[1][w] == "tata"

    def test_point_marks_within_row_token_only(self, corpus):
        headers, rows = _run(
            corpus, ["words", "phones", "Target", "Stress"])
        tcol = headers.index("Target")
        scol = headers.index("Stress")
        assert rows[0][tcol] == "rel"
        assert rows[1][tcol] == "T1; T2"
        assert rows[0][scol] == ""              # stress point is in 'a'
        assert rows[1][scol] == "1"

    def test_encapsulating_label_when_child_is_row_tier(self, corpus):
        # Flip: allophone drives rows → phones label appears on both rows
        headers, rows = _run(
            corpus, ["phones", "allophone", "Target"],
            formant_mode=None, extract_formants=False)
        assert len(rows) == 2                   # t0, tH
        p = headers.index("phones")
        assert rows[0][p] == "t" and rows[1][p] == "t"


class TestPointTierPrimary:
    """Primary can be a point tier: one row per point."""

    def test_one_row_per_point(self, corpus):
        headers, rows = _run(
            corpus, ["words", "phones", "Target"],
            primary_tier_name="Target",
            formant_mode="at_points", point_tier_name="Target")
        assert len(rows) == 3          # rel, T1, T2 — one row each

    def test_point_row_bounds_equal(self, corpus):
        _, rows = _run(
            corpus, ["words", "phones", "Target"],
            primary_tier_name="Target",
            formant_mode="at_points", point_tier_name="Target")
        headers, _ = _run(
            corpus, ["words", "phones", "Target"],
            primary_tier_name="Target",
            formant_mode="at_points", point_tier_name="Target")
        s, e = headers.index("row_start"), headers.index("row_end")
        for r in rows:
            assert r[s] == r[e]        # zero-width unit

    def test_point_primary_pulls_containing_interval_labels(self, corpus):
        headers, rows = _run(
            corpus, ["words", "phones", "Target"],
            primary_tier_name="Target",
            formant_mode="at_points", point_tier_name="Target")
        pcol = headers.index("phones")
        tcol = headers.index("Target")
        # rel@0.10 is in phone 't'; T1@0.25 and T2@0.40 are in 'a'
        by_target = {r[tcol]: r[pcol] for r in rows}
        assert by_target["rel"] == "t"
        assert by_target["T1"] == "a"
        assert by_target["T2"] == "a"

    def test_point_primary_formant_at_the_point(self, corpus):
        headers, rows = _run(
            corpus, ["words", "phones", "Target"],
            primary_tier_name="Target",
            formant_mode="at_points", point_tier_name="Target")
        # one target per zero-width unit → single wide set, F1_Target1
        f1 = headers.index("F1_Target1")
        tcol = headers.index("Target")
        vals = {r[tcol]: float(r[f1]) for r in rows}
        assert vals["rel"] == pytest.approx(400.0, abs=6)   # 300+1000*0.10
        assert vals["T1"] == pytest.approx(550.0, abs=6)    # 0.25
        assert vals["T2"] == pytest.approx(700.0, abs=6)    # 0.40


class TestRowTierMissing:
    def test_missing_row_tier_reported_not_blank(self, corpus):
        skipped = []
        fake = Tier("nonexistent", "IntervalTier", 0.0, DUR,
                    intervals=[Interval(0.0, DUR, "x")])
        headers, rows = _run(
            corpus, ["words", "phones", "Target"],
            selected_tiers=[fake], skipped_files=skipped,
            extract_formants=False, formant_mode=None)
        assert rows == []
        assert len(skipped) == 1
        assert "primary tier" in skipped[0][1]
