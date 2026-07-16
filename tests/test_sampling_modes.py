"""
Tests for flexible formant sampling modes — percentage list, percentage step,
and time-based (ms) stepping in CSV export.

Covers:
- Percentage step expansion (wizard-level logic)
- _build_csv_data header generation for all three modes
- _build_csv_data row generation with time-based stepping
- Edge cases: short vowels, step larger than duration, single offset
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from formant_editor import (
    FormantData,
    Interval,
    TextGrid,
    Tier,
    _build_csv_data,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_formant_data(duration_s=0.2, time_step=0.005, n_formants=5):
    """Create FormantData spanning *duration_s* with known linear F1-F3."""
    n_frames = int(duration_s / time_step) + 1
    times = np.linspace(0.0, duration_s, n_frames)
    values = np.full((n_formants, n_frames), np.nan)
    values[0] = np.linspace(300, 800, n_frames)   # F1: 300 -> 800
    values[1] = np.linspace(1200, 2000, n_frames)  # F2: 1200 -> 2000
    values[2] = np.linspace(2500, 3000, n_frames)  # F3: 2500 -> 3000
    return FormantData(times=times, values=values, n_formants=n_formants,
                       time_step=time_step)


def _setup_test_corpus(tmpdir, intervals, duration_s=0.5,
                       tier_name="phones", time_step=0.005):
    """Create a minimal test corpus with one audio file stub,
    one TextGrid, and one .formants file.

    *intervals* is a list of (xmin, xmax, label) tuples.
    Returns (audio_dir, textgrid_dir, formants_dir).
    """
    audio_dir = os.path.join(tmpdir, "audio")
    tg_dir = os.path.join(tmpdir, "textgrids")
    fmt_dir = os.path.join(tmpdir, "formants")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(tg_dir, exist_ok=True)
    os.makedirs(fmt_dir, exist_ok=True)

    basename = "test"

    # Stub WAV (empty — _build_csv_data only checks existence)
    with open(os.path.join(audio_dir, f"{basename}.wav"), "wb") as f:
        f.write(b"\x00")

    # TextGrid
    ivs = [Interval(xmin, xmax, label) for xmin, xmax, label in intervals]
    tier = Tier(tier_name, "IntervalTier", 0.0, duration_s, intervals=ivs)
    tg = TextGrid(0.0, duration_s, [tier])
    tg_path = os.path.join(tg_dir, f"{basename}.TextGrid")
    tg.save(tg_path)

    # FormantData
    fd = _make_formant_data(duration_s=duration_s, time_step=time_step)
    fd_path = os.path.join(fmt_dir, f"{basename}.formants")
    fd.save(fd_path)

    return audio_dir, tg_dir, fmt_dir


# ---------------------------------------------------------------------------
# Percentage step expansion
# ---------------------------------------------------------------------------

class TestPercentageStepExpansion:
    """Test the logic that expands a single step value into a percentage list.

    This mirrors the wizard's validatePage expansion for 'Percentage step' mode.
    """

    @staticmethod
    def _expand_step(step):
        """Replicate the wizard expansion logic."""
        pcts = []
        v = 0.0
        while v <= 100.0 + 1e-9:
            pcts.append(round(v, 4))
            v += step
        if pcts[-1] < 100.0:
            pcts.append(100.0)
        return pcts

    def test_step_10(self):
        result = self._expand_step(10)
        assert result == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    def test_step_25(self):
        result = self._expand_step(25)
        assert result == [0, 25, 50, 75, 100]

    def test_step_100(self):
        result = self._expand_step(100)
        assert result == [0, 100]

    def test_step_7_includes_endpoint(self):
        result = self._expand_step(7)
        assert result[0] == 0
        assert result[-1] == 100.0
        assert 100.0 in result

    def test_step_33_3(self):
        result = self._expand_step(33.3)
        assert result[0] == 0
        assert result[-1] == 100.0
        assert len(result) == 5  # 0, 33.3, 66.6, 99.9, 100

    def test_step_50(self):
        result = self._expand_step(50)
        assert result == [0, 50, 100]

    def test_step_5(self):
        result = self._expand_step(5)
        assert len(result) == 21  # 0, 5, 10, ..., 100
        assert result[0] == 0
        assert result[-1] == 100
        assert result[10] == 50


# ---------------------------------------------------------------------------
# _build_csv_data: percentage list mode (existing behaviour preserved)
# ---------------------------------------------------------------------------

class TestBuildCsvPercentageList:

    def test_headers_percentage_list(self, tmp_path):
        audio_dir, tg_dir, fmt_dir = _setup_test_corpus(
            str(tmp_path),
            intervals=[(0.0, 0.1, "a")],
            tier_name="phones",
        )
        tier = Tier("phones", "IntervalTier", 0.0, 0.5)
        headers, rows = _build_csv_data(
            audio_dir=audio_dir,
            textgrid_dir=tg_dir,
            formants_dir=fmt_dir,
            selected_tiers=[tier],
            extract_formants=True,
            formant_mode="for_segments",
            point_tier_name=None,
            segment_tier_name="phones",
            percentage_markers=[0, 50, 100],
            extract_durations=False,
            duration_tier_names=[],
        )
        assert "F1_0%" in headers
        assert "F2_50%" in headers
        assert "F3_100%" in headers
        assert len([h for h in headers if h.startswith("F1_")]) == 3

    def test_data_rows_percentage_list(self, tmp_path):
        audio_dir, tg_dir, fmt_dir = _setup_test_corpus(
            str(tmp_path),
            intervals=[(0.0, 0.1, "a"), (0.1, 0.2, "e")],
            tier_name="phones",
        )
        tier = Tier("phones", "IntervalTier", 0.0, 0.5)
        headers, rows = _build_csv_data(
            audio_dir=audio_dir,
            textgrid_dir=tg_dir,
            formants_dir=fmt_dir,
            selected_tiers=[tier],
            extract_formants=True,
            formant_mode="for_segments",
            point_tier_name=None,
            segment_tier_name="phones",
            percentage_markers=[0, 100],
            extract_durations=False,
            duration_tier_names=[],
        )
        assert len(rows) == 2
        # Each row should have filename + label + 6 formant values (2 pcts * 3)
        assert len(rows[0]) == len(headers)


# ---------------------------------------------------------------------------
# _build_csv_data: time-step mode
# ---------------------------------------------------------------------------

class TestBuildCsvTimeStep:

    def test_headers_time_step(self, tmp_path):
        """Time-step mode should produce F1_Xms headers."""
        audio_dir, tg_dir, fmt_dir = _setup_test_corpus(
            str(tmp_path),
            intervals=[(0.0, 0.020, "a")],
            tier_name="phones",
        )
        tier = Tier("phones", "IntervalTier", 0.0, 0.5)
        headers, rows = _build_csv_data(
            audio_dir=audio_dir,
            textgrid_dir=tg_dir,
            formants_dir=fmt_dir,
            selected_tiers=[tier],
            extract_formants=True,
            formant_mode="for_segments",
            point_tier_name=None,
            segment_tier_name="phones",
            percentage_markers=[],
            extract_durations=False,
            duration_tier_names=[],
            time_step_ms=5,
        )
        # 20ms vowel with 5ms step => offsets 0, 5, 10, 15, 20 => 5 offsets
        formant_headers = [h for h in headers if h.endswith("ms")]
        assert "F1_0ms" in headers
        assert "F1_5ms" in headers
        assert "F1_10ms" in headers
        assert "F1_15ms" in headers
        assert "F1_20ms" in headers
        assert len(formant_headers) == 15  # 5 offsets * 3 formants
        # No percentage headers
        assert not any("%" in h for h in headers)

    def test_data_rows_time_step_filled(self, tmp_path):
        """All offsets within duration should have formant values."""
        audio_dir, tg_dir, fmt_dir = _setup_test_corpus(
            str(tmp_path),
            intervals=[(0.0, 0.020, "a")],
            tier_name="phones",
        )
        tier = Tier("phones", "IntervalTier", 0.0, 0.5)
        headers, rows = _build_csv_data(
            audio_dir=audio_dir,
            textgrid_dir=tg_dir,
            formants_dir=fmt_dir,
            selected_tiers=[tier],
            extract_formants=True,
            formant_mode="for_segments",
            point_tier_name=None,
            segment_tier_name="phones",
            percentage_markers=[],
            extract_durations=False,
            duration_tier_names=[],
            time_step_ms=5,
        )
        assert len(rows) == 1
        row = rows[0]
        # filename + phones label + 15 formant cells
        assert len(row) == len(headers)
        # All formant cells should be filled (vowel spans all offsets)
        f1_0_idx = headers.index("F1_0ms")
        for i in range(f1_0_idx, len(row)):
            assert row[i] != "", f"Cell at index {i} ({headers[i]}) is empty"

    def test_trailing_empty_cells_for_short_vowel(self, tmp_path):
        """Shorter vowels should have trailing empty cells."""
        audio_dir, tg_dir, fmt_dir = _setup_test_corpus(
            str(tmp_path),
            intervals=[
                (0.0, 0.020, "a"),    # 20ms
                (0.02, 0.030, "e"),   # 10ms — shorter
            ],
            tier_name="phones",
        )
        tier = Tier("phones", "IntervalTier", 0.0, 0.5)
        headers, rows = _build_csv_data(
            audio_dir=audio_dir,
            textgrid_dir=tg_dir,
            formants_dir=fmt_dir,
            selected_tiers=[tier],
            extract_formants=True,
            formant_mode="for_segments",
            point_tier_name=None,
            segment_tier_name="phones",
            percentage_markers=[],
            extract_durations=False,
            duration_tier_names=[],
            time_step_ms=5,
        )
        # Max duration 20ms, step 5ms => offsets 0,5,10,15,20 => 5 offsets
        assert len(rows) == 2
        row_short = rows[1]  # "e" at 10ms duration

        # Offsets 0,5,10 should be filled; 15,20 should be empty
        f1_15_idx = headers.index("F1_15ms")
        f3_20_idx = headers.index("F3_20ms")
        for i in range(f1_15_idx, f3_20_idx + 1):
            assert row_short[i] == "", \
                f"Cell {headers[i]} should be empty for short vowel"

        # But offsets 0,5,10 should have values
        f1_0_idx = headers.index("F1_0ms")
        f3_10_idx = headers.index("F3_10ms")
        for i in range(f1_0_idx, f3_10_idx + 1):
            assert row_short[i] != "", \
                f"Cell {headers[i]} should be filled for short vowel"

    def test_step_larger_than_all_vowels(self, tmp_path):
        """Step much larger than vowel duration => only 0ms offset."""
        audio_dir, tg_dir, fmt_dir = _setup_test_corpus(
            str(tmp_path),
            intervals=[(0.0, 0.005, "a")],  # 5ms vowel
            tier_name="phones",
        )
        tier = Tier("phones", "IntervalTier", 0.0, 0.5)
        headers, rows = _build_csv_data(
            audio_dir=audio_dir,
            textgrid_dir=tg_dir,
            formants_dir=fmt_dir,
            selected_tiers=[tier],
            extract_formants=True,
            formant_mode="for_segments",
            point_tier_name=None,
            segment_tier_name="phones",
            percentage_markers=[],
            extract_durations=False,
            duration_tier_names=[],
            time_step_ms=500,  # 500ms step, 5ms vowel
        )
        formant_headers = [h for h in headers if h.endswith("ms")]
        assert formant_headers == ["F1_0ms", "F2_0ms", "F3_0ms"]

    def test_no_labelled_intervals(self, tmp_path):
        """Empty labels => no data rows, but headers still generated
        (max_dur stays 0 => only 0ms offset)."""
        audio_dir, tg_dir, fmt_dir = _setup_test_corpus(
            str(tmp_path),
            intervals=[(0.0, 0.1, "")],  # empty label
            tier_name="phones",
        )
        tier = Tier("phones", "IntervalTier", 0.0, 0.5)
        headers, rows = _build_csv_data(
            audio_dir=audio_dir,
            textgrid_dir=tg_dir,
            formants_dir=fmt_dir,
            selected_tiers=[tier],
            extract_formants=True,
            formant_mode="for_segments",
            point_tier_name=None,
            segment_tier_name="phones",
            percentage_markers=[],
            extract_durations=False,
            duration_tier_names=[],
            time_step_ms=5,
        )
        # No labelled segments, so single stub row (filename only) or no rows
        # from the segment processing. The function still generates a row per
        # audio file when no TextGrid match → but we have a TextGrid, just no
        # labelled intervals, so rows from the interval loop = 0.
        # However, the row for the audio file is still created from the outer
        # loop fallback IF no matching intervals exist.
        # With time_mode and no labelled intervals, max_dur_s stays 0.0,
        # so time_offsets_ms = [0.0] => 3 formant columns.
        formant_headers = [h for h in headers if h.endswith("ms")]
        assert "F1_0ms" in formant_headers


# ---------------------------------------------------------------------------
# _build_csv_data: percentage step mode (expansion verified)
# ---------------------------------------------------------------------------

class TestBuildCsvPercentageStep:
    """Percentage step expansion happens in the wizard, so _build_csv_data
    just receives an expanded list. Verify the expanded list works correctly."""

    def test_expanded_step_10_headers(self, tmp_path):
        """Step=10 expanded to [0,10,...,100] should produce 11 offsets."""
        audio_dir, tg_dir, fmt_dir = _setup_test_corpus(
            str(tmp_path),
            intervals=[(0.0, 0.1, "a")],
            tier_name="phones",
        )
        markers = list(range(0, 101, 10))
        tier = Tier("phones", "IntervalTier", 0.0, 0.5)
        headers, rows = _build_csv_data(
            audio_dir=audio_dir,
            textgrid_dir=tg_dir,
            formants_dir=fmt_dir,
            selected_tiers=[tier],
            extract_formants=True,
            formant_mode="for_segments",
            point_tier_name=None,
            segment_tier_name="phones",
            percentage_markers=markers,
            extract_durations=False,
            duration_tier_names=[],
        )
        pct_headers = [h for h in headers if "%" in h]
        assert len(pct_headers) == 33  # 11 offsets * 3 formants


# ---------------------------------------------------------------------------
# Validation edge cases
# ---------------------------------------------------------------------------

class TestValidationEdgeCases:

    def test_step_expansion_no_float_drift(self):
        """Verify that float accumulation doesn't cause issues."""
        pcts = []
        v = 0.0
        step = 0.1
        while v <= 100.0 + 1e-9:
            pcts.append(round(v, 4))
            v += step
        if pcts[-1] < 100.0:
            pcts.append(100.0)
        # Should not have duplicate 100.0
        assert pcts.count(100.0) == 1
        assert pcts[-1] == 100.0

    def test_step_expansion_exact_divisor(self):
        """Step that evenly divides 100 should end exactly at 100."""
        pcts = []
        v = 0.0
        step = 20.0
        while v <= 100.0 + 1e-9:
            pcts.append(round(v, 4))
            v += step
        if pcts[-1] < 100.0:
            pcts.append(100.0)
        assert pcts == [0, 20, 40, 60, 80, 100]
