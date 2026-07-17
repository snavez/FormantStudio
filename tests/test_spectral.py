"""
Tests for spectral-moment analysis helpers (consonant/fricative spectra).

Covers window clipping, high-pass filtering, and moment computation, with
both correct-pass and correct-fail scenarios.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from formant_editor import (
    clip_window_to_segment, highpass_sound, compute_spectral_moments,
    _spectral_window_shape, SPECTRAL_WINDOW_SHAPES,
)

import parselmouth


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pure_tone_sound():
    """1000 Hz pure tone (0.5 s)."""
    sr = 44100
    t = np.arange(0, 0.5, 1 / sr)
    return parselmouth.Sound(0.8 * np.sin(2 * np.pi * 1000.0 * t),
                             sampling_frequency=sr)


@pytest.fixture
def low_plus_high_sound():
    """Strong 200 Hz tone + weak 6000 Hz tone (voicing + frication mock)."""
    sr = 44100
    t = np.arange(0, 0.5, 1 / sr)
    sig = 0.9 * np.sin(2 * np.pi * 200.0 * t) + 0.1 * np.sin(2 * np.pi * 6000.0 * t)
    return parselmouth.Sound(sig, sampling_frequency=sr)


@pytest.fixture
def hf_noise_sound():
    """High-pass-ish noise, fricative-like (energy biased high)."""
    sr = 44100
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(int(sr * 0.3))
    snd = parselmouth.Sound(noise * 0.3, sampling_frequency=sr)
    # Pre-emphasise to push energy up, mimicking a sibilant
    return parselmouth.praat.call(snd, "Filter (pass Hann band)...", 4000.0, 0.0, 100.0)


# ---------------------------------------------------------------------------
# clip_window_to_segment
# ---------------------------------------------------------------------------

class TestClipWindow:
    def test_centre_window_fits(self):
        t0, t1 = clip_window_to_segment(0.5, 0.04, 0.0, 1.0)
        assert t0 == pytest.approx(0.48)
        assert t1 == pytest.approx(0.52)

    def test_left_edge_clipped(self):
        # 10% point of a 0.1 s segment with a 40 ms window spills left
        t0, t1 = clip_window_to_segment(0.01, 0.04, 0.0, 0.1)
        assert t0 == pytest.approx(0.0)      # clipped at segment start
        assert t1 == pytest.approx(0.03)     # centre preserved on the right
        assert (t1 - t0) < 0.04              # window is shorter than requested

    def test_right_edge_clipped(self):
        t0, t1 = clip_window_to_segment(0.095, 0.04, 0.0, 0.1)
        assert t0 == pytest.approx(0.075)
        assert t1 == pytest.approx(0.1)

    def test_window_wider_than_segment(self):
        # Requested window exceeds the whole segment → clamps to segment
        t0, t1 = clip_window_to_segment(0.05, 0.5, 0.0, 0.1)
        assert t0 == pytest.approx(0.0)
        assert t1 == pytest.approx(0.1)

    def test_centre_out_of_range_returns_empty(self):
        t0, t1 = clip_window_to_segment(-1.0, 0.04, 0.0, 0.1)
        assert t1 == t0  # empty window, never negative-length


# ---------------------------------------------------------------------------
# highpass_sound
# ---------------------------------------------------------------------------

class TestHighpass:
    def test_zero_cutoff_returns_same_object(self, pure_tone_sound):
        assert highpass_sound(pure_tone_sound, 0.0) is pure_tone_sound
        assert highpass_sound(pure_tone_sound, -5.0) is pure_tone_sound

    def test_highpass_raises_cog(self, low_plus_high_sound):
        shape = _spectral_window_shape("Hamming")
        cog_raw, *_ = compute_spectral_moments(
            low_plus_high_sound, 0.2, 0.24, shape)
        filtered = highpass_sound(low_plus_high_sound, 1000.0)
        cog_hp, *_ = compute_spectral_moments(filtered, 0.2, 0.24, shape)
        # Removing the dominant 200 Hz energy pulls the centre of gravity up
        assert cog_hp > cog_raw
        assert cog_raw < 1000.0   # dominated by the low tone
        assert cog_hp > 2000.0    # now dominated by the 6000 Hz tone


# ---------------------------------------------------------------------------
# compute_spectral_moments
# ---------------------------------------------------------------------------

class TestSpectralMoments:
    def test_pure_tone_cog_near_tone(self, pure_tone_sound):
        shape = _spectral_window_shape("Hamming")
        cog, sd, skew, kurt = compute_spectral_moments(
            pure_tone_sound, 0.2, 0.24, shape)
        assert cog == pytest.approx(1000.0, abs=60.0)
        assert sd >= 0.0

    def test_fricative_like_high_cog(self, hf_noise_sound):
        shape = _spectral_window_shape("Hamming")
        cog, *_ = compute_spectral_moments(hf_noise_sound, 0.1, 0.14, shape)
        assert cog > 3000.0  # energy biased high

    def test_empty_window_returns_nan(self, pure_tone_sound):
        shape = _spectral_window_shape("Hamming")
        vals = compute_spectral_moments(pure_tone_sound, 0.2, 0.2, shape)
        assert all(np.isnan(v) for v in vals)

    def test_reversed_window_returns_nan(self, pure_tone_sound):
        shape = _spectral_window_shape("Hamming")
        vals = compute_spectral_moments(pure_tone_sound, 0.24, 0.2, shape)
        assert all(np.isnan(v) for v in vals)

    def test_window_shape_names_resolve(self):
        for name in SPECTRAL_WINDOW_SHAPES:
            shape = _spectral_window_shape(name)
            assert isinstance(shape, parselmouth.WindowShape)
        # Unknown falls back to Hamming, never raises
        assert _spectral_window_shape("nonsense") == parselmouth.WindowShape.HAMMING


# ---------------------------------------------------------------------------
# _build_csv_data integration (spectral columns)
# ---------------------------------------------------------------------------

from formant_editor import _build_csv_data, TextGrid, Tier, Interval


def _write_corpus(tmp_path, intervals, duration_s=0.6, sr=44100):
    """Write a real noise WAV + matching TextGrid; return (audio, tg) dirs."""
    audio_dir = tmp_path / "audio"
    tg_dir = tmp_path / "tg"
    audio_dir.mkdir()
    tg_dir.mkdir()
    rng = np.random.default_rng(1)
    sig = rng.standard_normal(int(sr * duration_s)) * 0.2
    snd = parselmouth.Sound(sig, sampling_frequency=sr)
    snd.save(str(audio_dir / "test.wav"), parselmouth.SoundFileFormat.WAV)
    ivs = [Interval(a, b, t) for a, b, t in intervals]
    tier = Tier("phones", "IntervalTier", 0.0, duration_s, intervals=ivs)
    TextGrid(0.0, duration_s, [tier]).save(str(tg_dir / "test.TextGrid"))
    return str(audio_dir), str(tg_dir)


def _spectral_kwargs(**over):
    base = dict(
        audio_dir=None, textgrid_dir=None, formants_dir=None,
        selected_tiers=None, extract_formants=False, formant_mode=None,
        point_tier_name=None, segment_tier_name=None,
        percentage_markers=[], extract_durations=False, duration_tier_names=[],
        extract_spectral=True, spectral_markers=[25, 50, 75],
        spectral_window_ms=25.0, spectral_window_type="Hamming",
        spectral_highpass_hz=0.0, spectral_min_window_ms=5.0,
    )
    base.update(over)
    return base


class TestSpectralCSV:
    def test_headers_present(self, tmp_path):
        audio, tg = _write_corpus(tmp_path, [(0.1, 0.4, "s")])
        tier = TextGrid.from_file(os.path.join(tg, "test.TextGrid")).tiers
        headers, _ = _build_csv_data(**_spectral_kwargs(
            audio_dir=audio, textgrid_dir=tg, selected_tiers=tier))
        for pct in (25, 50, 75):
            for m in ("COG", "SD", "skew", "kurt", "winms"):
                assert f"{m}_{pct}%" in headers

    def test_normal_segment_has_moments(self, tmp_path):
        audio, tg = _write_corpus(tmp_path, [(0.1, 0.4, "s")])
        tier = TextGrid.from_file(os.path.join(tg, "test.TextGrid")).tiers
        headers, rows = _build_csv_data(**_spectral_kwargs(
            audio_dir=audio, textgrid_dir=tg, selected_tiers=tier))
        row = rows[0]
        cog = row[headers.index("COG_50%")]
        win = row[headers.index("winms_50%")]
        assert cog != ""              # a numeric COG was produced
        assert float(win) == pytest.approx(25.0, abs=0.5)  # full window

    def test_short_segment_blanks_moments_but_reports_window(self, tmp_path):
        # 4 ms segment < 5 ms floor even at the midpoint → blanked, flagged
        audio, tg = _write_corpus(tmp_path, [(0.20, 0.204, "t")])
        tier = TextGrid.from_file(os.path.join(tg, "test.TextGrid")).tiers
        headers, rows = _build_csv_data(**_spectral_kwargs(
            audio_dir=audio, textgrid_dir=tg, selected_tiers=tier))
        row = rows[0]
        assert row[headers.index("COG_50%")] == ""      # unreliable → blank
        win = float(row[headers.index("winms_50%")])
        assert win < 5.0 and win > 0.0                  # actual width reported

    def test_spectral_only_still_emits_label_rows(self, tmp_path):
        audio, tg = _write_corpus(
            tmp_path, [(0.1, 0.3, "s"), (0.3, 0.5, "a")])
        tier = TextGrid.from_file(os.path.join(tg, "test.TextGrid")).tiers
        headers, rows = _build_csv_data(**_spectral_kwargs(
            audio_dir=audio, textgrid_dir=tg, selected_tiers=tier))
        # One row per labelled segment, label preserved
        labels = [r[headers.index("phones")] for r in rows]
        assert "s" in labels and "a" in labels

    def test_highpass_shifts_cog_up(self, tmp_path):
        audio, tg = _write_corpus(tmp_path, [(0.1, 0.4, "z")])
        tier = TextGrid.from_file(os.path.join(tg, "test.TextGrid")).tiers
        h0, r0 = _build_csv_data(**_spectral_kwargs(
            audio_dir=audio, textgrid_dir=tg, selected_tiers=tier,
            spectral_highpass_hz=0.0))
        h1, r1 = _build_csv_data(**_spectral_kwargs(
            audio_dir=audio, textgrid_dir=tg, selected_tiers=tier,
            spectral_highpass_hz=2000.0))
        cog_raw = float(r0[0][h0.index("COG_50%")])
        cog_hp = float(r1[0][h1.index("COG_50%")])
        assert cog_hp > cog_raw


def _write_two_tier_corpus(tmp_path, duration_s=0.6, sr=44100):
    """Word tier 0.1-0.5 'ta' containing phones 't' (0.1-0.2), 'a' (0.2-0.5)."""
    audio_dir = tmp_path / "audio"
    tg_dir = tmp_path / "tg"
    audio_dir.mkdir()
    tg_dir.mkdir()
    rng = np.random.default_rng(2)
    sig = rng.standard_normal(int(sr * duration_s)) * 0.2
    snd = parselmouth.Sound(sig, sampling_frequency=sr)
    snd.save(str(audio_dir / "test.wav"), parselmouth.SoundFileFormat.WAV)
    words = Tier("words", "IntervalTier", 0.0, duration_s,
                 intervals=[Interval(0.1, 0.5, "ta")])
    phones = Tier("phones", "IntervalTier", 0.0, duration_s,
                  intervals=[Interval(0.1, 0.2, "t"),
                             Interval(0.2, 0.5, "a")])
    TextGrid(0.0, duration_s, [words, phones]).save(
        str(tg_dir / "test.TextGrid"))
    return str(audio_dir), str(tg_dir)


class TestSpectralTierSelection:
    """Spectral sampling follows the chosen tier, not the row tier."""

    def test_row_tier_sampling_differs_per_phone(self, tmp_path):
        audio, tg = _write_two_tier_corpus(tmp_path)
        tiers = TextGrid.from_file(os.path.join(tg, "test.TextGrid")).tiers
        headers, rows = _build_csv_data(**_spectral_kwargs(
            audio_dir=audio, textgrid_dir=tg, selected_tiers=tiers,
            spectral_tier_name="phones", spectral_markers=[50]))
        assert len(rows) == 2  # rows driven by phones (lowest tier)
        wins = [float(r[headers.index("winms_50%")]) for r in rows]
        assert all(w == pytest.approx(25.0, abs=0.5) for w in wins)
        cogs = [r[headers.index("COG_50%")] for r in rows]
        assert cogs[0] != cogs[1]  # different windows → different spectra

    def test_parent_tier_sampling_repeats_across_rows(self, tmp_path):
        audio, tg = _write_two_tier_corpus(tmp_path)
        tiers = TextGrid.from_file(os.path.join(tg, "test.TextGrid")).tiers
        headers, rows = _build_csv_data(**_spectral_kwargs(
            audio_dir=audio, textgrid_dir=tg, selected_tiers=tiers,
            spectral_tier_name="words", spectral_markers=[50]))
        assert len(rows) == 2
        # Both phone rows sample the containing WORD's midpoint window
        cogs = [r[headers.index("COG_50%")] for r in rows]
        assert cogs[0] == cogs[1] and cogs[0] != ""

    def test_unlabelled_container_blanks_rather_than_measuring_gap(
            self, tmp_path):
        """A row inside an UNLABELLED stretch of the chosen tier must blank.

        Regression: a sub-phone tier ('allophone') has long empty intervals
        between its labels. Resolving a row to one of those and sampling its
        midpoint reported COG measured from arbitrary unrelated audio.
        """
        audio_dir = tmp_path / "audio"
        tg_dir = tmp_path / "tg"
        audio_dir.mkdir()
        tg_dir.mkdir()
        sr, dur = 44100, 1.0
        rng = np.random.default_rng(3)
        parselmouth.Sound(rng.standard_normal(int(sr * dur)) * 0.2,
                          sampling_frequency=sr).save(
            str(audio_dir / "test.wav"), parselmouth.SoundFileFormat.WAV)
        # phones drive rows; 'allophone' labels only the LAST phone's span
        phones = Tier("phones", "IntervalTier", 0.0, dur, intervals=[
            Interval(0.0, 0.4, "a"),      # sits in the empty allophone gap
            Interval(0.4, 0.8, "t"),      # inside the labelled allophone
        ])
        allo = Tier("allophone", "IntervalTier", 0.0, dur, intervals=[
            Interval(0.0, 0.4, ""),       # long unlabelled gap
            Interval(0.4, 0.8, "tH"),     # the only labelled span
            Interval(0.8, dur, ""),
        ])
        TextGrid(0.0, dur, [phones, allo]).save(str(tg_dir / "test.TextGrid"))
        tiers = TextGrid.from_file(str(tg_dir / "test.TextGrid")).tiers

        headers, rows = _build_csv_data(**_spectral_kwargs(
            audio_dir=str(audio_dir), textgrid_dir=str(tg_dir),
            selected_tiers=[tiers[0]],           # phones drives rows
            spectral_tier_name="allophone", spectral_markers=[50]))
        assert len(rows) == 2
        cog_i = headers.index("COG_50%")
        assert rows[0][cog_i] == ""      # 'a' -> unlabelled gap -> blank
        assert rows[1][cog_i] != ""      # 't' -> inside 'tH' -> measured

    def test_missing_tier_blanks_spectral_cells(self, tmp_path):
        audio, tg = _write_two_tier_corpus(tmp_path)
        tiers = TextGrid.from_file(os.path.join(tg, "test.TextGrid")).tiers
        headers, rows = _build_csv_data(**_spectral_kwargs(
            audio_dir=audio, textgrid_dir=tg, selected_tiers=tiers,
            spectral_tier_name="nonexistent", spectral_markers=[50]))
        for r in rows:
            assert r[headers.index("COG_50%")] == ""
            assert r[headers.index("winms_50%")] == ""


class TestSkippedFiles:
    """Audio files with no usable TextGrid are reported, not silently blank."""

    def test_missing_textgrid_is_skipped_and_reported(self, tmp_path):
        audio, tg = _write_corpus(tmp_path, [(0.1, 0.4, "s")])
        tiers = TextGrid.from_file(os.path.join(tg, "test.TextGrid")).tiers
        # Rename the TextGrid so nothing matches the audio's base name
        os.rename(os.path.join(tg, "test.TextGrid"),
                  os.path.join(tg, "testTS.TextGrid"))
        skipped = []
        _, rows = _build_csv_data(**_spectral_kwargs(
            audio_dir=audio, textgrid_dir=tg, selected_tiers=tiers,
            skipped_files=skipped))
        assert rows == []                       # no silent blank row
        assert len(skipped) == 1
        assert skipped[0][0] == "test.wav"
        assert "test.TextGrid" in skipped[0][1]

    def test_unreadable_textgrid_is_skipped_and_reported(self, tmp_path):
        audio, tg = _write_corpus(tmp_path, [(0.1, 0.4, "s")])
        tiers = TextGrid.from_file(os.path.join(tg, "test.TextGrid")).tiers
        with open(os.path.join(tg, "test.TextGrid"), "w") as f:
            f.write("this is not a TextGrid")
        skipped = []
        _, rows = _build_csv_data(**_spectral_kwargs(
            audio_dir=audio, textgrid_dir=tg, selected_tiers=tiers,
            skipped_files=skipped))
        assert rows == []
        assert len(skipped) == 1
        assert "could not read" in skipped[0][1]

    def test_matched_files_unaffected_by_collector(self, tmp_path):
        audio, tg = _write_corpus(tmp_path, [(0.1, 0.4, "s")])
        tiers = TextGrid.from_file(os.path.join(tg, "test.TextGrid")).tiers
        skipped = []
        _, rows = _build_csv_data(**_spectral_kwargs(
            audio_dir=audio, textgrid_dir=tg, selected_tiers=tiers,
            skipped_files=skipped))
        assert len(rows) == 1
        assert skipped == []


class TestFormantSegmentTierResolution:
    """The formant segment tier now genuinely controls sampling positions."""

    def test_word_tier_sampling_repeats_across_phone_rows(self, tmp_path):
        from formant_editor import FormantData
        audio, tg = _write_two_tier_corpus(tmp_path)
        tiers = TextGrid.from_file(os.path.join(tg, "test.TextGrid")).tiers
        # Linear F1 ramp so the sampled time is recoverable from the value
        fmt_dir = tmp_path / "fmt"
        fmt_dir.mkdir()
        n = 121
        times = np.linspace(0.0, 0.6, n)
        values = np.full((5, n), np.nan)
        values[0] = np.linspace(0.0, 600.0, n)   # F1 = time_ms
        values[1] = np.linspace(1000.0, 1000.0, n)
        values[2] = np.linspace(2500.0, 2500.0, n)
        FormantData(times=times, values=values, n_formants=5,
                    time_step=times[1] - times[0]).save(
            str(fmt_dir / "test.formants"))

        headers, rows = _build_csv_data(
            audio_dir=audio, textgrid_dir=tg, formants_dir=str(fmt_dir),
            selected_tiers=tiers,
            extract_formants=True, formant_mode="for_segments",
            point_tier_name=None, segment_tier_name="words",
            percentage_markers=[50], extract_durations=False,
            duration_tier_names=[],
        )
        f1s = [float(r[headers.index("F1_50%")]) for r in rows]
        # Word midpoint = 0.3 s → F1 ≈ 300 for BOTH phone rows
        assert f1s[0] == pytest.approx(300.0, abs=6.0)
        assert f1s[1] == pytest.approx(300.0, abs=6.0)

        headers, rows = _build_csv_data(
            audio_dir=audio, textgrid_dir=tg, formants_dir=str(fmt_dir),
            selected_tiers=tiers,
            extract_formants=True, formant_mode="for_segments",
            point_tier_name=None, segment_tier_name="phones",
            percentage_markers=[50], extract_durations=False,
            duration_tier_names=[],
        )
        f1s = [float(r[headers.index("F1_50%")]) for r in rows]
        # Phone midpoints: 't' → 0.15 s, 'a' → 0.35 s
        assert f1s[0] == pytest.approx(150.0, abs=6.0)
        assert f1s[1] == pytest.approx(350.0, abs=6.0)
