"""
Tests for the release-noisiness measures: spectral flatness, the high-band
level (absolute and relative to the file's speech level), the calibrated
spectrum they rest on, and the per-file clipping check.
"""

import math
import os
import sys

import numpy as np
import parselmouth
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from formant_editor import (  # noqa: E402
    FILE_SIGNAL_COLUMNS, P_REF_PA, SPECTRAL_MARKER_COLUMNS,
    SPECTRAL_TRACK_NAMES, Interval, TextGrid, Tier, _build_csv_data,
    band_level_db, clipped_percent, measure_window, spectral_flatness,
    spectrum_of, speech_level_db,
)

SR = 44100.0


def _db(mean_square):
    return 10.0 * math.log10(mean_square / P_REF_PA ** 2)


def _noise(seconds=0.2, sigma=0.1, seed=0):
    n = int(SR * seconds)
    return np.random.default_rng(seed).standard_normal(n) * sigma


def _tone(freq, seconds=0.2, amp=0.5):
    t = np.arange(int(SR * seconds)) / SR
    return amp * np.sin(2 * np.pi * freq * t)


def _sound(x):
    return parselmouth.Sound(np.asarray(x, dtype=float), sampling_frequency=SR)


ESTIMATORS = ["single_taper", "multitaper"]


# ---------------------------------------------------------------------------
# Calibrated spectrum and band level
# ---------------------------------------------------------------------------

class TestBandLevel:
    @pytest.mark.parametrize("estimator", ESTIMATORS)
    def test_white_noise_full_band_matches_its_mean_square(self, estimator):
        sigma = 0.1
        freqs, power = spectrum_of(_sound(_noise(sigma=sigma)), 0.05, 0.15,
                                   estimator=estimator)
        assert band_level_db(freqs, power, (0.0, 0.0)) == pytest.approx(
            _db(sigma ** 2), abs=0.5)

    @pytest.mark.parametrize("estimator", ESTIMATORS)
    def test_a_tone_in_band_reads_amplitude_squared_over_two(self, estimator):
        amp = 0.3
        freqs, power = spectrum_of(_sound(_tone(4000.0, amp=amp)), 0.05, 0.15,
                                   estimator=estimator)
        assert band_level_db(freqs, power, (2000.0, 8000.0)) == \
            pytest.approx(_db(amp ** 2 / 2.0), abs=0.3)

    def test_a_tone_outside_the_band_barely_registers(self):
        freqs, power = spectrum_of(_sound(_tone(500.0)), 0.05, 0.15,
                                   estimator="multitaper")
        inside = band_level_db(freqs, power, (300.0, 800.0))
        outside = band_level_db(freqs, power, (2000.0, 8000.0))
        assert inside - outside > 60.0

    def test_doubling_the_gain_adds_six_db(self):
        x = _noise()
        f1, p1 = spectrum_of(_sound(x), 0.05, 0.15, estimator="multitaper")
        f2, p2 = spectrum_of(_sound(2 * x), 0.05, 0.15,
                             estimator="multitaper")
        assert band_level_db(f2, p2, (2000.0, 8000.0)) - band_level_db(
            f1, p1, (2000.0, 8000.0)) == pytest.approx(20 * math.log10(2),
                                                       abs=1e-6)

    def test_is_nan_without_a_spectrum_or_an_empty_band(self):
        freqs, power = spectrum_of(_sound(_noise()), 0.05, 0.15)
        assert np.isnan(band_level_db(None, None, (0.0, 8000.0)))
        assert np.isnan(band_level_db(freqs, power, (30000.0, 40000.0)))

    def test_is_nan_for_digital_silence(self):
        freqs, power = spectrum_of(_sound(np.zeros(8820)), 0.05, 0.15)
        assert np.isnan(band_level_db(freqs, power, (2000.0, 8000.0)))


# ---------------------------------------------------------------------------
# Spectral flatness
# ---------------------------------------------------------------------------

class TestFlatness:
    def test_noise_is_flat_and_a_tone_is_not(self):
        f_n, p_n = spectrum_of(_sound(_noise()), 0.05, 0.15,
                               estimator="multitaper")
        f_t, p_t = spectrum_of(_sound(_tone(3000.0)), 0.05, 0.15,
                               estimator="multitaper")
        assert spectral_flatness(p_n) > 0.8
        assert spectral_flatness(p_t) < 0.05

    def test_a_harmonic_vowel_like_sound_is_less_flat_than_noise(self):
        voiced = sum(_tone(120.0 * k, amp=0.3 / k) for k in range(1, 40))
        _, p_v = spectrum_of(_sound(voiced), 0.05, 0.15,
                             estimator="multitaper")
        _, p_n = spectrum_of(_sound(_noise()), 0.05, 0.15,
                             estimator="multitaper")
        assert spectral_flatness(p_v) < 0.5 * spectral_flatness(p_n)

    def test_is_scale_free(self):
        p = np.random.default_rng(3).random(200) + 0.1
        assert spectral_flatness(p * 1e6) == pytest.approx(
            spectral_flatness(p))

    def test_a_perfectly_flat_spectrum_is_one(self):
        assert spectral_flatness(np.full(64, 3.0)) == pytest.approx(1.0)

    @pytest.mark.parametrize("bad", [None, np.zeros(10), np.array([]),
                                     np.array([1.0, np.nan])])
    def test_is_nan_when_unmeasurable(self, bad):
        assert np.isnan(spectral_flatness(bad))


class TestMeasureWindow:
    def test_flatness_and_level_arrive_in_track_order(self):
        vals = dict(zip(SPECTRAL_TRACK_NAMES,
                        measure_window(_sound(_noise()), 0.05, 0.15)))
        assert 0.0 < vals["flatness"] <= 1.0
        assert np.isfinite(vals["hblevel"])

    def test_the_high_band_level_follows_the_ratio_high_band(self):
        snd = _sound(_tone(1000.0, amp=0.5) + _tone(5000.0, amp=0.05))
        low = measure_window(snd, 0.05, 0.15, ratio_high=(500.0, 2000.0))
        high = measure_window(snd, 0.05, 0.15, ratio_high=(4000.0, 8000.0))
        i = SPECTRAL_TRACK_NAMES.index("hblevel")
        assert low[i] - high[i] == pytest.approx(20.0, abs=0.5)


# ---------------------------------------------------------------------------
# File-level: speech level and clipping
# ---------------------------------------------------------------------------

class TestSpeechLevel:
    def test_a_steady_tone_reads_its_own_level(self):
        amp = 0.2
        assert speech_level_db(_sound(_tone(440.0, seconds=1.0, amp=amp))) \
            == pytest.approx(_db(amp ** 2 / 2.0), abs=0.1)

    def test_silence_does_not_pull_it_down(self):
        tone = _tone(440.0, seconds=1.0, amp=0.2)
        padded = np.concatenate([np.zeros(int(SR * 3)), tone,
                                 np.zeros(int(SR * 3))])
        assert speech_level_db(_sound(padded)) == pytest.approx(
            speech_level_db(_sound(tone)), abs=0.2)

    def test_follows_the_recording_gain(self):
        x = _noise(seconds=1.0)
        assert speech_level_db(_sound(2 * x)) - speech_level_db(_sound(x)) \
            == pytest.approx(20 * math.log10(2), abs=1e-6)

    def test_is_nan_when_too_short_or_silent(self):
        assert np.isnan(speech_level_db(_sound(_noise(seconds=0.01))))
        assert np.isnan(speech_level_db(_sound(np.zeros(int(SR)))))


class TestClipping:
    @pytest.mark.parametrize("freq", [100.0, 440.0, 3000.0])
    def test_a_clean_sine_is_not_clipped(self, freq):
        assert clipped_percent(_sound(_tone(freq, seconds=0.5))) == 0.0

    def test_a_hard_clipped_sine_is(self):
        x = np.clip(_tone(440.0, seconds=0.5, amp=1.0), -0.5, 0.5)
        # A sine above half its amplitude a third of the time, both signs
        assert clipped_percent(_sound(x)) == pytest.approx(66.7, abs=1.0)

    def test_clipping_survives_later_normalisation(self):
        x = np.clip(_tone(440.0, seconds=0.5, amp=1.0), -0.5, 0.5)
        assert clipped_percent(_sound(0.3 * x)) == pytest.approx(
            clipped_percent(_sound(x)))

    def test_a_lone_spike_at_the_peak_is_not_a_flat_top(self):
        x = _noise(seconds=0.5)
        x[1000] = 1.0
        x[1001] = 1.0
        assert clipped_percent(_sound(x)) == 0.0

    def test_one_sided_clipping_counts(self):
        x = _tone(440.0, seconds=0.5, amp=1.0)
        x = np.minimum(x, 0.5)
        assert clipped_percent(_sound(x)) == pytest.approx(33.3, abs=1.0)

    def test_is_nan_for_silence(self):
        assert np.isnan(clipped_percent(_sound(np.zeros(1000))))


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

DUR = 1.0


def _corpus(tmp_path, signals):
    """One file per named signal, each with an 's' interval at 0.3–0.7 s."""
    audio_dir = tmp_path / "audio"
    tg_dir = tmp_path / "tg"
    audio_dir.mkdir()
    tg_dir.mkdir()
    for name, x in signals.items():
        path = audio_dir / f"{name}.wav"
        if x is None:
            path.write_bytes(b"not a wav file")
        else:
            _sound(x).save(str(path), parselmouth.SoundFileFormat.WAV)
        tier = Tier("phones", "IntervalTier", 0.0, DUR,
                    intervals=[Interval(0.3, 0.7, "s")])
        TextGrid(0.0, DUR, [tier]).save(str(tg_dir / f"{name}.TextGrid"))
    return str(audio_dir), str(tg_dir)


def _export(dirs, **kwargs):
    audio_dir, tg_dir = dirs
    opts = dict(
        selected_tiers=[Tier("phones", "IntervalTier", 0.0, DUR,
                             intervals=[])],
        extract_formants=False, formant_mode="at_points",
        point_tier_name=None, segment_tier_name="phones",
        percentage_markers=[], extract_durations=False,
        duration_tier_names=[], primary_tier_name="phones",
        spectral_tier_name="phones", spectral_markers=[50.0],
        extract_spectral=True,
    )
    opts.update(kwargs)
    return _build_csv_data(audio_dir, tg_dir, None, **opts)


def _by_file(headers, rows):
    return {r[0]: dict(zip(headers, r)) for r in rows}


# Kept under full scale so the louder copy is not itself clipped
_BASE = _noise(seconds=DUR, sigma=0.05, seed=7)


class TestCSV:
    def test_file_columns_follow_the_filename(self, tmp_path):
        headers, _ = _export(_corpus(tmp_path, {"a": _BASE}))
        assert tuple(headers[1:1 + len(FILE_SIGNAL_COLUMNS)]) == \
            FILE_SIGNAL_COLUMNS

    def test_every_marker_column_is_present(self, tmp_path):
        headers, _ = _export(_corpus(tmp_path, {"a": _BASE}))
        for col in SPECTRAL_MARKER_COLUMNS:
            assert f"{col}_50%" in headers

    def test_hbrel_is_hblevel_less_the_speech_level(self, tmp_path):
        headers, rows = _export(_corpus(tmp_path, {"a": _BASE}))
        r = _by_file(headers, rows)["a"]
        assert float(r["hbrel_50%"]) == pytest.approx(
            float(r["hblevel_50%"]) - float(r["speech_level_db"]), abs=0.11)

    def test_gain_moves_the_level_but_not_the_relative_level(self, tmp_path):
        headers, rows = _export(_corpus(
            tmp_path, {"quiet": _BASE, "loud": 4 * _BASE}))
        r = _by_file(headers, rows)
        step = 20 * math.log10(4)
        assert float(r["loud"]["hblevel_50%"]) - float(
            r["quiet"]["hblevel_50%"]) == pytest.approx(step, abs=0.11)
        assert float(r["loud"]["hbrel_50%"]) == pytest.approx(
            float(r["quiet"]["hbrel_50%"]), abs=0.11)
        assert r["loud"]["flatness_50%"] == r["quiet"]["flatness_50%"]

    def test_clipping_is_reported_per_file(self, tmp_path):
        clipped = np.clip(_tone(440.0, seconds=DUR, amp=1.0), -0.5, 0.5)
        headers, rows = _export(_corpus(
            tmp_path, {"clean": _BASE, "clipped": clipped}))
        r = _by_file(headers, rows)
        assert float(r["clean"]["clipped_pct"]) == 0.0
        assert float(r["clipped"]["clipped_pct"]) > 50.0

    def test_too_short_blanks_every_measure(self, tmp_path):
        dirs = _corpus(tmp_path, {"a": _BASE})
        headers, rows = _export(dirs, spectral_window_mode="fixed",
                                spectral_window_ms=500.0)
        r = _by_file(headers, rows)["a"]
        assert r["winsource_50%"] == "too_short"
        for col in SPECTRAL_TRACK_NAMES + ("hbrel", "winms", "nsamples"):
            assert r[f"{col}_50%"] == ""

    def test_an_unreadable_recording_keeps_its_row_aligned(self, tmp_path):
        """Every row is as wide as the header, even with no audio to measure.

        The blank for a missing sound once came up one cell short per
        marker, shifting every later column of that row.
        """
        headers, rows = _export(_corpus(
            tmp_path, {"good": _BASE, "broken": None}),
            spectral_markers=[20.0, 50.0, 80.0], extract_durations=True,
            duration_tier_names=["phones"])
        assert rows
        for row in rows:
            assert len(row) == len(headers)

    def test_trajectories_can_track_flatness(self, tmp_path):
        headers, rows = _export(_corpus(tmp_path, {"a": _BASE}),
                                extract_spectral=False,
                                extract_trajectory=True,
                                traj_moments=["flatness", "hblevel"])
        r = _by_file(headers, rows)["a"]
        assert 0.0 < float(r["flatness_t0"]) <= 1.0
        assert r["hblevel_k0"] != ""
