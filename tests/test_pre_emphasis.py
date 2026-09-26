"""
Tests for the pre-emphasis applied to the spectral-analysis waveform.

Covers the corner-to-coefficient conversion, the filter itself, and how the
setting reaches the CSV: applied before any window for both the moments and
the trajectories, and recorded per file as ``pre_emphasis_alpha``.
"""

import math
import os
import sys

import numpy as np
import parselmouth
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from formant_editor import (  # noqa: E402
    Interval, Tier, TextGrid, _build_csv_data, pre_emphasis_alpha,
    pre_emphasize,
)

SR = 44100.0
DUR = 0.6


def _noise(n=4410, seed=0):
    return np.random.default_rng(seed).standard_normal(n) * 0.2


class TestAlpha:
    def test_follows_praats_definition(self):
        assert pre_emphasis_alpha(50.0, SR) == pytest.approx(
            math.exp(-2 * math.pi * 50.0 / SR))

    def test_a_142_hz_corner_is_0_98_at_44_1_khz(self):
        assert pre_emphasis_alpha(141.797, SR) == pytest.approx(0.98, abs=1e-5)

    def test_the_same_corner_differs_by_sampling_rate(self):
        assert pre_emphasis_alpha(100.0, 16000.0) < pre_emphasis_alpha(
            100.0, SR)

    @pytest.mark.parametrize("off", [0.0, -5.0, None])
    def test_is_zero_when_off(self, off):
        assert pre_emphasis_alpha(off, SR) == 0.0


class TestPreEmphasize:
    def test_applies_the_first_difference_without_rescaling(self):
        x = _noise()
        snd = parselmouth.Sound(x, sampling_frequency=SR)
        y = pre_emphasize(snd, 50.0).values[0]
        a = pre_emphasis_alpha(50.0, SR)
        assert y[0] == pytest.approx(x[0])
        np.testing.assert_allclose(y[1:], x[1:] - a * x[:-1], atol=1e-12)

    def test_off_returns_the_same_sound(self):
        snd = parselmouth.Sound(_noise(), sampling_frequency=SR)
        assert pre_emphasize(snd, 0.0) is snd

    def test_keeps_rate_and_start_time(self):
        snd = parselmouth.Sound(_noise(), sampling_frequency=SR,
                                start_time=1.5)
        out = pre_emphasize(snd, 50.0)
        assert out.sampling_frequency == SR
        assert out.xmin == pytest.approx(1.5)

    def test_lifts_high_frequencies_relative_to_low(self):
        t = np.arange(int(SR * 0.2)) / SR
        low = parselmouth.Sound(np.sin(2 * np.pi * 200 * t),
                                sampling_frequency=SR)
        high = parselmouth.Sound(np.sin(2 * np.pi * 6000 * t),
                                 sampling_frequency=SR)

        def gain(snd):
            return (np.std(pre_emphasize(snd, 100.0).values)
                    / np.std(snd.values))

        assert gain(high) > 10 * gain(low)

    def test_does_not_touch_the_input(self):
        x = _noise()
        snd = parselmouth.Sound(x.copy(), sampling_frequency=SR)
        pre_emphasize(snd, 50.0)
        np.testing.assert_array_equal(snd.values[0], x)


@pytest.fixture
def corpus(tmp_path):
    """One pink-ish (low-heavy) recording with a single 's' interval."""
    audio_dir = tmp_path / "audio"
    tg_dir = tmp_path / "tg"
    audio_dir.mkdir()
    tg_dir.mkdir()
    x = np.cumsum(_noise(int(SR * DUR)))
    x = 0.5 * x / np.abs(x).max()
    parselmouth.Sound(x, sampling_frequency=SR).save(
        str(audio_dir / "a.wav"), parselmouth.SoundFileFormat.WAV)
    tier = Tier("phones", "IntervalTier", 0.0, DUR,
                intervals=[Interval(0.1, 0.5, "s")])
    TextGrid(0.0, DUR, [tier]).save(str(tg_dir / "a.TextGrid"))
    return str(audio_dir), str(tg_dir)


def _export(corpus, **kwargs):
    audio_dir, tg_dir = corpus
    opts = dict(
        selected_tiers=[Tier("phones", "IntervalTier", 0.0, DUR,
                             intervals=[])],
        extract_formants=False, formant_mode="at_points",
        point_tier_name=None, segment_tier_name="phones",
        percentage_markers=[], extract_durations=False,
        duration_tier_names=[], primary_tier_name="phones",
        spectral_tier_name="phones", spectral_markers=[50.0],
    )
    opts.update(kwargs)
    return _build_csv_data(audio_dir, tg_dir, None, **opts)


def _cell(headers, rows, name):
    return rows[0][headers.index(name)]


class TestCSV:
    def test_the_alpha_column_sits_beside_the_cutoff(self, corpus):
        headers, _ = _export(corpus, extract_spectral=True)
        assert headers[:3] == ["filename", "hf_cutoff_hz",
                               "pre_emphasis_alpha"]

    def test_off_is_written_as_zero_not_left_blank(self, corpus):
        headers, rows = _export(corpus, extract_spectral=True)
        assert float(_cell(headers, rows, "pre_emphasis_alpha")) == 0.0

    def test_on_writes_the_files_own_alpha(self, corpus):
        headers, rows = _export(corpus, extract_spectral=True,
                                spectral_preemph_hz=141.797)
        assert float(_cell(headers, rows, "pre_emphasis_alpha")) == \
            pytest.approx(0.98, abs=1e-5)

    def test_an_export_without_spectra_has_no_alpha_column(self, corpus):
        headers, _ = _export(corpus, extract_durations=True,
                             duration_tier_names=["phones"])
        assert "pre_emphasis_alpha" not in headers

    def test_raises_the_moments_centre_of_gravity(self, corpus):
        h0, r0 = _export(corpus, extract_spectral=True)
        h1, r1 = _export(corpus, extract_spectral=True,
                         spectral_preemph_hz=50.0)
        assert float(_cell(h1, r1, "COG_50%")) > \
            float(_cell(h0, r0, "COG_50%")) + 200.0

    def test_reaches_the_trajectories_too(self, corpus):
        # k0 of an orthonormal DCT is the track's scaled mean
        kw = dict(extract_trajectory=True, traj_moments=["COG"],
                  traj_include_track=False)
        h0, r0 = _export(corpus, **kw)
        h1, r1 = _export(corpus, spectral_preemph_hz=50.0, **kw)
        k0 = next(i for i, h in enumerate(h0)
                  if h.startswith("COG") and h.endswith("_k0"))
        assert h0 == h1
        assert float(r1[0][k0]) > float(r0[0][k0])
