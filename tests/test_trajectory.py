"""
Tests for the spectral trajectory / DCT pass.

Covers the sliding-frame geometry, time normalisation, DCT reduction and
its sign conventions, the wide CSV columns, and the degenerate cases
(segment too short, mostly-unmeasurable track).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import parselmouth

from formant_editor import (
    Interval,
    TextGrid,
    Tier,
    _build_csv_data,
    compute_spectral_trajectory,
    dct_coefficients,
)

SR = 44100


def _chirp(f_start, f_end, dur=0.12, sr=SR):
    """A linear frequency sweep — COG should track the sweep."""
    t = np.arange(0, dur, 1.0 / sr)
    phase = 2 * np.pi * (f_start * t + (f_end - f_start) / (2 * dur) * t ** 2)
    return parselmouth.Sound(0.8 * np.sin(phase), sampling_frequency=sr)


class TestTrajectoryGeometry:
    def test_rising_chirp_gives_rising_cog_track(self):
        snd = _chirp(2000.0, 8000.0)
        tracks = compute_spectral_trajectory(
            snd, 0.0, 0.12, win_ms=6.0, hop_ms=1.0, norm_points=11,
            estimator="multitaper")
        cog = tracks["COG"]
        assert cog.size == 11
        assert np.all(np.diff(cog) > 0)          # monotonically rising
        assert cog[0] < 3500 and cog[-1] > 7000  # spans the sweep

    def test_norm_points_controls_track_length(self):
        snd = _chirp(2000.0, 6000.0)
        for n in (3, 5, 20):
            tracks = compute_spectral_trajectory(
                snd, 0.0, 0.12, win_ms=6.0, hop_ms=1.0, norm_points=n,
                estimator="multitaper")
            assert tracks["COG"].size == n

    def test_all_four_moments_returned(self):
        snd = _chirp(2000.0, 6000.0)
        tracks = compute_spectral_trajectory(
            snd, 0.0, 0.12, win_ms=6.0, hop_ms=1.0, norm_points=5,
            estimator="multitaper")
        assert set(tracks) == {"COG", "SD", "skew", "kurt"}

    def test_segment_too_short_returns_none(self):
        snd = _chirp(2000.0, 6000.0, dur=0.05)
        # a 6 ms window with 1 ms hop needs room for >= 3 frames
        assert compute_spectral_trajectory(
            snd, 0.0, 0.007, win_ms=6.0, hop_ms=1.0, norm_points=11) is None

    def test_frames_stay_inside_the_segment(self):
        # Sweep only in the middle of a longer sound; analysing the middle
        # span must not pull in the flanking silence.
        sr = SR
        t = np.arange(0, 0.30, 1.0 / sr)
        sig = np.zeros_like(t)
        mid = (t >= 0.10) & (t < 0.22)
        tm = t[mid] - 0.10
        sig[mid] = 0.8 * np.sin(2 * np.pi * 6000.0 * tm)
        snd = parselmouth.Sound(sig, sampling_frequency=sr)
        tracks = compute_spectral_trajectory(
            snd, 0.10, 0.22, win_ms=6.0, hop_ms=1.0, norm_points=7,
            estimator="multitaper")
        cog = tracks["COG"]
        # every frame sits on the 6 kHz tone, not the surrounding silence
        assert np.all(np.abs(cog - 6000.0) < 400.0)


class TestDCT:
    def test_k0_is_mean_times_sqrt_n(self):
        track = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        k = dct_coefficients(track, 3)
        assert k[0] == pytest.approx(track.mean() * np.sqrt(track.size))

    def test_k1_sign_flips_with_slope(self):
        rising = dct_coefficients(np.linspace(1000.0, 5000.0, 11), 2)
        falling = dct_coefficients(np.linspace(5000.0, 1000.0, 11), 2)
        # DCT-II: a rising track gives a negative k1, falling positive
        assert rising[1] < 0 < falling[1]
        assert rising[1] == pytest.approx(-falling[1])

    def test_flat_track_has_zero_slope_and_curvature(self):
        k = dct_coefficients(np.full(11, 4200.0), 4)
        assert k[0] == pytest.approx(4200.0 * np.sqrt(11))
        assert np.allclose(k[1:], 0.0, atol=1e-9)

    def test_requesting_more_coeffs_than_points_pads_with_nan(self):
        k = dct_coefficients(np.array([1.0, 2.0, 3.0]), 5)
        assert k.size == 5
        assert np.all(np.isfinite(k[:3])) and np.all(np.isnan(k[3:]))


def _corpus(tmp_path, intervals, dur=0.6):
    """Noise WAV + TextGrid with a 'phones' tier."""
    audio_dir = tmp_path / "audio"
    tg_dir = tmp_path / "tg"
    audio_dir.mkdir()
    tg_dir.mkdir()
    rng = np.random.default_rng(4)
    parselmouth.Sound(rng.standard_normal(int(SR * dur)) * 0.2,
                      sampling_frequency=SR).save(
        str(audio_dir / "test.wav"), parselmouth.SoundFileFormat.WAV)
    tier = Tier("phones", "IntervalTier", 0.0, dur,
                intervals=[Interval(a, b, t) for a, b, t in intervals])
    TextGrid(0.0, dur, [tier]).save(str(tg_dir / "test.TextGrid"))
    tiers = TextGrid.from_file(str(tg_dir / "test.TextGrid")).tiers
    return str(audio_dir), str(tg_dir), tiers


def _kwargs(**over):
    base = dict(
        audio_dir=None, textgrid_dir=None, formants_dir=None,
        selected_tiers=None, extract_formants=False, formant_mode=None,
        point_tier_name=None, segment_tier_name=None, percentage_markers=[],
        extract_durations=False, duration_tier_names=[],
        extract_spectral=True, spectral_markers=[50],
        spectral_tier_name="phones", spectral_highpass_hz=0.0,
        extract_trajectory=True, traj_moments=["COG", "SD"],
        traj_norm_points=5, traj_dct_coeffs=3,
    )
    base.update(over)
    return base


class TestTrajectoryColumns:
    def test_wide_columns_in_canonical_order(self, tmp_path):
        audio, tg, tiers = _corpus(tmp_path, [(0.1, 0.4, "s")])
        headers, _ = _build_csv_data(**_kwargs(
            audio_dir=audio, textgrid_dir=tg, selected_tiers=tiers))
        i = headers.index("COG_k0")
        assert headers[i:i + 3] == ["COG_k0", "COG_k1", "COG_k2"]
        assert headers[i + 3:i + 8] == [f"COG_t{n}" for n in range(5)]
        # SD block follows COG, and no unrequested moments appear
        assert "SD_k0" in headers
        assert "skew_k0" not in headers and "kurt_k0" not in headers

    def test_moment_order_is_canonical_regardless_of_input(self, tmp_path):
        audio, tg, tiers = _corpus(tmp_path, [(0.1, 0.4, "s")])
        headers, _ = _build_csv_data(**_kwargs(
            audio_dir=audio, textgrid_dir=tg, selected_tiers=tiers,
            traj_moments=["kurt", "COG"]))
        assert headers.index("COG_k0") < headers.index("kurt_k0")

    def test_track_columns_omitted_when_not_requested(self, tmp_path):
        audio, tg, tiers = _corpus(tmp_path, [(0.1, 0.4, "s")])
        headers, rows = _build_csv_data(**_kwargs(
            audio_dir=audio, textgrid_dir=tg, selected_tiers=tiers,
            traj_include_track=False))
        assert "COG_k0" in headers
        assert "COG_t0" not in headers
        assert rows[0][headers.index("COG_k0")] != ""

    def test_values_populated_for_a_normal_segment(self, tmp_path):
        audio, tg, tiers = _corpus(tmp_path, [(0.1, 0.4, "s")])
        headers, rows = _build_csv_data(**_kwargs(
            audio_dir=audio, textgrid_dir=tg, selected_tiers=tiers))
        row = rows[0]
        assert row[headers.index("COG_k0")] != ""
        for n in range(5):
            assert row[headers.index(f"COG_t{n}")] != ""

    def test_short_segment_blanks_trajectory_but_keeps_row(self, tmp_path):
        # 4 ms segment cannot host 6 ms frames -> blanks, row still emitted
        audio, tg, tiers = _corpus(tmp_path, [(0.20, 0.204, "t")])
        headers, rows = _build_csv_data(**_kwargs(
            audio_dir=audio, textgrid_dir=tg, selected_tiers=tiers))
        assert len(rows) == 1
        assert rows[0][headers.index("COG_k0")] == ""
        assert rows[0][headers.index("COG_t0")] == ""

    def test_trajectory_off_emits_no_columns(self, tmp_path):
        audio, tg, tiers = _corpus(tmp_path, [(0.1, 0.4, "s")])
        headers, _ = _build_csv_data(**_kwargs(
            audio_dir=audio, textgrid_dir=tg, selected_tiers=tiers,
            extract_trajectory=False))
        assert not [h for h in headers if "_k0" in h or "_t0" in h]
