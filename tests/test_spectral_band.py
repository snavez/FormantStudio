"""
Tests for the stated analysis band and the band-energy ratio: the band decides
what a moment is a moment of, and the ratio indexes high-frequency energy
without describing the spectrum's shape.
"""

import math
import os
import sys

import numpy as np
import parselmouth
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import formant_editor as fe
from formant_editor import (
    band_energy_ratio, band_mask, measure_window, spectrum_of,
)


def _tone_sound(freqs, sr=44100, dur=0.2, amps=None):
    """A sound made of pure tones, so its spectrum is known in advance."""
    t = np.arange(int(sr * dur)) / sr
    amps = amps or [1.0] * len(freqs)
    x = sum(a * np.sin(2 * np.pi * f * t) for f, a in zip(freqs, amps))
    return parselmouth.Sound(x, sampling_frequency=sr)


# ---------------------------------------------------------------------------
# The band
# ---------------------------------------------------------------------------

def test_band_mask_keeps_only_what_is_inside():
    freqs = np.array([0.0, 100.0, 1000.0, 5000.0, 12000.0])
    m = band_mask(freqs, 100.0, 5000.0)
    assert list(freqs[m]) == [100.0, 1000.0, 5000.0]


def test_band_mask_open_ended():
    freqs = np.array([0.0, 500.0, 9000.0])
    assert band_mask(freqs, 0, 0).all()          # no limits set


@pytest.mark.parametrize("estimator", ["single_taper", "multitaper"])
def test_band_excludes_energy_above_the_ceiling(estimator):
    """A tone above the ceiling must not pull the centre of gravity up."""
    snd = _tone_sound([1000.0, 10000.0])
    inside = measure_window(snd, 0.02, 0.18, estimator=estimator,
                            band_low=0, band_high=5000)[0]
    across = measure_window(snd, 0.02, 0.18, estimator=estimator,
                            band_low=0, band_high=15000)[0]
    assert inside == pytest.approx(1000.0, abs=120)
    assert across > inside + 1000


def test_band_floor_excludes_low_energy():
    snd = _tone_sound([200.0, 6000.0])
    above = measure_window(snd, 0.02, 0.18, band_low=1000, band_high=12000)[0]
    assert above == pytest.approx(6000.0, abs=200)


@pytest.mark.parametrize("estimator", ["single_taper", "multitaper"])
def test_both_estimators_agree_within_tolerance(estimator):
    """The two differ in how the spectrum is estimated, not in what a moment
    means, so they should not disagree much on a clean signal."""
    snd = _tone_sound([3000.0])
    cog = measure_window(snd, 0.02, 0.18, estimator=estimator,
                         band_low=0, band_high=12000)[0]
    assert cog == pytest.approx(3000.0, abs=200)


def test_spectrum_of_returns_nothing_for_an_empty_region():
    snd = _tone_sound([1000.0])
    assert spectrum_of(snd, 0.1, 0.1) == (None, None)


# ---------------------------------------------------------------------------
# The band-energy ratio
# ---------------------------------------------------------------------------

def test_ratio_is_positive_when_the_high_band_dominates():
    snd = _tone_sound([500.0, 5000.0], amps=[0.1, 1.0])
    r = measure_window(snd, 0.02, 0.18,
                       ratio_low=(0, 2000), ratio_high=(2000, 8000))[4]
    assert r > 6


def test_ratio_is_negative_when_the_low_band_dominates():
    snd = _tone_sound([500.0, 5000.0], amps=[1.0, 0.1])
    r = measure_window(snd, 0.02, 0.18,
                       ratio_low=(0, 2000), ratio_high=(2000, 8000))[4]
    assert r < -6


def test_ratio_tracks_the_amplitude_difference_in_db():
    """Ten times the amplitude is twenty dB of power, so the ratio should
    move by about that much."""
    quiet = _tone_sound([500.0, 5000.0], amps=[1.0, 0.1])
    loud = _tone_sound([500.0, 5000.0], amps=[1.0, 1.0])
    r_q = measure_window(quiet, 0.02, 0.18,
                         ratio_low=(0, 2000), ratio_high=(2000, 8000))[4]
    r_l = measure_window(loud, 0.02, 0.18,
                         ratio_low=(0, 2000), ratio_high=(2000, 8000))[4]
    assert r_l - r_q == pytest.approx(20.0, abs=3.0)


def test_ratio_stays_finite_on_silence():
    """A near-silent window must not produce an infinity."""
    snd = parselmouth.Sound(np.zeros(4410), sampling_frequency=44100)
    r = band_energy_ratio(*spectrum_of(snd, 0.01, 0.09),
                          (0, 2000), (2000, 8000))
    assert np.isnan(r) or math.isfinite(r)


def test_ratio_is_nan_without_a_spectrum():
    assert np.isnan(band_energy_ratio(None, None, (0, 2000), (2000, 8000)))


# ---------------------------------------------------------------------------
# What measure_window promises
# ---------------------------------------------------------------------------

def test_measure_window_returns_four_moments_and_a_ratio():
    vals = measure_window(_tone_sound([2000.0]), 0.02, 0.18)
    assert len(vals) == 5
    assert all(np.isfinite(v) for v in vals[:2])


def test_measure_window_is_all_nan_for_an_impossible_region():
    assert all(np.isnan(v) for v in measure_window(_tone_sound([1000.0]),
                                                   0.1, 0.05))


def test_the_ratio_is_tracked_alongside_the_moments():
    assert fe.BAND_RATIO_NAME in fe.SPECTRAL_TRACK_NAMES
    assert set(fe.SPECTRAL_MOMENT_NAMES) < set(fe.SPECTRAL_TRACK_NAMES)


def test_trajectory_carries_a_ratio_track():
    snd = _tone_sound([3000.0], dur=0.4)
    tracks = fe.compute_spectral_trajectory(snd, 0.05, 0.35)
    assert tracks is not None
    assert fe.BAND_RATIO_NAME in tracks


# ---------------------------------------------------------------------------
# The two ratio bands are independent
# ---------------------------------------------------------------------------

def test_ratio_bands_need_not_touch():
    """A gap between the bands ignores whatever falls in it."""
    snd = _tone_sound([500.0, 2500.0, 6000.0], amps=[1.0, 5.0, 1.0])
    touching = measure_window(snd, 0.02, 0.18,
                              ratio_low=(0, 2000), ratio_high=(2000, 8000))[4]
    with_gap = measure_window(snd, 0.02, 0.18,
                              ratio_low=(0, 1000), ratio_high=(4000, 8000))[4]
    # The loud 2500 Hz tone counts as high-band energy when the bands touch,
    # and is excluded entirely when the gap steps over it.
    assert touching > with_gap + 10


def test_ratio_low_band_need_not_start_at_zero():
    """With a high-pass already applied, starting above zero is reasonable."""
    snd = _tone_sound([100.0, 1500.0, 6000.0], amps=[10.0, 1.0, 1.0])
    from_zero = measure_window(snd, 0.02, 0.18,
                               ratio_low=(0, 2000), ratio_high=(2000, 8000))[4]
    above = measure_window(snd, 0.02, 0.18,
                           ratio_low=(1000, 2000), ratio_high=(2000, 8000))[4]
    # Excluding the very loud 100 Hz tone shrinks the denominator, so the
    # ratio rises.
    assert above > from_zero + 10


def test_ratio_bands_may_be_placed_anywhere():
    snd = _tone_sound([3000.0, 9000.0], amps=[1.0, 1.0])
    r = measure_window(snd, 0.02, 0.18,
                       ratio_low=(2500, 3500), ratio_high=(8500, 9500))[4]
    assert np.isfinite(r) and abs(r) < 6      # two equal tones, near 0 dB
