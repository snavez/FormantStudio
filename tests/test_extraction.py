"""
Tests for formant extraction and spectrogram computation — spec sections 6.4 & 6.5.

Uses synthetic audio signals generated via Parselmouth/NumPy.
"""

import numpy as np
import numpy.testing as npt
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from formant_editor import FormantData, extract_formants_from_praat

import parselmouth


# ---------------------------------------------------------------------------
# Fixtures — synthetic audio signals
# ---------------------------------------------------------------------------

@pytest.fixture
def vowel_sound():
    """Synthetic vowel-like signal (~0.5s, 44100 Hz, with harmonics)."""
    sr = 44100
    duration = 0.5
    t = np.arange(0, duration, 1 / sr)
    f0 = 120.0
    signal = np.zeros_like(t)
    for h in range(1, 31):
        signal += (1.0 / h) * np.sin(2 * np.pi * f0 * h * t)
    signal = signal / np.max(np.abs(signal)) * 0.8
    return parselmouth.Sound(signal, sampling_frequency=sr)


@pytest.fixture
def silence_sound():
    """Silent audio (0.5s)."""
    sr = 44100
    signal = np.zeros(int(sr * 0.5))
    return parselmouth.Sound(signal, sampling_frequency=sr)


@pytest.fixture
def pure_tone_sound():
    """1000 Hz pure tone (0.5s)."""
    sr = 44100
    duration = 0.5
    t = np.arange(0, duration, 1 / sr)
    signal = 0.8 * np.sin(2 * np.pi * 1000.0 * t)
    return parselmouth.Sound(signal, sampling_frequency=sr)


@pytest.fixture
def short_sound():
    """Very short audio (~10 ms)."""
    sr = 44100
    signal = np.random.randn(int(sr * 0.01)) * 0.1
    return parselmouth.Sound(signal, sampling_frequency=sr)


@pytest.fixture
def high_sr_sound():
    """48 kHz sample rate vowel-like signal."""
    sr = 48000
    duration = 0.5
    t = np.arange(0, duration, 1 / sr)
    f0 = 120.0
    signal = np.zeros_like(t)
    for h in range(1, 20):
        signal += (1.0 / h) * np.sin(2 * np.pi * f0 * h * t)
    signal = signal / np.max(np.abs(signal)) * 0.8
    return parselmouth.Sound(signal, sampling_frequency=sr)


@pytest.fixture
def low_sr_sound():
    """8 kHz sample rate signal."""
    sr = 8000
    duration = 0.5
    t = np.arange(0, duration, 1 / sr)
    f0 = 120.0
    signal = np.zeros_like(t)
    for h in range(1, 10):
        signal += (1.0 / h) * np.sin(2 * np.pi * f0 * h * t)
    signal = signal / np.max(np.abs(signal)) * 0.8
    return parselmouth.Sound(signal, sampling_frequency=sr)


# ===================================================================
# 6.4.1 Praat Equivalence
# ===================================================================

class TestPraatEquivalence:

    def test_extraction_returns_formant_data(self, vowel_sound):
        result = extract_formants_from_praat(vowel_sound)
        assert isinstance(result, FormantData)

    def test_extraction_frame_count(self, vowel_sound):
        formant_obj = vowel_sound.to_formant_burg()
        result = extract_formants_from_praat(vowel_sound)
        assert result.n_frames == formant_obj.n_frames

    def test_extraction_times_match_praat(self, vowel_sound):
        formant_obj = vowel_sound.to_formant_burg()
        result = extract_formants_from_praat(vowel_sound)
        praat_times = np.array(formant_obj.xs())
        npt.assert_array_almost_equal(result.times, praat_times)

    def test_extraction_values_match_praat(self, vowel_sound):
        formant_obj = vowel_sound.to_formant_burg()
        result = extract_formants_from_praat(vowel_sound)
        for f_idx in range(5):
            for t_idx in range(result.n_frames):
                try:
                    praat_val = formant_obj.get_value_at_time(
                        f_idx + 1, result.times[t_idx]
                    )
                except Exception:
                    praat_val = np.nan
                result_val = result.values[f_idx, t_idx]
                if praat_val is None or np.isnan(praat_val) or praat_val <= 0:
                    assert np.isnan(result_val), \
                        f"F{f_idx+1} frame {t_idx}: expected NaN, got {result_val}"
                else:
                    assert result_val == pytest.approx(praat_val, rel=1e-6), \
                        f"F{f_idx+1} frame {t_idx}: expected {praat_val}, got {result_val}"

    def test_extraction_default_3_formants(self, vowel_sound):
        result = extract_formants_from_praat(vowel_sound, n_formants=3)
        # F4 and F5 should be all NaN
        assert np.all(np.isnan(result.values[3]))
        assert np.all(np.isnan(result.values[4]))
        # F1 should have some non-NaN values
        assert np.any(~np.isnan(result.values[0]))

    def test_extraction_5_formants(self, vowel_sound):
        result = extract_formants_from_praat(vowel_sound, n_formants=5)
        # At least F1–F3 should have some data
        for f in range(3):
            assert np.any(~np.isnan(result.values[f])), f"F{f+1} has no data"

    def test_extraction_time_step_computed(self, vowel_sound):
        result = extract_formants_from_praat(vowel_sound)
        if result.n_frames >= 2:
            expected_step = result.times[1] - result.times[0]
            assert result.time_step == pytest.approx(expected_step)


# ===================================================================
# 6.4.2 Parameter Sensitivity
# ===================================================================

class TestParameterSensitivity:

    def test_max_formant_affects_values(self, vowel_sound):
        r1 = extract_formants_from_praat(vowel_sound, max_formant=5500.0)
        r2 = extract_formants_from_praat(vowel_sound, max_formant=5000.0)
        # Different max_formant can change frame count, so compare over
        # the shorter common length
        n = min(r1.n_frames, r2.n_frames)
        v1 = r1.values[0, :n]
        v2 = r2.values[0, :n]
        mask = ~np.isnan(v1) & ~np.isnan(v2)
        if np.any(mask):
            assert not np.allclose(v1[mask], v2[mask])

    def test_window_length_affects_frame_count(self, vowel_sound):
        r1 = extract_formants_from_praat(vowel_sound, window_length=0.025)
        r2 = extract_formants_from_praat(vowel_sound, window_length=0.010)
        # Shorter window → more frames
        assert r2.n_frames > r1.n_frames

    def test_pre_emphasis_affects_values(self, vowel_sound):
        r1 = extract_formants_from_praat(vowel_sound, pre_emphasis=50.0)
        r2 = extract_formants_from_praat(vowel_sound, pre_emphasis=150.0)
        mask1 = ~np.isnan(r1.values[0])
        mask2 = ~np.isnan(r2.values[0])
        common = mask1 & mask2
        if np.any(common):
            assert not np.allclose(r1.values[0][common], r2.values[0][common])

    def test_time_step_none_uses_auto(self, vowel_sound):
        # time_step=0.0 should be converted to None (auto)
        result = extract_formants_from_praat(vowel_sound, time_step=0.0)
        assert result.n_frames > 0


# ===================================================================
# 6.4.3 Edge Cases
# ===================================================================

class TestExtractionEdgeCases:

    def test_extraction_silence(self, silence_sound):
        result = extract_formants_from_praat(silence_sound)
        assert isinstance(result, FormantData)
        # Most values should be NaN for silence
        nan_ratio = np.sum(np.isnan(result.values)) / result.values.size
        assert nan_ratio > 0.5

    def test_extraction_very_short_audio(self, short_sound):
        result = extract_formants_from_praat(short_sound)
        assert isinstance(result, FormantData)
        # Should not crash; may have 0 or 1 frames
        assert result.n_frames >= 0

    def test_extraction_high_sample_rate(self, high_sr_sound):
        result = extract_formants_from_praat(high_sr_sound)
        assert isinstance(result, FormantData)
        assert result.n_frames > 0

    def test_extraction_low_sample_rate(self, low_sr_sound):
        result = extract_formants_from_praat(low_sr_sound, max_formant=3500.0)
        assert isinstance(result, FormantData)
        assert result.n_frames > 0


# ===================================================================
# 6.5 Spectrogram Computation
# ===================================================================

class TestSpectrogram:

    def test_spectrogram_shape(self, vowel_sound):
        spec = vowel_sound.to_spectrogram(
            window_length=0.005, maximum_frequency=5500.0,
            time_step=0.002, frequency_step=20.0,
        )
        vals = spec.values
        freqs = np.array(spec.ys())
        times = np.array(spec.xs())
        assert vals.shape == (len(freqs), len(times))

    def test_spectrogram_freq_axis(self, vowel_sound):
        max_freq = 5500.0
        spec = vowel_sound.to_spectrogram(
            window_length=0.005, maximum_frequency=max_freq,
            time_step=0.002, frequency_step=20.0,
        )
        freqs = np.array(spec.ys())
        assert freqs[0] >= 0
        assert freqs[-1] <= max_freq + 100  # some tolerance

    def test_spectrogram_time_axis(self, vowel_sound):
        spec = vowel_sound.to_spectrogram(
            window_length=0.005, maximum_frequency=5500.0,
            time_step=0.002, frequency_step=20.0,
        )
        times = np.array(spec.xs())
        assert times[0] >= 0
        assert times[-1] <= vowel_sound.duration + 0.01

    def test_spectrogram_pure_tone(self, pure_tone_sound):
        spec = pure_tone_sound.to_spectrogram(
            window_length=0.005, maximum_frequency=5500.0,
            time_step=0.002, frequency_step=20.0,
        )
        freqs = np.array(spec.ys())
        vals = spec.values
        # Average power across time
        avg_power = np.mean(vals, axis=1)
        # Peak should be near 1000 Hz
        peak_freq = freqs[np.argmax(avg_power)]
        assert abs(peak_freq - 1000.0) < 100.0, \
            f"Peak at {peak_freq} Hz, expected ~1000 Hz"

    def test_spectrogram_silence(self, silence_sound):
        spec = silence_sound.to_spectrogram(
            window_length=0.005, maximum_frequency=5500.0,
            time_step=0.002, frequency_step=20.0,
        )
        vals = spec.values
        # All values should be very small
        assert np.max(vals) < 1e-10

    def test_spectrogram_values_non_negative(self, vowel_sound):
        spec = vowel_sound.to_spectrogram(
            window_length=0.005, maximum_frequency=5500.0,
            time_step=0.002, frequency_step=20.0,
        )
        assert np.all(spec.values >= 0)


# ===================================================================
# 6.8 Regression Tests
# ===================================================================

class TestRegressions:

    def test_pre_emphasis_minimum_1(self, vowel_sound):
        # Pre-emphasis of 0 should be clamped to 1
        result = extract_formants_from_praat(vowel_sound, pre_emphasis=0.0)
        assert isinstance(result, FormantData)

    def test_time_step_none_not_zero(self, vowel_sound):
        # time_step=0.0 should pass None to Praat, not 0.0
        result = extract_formants_from_praat(vowel_sound, time_step=0.0)
        assert result.n_frames > 0

    def test_n_formants_is_float(self, vowel_sound):
        # Parselmouth expects float for max_number_of_formants
        # This should not raise a TypeError
        result = extract_formants_from_praat(vowel_sound, n_formants=3)
        assert isinstance(result, FormantData)
