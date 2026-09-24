"""
Tests for the per-file recording bandwidth reported as ``hf_cutoff_hz``.

The measure exists to identify recordings whose top octave the recording chain
removed, since those yield an artificially low centre of gravity however the
speaker actually produced the sound. So the tests care about two things: that
a lowpassed file returns its real cutoff, and that the answer depends on the
recording's bandwidth alone — not on its gain, its silence, or its Nyquist.
"""

import os
import sys

import numpy as np
import parselmouth
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import formant_editor as fe
from formant_editor import (
    Interval,
    TextGrid,
    Tier,
    _build_csv_data,
    estimate_hf_cutoff,
    long_term_average_spectrum,
)


SR = 32000
NYQUIST = SR / 2.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _speech_like(duration_s=2.0, sr=SR, seed=0, floor=1e-4):
    """Broadband noise with a speech-like tilt, over a faint noise floor.

    The tilt matters: a flat spectrum would sit far above its own floor right
    up to Nyquist, which is not the shape any real recording has.
    """
    rng = np.random.default_rng(seed)
    n = int(sr * duration_s)
    spec = np.fft.rfft(rng.normal(0, 1, n))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    spec /= np.maximum(freqs, 50.0)             # ~-6 dB/octave
    x = np.fft.irfft(spec, n)
    x /= np.abs(x).max()
    return x + rng.normal(0, floor, n)


def _lowpass(x, cutoff_hz, sr=SR, seed=1, floor=1e-4):
    """Brick-wall lowpass, with the noise floor re-added AFTER the filter.

    A filter applied to the floor as well would leave nothing above the
    cutoff at all, which no real recording chain produces and which would let
    a broken estimator pass.
    """
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, 1.0 / sr)
    spec[freqs > cutoff_hz] = 0.0
    y = np.fft.irfft(spec, x.size)
    return y + np.random.default_rng(seed).normal(0, floor, y.size)


def _sound(x, sr=SR):
    return parselmouth.Sound(np.asarray(x, dtype=float), sampling_frequency=sr)


# ---------------------------------------------------------------------------
# The LTAS
# ---------------------------------------------------------------------------

def test_ltas_spans_dc_to_nyquist():
    freqs, db = long_term_average_spectrum(_sound(_speech_like()))
    assert freqs[0] == 0.0
    assert freqs[-1] == pytest.approx(NYQUIST)
    assert db.shape == freqs.shape


def test_ltas_declines_with_a_falling_spectrum():
    freqs, db = long_term_average_spectrum(_sound(_speech_like()))
    assert db[(freqs > 200) & (freqs < 400)].mean() > \
        db[(freqs > 4000) & (freqs < 6000)].mean()


def test_ltas_needs_a_full_frame():
    short = _sound(np.zeros(fe.HF_LTAS_WINDOW - 1))
    assert long_term_average_spectrum(short) == (None, None)


def test_ltas_ignores_silence_between_the_speech():
    """Padding a recording with room tone must not reshape its average."""
    x = _speech_like(duration_s=1.0)
    rng = np.random.default_rng(3)
    room = rng.normal(0, 1e-3, SR * 3)
    padded = np.concatenate([room, x, room])

    _, clean = long_term_average_spectrum(_sound(x))
    _, gated = long_term_average_spectrum(_sound(padded))
    # Whole-file averaging pulls the top end down; the gated one should not.
    _, ungated = long_term_average_spectrum(_sound(padded), active_db=1e9)
    top = slice(-200, None)
    assert abs(gated[top].mean() - clean[top].mean()) < \
        abs(ungated[top].mean() - clean[top].mean())


def test_smoothing_leaves_a_flat_spectrum_flat_at_its_edges():
    """Edge bins are normalised by their contributors, not the kernel width."""
    freqs = np.arange(0, 8000, 10.0)
    db = np.full(freqs.size, -42.0)
    assert fe._smooth_db(db, freqs) == pytest.approx(-42.0)


# ---------------------------------------------------------------------------
# The cutoff
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cutoff", [3400.0, 5000.0, 8000.0, 11000.0])
def test_a_lowpassed_file_returns_its_own_cutoff(cutoff):
    est = estimate_hf_cutoff(_sound(_lowpass(_speech_like(), cutoff)))
    assert est == pytest.approx(cutoff, rel=0.05)


def test_a_full_bandwidth_file_reaches_towards_nyquist():
    """A clean recording's edge is its ADC's anti-alias filter, just under
    Nyquist — the same shape as a lowpassed file, only much higher up."""
    est = estimate_hf_cutoff(_sound(_lowpass(_speech_like(), 0.95 * NYQUIST)))
    assert est > 0.9 * NYQUIST


def test_a_lowpassed_file_scores_below_a_full_bandwidth_one():
    """The discrimination the measure exists for."""
    full = _lowpass(_speech_like(), 0.95 * NYQUIST)
    assert estimate_hf_cutoff(_sound(_lowpass(full, 4000.0))) < \
        estimate_hf_cutoff(_sound(full)) - 1000.0


def test_the_cutoff_is_not_merely_nyquist():
    """The anti-requirement: any-energy would answer Nyquist every time."""
    est = estimate_hf_cutoff(_sound(_lowpass(_speech_like(), 4000.0)))
    assert est < 0.5 * NYQUIST


def test_the_cutoff_is_invariant_to_gain():
    """Self-calibrating against the file's own floor, not an absolute level."""
    x = _lowpass(_speech_like(), 6000.0)
    quiet = estimate_hf_cutoff(_sound(x * 0.01))
    loud = estimate_hf_cutoff(_sound(x))
    assert quiet == pytest.approx(loud, rel=0.02)


def test_the_cutoff_survives_leading_and_trailing_silence():
    x = _lowpass(_speech_like(duration_s=1.0), 5000.0)
    rng = np.random.default_rng(4)
    padded = np.concatenate([rng.normal(0, 1e-4, SR * 2), x,
                             rng.normal(0, 1e-4, SR * 2)])
    assert estimate_hf_cutoff(_sound(padded)) == \
        pytest.approx(estimate_hf_cutoff(_sound(x)), rel=0.05)


def test_a_sound_too_short_to_analyse_has_no_cutoff():
    assert np.isnan(estimate_hf_cutoff(_sound(np.zeros(64))))


def test_a_flat_noise_floor_has_no_cutoff():
    """Nothing clears its own floor by 6 dB, so there is nothing to report."""
    rng = np.random.default_rng(5)
    assert np.isnan(estimate_hf_cutoff(_sound(rng.normal(0, 1e-3, SR))))


def test_a_wider_margin_reports_a_lower_cutoff():
    x = _lowpass(_speech_like(), 0.95 * NYQUIST)
    strict = estimate_hf_cutoff(_sound(x), margin_db=30.0)
    lenient = estimate_hf_cutoff(_sound(x), margin_db=6.0)
    assert strict < lenient


def test_a_signal_with_no_rolloff_edge_reads_low():
    """The documented limit of the method, pinned so it stays known.

    An unfiltered power-law spectrum never flattens, so its minimum above
    1 kHz is Nyquist itself and the answer is wherever the last 6 dB of
    decline happens rather than where content stops. Digitised audio always
    carries an anti-alias edge, so this shape does not arise in a real file.
    """
    est = estimate_hf_cutoff(_sound(_speech_like(floor=0.0)))
    assert est < 0.7 * NYQUIST


# ---------------------------------------------------------------------------
# The CSV column
# ---------------------------------------------------------------------------

DUR = 1.5


@pytest.fixture
def corpus(tmp_path):
    """Two recordings of the same annotation: one full-band, one lowpassed."""
    audio_dir = tmp_path / "audio"
    tg_dir = tmp_path / "tg"
    audio_dir.mkdir()
    tg_dir.mkdir()

    full = _lowpass(_speech_like(duration_s=DUR), 0.95 * NYQUIST)
    sounds = {"wide": full, "narrow": _lowpass(full, 3400.0)}
    for name, x in sounds.items():
        _sound(x).save(str(audio_dir / f"{name}.wav"),
                       parselmouth.SoundFileFormat.WAV)
        tier = Tier("phones", "IntervalTier", 0.0, DUR,
                    intervals=[Interval(0.2, 0.7, "s"),
                               Interval(0.7, 1.2, "a")])
        TextGrid(0.0, DUR, [tier]).save(str(tg_dir / f"{name}.TextGrid"))
    return str(audio_dir), str(tg_dir), sounds


def _export(audio_dir, tg_dir, **kwargs):
    tiers = [Tier("phones", "IntervalTier", 0.0, DUR, intervals=[])]
    opts = dict(
        selected_tiers=tiers, extract_formants=False, formant_mode="at_points",
        point_tier_name=None, segment_tier_name="phones",
        percentage_markers=[], extract_durations=False,
        duration_tier_names=[], primary_tier_name="phones",
    )
    opts.update(kwargs)
    return _build_csv_data(audio_dir, tg_dir, None, **opts)


def test_spectral_export_carries_the_cutoff_beside_the_filename(corpus):
    audio_dir, tg_dir, _ = corpus
    headers, rows = _export(audio_dir, tg_dir, extract_spectral=True,
                            spectral_markers=[50.0],
                            spectral_tier_name="phones")
    assert headers[:2] == ["filename", "hf_cutoff_hz"]
    assert rows


def test_a_trajectory_only_export_carries_it_too(corpus):
    audio_dir, tg_dir, _ = corpus
    headers, _ = _export(audio_dir, tg_dir, extract_trajectory=True,
                         spectral_tier_name="phones")
    assert headers[:2] == ["filename", "hf_cutoff_hz"]


def test_an_export_without_spectral_data_has_no_cutoff_column(corpus):
    audio_dir, tg_dir, _ = corpus
    headers, _ = _export(audio_dir, tg_dir, extract_durations=True,
                         duration_tier_names=["phones"])
    assert "hf_cutoff_hz" not in headers
    assert headers[0] == "filename"


def test_the_cutoff_is_constant_within_a_file_and_differs_between(corpus):
    audio_dir, tg_dir, _sounds = corpus
    _, rows = _export(audio_dir, tg_dir, extract_spectral=True,
                      spectral_markers=[50.0], spectral_tier_name="phones")

    per_file = {}
    for name, cutoff, *_rest in rows:
        per_file.setdefault(name, set()).add(cutoff)

    assert set(per_file) == {"wide", "narrow"}
    for name, values in per_file.items():
        assert len(values) == 1, f"{name} reported {values}"

    narrow = float(per_file["narrow"].pop())
    wide = float(per_file["wide"].pop())
    assert narrow == pytest.approx(3400.0, rel=0.05)
    assert wide > narrow + 1000.0


def test_the_column_matches_the_estimator_on_the_unfiltered_sound(corpus):
    audio_dir, tg_dir, _sounds = corpus
    _, rows = _export(audio_dir, tg_dir, extract_spectral=True,
                      spectral_markers=[50.0], spectral_tier_name="phones",
                      spectral_highpass_hz=300.0)
    reported = {name: float(cutoff) for name, cutoff, *_rest in rows}
    for name, value in reported.items():
        direct = estimate_hf_cutoff(
            parselmouth.Sound(os.path.join(audio_dir, f"{name}.wav")))
        assert value == pytest.approx(direct, abs=1.0)
