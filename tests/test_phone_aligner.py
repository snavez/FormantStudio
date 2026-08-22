"""
Tests for phone-class forced alignment: the SAMPA class map, the time warp,
Viterbi decoding, and the guarantee that every tier moves together.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import phone_aligner
from phone_aligner import (
    AFFR, APPROX, CLASS_INDEX, FRIC, NASAL, SIL, STOP, VOWEL,
    AlignmentError, ClassModel, TimeWarp, alignable_tiers, classify, is_pause,
    warp_textgrid, _viterbi,
)
from formant_editor import Interval, Point, Tier, TextGrid


# ---------------------------------------------------------------------------
# SAMPA classification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,expected", [
    ("<p:>", SIL), ("", SIL), ("sp", SIL),
    ("6", VOWEL), ("e", VOWEL), ("6:", VOWEL), ("6i", VOWEL), ("@}", VOWEL),
    ("m", NASAL), ("N", NASAL), ("n", NASAL), ("m=", NASAL),
    ("w", APPROX), ("4", APPROX), ("l", APPROX), ("r", APPROX), ("j", APPROX),
    ("f", FRIC), ("h", FRIC), ("s", FRIC), ("v", FRIC),
    ("t", STOP), ("k", STOP), ("p", STOP), ("?", STOP),
    ("t_h", STOP), ("k_h", STOP), ("p_h", STOP),
    ("ts", AFFR), ("kC", AFFR),
])
def test_classify_sampa(label, expected):
    assert classify(label) == expected


def test_diacritics_do_not_change_class():
    for base in ("t", "6", "m", "s"):
        assert classify(base) == classify(base + "_h") == classify(base + ":")


def test_unknown_symbol_falls_back_to_vowel():
    assert classify("øøø") == VOWEL


def test_is_pause():
    assert is_pause("<p:>") and is_pause("  ") and not is_pause("6")


# ---------------------------------------------------------------------------
# TimeWarp
# ---------------------------------------------------------------------------

def test_linear_warp_scales_proportionally():
    w = TimeWarp.linear(10.0, 5.0)
    assert w(0.0) == 0.0
    assert w(10.0) == pytest.approx(5.0)
    assert w(4.0) == pytest.approx(2.0)


def test_warp_interpolates_between_knots():
    w = TimeWarp([(0, 0), (1, 5), (2, 6)])
    assert w(0.5) == pytest.approx(2.5)
    assert w(1.5) == pytest.approx(5.5)


def test_warp_is_monotonic_even_from_unsorted_knots():
    w = TimeWarp([(0, 0), (1, 5), (2, 3), (3, 9)])
    ys = [w(t) for t in np.linspace(0, 3, 40)]
    assert all(b >= a - 1e-9 for a, b in zip(ys, ys[1:]))


def test_warp_maps_equal_inputs_to_equal_outputs():
    """The property that keeps time-aligned tiers time-aligned."""
    w = TimeWarp([(0, 0), (1.5, 2.0), (3.0, 7.0)])
    assert w(1.5) == w(1.5)
    for t in (0.0, 0.7, 1.5, 2.9, 3.0):
        assert w(t) == w(t)


# ---------------------------------------------------------------------------
# Viterbi
# ---------------------------------------------------------------------------

def _separable_model(n_feat=4):
    n_cls = len(phone_aligner.CLASSES)
    means = np.zeros((n_cls, n_feat))
    for c in range(n_cls):
        means[c] = np.roll([3.0, 0.0, 0.0, 0.0], c % n_feat) * (1 + c * 0.3)
    return ClassModel(
        mu=np.zeros(n_feat), sd=np.ones(n_feat), means=means,
        variances=np.full((n_cls, n_feat), 0.1),
        log_prior=np.zeros(n_cls), log_dur=np.log(np.full(n_cls, 10.0)),
        min_frames=np.full(n_cls, 2.0))


def _synthetic(labels, lengths, model, noise=0.15, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for lab, n in zip(labels, lengths):
        c = CLASS_INDEX[classify(lab)]
        rows.append(model.means[c][None, :] + rng.normal(0, noise, (n, 4)))
    return np.vstack(rows)


def test_viterbi_recovers_known_boundaries():
    model = _separable_model()
    labels = ["<p:>", "k", "o", "m", "6", "<p:>", "t", "e"]
    lengths = [20, 8, 15, 9, 18, 25, 7, 12]
    frames = _synthetic(labels, lengths, model)
    classes = np.array([CLASS_INDEX[classify(l)] for l in labels])

    ends = _viterbi(classes, frames, model)
    assert list(ends) == list(np.cumsum(lengths) - 1)


def test_viterbi_output_is_monotonic():
    model = _separable_model()
    labels = ["<p:>", "t", "6", "n", "6", "<p:>"]
    frames = _synthetic(labels, [15, 6, 20, 8, 16, 20], model, noise=0.8, seed=3)
    classes = np.array([CLASS_INDEX[classify(l)] for l in labels])
    ends = _viterbi(classes, frames, model)
    assert all(b >= a for a, b in zip(ends, ends[1:]))
    assert ends[-1] == frames.shape[0] - 1


def test_viterbi_rejects_audio_too_short_for_the_labels():
    model = _separable_model()
    labels = ["k", "o", "m", "6", "t", "e", "n", "6"]
    frames = _synthetic(["6"], [3], model)
    classes = np.array([CLASS_INDEX[classify(l)] for l in labels])
    with pytest.raises(AlignmentError):
        _viterbi(classes, frames, model)


# ---------------------------------------------------------------------------
# Applying a warp to a whole TextGrid
# ---------------------------------------------------------------------------

def _two_tier_grid():
    """Two tiers sharing boundaries at 1.0 and 2.0."""
    words = Tier("words", "IntervalTier", 0, 3.0, intervals=[
        Interval(0.0, 1.0, "ko"), Interval(1.0, 2.0, "te"),
        Interval(2.0, 3.0, "reo")])
    phones = Tier("phones", "IntervalTier", 0, 3.0, intervals=[
        Interval(0.0, 0.5, "k"), Interval(0.5, 1.0, "o"),
        Interval(1.0, 1.5, "t"), Interval(1.5, 2.0, "e"),
        Interval(2.0, 3.0, "r")])
    marks = Tier("marks", "TextTier", 0, 3.0,
                 points=[Point(1.0, "a"), Point(2.5, "b")])
    return TextGrid(0, 3.0, [words, phones, marks])


def test_warp_textgrid_preserves_shared_boundaries():
    tg = _two_tier_grid()
    warp = TimeWarp([(0, 0), (1.0, 0.4), (2.0, 3.5), (3.0, 4.0)])
    out = warp_textgrid(tg, warp, 4.0)

    words, phones, marks = out.tiers
    assert words.intervals[0].xmax == phones.intervals[1].xmax
    assert words.intervals[1].xmax == phones.intervals[3].xmax
    assert marks.points[0].time == words.intervals[0].xmax


def test_warp_textgrid_spans_the_new_duration():
    out = warp_textgrid(_two_tier_grid(), TimeWarp.linear(3.0, 6.0), 6.0)
    assert out.xmax == 6.0
    for tier in out.tiers:
        assert tier.xmax == 6.0
        if tier.tier_class == "IntervalTier":
            assert tier.intervals[0].xmin == 0
            assert tier.intervals[-1].xmax == 6.0


def test_warp_textgrid_keeps_labels_and_tier_order():
    out = warp_textgrid(_two_tier_grid(), TimeWarp.linear(3.0, 6.0), 6.0)
    assert [t.name for t in out.tiers] == ["words", "phones", "marks"]
    assert [iv.text for iv in out.tiers[0].intervals] == ["ko", "te", "reo"]
    assert [p.mark for p in out.tiers[2].points] == ["a", "b"]


def test_warp_textgrid_output_is_monotonic():
    out = warp_textgrid(_two_tier_grid(),
                        TimeWarp([(0, 0), (1.0, 2.9), (2.0, 3.0), (3.0, 4.0)]), 4.0)
    for tier in out.tiers:
        if tier.tier_class == "IntervalTier":
            xs = [iv.xmin for iv in tier.intervals]
            assert all(b >= a for a, b in zip(xs, xs[1:]))


# ---------------------------------------------------------------------------
# Choosing a tier to drive the alignment
# ---------------------------------------------------------------------------

def test_alignable_tiers_prefers_the_finest_tier():
    tg = _two_tier_grid()
    assert alignable_tiers(tg)[0] == "phones"


def test_alignable_tiers_skips_point_tiers():
    assert "marks" not in alignable_tiers(_two_tier_grid())


def test_alignable_tiers_empty_when_nothing_usable():
    tg = TextGrid(0, 1.0, [Tier("pts", "TextTier", 0, 1.0, points=[Point(0.5, "x")])])
    assert alignable_tiers(tg) == []


# ---------------------------------------------------------------------------
# The shipped model
# ---------------------------------------------------------------------------

def test_shipped_model_loads_and_matches_the_feature_set():
    model = ClassModel.load()
    assert model.means.shape == (len(phone_aligner.MODEL_CLASSES),
                                 len(phone_aligner.FEATURE_NAMES))
    assert model.min_frames.shape == (len(phone_aligner.MODEL_CLASSES),)
    assert np.all(model.min_frames >= 1)


def test_shipped_model_scores_frames():
    model = ClassModel.load()
    frames = np.zeros((5, len(phone_aligner.FEATURE_NAMES)), dtype=np.float32)
    ll = model.log_likelihood(frames)
    assert ll.shape == (5, len(phone_aligner.MODEL_CLASSES))
    assert np.all(np.isfinite(ll))


# ---------------------------------------------------------------------------
# Independence from any one labelling convention
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,expected", [
    ("a", VOWEL), ("\u0254", VOWEL), ("\u0259", VOWEL), ("\u025b", VOWEL),
    ("\u014b", NASAL), ("\u0272", NASAL),
    ("\u0283", FRIC), ("\u03b8", FRIC), ("\u0278", FRIC),
    ("\u0279", APPROX), ("\u027e", APPROX), ("\u026b", APPROX),
    ("\u0294", STOP), ("\u0261", STOP),
])
def test_classify_ipa(label, expected):
    """Templates may be transcribed in IPA rather than SAMPA."""
    assert classify(label) == expected


@pytest.mark.parametrize("label", [
    "<p:>", "SIL", "sil", "Sil", "pause", "PAUSE", "<sil>", "sp", "#", "", "   ",
])
def test_pause_conventions_are_matched_case_insensitively(label):
    assert is_pause(label)


def test_spoken_labels_are_not_pauses():
    for label in ("6", "k", "\u014b", "tS"):
        assert not is_pause(label)


def test_unrecognised_labels_are_reported():
    from phone_aligner import unrecognised_labels
    assert unrecognised_labels(["6", "k", "\u2603", "<p:>"]) == ["\u2603"]
    assert unrecognised_labels(["6", "k", "m"]) == []


def test_unrecognised_labels_still_classify_as_vowel():
    """An unfamiliar alphabet degrades the alignment; it must not crash it."""
    assert classify("\u2603") == VOWEL


def test_alignable_tiers_ignores_tier_names_entirely():
    """Tier naming is local convention; ranking must be structural only."""
    fine = Tier("zzz_obscure_name", "IntervalTier", 0, 2.0, intervals=[
        Interval(0.0, 0.4, "k"), Interval(0.4, 0.9, "o"),
        Interval(0.9, 1.4, "t"), Interval(1.4, 2.0, "e")])
    coarse = Tier("MAU", "IntervalTier", 0, 2.0, intervals=[
        Interval(0.0, 1.0, "kotahi"), Interval(1.0, 2.0, "tekau")])
    ranked = alignable_tiers(TextGrid(0, 2.0, [coarse, fine]))
    assert ranked[0] == "zzz_obscure_name"
    assert set(ranked) == {"zzz_obscure_name", "MAU"}


def test_alignable_tiers_offers_every_interval_tier():
    """The user must never be blocked from choosing a tier we ranked low."""
    a = Tier("a", "IntervalTier", 0, 2.0,
             intervals=[Interval(0, 1, ""), Interval(1, 2, "")])
    b = Tier("b", "IntervalTier", 0, 2.0,
             intervals=[Interval(0, 1, "x"), Interval(1, 2, "y")])
    ranked = alignable_tiers(TextGrid(0, 2.0, [a, b]))
    assert set(ranked) == {"a", "b"}
    assert ranked[0] == "b"


def test_warp_textgrid_keeps_arbitrary_tier_names_and_order():
    tiers = [
        Tier("Speaker A : words", "IntervalTier", 0, 2.0,
             intervals=[Interval(0, 1, "ko"), Interval(1, 2, "te")]),
        Tier("segmental", "IntervalTier", 0, 2.0,
             intervals=[Interval(0, 1, "k"), Interval(1, 2, "t")]),
        Tier("notes", "TextTier", 0, 2.0, points=[Point(1.0, "n")]),
    ]
    out = warp_textgrid(TextGrid(0, 2.0, tiers), TimeWarp.linear(2.0, 5.0), 5.0)
    assert [t.name for t in out.tiers] == [
        "Speaker A : words", "segmental", "notes"]
    assert [t.tier_class for t in out.tiers] == [
        "IntervalTier", "IntervalTier", "TextTier"]


# ---------------------------------------------------------------------------
# Vowel-quality sub-classes
# ---------------------------------------------------------------------------

from phone_aligner import (
    MODEL_CLASSES, MODEL_CLASS_INDEX, UNKNOWN_QUALITY, VOWEL_FRONTINGS,
    VOWEL_HEIGHTS, model_class, vowel_quality,
)


def test_manner_classification_is_unchanged_by_sub_classing():
    """classify() stays the broad phonetic answer; only the model splits."""
    assert classify("i") == classify("u") == classify("6") == VOWEL
    assert classify("k") == STOP and classify("m") == NASAL


@pytest.mark.parametrize("label,height,fronting", [
    ("i", "high", "front"),
    ("u", "high", "back"),
    ("e", "mid-high", "front"),
    ("o", "mid-high", "back"),
    ("6", "mid-low", "centre"),
    ("@", "mid", "centre"),
    ("}", "high", "centre"),
])
def test_vowel_quality_from_the_chart(label, height, fronting):
    assert vowel_quality(label) == (height, fronting)


def test_vowel_quality_reads_ipa_as_well_as_sampa():
    assert vowel_quality("\u0254") == ("mid-low", "back")
    assert vowel_quality("\u025b") is not None


def test_length_and_devoicing_do_not_change_quality():
    assert vowel_quality("6:") == vowel_quality("6")
    assert vowel_quality("6_0") == vowel_quality("6")


def test_diphthong_takes_its_onset_quality():
    """A diphthong is one segment, so it is classed by where it begins."""
    assert vowel_quality("6i") == vowel_quality("6")
    assert vowel_quality("6u") == vowel_quality("6")
    assert model_class("6i") == model_class("6")


def test_distinct_vowels_get_distinct_model_classes():
    classes = {model_class(v) for v in ("i", "e", "6", "o", "u", "@", "}")}
    assert len(classes) == 7


def test_unknown_vowel_falls_back_to_its_own_class():
    unknown = model_class("\u2603")
    assert UNKNOWN_QUALITY in unknown
    assert unknown in MODEL_CLASS_INDEX


def test_non_vowels_are_not_sub_classed():
    for label in ("k", "m", "f", "ts", "w", "<p:>", "-"):
        assert model_class(label) == classify(label)


def test_model_class_list_is_fixed_and_complete():
    """The class list must not depend on any corpus, or a stored model would
    stop lining up with the code that loads it."""
    expected = ((len(phone_aligner.CLASSES) - 1)
                + len(VOWEL_HEIGHTS) * len(VOWEL_FRONTINGS) + 1)
    assert len(MODEL_CLASSES) == expected
    assert len(set(MODEL_CLASSES)) == len(MODEL_CLASSES)
    for label in ("i", "u", "6", "\u2603", "k", "<p:>"):
        assert model_class(label) in MODEL_CLASS_INDEX


def test_bark_difference_features_are_present():
    for name in ("bark_f1_f0", "bark_f2_f1", "bark_f3_f2"):
        assert name in phone_aligner.FEATURE_NAMES


def test_bark_scale_is_monotonic_and_compresses_high_frequencies():
    from phone_aligner import _bark
    b = _bark(np.array([100.0, 500.0, 1000.0, 2000.0, 4000.0]))
    assert np.all(np.diff(b) > 0)

    # The same distance in Hz spans far fewer Bark higher up. That compression
    # is what lets a formant difference mean the same thing for a small vocal
    # tract as for a large one.
    low = _bark(np.array([100.0, 1100.0]))
    high = _bark(np.array([3000.0, 4000.0]))
    assert (low[1] - low[0]) > 4 * (high[1] - high[0])


def test_chart_and_model_resolve_inside_a_bundle(tmp_path, monkeypatch):
    """Frozen builds resolve data files against sys._MEIPASS.

    A chart the aligner cannot find is not an error — every vowel quietly
    becomes the unknown quality — so the failure would only show as worse
    alignment. Worth asserting rather than trusting.
    """
    import shutil
    root = os.path.join(os.path.dirname(__file__), "..")
    (tmp_path / "Docs").mkdir()
    shutil.copy(os.path.join(root, "Docs", "ipa_symbol_chart.csv"),
                str(tmp_path / "Docs"))
    shutil.copy(os.path.join(root, "phone_class_model.npz"), str(tmp_path))

    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    monkeypatch.setattr(phone_aligner, "_vowel_quality_cache", None)

    assert phone_aligner.default_model_path().startswith(str(tmp_path))
    assert ClassModel.load().means.shape == (len(MODEL_CLASSES),
                                             len(phone_aligner.FEATURE_NAMES))
    assert UNKNOWN_QUALITY not in model_class("i")
    assert model_class("i") != model_class("u")


def test_missing_chart_degrades_without_raising(tmp_path, monkeypatch):
    """No chart must mean uninformative vowels, not a crash mid-alignment."""
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    monkeypatch.setattr(phone_aligner, "_vowel_quality_cache", None)

    assert vowel_quality("i") is None
    assert model_class("i") in MODEL_CLASS_INDEX
    assert UNKNOWN_QUALITY in model_class("i")
    assert model_class("k") == STOP
