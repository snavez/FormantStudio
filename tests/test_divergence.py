"""
Tests for reading a phoneme tier against an allophone tier: the four divergence
types, the span a coalesced sound is measured over, and the structural rules
that keep the two tiers readable together.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import divergence
from divergence import (
    MATCH, DELETION, INSERTION, NOT_ANALYSED, SUBSTITUTION, UNANNOTATED,
    AnnotationError, resolve, summarise, validate,
)


def tiers(pairs, step=0.1):
    """Build a matching phoneme/allophone tier pair from (phoneme, allophone)."""
    ph = [(i * step, (i + 1) * step, p) for i, (p, _) in enumerate(pairs)]
    al = [(i * step, (i + 1) * step, a) for i, (_, a) in enumerate(pairs)]
    return ph, al


def kinds(pairs):
    return [d.kind for d in resolve(*tiers(pairs))]


# ---------------------------------------------------------------------------
# The four divergence types
# ---------------------------------------------------------------------------

def test_matching_labels_are_a_match():
    assert kinds([("k", "k"), ("o", "o")]) == [MATCH, MATCH]


def test_differing_labels_are_a_substitution():
    assert kinds([("N", "n")]) == [SUBSTITUTION]


def test_empty_phoneme_against_a_label_is_an_insertion():
    assert kinds([("N", "n"), ("-", "g")]) == [SUBSTITUTION, INSERTION]


def test_an_arrow_is_a_deletion():
    assert kinds([("k", "k"), ("i", ">"), ("e", "@")]) == [
        MATCH, DELETION, SUBSTITUTION]


def test_empty_on_both_tiers_is_not_analysed():
    """A gap, hesitation or pause: nothing expected and nothing produced."""
    assert kinds([("-", "-")]) == [NOT_ANALYSED]


def test_an_expected_sound_with_nothing_produced_is_a_deletion():
    """"-" against a real phoneme means it was not produced, so no data."""
    out = resolve(*tiers([("k", "-"), ("o", "o")]))
    assert out[0].kind == DELETION
    assert out[0].xmin is None and out[0].xmax is None


def test_only_four_kinds_can_reach_a_csv_row():
    """Gaps and unannotated intervals are skipped before a row is built."""
    measurable = {MATCH, SUBSTITUTION, INSERTION, DELETION}
    pairs = [("k", "k"), ("N", "n"), ("-", "g"), ("i", ">"), ("e", "@")]
    assert {d.kind for d in resolve(*tiers(pairs))} <= measurable


def test_blank_on_both_tiers_is_unannotated():
    """The case the '-' marker exists to distinguish."""
    assert kinds([("", "")]) == [UNANNOTATED]


def test_every_expected_sound_gets_a_record():
    pairs = [("k", "k"), ("i", ">"), ("e", "@"), ("a", "<")]
    assert len(resolve(*tiers(pairs))) == len(pairs)


# ---------------------------------------------------------------------------
# The span a coalesced sound is measured over
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("allos,survivor", [
    ([">", ">", "@"], 2),
    ([">", "@", "<"], 1),
    (["@", "<", "<"], 0),
])
def test_coalescence_measures_the_whole_span(allos, survivor):
    """However the arrows fall, the surviving sound spans all three intervals."""
    pairs = list(zip(["i", "e", "a"], allos))
    out = resolve(*tiers(pairs))
    assert out[survivor].kind == SUBSTITUTION
    assert out[survivor].xmin == pytest.approx(0.0)
    assert out[survivor].xmax == pytest.approx(0.3)
    for i, d in enumerate(out):
        if i != survivor:
            assert d.kind == DELETION


def test_a_deleted_sound_has_no_span_to_measure():
    out = resolve(*tiers([("i", ">"), ("e", "@")]))
    assert out[0].xmin is None and out[0].xmax is None


def test_an_ordinary_sound_spans_only_its_own_interval():
    out = resolve(*tiers([("k", "k"), ("o", "o")]))
    assert out[0].xmin == pytest.approx(0.0) and out[0].xmax == pytest.approx(0.1)


def test_arrows_only_attach_to_the_sound_they_point_at():
    """A '<' after one sound must not extend a different one."""
    out = resolve(*tiers([("a", "x"), ("i", "<"), ("e", "y")]))
    assert out[0].xmax == pytest.approx(0.2)      # 'x' absorbed the '<'
    assert out[2].xmin == pytest.approx(0.2)      # 'y' is untouched
    assert out[2].xmax == pytest.approx(0.3)


def test_insertion_can_also_absorb_neighbours():
    out = resolve(*tiers([("-", "g"), ("i", "<")]))
    assert out[0].kind == INSERTION
    assert out[0].xmax == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Structural validation
# ---------------------------------------------------------------------------

def test_well_formed_annotation_has_no_problems():
    for pairs in ([("k", "k")], [("i", ">"), ("e", "@")],
                  [("i", "@"), ("e", "<")], [("-", "g")], [("-", "-")]):
        assert validate(*tiers(pairs)) == []


def test_arrow_may_not_open_the_tier():
    problems = validate(*tiers([("i", "<"), ("e", "@")]))
    assert len(problems) == 1 and "cannot open" in problems[0]


def test_arrow_may_not_close_the_tier():
    problems = validate(*tiers([("e", "@"), ("i", ">")]))
    assert len(problems) == 1 and "cannot close" in problems[0]


def test_arrows_pointing_at_each_other_are_reported_once():
    problems = validate(*tiers([("a", "x"), ("i", ">"), ("e", "<"), ("o", "y")]))
    assert len(problems) == 1 and "point at each other" in problems[0]


def test_an_arrow_may_not_reach_across_an_empty_interval():
    problems = validate(*tiers([("a", "@"), ("x", "-"), ("e", "<")]))
    assert len(problems) == 1 and "cannot reach" in problems[0]


def test_mismatched_boundaries_are_reported():
    ph, al = tiers([("k", "k"), ("N", "n")])
    al[1] = (al[1][0] + 0.02, al[1][1], al[1][2])
    problems = validate(ph, al)
    assert any("boundaries differ" in p for p in problems)


def test_mismatched_interval_counts_are_reported():
    ph, al = tiers([("k", "k"), ("N", "n")])
    problems = validate(ph, al[:1])
    assert len(problems) == 1 and "different numbers of intervals" in problems[0]


def test_resolve_refuses_tiers_it_cannot_pair():
    ph, al = tiers([("k", "k"), ("N", "n")])
    with pytest.raises(AnnotationError):
        resolve(ph, al[:1])


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def test_summarise_counts_each_kind():
    pairs = [("k", "k"), ("N", "n"), ("-", "g"), ("i", ">"), ("e", "@")]
    counts = summarise(resolve(*tiers(pairs)))
    assert counts == {MATCH: 1, SUBSTITUTION: 2, INSERTION: 1, DELETION: 1}


def test_label_helpers():
    assert divergence.is_empty("-") and divergence.is_empty("  -  ")
    assert divergence.is_arrow("<") and divergence.is_arrow(">")
    assert divergence.is_anchor("@") and not divergence.is_anchor("-")
    assert not divergence.is_anchor("") and not divergence.is_anchor(">")


# ---------------------------------------------------------------------------
# What the CSV builder relies on
# ---------------------------------------------------------------------------

def test_row_units_come_only_from_reportable_kinds():
    """Gaps and unannotated intervals must not become rows."""
    pairs = [("k", "k"), ("-", "-"), ("", ""), ("N", "n")]
    reportable = [d for d in resolve(*tiers(pairs))
                  if d.kind not in (NOT_ANALYSED, UNANNOTATED)]
    assert [d.kind for d in reportable] == [MATCH, SUBSTITUTION]


def test_a_deletion_keeps_its_labels_for_the_row():
    """A deletion has no span, but its labels still identify the row."""
    out = resolve(*tiers([("i", ">"), ("e", "@")]))
    assert out[0].phoneme == "i" and out[0].realised == ">"


def test_the_realised_label_is_the_anchors_own():
    """The widened span covers absorbed intervals, so the label must not be
    looked up across it."""
    out = resolve(*tiers([("i", ">"), ("e", "@"), ("a", "<")]))
    assert out[1].realised == "@"
    assert out[1].xmin == pytest.approx(0.0) and out[1].xmax == pytest.approx(0.3)
