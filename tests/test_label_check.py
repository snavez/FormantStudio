"""
Tests for the batch label-file checks.

Each check earns its place by catching something that would otherwise corrupt
an export silently, so the tests are paired: a fault that must be reported, and
the legitimate annotation next door that must not be. The pause marker, the
divergence arrows and the silence either side of the speech are all things a
careless check would flag by the hundred.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import label_check as lc
from formant_editor import Interval, Point, TextGrid, Tier


DUR = 2.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _itier(name, spans):
    """An interval tier from (xmin, xmax, text) triples."""
    return Tier(name, "IntervalTier", 0.0, DUR,
                intervals=[Interval(a, b, t) for a, b, t in spans])


def _ptier(name, marks):
    """A point tier from (time, mark) pairs."""
    return Tier(name, "TextTier", 0.0, DUR,
                points=[Point(t, m) for t, m in marks])


def _grid(*tiers):
    return TextGrid(0.0, DUR, list(tiers))


# A tier with silence at each end, a real hole in the middle, and a pause the
# annotator marked deliberately.
def _holey():
    return _itier("phones", [
        (0.0, 0.2, ""),        # 0  leading silence — not an omission
        (0.2, 0.4, "t"),       # 1
        (0.4, 0.6, ""),        # 2  a hole inside the annotation
        (0.6, 0.8, "-"),       # 3  deliberate pause
        (0.8, 1.0, "a"),       # 4
        (1.0, DUR, ""),        # 5  trailing silence — not an omission
    ])


# ---------------------------------------------------------------------------
# Unlabelled
# ---------------------------------------------------------------------------

def test_reports_only_the_hole_inside_the_annotation():
    issues = lc.check_unlabelled(_grid(_holey()), "f", ["phones"])
    assert [i.index for i in issues] == [2]
    assert issues[0].kind == lc.UNLABELLED


def test_the_deliberate_pause_is_not_an_omission():
    issues = lc.check_unlabelled(_grid(_holey()), "f", ["phones"])
    assert all(i.index != 3 for i in issues)


def test_leading_and_trailing_silence_are_not_omissions():
    issues = lc.check_unlabelled(_grid(_holey()), "f", ["phones"])
    assert all(i.index not in (0, 5) for i in issues)


def test_a_fully_labelled_tier_reports_nothing():
    tier = _itier("phones", [(0.0, 1.0, "t"), (1.0, DUR, "a")])
    assert lc.check_unlabelled(_grid(tier), "f", ["phones"]) == []


def test_an_entirely_blank_tier_reports_nothing():
    """No annotated extent means nothing was omitted — it was never started."""
    tier = _itier("phones", [(0.0, 1.0, ""), (1.0, DUR, "")])
    assert lc.check_unlabelled(_grid(tier), "f", ["phones"]) == []


def test_the_proposed_fix_for_a_hole_is_the_pause_marker():
    issues = lc.check_unlabelled(_grid(_holey()), "f", ["phones"])
    assert issues[0].proposed == lc.PAUSE_MARKER
    assert issues[0].fixable


def test_a_blank_point_is_reported_with_no_fix_to_guess():
    tier = _ptier("targets", [(0.5, "T1"), (1.0, "")])
    issues = lc.check_unlabelled(_grid(tier), "f", ["targets"])
    assert [i.index for i in issues] == [1]
    assert not issues[0].fixable


def test_a_tier_missing_from_this_file_is_skipped_quietly():
    assert lc.check_unlabelled(_grid(_holey()), "f", ["nosuch"]) == []


# ---------------------------------------------------------------------------
# Symbols
# ---------------------------------------------------------------------------

MAORI = {"p", "t", "k", "m", "n", "ŋ", "f", "h", "r", "w",
         "a", "e", "i", "o", "u"}


def _symbols(tg, tiers=("phones",), inventory=MAORI):
    return lc.check_symbols(tg, "f", list(tiers), inventory.__contains__,
                            inventory, lc.NOT_IN_INVENTORY)


def test_a_symbol_outside_the_inventory_is_flagged():
    tier = _itier("phones", [(0.0, 1.0, "θ"), (1.0, DUR, "a")])
    issues = _symbols(_grid(tier))
    assert [i.current for i in issues] == ["θ"]


def test_symbols_inside_the_inventory_pass():
    tier = _itier("phones", [(0.0, 1.0, "ŋ"), (1.0, DUR, "u")])
    assert _symbols(_grid(tier)) == []


def test_a_mistyped_symbol_gets_the_nearest_inventory_member_proposed():
    tier = _itier("phones", [(0.0, DUR, "tt")])
    issues = _symbols(_grid(tier))
    assert issues[0].proposed == "t"


def test_a_label_close_to_nothing_gets_no_proposal():
    tier = _itier("phones", [(0.0, DUR, "XYZZY")])
    issues = _symbols(_grid(tier))
    assert issues[0].current == "XYZZY"
    assert not issues[0].fixable


def test_pause_markers_and_arrows_are_syntax_not_symbols():
    tier = _itier("phones", [(0.0, 0.5, "-"), (0.5, 1.0, ">"),
                             (1.0, 1.5, "<"), (1.5, DUR, "")])
    assert _symbols(_grid(tier)) == []


def test_the_chart_mode_reports_its_own_kind():
    chart = {"t", "a", "s"}
    tier = _itier("phones", [(0.0, DUR, "θ")])
    issues = lc.check_symbols(_grid(tier), "f", ["phones"],
                              chart.__contains__, chart, lc.UNKNOWN_SYMBOL)
    assert issues[0].kind == lc.UNKNOWN_SYMBOL
    assert "recognised" in issues[0].detail


def test_point_marks_are_validated_too():
    tier = _ptier("targets", [(0.5, "a"), (1.0, "θ")])
    issues = _symbols(_grid(tier), tiers=("targets",))
    assert [i.index for i in issues] == [1]


# ---------------------------------------------------------------------------
# Boundary drift
# ---------------------------------------------------------------------------

def _pair(allo_second_start):
    """A phone tier and an allophone tier whose middle boundary is movable."""
    phones = _itier("phones", [(0.0, 0.5, "t"), (0.5, DUR, "a")])
    allo = _itier("allo", [(0.0, allo_second_start, "t0"),
                           (allo_second_start, DUR, "a")])
    return _grid(phones, allo)


def test_a_near_miss_is_reported_and_snapped_to_the_reference():
    issues = lc.check_boundaries(_pair(0.4968), "f", "phones", ["allo"], 0.020)
    assert len(issues) == 1
    assert issues[0].kind == lc.BOUNDARY_DRIFT
    assert float(issues[0].proposed) == pytest.approx(0.5)


def test_an_exactly_aligned_boundary_is_not_drift():
    assert lc.check_boundaries(_pair(0.5), "f", "phones", ["allo"], 0.020) == []


def test_a_genuine_subdivision_is_left_alone():
    """Further than the tolerance is a real extra boundary, not a slip."""
    assert lc.check_boundaries(_pair(0.30), "f", "phones", ["allo"],
                               0.020) == []


def test_the_tolerance_is_inclusive_at_its_edge():
    at = lc.check_boundaries(_pair(0.480), "f", "phones", ["allo"], 0.020)
    beyond = lc.check_boundaries(_pair(0.479), "f", "phones", ["allo"], 0.020)
    assert len(at) == 1 and beyond == []


def test_the_reference_tier_is_never_checked_against_itself():
    issues = lc.check_boundaries(_pair(0.4968), "f", "phones",
                                 ["phones", "allo"], 0.020)
    assert all(i.tier == "allo" for i in issues)


def test_the_reference_tiers_final_edge_counts_as_a_boundary():
    """It is a candidate to measure against, but never one to snap to.

    Every tier spans the same duration, so moving a boundary onto the far edge
    would collapse the last interval to nothing. The drift is worth reporting;
    the repair is not ours to make.
    """
    phones = _itier("phones", [(0.0, DUR, "a")])
    allo = _itier("allo", [(0.0, 1.99, "a"), (1.99, DUR, "x")])
    issues = lc.check_boundaries(_grid(phones, allo), "f", "phones",
                                 ["allo"], 0.020)
    assert len(issues) == 1, "the far edge was not considered"
    assert not issues[0].fixable
    assert "cross" in issues[0].detail


def test_a_snap_that_would_cross_a_neighbour_is_reported_without_a_fix():
    phones = _itier("phones", [(0.0, 0.5, "t"), (0.5, DUR, "a")])
    allo = _itier("allo", [(0.0, 0.49, "t0"), (0.49, 0.495, "tH"),
                           (0.495, DUR, "a")])
    issues = lc.check_boundaries(_grid(phones, allo), "f", "phones",
                                 ["allo"], 0.020)
    crossing = [i for i in issues if not i.fixable]
    assert crossing, "a boundary whose snap would invert an interval"
    assert "cross" in crossing[0].detail


def test_a_missing_reference_tier_reports_nothing():
    assert lc.check_boundaries(_pair(0.4968), "f", "nosuch", ["allo"],
                               0.020) == []


def test_points_drift_against_interval_boundaries():
    phones = _itier("phones", [(0.0, 0.5, "t"), (0.5, DUR, "a")])
    marks = _ptier("targets", [(0.4968, "T")])
    issues = lc.check_boundaries(_grid(phones, marks), "f", "phones",
                                 ["targets"], 0.020)
    assert float(issues[0].proposed) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Applying fixes
# ---------------------------------------------------------------------------

def test_a_label_fix_is_written_to_the_interval():
    tg = _grid(_holey())
    issues = lc.check_unlabelled(tg, "f", ["phones"])
    applied, skipped = lc.apply_fixes(tg, issues)
    assert (applied, skipped) == (1, 0)
    assert tg.tiers[0].intervals[2].text == lc.PAUSE_MARKER


def test_a_boundary_fix_moves_both_sides_of_the_seam():
    tg = _pair(0.4968)
    issues = lc.check_boundaries(tg, "f", "phones", ["allo"], 0.020)
    lc.apply_fixes(tg, issues)
    allo = tg.tiers[1]
    assert allo.intervals[0].xmax == pytest.approx(0.5)
    assert allo.intervals[1].xmin == pytest.approx(0.5)


def test_the_tier_stays_contiguous_after_a_boundary_fix():
    tg = _pair(0.4968)
    lc.apply_fixes(tg, lc.check_boundaries(tg, "f", "phones", ["allo"], 0.020))
    ivs = tg.tiers[1].intervals
    assert all(a.xmax == b.xmin for a, b in zip(ivs, ivs[1:]))


def test_unfixable_issues_are_skipped_not_applied():
    tier = _ptier("targets", [(1.0, "")])
    tg = _grid(tier)
    applied, skipped = lc.apply_fixes(
        tg, lc.check_unlabelled(tg, "f", ["targets"]))
    assert (applied, skipped) == (0, 1)
    assert tg.tiers[0].points[0].mark == ""


def test_a_stale_boundary_fix_is_revalidated_and_refused():
    """A proposal checked before a neighbour moved must not be trusted."""
    tg = _pair(0.4968)
    stale = lc.Issue(file="f", kind=lc.BOUNDARY_DRIFT, tier="allo", index=1,
                     time=0.4968, current="0.4968", proposed="9.99",
                     detail="from another run")
    applied, skipped = lc.apply_fixes(tg, [stale])
    assert (applied, skipped) == (0, 1)
    assert tg.tiers[1].intervals[1].xmin == pytest.approx(0.4968)


def test_a_point_fix_moves_the_point():
    phones = _itier("phones", [(0.0, 0.5, "t"), (0.5, DUR, "a")])
    tg = _grid(phones, _ptier("targets", [(0.4968, "T")]))
    lc.apply_fixes(tg, lc.check_boundaries(tg, "f", "phones", ["targets"],
                                           0.020))
    assert tg.tiers[1].points[0].time == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Running over a directory
# ---------------------------------------------------------------------------

@pytest.fixture
def corpus(tmp_path):
    """Two grids with a fault each, plus one that cannot be parsed."""
    good = _grid(_itier("phones", [(0.0, 0.5, "t"), (0.5, DUR, "a")]))
    good.save(str(tmp_path / "clean.TextGrid"))
    _grid(_holey()).save(str(tmp_path / "holey.TextGrid"))
    (tmp_path / "broken.TextGrid").write_text("not a TextGrid at all")
    return tmp_path


def test_the_directory_run_finds_faults_and_names_what_it_could_not_read(corpus):
    opts = lc.CheckOptions(unlabelled_tiers=("phones",))
    issues, unreadable = lc.check_directory(str(corpus), opts,
                                            TextGrid.from_file)
    assert {i.file for i in issues} == {"holey"}
    assert [f for f, _ in unreadable] == ["broken.TextGrid"]


def test_one_unreadable_grid_does_not_hide_the_rest(corpus):
    """The corrupt file sorts first; the others must still be checked."""
    opts = lc.CheckOptions(unlabelled_tiers=("phones",))
    issues, unreadable = lc.check_directory(str(corpus), opts,
                                            TextGrid.from_file)
    assert issues and unreadable


def test_an_empty_options_set_runs_no_checks(corpus):
    issues, _ = lc.check_directory(str(corpus), lc.CheckOptions(),
                                   TextGrid.from_file)
    assert issues == []


def test_tier_names_are_unioned_across_the_corpus(tmp_path):
    _grid(_itier("phones", [(0.0, DUR, "a")])).save(
        str(tmp_path / "a.TextGrid"))
    _grid(_itier("words", [(0.0, DUR, "kupu")]),
          _ptier("targets", [(1.0, "T")])).save(str(tmp_path / "b.TextGrid"))
    names = dict(lc.tier_names_in(str(tmp_path), TextGrid.from_file))
    assert names == {"phones": "IntervalTier", "words": "IntervalTier",
                     "targets": "TextTier"}


def test_the_symbol_check_needs_something_to_validate_against(corpus):
    """Naming tiers is not enough; with no policy there is nothing to say."""
    opts = lc.CheckOptions(symbol_tiers=("phones",))
    issues, _ = lc.check_directory(str(corpus), opts, TextGrid.from_file)
    assert issues == []


# ---------------------------------------------------------------------------
# Writing fixes back to disk
# ---------------------------------------------------------------------------

def test_saving_keeps_the_previous_file_beside_it(tmp_path):
    path = str(tmp_path / "g.TextGrid")
    _grid(_holey()).save(path)
    original = open(path, encoding="utf-8").read()

    tg = TextGrid.from_file(path)
    lc.apply_fixes(tg, lc.check_unlabelled(tg, "g", ["phones"]))
    lc.save_with_backup(tg, path)

    backup = path + lc.BACKUP_SUFFIX
    assert os.path.exists(backup)
    assert open(backup, encoding="utf-8").read() == original
    assert open(path, encoding="utf-8").read() != original


def test_a_second_run_does_not_overwrite_the_pristine_backup(tmp_path):
    """The first backup is the annotator's own work — the one worth keeping."""
    path = str(tmp_path / "g.TextGrid")
    _grid(_holey()).save(path)
    original = open(path, encoding="utf-8").read()

    for _ in range(2):
        tg = TextGrid.from_file(path)
        lc.apply_fixes(tg, lc.check_unlabelled(tg, "g", ["phones"]))
        lc.save_with_backup(tg, path)

    assert open(path + lc.BACKUP_SUFFIX, encoding="utf-8").read() == original


def test_the_fix_survives_the_round_trip_through_the_file(tmp_path):
    path = str(tmp_path / "g.TextGrid")
    _grid(_holey()).save(path)
    tg = TextGrid.from_file(path)
    lc.apply_fixes(tg, lc.check_unlabelled(tg, "g", ["phones"]))
    lc.save_with_backup(tg, path)

    reloaded = TextGrid.from_file(path)
    assert reloaded.tiers[0].intervals[2].text == lc.PAUSE_MARKER
    assert lc.check_unlabelled(reloaded, "g", ["phones"]) == []


def test_a_boundary_fix_survives_the_round_trip(tmp_path):
    path = str(tmp_path / "g.TextGrid")
    _pair(0.4968).save(path)
    tg = TextGrid.from_file(path)
    lc.apply_fixes(tg, lc.check_boundaries(tg, "g", "phones", ["allo"], 0.020))
    lc.save_with_backup(tg, path)

    reloaded = TextGrid.from_file(path)
    assert reloaded.tiers[1].intervals[1].xmin == pytest.approx(0.5)
    assert lc.check_boundaries(reloaded, "g", "phones", ["allo"], 0.020) == []


def test_an_unwritable_path_raises_rather_than_failing_silently(tmp_path):
    """The caller reports the file by name; it must not think it succeeded."""
    target = tmp_path / "nosuchdir" / "g.TextGrid"
    with pytest.raises(OSError):
        lc.save_with_backup(_grid(_holey()), str(target))


def test_names_map_back_to_their_files_whatever_the_extension_case(tmp_path):
    _grid(_holey()).save(str(tmp_path / "upper.TextGrid"))
    _grid(_holey()).save(str(tmp_path / "lower.textgrid"))
    found = lc.paths_by_name(str(tmp_path))
    assert set(found) == {"upper", "lower"}
    assert found["lower"].endswith(".textgrid")
