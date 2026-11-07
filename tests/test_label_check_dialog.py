"""
Tests for the batch label-check dialog: the wiring between what the user
selects and what the checking module is asked to do, and the ticking rules
that decide which fixes get written.

The checks themselves are covered in test_label_check.py; what matters here is
that an unticked check contributes no tiers, that an inventory takes
precedence over the chart, and that a fault with no proposed fix can never be
ticked into an edit.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtCore import Qt

import label_check as lc
from formant_editor import (
    Interval, Point, TextGrid, Tier, _LabelCheckDialog,
)


DUR = 2.0


@pytest.fixture(scope="session")
def qapp():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


@pytest.fixture
def corpus(tmp_path):
    """One grid with a hole, a bad symbol and a drifted boundary."""
    phones = Tier("phones", "IntervalTier", 0.0, DUR, intervals=[
        Interval(0.0, 0.2, ""), Interval(0.2, 0.5, "t"),
        Interval(0.5, 0.8, ""), Interval(0.8, DUR, "θ")])
    allo = Tier("allo", "IntervalTier", 0.0, DUR, intervals=[
        Interval(0.0, 0.4968, "t0"), Interval(0.4968, DUR, "a")])
    marks = Tier("targets", "TextTier", 0.0, DUR, points=[Point(1.0, "T1")])
    TextGrid(0.0, DUR, [phones, allo, marks]).save(
        str(tmp_path / "one.TextGrid"))
    return str(tmp_path)


@pytest.fixture
def dlg(qapp, corpus):
    d = _LabelCheckDialog(corpus)
    yield d
    d.deleteLater()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def test_tiers_are_discovered_from_the_folder(dlg):
    opts = dlg.options()
    assert set(opts.unlabelled_tiers) == {"phones", "allo", "targets"}
    assert dlg._ref_combo.count() == 3


def test_an_unticked_check_contributes_no_tiers(dlg):
    dlg._symbol_cb.setChecked(False)
    opts = dlg.options()
    assert opts.symbol_tiers == ()
    assert opts.unlabelled_tiers, "the other checks are unaffected"


def test_unticking_the_boundary_check_drops_its_reference_tier(dlg):
    dlg._boundary_cb.setChecked(False)
    assert dlg.options().reference_tier == ""


def test_the_tolerance_is_reported_in_seconds(dlg):
    dlg._tolerance_spin.setValue(12.5)
    assert dlg.options().tolerance_s == pytest.approx(0.0125)


def test_the_inventory_accepts_spaces_or_commas(dlg):
    dlg._inventory_edit.setText("p, t ,k  a")
    assert dlg.options().inventory == frozenset({"p", "t", "k", "a"})


def test_without_an_inventory_the_policy_falls_back_to_the_chart(dlg):
    accept, candidates = dlg._symbol_policy()
    assert accept("t") and accept("θ"), "both are valid IPA"
    assert len(candidates) > 100, "the whole chart is the candidate pool"


def test_an_inventory_overrides_the_chart(dlg):
    dlg._inventory_edit.setText("p t k a e i o u")
    accept, candidates = dlg._symbol_policy()
    assert accept("t")
    assert not accept("θ"), "valid IPA, but not in this corpus"
    assert candidates == frozenset({"p", "t", "k", "a", "e", "i", "o", "u"})


def test_the_notation_choice_reaches_the_options(dlg):
    dlg._notation_combo.setCurrentText("X-SAMPA")
    assert dlg.options().notation == "xsampa"


# ---------------------------------------------------------------------------
# The results table
# ---------------------------------------------------------------------------

def _issues(fixable=True):
    return [
        lc.Issue("one", lc.UNLABELLED, "phones", 2, 0.5, "", "-", "a hole"),
        lc.Issue("one", lc.NOT_IN_INVENTORY, "phones", 3, 0.8, "θ",
                 "t" if fixable else "", "not in the inventory"),
    ]


def test_every_issue_gets_a_row(dlg):
    dlg._populate(_issues())
    assert dlg._table.rowCount() == 2


def test_nothing_is_ticked_until_the_user_ticks_it(dlg):
    """Each fix edits the annotation, so it is opted into, not out of."""
    dlg._populate(_issues())
    assert dlg.accepted_issues() == []


def test_a_fault_with_no_proposal_cannot_be_ticked(dlg):
    dlg._populate(_issues(fixable=False))
    dlg._set_all(True)
    accepted = dlg.accepted_issues()
    assert [i.kind for i in accepted] == [lc.UNLABELLED]


def test_ticking_all_selects_every_fixable_row(dlg):
    dlg._populate(_issues())
    dlg._set_all(True)
    assert len(dlg.accepted_issues()) == 2
    dlg._set_all(False)
    assert dlg.accepted_issues() == []


def test_ticking_by_type_leaves_the_other_types_alone(dlg):
    dlg._populate(_issues())
    dlg._kind_combo.setCurrentText(
        _LabelCheckDialog.ISSUE_LABELS[lc.NOT_IN_INVENTORY])
    dlg._tick_kind()
    assert [i.kind for i in dlg.accepted_issues()] == [lc.NOT_IN_INVENTORY]


def test_apply_stays_disabled_when_nothing_can_be_fixed(dlg):
    dlg._populate([lc.Issue("one", lc.UNLABELLED, "targets", 0, 1.0, "", "",
                            "point carries no mark")])
    assert not dlg._apply_btn.isEnabled()


def test_apply_is_offered_once_something_is_fixable(dlg):
    dlg._populate(_issues())
    assert dlg._apply_btn.isEnabled()


def test_a_blank_current_value_is_shown_as_blank_not_as_nothing(dlg):
    """An empty cell would read as "no data", which is the opposite fault."""
    dlg._populate(_issues())
    assert dlg._table.item(0, 5).text() == "(blank)"


def test_repopulating_replaces_the_previous_results(dlg):
    dlg._populate(_issues())
    dlg._populate(_issues()[:1])
    assert dlg._table.rowCount() == 1


def _proposed_item(dlg, row):
    return dlg._table.item(row, dlg._PROPOSED_COL)


def test_typing_a_fix_makes_the_row_tickable_and_ticks_it(dlg):
    """A label with no suggestion can still be repaired from the table."""
    dlg._populate(_issues(fixable=False))
    _proposed_item(dlg, 1).setText("t")
    accepted = dlg.accepted_issues()
    assert [(i.current, i.proposed) for i in accepted] == [("θ", "t")]
    assert dlg._apply_btn.isEnabled()


def test_typing_over_a_suggestion_replaces_it(dlg):
    dlg._populate(_issues())
    _proposed_item(dlg, 1).setText("s")
    assert dlg._issues[1].proposed == "s"


def test_clearing_a_typed_fix_withdraws_it(dlg):
    dlg._populate(_issues(fixable=False))
    _proposed_item(dlg, 1).setText("t")
    _proposed_item(dlg, 1).setText("  ")
    assert dlg.accepted_issues() == []
    tick = dlg._table.item(1, 0)
    assert not tick.flags() & Qt.ItemFlag.ItemIsUserCheckable


def test_a_boundary_proposal_cannot_be_typed_over(dlg):
    """A hand-typed time is no repair; only the snap is offered."""
    dlg._populate([lc.Issue("one", lc.BOUNDARY_DRIFT, "allo", 1, 0.4968,
                            "0.4968", "", "crosses a neighbour")])
    assert not _proposed_item(dlg, 0).flags() & Qt.ItemFlag.ItemIsEditable


def test_populating_does_not_count_as_typing(dlg):
    """Filling the table must not tick suggestions on the user's behalf."""
    dlg._populate(_issues())
    assert dlg.accepted_issues() == []


def test_the_table_tick_boxes_are_styled(dlg):
    """Unstyled, they draw dark grey on the dark theme and vanish."""
    assert "QTableView::indicator" in dlg.styleSheet()


# ---------------------------------------------------------------------------
# End to end, against real files
# ---------------------------------------------------------------------------

def test_the_configured_run_finds_all_three_fault_kinds(dlg, corpus):
    dlg._inventory_edit.setText("t a t0")
    accept, candidates = dlg._symbol_policy()
    issues, unreadable = lc.check_directory(
        corpus, dlg.options(), TextGrid.from_file,
        accept=accept, candidates=candidates)
    assert not unreadable
    assert {i.kind for i in issues} == {
        lc.UNLABELLED, lc.NOT_IN_INVENTORY, lc.BOUNDARY_DRIFT}


def test_accepted_fixes_reach_the_file_and_clear_the_fault(dlg, corpus):
    dlg._boundary_cb.setChecked(True)
    dlg._symbol_cb.setChecked(False)
    dlg._unlabelled_cb.setChecked(False)
    issues, _ = lc.check_directory(corpus, dlg.options(), TextGrid.from_file)
    dlg._populate(issues)
    dlg._set_all(True)

    path = lc.paths_by_name(corpus)["one"]
    tg = TextGrid.from_file(path)
    lc.apply_fixes(tg, dlg.accepted_issues())
    lc.save_with_backup(tg, path)

    again, _ = lc.check_directory(corpus, dlg.options(), TextGrid.from_file)
    assert again == []
    assert os.path.exists(path + lc.BACKUP_SUFFIX)
