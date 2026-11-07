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


def _rule(dlg, tier, kind, labels=""):
    """Set *tier*'s row in the rules table."""
    table = dlg._rules_table
    row = table._tiers.index(tier)
    combo = table.cellWidget(row, 1)
    combo.setCurrentIndex(combo.findData(kind))
    table.cellWidget(row, 2).setText(labels)


def test_an_unticked_check_contributes_no_tiers(dlg):
    _rule(dlg, "phones", "ipa")
    dlg._labels_cb.setChecked(False)
    opts = dlg.options()
    assert opts.label_policies == {}
    assert opts.unlabelled_tiers, "the other checks are unaffected"


def test_unticking_the_boundary_check_drops_its_reference_tier(dlg):
    dlg._boundary_cb.setChecked(False)
    assert dlg.options().reference_tier == ""


def test_the_tolerance_is_reported_in_seconds(dlg):
    dlg._tolerance_spin.setValue(12.5)
    assert dlg.options().tolerance_s == pytest.approx(0.0125)


def test_every_tier_gets_a_row_and_starts_unchecked(dlg):
    """A fresh corpus judges no labels until a tier is given a rule."""
    assert dlg._rules_table._tiers == ["phones", "allo", "targets"]
    assert dlg.options().label_policies == {}


def test_a_notation_without_an_inventory_accepts_the_chart(dlg):
    _rule(dlg, "phones", "ipa")
    policy = dlg.options().label_policies["phones"]
    assert policy.accept("t") and policy.accept("\u03B8")
    assert policy.kind == lc.UNKNOWN_SYMBOL
    assert len(policy.candidates) > 100


def test_an_inventory_narrows_the_notation(dlg):
    _rule(dlg, "phones", "ipa", "p, t ,k  a")
    policy = dlg.options().label_policies["phones"]
    assert policy.accept("t")
    assert not policy.accept("\u03B8"), "valid IPA, but not in this corpus"
    assert policy.candidates == frozenset({"p", "t", "k", "a"})
    assert policy.kind == lc.NOT_IN_INVENTORY


def test_each_tier_can_use_its_own_notation(dlg):
    _rule(dlg, "phones", "ipa")
    _rule(dlg, "allo", "xsampa")
    policies = dlg.options().label_policies
    assert policies["allo"].accept("t_h")
    assert not policies["phones"].accept("t_h")


def test_a_tier_can_list_its_own_labels(dlg):
    _rule(dlg, "targets", "list", "T, T1, T2")
    policy = dlg.options().label_policies["targets"]
    assert policy.accept("T1")
    assert not policy.accept("t1"), "lists are case-sensitive"
    assert policy.kind == lc.NOT_ALLOWED


def test_the_labels_field_is_off_until_a_rule_is_chosen(dlg):
    row = dlg._rules_table._tiers.index("targets")
    assert not dlg._rules_table.cellWidget(row, 2).isEnabled()
    _rule(dlg, "targets", "list", "T")
    assert dlg._rules_table.cellWidget(row, 2).isEnabled()


def test_rules_for_tiers_this_corpus_lacks_are_kept(dlg):
    """A profile shared from another corpus loses nothing on saving."""
    foreign = cp.LabelRule("syllables", "list", ("S", "W"))
    dlg._rules_table.set_rules((cp.LabelRule("phones", "ipa"), foreign))
    assert foreign in dlg._rules_table.rules()
    # Still applied: tiers are listed from the first files only, and a tier
    # missing there may yet turn up further into the corpus.
    assert "syllables" in dlg.options().label_policies


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

def test_the_configured_run_finds_every_fault_kind(dlg, corpus):
    _rule(dlg, "phones", "ipa", "t a")
    _rule(dlg, "targets", "list", "T2")
    issues, unreadable = lc.check_directory(
        corpus, dlg.options(), TextGrid.from_file)
    assert not unreadable
    assert {i.kind for i in issues} == {
        lc.UNLABELLED, lc.NOT_IN_INVENTORY, lc.NOT_ALLOWED,
        lc.BOUNDARY_DRIFT}


def test_accepted_fixes_reach_the_file_and_clear_the_fault(dlg, corpus):
    dlg._boundary_cb.setChecked(True)
    dlg._labels_cb.setChecked(False)
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


# ---------------------------------------------------------------------------
# The corpus profile
# ---------------------------------------------------------------------------

import corpus_profile as cp  # noqa: E402
from corpus_profile import CorpusProfile  # noqa: E402


def _saved(corpus, **check):
    profile = CorpusProfile(
        non_speech_labels=("-", "<noise>"), audio_dir="D:/audio",
    ).with_label_check(**check)
    profile.save_to_folder(corpus)
    return profile


def test_a_saved_profile_sets_up_the_dialog(qapp, corpus):
    _saved(corpus, label_rules=(cp.LabelRule("phones", "sampa", ("t", "a")),
                                cp.LabelRule("targets", "list", ("T1",))),
           tolerance_ms=7.5, boundaries=False, reference_tier="allo")
    d = _LabelCheckDialog(corpus)
    assert d._non_speech_edit.text() == "-, <noise>"
    assert d._audio_dir_edit.text() == "D:/audio"
    table = d._rules_table
    assert table.cellWidget(0, 1).currentData() == "sampa"
    assert table.cellWidget(0, 2).text() == "t a"
    assert table.cellWidget(2, 2).text() == "T1"
    assert table.cellWidget(1, 1).currentData() == ""
    assert d._tolerance_spin.value() == 7.5
    assert not d._boundary_cb.isChecked()
    assert d._ref_combo.currentText() == "allo"


def test_what_is_shown_is_what_was_saved(qapp, corpus):
    saved = _saved(corpus, unlabelled_tiers=("phones", "targets"),
                   label_rules=(cp.LabelRule("phones", "ipa", ("a", "t")),),
                   reference_tier="phones")
    d = _LabelCheckDialog(corpus)
    assert d.profile() == saved


def test_the_non_speech_labels_reach_the_check(dlg):
    dlg._non_speech_edit.setText("-, <p:>")
    assert dlg.options().non_speech == ("-", "<p:>")


def test_save_to_corpus_creates_the_profile(dlg, corpus):
    assert not os.path.exists(cp.profile_path(corpus))
    _rule(dlg, "targets", "list", "T, T1, T2")
    dlg._non_speech_edit.setText("<p:>")
    dlg._save_to_corpus()
    saved = CorpusProfile.for_folder(corpus)
    assert saved.non_speech_labels == ("<p:>",)
    assert cp.LabelRule("targets", "list", ("T", "T1", "T2")) in \
        saved.label_check.label_rules


def test_running_a_check_saves_nothing(dlg, corpus, monkeypatch):
    """Saving is the user's explicit choice, never a side effect."""
    from PyQt6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(
        lambda *a, **k: None))
    _rule(dlg, "targets", "list", "T")
    dlg._run()
    assert not os.path.exists(cp.profile_path(corpus))


def test_the_status_says_when_there_is_no_profile(dlg):
    assert "none saved yet" in dlg._profile_status.text()


def test_the_status_tracks_unsaved_changes(dlg):
    dlg._save_to_corpus()
    assert "saved with this corpus" in dlg._profile_status.text()
    _rule(dlg, "targets", "list", "T")
    assert "unsaved changes" in dlg._profile_status.text()
    dlg._save_to_corpus()
    assert "saved with this corpus" in dlg._profile_status.text()


def test_a_list_with_no_labels_is_refused_at_run(dlg, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(
        lambda *a, **k: warned.append(a)))
    _rule(dlg, "targets", "list", "")
    dlg._run()
    assert warned and "targets" in warned[0][2]


def test_a_broken_profile_is_reported_not_overwritten(
        qapp, corpus, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    path = cp.profile_path(corpus)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("{broken")
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(
        lambda *a, **k: warned.append(a)))
    d = _LabelCheckDialog(corpus)
    assert warned
    assert "unreadable" in d._profile_status.text()
    assert open(path, encoding="utf-8").read() == "{broken"


def test_a_profile_can_be_loaded_from_another_corpus(
        dlg, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog
    other = tmp_path / "elsewhere.json"
    CorpusProfile(non_speech_labels=("<sil>",)).save(str(other))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(
        lambda *a, **k: (str(other), "")))
    dlg._load_profile_from()
    assert dlg._non_speech_labels() == ("<sil>",)


def test_save_as_writes_the_settings_on_screen(dlg, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog
    target = tmp_path / "shared.json"
    dlg._non_speech_edit.setText("-, <noise>")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(
        lambda *a, **k: (str(target), "")))
    dlg._save_profile_as()
    assert CorpusProfile.load(str(target)) == dlg.profile()


def test_a_loaded_profile_is_not_an_unsaved_change(qapp, corpus):
    """Settings the file leaves open, like the reference tier, are filled
    in on screen; that alone must not read as unsaved."""
    CorpusProfile().with_label_check(
        label_rules=(cp.LabelRule("targets", "list", ("T",)),)
    ).save_to_folder(corpus)
    d = _LabelCheckDialog(corpus)
    assert "saved with this corpus" in d._profile_status.text()
