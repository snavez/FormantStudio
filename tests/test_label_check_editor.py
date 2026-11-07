"""
Tests for working through Label Check's results in the main window.

The check window stays open beside the editor, and an issue is opened by
double-clicking it: the recording and its TextGrid load, the issue's tier
becomes active and the offending interval or point is selected and in view.
Fixes written while that file is open must not be lost to a later save.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox

import label_check as lc
from formant_editor import (
    Interval, MainWindow, Point, TextGrid, Tier, _LabelCheckDialog,
    _PlaceholderDelegate, find_audio,
)

DUR = 1.0


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


def _write_wav(path, seconds=DUR, sr=16000):
    import parselmouth
    samples = np.random.default_rng(0).standard_normal(int(sr * seconds))
    parselmouth.Sound(samples * 0.05, sampling_frequency=sr).save(
        path, "WAV")


def _write_grid(path, phone_labels=("a", "θ", "u")):
    edges = np.linspace(0.0, DUR, len(phone_labels) + 1)
    phones = Tier("phones", "IntervalTier", 0.0, DUR, intervals=[
        Interval(float(a), float(b), lab)
        for a, b, lab in zip(edges[:-1], edges[1:], phone_labels)])
    marks = Tier("marks", "TextTier", 0.0, DUR, points=[
        Point(0.2, "yes"), Point(0.8, "no")])
    TextGrid(0.0, DUR, [phones, marks]).save(path)


@pytest.fixture
def corpus(tmp_path):
    """Two recordings, each with its TextGrid beside it."""
    for name in ("one", "two"):
        _write_wav(str(tmp_path / f"{name}.wav"))
        _write_grid(str(tmp_path / f"{name}.TextGrid"))
    return tmp_path


@pytest.fixture
def window(qapp):
    mw = MainWindow()
    yield mw
    mw._textgrid_dirty = mw._formants_dirty = False
    mw.close()
    # Destroy the window now rather than whenever the garbage collector
    # reaches it: a pile of half-torn-down plot widgets otherwise crashes a
    # later test in the middle of building its own.
    mw.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def _issue(file="one", tier="phones", index=1, kind=lc.UNKNOWN_SYMBOL,
           proposed=""):
    return lc.Issue(file, kind, tier, index, 0.0, "θ", proposed,
                    "not a recognised symbol")


class TestFindAudio:

    def test_finds_the_recording_whatever_the_extension_case(self, tmp_path):
        _write_wav(str(tmp_path / "x.WAV"))
        assert find_audio(str(tmp_path), "x") == str(tmp_path / "x.WAV")

    def test_none_when_absent(self, tmp_path):
        assert find_audio(str(tmp_path), "nothing") is None

    def test_none_for_a_missing_folder(self, tmp_path):
        assert find_audio(str(tmp_path / "no-such-dir"), "x") is None


class TestShowItem:

    def test_selects_and_activates(self, window, corpus):
        window.open_audio(str(corpus / "one.wav"))
        assert window.canvas.show_item("phones", 2)
        assert window.canvas._active_tier == 0
        assert window.canvas._selected_interval == (0, 2)

    def test_brings_the_item_into_view(self, window, corpus):
        window.open_audio(str(corpus / "one.wav"))
        window.canvas.set_view(0.0, 0.1)
        window.canvas.show_item("marks", 1)          # the point at 0.8 s
        c = window.canvas
        assert c.view_start <= 0.8 <= c.view_end
        assert c._selected_interval == (1, 1)

    def test_missing_tier_or_item_is_refused(self, window, corpus):
        window.open_audio(str(corpus / "one.wav"))
        assert not window.canvas.show_item("nosuch", 0)
        assert not window.canvas.show_item("phones", 99)


class TestOpenIssue:

    def test_opens_recording_and_grid_at_the_issue(self, window, corpus):
        window._open_issue(str(corpus / "one.TextGrid"), "phones", 1)
        assert os.path.basename(window._filepath) == "one.wav"
        assert window.canvas._selected_interval == (0, 1)
        assert window.label_edit.text() == "θ"

    def test_a_second_issue_in_the_open_file_does_not_reload_it(
            self, window, corpus):
        """Edits in progress survive moving between a file's issues."""
        window._open_issue(str(corpus / "one.TextGrid"), "phones", 1)
        window.canvas.textgrid_data.tiers[0].intervals[0].text = "edited"
        window._textgrid_dirty = True
        window._open_issue(str(corpus / "one.TextGrid"), "phones", 2)
        assert window.canvas.textgrid_data.tiers[0].intervals[0].text \
            == "edited"
        assert window.canvas._selected_interval == (0, 2)

    def test_switching_file_asks_about_unsaved_work(
            self, window, corpus, monkeypatch):
        window._open_issue(str(corpus / "one.TextGrid"), "phones", 1)
        window._textgrid_dirty = True
        monkeypatch.setattr(QMessageBox, "question", staticmethod(
            lambda *a, **k: QMessageBox.StandardButton.Cancel))
        window._open_issue(str(corpus / "two.TextGrid"), "phones", 1)
        assert os.path.basename(window._filepath) == "one.wav"

    def test_without_its_recording_the_grid_opens_alone(
            self, window, tmp_path, monkeypatch):
        """No prompt: the issue is shown, and the recording can follow."""
        _write_grid(str(tmp_path / "lonely.TextGrid"))
        asked = []
        monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                            staticmethod(lambda *a, **k: asked.append(a)))
        window._open_issue(str(tmp_path / "lonely.TextGrid"), "phones", 1)
        assert asked == []
        assert window.canvas.sound is None
        assert window.canvas._selected_interval == (0, 1)
        assert window.label_edit.text() == "\u03B8"

    def test_the_issue_is_centred_in_a_grid_alone(self, window, tmp_path):
        path = str(tmp_path / "long.TextGrid")
        marks = Tier("marks", "TextTier", 0.0, 6.0, points=[
            Point(1.0, "yes"), Point(4.0, "no")])
        TextGrid(0.0, 6.0, [marks]).save(path)
        window._open_issue(path, "marks", 1)
        c = window.canvas
        assert c.view_end - c.view_start < 6.0
        assert abs((c.view_start + c.view_end) / 2 - 4.0) < 1e-6

    def test_recordings_kept_apart_are_found_once_pointed_at(
            self, window, tmp_path, monkeypatch):
        grids, audio = tmp_path / "grids", tmp_path / "audio"
        grids.mkdir()
        audio.mkdir()
        for name in ("one", "two"):
            _write_grid(str(grids / f"{name}.TextGrid"))
            _write_wav(str(audio / f"{name}.wav"))
        window._open_issue(str(grids / "one.TextGrid"), "phones", 0)
        assert window.canvas.sound is None
        monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(
            lambda *a, **k: (str(audio / "one.wav"), "")))
        window._choose_recording_for_textgrid()
        window._open_issue(str(grids / "two.TextGrid"), "phones", 0)
        assert os.path.basename(window._filepath) == "two.wav"


class TestAudioFolder:
    """Recordings kept apart from their TextGrids, named in Label Check."""

    @pytest.fixture
    def split(self, tmp_path):
        grids, audio = tmp_path / "grids", tmp_path / "audio"
        grids.mkdir()
        audio.mkdir()
        for name in ("one", "two"):
            _write_grid(str(grids / f"{name}.TextGrid"))
            _write_wav(str(audio / f"{name}.wav"))
        return grids, audio

    def test_the_recording_opens_from_the_audio_folder(self, window, split):
        grids, audio = split
        window._open_issue(str(grids / "two.TextGrid"), "phones", 1,
                           str(audio))
        assert os.path.basename(window._filepath) == "two.wav"
        assert window._textgrid_path == str(grids / "two.TextGrid")
        assert window.canvas._selected_interval == (0, 1)

    def test_a_grid_open_alone_gains_its_recording(self, window, split):
        """Naming the folder later attaches the recording, edits kept."""
        grids, audio = split
        path = str(grids / "one.TextGrid")
        window._open_issue(path, "phones", 0)
        assert window.canvas.sound is None
        window.canvas.textgrid_data.tiers[0].intervals[2].text = "mine"
        window._textgrid_dirty = True
        window._open_issue(path, "phones", 1, str(audio))
        assert window.canvas.sound is not None
        assert window.canvas.textgrid_data.tiers[0].intervals[2].text \
            == "mine"
        assert window.canvas._selected_interval == (0, 1)

    def test_a_wrong_audio_folder_still_opens_the_grid(
            self, window, split, tmp_path):
        grids, _ = split
        window._open_issue(str(grids / "one.TextGrid"), "phones", 0,
                           str(tmp_path / "nowhere"))
        assert window.canvas.sound is None
        assert window._textgrid_path == str(grids / "one.TextGrid")


class TestTextGridAlone:

    @pytest.fixture
    def grid(self, tmp_path):
        path = str(tmp_path / "solo.TextGrid")
        _write_grid(path)
        return path

    def test_the_grid_gives_the_time_axis(self, window, grid):
        assert window.open_textgrid_alone(grid)
        c = window.canvas
        assert c.has_timeline
        assert (c.view_start, c.view_end) == (0.0, DUR)
        assert c._tier_plots

    def test_any_open_recording_is_closed(self, window, corpus, grid):
        window.open_audio(str(corpus / "one.wav"))
        window.open_textgrid_alone(grid)
        assert window.canvas.sound is None
        assert window.canvas.spectrogram_data is None
        assert window._filepath is None

    def test_labels_can_be_edited_and_saved(self, window, grid):
        window.open_textgrid_alone(grid)
        window.canvas.show_item("phones", 1)
        window.label_edit.selectAll()
        window.label_edit.insert("t")
        assert window._textgrid_dirty
        window.save_textgrid()
        assert TextGrid.from_file(grid).tiers[0].intervals[1].text == "t"

    def test_zoom_works_without_a_recording(self, window, grid):
        window.open_textgrid_alone(grid)
        before = window.canvas.view_width
        window._menu_action(lambda: window.canvas.zoom(0.5))
        assert window.canvas.view_width < before

    def test_loading_a_textgrid_with_nothing_open_opens_it_alone(
            self, window, grid):
        window.load_textgrid(grid)
        assert window.canvas.has_timeline
        assert window._textgrid_path == grid

    def test_attaching_the_recording_keeps_unsaved_edits(
            self, window, corpus):
        window.open_textgrid_alone(str(corpus / "one.TextGrid"))
        window.canvas.textgrid_data.tiers[0].intervals[0].text = "mine"
        window._textgrid_dirty = True
        assert window.attach_recording(str(corpus / "one.wav"))
        assert window.canvas.sound is not None
        assert window.canvas.textgrid_data.tiers[0].intervals[0].text == "mine"
        assert window._textgrid_dirty

    def test_attaching_keeps_the_view(self, window, corpus):
        window.open_textgrid_alone(str(corpus / "one.TextGrid"))
        window.canvas.set_view(0.2, 0.4)
        window.attach_recording(str(corpus / "one.wav"))
        c = window.canvas
        assert (round(c.view_start, 6), round(c.view_end, 6)) == (0.2, 0.4)

    def test_attach_without_a_grid_says_so(self, window, monkeypatch):
        told = []
        monkeypatch.setattr(QMessageBox, "information", staticmethod(
            lambda *a, **k: told.append(a)))
        window._choose_recording_for_textgrid()
        assert told

    def test_a_bad_grid_is_reported_not_raised(
            self, window, tmp_path, monkeypatch):
        bad = tmp_path / "bad.TextGrid"
        bad.write_text("not a textgrid", encoding="utf-8")
        shown = []
        monkeypatch.setattr(QMessageBox, "critical", staticmethod(
            lambda *a, **k: shown.append(a)))
        assert not window.open_textgrid_alone(str(bad))
        assert shown


class TestFixesWhileOpen:

    def test_a_clean_open_file_is_reloaded(self, window, corpus):
        path = str(corpus / "one.TextGrid")
        window._open_issue(path, "phones", 1)
        tg = TextGrid.from_file(path)
        tg.tiers[0].intervals[1].text = "t"
        tg.save(path)
        window._on_files_fixed([path])
        assert window.canvas.textgrid_data.tiers[0].intervals[1].text == "t"

    def test_unsaved_edits_are_kept_and_the_user_warned(
            self, window, corpus, monkeypatch):
        path = str(corpus / "one.TextGrid")
        window._open_issue(path, "phones", 1)
        window.canvas.textgrid_data.tiers[0].intervals[0].text = "mine"
        window._textgrid_dirty = True
        warned = []
        monkeypatch.setattr(QMessageBox, "warning", staticmethod(
            lambda *a, **k: warned.append(a)))
        window._on_files_fixed([path])
        assert warned
        assert window.canvas.textgrid_data.tiers[0].intervals[0].text == "mine"

    def test_fixes_to_other_files_leave_the_editor_alone(
            self, window, corpus, monkeypatch):
        window._open_issue(str(corpus / "one.TextGrid"), "phones", 1)
        reloaded = []
        monkeypatch.setattr(window, "load_textgrid",
                            lambda *a, **k: reloaded.append(a))
        window._on_files_fixed([str(corpus / "two.TextGrid")])
        assert reloaded == []


class TestDialog:

    @pytest.fixture
    def dlg(self, qapp, corpus):
        d = _LabelCheckDialog(str(corpus))
        yield d
        d.deleteLater()

    def test_double_click_asks_to_open_the_issue(self, dlg, corpus):
        dlg._populate([_issue(index=2)])
        got = []
        dlg.open_requested.connect(lambda *a: got.append(a))
        dlg._on_cell_double_clicked(0, 1)              # the File column
        assert got == [(str(corpus / "one.TextGrid"), "phones", 2,
                        str(corpus))]

    def test_without_an_audio_folder_the_textgrid_folder_is_used(
            self, dlg, corpus):
        assert dlg.audio_dir() == str(corpus)

    def test_the_audio_folder_is_passed_with_the_issue(
            self, dlg, tmp_path):
        dlg._audio_dir_edit.setText(str(tmp_path / "audio"))
        dlg._populate([_issue()])
        got = []
        dlg.open_requested.connect(lambda *a: got.append(a))
        dlg._open_row(0)
        assert got[0][3] == str(tmp_path / "audio")

    def test_double_click_on_an_editable_fix_edits_instead(self, dlg):
        dlg._populate([_issue()])
        got = []
        dlg.open_requested.connect(lambda *a: got.append(a))
        dlg._on_cell_double_clicked(0, dlg._PROPOSED_COL)
        assert got == []

    def test_open_in_editor_with_nothing_selected_says_so(
            self, dlg, monkeypatch):
        dlg._populate([_issue()])
        dlg._table.setCurrentCell(-1, -1)
        told = []
        monkeypatch.setattr(QMessageBox, "information", staticmethod(
            lambda *a, **k: told.append(a)))
        dlg._open_selected()
        assert told

    def test_label_without_a_fix_shows_a_hint_not_a_dash(self, dlg):
        """A dash reads as the pause marker."""
        dlg._populate([_issue()])
        item = dlg._table.item(0, dlg._PROPOSED_COL)
        assert item.text() == ""
        assert "no suggestion" in item.data(_PlaceholderDelegate.ROLE)

    def test_unsafe_snap_says_why_there_is_no_fix(self, dlg):
        dlg._populate([_issue(kind=lc.BOUNDARY_DRIFT)])
        hint = dlg._table.item(0, dlg._PROPOSED_COL).data(
            _PlaceholderDelegate.ROLE)
        assert "cannot snap" in hint

    def test_the_hint_is_painted_only(self, dlg):
        """Editing must start from an empty cell, not from the hint."""
        dlg._populate([_issue()])
        item = dlg._table.item(0, dlg._PROPOSED_COL)
        assert item.data(Qt.ItemDataRole.EditRole) == ""


class TestFileMenu:

    def test_open_names_audio_not_just_wav(self, window):
        file_menu = next(a.menu() for a in window.menuBar().actions()
                         if a.text() == "&File")
        texts = [a.text() for a in file_menu.actions()]
        assert "&Open Audio..." in texts
        assert not any("WAV" in x for x in texts)


class TestWindowStaysOpen:

    def test_label_check_is_not_modal(self, window):
        window._check_label_files()
        assert not window._label_check_dialog.isModal()

    def test_reopening_finds_the_same_window(self, window):
        window._check_label_files()
        first = window._label_check_dialog
        first._populate([_issue()])
        window._check_label_files()
        assert window._label_check_dialog is first
        assert first._table.rowCount() == 1
