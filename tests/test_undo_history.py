"""
Tests for the lifetime of the formant edit history.

An undo entry holds (formant, frame index, old value) triples that mean
something only against the array they were recorded from. The history must
therefore end when the track does -- a new audio file, a loaded .formants, a
re-analysis -- while surviving edits made to a track in place.

These drive the real SpectrogramCanvas rather than a stand-in, because the
invariant lives in the property that installs a track, not in the undo
routine itself.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication

from formant_editor import FormantData, SpectrogramCanvas, UndoEntry


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


def _track(n_frames, fill, n_formants=3):
    return FormantData(times=np.linspace(0.0, 0.1 * n_frames, n_frames),
                       values=np.full((n_formants, n_frames), float(fill)),
                       n_formants=n_formants)


def _record_edit(canvas, f_idx, frame_idx, new_val):
    """Edit the track in place the way a stroke does, and log it for undo."""
    fd = canvas.formant_data
    old_val = float(fd.values[f_idx, frame_idx])
    old_mask = bool(fd.edited_mask[f_idx, frame_idx])
    fd.values[f_idx, frame_idx] = new_val
    fd.edited_mask[f_idx, frame_idx] = True
    canvas._undo_stack.append(
        UndoEntry("Draw", [(f_idx, frame_idx, old_val, old_mask,
                            new_val, True)]))


@pytest.fixture
def canvas(qapp):
    c = SpectrogramCanvas()
    c.formant_data = _track(100, 500.0)
    yield c


class TestHistorySurvivesEditing:
    """Editing a track in place must not disturb its history."""

    def test_an_edit_is_undoable(self, canvas):
        _record_edit(canvas, 0, 50, 999.0)
        assert canvas.undo() is True
        assert canvas.formant_data.values[0, 50] == 500.0

    def test_an_undone_edit_is_redoable(self, canvas):
        _record_edit(canvas, 0, 50, 999.0)
        canvas.undo()
        assert canvas.redo() is True
        assert canvas.formant_data.values[0, 50] == 999.0

    def test_several_edits_unwind_in_order(self, canvas):
        _record_edit(canvas, 0, 10, 111.0)
        _record_edit(canvas, 0, 20, 222.0)
        assert len(canvas._undo_stack) == 2
        canvas.undo()
        assert canvas.formant_data.values[0, 20] == 500.0
        assert canvas.formant_data.values[0, 10] == 111.0

    def test_resetting_in_place_keeps_the_history(self, canvas):
        """``reset_to_original`` mutates the track; it does not replace it."""
        _record_edit(canvas, 0, 50, 999.0)
        canvas.formant_data.reset_to_original(0)
        assert len(canvas._undo_stack) == 1


class TestHistoryEndsWithTheTrack:
    """The bug: Ctrl+Z after opening a new file replayed the old file's edits."""

    def test_installing_a_new_track_drops_the_history(self, canvas):
        _record_edit(canvas, 0, 50, 999.0)
        canvas.formant_data = _track(100, 300.0)
        assert canvas._undo_stack == []

    def test_the_redo_stack_goes_too(self, canvas):
        _record_edit(canvas, 0, 50, 999.0)
        canvas.undo()
        assert canvas._redo_stack, "fixture precondition: something to redo"
        canvas.formant_data = _track(100, 300.0)
        assert canvas._redo_stack == []

    def test_undo_on_the_new_track_does_nothing(self, canvas):
        _record_edit(canvas, 0, 50, 999.0)
        canvas.formant_data = _track(100, 300.0)
        assert canvas.undo() is False

    def test_the_new_track_is_left_alone(self, canvas):
        """The corruption case: frame 50 of the new file must keep its value."""
        _record_edit(canvas, 0, 50, 999.0)
        canvas.formant_data = _track(100, 300.0)
        canvas.undo()
        assert canvas.formant_data.values[0, 50] == 300.0

    def test_a_shorter_new_track_does_not_raise(self, canvas):
        """Frame 50 does not exist in a 20-frame file -- once an IndexError."""
        _record_edit(canvas, 0, 50, 999.0)
        canvas.formant_data = _track(20, 300.0)
        assert canvas.undo() is False
        assert canvas.formant_data.values[0, 10] == 300.0

    def test_clearing_the_track_clears_the_history(self, canvas):
        _record_edit(canvas, 0, 50, 999.0)
        canvas.formant_data = None
        assert canvas._undo_stack == []
        assert canvas.undo() is False


class TestFormantDataProperty:
    """The property itself: a plain attribute from the outside."""

    def test_it_reads_back_what_was_installed(self, canvas):
        fd = _track(10, 400.0)
        canvas.formant_data = fd
        assert canvas.formant_data is fd

    def test_a_fresh_canvas_has_no_track_and_no_history(self, qapp):
        c = SpectrogramCanvas()
        assert c.formant_data is None
        assert c._undo_stack == []
        assert c._redo_stack == []
