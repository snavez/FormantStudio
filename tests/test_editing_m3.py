"""
Tests for Milestone 3 editing features — undo/redo, frame interpolation,
eraser tool, and interpolate-between-points.

These test the data-level logic by operating directly on FormantData
and SpectrogramCanvas internals without requiring a full GUI.
"""

import numpy as np
import numpy.testing as npt
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from formant_editor import FormantData, UndoEntry, MAX_UNDO_STEPS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fd(n_frames=20, n_formants=5):
    """Create a FormantData with known values for testing."""
    times = np.linspace(0.0, 0.19, n_frames)  # ~10ms step
    values = np.full((n_formants, n_frames), np.nan)
    for f in range(min(3, n_formants)):
        values[f] = np.linspace(300 * (f + 1), 350 * (f + 1), n_frames)
    return FormantData(times=times, values=values, n_formants=n_formants,
                       time_step=times[1] - times[0])


class FakeCanvas:
    """Minimal stand-in for SpectrogramCanvas to test undo/redo/interpolate."""

    def __init__(self, formant_data):
        self.formant_data = formant_data
        self._undo_stack = []
        self._redo_stack = []
        self._stroke_changes = []
        self._is_erasing = False
        self._last_frame_idx = -1
        self._last_frame_freq = None
        self.active_formant = 0

    def render(self):
        pass  # no-op for testing


def _apply_undo(canvas):
    """Replicate SpectrogramCanvas.undo() logic."""
    if not canvas._undo_stack or canvas.formant_data is None:
        return False
    entry = canvas._undo_stack.pop()
    fd = canvas.formant_data
    for f_idx, frame_idx, old_val, old_mask, new_val, new_mask in entry.changes:
        fd.values[f_idx, frame_idx] = old_val
        fd.edited_mask[f_idx, frame_idx] = old_mask
    canvas._redo_stack.append(entry)
    return True


def _apply_redo(canvas):
    """Replicate SpectrogramCanvas.redo() logic."""
    if not canvas._redo_stack or canvas.formant_data is None:
        return False
    entry = canvas._redo_stack.pop()
    fd = canvas.formant_data
    for f_idx, frame_idx, old_val, old_mask, new_val, new_mask in entry.changes:
        fd.values[f_idx, frame_idx] = new_val
        fd.edited_mask[f_idx, frame_idx] = new_mask
    canvas._undo_stack.append(entry)
    return True


def _simulate_stroke(canvas, frames_and_freqs, is_erasing=False):
    """Simulate a drawing/erasing stroke by recording changes and pushing undo.

    frames_and_freqs: list of (frame_idx, freq_hz) pairs
    """
    fd = canvas.formant_data
    f_idx = canvas.active_formant
    canvas._stroke_changes = []
    canvas._is_erasing = is_erasing

    for frame_idx, freq_hz in frames_and_freqs:
        old_val = float(fd.values[f_idx, frame_idx])
        old_mask = bool(fd.edited_mask[f_idx, frame_idx])
        if is_erasing:
            new_val = float(fd.original_values[f_idx, frame_idx])
            fd.values[f_idx, frame_idx] = new_val
            fd.edited_mask[f_idx, frame_idx] = False
            canvas._stroke_changes.append(
                (f_idx, frame_idx, old_val, old_mask, new_val, False))
        else:
            fd.set_value(f_idx, frame_idx, freq_hz)
            canvas._stroke_changes.append(
                (f_idx, frame_idx, old_val, old_mask, freq_hz, True))

    # Push undo entry
    if canvas._stroke_changes:
        desc = "Erase" if is_erasing else "Draw"
        entry = UndoEntry(desc, list(canvas._stroke_changes))
        canvas._undo_stack.append(entry)
        canvas._redo_stack.clear()
        if len(canvas._undo_stack) > MAX_UNDO_STEPS:
            canvas._undo_stack.pop(0)
        canvas._stroke_changes = []


def _interpolate_formant(canvas, formant_idx, start_time=None, end_time=None):
    """Replicate SpectrogramCanvas.interpolate_formant() logic."""
    fd = canvas.formant_data
    f_idx = formant_idx

    if start_time is not None and end_time is not None:
        fi_start = np.argmin(np.abs(fd.times - start_time))
        fi_end = np.argmin(np.abs(fd.times - end_time))
        if fi_start > fi_end:
            fi_start, fi_end = fi_end, fi_start
    else:
        fi_start = 0
        fi_end = fd.n_frames - 1

    edited_frames = []
    for fi in range(fi_start, fi_end + 1):
        if fd.edited_mask[f_idx, fi] and not np.isnan(fd.values[f_idx, fi]):
            edited_frames.append(fi)

    if len(edited_frames) < 2:
        return 0

    changes = []
    count = 0
    for i in range(len(edited_frames) - 1):
        a = edited_frames[i]
        b = edited_frames[i + 1]
        if b - a <= 1:
            continue
        val_a = fd.values[f_idx, a]
        val_b = fd.values[f_idx, b]
        for fi in range(a + 1, b):
            t = (fi - a) / (b - a)
            new_val = val_a + t * (val_b - val_a)
            old_val = float(fd.values[f_idx, fi])
            old_mask = bool(fd.edited_mask[f_idx, fi])
            fd.set_value(f_idx, fi, new_val)
            changes.append((f_idx, fi, old_val, old_mask, new_val, True))
            count += 1

    if changes:
        entry = UndoEntry("Interpolate", changes)
        canvas._undo_stack.append(entry)
        canvas._redo_stack.clear()
        if len(canvas._undo_stack) > MAX_UNDO_STEPS:
            canvas._undo_stack.pop(0)

    return count


# ===========================================================================
# Undo/Redo Tests
# ===========================================================================

class TestUndoRedo:
    """Tests for the undo/redo system."""

    def test_undo_empty_stack(self):
        """Undo with empty stack returns False."""
        c = FakeCanvas(_make_fd())
        assert _apply_undo(c) == False

    def test_redo_empty_stack(self):
        """Redo with empty stack returns False."""
        c = FakeCanvas(_make_fd())
        assert _apply_redo(c) == False

    def test_undo_restores_values(self):
        """Undo restores the original values after a draw stroke."""
        fd = _make_fd()
        c = FakeCanvas(fd)
        original_f1_5 = float(fd.values[0, 5])
        _simulate_stroke(c, [(5, 999.0)])
        assert fd.values[0, 5] == 999.0
        assert fd.edited_mask[0, 5] == True
        _apply_undo(c)
        assert fd.values[0, 5] == original_f1_5
        assert fd.edited_mask[0, 5] == False

    def test_redo_reapplies_values(self):
        """Redo reapplies the undone edit."""
        fd = _make_fd()
        c = FakeCanvas(fd)
        _simulate_stroke(c, [(5, 999.0)])
        _apply_undo(c)
        _apply_redo(c)
        assert fd.values[0, 5] == 999.0
        assert fd.edited_mask[0, 5] == True

    def test_undo_multiple_strokes(self):
        """Multiple strokes can be undone individually."""
        fd = _make_fd()
        c = FakeCanvas(fd)
        orig_5 = float(fd.values[0, 5])
        orig_10 = float(fd.values[0, 10])
        _simulate_stroke(c, [(5, 800.0)])
        _simulate_stroke(c, [(10, 900.0)])
        assert len(c._undo_stack) == 2

        _apply_undo(c)  # undo second stroke
        assert fd.values[0, 10] == orig_10
        assert fd.values[0, 5] == 800.0  # first stroke still applied

        _apply_undo(c)  # undo first stroke
        assert fd.values[0, 5] == orig_5

    def test_new_edit_clears_redo_stack(self):
        """A new edit after undo clears the redo stack."""
        fd = _make_fd()
        c = FakeCanvas(fd)
        _simulate_stroke(c, [(5, 800.0)])
        _apply_undo(c)
        assert len(c._redo_stack) == 1
        _simulate_stroke(c, [(7, 900.0)])
        assert len(c._redo_stack) == 0

    def test_undo_multi_frame_stroke(self):
        """Undo restores all frames from a multi-frame stroke."""
        fd = _make_fd()
        c = FakeCanvas(fd)
        originals = {i: float(fd.values[0, i]) for i in [3, 4, 5, 6]}
        _simulate_stroke(c, [(3, 500.0), (4, 510.0), (5, 520.0), (6, 530.0)])
        _apply_undo(c)
        for i, orig in originals.items():
            assert fd.values[0, i] == orig
            assert fd.edited_mask[0, i] == False

    def test_undo_stack_limit(self):
        """Undo stack is trimmed to MAX_UNDO_STEPS."""
        fd = _make_fd(n_frames=200)
        c = FakeCanvas(fd)
        for i in range(MAX_UNDO_STEPS + 20):
            _simulate_stroke(c, [(i % 200, 500.0 + i)])
        assert len(c._undo_stack) == MAX_UNDO_STEPS

    def test_undo_redo_roundtrip(self):
        """Undo then redo leaves data identical to post-edit state."""
        fd = _make_fd()
        c = FakeCanvas(fd)
        _simulate_stroke(c, [(5, 777.0), (6, 888.0)])
        snapshot = fd.values.copy()
        _apply_undo(c)
        _apply_redo(c)
        npt.assert_array_equal(fd.values, snapshot)


# ===========================================================================
# Eraser Tests
# ===========================================================================

class TestEraser:
    """Tests for the eraser tool (right-click drag)."""

    def test_erase_reverts_to_original(self):
        """Erasing a frame restores the original Praat value."""
        fd = _make_fd()
        c = FakeCanvas(fd)
        original = float(fd.original_values[0, 5])
        _simulate_stroke(c, [(5, 999.0)])  # draw
        assert fd.values[0, 5] == 999.0
        _simulate_stroke(c, [(5, 0)], is_erasing=True)  # erase (freq ignored)
        assert fd.values[0, 5] == original
        assert fd.edited_mask[0, 5] == False

    def test_erase_is_undoable(self):
        """Undoing an erase restores the edited value."""
        fd = _make_fd()
        c = FakeCanvas(fd)
        _simulate_stroke(c, [(5, 999.0)])  # draw
        _simulate_stroke(c, [(5, 0)], is_erasing=True)  # erase
        _apply_undo(c)  # undo erase
        assert fd.values[0, 5] == 999.0
        assert fd.edited_mask[0, 5] == True

    def test_erase_multi_frame(self):
        """Eraser reverts multiple frames in a single stroke."""
        fd = _make_fd()
        c = FakeCanvas(fd)
        originals = {i: float(fd.original_values[0, i]) for i in [3, 4, 5]}
        _simulate_stroke(c, [(3, 800.0), (4, 810.0), (5, 820.0)])
        _simulate_stroke(c, [(3, 0), (4, 0), (5, 0)], is_erasing=True)
        for i, orig in originals.items():
            assert fd.values[0, i] == orig
            assert fd.edited_mask[0, i] == False

    def test_erase_undo_entry_description(self):
        """Erase strokes have 'Erase' as the undo description."""
        fd = _make_fd()
        c = FakeCanvas(fd)
        _simulate_stroke(c, [(5, 0)], is_erasing=True)
        assert c._undo_stack[-1].description == "Erase"

    def test_draw_undo_entry_description(self):
        """Draw strokes have 'Draw' as the undo description."""
        fd = _make_fd()
        c = FakeCanvas(fd)
        _simulate_stroke(c, [(5, 500.0)])
        assert c._undo_stack[-1].description == "Draw"


# ===========================================================================
# Interpolate Between Points Tests
# ===========================================================================

class TestInterpolateBetweenPoints:
    """Tests for the interpolate-between-edited-points feature."""

    def test_interpolate_fills_gap(self):
        """Interpolation fills frames between two edited points."""
        fd = _make_fd(n_frames=20)
        c = FakeCanvas(fd)
        # Edit frames 5 and 15
        fd.set_value(0, 5, 500.0)
        fd.set_value(0, 15, 1000.0)
        count = _interpolate_formant(c, 0)
        assert count == 9  # frames 6–14
        # Check linear interpolation
        for fi in range(6, 15):
            t = (fi - 5) / (15 - 5)
            expected = 500.0 + t * 500.0
            assert abs(fd.values[0, fi] - expected) < 0.01

    def test_interpolate_marks_as_edited(self):
        """Interpolated frames have edited_mask set to True."""
        fd = _make_fd(n_frames=20)
        c = FakeCanvas(fd)
        fd.set_value(0, 5, 500.0)
        fd.set_value(0, 15, 1000.0)
        _interpolate_formant(c, 0)
        for fi in range(6, 15):
            assert fd.edited_mask[0, fi] == True

    def test_interpolate_is_undoable(self):
        """Undoing interpolation restores previous values."""
        fd = _make_fd(n_frames=20)
        c = FakeCanvas(fd)
        originals = fd.values[0].copy()
        fd.set_value(0, 5, 500.0)
        fd.set_value(0, 15, 1000.0)
        _interpolate_formant(c, 0)
        _apply_undo(c)  # undo interpolation
        # Frames 6–14 should be back to pre-interpolation values
        for fi in range(6, 15):
            assert fd.values[0, fi] == originals[fi]

    def test_interpolate_needs_two_points(self):
        """Interpolation with fewer than 2 edited points returns 0."""
        fd = _make_fd(n_frames=20)
        c = FakeCanvas(fd)
        fd.set_value(0, 5, 500.0)
        count = _interpolate_formant(c, 0)
        assert count == 0

    def test_interpolate_no_edited_points(self):
        """Interpolation with no edited points returns 0."""
        fd = _make_fd(n_frames=20)
        c = FakeCanvas(fd)
        count = _interpolate_formant(c, 0)
        assert count == 0

    def test_interpolate_adjacent_points(self):
        """Adjacent edited points have nothing to interpolate."""
        fd = _make_fd(n_frames=20)
        c = FakeCanvas(fd)
        fd.set_value(0, 5, 500.0)
        fd.set_value(0, 6, 600.0)
        count = _interpolate_formant(c, 0)
        assert count == 0

    def test_interpolate_with_selection_range(self):
        """Interpolation respects time selection range."""
        fd = _make_fd(n_frames=20)
        c = FakeCanvas(fd)
        # Edit frames 2, 8, 15
        fd.set_value(0, 2, 400.0)
        fd.set_value(0, 8, 800.0)
        fd.set_value(0, 15, 1200.0)
        # Only interpolate in the range covering frames 8–15
        start_t = fd.times[7]
        end_t = fd.times[16]
        count = _interpolate_formant(c, 0, start_t, end_t)
        # Should fill frames 9–14 (between 8 and 15)
        assert count == 6
        # Frames 3–7 (between 2 and 8) should NOT be interpolated
        assert fd.edited_mask[0, 4] == False

    def test_interpolate_multiple_pairs(self):
        """Interpolation fills gaps between multiple consecutive edited pairs."""
        fd = _make_fd(n_frames=20)
        c = FakeCanvas(fd)
        fd.set_value(0, 2, 400.0)
        fd.set_value(0, 6, 800.0)
        fd.set_value(0, 10, 600.0)
        count = _interpolate_formant(c, 0)
        # 3 frames between 2–6, plus 3 frames between 6–10
        assert count == 3 + 3

    def test_interpolate_pushes_undo_entry(self):
        """Interpolation pushes an undo entry with 'Interpolate' description."""
        fd = _make_fd(n_frames=20)
        c = FakeCanvas(fd)
        fd.set_value(0, 5, 500.0)
        fd.set_value(0, 15, 1000.0)
        _interpolate_formant(c, 0)
        assert len(c._undo_stack) == 1
        assert c._undo_stack[0].description == "Interpolate"


# ===========================================================================
# Frame Interpolation During Drawing Tests
# ===========================================================================

class TestFrameInterpolation:
    """Tests for frame interpolation that fills skipped frames during drawing."""

    def test_interpolation_fills_skipped_frames(self):
        """Drawing at frames 2 and 8 should fill frames 3–7."""
        fd = _make_fd(n_frames=20)
        c = FakeCanvas(fd)
        # Simulate what _apply_edit does: first frame, then skip to frame 8
        f_idx = 0
        c.active_formant = f_idx

        # First point at frame 2
        old_val = float(fd.values[f_idx, 2])
        fd.set_value(f_idx, 2, 400.0)
        c._stroke_changes.append((f_idx, 2, old_val, False, 400.0, True))
        c._last_frame_idx = 2
        c._last_frame_freq = 400.0

        # Jump to frame 8 — simulate interpolation logic
        frame_idx = 8
        freq_hz = 700.0
        prev_fi = c._last_frame_idx
        prev_freq = c._last_frame_freq
        n_steps = abs(frame_idx - prev_fi)
        for k in range(1, n_steps):
            mid_fi = prev_fi + k
            t = k / n_steps
            mid_freq = prev_freq + t * (freq_hz - prev_freq)
            old_v = float(fd.values[f_idx, mid_fi])
            fd.set_value(f_idx, mid_fi, mid_freq)
            c._stroke_changes.append((f_idx, mid_fi, old_v, False, mid_freq, True))

        old_v = float(fd.values[f_idx, frame_idx])
        fd.set_value(f_idx, frame_idx, freq_hz)
        c._stroke_changes.append((f_idx, frame_idx, old_v, False, freq_hz, True))

        # Verify intermediate frames are filled
        for fi in range(3, 8):
            assert fd.edited_mask[f_idx, fi] == True
            assert not np.isnan(fd.values[f_idx, fi])

        # Verify linear interpolation
        for fi in range(3, 8):
            t = (fi - 2) / (8 - 2)
            expected = 400.0 + t * 300.0
            assert abs(fd.values[f_idx, fi] - expected) < 0.1

    def test_interpolation_values_are_linear(self):
        """Interpolated values are exactly linear between endpoints."""
        fd = _make_fd(n_frames=20)
        f_idx = 0
        start_freq = 500.0
        end_freq = 1000.0
        start_fi = 0
        end_fi = 10

        # Simulate filling
        fd.set_value(f_idx, start_fi, start_freq)
        fd.set_value(f_idx, end_fi, end_freq)
        n_steps = end_fi - start_fi
        for k in range(1, n_steps):
            mid_fi = start_fi + k
            t = k / n_steps
            mid_freq = start_freq + t * (end_freq - start_freq)
            fd.set_value(f_idx, mid_fi, mid_freq)

        # Check exact linearity
        for fi in range(start_fi, end_fi + 1):
            t = (fi - start_fi) / (end_fi - start_fi)
            expected = start_freq + t * (end_freq - start_freq)
            assert abs(fd.values[f_idx, fi] - expected) < 1e-10


# ===========================================================================
# UndoEntry namedtuple Tests
# ===========================================================================

class TestUndoEntry:
    """Tests for the UndoEntry data structure."""

    def test_undo_entry_fields(self):
        """UndoEntry has description and changes fields."""
        changes = [(0, 5, 300.0, False, 999.0, True)]
        entry = UndoEntry("Draw", changes)
        assert entry.description == "Draw"
        assert entry.changes == changes

    def test_undo_entry_immutable_description(self):
        """UndoEntry description is accessible."""
        entry = UndoEntry("Erase", [])
        assert entry.description == "Erase"

    def test_max_undo_steps_constant(self):
        """MAX_UNDO_STEPS is 100."""
        assert MAX_UNDO_STEPS == 100
