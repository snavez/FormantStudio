"""
Tests for keyboard navigation of the annotation.

Ctrl+Left/Right step along the active tier, Ctrl+Up/Down move between tiers
carrying the selection to the segment covering the same moment. The bare
arrows belong to the label editor's text.

The binding is Ctrl rather than the bare arrows because selecting a segment
focuses the label editor, so a bare-arrow binding is unreachable in the one
workflow it exists for: select, relabel, move on.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QApplication

from formant_editor import (
    Interval,
    LabelEdit,
    Point,
    SpectrogramCanvas,
    TextGrid,
    Tier,
)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


@pytest.fixture
def stacked_textgrid():
    """Three tiers over the same second: words over phones, plus targets.

    ``words`` is coarser than ``phones``, so a phone's midpoint lands inside
    exactly one word -- which is what stepping between tiers has to follow.
    """
    return TextGrid(0, 1.0, [
        Tier("words", "IntervalTier", 0, 1.0,
             intervals=[Interval(0, 0.5, "hello"),
                        Interval(0.5, 1.0, "world")]),
        Tier("phones", "IntervalTier", 0, 1.0,
             intervals=[Interval(0, 0.3, "h"), Interval(0.3, 0.5, "ow"),
                        Interval(0.5, 0.7, "w"), Interval(0.7, 1.0, "ld")]),
        Tier("targets", "TextTier", 0, 1.0,
             points=[Point(0.15, "T1"), Point(0.85, "T2")]),
    ])


@pytest.fixture
def canvas(qapp, stacked_textgrid):
    c = SpectrogramCanvas()
    c.textgrid_data = stacked_textgrid
    c.total_duration = 1.0
    c.view_start = 0.0
    c.view_end = 1.0
    c._active_tier = 1          # phones
    yield c


def _key(key, ctrl=True):
    mods = (Qt.KeyboardModifier.ControlModifier if ctrl
            else Qt.KeyboardModifier.NoModifier)
    return QKeyEvent(QKeyEvent.Type.KeyPress, key, mods)


def _selected_label(canvas):
    sel = canvas._selected_interval
    if sel is None:
        return None
    tier = canvas.textgrid_data.tiers[sel[0]]
    if tier.tier_class == "IntervalTier":
        return tier.intervals[sel[1]].text
    return tier.points[sel[1]].mark


class TestStepSelection:
    """Ctrl+Left/Right walk the active tier."""

    def test_stepping_forward_takes_the_next_interval(self, canvas):
        canvas._select_interval(1, 0.15)          # "h"
        iv = canvas.step_selection(1)
        assert iv.text == "ow"
        assert _selected_label(canvas) == "ow"

    def test_stepping_back_takes_the_previous_one(self, canvas):
        canvas._select_interval(1, 0.6)           # "w"
        assert canvas.step_selection(-1).text == "ow"

    def test_it_stops_at_the_end_rather_than_wrapping(self, canvas):
        canvas._select_interval(1, 0.85)          # "ld", the last phone
        assert canvas.step_selection(1).text == "ld"

    def test_it_stops_at_the_start(self, canvas):
        canvas._select_interval(1, 0.15)          # "h", the first phone
        assert canvas.step_selection(-1).text == "h"

    def test_with_nothing_selected_it_starts_from_the_view(self, canvas):
        """The keyboard picks up where the eye already is."""
        canvas.set_view(0.5, 0.7)                 # centred on "w"
        assert canvas.step_selection(1).text == "ld"

    def test_a_point_tier_has_no_intervals_to_step(self, canvas):
        canvas._active_tier = 2                   # targets
        assert canvas.step_selection(1) is None


class TestStepActiveTier:
    """Ctrl+Up/Down move between tiers, carrying the selection."""

    def test_going_up_selects_the_word_over_the_phone(self, canvas):
        canvas._select_interval(1, 0.35)          # "ow", inside "hello"
        assert canvas.step_active_tier(-1) == "words"
        assert _selected_label(canvas) == "hello"

    def test_going_down_returns_to_a_phone_at_that_moment(self, canvas):
        canvas._active_tier = 0
        canvas._select_interval(0, 0.75)          # "world"
        assert canvas.step_active_tier(1) == "phones"
        # "world" spans 0.5-1.0, midpoint 0.75, which is inside "ld"
        assert _selected_label(canvas) == "ld"

    def test_the_selection_survives_a_round_trip(self, canvas):
        canvas._select_interval(1, 0.85)          # "ld"
        canvas.step_active_tier(-1)
        assert _selected_label(canvas) == "world"
        canvas.step_active_tier(1)
        assert _selected_label(canvas) == "ld"

    def test_with_nothing_selected_the_view_centre_anchors_it(self, canvas):
        canvas.set_view(0.0, 0.4)                 # centre 0.2, inside "hello"
        assert canvas.step_active_tier(-1) == "words"
        assert _selected_label(canvas) == "hello"

    def test_it_stops_at_the_top_tier(self, canvas):
        canvas._active_tier = 0
        assert canvas.step_active_tier(-1) == "words"

    def test_it_stops_at_the_bottom_tier(self, canvas):
        canvas._active_tier = 2
        assert canvas.step_active_tier(1) == "targets"

    def test_hidden_tiers_are_skipped(self, canvas):
        """Stepping follows what is on screen, not what is in the file."""
        canvas.hidden_tiers = {0}                 # hide words
        canvas._select_interval(1, 0.35)
        assert canvas.step_active_tier(-1) == "phones"

    def test_a_tier_with_nothing_there_leaves_no_selection(self, canvas):
        """The targets tier has no point near 0.35, so nothing is selected."""
        canvas._select_interval(1, 0.35)          # "ow", midpoint 0.4
        assert canvas.step_active_tier(1) == "targets"
        assert canvas._selected_interval is None
        assert canvas._selection_start is None

    def test_a_point_under_the_selection_is_picked_up(self, canvas):
        """T2 sits at 0.85, the midpoint of "ld" (0.7-1.0)."""
        canvas._select_interval(1, 0.85)          # "ld"
        assert canvas.step_active_tier(1) == "targets"
        assert _selected_label(canvas) == "T2"

    def test_no_visible_tiers_is_no_move(self, canvas):
        canvas.hidden_tiers = {0, 1, 2}
        assert canvas.step_active_tier(1) is None


class TestFocusLabelForSelection:
    """Where the caret lands, so the next keystroke does the right thing."""

    def test_an_interval_label_is_selected_whole(self, canvas, qapp):
        edit = LabelEdit()
        canvas._label_edit = edit
        canvas._select_interval(1, 0.35)
        canvas.focus_label_for_selection(1)
        assert edit.text() == "ow"
        assert edit.selectedText() == "ow"

    def test_a_point_mark_puts_the_caret_at_the_end(self, canvas, qapp):
        """Delete must remove the point, not a character of its mark."""
        edit = LabelEdit()
        canvas._label_edit = edit
        canvas._select_interval(2, 0.15)
        canvas.focus_label_for_selection(2)
        assert edit.text() == "T1"
        assert edit.selectedText() == ""
        assert edit.cursorPosition() == len("T1")

    def test_no_editor_is_not_an_error(self, canvas):
        canvas._label_edit = None
        canvas.focus_label_for_selection(1)


class TestLabelEditKeyPassthrough:
    """The editor holds focus while relabelling, so it must pass Ctrl+arrows."""

    @pytest.mark.parametrize("key", [Qt.Key.Key_Left, Qt.Key.Key_Right,
                                     Qt.Key.Key_Up, Qt.Key.Key_Down])
    def test_ctrl_arrows_travel_up_to_the_window(self, qapp, key):
        edit = LabelEdit()
        edit.setEnabled(True)
        edit.setText("ow")
        ev = _key(key, ctrl=True)
        edit.keyPressEvent(ev)
        assert not ev.isAccepted(), "Ctrl+arrow must propagate, not be eaten"

    @pytest.mark.parametrize("key", [Qt.Key.Key_Left, Qt.Key.Key_Right])
    def test_bare_arrows_stay_with_the_text(self, qapp, key):
        """Retired as a navigation binding: they belong to the editor."""
        edit = LabelEdit()
        edit.setEnabled(True)
        edit.setText("ow")
        edit.setCursorPosition(1)
        ev = _key(key, ctrl=False)
        edit.keyPressEvent(ev)
        assert ev.isAccepted()

    def test_typing_still_reaches_the_text(self, qapp):
        edit = LabelEdit()
        edit.setEnabled(True)
        ev = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_A,
                       Qt.KeyboardModifier.NoModifier, "a")
        edit.keyPressEvent(ev)
        assert edit.text() == "a"


class TestMainWindowBinding:
    """The wiring, exercised the way a keystroke actually travels.

    Events are sent to the label editor, not to the window, so these cover the
    whole chain: LabelEdit declines the Ctrl+arrow, Qt propagates it up the
    parent widgets, and MainWindow acts on it. Calling the window's handler
    directly would skip the very step that used to swallow the key.
    """

    @pytest.fixture
    def window(self, qapp, stacked_textgrid):
        from formant_editor import MainWindow
        mw = MainWindow()
        c = mw.canvas
        c.textgrid_data = stacked_textgrid
        c.total_duration = 1.0
        c.view_start, c.view_end = 0.0, 1.0
        c._active_tier = 1                        # phones
        c._select_interval(1, 0.15)               # "h"
        mw.label_edit.setFocus()
        yield mw
        mw.close()

    def _press(self, window, key, ctrl=True):
        ev = _key(key, ctrl=ctrl)
        QApplication.sendEvent(window.label_edit, ev)
        return ev

    def test_ctrl_right_steps_while_the_editor_holds_focus(self, window):
        """The case the old bare-arrow binding could never reach."""
        ev = self._press(window, Qt.Key.Key_Right)
        assert ev.isAccepted()
        assert _selected_label(window.canvas) == "ow"

    def test_ctrl_left_steps_back(self, window):
        window.canvas._select_interval(1, 0.6)    # "w"
        self._press(window, Qt.Key.Key_Left)
        assert _selected_label(window.canvas) == "ow"

    def test_ctrl_up_moves_tier_and_carries_the_selection(self, window):
        ev = self._press(window, Qt.Key.Key_Up)
        assert ev.isAccepted()
        assert window.canvas._active_tier == 0
        assert _selected_label(window.canvas) == "hello"

    def test_the_move_leaves_the_label_ready_to_retype(self, window):
        """Navigate, type, navigate -- without reaching for the mouse.

        Asserted on the selected text rather than ``hasFocus``: whether the
        OS actually grants focus depends on the window being shown and
        active, which a test run cannot rely on, but the whole-label
        selection only happens on the path that also calls ``setFocus``.
        """
        self._press(window, Qt.Key.Key_Right)
        assert window.label_edit.text() == "ow"
        assert window.label_edit.selectedText() == "ow"

    def test_bare_arrows_stay_with_the_text(self, window):
        """Retired as navigation: the editor keeps them."""
        self._press(window, Qt.Key.Key_Right, ctrl=False)
        assert _selected_label(window.canvas) == "h"

    def test_no_textgrid_means_nothing_to_navigate(self, window):
        window.canvas.textgrid_data = None
        ev = self._press(window, Qt.Key.Key_Right)
        assert not ev.isAccepted()


class TestNavigationDoesNotDirtyTheTextGrid:
    """Looking through a file is not editing it.

    The label editor is refilled every time the selection moves, and that
    programmatic ``setText`` used to run the same handler as typing -- so
    navigating marked the TextGrid unsaved and the app asked to save a file
    nothing had been done to. Navigation got much cheaper with Ctrl+arrows,
    which made a pre-existing wart constant.
    """

    @pytest.fixture
    def window(self, qapp, stacked_textgrid):
        from formant_editor import MainWindow
        mw = MainWindow()
        c = mw.canvas
        c.textgrid_data = stacked_textgrid
        c.total_duration = 1.0
        c.view_start, c.view_end = 0.0, 1.0
        c._active_tier = 1
        yield mw

    def test_selecting_is_not_an_edit(self, window):
        window.canvas._select_interval(1, 0.15)
        assert window._textgrid_dirty is False

    def test_stepping_along_a_tier_is_not_an_edit(self, window):
        window.canvas._select_interval(1, 0.15)
        window.canvas.step_selection(1)
        assert window._textgrid_dirty is False

    def test_stepping_between_tiers_is_not_an_edit(self, window):
        window.canvas._select_interval(1, 0.35)
        window.canvas.step_active_tier(-1)
        assert window._textgrid_dirty is False

    def test_typing_is_an_edit_and_writes_through(self, window, qapp):
        """The other half: real keystrokes must still reach the data."""
        from PyQt6.QtTest import QTest
        window.canvas._select_interval(1, 0.15)     # "h"
        window.label_edit.setFocus()
        window.label_edit.clear()
        QTest.keyClicks(window.label_edit, "tS")
        assert window._textgrid_dirty is True
        assert window.canvas.textgrid_data.tiers[1].intervals[0].text == "tS"


# A canvas needs a spectrogram before render() will apply a view range --
# without one it draws empty axes and returns early. Synthesising the sound
# in memory keeps these tests independent of any wav file on disk.
def _sounded_canvas(seconds=6.0, seg=0.1):
    import numpy as np
    import parselmouth
    sr = 16000
    snd = parselmouth.Sound(
        np.random.default_rng(0).standard_normal(int(sr * seconds)) * 0.05,
        sampling_frequency=sr)
    c = SpectrogramCanvas()
    c.sound = snd
    c.total_duration = snd.duration
    c.view_start, c.view_end = 0.0, snd.duration
    c._compute_spectrogram()
    c._setup_axes()
    n = int(snd.duration / seg)
    c.textgrid_data = TextGrid(0, snd.duration, [
        Tier("phones", "IntervalTier", 0, snd.duration,
             intervals=[Interval(i * seg, (i + 1) * seg, f"s{i}")
                        for i in range(n)])])
    c._active_tier = 0
    c.set_view(0.0, 1.0)                      # 1 s window over 0.1 s segments
    c.render()
    c._select_interval(0, 0.05)               # s0
    return c


class TestSteppingKeepsTheSelectionOnScreen:
    """A step that leaves the window has to move the window.

    The view range only reaches the plot through ``render()``, so stepping
    used to scroll the model and redraw nothing: the selection walked off the
    edge and stayed there, which is invisible in the data and obvious on
    screen.
    """

    @pytest.fixture
    def canvas(self, qapp):
        return _sounded_canvas()

    def _plot_range(self, canvas):
        return canvas._spec_plot.getViewBox().viewRange()[0]

    def _selected(self, canvas):
        idx = canvas._selected_interval[1]
        return canvas.textgrid_data.tiers[0].intervals[idx]

    def _centred(self, canvas):
        iv = self._selected(canvas)
        lo, hi = self._plot_range(canvas)
        return (lo + hi) / 2 == pytest.approx((iv.xmin + iv.xmax) / 2, abs=0.02)

    def test_every_step_re_centres_the_window(self, canvas):
        """Not only the ones that would leave it: context matters at the edge.

        Checked from half a window in, past the point where centring is
        clamped by the start of the recording.
        """
        for _ in range(6):                # clear of the start-of-file clamp
            canvas.step_selection(1)
        for n in range(1, 15):
            canvas.step_selection(1)
            canvas._render_timer.stop()
            canvas.render()               # flush the debounced repaint
            assert self._centred(canvas), f"step {n} left the selection off-centre"

    def test_the_first_segment_sits_off_centre_at_the_start(self, canvas):
        """Centring stops at the recording's edge rather than showing nothing."""
        for _ in range(20):
            canvas.step_selection(-1)     # back past the beginning
        canvas._render_timer.stop()
        canvas.render()
        lo, hi = self._plot_range(canvas)
        assert lo == pytest.approx(0.0, abs=0.02)
        assert not self._centred(canvas)

    def test_forty_steps_forward_stay_visible(self, canvas):
        for n in range(40):
            canvas.step_selection(1)
            canvas._render_timer.stop()
            canvas.render()
            iv = self._selected(canvas)
            lo, hi = self._plot_range(canvas)
            assert lo <= iv.xmin and iv.xmax <= hi, f"step {n} went off screen"

    def test_forty_steps_back_stay_visible(self, canvas):
        for _ in range(40):
            canvas.step_selection(1)
        for n in range(40):
            canvas.step_selection(-1)
            canvas._render_timer.stop()
            canvas.render()
            iv = self._selected(canvas)
            lo, hi = self._plot_range(canvas)
            assert lo <= iv.xmin and iv.xmax <= hi, f"back step {n} off screen"

    def test_the_window_keeps_its_width(self, canvas):
        """Following the selection scrolls; it must not zoom."""
        lo, hi = self._plot_range(canvas)
        width = hi - lo
        for _ in range(25):
            canvas.step_selection(1)
        canvas._render_timer.stop()
        canvas.render()
        lo, hi = self._plot_range(canvas)
        assert hi - lo == pytest.approx(width, abs=1e-6)
