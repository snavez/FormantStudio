"""
Integration tests for SpectrogramCanvas (pyqtgraph backend).

These tests create real widget instances and exercise the rendering pipeline
to catch segfaults and runtime errors that data-only tests miss.
They require a running QApplication but use offscreen rendering where possible.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from formant_editor import (
    SpectrogramCanvas, MainWindow, FormantData, TextGrid, Tier, Interval,
    Point, LabelEdit, extract_formants_from_praat, FORMANT_COLORS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def qapp():
    """Create a QApplication once for the whole test session."""
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


@pytest.fixture
def canvas(qapp):
    """Create a fresh SpectrogramCanvas."""
    c = SpectrogramCanvas()
    yield c


@pytest.fixture
def wav_path():
    """Path to a test WAV file."""
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test.wav")
    if not os.path.exists(path):
        pytest.skip("test.wav not found")
    return path


@pytest.fixture
def canvas_with_sound(canvas, wav_path):
    """Canvas with a loaded sound file."""
    canvas.load_sound(wav_path)
    return canvas


@pytest.fixture
def simple_textgrid():
    """Create a simple 2-tier TextGrid for testing."""
    return TextGrid(0, 1.0, [
        Tier("words", "IntervalTier", 0, 1.0,
             intervals=[Interval(0, 0.5, "hello"), Interval(0.5, 1.0, "world")]),
        Tier("phones", "IntervalTier", 0, 1.0,
             intervals=[Interval(0, 0.3, "h"), Interval(0.3, 0.5, "ow"),
                        Interval(0.5, 0.7, "w"), Interval(0.7, 1.0, "ld")]),
    ])


@pytest.fixture
def point_textgrid():
    """Create a TextGrid with a TextTier for testing."""
    return TextGrid(0, 1.0, [
        Tier("words", "IntervalTier", 0, 1.0,
             intervals=[Interval(0, 0.5, "a"), Interval(0.5, 1.0, "b")]),
        Tier("events", "TextTier", 0, 1.0,
             points=[Point(0.25, "click"), Point(0.75, "beep")]),
    ])


# ---------------------------------------------------------------------------
# Canvas creation and empty state
# ---------------------------------------------------------------------------

class TestCanvasCreation:
    def test_canvas_creates_without_crash(self, canvas):
        assert canvas is not None
        assert canvas._glw is not None
        assert canvas._spec_plot is not None

    def test_canvas_initial_state(self, canvas):
        assert canvas.sound is None
        assert canvas.spectrogram_data is None
        assert canvas.formant_data is None
        assert canvas.textgrid_data is None
        assert canvas.view_start == 0.0
        assert canvas.view_end == 0.0
        assert canvas.edit_mode is False

    def test_empty_render_no_crash(self, canvas):
        """render() on empty canvas should not crash."""
        canvas.render()


# ---------------------------------------------------------------------------
# Sound loading
# ---------------------------------------------------------------------------

class TestSoundLoading:
    def test_load_sound(self, canvas_with_sound):
        c = canvas_with_sound
        assert c.sound is not None
        assert c.total_duration > 0
        assert c.spectrogram_data is not None
        assert c.spec_times is not None
        assert c.spec_freqs is not None
        assert c._spec_plot is not None
        assert c._wave_plot is not None

    def test_render_after_load(self, canvas_with_sound):
        """render() after loading sound should not crash."""
        canvas_with_sound.render()

    def test_spec_image_exists(self, canvas_with_sound):
        assert canvas_with_sound._spec_image is not None

    def test_view_range_set(self, canvas_with_sound):
        c = canvas_with_sound
        assert c.view_start == 0.0
        assert c.view_end == c.total_duration
        assert c.view_width > 0


# ---------------------------------------------------------------------------
# Rendering with formants
# ---------------------------------------------------------------------------

class TestFormantRendering:
    def test_render_with_formants(self, canvas_with_sound, wav_path):
        c = canvas_with_sound
        import parselmouth
        sound = parselmouth.Sound(wav_path)
        c.formant_data = extract_formants_from_praat(sound)
        c.show_formants = True
        c.render()
        # Should have created scatter items
        assert len(c._formant_scatters) > 0

    def test_render_formants_hidden(self, canvas_with_sound, wav_path):
        c = canvas_with_sound
        import parselmouth
        sound = parselmouth.Sound(wav_path)
        c.formant_data = extract_formants_from_praat(sound)
        c.show_formants = False
        c.render()
        assert len(c._formant_scatters) == 0

    def test_render_edit_mode(self, canvas_with_sound, wav_path):
        c = canvas_with_sound
        import parselmouth
        sound = parselmouth.Sound(wav_path)
        c.formant_data = extract_formants_from_praat(sound)
        c.edit_mode = True
        c.active_formant = 0
        c.render()


# ---------------------------------------------------------------------------
# Rendering with TextGrid
# ---------------------------------------------------------------------------

class TestTextGridRendering:
    def test_render_with_interval_tiers(self, canvas_with_sound, simple_textgrid):
        c = canvas_with_sound
        c.textgrid_data = simple_textgrid
        c._setup_axes()
        c.render()
        assert len(c._tier_plots) == 2
        assert len(c.tier_axes) == 2
        # Boundary lines are batched into persistent per-plot items
        assert len(c._batched_lines) > 0

    def test_render_with_point_tier(self, canvas_with_sound, point_textgrid):
        c = canvas_with_sound
        c.textgrid_data = point_textgrid
        c._setup_axes()
        c.render()
        assert len(c._tier_plots) == 2

    def test_setup_axes_no_tiers(self, canvas_with_sound):
        c = canvas_with_sound
        c.textgrid_data = None
        c._setup_axes()
        c.render()
        assert len(c._tier_plots) == 0

    def test_active_tier_highlighting(self, canvas_with_sound, simple_textgrid):
        c = canvas_with_sound
        c.textgrid_data = simple_textgrid
        c._setup_axes()
        c._active_tier = 0
        c.render()
        c._active_tier = 1
        c.render()

    def test_hide_tier(self, canvas_with_sound, simple_textgrid):
        """Hiding a tier should exclude it from rendering."""
        c = canvas_with_sound
        c.textgrid_data = simple_textgrid
        c._setup_axes()
        assert len(c._tier_plots) == 2
        # Hide tier 0
        c.hidden_tiers = {0}
        c._setup_axes()
        c.render()
        assert len(c._tier_plots) == 1
        assert c._tier_plot_indices == [1]

    def test_hide_all_tiers(self, canvas_with_sound, simple_textgrid):
        """Hiding all tiers should show spectrogram x-axis."""
        c = canvas_with_sound
        c.textgrid_data = simple_textgrid
        c.hidden_tiers = {0, 1}
        c._setup_axes()
        c.render()
        assert len(c._tier_plots) == 0

    def test_tier_index_mapping(self, canvas_with_sound, simple_textgrid):
        """_tier_index_for_plot should return actual tier index, not plot index."""
        c = canvas_with_sound
        c.textgrid_data = simple_textgrid
        c.hidden_tiers = {0}  # hide first tier
        c._setup_axes()
        c.render()
        # The one visible plot should map to tier index 1
        assert c._tier_index_for_plot(c._tier_plots[0]) == 1
        assert c._plot_for_tier(1) is c._tier_plots[0]
        assert c._plot_for_tier(0) is None  # hidden


# ---------------------------------------------------------------------------
# Zoom and view
# ---------------------------------------------------------------------------

class TestZoomAndView:
    def test_zoom_in(self, canvas_with_sound):
        c = canvas_with_sound
        old_width = c.view_width
        c.zoom(0.5)
        assert c.view_width < old_width
        c.render()

    def test_zoom_out(self, canvas_with_sound):
        c = canvas_with_sound
        c.zoom(0.5)  # zoom in first
        old_width = c.view_width
        c.zoom(2.0)
        assert c.view_width > old_width
        c.render()

    def test_set_view(self, canvas_with_sound):
        c = canvas_with_sound
        mid = c.total_duration / 2
        c.set_view(mid - 0.5, mid + 0.5)
        c.render()
        assert abs(c.view_start - (mid - 0.5)) < 0.01

    def test_render_zoomed_past_max(self, canvas_with_sound):
        """When zoomed out past MAX_SPECTROGRAM_VIEW, spectrogram hides."""
        c = canvas_with_sound
        c.set_view(0, c.total_duration)
        c.render()
        # If total_duration > 10s, spec image should be hidden
        if c.total_duration > 10.0:
            assert not c._spec_image.isVisible()


# ---------------------------------------------------------------------------
# Selection overlay
# ---------------------------------------------------------------------------

class TestSelectionOverlay:
    def test_time_selection(self, canvas_with_sound):
        c = canvas_with_sound
        c.render()
        c._selection_start = 0.1
        c._selection_end = 0.3
        c._draw_selection_overlay()
        assert len(c._overlay_items) > 0

    def test_boundary_selection(self, canvas_with_sound, simple_textgrid):
        c = canvas_with_sound
        c.textgrid_data = simple_textgrid
        c._setup_axes()
        c.render()
        c._selected_boundary = (0, 0.5)
        c._draw_selection_overlay()
        assert len(c._overlay_items) > 0

    def test_interval_selection(self, canvas_with_sound, simple_textgrid):
        c = canvas_with_sound
        c.textgrid_data = simple_textgrid
        c._setup_axes()
        c.render()
        c._selected_interval = (0, 0)
        c._draw_selection_overlay()
        assert len(c._overlay_items) > 0

    def test_clear_selection(self, canvas_with_sound):
        c = canvas_with_sound
        c._selection_start = 0.1
        c._selection_end = 0.3
        c._clear_selection()
        assert c._selection_start is None
        assert c._selection_end is None


# ---------------------------------------------------------------------------
# Crosshair
# ---------------------------------------------------------------------------

class TestCrosshair:
    def test_crosshair_update(self, canvas_with_sound):
        c = canvas_with_sound
        c.render()
        c._update_crosshair(0.5, 1000.0, on_spectrogram=True)
        assert c._crosshair_visible
        assert c._crosshair_v.isVisible()
        assert c._crosshair_h.isVisible()

    def test_crosshair_hide(self, canvas_with_sound):
        c = canvas_with_sound
        c.render()
        c._update_crosshair(0.5, 1000.0, on_spectrogram=True)
        c._hide_crosshair()
        assert not c._crosshair_visible

    def test_crosshair_wave_only(self, canvas_with_sound):
        c = canvas_with_sound
        c.render()
        c._update_crosshair(0.5, 0.1, on_spectrogram=False)
        assert c._crosshair_visible
        assert not c._crosshair_h.isVisible()


# ---------------------------------------------------------------------------
# Playback
# ---------------------------------------------------------------------------

class TestPlayback:
    def test_play_creates_cursors(self, canvas_with_sound):
        c = canvas_with_sound
        c.render()
        c.play_audio(0.0, 0.1)
        assert c._playback_playing
        assert len(c._playback_cursors) > 0
        c.stop_playback()
        assert not c._playback_playing
        assert len(c._playback_cursors) == 0

    def test_stop_when_not_playing(self, canvas_with_sound):
        c = canvas_with_sound
        c.render()
        c.stop_playback()  # should not crash


# ---------------------------------------------------------------------------
# Click-vs-drag threshold on boundary/point markers
# ---------------------------------------------------------------------------

class _FakeMouseEvent:
    """Minimal stand-in for QMouseEvent (position/modifiers/button)."""

    def __init__(self, x, y):
        from PyQt6.QtCore import QPointF
        self._pos = QPointF(x, y)

    def position(self):
        return self._pos

    def modifiers(self):
        from PyQt6.QtCore import Qt
        return Qt.KeyboardModifier.NoModifier

    def button(self):
        from PyQt6.QtCore import Qt
        return Qt.MouseButton.LeftButton


class TestClickVsDrag:
    def test_plain_click_selects_without_moving(
            self, canvas_with_sound, simple_textgrid):
        c = canvas_with_sound
        c.textgrid_data = simple_textgrid
        c._setup_axes()
        c.render()
        edits = []
        c._on_textgrid_edited = lambda: edits.append(1)

        c._start_boundary_drag(0, 0.5, _FakeMouseEvent(100, 50))
        assert c._dragging_boundary and not c._drag_active
        # Jitter below the threshold must not activate the drag
        c._on_mouse_move(_FakeMouseEvent(102, 51))
        assert not c._drag_active
        c._on_mouse_release(_FakeMouseEvent(102, 51))

        assert not c._dragging_boundary
        assert c._selected_boundary == (0, 0.5)          # selected...
        assert c.textgrid_data.tiers[0].intervals[0].xmax == 0.5  # ...unmoved
        assert edits == []                               # not marked dirty

    def test_move_past_threshold_activates_drag(
            self, canvas_with_sound, simple_textgrid):
        c = canvas_with_sound
        c.textgrid_data = simple_textgrid
        c._setup_axes()
        c.render()
        edits = []
        c._on_textgrid_edited = lambda: edits.append(1)

        c._start_boundary_drag(0, 0.5, _FakeMouseEvent(100, 50))
        c._on_mouse_move(_FakeMouseEvent(110, 50))  # 10 px > threshold
        assert c._drag_active
        c._on_mouse_release(_FakeMouseEvent(110, 50))
        assert not c._dragging_boundary and not c._drag_active
        assert edits == [1]  # a real move commits and marks dirty

    def test_plain_click_on_point_keeps_time(
            self, canvas_with_sound, point_textgrid):
        c = canvas_with_sound
        c.textgrid_data = point_textgrid
        c._setup_axes()
        c.render()
        edits = []
        c._on_textgrid_edited = lambda: edits.append(1)

        c._start_boundary_drag(1, 0.25, _FakeMouseEvent(200, 80))
        c._on_mouse_release(_FakeMouseEvent(200, 80))
        assert c.textgrid_data.tiers[1].points[0].time == 0.25
        assert edits == []


# ---------------------------------------------------------------------------
# Tier rename
# ---------------------------------------------------------------------------

class TestRenameTier:
    def test_rename_updates_name_and_fires_callbacks(
            self, canvas_with_sound, simple_textgrid):
        c = canvas_with_sound
        c.textgrid_data = simple_textgrid
        c._setup_axes()
        c.render()
        edits, changes = [], []
        c._on_textgrid_edited = lambda: edits.append(1)
        c._on_tiers_changed = lambda: changes.append(1)

        assert c.rename_tier(0, "  lexical ") is True
        assert c.textgrid_data.tiers[0].name == "lexical"
        assert edits == [1] and changes == [1]

    def test_rename_rejects_empty_same_or_bad_index(
            self, canvas_with_sound, simple_textgrid):
        c = canvas_with_sound
        c.textgrid_data = simple_textgrid
        c._setup_axes()
        edits = []
        c._on_textgrid_edited = lambda: edits.append(1)

        assert c.rename_tier(0, "words") is False   # unchanged name
        assert c.rename_tier(0, "   ") is False     # empty
        assert c.rename_tier(99, "x") is False      # bad index
        assert edits == []
        assert c.textgrid_data.tiers[0].name == "words"

    def test_rename_without_textgrid_is_safe(self, canvas):
        assert canvas.rename_tier(0, "x") is False


class TestDuplicateTier:
    def test_insert_tier_places_copy_and_shifts_hidden(
            self, qapp, wav_path, simple_textgrid):
        mw = MainWindow()
        mw.canvas.load_sound(wav_path)
        mw.canvas.textgrid_data = simple_textgrid
        mw._setup_tier_checkboxes()
        mw.canvas.hidden_tiers = {1}      # 'phones' hidden
        mw.canvas._setup_axes()
        mw.canvas.render()

        src = mw.canvas.textgrid_data.tiers[0]   # 'words'
        dup = src.copy(name="words_copy")
        mw._insert_tier(dup, 1)                   # insert between the two

        tiers = mw.canvas.textgrid_data.tiers
        assert [t.name for t in tiers] == ["words", "words_copy", "phones"]
        # 'phones' was index 1, now index 2 — its hidden flag must follow
        assert mw.canvas.hidden_tiers == {2}
        assert mw._textgrid_dirty is True
        # Deep copy: editing the duplicate leaves the source intact
        dup.intervals[0].text = "ZZZ"
        assert tiers[0].intervals[0].text == "hello"

    def test_rename_preserves_hidden_tiers_via_mainwindow(
            self, qapp, wav_path, simple_textgrid):
        mw = MainWindow()
        mw.canvas.load_sound(wav_path)
        mw.canvas.textgrid_data = simple_textgrid
        mw._setup_tier_checkboxes()
        mw.canvas.hidden_tiers = {1}
        mw.canvas._setup_axes()
        mw.canvas.render()
        assert mw.canvas.rename_tier(0, "renamed") is True
        assert mw.canvas.hidden_tiers == {1}  # hidden state survives rename
        assert mw.canvas.textgrid_data.tiers[0].name == "renamed"


# ---------------------------------------------------------------------------
# Undo/redo (via canvas, not FakeCanvas)
# ---------------------------------------------------------------------------

class TestUndoRedo:
    def test_undo_empty(self, canvas):
        assert canvas.undo() is False

    def test_redo_empty(self, canvas):
        assert canvas.redo() is False


# ---------------------------------------------------------------------------
# MainWindow integration
# ---------------------------------------------------------------------------

class TestMainWindow:
    def test_mainwindow_creates(self, qapp):
        mw = MainWindow()
        assert mw.canvas is not None
        assert mw.label_edit is not None
        assert mw.controls is not None

    def test_mainwindow_load_sound(self, qapp, wav_path):
        mw = MainWindow()
        mw._filepath = wav_path
        mw.canvas._filepath = wav_path
        mw.canvas.load_sound(wav_path)
        mw._run_formant_analysis()
        mw._update_scrollbar()
        mw.canvas.render()
        assert mw.canvas.sound is not None
        assert mw.canvas.formant_data is not None

    def test_mainwindow_load_with_textgrid(self, qapp, wav_path, simple_textgrid):
        mw = MainWindow()
        mw._filepath = wav_path
        mw.canvas._filepath = wav_path
        mw.canvas.load_sound(wav_path)
        mw._run_formant_analysis()
        mw.canvas.textgrid_data = simple_textgrid
        mw.canvas._setup_axes()
        mw.canvas.render()
        mw._update_scrollbar()

    def test_scrollbar_sync(self, qapp, wav_path):
        mw = MainWindow()
        mw._filepath = wav_path
        mw.canvas._filepath = wav_path
        mw.canvas.load_sound(wav_path)
        # Zoom in so view < total duration, then scrollbar should have range
        mw.canvas.zoom(0.5)
        mw._update_scrollbar()
        assert mw.scrollbar.maximum() > 0

    def test_spec_setting_change(self, qapp, wav_path):
        mw = MainWindow()
        mw._filepath = wav_path
        mw.canvas._filepath = wav_path
        mw.canvas.load_sound(wav_path)
        mw._run_formant_analysis()
        mw.canvas.render()
        # Change dynamic range
        mw.canvas.dynamic_range = 50.0
        mw.canvas.render()
        # Change brightness
        mw.canvas.brightness = 5.0
        mw.canvas.render()

    def test_edit_mode_toggle(self, qapp, wav_path):
        mw = MainWindow()
        mw._filepath = wav_path
        mw.canvas._filepath = wav_path
        mw.canvas.load_sound(wav_path)
        mw._run_formant_analysis()
        mw.canvas.render()
        mw.canvas.edit_mode = True
        mw.canvas.render()
        mw.canvas.edit_mode = False
        mw.canvas.render()


# ---------------------------------------------------------------------------
# Waveform rendering
# ---------------------------------------------------------------------------

class TestWaveform:
    def test_waveform_renders(self, canvas_with_sound):
        c = canvas_with_sound
        c.render()
        assert c._wave_plot is not None
        # Either _wave_line or _wave_fill_pos should exist
        assert (c._wave_line is not None or c._wave_fill_pos is not None)

    def test_waveform_zoomed_in(self, canvas_with_sound):
        c = canvas_with_sound
        c.set_view(0, 0.1)
        c.render()

    def test_waveform_zoomed_out(self, canvas_with_sound):
        c = canvas_with_sound
        c.set_view(0, c.total_duration)
        c.render()


# ---------------------------------------------------------------------------
# Repeated render (catch stale item cleanup bugs)
# ---------------------------------------------------------------------------

class TestRepeatedRender:
    def test_multiple_renders(self, canvas_with_sound, simple_textgrid):
        c = canvas_with_sound
        c.textgrid_data = simple_textgrid
        c._setup_axes()
        for _ in range(5):
            c.render()

    def test_render_after_zoom(self, canvas_with_sound, simple_textgrid):
        c = canvas_with_sound
        c.textgrid_data = simple_textgrid
        c._setup_axes()
        c.render()
        c.zoom(0.5)
        c.render()
        c.zoom(2.0)
        c.render()

    def test_transient_items_cleared(self, canvas_with_sound, simple_textgrid):
        c = canvas_with_sound
        c.textgrid_data = simple_textgrid
        c._setup_axes()
        c.render()
        n1 = len(c._transient_items)
        c.render()
        n2 = len(c._transient_items)
        # Counts should be similar (not accumulating)
        assert n2 <= n1 * 1.1 + 5


# ---------------------------------------------------------------------------
# Time threshold utility
# ---------------------------------------------------------------------------

class TestTimeThreshold:
    def test_time_threshold(self, canvas_with_sound):
        c = canvas_with_sound
        c.render()
        threshold = c._time_threshold_for_pixels(5)
        assert threshold > 0
        assert threshold < c.view_width


# ---------------------------------------------------------------------------
# Horizontal alignment between the audio plots and the tier panes
# ---------------------------------------------------------------------------

class TestPlotAlignment:
    """The waveform, spectrogram and tier panes are separate PlotItems.

    Linking their x ranges only aligns the data if their plot areas also start
    at the same pixel, which requires their left axes to share a width. When
    they did not, a boundary in a tier sat well away from the moment in the
    spectrogram it marked.
    """

    def _plot_areas(self, canvas):
        plots = []
        if canvas._wave_plot is not None:
            plots.append(canvas._wave_plot)
        if canvas._spec_plot is not None:
            plots.append(canvas._spec_plot)
        plots.extend(canvas._tier_plots)
        return [p.getViewBox().sceneBoundingRect() for p in plots]

    def test_plot_areas_share_horizontal_extent(self, canvas_with_sound,
                                                simple_textgrid, qapp):
        canvas_with_sound.textgrid_data = simple_textgrid
        canvas_with_sound._setup_axes()
        canvas_with_sound.render()
        qapp.processEvents()

        rects = self._plot_areas(canvas_with_sound)
        assert len(rects) >= 3
        assert max(r.left() for r in rects) - min(r.left() for r in rects) < 0.5
        assert max(r.right() for r in rects) - min(r.right() for r in rects) < 0.5

    def test_left_axis_width_is_uniform(self, canvas_with_sound,
                                        simple_textgrid, qapp):
        canvas_with_sound.textgrid_data = simple_textgrid
        canvas_with_sound._setup_axes()
        qapp.processEvents()

        plots = [canvas_with_sound._spec_plot] + canvas_with_sound._tier_plots
        if canvas_with_sound._wave_plot is not None:
            plots.append(canvas_with_sound._wave_plot)
        widths = {p.getAxis('left').width() for p in plots}
        assert len(widths) == 1, widths

    def test_alignment_survives_a_very_long_tier_name(self, canvas_with_sound,
                                                      simple_textgrid, qapp):
        simple_textgrid.tiers[0].name = "a"
        simple_textgrid.tiers[1].name = "an extremely long tier name indeed"
        canvas_with_sound.textgrid_data = simple_textgrid
        canvas_with_sound._setup_axes()
        canvas_with_sound.render()
        qapp.processEvents()

        rects = self._plot_areas(canvas_with_sound)
        assert max(r.left() for r in rects) - min(r.left() for r in rects) < 0.5

    def test_alignment_holds_without_a_textgrid(self, canvas_with_sound, qapp):
        canvas_with_sound.textgrid_data = None
        canvas_with_sound._setup_axes()
        canvas_with_sound.render()
        qapp.processEvents()

        rects = self._plot_areas(canvas_with_sound)
        assert max(r.left() for r in rects) - min(r.left() for r in rects) < 0.5

    def test_left_axis_width_accommodates_frequency_labels(self, canvas):
        """The shared width must not crop the spectrogram's own axis."""
        assert canvas._left_axis_width([]) >= canvas.MIN_LEFT_AXIS_WIDTH
        assert canvas._left_axis_width(["x"]) >= canvas.MIN_LEFT_AXIS_WIDTH

    def test_left_axis_width_grows_with_tier_names(self, canvas):
        narrow = canvas._left_axis_width(["ab"])
        wide = canvas._left_axis_width(["ab", "a very much longer tier name"])
        assert wide > narrow
