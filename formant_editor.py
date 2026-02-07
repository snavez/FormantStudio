"""
FormantStudio — Manual Formant Editor
Built on Praat's analysis engine (via Parselmouth) with a PyQt6 editing interface.

Milestone 1:
  - Open WAV file, display spectrogram
  - Overlay Praat-calculated formants (F1–F3 default, up to F5), color-coded
  - Spectrogram contrast/brightness/pre-emphasis sliders
  - Click-drag formant editing with F1–F5 key selection
  - Save/load formant data alongside TextGrid files

Formant analysis uses Praat (Boersma & Weenink) via Parselmouth.
"""

import sys
import os
import re
import json
import time as _time
import numpy as np
import parselmouth
from parselmouth import praat
import sounddevice as sd

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QToolBar, QFileDialog, QLabel, QSlider, QGroupBox, QStatusBar,
    QPushButton, QSplitter, QComboBox, QDoubleSpinBox, QCheckBox,
    QMessageBox, QSizePolicy, QScrollBar, QLineEdit, QDialog,
    QDialogButtonBox, QFormLayout, QGridLayout
)
from PyQt6.QtCore import Qt, QTimer, QEvent, QObject, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence, QColor, QFont

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle as MplRectangle
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FORMANT_COLORS = {
    1: "#3399FF",   # F1 — blue
    2: "#FFD700",   # F2 — yellow/gold
    3: "#FF4444",   # F3 — red
    4: "#44CC44",   # F4 — green
    5: "#CC44CC",   # F5 — magenta
}

FORMANT_LABELS = {1: "F1", 2: "F2", 3: "F3", 4: "F4", 5: "F5"}

DEFAULT_NUM_FORMANTS = 3       # Show F1-F3 by default (user can increase)
MAX_FORMANT_HZ = 5500.0        # Default max formant frequency (Hz)
DEFAULT_WINDOW_LENGTH = 0.025  # 25 ms — Praat default
DEFAULT_TIME_STEP = 0.0       # 0 = auto (Praat default: 25% of window length)
DEFAULT_PRE_EMPHASIS = 50.0    # Hz — Praat default

# Spectrogram display defaults
DEFAULT_DYNAMIC_RANGE = 70.0   # dB
DEFAULT_SPEC_MAX_FREQ = 5500.0
DEFAULT_SPEC_WINDOW = 0.005    # seconds — spectrogram analysis window

# View limits
MAX_SPECTROGRAM_VIEW = 10.0    # seconds — hide spectrogram beyond this
MIN_VIEW_WIDTH = 0.1           # seconds — maximum zoom
ZOOM_FACTOR = 1.3              # per scroll wheel notch


# ---------------------------------------------------------------------------
# Formant data container
# ---------------------------------------------------------------------------

class FormantData:
    """
    Stores formant values (original + edited) for an audio file.

    Shape: (n_formants, n_frames) — NaN where no value exists.
    """

    def __init__(self, times=None, values=None, n_formants=5, time_step=None):
        self.n_formants = n_formants
        self.time_step = time_step  # seconds between frames
        self.times = times if times is not None else np.array([])
        # values[f, t] = frequency in Hz for formant f+1 at time index t
        if values is not None:
            self.values = values.copy()
        else:
            self.values = np.full((n_formants, 0), np.nan)

        # Track which points have been manually edited
        self.edited_mask = np.zeros_like(self.values, dtype=bool)
        # Keep original for undo/comparison
        self.original_values = self.values.copy()

    @property
    def n_frames(self):
        return len(self.times)

    def set_value(self, formant_idx, frame_idx, value_hz):
        """Set a formant value (0-indexed formant, 0-indexed frame)."""
        if 0 <= formant_idx < self.n_formants and 0 <= frame_idx < self.n_frames:
            self.values[formant_idx, frame_idx] = value_hz
            self.edited_mask[formant_idx, frame_idx] = True

    def reset_to_original(self, formant_idx=None):
        """Reset edited values back to Praat's original calculation."""
        if formant_idx is not None:
            self.values[formant_idx] = self.original_values[formant_idx].copy()
            self.edited_mask[formant_idx] = False
        else:
            self.values = self.original_values.copy()
            self.edited_mask[:] = False

    def save(self, filepath):
        """
        Save formant data to a .formants JSON file.

        Includes time_step, times, values (with NaN preserved as null),
        edited_mask, and metadata.
        """
        data = {
            "format_version": 1,
            "generator": "FormantStudio",
            "analysis_engine": "Praat (via Parselmouth)",
            "n_formants": self.n_formants,
            "n_frames": self.n_frames,
            "time_step_seconds": self.time_step,
            "times": self.times.tolist(),
            "values": [],
            "edited_mask": [],
        }
        for f in range(self.n_formants):
            # Convert NaN to None for JSON
            vals = [None if np.isnan(v) else round(float(v), 2) for v in self.values[f]]
            mask = self.edited_mask[f].tolist()
            data["values"].append(vals)
            data["edited_mask"].append(mask)

        with open(filepath, "w") as fh:
            json.dump(data, fh, indent=2)

    @classmethod
    def load(cls, filepath):
        """Load formant data from a .formants JSON file."""
        with open(filepath, "r") as fh:
            data = json.load(fh)

        n_formants = data["n_formants"]
        times = np.array(data["times"])
        values = np.array(
            [[np.nan if v is None else v for v in row] for row in data["values"]]
        )
        obj = cls(
            times=times,
            values=values,
            n_formants=n_formants,
            time_step=data.get("time_step_seconds"),
        )
        if "edited_mask" in data:
            obj.edited_mask = np.array(data["edited_mask"], dtype=bool)
        # When loading, the "original" is whatever was saved
        # (we don't have the raw Praat output anymore unless we re-analyse)
        obj.original_values = values.copy()
        return obj


# ---------------------------------------------------------------------------
# TextGrid data model and parser
# ---------------------------------------------------------------------------

class Interval:
    """One labeled segment in an IntervalTier."""
    __slots__ = ("xmin", "xmax", "text")

    def __init__(self, xmin, xmax, text):
        self.xmin = xmin
        self.xmax = xmax
        self.text = text

    def __eq__(self, other):
        if not isinstance(other, Interval):
            return NotImplemented
        return (self.xmin == other.xmin and self.xmax == other.xmax
                and self.text == other.text)

    def __repr__(self):
        return f"Interval({self.xmin}, {self.xmax}, {self.text!r})"


class Point:
    """One labeled time point in a TextTier."""
    __slots__ = ("time", "mark")

    def __init__(self, time, mark):
        self.time = time
        self.mark = mark

    def __eq__(self, other):
        if not isinstance(other, Point):
            return NotImplemented
        return self.time == other.time and self.mark == other.mark

    def __repr__(self):
        return f"Point({self.time}, {self.mark!r})"


class Tier:
    """One tier in a TextGrid."""
    __slots__ = ("name", "tier_class", "xmin", "xmax", "intervals", "points")

    def __init__(self, name, tier_class, xmin, xmax, intervals=None, points=None):
        self.name = name
        self.tier_class = tier_class  # "IntervalTier" or "TextTier"
        self.xmin = xmin
        self.xmax = xmax
        self.intervals = intervals or []
        self.points = points or []

    def __eq__(self, other):
        if not isinstance(other, Tier):
            return NotImplemented
        return (self.name == other.name and self.tier_class == other.tier_class
                and self.xmin == other.xmin and self.xmax == other.xmax
                and self.intervals == other.intervals
                and self.points == other.points)

    def __repr__(self):
        if self.tier_class == "IntervalTier":
            return (f"Tier({self.name!r}, {self.tier_class!r}, "
                    f"{len(self.intervals)} intervals)")
        return (f"Tier({self.name!r}, {self.tier_class!r}, "
                f"{len(self.points)} points)")


class TextGrid:
    """Top-level TextGrid container."""

    def __init__(self, xmin, xmax, tiers):
        self.xmin = xmin
        self.xmax = xmax
        self.tiers = tiers  # list of Tier

    def __eq__(self, other):
        if not isinstance(other, TextGrid):
            return NotImplemented
        return (self.xmin == other.xmin and self.xmax == other.xmax
                and self.tiers == other.tiers)

    def save(self, filepath):
        """Save TextGrid to a file in Praat normal text format."""
        def _escape(s):
            return s.replace('"', '""')

        lines = []
        lines.append('File type = "ooTextFile"')
        lines.append('Object class = "TextGrid"')
        lines.append('')
        lines.append(f'xmin = {self.xmin} ')
        lines.append(f'xmax = {self.xmax} ')
        lines.append('tiers? <exists> ')
        lines.append(f'size = {len(self.tiers)} ')
        lines.append('item []: ')

        for ti, tier in enumerate(self.tiers, 1):
            lines.append(f'    item [{ti}]:')
            lines.append(f'        class = "{_escape(tier.tier_class)}" ')
            lines.append(f'        name = "{_escape(tier.name)}" ')
            lines.append(f'        xmin = {tier.xmin} ')
            lines.append(f'        xmax = {tier.xmax} ')

            if tier.tier_class == "IntervalTier":
                lines.append(f'        intervals: size = {len(tier.intervals)} ')
                for ii, iv in enumerate(tier.intervals, 1):
                    lines.append(f'        intervals [{ii}]:')
                    lines.append(f'            xmin = {iv.xmin} ')
                    lines.append(f'            xmax = {iv.xmax} ')
                    lines.append(f'            text = "{_escape(iv.text)}" ')
            elif tier.tier_class == "TextTier":
                lines.append(f'        points: size = {len(tier.points)} ')
                for pi, pt in enumerate(tier.points, 1):
                    lines.append(f'        points [{pi}]:')
                    lines.append(f'            number = {pt.time} ')
                    lines.append(f'            mark = "{_escape(pt.mark)}" ')

        with open(filepath, "w", encoding="utf-8") as fh:
            fh.write('\n'.join(lines) + '\n')

    @classmethod
    def from_file(cls, filepath):
        """Parse a TextGrid file (normal or short format)."""
        with open(filepath, "r", encoding="utf-8") as fh:
            text = fh.read()
        tokens = _tokenize_textgrid(text)
        return _parse_textgrid_tokens(tokens)


# Token pattern: quoted strings (with escaped ""), numbers, or flag tokens.
# The first alternative matches [N] bracket indices (no capture) to skip them.
_TG_TOKEN_RE = re.compile(
    r'\[\d+\]'                      # bracketed index [1], [2] — skip (no capture)
    r'|'
    r'"((?:[^"]*(?:""[^"]*)*))"'   # quoted string (captures content inside quotes)
    r'|'
    r'(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)'  # number
    r'|'
    r'(<exists>)'                   # exists flag
)


def _tokenize_textgrid(text):
    """
    Extract tokens from TextGrid text. Works identically for normal and
    short formats since both contain the same quoted strings, numbers,
    and flags in the same order — normal format just has extra labels
    between them which this regex skips.
    """
    tokens = []
    for m in _TG_TOKEN_RE.finditer(text):
        if m.group(1) is not None:
            # Quoted string — unescape Praat's "" -> "
            tokens.append(m.group(1).replace('""', '"'))
        elif m.group(2) is not None:
            # Number
            tokens.append(float(m.group(2)))
        elif m.group(3) is not None:
            # Flag
            tokens.append(m.group(3))
    return tokens


def _parse_textgrid_tokens(tokens):
    """
    Parse the token sequence into a TextGrid object.

    Expected token order:
      "ooTextFile" "TextGrid" xmin xmax <exists> n_tiers
      [per tier: class_str name_str xmin xmax n_items [per item: values...]]
    """
    i = 0

    def next_str():
        nonlocal i
        val = tokens[i]
        i += 1
        return str(val)

    def next_num():
        nonlocal i
        val = tokens[i]
        i += 1
        return float(val)

    def next_int():
        return int(next_num())

    def skip_flag():
        nonlocal i
        if i < len(tokens) and tokens[i] == "<exists>":
            i += 1

    # Header: "ooTextFile" "TextGrid"
    header = next_str()
    if header != "ooTextFile":
        raise ValueError(f"Not a TextGrid file (expected 'ooTextFile', got {header!r})")
    file_type = next_str()
    if file_type != "TextGrid":
        raise ValueError(f"Not a TextGrid file (expected 'TextGrid', got {file_type!r})")

    tg_xmin = next_num()
    tg_xmax = next_num()
    skip_flag()
    n_tiers = next_int()

    tiers = []
    for _ in range(n_tiers):
        tier_class = next_str()
        tier_name = next_str()
        tier_xmin = next_num()
        tier_xmax = next_num()
        n_items = next_int()

        if tier_class == "IntervalTier":
            intervals = []
            for _ in range(n_items):
                ival_xmin = next_num()
                ival_xmax = next_num()
                ival_text = next_str()
                intervals.append(Interval(ival_xmin, ival_xmax, ival_text))
            tiers.append(Tier(tier_name, tier_class, tier_xmin, tier_xmax,
                              intervals=intervals))
        elif tier_class == "TextTier":
            points = []
            for _ in range(n_items):
                pt_time = next_num()
                pt_mark = next_str()
                points.append(Point(pt_time, pt_mark))
            tiers.append(Tier(tier_name, tier_class, tier_xmin, tier_xmax,
                              points=points))
        else:
            raise ValueError(f"Unknown tier class: {tier_class!r}")

    return TextGrid(tg_xmin, tg_xmax, tiers)


# ---------------------------------------------------------------------------
# Extract formant data from Praat Formant object
# ---------------------------------------------------------------------------

def extract_formants_from_praat(sound, n_formants=5, max_formant=5500.0,
                                 window_length=0.025, time_step=0.0,
                                 pre_emphasis=50.0):
    """
    Run Praat's Burg formant analysis and return a FormantData object.
    """
    formant_obj = sound.to_formant_burg(
        time_step=time_step if time_step > 0 else None,
        max_number_of_formants=float(n_formants),
        maximum_formant=max_formant,
        window_length=window_length,
        pre_emphasis_from=max(pre_emphasis, 1.0),
    )

    n_frames = formant_obj.n_frames
    times = np.array(formant_obj.xs())

    # Compute actual time step from first two frames (if available)
    if n_frames >= 2:
        actual_time_step = times[1] - times[0]
    else:
        actual_time_step = window_length * 0.25  # Praat default

    values = np.full((5, n_frames), np.nan)  # Always store 5 slots
    for f_idx in range(n_formants):
        for t_idx in range(n_frames):
            try:
                val = formant_obj.get_value_at_time(f_idx + 1, times[t_idx])
                if val is not None and not np.isnan(val) and val > 0:
                    values[f_idx, t_idx] = val
            except Exception:
                pass

    return FormantData(
        times=times,
        values=values,
        n_formants=5,
        time_step=actual_time_step,
    )


# ---------------------------------------------------------------------------
# Inline Label Editor
# ---------------------------------------------------------------------------

class LabelEdit(QLineEdit):
    """Inline label editor for TextGrid intervals/points.

    Positioned between the canvas and the scrollbar. Overrides Tab/Escape/Enter
    keys for playback and editing workflow.
    """

    escape_pressed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._play_callback = None
        self.setFixedHeight(36)
        self.setMaximumWidth(600)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setPlaceholderText("Select an interval to edit label")
        self.setEnabled(False)
        self.setStyleSheet("""
            QLineEdit {
                background-color: #ffffff;
                color: #000000;
                border: 1px solid #aaaaaa;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 16px;
            }
            QLineEdit:disabled {
                background-color: #f0f0f0;
                color: #999999;
            }
            QLineEdit:focus {
                border-color: #3366cc;
                border-width: 2px;
            }
        """)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_Escape:
            self.clearFocus()
            self.escape_pressed.emit()
            event.accept()
            return
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.clearFocus()
            event.accept()
            return
        super().keyPressEvent(event)


# ---------------------------------------------------------------------------
# Spectrogram Canvas (matplotlib embedded in Qt)
# ---------------------------------------------------------------------------

class SpectrogramCanvas(FigureCanvas):
    """
    Matplotlib canvas displaying:
      - Spectrogram (with adjustable display settings)
      - Formant overlay (color-coded dots/lines)
      - Interactive editing with blitting for responsiveness
    """

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(14, 5), dpi=100)
        self.fig.set_facecolor("#1e1e1e")
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Data
        self.sound = None
        self.spectrogram_data = None
        self.spec_times = None
        self.spec_freqs = None
        self.formant_data = None
        self.textgrid_data = None   # TextGrid object or None
        self.tier_axes = []         # list of axes for TextGrid tiers

        # Display settings
        self.dynamic_range = DEFAULT_DYNAMIC_RANGE
        self.brightness = 0.0  # offset in dB
        self.max_freq = DEFAULT_SPEC_MAX_FREQ
        self.spec_window_length = DEFAULT_SPEC_WINDOW
        self.show_formants = True
        self.display_n_formants = DEFAULT_NUM_FORMANTS

        # View (zoom/scroll)
        self.view_start = 0.0
        self.view_end = 0.0
        self.total_duration = 0.0

        # Edit state
        self.edit_mode = False
        self.active_formant = 0  # 0-indexed (0 = F1)
        self.is_drawing = False
        self._last_frame_idx = -1
        self._bg_cache = None
        self._stroke_scatter = None
        self._stroke_times = []
        self._stroke_freqs = []

        # Playback state
        self._playback_playing = False
        self._playback_start_time = 0.0   # audio time where playback starts
        self._playback_start_wall = 0.0   # wall-clock time at playback start
        self._playback_end_time = 0.0     # audio time where playback ends
        self._playback_cursor = None      # axvline artist
        self._playback_timer = None       # QTimer for cursor animation
        self._playback_cursor_bg = None   # blit cache
        self._click_time = None           # last left-click position
        self._hover_time = None           # current mouse hover time on spectrogram

        # TextGrid editing state
        self._selected_boundary = None      # (tier_index, time) or None
        self._dragging_boundary = False
        self._drag_tier_index = None
        self._drag_original_time = None
        self._drag_min_time = 0.0
        self._drag_max_time = 0.0

        # Interval/point selection state
        self._selected_interval = None      # (tier_idx, interval_idx) or None
        self._label_edit = None             # set by MainWindow (LabelEdit widget)

        # Spectrogram drag selection state
        self._selection_start = None        # time range start
        self._selection_end = None          # time range end
        self._spec_dragging = False         # currently drag-selecting on spectrogram
        self._spec_drag_start = None        # start time of spectrogram drag

        # Blit state for drag operations (avoid full render per frame)
        self._drag_bg = None                # cached figure background
        self._drag_line_spec = None         # boundary drag: axvline on spectrogram
        self._drag_line_tier = None         # boundary drag: axvline on tier
        self._spec_sel_artist = None        # spec drag: axvspan on spectrogram
        self._spec_sel_tier_artists = []    # spec drag: axvspan on each tier

        # Active tier state
        self._active_tier = None  # int index of the active/selected tier

        # Debounce timers
        self._render_timer = QTimer()
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(30)
        self._render_timer.timeout.connect(self._debounced_render)
        self._label_render_timer = QTimer()
        self._label_render_timer.setSingleShot(True)
        self._label_render_timer.setInterval(100)
        self._label_render_timer.timeout.connect(self._debounced_render)

        # Overlay blit state (selection/boundary highlights without full render)
        self._overlay_bg = None       # cached base content (no selection overlays)
        self._overlay_artists = []    # current overlay artists (for cleanup)

        # Crosshair state
        self._crosshair_h = None  # horizontal line artist
        self._crosshair_v = None  # vertical line artist
        self._crosshair_visible = False
        self._status_callback = None  # set by MainWindow

        # Connect mouse events
        self.mpl_connect("button_press_event", self._on_mouse_press)
        self.mpl_connect("button_release_event", self._on_mouse_release)
        self.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.mpl_connect("scroll_event", self._on_scroll)
        self._setup_empty_axes()

    def _setup_empty_axes(self):
        self.ax.set_facecolor("#1a1a2e")
        self.ax.set_xlabel("Time (s)", color="#cccccc", fontsize=9)
        self.ax.set_ylabel("Frequency (Hz)", color="#cccccc", fontsize=9)
        self.ax.tick_params(colors="#999999", labelsize=8)
        self.ax.set_title("Open a WAV file to begin", color="#cccccc", fontsize=11)
        self.fig.tight_layout()
        self.draw()


    def load_sound(self, filepath):
        """Load audio and compute spectrogram."""
        self.sound = parselmouth.Sound(filepath)
        self.total_duration = self.sound.duration
        self.view_start = 0.0
        self.view_end = self.total_duration
        self._compute_spectrogram()

    def _compute_spectrogram(self):
        """Compute spectrogram using Praat."""
        if self.sound is None:
            return

        spec_obj = self.sound.to_spectrogram(
            window_length=self.spec_window_length,
            maximum_frequency=self.max_freq,
            time_step=0.002,
            frequency_step=20.0,
        )

        self.spectrogram_data = spec_obj.values  # shape: (n_freq, n_time)
        self.spec_freqs = np.array(spec_obj.ys())
        self.spec_times = np.array(spec_obj.xs())

    @property
    def view_width(self):
        return self.view_end - self.view_start

    def set_view(self, start, end):
        """Set the visible time range, clamping to valid bounds."""
        width = end - start
        width = max(MIN_VIEW_WIDTH, min(width, self.total_duration))
        start = max(0.0, start)
        end = start + width
        if end > self.total_duration:
            end = self.total_duration
            start = max(0.0, end - width)
        self.view_start = start
        self.view_end = end

    def zoom(self, factor, center_time=None):
        """Zoom in/out by factor, centered on center_time."""
        if center_time is None:
            center_time = (self.view_start + self.view_end) / 2.0

        old_width = self.view_width
        new_width = old_width * factor
        new_width = max(MIN_VIEW_WIDTH, min(new_width, self.total_duration))

        # Keep center_time at the same relative position in the view
        if old_width > 0:
            ratio = (center_time - self.view_start) / old_width
        else:
            ratio = 0.5
        ratio = max(0.0, min(1.0, ratio))

        new_start = center_time - ratio * new_width
        self.set_view(new_start, new_start + new_width)

    def _debounced_render(self):
        """Slot for debounce timers — just calls render()."""
        self.render()

    def render(self):
        """Full re-render of spectrogram + formants + TextGrid tiers."""
        if self._playback_playing:
            self.stop_playback()
        self.ax.clear()
        self._bg_cache = None

        if self.spectrogram_data is None:
            self._setup_empty_axes()
            return

        view_width = self.view_width
        show_spectrogram = view_width <= MAX_SPECTROGRAM_VIEW

        if show_spectrogram:
            # Slice spectrogram to visible time range for performance
            t_mask = ((self.spec_times >= self.view_start) &
                      (self.spec_times <= self.view_end))
            vis_times = self.spec_times[t_mask]
            vis_data = self.spectrogram_data[:, t_mask]

            if len(vis_times) > 0:
                power_db = 10 * np.log10(vis_data + 1e-20)
                peak_db = np.max(power_db)
                vmax = peak_db + self.brightness
                vmin = vmax - self.dynamic_range

                self.ax.imshow(
                    power_db,
                    aspect="auto",
                    origin="lower",
                    cmap="gray_r",
                    vmin=vmin, vmax=vmax,
                    extent=[vis_times[0], vis_times[-1],
                            self.spec_freqs[0], self.spec_freqs[-1]],
                    interpolation="bilinear",
                )
        else:
            # View too wide for spectrogram — show message
            mid_t = (self.view_start + self.view_end) / 2
            mid_f = self.max_freq / 2
            self.ax.text(
                mid_t, mid_f,
                f"Zoom in to view spectrogram\n"
                f"(current view: {view_width:.1f}s, max: {MAX_SPECTROGRAM_VIEW:.0f}s)\n\n"
                f"Use mouse wheel to zoom",
                ha="center", va="center", color="#666688",
                fontsize=13, fontstyle="italic",
            )

        self.ax.set_xlim(self.view_start, self.view_end)
        self.ax.set_ylim(0, self.max_freq)
        self.ax.set_ylabel("Frequency (Hz)", color="#cccccc", fontsize=9)
        self.ax.tick_params(colors="#999999", labelsize=8)
        self.ax.set_facecolor("#1a1a2e")

        # Draw formants
        if self.show_formants and self.formant_data is not None:
            self._draw_formants()

        # Draw TextGrid tiers
        if self.textgrid_data is not None and len(self.tier_axes) > 0:
            self._draw_textgrid()
        else:
            # No tiers — spectrogram shows its own x-axis label
            self.ax.set_xlabel("Time (s)", color="#cccccc", fontsize=9)

        # Title with edit mode indicator
        title = os.path.basename(self._filepath) if hasattr(self, "_filepath") else ""
        if self.edit_mode:
            fn = self.active_formant + 1
            color = FORMANT_COLORS.get(fn, "#ffffff")
            title += f"  |  EDIT MODE — Drawing {FORMANT_LABELS[fn]}"
            self.ax.set_title(title, color=color, fontsize=11, fontweight="bold")
        else:
            self.ax.set_title(title, color="#cccccc", fontsize=11)

        if self.tier_axes:
            self.fig.subplots_adjust(
                left=0.12, right=0.98, top=0.94, bottom=0.08, hspace=0.05
            )
        else:
            self.fig.subplots_adjust(
                left=0.12, right=0.98, top=0.94, bottom=0.10
            )

        # Create crosshair artists (invisible) before draw so they are
        # part of the artist tree but don't trigger extra redraws.
        self._crosshair_h = self.ax.axhline(
            y=0, color="#00ffcc", linewidth=0.7, alpha=0.8,
            linestyle="-", visible=False, zorder=10,
        )
        self._crosshair_v = self.ax.axvline(
            x=0, color="#00ffcc", linewidth=0.7, alpha=0.8,
            linestyle="-", visible=False, zorder=10,
        )
        self._crosshair_visible = False

        self.draw()

        # Cache base content for overlay blitting (before selection overlays)
        self._overlay_bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)

        # Draw selection/boundary overlays on top via blitting
        self._draw_selection_overlay()

    def _draw_selection_overlay(self):
        """Draw selection/boundary highlights via blitting over cached base.

        This avoids a full render() for click interactions that only change
        which interval/boundary/time-range is highlighted.
        """
        # Remove previous overlay artists
        for a in self._overlay_artists:
            a.remove()
        self._overlay_artists = []

        if self._overlay_bg is None:
            return

        self.fig.canvas.restore_region(self._overlay_bg)

        # --- Time selection highlight (spectrogram + tier axes) ---
        if self._selection_start is not None and self._selection_end is not None:
            a = self.ax.axvspan(
                self._selection_start, self._selection_end,
                color="#3366cc", alpha=0.25, zorder=5,
            )
            self._overlay_artists.append(a)
            for tier_ax in self.tier_axes:
                a = tier_ax.axvspan(
                    self._selection_start, self._selection_end,
                    color="#3366cc", alpha=0.15, zorder=0,
                )
                self._overlay_artists.append(a)

        # --- Selected interval highlight ---
        if self._selected_interval is not None:
            sel_tier_idx, sel_i = self._selected_interval
            if (self.textgrid_data is not None
                    and 0 <= sel_tier_idx < len(self.textgrid_data.tiers)):
                tier = self.textgrid_data.tiers[sel_tier_idx]
                if (tier.tier_class == "IntervalTier"
                        and sel_i < len(tier.intervals)):
                    iv = tier.intervals[sel_i]
                    if sel_tier_idx < len(self.tier_axes):
                        a = self.tier_axes[sel_tier_idx].axvspan(
                            iv.xmin, iv.xmax,
                            color="#3366cc", alpha=0.25, zorder=0)
                        self._overlay_artists.append(a)

        # --- Selected boundary highlight + shadow boundaries ---
        if self._selected_boundary is not None:
            sel_tier, sel_time = self._selected_boundary
            view_start = self.view_start
            view_end = self.view_end
            if view_start <= sel_time <= view_end:
                # Red highlight on spectrogram
                a = self.ax.axvline(sel_time, color="#ff0000", linewidth=2.5,
                                    alpha=0.9, zorder=6)
                self._overlay_artists.append(a)
                # Red highlight on the selected tier
                if 0 <= sel_tier < len(self.tier_axes):
                    a = self.tier_axes[sel_tier].axvline(
                        sel_time, color="#ff0000", linewidth=2.5,
                        alpha=0.9, zorder=6)
                    self._overlay_artists.append(a)

                # Shadow boundaries on OTHER tiers
                for other_idx, other_ax in enumerate(self.tier_axes):
                    if other_idx == sel_tier:
                        continue
                    # Dashed gray line
                    a = other_ax.axvline(sel_time, color="#888888",
                                         linewidth=1.0, linestyle="--",
                                         alpha=0.6, zorder=4)
                    self._overlay_artists.append(a)
                    # Clickable circle
                    a = other_ax.scatter(
                        [sel_time], [0.5], s=40, c="#6688cc",
                        edgecolors="white", linewidths=1.0, zorder=5,
                    )
                    self._overlay_artists.append(a)

        # Draw all overlay artists via blit
        for a in self._overlay_artists:
            a.axes.draw_artist(a)

        self.fig.canvas.blit(self.fig.bbox)

        # Recapture crosshair bg to include overlays
        self._crosshair_bg = self.fig.canvas.copy_from_bbox(self.ax.bbox)

    def _draw_formants(self):
        """Draw formant overlay on the spectrogram."""
        fd = self.formant_data

        # Only draw points within the visible time range
        t_mask = ((fd.times >= self.view_start) & (fd.times <= self.view_end))

        for f_idx in range(self.display_n_formants):
            fn = f_idx + 1
            color = FORMANT_COLORS.get(fn, "#ffffff")

            vals = fd.values[f_idx]
            edited = fd.edited_mask[f_idx]
            valid = ~np.isnan(vals) & t_mask

            if not np.any(valid):
                continue

            is_active = (self.edit_mode and f_idx == self.active_formant)

            if is_active:
                # In edit mode for this formant: only show edited points
                # so the user sees their corrections against the clean spectrogram
                edited_valid = valid & edited
                if np.any(edited_valid):
                    self.ax.scatter(
                        fd.times[edited_valid], vals[edited_valid],
                        c=color, s=8, alpha=1.0, zorder=4,
                        edgecolors="white", linewidths=0.3,
                    )
            else:
                # Draw unedited points as small dots
                unedited_mask = valid & ~edited
                if np.any(unedited_mask):
                    self.ax.scatter(
                        fd.times[unedited_mask], vals[unedited_mask],
                        c=color, s=2, alpha=0.7, zorder=3,
                    )

                # Draw edited points as slightly larger, brighter dots
                edited_valid = valid & edited
                if np.any(edited_valid):
                    self.ax.scatter(
                        fd.times[edited_valid], vals[edited_valid],
                        c=color, s=8, alpha=1.0, zorder=4,
                        edgecolors="white", linewidths=0.3,
                    )

    # -------------------------------------------------------------------
    # TextGrid axes layout and drawing
    # -------------------------------------------------------------------

    def _setup_axes(self):
        """
        Recreate figure axes layout based on whether a TextGrid is loaded.
        No TextGrid: single axes. With TextGrid: GridSpec with spectrogram
        on top and one axes per tier below, all sharing x-axis.
        """
        from matplotlib.gridspec import GridSpec

        self.fig.clear()
        self.tier_axes = []

        tg = self.textgrid_data
        if tg is None or len(tg.tiers) == 0:
            # Simple layout — single axes
            self.ax = self.fig.add_subplot(111)
            return

        n_tiers = len(tg.tiers)
        # Height ratios: spectrogram ~65%, tiers share ~35% (generous for readability)
        tier_share = max(3, 35 // n_tiers)  # per-tier height ratio
        spec_share = 65
        ratios = [spec_share] + [tier_share] * n_tiers

        gs = GridSpec(1 + n_tiers, 1, height_ratios=ratios, hspace=0.05,
                      figure=self.fig)

        self.ax = self.fig.add_subplot(gs[0])
        for i in range(n_tiers):
            tier_ax = self.fig.add_subplot(gs[1 + i], sharex=self.ax)
            self.tier_axes.append(tier_ax)

    def _draw_textgrid(self):
        """Render TextGrid tiers into their axes and boundary lines on spectrogram.

        Uses batch draw calls (vlines, broken_barh, scatter) instead of
        per-interval artists to stay fast with large TextGrids.
        """
        tg = self.textgrid_data
        if tg is None or len(self.tier_axes) == 0:
            return

        view_start = self.view_start
        view_end = self.view_end

        # Collect all boundary times for a single batch vlines on spectrogram
        spec_interval_boundaries = set()
        spec_point_times = []

        for tier_idx, (tier, tier_ax) in enumerate(zip(tg.tiers, self.tier_axes)):
            tier_ax.clear()
            is_active = (tier_idx == self._active_tier)
            tier_ax.set_facecolor("#fff8c4" if is_active else "#ffffff")
            tier_ax.set_xlim(view_start, view_end)
            tier_ax.set_ylim(0, 1)
            tier_ax.set_yticks([])
            tier_ax.tick_params(colors="#555555", labelsize=8)
            tier_ax.set_ylabel(tier.name,
                               color="#cc2200" if is_active else "#99bbdd",
                               fontsize=10,
                               fontweight="bold" if is_active else "normal",
                               rotation=0, ha="right", va="center",
                               labelpad=10)
            # Subtle border around tier
            for spine in tier_ax.spines.values():
                spine.set_color("#999999")
                spine.set_linewidth(0.5)

            if tier.tier_class == "IntervalTier":
                # Filter to visible intervals
                vis = [iv for iv in tier.intervals
                       if iv.xmax > view_start and iv.xmin < view_end]

                if vis:
                    # Batch boundary lines on tier axes (one vlines call)
                    boundaries = set()
                    for iv in vis:
                        boundaries.add(iv.xmin)
                        boundaries.add(iv.xmax)
                        spec_interval_boundaries.add(iv.xmin)
                        spec_interval_boundaries.add(iv.xmax)
                    tier_ax.vlines(sorted(boundaries), 0, 1,
                                   colors="#4444aa", linewidths=0.8, zorder=1)

                    # Labels — black text, larger font
                    for iv in vis:
                        if iv.text:
                            mid_t = (iv.xmin + iv.xmax) / 2.0
                            if view_start <= mid_t <= view_end:
                                tier_ax.text(
                                    mid_t, 0.5, iv.text,
                                    ha="center", va="center",
                                    color="#000000", fontsize=10,
                                    clip_on=True, zorder=2,
                                )

            elif tier.tier_class == "TextTier":
                vis = [pt for pt in tier.points
                       if view_start <= pt.time <= view_end]

                if vis:
                    times = [pt.time for pt in vis]
                    spec_point_times.extend(times)

                    # Batch marker lines on tier axes
                    tier_ax.vlines(times, 0, 1,
                                   colors="#cc4400", linewidths=1.0, zorder=1)

                    # Batch diamond markers — centered at 0.5
                    tier_ax.scatter(times, [0.5] * len(vis), marker="D",
                                   c="#cc4400", s=30, zorder=3)

                    # Labels — centered on point line (va="center")
                    for pt in vis:
                        if pt.mark:
                            tier_ax.text(
                                pt.time, 0.5, "  " + pt.mark,
                                ha="left", va="center",
                                color="#000000", fontsize=10,
                                fontweight="bold",
                                clip_on=True, zorder=2,
                            )

        # Single batch vlines call on spectrogram for interval boundaries
        if spec_interval_boundaries:
            self.ax.vlines(sorted(spec_interval_boundaries),
                           0, self.max_freq,
                           colors="#4488ff", linewidths=1.0,
                           linestyles="--", alpha=0.7, zorder=2)

        # Single batch vlines call on spectrogram for point markers
        if spec_point_times:
            self.ax.vlines(spec_point_times,
                           0, self.max_freq,
                           colors="#ff6622", linewidths=1.2,
                           linestyles="--", alpha=0.7, zorder=2)

        # Hide x-axis labels on all but the bottom tier
        self.ax.tick_params(labelbottom=False)
        for tier_ax in self.tier_axes[:-1]:
            tier_ax.tick_params(labelbottom=False)
        # Bottom tier shows time axis
        self.tier_axes[-1].set_xlabel("Time (s)", color="#333333", fontsize=9)
        self.tier_axes[-1].tick_params(colors="#555555", labelsize=8)

    # -------------------------------------------------------------------
    # TextGrid editing helpers
    # -------------------------------------------------------------------

    def _tier_index_for_axes(self, axes):
        """Return the tier index for a given axes, or None."""
        for i, ta in enumerate(self.tier_axes):
            if ta is axes:
                return i
        return None

    def _find_nearest_boundary(self, tier_idx, time):
        """Return (boundary_time, distance) for the nearest boundary in tier."""
        tier = self.textgrid_data.tiers[tier_idx]
        best_time = None
        best_dist = float('inf')

        if tier.tier_class == "IntervalTier":
            for iv in tier.intervals:
                for bt in (iv.xmin, iv.xmax):
                    d = abs(bt - time)
                    if d < best_dist:
                        best_dist = d
                        best_time = bt
        elif tier.tier_class == "TextTier":
            for pt in tier.points:
                d = abs(pt.time - time)
                if d < best_dist:
                    best_dist = d
                    best_time = pt.time

        return best_time, best_dist

    def _time_threshold_for_pixels(self, pixels=5):
        """Convert a pixel distance to time units for hit detection."""
        if self.ax.get_xlim()[1] == self.ax.get_xlim()[0]:
            return 0.01
        ax_width_px = self.ax.get_window_extent().width
        view_width = self.view_end - self.view_start
        if ax_width_px <= 0:
            return 0.01
        return pixels * view_width / ax_width_px

    def _add_boundary(self, tier_idx, time):
        """Add a boundary at the given time in the specified tier."""
        tier = self.textgrid_data.tiers[tier_idx]

        if tier.tier_class == "IntervalTier":
            # Find the interval containing this time
            for i, iv in enumerate(tier.intervals):
                if iv.xmin < time < iv.xmax:
                    # Split: left keeps text, right gets ""
                    old_text = iv.text
                    tier.intervals[i] = Interval(iv.xmin, time, old_text)
                    tier.intervals.insert(i + 1, Interval(time, iv.xmax, ""))
                    return True
        elif tier.tier_class == "TextTier":
            # Insert point in sorted order
            pt = Point(time, "")
            for i, existing in enumerate(tier.points):
                if time < existing.time:
                    tier.points.insert(i, pt)
                    return True
            tier.points.append(pt)
            return True

        return False

    def _move_boundary(self, tier_idx, old_time, new_time):
        """Move a boundary from old_time to new_time in the specified tier."""
        tier = self.textgrid_data.tiers[tier_idx]

        if tier.tier_class == "IntervalTier":
            for iv in tier.intervals:
                if iv.xmax == old_time:
                    iv.xmax = new_time
                elif iv.xmin == old_time:
                    iv.xmin = new_time
        elif tier.tier_class == "TextTier":
            for pt in tier.points:
                if pt.time == old_time:
                    pt.time = new_time
                    break

    def _delete_boundary(self, tier_idx, boundary_time):
        """Delete a boundary, merging adjacent intervals (left keeps text)."""
        tier = self.textgrid_data.tiers[tier_idx]

        if tier.tier_class == "IntervalTier":
            # Can't delete tier start/end boundaries
            if boundary_time == tier.xmin or boundary_time == tier.xmax:
                return False
            # Find the two intervals sharing this boundary
            left_idx = None
            right_idx = None
            for i, iv in enumerate(tier.intervals):
                if iv.xmax == boundary_time:
                    left_idx = i
                if iv.xmin == boundary_time:
                    right_idx = i
            if left_idx is not None and right_idx is not None:
                # Merge: extend left to cover right, keep left's text
                tier.intervals[left_idx] = Interval(
                    tier.intervals[left_idx].xmin,
                    tier.intervals[right_idx].xmax,
                    tier.intervals[left_idx].text,
                )
                tier.intervals.pop(right_idx)
                return True
        elif tier.tier_class == "TextTier":
            for i, pt in enumerate(tier.points):
                if pt.time == boundary_time:
                    tier.points.pop(i)
                    return True

        return False

    def _compute_drag_constraints(self, tier_idx, boundary_time):
        """Compute min/max times for dragging a boundary."""
        tier = self.textgrid_data.tiers[tier_idx]
        epsilon = 0.001

        if tier.tier_class == "IntervalTier":
            # Find neighbors
            prev_boundary = tier.xmin
            next_boundary = tier.xmax
            for iv in tier.intervals:
                if iv.xmax < boundary_time and iv.xmax > prev_boundary:
                    prev_boundary = iv.xmax
                if iv.xmin > boundary_time and iv.xmin < next_boundary:
                    next_boundary = iv.xmin
            self._drag_min_time = prev_boundary + epsilon
            self._drag_max_time = next_boundary - epsilon
        elif tier.tier_class == "TextTier":
            # Points can move freely within tier bounds
            prev_time = tier.xmin
            next_time = tier.xmax
            for pt in tier.points:
                if pt.time < boundary_time and pt.time > prev_time:
                    prev_time = pt.time
                if pt.time > boundary_time and pt.time < next_time:
                    next_time = pt.time
            self._drag_min_time = prev_time + epsilon
            self._drag_max_time = next_time - epsilon

    # -------------------------------------------------------------------
    # Mouse scroll — zoom
    # -------------------------------------------------------------------

    def _on_scroll(self, event):
        """Mouse wheel zoom, centered on cursor position."""
        if self.sound is None:
            return
        # Allow scroll on spectrogram axes or any tier axes
        valid_axes = [self.ax] + self.tier_axes
        if event.inaxes not in valid_axes:
            return

        if event.button == "up":
            self.zoom(1.0 / ZOOM_FACTOR, center_time=event.xdata)
        elif event.button == "down":
            self.zoom(ZOOM_FACTOR, center_time=event.xdata)

        self._render_timer.start()  # debounced render

        # Notify the main window to update scrollbar
        if hasattr(self, "_on_view_changed_callback"):
            self._on_view_changed_callback()

    # -------------------------------------------------------------------
    # Mouse interaction for formant editing
    # -------------------------------------------------------------------

    def _select_interval(self, tier_idx, click_time):
        """Select the interval or point containing/nearest to click_time."""
        tier = self.textgrid_data.tiers[tier_idx]

        if tier.tier_class == "IntervalTier":
            for i, iv in enumerate(tier.intervals):
                if iv.xmin <= click_time <= iv.xmax:
                    self._selected_interval = (tier_idx, i)
                    self._selected_boundary = None
                    self._selection_start = iv.xmin
                    self._selection_end = iv.xmax
                    # Populate label editor
                    if self._label_edit is not None:
                        self._label_edit.setEnabled(True)
                        self._label_edit.setText(iv.text)
                    return True
        elif tier.tier_class == "TextTier":
            # Find nearest point
            threshold = self._time_threshold_for_pixels(10)
            best_pt_idx = None
            best_dist = float('inf')
            for i, pt in enumerate(tier.points):
                d = abs(pt.time - click_time)
                if d < best_dist:
                    best_dist = d
                    best_pt_idx = i
            if best_pt_idx is not None and best_dist <= threshold:
                self._selected_interval = (tier_idx, best_pt_idx)
                self._selected_boundary = None
                pt = tier.points[best_pt_idx]
                self._selection_start = pt.time
                self._selection_end = pt.time
                if self._label_edit is not None:
                    self._label_edit.setEnabled(True)
                    self._label_edit.setText(pt.mark)
                return True
        return False

    def _clear_selection(self):
        """Clear interval/boundary/time selection."""
        self._selected_interval = None
        self._selected_boundary = None
        self._selection_start = None
        self._selection_end = None
        if self._label_edit is not None:
            self._label_edit.setEnabled(False)
            self._label_edit.clear()

    def _on_mouse_press(self, event):
        if event.button != 1:
            return

        # --- TextGrid tier click detection (works in ALL modes) ---
        if self.textgrid_data is not None:
            tier_idx = self._tier_index_for_axes(event.inaxes)
            if tier_idx is not None and event.xdata is not None:
                old_active = self._active_tier
                self._active_tier = tier_idx  # always set active tier

                if event.dblclick:
                    # Double-click: select interval + focus label editor
                    self._select_interval(tier_idx, event.xdata)
                    if self._label_edit is not None:
                        self._label_edit.setFocus()
                        self._label_edit.selectAll()
                    if old_active != tier_idx:
                        self.render()
                    else:
                        self._draw_selection_overlay()
                    return

                # Check if clicking on a shadow boundary circle (other tiers)
                if self._selected_boundary is not None:
                    sel_tier, sel_time = self._selected_boundary
                    if tier_idx != sel_tier:
                        threshold = self._time_threshold_for_pixels(8)
                        if abs(event.xdata - sel_time) <= threshold:
                            # Create aligned boundary at shadow time
                            if self._add_boundary(tier_idx, sel_time):
                                self._selected_boundary = (tier_idx, sel_time)
                                self._selected_interval = None
                                self.render()
                            return

                # Single click: check if near a boundary
                tier = self.textgrid_data.tiers[tier_idx]
                bt, dist = self._find_nearest_boundary(tier_idx, event.xdata)
                threshold = self._time_threshold_for_pixels(5)

                # For TextTier, single-click always selects the point
                # for label editing; drag is initiated separately
                if tier.tier_class == "TextTier":
                    if bt is not None and dist <= threshold:
                        # Select point for label editing + prepare drag
                        self._select_interval(tier_idx, event.xdata)
                        self._selected_boundary = None
                        self._dragging_boundary = True
                        self._drag_tier_index = tier_idx
                        self._drag_original_time = bt
                        self._compute_drag_constraints(tier_idx, bt)
                        self.render()
                        self._drag_bg = self.fig.canvas.copy_from_bbox(
                            self.fig.bbox)
                        self._drag_line_spec = self.ax.axvline(
                            bt, color="#ff0000", linewidth=2.5,
                            alpha=0.9, zorder=6,
                        )
                        self._drag_line_tier = self.tier_axes[tier_idx].axvline(
                            bt, color="#ff0000", linewidth=2.5,
                            alpha=0.9, zorder=6,
                        )
                        self.ax.draw_artist(self._drag_line_spec)
                        self.tier_axes[tier_idx].draw_artist(self._drag_line_tier)
                        self.fig.canvas.blit(self.fig.bbox)
                    else:
                        # Click away from any point: just set active tier
                        self._selected_interval = None
                        self._selected_boundary = None
                        if self._label_edit is not None:
                            self._label_edit.setEnabled(False)
                            self._label_edit.clear()
                        if old_active != tier_idx:
                            self.render()
                        else:
                            self._draw_selection_overlay()
                    return

                # IntervalTier boundary/interval click
                if bt is not None and dist <= threshold:
                    # Select boundary
                    self._selected_interval = None
                    if self._label_edit is not None:
                        self._label_edit.setEnabled(False)
                        self._label_edit.clear()
                    if tier.tier_class == "IntervalTier" and (bt == tier.xmin or bt == tier.xmax):
                        # Can't drag tier start/end
                        self._selected_boundary = (tier_idx, bt)
                        if old_active != tier_idx:
                            self.render()
                        else:
                            self._draw_selection_overlay()
                        return
                    # Select and prepare for drag (blitting)
                    self._selected_boundary = None  # render without highlight
                    self._selected_interval = None
                    self._dragging_boundary = True
                    self._drag_tier_index = tier_idx
                    self._drag_original_time = bt
                    self._compute_drag_constraints(tier_idx, bt)
                    self.render()  # clean background (no red line)
                    # Cache bg BEFORE creating overlay artists
                    self._drag_bg = self.fig.canvas.copy_from_bbox(
                        self.fig.bbox)
                    # Create movable line artists
                    self._drag_line_spec = self.ax.axvline(
                        bt, color="#ff0000", linewidth=2.5,
                        alpha=0.9, zorder=6,
                    )
                    self._drag_line_tier = self.tier_axes[tier_idx].axvline(
                        bt, color="#ff0000", linewidth=2.5,
                        alpha=0.9, zorder=6,
                    )
                    # Show initial position via blit
                    self.ax.draw_artist(self._drag_line_spec)
                    self.tier_axes[tier_idx].draw_artist(self._drag_line_tier)
                    self.fig.canvas.blit(self.fig.bbox)
                    return
                else:
                    # Click in interval body: select interval
                    self._select_interval(tier_idx, event.xdata)
                    if old_active != tier_idx:
                        self.render()
                    else:
                        self._draw_selection_overlay()
                    return

        # --- Spectrogram click/drag (non-edit mode) ---
        if not self.edit_mode:
            if event.inaxes == self.ax and event.xdata is not None:
                self._click_time = event.xdata
                # Clear interval/boundary selection but keep _active_tier
                self._selected_interval = None
                self._selected_boundary = None
                self._selection_start = None
                self._selection_end = None
                if self._label_edit is not None:
                    self._label_edit.setEnabled(False)
                    self._label_edit.clear()
                if self._crosshair_visible:
                    self._hide_crosshair()
                self._spec_dragging = True
                self._spec_drag_start = event.xdata
                self._drag_bg = None  # will be set up on first move
                if self._status_callback:
                    self._status_callback(
                        f"Time: {event.xdata:.4f} s  |  "
                        f"Frequency: {event.ydata:.1f} Hz"
                    )
            return

        # --- Edit mode: formant drawing (only on spectrogram axes) ---
        if event.inaxes != self.ax:
            return
        self.is_drawing = True
        self._last_frame_idx = -1
        self._stroke_times = []
        self._stroke_freqs = []

        # Cache background for blitting
        self.fig.canvas.draw()
        self._bg_cache = self.fig.canvas.copy_from_bbox(self.ax.bbox)

        # Create a scatter artist for the stroke
        fn = self.active_formant + 1
        color = FORMANT_COLORS.get(fn, "#ffffff")
        self._stroke_scatter = self.ax.scatter(
            [], [], c=color, s=12, edgecolors="white",
            linewidths=0.3, zorder=5,
        )

        self._apply_edit(event.xdata, event.ydata)

    def _on_mouse_release(self, event):
        # --- End boundary drag (blit-based) ---
        if self._dragging_boundary:
            # Compute final position
            final_time = self._drag_original_time
            if event.xdata is not None:
                final_time = max(self._drag_min_time,
                                 min(self._drag_max_time, event.xdata))
            # Commit data model change once
            self._move_boundary(self._drag_tier_index,
                                self._drag_original_time, final_time)
            self._selected_boundary = (self._drag_tier_index, final_time)
            # Clean up blit artists
            if self._drag_line_spec is not None:
                self._drag_line_spec.remove()
                self._drag_line_spec = None
            if self._drag_line_tier is not None:
                self._drag_line_tier.remove()
                self._drag_line_tier = None
            self._drag_bg = None
            self._dragging_boundary = False
            self._drag_tier_index = None
            self._drag_original_time = None
            self.render()
            return

        # --- End spectrogram drag selection (blit-based) ---
        if self._spec_dragging:
            self._spec_dragging = False
            # Clean up blit artists
            if self._spec_sel_artist is not None:
                self._spec_sel_artist.remove()
                self._spec_sel_artist = None
            for a in self._spec_sel_tier_artists:
                a.remove()
            self._spec_sel_tier_artists = []
            self._drag_bg = None
            # Determine if real drag or just a click
            if event.xdata is not None and self._spec_drag_start is not None:
                drag_dist = abs(event.xdata - self._spec_drag_start)
                threshold = self._time_threshold_for_pixels(5)
                if drag_dist > threshold:
                    self._selection_start = min(self._spec_drag_start,
                                                event.xdata)
                    self._selection_end = max(self._spec_drag_start,
                                              event.xdata)
                else:
                    self._selection_start = None
                    self._selection_end = None
            self._spec_drag_start = None
            self._draw_selection_overlay()
            return

        if not self.is_drawing:
            return
        self.is_drawing = False
        self._last_frame_idx = -1

        # Clean up stroke artist
        if self._stroke_scatter is not None:
            self._stroke_scatter.remove()
            self._stroke_scatter = None
        self._stroke_times = []
        self._stroke_freqs = []
        self._bg_cache = None

        if self.edit_mode:
            self.render()

    def _on_mouse_move(self, event):
        # --- Boundary drag (blit-based) ---
        if self._dragging_boundary and event.xdata is not None:
            new_time = max(self._drag_min_time,
                           min(self._drag_max_time, event.xdata))
            # Update visual only — no data mutation until release
            if self._drag_bg is not None:
                self.fig.canvas.restore_region(self._drag_bg)
                if self._drag_line_spec is not None:
                    self._drag_line_spec.set_xdata([new_time])
                    self.ax.draw_artist(self._drag_line_spec)
                if self._drag_line_tier is not None:
                    self._drag_line_tier.set_xdata([new_time])
                    ta = self.tier_axes[self._drag_tier_index]
                    ta.draw_artist(self._drag_line_tier)
                self.fig.canvas.blit(self.fig.bbox)
            return

        # --- Spectrogram drag selection (blit-based, lazy init) ---
        if self._spec_dragging and event.xdata is not None:
            # Lazy init: render + create artists on first move
            if self._drag_bg is None:
                self.render()  # clean background
                self._drag_bg = self.fig.canvas.copy_from_bbox(
                    self.fig.bbox)
                ymin, ymax = self.ax.get_ylim()
                self._spec_sel_artist = MplRectangle(
                    (self._spec_drag_start, ymin), 0, ymax - ymin,
                    color="#3366cc", alpha=0.25, zorder=5,
                )
                self.ax.add_patch(self._spec_sel_artist)
                self._spec_sel_tier_artists = []
                for tier_ax in self.tier_axes:
                    r = MplRectangle(
                        (self._spec_drag_start, 0), 0, 1,
                        color="#3366cc", alpha=0.15, zorder=0,
                    )
                    tier_ax.add_patch(r)
                    self._spec_sel_tier_artists.append(r)
            # Update rectangles
            x0 = min(self._spec_drag_start, event.xdata)
            x1 = max(self._spec_drag_start, event.xdata)
            self._spec_sel_artist.set_x(x0)
            self._spec_sel_artist.set_width(x1 - x0)
            for a in self._spec_sel_tier_artists:
                a.set_x(x0)
                a.set_width(x1 - x0)
            # Blit
            self.fig.canvas.restore_region(self._drag_bg)
            self.ax.draw_artist(self._spec_sel_artist)
            for a, ta in zip(self._spec_sel_tier_artists,
                             self.tier_axes):
                ta.draw_artist(a)
            self.fig.canvas.blit(self.fig.bbox)
            return

        # --- Crosshair + readout (always active over spectrogram, not during playback) ---
        if self._playback_playing:
            return
        if event.inaxes == self.ax and event.xdata is not None:
            self._hover_time = event.xdata
            self._update_crosshair(event.xdata, event.ydata)
            if self._status_callback and not self.is_drawing:
                self._status_callback(
                    f"Time: {event.xdata:.4f} s  |  "
                    f"Frequency: {event.ydata:.1f} Hz"
                )
        elif self._crosshair_visible:
            self._hide_crosshair()

        # --- Formant editing (only while drawing) ---
        if not self.is_drawing or not self.edit_mode:
            return
        if event.inaxes != self.ax or event.xdata is None:
            return
        self._apply_edit(event.xdata, event.ydata)

    def _update_crosshair(self, xdata, ydata):
        """Move crosshair lines to cursor position via blitting."""
        if self._crosshair_h is None or self._crosshair_bg is None:
            return
        self._crosshair_h.set_ydata([ydata])
        self._crosshair_v.set_xdata([xdata])
        if not self._crosshair_visible:
            self._crosshair_h.set_visible(True)
            self._crosshair_v.set_visible(True)
            self._crosshair_visible = True
        self.fig.canvas.restore_region(self._crosshair_bg)
        self.ax.draw_artist(self._crosshair_h)
        self.ax.draw_artist(self._crosshair_v)
        self.fig.canvas.blit(self.ax.bbox)

    def _hide_crosshair(self):
        """Hide crosshair when cursor leaves the spectrogram."""
        if self._crosshair_h is None:
            return
        self._crosshair_h.set_visible(False)
        self._crosshair_v.set_visible(False)
        self._crosshair_visible = False
        if self._crosshair_bg is not None:
            self.fig.canvas.restore_region(self._crosshair_bg)
            self.fig.canvas.blit(self.ax.bbox)

    # -------------------------------------------------------------------
    # Audio playback
    # -------------------------------------------------------------------

    def play_audio(self, start_time, end_time):
        """Play audio from start_time to end_time with animated cursor."""
        if self.sound is None:
            return

        # Stop any current playback
        if self._playback_playing:
            self.stop_playback()

        # Extract samples for the requested range
        sr = int(self.sound.sampling_frequency)
        samples = self.sound.values[0]  # mono float64
        start_sample = max(0, int(start_time * sr))
        end_sample = min(len(samples), int(end_time * sr))
        if end_sample <= start_sample:
            return

        audio_chunk = samples[start_sample:end_sample].copy()

        self._playback_start_time = start_time
        self._playback_end_time = end_time
        self._playback_playing = True

        # Start audio
        sd.play(audio_chunk, sr)
        self._playback_start_wall = _time.monotonic()

        # Create green cursor line
        self._playback_cursor = self.ax.axvline(
            x=start_time, color="#00ff44", linewidth=1.5,
            alpha=0.9, zorder=15,
        )
        # Also add cursor lines to tier axes
        self._playback_tier_cursors = []
        for tier_ax in self.tier_axes:
            tc = tier_ax.axvline(
                x=start_time, color="#00ff44", linewidth=1.5,
                alpha=0.9, zorder=15,
            )
            self._playback_tier_cursors.append(tc)

        self.fig.canvas.draw()
        self._playback_cursor_bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)

        # Start timer for cursor animation (30ms interval)
        self._playback_timer = QTimer()
        self._playback_timer.timeout.connect(self._update_playback_cursor)
        self._playback_timer.start(30)

    def stop_playback(self):
        """Stop audio playback and clean up cursor."""
        if not self._playback_playing:
            return

        sd.stop()
        self._playback_playing = False

        if self._playback_timer is not None:
            self._playback_timer.stop()
            self._playback_timer = None

        # Remove cursor artists
        if self._playback_cursor is not None:
            self._playback_cursor.remove()
            self._playback_cursor = None
        for tc in getattr(self, '_playback_tier_cursors', []):
            tc.remove()
        self._playback_tier_cursors = []
        self._playback_cursor_bg = None

        self.render()

    def _update_playback_cursor(self):
        """Timer callback: update playback cursor position via blitting."""
        if not self._playback_playing:
            return

        elapsed = _time.monotonic() - self._playback_start_wall
        current_time = self._playback_start_time + elapsed

        if current_time >= self._playback_end_time:
            self.stop_playback()
            return

        # Update cursor position
        self._playback_cursor.set_xdata([current_time])
        for tc in self._playback_tier_cursors:
            tc.set_xdata([current_time])

        # Blit
        if self._playback_cursor_bg is not None:
            self.fig.canvas.restore_region(self._playback_cursor_bg)
            self.ax.draw_artist(self._playback_cursor)
            for tc, tier_ax in zip(self._playback_tier_cursors, self.tier_axes):
                tier_ax.draw_artist(tc)
            self.fig.canvas.blit(self.fig.bbox)

    def _apply_edit(self, time_s, freq_hz):
        """Apply an edit at the given time/frequency position."""
        if self.formant_data is None:
            return

        fd = self.formant_data
        # Find nearest frame
        frame_idx = np.argmin(np.abs(fd.times - time_s))

        # Avoid re-editing the exact same frame in one drag
        if frame_idx == self._last_frame_idx:
            return
        self._last_frame_idx = frame_idx

        fd.set_value(self.active_formant, frame_idx, freq_hz)

        # Quick incremental draw via blitting
        self._quick_draw_edit_point(frame_idx, freq_hz)

    def _quick_draw_edit_point(self, frame_idx, freq_hz):
        """Draw edited points using blitting for responsiveness."""
        if self._bg_cache is None or self._stroke_scatter is None:
            return

        t = self.formant_data.times[frame_idx]
        self._stroke_times.append(t)
        self._stroke_freqs.append(freq_hz)

        # Update the scatter artist with all stroke points
        offsets = np.column_stack([self._stroke_times, self._stroke_freqs])
        self._stroke_scatter.set_offsets(offsets)

        # Blit: restore background, draw artist, update canvas
        self.fig.canvas.restore_region(self._bg_cache)
        self.ax.draw_artist(self._stroke_scatter)
        self.fig.canvas.blit(self.ax.bbox)


# ---------------------------------------------------------------------------
# Control Panel (sliders, settings)
# ---------------------------------------------------------------------------

class ControlPanel(QWidget):
    """Side panel with spectrogram and formant controls."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(260)
        self.setStyleSheet("""
            QWidget { background-color: #252530; color: #cccccc; }
            QGroupBox {
                font-weight: bold; border: 1px solid #444455;
                border-radius: 4px; margin-top: 8px; padding-top: 14px;
            }
            QGroupBox::title { padding: 0 6px; }
            QSlider::groove:horizontal {
                height: 6px; background: #444455; border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #6699cc; width: 14px; margin: -4px 0;
                border-radius: 7px;
            }
            QPushButton {
                background-color: #334455; border: 1px solid #556677;
                border-radius: 4px; padding: 6px 12px; color: #cccccc;
            }
            QPushButton:hover { background-color: #445566; }
            QPushButton:checked { background-color: #cc5544; border-color: #ee7766; }
            QLabel { font-size: 11px; }
            QComboBox, QDoubleSpinBox {
                background-color: #334455; border: 1px solid #556677;
                border-radius: 3px; padding: 3px; color: #cccccc;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # --- Spectrogram Display ---
        spec_group = QGroupBox("Spectrogram Display")
        spec_layout = QVBoxLayout(spec_group)

        self.dynamic_range_slider = self._make_slider(
            "Dynamic Range (dB)", 10, 100, int(DEFAULT_DYNAMIC_RANGE), spec_layout
        )
        self.brightness_slider = self._make_slider(
            "Brightness", -30, 30, 0, spec_layout
        )

        # Spectrogram window length (ms for display, stored as seconds)
        self.spec_window_slider = self._make_slider(
            "Window (ms)", 1, 20, int(DEFAULT_SPEC_WINDOW * 1000), spec_layout
        )

        # Max frequency
        freq_row = QHBoxLayout()
        freq_row.addWidget(QLabel("Max Freq (Hz):"))
        self.max_freq_spin = QDoubleSpinBox()
        self.max_freq_spin.setRange(1000, 12000)
        self.max_freq_spin.setValue(DEFAULT_SPEC_MAX_FREQ)
        self.max_freq_spin.setSingleStep(500)
        freq_row.addWidget(self.max_freq_spin)
        spec_layout.addLayout(freq_row)

        layout.addWidget(spec_group)

        # --- Formant Settings ---
        fmt_group = QGroupBox("Formant Analysis")
        fmt_layout = QVBoxLayout(fmt_group)

        # Number of formants to display
        nf_row = QHBoxLayout()
        nf_row.addWidget(QLabel("Show formants:"))
        self.num_formants_combo = QComboBox()
        for n in range(1, 6):
            self.num_formants_combo.addItem(f"F1–F{n}", n)
        self.num_formants_combo.setCurrentIndex(2)  # F1–F3 default
        nf_row.addWidget(self.num_formants_combo)
        fmt_layout.addLayout(nf_row)

        # Max formant for analysis
        mf_row = QHBoxLayout()
        mf_row.addWidget(QLabel("Max formant (Hz):"))
        self.max_formant_spin = QDoubleSpinBox()
        self.max_formant_spin.setRange(3000, 8000)
        self.max_formant_spin.setValue(MAX_FORMANT_HZ)
        self.max_formant_spin.setSingleStep(500)
        mf_row.addWidget(self.max_formant_spin)
        fmt_layout.addLayout(mf_row)

        # Window length
        wl_row = QHBoxLayout()
        wl_row.addWidget(QLabel("Window (s):"))
        self.window_length_spin = QDoubleSpinBox()
        self.window_length_spin.setRange(0.005, 0.1)
        self.window_length_spin.setValue(DEFAULT_WINDOW_LENGTH)
        self.window_length_spin.setSingleStep(0.005)
        self.window_length_spin.setDecimals(3)
        wl_row.addWidget(self.window_length_spin)
        fmt_layout.addLayout(wl_row)

        # Pre-emphasis (affects formant analysis, not spectrogram)
        self.contrast_slider = self._make_slider(
            "Pre-emphasis (Hz)", 1, 200, int(DEFAULT_PRE_EMPHASIS), fmt_layout
        )

        # Re-analyse button
        self.reanalyse_btn = QPushButton("Re-analyse Formants")
        fmt_layout.addWidget(self.reanalyse_btn)

        # Show/hide formants
        self.show_formants_cb = QCheckBox("Show formants")
        self.show_formants_cb.setChecked(True)
        fmt_layout.addWidget(self.show_formants_cb)

        layout.addWidget(fmt_group)

        # --- Edit Mode ---
        edit_group = QGroupBox("Formant Editing")
        edit_layout = QVBoxLayout(edit_group)

        self.edit_btn = QPushButton("✏  EDIT MODE")
        self.edit_btn.setCheckable(True)
        self.edit_btn.setStyleSheet("""
            QPushButton {
                font-weight: bold; font-size: 13px;
                padding: 8px; background-color: #334455;
            }
            QPushButton:checked {
                background-color: #cc4433; color: white;
            }
        """)
        edit_layout.addWidget(self.edit_btn)

        # Active formant indicator
        self.active_formant_label = QLabel("Active: F1")
        self.active_formant_label.setStyleSheet(
            f"font-weight: bold; font-size: 14px; color: {FORMANT_COLORS[1]};"
        )
        self.active_formant_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        edit_layout.addWidget(self.active_formant_label)

        edit_layout.addWidget(QLabel("Press F1–F5 to select formant"))

        # Reset buttons
        self.reset_current_btn = QPushButton("Reset Current Formant")
        self.reset_all_btn = QPushButton("Reset All Edits")
        edit_layout.addWidget(self.reset_current_btn)
        edit_layout.addWidget(self.reset_all_btn)

        layout.addWidget(edit_group)

        layout.addStretch()

    def _make_slider(self, label_text, min_val, max_val, default, parent_layout):
        lbl = QLabel(f"{label_text}: {default}")
        parent_layout.addWidget(lbl)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default)
        slider.valueChanged.connect(lambda v, l=lbl, t=label_text: l.setText(f"{t}: {v}"))
        parent_layout.addWidget(slider)
        return slider

    def update_active_formant_display(self, formant_idx):
        """Update the active formant label (0-indexed input)."""
        fn = formant_idx + 1
        color = FORMANT_COLORS.get(fn, "#ffffff")
        self.active_formant_label.setText(f"Active: {FORMANT_LABELS[fn]}")
        self.active_formant_label.setStyleSheet(
            f"font-weight: bold; font-size: 14px; color: {color};"
        )


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------

class _TabPlayFilter(QObject):
    """Application event filter: intercepts Tab for audio playback."""

    def __init__(self, main_window):
        super().__init__(main_window)
        self._mw = main_window

    def eventFilter(self, obj, event):
        if (event.type() == QEvent.Type.KeyPress
                and event.key() == Qt.Key.Key_Tab
                and self._mw.isActiveWindow()):
            self._mw._play_selection()
            return True
        return False


# ---------------------------------------------------------------------------
# Create TextGrid dialog
# ---------------------------------------------------------------------------

_TIER_TYPE_DISPLAY = {"Interval Tier": "IntervalTier", "Point Tier": "TextTier"}
_TIER_TYPE_LABELS = list(_TIER_TYPE_DISPLAY.keys())

_DIALOG_FIELD_STYLE = """
    QLineEdit, QComboBox {
        background-color: #ffffff; color: #000000;
    }
    QComboBox QAbstractItemView {
        background-color: #ffffff; color: #000000;
        selection-background-color: #d0d0d0; selection-color: #000000;
    }
"""


class CreateTextGridDialog(QDialog):
    """Dialog for creating a new TextGrid with user-defined tiers."""

    def __init__(self, duration, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create TextGrid")
        self.setMinimumWidth(400)
        self.setStyleSheet(_DIALOG_FIELD_STYLE)
        self._duration = duration
        self._tier_rows = []

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"No TextGrid found. Create one for this file?\n"
                                f"Duration: {duration:.3f} s"))

        # Tier list area
        self._tier_layout = QVBoxLayout()
        layout.addLayout(self._tier_layout)

        # Add initial default tier
        self._add_tier_row("phones", "Interval Tier")

        # Add Tier button
        add_btn = QPushButton("+ Add Tier")
        add_btn.clicked.connect(lambda: self._add_tier_row())
        layout.addWidget(add_btn)

        # OK/Cancel
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _add_tier_row(self, name="", tier_class="Interval Tier"):
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)

        name_edit = QLineEdit(name)
        name_edit.setPlaceholderText("Tier name")
        type_combo = QComboBox()
        type_combo.addItems(_TIER_TYPE_LABELS)
        type_combo.setCurrentText(tier_class)
        remove_btn = QPushButton("Remove")

        row_layout.addWidget(name_edit, 1)
        row_layout.addWidget(type_combo)
        row_layout.addWidget(remove_btn)

        self._tier_layout.addWidget(row_widget)
        row_data = {"widget": row_widget, "name": name_edit, "type": type_combo}
        self._tier_rows.append(row_data)

        remove_btn.clicked.connect(lambda: self._remove_tier_row(row_data))
        self._update_remove_buttons()

    def _remove_tier_row(self, row_data):
        if len(self._tier_rows) <= 1:
            return
        self._tier_rows.remove(row_data)
        row_data["widget"].deleteLater()
        self._update_remove_buttons()

    def _update_remove_buttons(self):
        enabled = len(self._tier_rows) > 1
        for row in self._tier_rows:
            # Find the remove button (last widget in layout)
            layout = row["widget"].layout()
            btn = layout.itemAt(layout.count() - 1).widget()
            btn.setEnabled(enabled)

    def get_textgrid(self):
        """Build a TextGrid from the dialog's current state."""
        tiers = []
        for row in self._tier_rows:
            name = row["name"].text().strip() or f"tier{len(tiers) + 1}"
            tc = _TIER_TYPE_DISPLAY[row["type"].currentText()]
            if tc == "IntervalTier":
                tier = Tier(name, tc, 0, self._duration,
                            intervals=[Interval(0, self._duration, "")])
            else:
                tier = Tier(name, tc, 0, self._duration, points=[])
            tiers.append(tier)
        return TextGrid(0, self._duration, tiers)


# ---------------------------------------------------------------------------
# Add Tier dialog
# ---------------------------------------------------------------------------

class AddTierDialog(QDialog):
    """Dialog for adding a tier to an existing TextGrid."""

    def __init__(self, textgrid, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Tier")
        self.setMinimumWidth(350)
        self.setStyleSheet(_DIALOG_FIELD_STYLE)
        self._tg = textgrid

        layout = QFormLayout(self)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Tier name")
        layout.addRow("Name:", self.name_edit)

        self.type_combo = QComboBox()
        self.type_combo.addItems(_TIER_TYPE_LABELS)
        layout.addRow("Type:", self.type_combo)

        self.position_combo = QComboBox()
        for i, tier in enumerate(textgrid.tiers):
            self.position_combo.addItem(f"Before \"{tier.name}\"", i)
        self.position_combo.addItem(
            f"After \"{textgrid.tiers[-1].name}\"" if textgrid.tiers
            else "At position 0",
            len(textgrid.tiers)
        )
        # Default to after last tier
        self.position_combo.setCurrentIndex(self.position_combo.count() - 1)
        layout.addRow("Position:", self.position_combo)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def get_tier(self):
        """Build a new Tier from the dialog's current state."""
        name = self.name_edit.text().strip() or "new_tier"
        tc = _TIER_TYPE_DISPLAY[self.type_combo.currentText()]
        xmin = self._tg.xmin
        xmax = self._tg.xmax
        if tc == "IntervalTier":
            return Tier(name, tc, xmin, xmax,
                        intervals=[Interval(xmin, xmax, "")])
        return Tier(name, tc, xmin, xmax, points=[])

    def get_position(self):
        return self.position_combo.currentData()


class MainWindow(QMainWindow):
    """FormantStudio main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("FormantStudio — Manual Formant Editor")
        self.setMinimumSize(1100, 600)
        self.resize(1400, 700)

        # Dark theme
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e2e; }
            QMenuBar { background-color: #252535; color: #cccccc; }
            QMenuBar::item:selected { background-color: #445566; }
            QStatusBar { background-color: #252535; color: #999999; }
        """)

        self._filepath = None
        self._textgrid_path = None
        self._scrollbar_updating = False

        self._setup_ui()
        self._setup_menu()
        self._connect_signals()

        # Debounce timer for max-frequency spinner (recomputes spectrogram)
        self._max_freq_timer = QTimer()
        self._max_freq_timer.setSingleShot(True)
        self._max_freq_timer.setInterval(200)
        self._max_freq_timer.timeout.connect(self._do_max_freq_update)

        # Application-level Tab filter for playback
        self._tab_filter = _TabPlayFilter(self)
        QApplication.instance().installEventFilter(self._tab_filter)

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Canvas + scrollbar in a vertical layout
        canvas_container = QWidget()
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.setSpacing(0)

        self.canvas = SpectrogramCanvas()
        self.canvas._on_view_changed_callback = self._update_scrollbar
        self.canvas._status_callback = lambda msg: self.status.showMessage(msg)

        # Inline label editor
        self.label_edit = LabelEdit()
        self.label_edit._play_callback = self._play_selection
        self.label_edit.textChanged.connect(self._on_label_text_changed)
        self.label_edit.escape_pressed.connect(self._on_label_escape)
        self.canvas._label_edit = self.label_edit

        self.scrollbar = QScrollBar(Qt.Orientation.Horizontal)
        self.scrollbar.setStyleSheet("""
            QScrollBar:horizontal {
                background: #1a1a2e; height: 14px;
                border: none;
            }
            QScrollBar::handle:horizontal {
                background: #556677; border-radius: 4px;
                min-width: 30px;
            }
            QScrollBar::add-line, QScrollBar::sub-line { width: 0; }
        """)

        canvas_layout.addWidget(self.canvas, 1)
        label_row = QHBoxLayout()
        label_row.addStretch(1)
        label_row.addWidget(self.label_edit)
        label_row.addStretch(1)
        canvas_layout.addLayout(label_row)
        canvas_layout.addWidget(self.scrollbar)

        # Control panel
        self.controls = ControlPanel()

        # Splitter for resizability
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(canvas_container)
        splitter.addWidget(self.controls)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

        main_layout.addWidget(splitter)

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready — Open a WAV file to begin (Ctrl+O)")

    def _setup_menu(self):
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        open_action = QAction("&Open WAV...", self)
        open_action.setShortcut(QKeySequence("Ctrl+O"))
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)

        save_action = QAction("&Save Formants", self)
        save_action.setShortcut(QKeySequence("Ctrl+S"))
        save_action.triggered.connect(self.save_formants)
        file_menu.addAction(save_action)

        load_fmt_action = QAction("&Load Formants...", self)
        load_fmt_action.setShortcut(QKeySequence("Ctrl+L"))
        load_fmt_action.triggered.connect(self.load_formants)
        file_menu.addAction(load_fmt_action)

        load_tg_action = QAction("Load &TextGrid...", self)
        load_tg_action.setShortcut(QKeySequence("Ctrl+T"))
        load_tg_action.triggered.connect(lambda: self.load_textgrid())
        file_menu.addAction(load_tg_action)

        save_tg_action = QAction("Save Text&Grid", self)
        save_tg_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        save_tg_action.triggered.connect(self.save_textgrid)
        file_menu.addAction(save_tg_action)

        add_tier_action = QAction("Add &Tier...", self)
        add_tier_action.setShortcut(QKeySequence("Ctrl+Shift+T"))
        add_tier_action.triggered.connect(self._add_tier_to_textgrid)
        file_menu.addAction(add_tier_action)

        file_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence("Ctrl+Q"))
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

    def _connect_signals(self):
        ctrl = self.controls

        # Spectrogram sliders -> re-render
        ctrl.dynamic_range_slider.valueChanged.connect(self._on_spec_setting_changed)
        ctrl.brightness_slider.valueChanged.connect(self._on_spec_setting_changed)
        ctrl.spec_window_slider.sliderReleased.connect(self._on_spec_window_changed)
        ctrl.max_freq_spin.valueChanged.connect(self._on_max_freq_changed)

        # Pre-emphasis -> re-analyse on release
        ctrl.contrast_slider.sliderReleased.connect(self._on_preemphasis_changed)

        # Show formants dropdown -> update display
        ctrl.num_formants_combo.currentIndexChanged.connect(
            self._on_num_formants_display_changed
        )

        # Formant settings
        ctrl.reanalyse_btn.clicked.connect(self._reanalyse_formants)
        ctrl.show_formants_cb.toggled.connect(self._on_show_formants_toggled)

        # Edit mode
        ctrl.edit_btn.toggled.connect(self._on_edit_toggled)
        ctrl.reset_current_btn.clicked.connect(self._reset_current_formant)
        ctrl.reset_all_btn.clicked.connect(self._reset_all_formants)

        # Scrollbar
        self.scrollbar.valueChanged.connect(self._on_scrollbar_changed)

    # -------------------------------------------------------------------
    # Key events — F1–F5 for formant selection
    # -------------------------------------------------------------------

    def _play_selection(self):
        """Play audio for the current selection or view."""
        if self.canvas._playback_playing:
            self.canvas.stop_playback()
            self.status.showMessage("Playback stopped")
            return
        if self.canvas.sound is None:
            return
        # Priority: selection range > click_time to view end > full view
        c = self.canvas
        if c._selection_start is not None and c._selection_end is not None:
            start, end = c._selection_start, c._selection_end
        elif c._click_time is not None:
            start = c._click_time
            end = c.view_end
        else:
            start = c.view_start
            end = c.view_end
        if start >= end:
            start = c.view_start
        c.play_audio(start, end)
        self.status.showMessage(f"Playing {start:.2f}s \u2013 {end:.2f}s")

    def _on_label_text_changed(self, text):
        """Live-update the label in the data model on every keystroke."""
        sel = self.canvas._selected_interval
        if sel is None or self.canvas.textgrid_data is None:
            return
        tier_idx, item_idx = sel
        tier = self.canvas.textgrid_data.tiers[tier_idx]
        if tier.tier_class == "IntervalTier" and item_idx < len(tier.intervals):
            tier.intervals[item_idx].text = text
        elif tier.tier_class == "TextTier" and item_idx < len(tier.points):
            tier.points[item_idx].mark = text
        self.canvas._label_render_timer.start()  # debounced render

    def _on_label_escape(self):
        """Escape pressed in label editor — clear selection."""
        self.canvas._clear_selection()
        self.canvas._draw_selection_overlay()
        self.status.showMessage("Selection cleared")

    def keyPressEvent(self, event):
        key = event.key()
        # F1=0x01000030, F2=0x01000031, ... F5=0x01000034
        if Qt.Key.Key_F1 <= key <= Qt.Key.Key_F5:
            if self.canvas.edit_mode:
                f_idx = key - Qt.Key.Key_F1  # 0-indexed
                n_display = self.controls.num_formants_combo.currentData()
                if f_idx < n_display:
                    self.canvas.active_formant = f_idx
                    self.controls.update_active_formant_display(f_idx)
                    self.canvas.render()
                    self.status.showMessage(
                        f"Now editing {FORMANT_LABELS[f_idx + 1]}"
                    )
                event.accept()
                return

        # Tab is handled by _TabPlayFilter (application event filter)

        # Escape — stop playback, clear selection, defocus label editor
        if key == Qt.Key.Key_Escape:
            if self.canvas._playback_playing:
                self.canvas.stop_playback()
                self.status.showMessage("Playback stopped")
                event.accept()
                return
            if (self.canvas._selected_boundary is not None
                    or self.canvas._selected_interval is not None
                    or self.canvas._selection_start is not None):
                self.canvas._clear_selection()
                self.label_edit.clearFocus()
                self.canvas._draw_selection_overlay()
                self.status.showMessage("Selection cleared")
                event.accept()
                return

        # Enter/Return — add boundary at hover (or click) position on active tier
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if not self.label_edit.hasFocus():
                c = self.canvas
                boundary_time = c._hover_time or c._click_time
                if (c._active_tier is not None and boundary_time is not None
                        and c.textgrid_data is not None):
                    if c._add_boundary(c._active_tier, boundary_time):
                        c._select_interval(c._active_tier, boundary_time)
                        c.render()
                        self.status.showMessage("Boundary added")
                    else:
                        self.status.showMessage("Cannot add boundary here")
                event.accept()
                return

        # Delete — remove selected boundary
        if key == Qt.Key.Key_Delete:
            if self.canvas._selected_boundary is not None:
                tier_idx, bt = self.canvas._selected_boundary
                if self.canvas._delete_boundary(tier_idx, bt):
                    self.canvas._selected_boundary = None
                    self.canvas.render()
                    self.status.showMessage("Boundary deleted")
                else:
                    self.status.showMessage("Cannot delete tier start/end boundary")
                event.accept()
                return

        # F key — toggle formant visibility
        if key == Qt.Key.Key_F and not event.modifiers():
            checked = not self.controls.show_formants_cb.isChecked()
            self.controls.show_formants_cb.setChecked(checked)
            self.status.showMessage(
                "Formants visible" if checked else "Formants hidden (press F to show)"
            )
            event.accept()
            return

        super().keyPressEvent(event)

    # -------------------------------------------------------------------
    # File operations
    # -------------------------------------------------------------------

    def open_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Audio File", "",
            "Audio Files (*.wav *.WAV *.aiff *.AIFF *.mp3);;All Files (*)"
        )
        if not filepath:
            return

        self._filepath = filepath
        self.canvas._filepath = filepath
        self.status.showMessage(f"Loading {os.path.basename(filepath)}...")
        QApplication.processEvents()

        try:
            self.canvas.load_sound(filepath)
            self._run_formant_analysis()
            self._update_scrollbar()
            self.canvas.render()
            self.status.showMessage(
                f"Loaded: {os.path.basename(filepath)} "
                f"({self.canvas.sound.duration:.2f}s, "
                f"{int(self.canvas.sound.sampling_frequency)} Hz)"
            )

            # Check for existing .formants file
            fmt_path = os.path.splitext(filepath)[0] + ".formants"
            if os.path.exists(fmt_path):
                reply = QMessageBox.question(
                    self, "Load saved formants?",
                    f"Found saved formant data:\n{os.path.basename(fmt_path)}\n\nLoad it?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self.canvas.formant_data = FormantData.load(fmt_path)
                    self.canvas.render()
                    self.status.showMessage(
                        f"Loaded formants from {os.path.basename(fmt_path)}"
                    )

            # Auto-detect TextGrid with same basename
            base = os.path.splitext(filepath)[0]
            tg_found = False
            for ext in (".TextGrid", ".textgrid"):
                tg_path = base + ext
                if os.path.exists(tg_path):
                    self.load_textgrid(tg_path)
                    tg_found = True
                    break
            if not tg_found:
                tg = self._create_textgrid_from_dialog()
                if tg is not None:
                    self.canvas.textgrid_data = tg
                    self.canvas._setup_axes()
                    self.canvas.render()
                    tier_desc = ", ".join(t.name for t in tg.tiers)
                    self.status.showMessage(
                        f"Created TextGrid "
                        f"({len(tg.tiers)} tier{'s' if len(tg.tiers) != 1 else ''}: {tier_desc})"
                    )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file:\n{e}")
            self.status.showMessage("Error loading file")

    def save_formants(self):
        if self.canvas.formant_data is None or self._filepath is None:
            self.status.showMessage("Nothing to save")
            return

        fmt_path = os.path.splitext(self._filepath)[0] + ".formants"
        try:
            self.canvas.formant_data.save(fmt_path)
            self.status.showMessage(f"Saved: {os.path.basename(fmt_path)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save:\n{e}")

    def load_formants(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Formant Data", "",
            "Formant Files (*.formants);;All Files (*)"
        )
        if not filepath:
            return
        try:
            self.canvas.formant_data = FormantData.load(filepath)
            self.canvas.render()
            self.status.showMessage(f"Loaded: {os.path.basename(filepath)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load:\n{e}")

    def load_textgrid(self, filepath=None):
        """Load a TextGrid file, optionally from a given path or via dialog."""
        if filepath is None:
            start_dir = os.path.dirname(self._filepath) if self._filepath else ""
            filepath, _ = QFileDialog.getOpenFileName(
                self, "Load TextGrid", start_dir,
                "TextGrid Files (*.TextGrid *.textgrid);;All Files (*)"
            )
            if not filepath:
                return

        try:
            tg = TextGrid.from_file(filepath)
            self._textgrid_path = filepath
            self.canvas.textgrid_data = tg
            self.canvas._setup_axes()
            self.canvas.render()
            tier_desc = ", ".join(t.name for t in tg.tiers)
            self.status.showMessage(
                f"TextGrid: {os.path.basename(filepath)} "
                f"({len(tg.tiers)} tier{'s' if len(tg.tiers) != 1 else ''}: {tier_desc})"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load TextGrid:\n{e}")

    def save_textgrid(self):
        """Save the current TextGrid to file."""
        if self.canvas.textgrid_data is None:
            self.status.showMessage("No TextGrid to save")
            return

        if self._textgrid_path:
            save_path = self._textgrid_path
        else:
            start_dir = os.path.dirname(self._filepath) if self._filepath else ""
            save_path, _ = QFileDialog.getSaveFileName(
                self, "Save TextGrid", start_dir,
                "TextGrid Files (*.TextGrid);;All Files (*)"
            )
            if not save_path:
                return

        try:
            self.canvas.textgrid_data.save(save_path)
            self._textgrid_path = save_path
            self.status.showMessage(f"Saved TextGrid: {os.path.basename(save_path)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save TextGrid:\n{e}")

    def _create_textgrid_from_dialog(self):
        """Show CreateTextGridDialog and return a TextGrid or None."""
        if self.canvas.sound is None:
            return None
        duration = self.canvas.sound.duration
        dlg = CreateTextGridDialog(duration, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            return dlg.get_textgrid()
        return None

    def _add_tier_to_textgrid(self):
        """Show AddTierDialog and insert a new tier into the current TextGrid."""
        tg = self.canvas.textgrid_data
        if tg is None:
            self.status.showMessage("No TextGrid loaded")
            return
        dlg = AddTierDialog(tg, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_tier = dlg.get_tier()
            pos = dlg.get_position()
            tg.tiers.insert(pos, new_tier)
            # Adjust active tier if it shifted
            if (self.canvas._active_tier is not None
                    and pos <= self.canvas._active_tier):
                self.canvas._active_tier += 1
            self.canvas._setup_axes()
            self.canvas.render()
            self.status.showMessage(f"Added tier \"{new_tier.name}\"")

    # -------------------------------------------------------------------
    # Formant analysis
    # -------------------------------------------------------------------

    def _run_formant_analysis(self):
        if self.canvas.sound is None:
            return

        ctrl = self.controls
        n_formants = ctrl.num_formants_combo.currentData()

        self.canvas.formant_data = extract_formants_from_praat(
            self.canvas.sound,
            n_formants=n_formants,
            max_formant=ctrl.max_formant_spin.value(),
            window_length=ctrl.window_length_spin.value(),
            pre_emphasis=ctrl.contrast_slider.value(),
        )

    def _reanalyse_formants(self):
        if self.canvas.sound is None:
            self.status.showMessage("No audio loaded")
            return

        self.status.showMessage("Re-analysing formants...")
        QApplication.processEvents()
        self._run_formant_analysis()
        self.canvas.render()
        n = self.controls.num_formants_combo.currentData()
        self.status.showMessage(
            f"Re-analysed: {self.canvas.formant_data.n_frames} frames, "
            f"F1–F{n}, time step = {self.canvas.formant_data.time_step*1000:.1f} ms"
        )

    # -------------------------------------------------------------------
    # Scrollbar
    # -------------------------------------------------------------------

    def _update_scrollbar(self):
        """Sync scrollbar range and position to the canvas view."""
        c = self.canvas
        if c.total_duration <= 0:
            return

        self._scrollbar_updating = True

        # Use milliseconds as scrollbar units
        total_ms = int(c.total_duration * 1000)
        view_ms = int(c.view_width * 1000)
        pos_ms = int(c.view_start * 1000)

        self.scrollbar.setRange(0, max(0, total_ms - view_ms))
        self.scrollbar.setPageStep(view_ms)
        self.scrollbar.setValue(pos_ms)

        self._scrollbar_updating = False

    def _on_scrollbar_changed(self, value):
        """Scrollbar dragged — pan the view."""
        if self._scrollbar_updating:
            return
        c = self.canvas
        new_start = value / 1000.0
        width = c.view_width
        c.set_view(new_start, new_start + width)
        c._render_timer.start()  # debounced render

    # -------------------------------------------------------------------
    # UI callbacks
    # -------------------------------------------------------------------

    def _on_spec_setting_changed(self):
        self.canvas.dynamic_range = self.controls.dynamic_range_slider.value()
        self.canvas.brightness = self.controls.brightness_slider.value()
        self.canvas._render_timer.start()  # debounced render

    def _on_spec_window_changed(self):
        """Spectrogram window length slider released — recompute spectrogram."""
        ms = self.controls.spec_window_slider.value()
        self.canvas.spec_window_length = ms / 1000.0
        self.canvas._compute_spectrogram()
        self.canvas.render()

    def _on_max_freq_changed(self, val):
        self.canvas.max_freq = val
        self._max_freq_timer.start()  # debounced recompute

    def _do_max_freq_update(self):
        self.canvas._compute_spectrogram()
        self.canvas.render()

    def _on_show_formants_toggled(self, checked):
        self.canvas.show_formants = checked
        self.canvas.render()

    def _on_preemphasis_changed(self):
        """Pre-emphasis slider released — re-analyse formants."""
        if self.canvas.sound is None:
            return
        self.status.showMessage("Re-analysing formants (pre-emphasis changed)...")
        QApplication.processEvents()
        self._run_formant_analysis()
        self.canvas.render()
        self.status.showMessage(
            f"Pre-emphasis: {self.controls.contrast_slider.value()} Hz"
        )

    def _on_num_formants_display_changed(self, index):
        """Show formants dropdown changed — update display immediately."""
        n = self.controls.num_formants_combo.currentData()
        self.canvas.display_n_formants = n
        # If active formant is now beyond display range, clamp it
        if self.canvas.active_formant >= n:
            self.canvas.active_formant = n - 1
            self.controls.update_active_formant_display(n - 1)
        self.canvas.render()

    def _on_edit_toggled(self, active):
        self.canvas.edit_mode = active
        if active:
            self.status.showMessage(
                "EDIT MODE — Click & drag to draw formants. "
                "F1–F5 to select formant. Press Edit again to exit."
            )
        else:
            self.status.showMessage("Edit mode off")
        self.canvas.render()

    def _reset_current_formant(self):
        if self.canvas.formant_data is None:
            return
        f_idx = self.canvas.active_formant
        self.canvas.formant_data.reset_to_original(f_idx)
        self.canvas.render()
        self.status.showMessage(f"Reset {FORMANT_LABELS[f_idx + 1]} to original")

    def _reset_all_formants(self):
        if self.canvas.formant_data is None:
            return
        self.canvas.formant_data.reset_to_original()
        self.canvas.render()
        self.status.showMessage("Reset all formants to original")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark palette
    from PyQt6.QtGui import QPalette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#1e1e2e"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#cccccc"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#252535"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#2a2a3a"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#cccccc"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#334455"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#cccccc"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#6699cc"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
