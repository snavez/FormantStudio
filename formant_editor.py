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
import io
import json
import csv
import time as _time
import numpy as np
import parselmouth
from parselmouth import praat

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QToolBar, QFileDialog, QLabel, QSlider, QGroupBox, QStatusBar,
    QPushButton, QSplitter, QComboBox, QDoubleSpinBox, QCheckBox,
    QMessageBox, QSizePolicy, QScrollBar, QLineEdit, QDialog,
    QDialogButtonBox, QFormLayout, QGridLayout,
    QWizard, QWizardPage, QProgressDialog, QListWidget, QListWidgetItem,
    QRadioButton, QButtonGroup, QTabWidget, QScrollArea
)
from PyQt6.QtCore import Qt, QTimer, QEvent, QObject, pyqtSignal, QByteArray, QBuffer, QIODevice
from PyQt6.QtMultimedia import QAudioSink, QAudioFormat
from PyQt6.QtGui import QAction, QKeySequence, QColor, QFont, QTransform

import pyqtgraph as pg

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
DEFAULT_SPEC_MAX_FREQ = 8000.0
DEFAULT_SPEC_WINDOW = 0.005    # seconds — spectrogram analysis window

# View limits
MAX_SPECTROGRAM_VIEW = 10.0    # seconds — hide spectrogram beyond this
MIN_VIEW_WIDTH = 0.1           # seconds — maximum zoom
ZOOM_FACTOR = 1.3              # per scroll wheel notch

# Undo system
from collections import namedtuple
UndoEntry = namedtuple('UndoEntry', ['description', 'changes'])
# changes = list of (formant_idx, frame_idx, old_value, old_mask, new_value, new_mask)
MAX_UNDO_STEPS = 100

# IPA symbol chart path (resolved for PyInstaller bundle)
_IPA_CHART_PATH = os.path.join(
    getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__))),
    "Docs", "ipa_symbol_chart.csv"
)

# Diacritic suffixes for SAMPA / X-SAMPA notation (longest first)
_DIACRITIC_SUFFIXES = [
    ("_h", "aspirated"),
    ("_0", "voiceless"),
    ("_t", "dental"),
    ("_k", "velarized"),
    ("_w", "labialized"),
    ("_j", "palatalized"),
    ("_~", "nasalized"),
    ("_d", "breathy"),
    ("_c", "creaky"),
    ("_:", "long"),
    (":", "long"),
]

# IPA combining diacritics (Unicode)
_IPA_COMBINING_DIACRITICS = [
    ("\u02B0", "aspirated"),     # superscript h
    ("\u0325", "voiceless"),     # combining ring below
    ("\u032A", "dental"),        # combining bridge below
    ("\u0334", "velarized"),     # combining tilde overlay
    ("\u02B7", "labialized"),    # superscript w
    ("\u02B2", "palatalized"),   # superscript j
    ("\u0303", "nasalized"),     # combining tilde
    ("\u0324", "breathy"),       # combining diaeresis below
    ("\u0330", "creaky"),        # combining tilde below
    ("\u02D0", "long"),          # length mark ː
]

# All known diacritic feature names (for column generation)
_ALL_DIACRITIC_FEATURES = [
    "aspirated", "voiceless_diac", "dental_diac", "velarized",
    "labialized", "palatalized", "nasalized", "breathy", "creaky",
]


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

    Always requests 5 formants from Praat so that all tracks are available
    for display regardless of the current dropdown setting.  The *n_formants*
    parameter is accepted for backwards compatibility but ignored — the
    display_n_formants attribute on SpectrogramCanvas controls how many
    tracks are rendered.
    """
    formant_obj = sound.to_formant_burg(
        time_step=time_step if time_step > 0 else None,
        max_number_of_formants=5.0,
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
    for f_idx in range(5):
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
# CSV Export Helpers
# ---------------------------------------------------------------------------


def _get_formant_at_time(fd, time_s):
    """Return (F1, F2, F3) at *time_s*, interpolating between nearest frames.

    If the exact time matches a frame (within half a time step), return that
    frame's values directly.  Otherwise linearly interpolate between the two
    bracketing frames.  Returns NaN for any formant that is NaN in both
    bracketing frames.
    """
    if fd is None or fd.n_frames == 0:
        return (np.nan, np.nan, np.nan)

    times = fd.times
    half_step = (fd.time_step or 0.005) / 2.0

    # Find nearest frame
    idx = int(np.argmin(np.abs(times - time_s)))

    # Close enough to an exact frame?
    if abs(times[idx] - time_s) <= half_step:
        return tuple(float(fd.values[f, idx]) for f in range(3))

    # Determine bracketing frames
    if time_s < times[idx]:
        lo, hi = max(0, idx - 1), idx
    else:
        lo, hi = idx, min(fd.n_frames - 1, idx + 1)

    if lo == hi:
        return tuple(float(fd.values[f, lo]) for f in range(3))

    # Linear interpolation factor
    t_lo, t_hi = times[lo], times[hi]
    frac = (time_s - t_lo) / (t_hi - t_lo) if t_hi != t_lo else 0.0

    result = []
    for f in range(3):
        v_lo = fd.values[f, lo]
        v_hi = fd.values[f, hi]
        if np.isnan(v_lo) and np.isnan(v_hi):
            result.append(np.nan)
        elif np.isnan(v_lo):
            result.append(float(v_hi))
        elif np.isnan(v_hi):
            result.append(float(v_lo))
        else:
            result.append(float(v_lo + frac * (v_hi - v_lo)))
    return tuple(result)


def _detect_tier_hierarchy(tiers):
    """Sort interval tiers by non-empty segment count (fewest = highest).

    *tiers* is a list of Tier objects.  Returns a new list sorted so that
    tiers with fewer non-empty intervals come first (e.g. words before
    phones).  Point tiers are appended at the end.
    """
    interval_tiers = []
    point_tiers = []
    for t in tiers:
        if t.tier_class == "IntervalTier":
            n = sum(1 for iv in t.intervals if iv.text.strip())
            interval_tiers.append((n, t))
        else:
            point_tiers.append(t)
    interval_tiers.sort(key=lambda x: x[0])
    return [t for _, t in interval_tiers] + point_tiers


def _find_containing_interval(tier, xmin, xmax):
    """Return the Interval in *tier* that contains [xmin, xmax], or None."""
    eps = 1e-6
    for iv in tier.intervals:
        if iv.xmin <= xmin + eps and iv.xmax >= xmax - eps:
            return iv
    return None


def _find_containing_interval_for_point(tier, time_s):
    """Return the Interval in *tier* that contains *time_s*, or None."""
    eps = 1e-6
    for iv in tier.intervals:
        if iv.xmin - eps <= time_s <= iv.xmax + eps:
            return iv
    return None


# ---------------------------------------------------------------------------
# IPA / SAMPA / X-SAMPA classification helpers
# ---------------------------------------------------------------------------

def _load_ipa_chart(csv_path):
    """Parse the IPA symbol chart CSV into a dict keyed by notation system.

    Returns ``{'ipa': {symbol: props}, 'sampa': {...}, 'xsampa': {...}}``
    where *props* is a dict with keys: type, subtype, height, fronting,
    rounding, length, voicing, place, manner.
    """
    with open(csv_path, "r", encoding="utf-8") as fh:
        raw_lines = [ln for ln in fh if not ln.strip().startswith("#")]
    reader = csv.DictReader(io.StringIO("".join(raw_lines)))
    chart = {"ipa": {}, "sampa": {}, "xsampa": {}}
    prop_keys = ["type", "subtype", "height", "fronting", "rounding",
                 "length", "voicing", "place", "manner"]
    for row in reader:
        props = {k: row.get(k, "").strip() for k in prop_keys}
        for notation in ("ipa", "sampa", "xsampa"):
            sym = row.get(notation, "").strip()
            if sym:
                chart[notation][sym] = props
    return chart


def _build_known_vowels(chart_lookup):
    """Return set of single-character vowel symbols from *chart_lookup*."""
    return {sym for sym, props in chart_lookup.items()
            if props.get("type") == "vowel" and len(sym) == 1}


def _classify_label(label, chart_lookup, notation, known_vowels):
    """Classify a single TextGrid label using the IPA chart.

    Returns ``(properties_dict, modifiers_list, matched_bool)``.
    """
    label = label.strip()
    if not label:
        return {}, [], True

    # Exact lookup
    if label in chart_lookup:
        return dict(chart_lookup[label]), [], True

    # Diacritic stripping
    if notation == "ipa":
        diac_list = _IPA_COMBINING_DIACRITICS
    else:
        diac_list = _DIACRITIC_SUFFIXES

    modifiers = []
    base = label
    changed = True
    while changed:
        changed = False
        for suffix, feature in diac_list:
            if base.endswith(suffix) and len(base) > len(suffix):
                base = base[:-len(suffix)]
                feat_name = feature
                # Avoid duplicating "voiceless"/"long" if already a base property
                if feat_name == "voiceless":
                    feat_name = "voiceless_diac"
                elif feat_name == "long":
                    feat_name = "long_diac"
                elif feat_name == "dental":
                    feat_name = "dental_diac"
                if feat_name not in modifiers:
                    modifiers.append(feat_name)
                changed = True
                # Try lookup after each strip
                if base in chart_lookup:
                    return dict(chart_lookup[base]), modifiers, True
                break  # restart loop for greedy matching

    # After full stripping, try lookup once more
    if base in chart_lookup:
        return dict(chart_lookup[base]), modifiers, True

    # Auto-detect diphthong: remaining is 2+ chars all in known vowels
    if len(base) >= 2 and all(ch in known_vowels for ch in base):
        props = {"type": "vowel", "subtype": "diphthong",
                 "voicing": "voiced", "length": "short"}
        return props, modifiers, True

    # Unmatched
    return {}, modifiers, False


def _build_csv_data(audio_dir, textgrid_dir, formants_dir,
                    selected_tiers, extract_formants, formant_mode,
                    point_tier_name, segment_tier_name,
                    percentage_markers, extract_durations,
                    duration_tier_names, point_tier_parents=None,
                    progress_callback=None,
                    include_point_times=False,
                    categorise=False, cat_chart=None,
                    cat_notation="ipa", cat_tier_names=None,
                    cat_vowel_props=None, cat_consonant_props=None,
                    auto_diphthong_candidates=None,
                    unmatched_labels=None):
    """Build CSV header and rows for batch formant/duration export.

    *formant_mode* can be ``"at_points"``, ``"for_segments"``, or ``"both"``.
    *point_tier_parents* maps point tier names to their parent interval tier
    name (determined by hierarchy order — the nearest interval tier above).
    When *categorise* is True, classification columns are appended.
    *auto_diphthong_candidates* and *unmatched_labels* are mutable sets
    that collect data for post-processing.
    Returns (headers: list[str], rows: list[list[str|float]]).
    """
    if point_tier_parents is None:
        point_tier_parents = {}
    if cat_tier_names is None:
        cat_tier_names = []
    if cat_vowel_props is None:
        cat_vowel_props = []
    if cat_consonant_props is None:
        cat_consonant_props = []
    do_at_points = formant_mode in ("at_points", "both") if extract_formants else False
    do_for_segments = formant_mode in ("for_segments", "both") if extract_formants else False

    # Discover audio files
    audio_exts = {".wav", ".aiff", ".mp3"}
    audio_files = sorted(
        f for f in os.listdir(audio_dir)
        if os.path.splitext(f)[1].lower() in audio_exts
    )

    # --- Build header ---
    headers = ["filename"]

    # Tier label columns — interval tiers in hierarchy order
    interval_tier_names = [t.name for t in selected_tiers
                           if t.tier_class == "IntervalTier"]
    headers.extend(interval_tier_names)

    # Point tier label columns — always included if the user selected them
    point_tier_names = [t.name for t in selected_tiers
                        if t.tier_class == "TextTier"]
    headers.extend(point_tier_names)

    # Formant columns — at-points (F1, F2, F3 named after point tier)
    if do_at_points:
        pt_suffix = point_tier_name or "pt"
        if include_point_times:
            headers.append(f"{pt_suffix}_time")
        headers.extend([f"F1_{pt_suffix}", f"F2_{pt_suffix}", f"F3_{pt_suffix}"])

    # Formant columns — for-segments (F1_P%, F2_P%, F3_P% per percentage)
    if do_for_segments:
        for pct in percentage_markers:
            pct_s = f"{pct:g}"
            headers.extend([f"F1_{pct_s}%", f"F2_{pct_s}%", f"F3_{pct_s}%"])

    # Duration columns
    if extract_durations:
        for tn in duration_tier_names:
            headers.append(f"dur_{tn}")

    # Categorisation columns
    cat_col_start = len(headers)
    cat_col_info = []  # [(header_name, tier_name, col_type, prop_name), ...]
    known_vowels_set = None
    if categorise and cat_chart:
        lookup = cat_chart[cat_notation]
        known_vowels_set = _build_known_vowels(lookup)
        for tn in cat_tier_names:
            # Type column (always)
            cat_col_info.append((f"{tn}_type", tn, "type", "type"))
            # Vowel property columns
            for vp in cat_vowel_props:
                cat_col_info.append((f"{tn}_V_{vp}", tn, "vowel", vp))
            # Consonant property columns
            for cp in cat_consonant_props:
                cat_col_info.append((f"{tn}_C_{cp}", tn, "consonant", cp))
            # Diacritic modifier columns (all known — empty ones filtered later)
            for feat in _ALL_DIACRITIC_FEATURES:
                cat_col_info.append((f"{tn}_{feat}", tn, "modifier", feat))
        headers.extend([ci[0] for ci in cat_col_info])

    rows = []

    for file_idx, audio_file in enumerate(audio_files):
        if progress_callback:
            progress_callback(file_idx, len(audio_files), audio_file)

        basename = os.path.splitext(audio_file)[0]

        # Find matching TextGrid
        tg_path = None
        for ext in (".TextGrid", ".textgrid"):
            candidate = os.path.join(textgrid_dir, basename + ext)
            if os.path.exists(candidate):
                tg_path = candidate
                break

        if tg_path is None:
            row = [audio_file] + [""] * (len(headers) - 1)
            rows.append(row)
            continue

        try:
            tg = TextGrid.from_file(tg_path)
        except Exception:
            row = [audio_file] + [""] * (len(headers) - 1)
            rows.append(row)
            continue

        # Map tier names → Tier objects for this file
        tier_map = {}
        for tier in tg.tiers:
            tier_map[tier.name] = tier

        # Load or extract formant data
        fd = None
        if extract_formants:
            fmt_path = None
            if formants_dir:
                candidate = os.path.join(formants_dir, basename + ".formants")
                if os.path.exists(candidate):
                    fmt_path = candidate
            if fmt_path:
                try:
                    fd = FormantData.load(fmt_path)
                except Exception:
                    fd = None
            if fd is None:
                audio_path = os.path.join(audio_dir, audio_file)
                try:
                    snd = parselmouth.Sound(audio_path)
                    fd = extract_formants_from_praat(snd)
                except Exception:
                    fd = None

        # Determine the lowest-level interval tier to drive rows
        lowest_tier = None
        for t_name in reversed(interval_tier_names):
            if t_name in tier_map:
                lowest_tier = tier_map[t_name]
                break

        if lowest_tier is None:
            row = [audio_file] + [""] * (len(headers) - 1)
            rows.append(row)
            continue

        # Resolve point tier for at-points formant extraction
        at_pt_tier = None
        if do_at_points and point_tier_name:
            at_pt_tier = tier_map.get(point_tier_name)
            if at_pt_tier and at_pt_tier.tier_class != "TextTier":
                at_pt_tier = None

        # Helper: build the common prefix columns for a lowest-tier interval
        def _label_cols(iv):
            cols = []
            # Interval tier labels — also cache for point tier parent lookups
            interval_cache = {}  # tier_name -> containing Interval
            for t_name in interval_tier_names:
                t = tier_map.get(t_name)
                if t is None:
                    cols.append("")
                elif t.name == lowest_tier.name:
                    cols.append(iv.text)
                    interval_cache[t_name] = iv
                else:
                    civ = _find_containing_interval(t, iv.xmin, iv.xmax)
                    cols.append(civ.text if civ else "")
                    if civ:
                        interval_cache[t_name] = civ
            # Point tier labels — look up within the PARENT interval tier
            for pt_name in point_tier_names:
                pt_tier = tier_map.get(pt_name)
                if pt_tier is None or pt_tier.tier_class != "TextTier":
                    cols.append("")
                    continue
                # Determine the search bounds from parent interval tier
                parent_name = point_tier_parents.get(pt_name)
                parent_iv = interval_cache.get(parent_name) if parent_name else None
                if parent_iv is not None:
                    search_xmin, search_xmax = parent_iv.xmin, parent_iv.xmax
                else:
                    # Fallback: use the lowest-tier interval bounds
                    search_xmin, search_xmax = iv.xmin, iv.xmax
                eps = 1e-6
                marks = [p.mark for p in pt_tier.points
                         if search_xmin - eps <= p.time <= search_xmax + eps]
                cols.append("; ".join(marks) if marks else "")
            return cols

        # Helper: duration columns for an interval
        def _dur_cols(iv):
            cols = []
            for tn in duration_tier_names:
                t = tier_map.get(tn)
                if t is None or t.tier_class != "IntervalTier":
                    cols.append("")
                elif tn == lowest_tier.name:
                    cols.append(f"{iv.xmax - iv.xmin:.4f}")
                else:
                    civ = _find_containing_interval(t, iv.xmin, iv.xmax)
                    if civ is None:
                        civ = _find_containing_interval_for_point(
                            t, (iv.xmin + iv.xmax) / 2)
                    cols.append(f"{civ.xmax - civ.xmin:.4f}" if civ else "")
            return cols

        # Helper: categorisation columns for a row (given label_cols already built)
        def _cat_cols(label_cols_list):
            """Return list of categorisation cell values."""
            if not categorise or not cat_col_info:
                return []
            cols = []
            for _hdr, tn, col_type, prop in cat_col_info:
                # Find the label for this tier from the already-built row
                # Tier label is at index: interval_tier_names.index(tn) or
                # point_tier_names offset
                label_text = ""
                if tn in interval_tier_names:
                    idx = interval_tier_names.index(tn)
                    if idx < len(label_cols_list):
                        label_text = label_cols_list[idx]
                elif tn in point_tier_names:
                    idx = len(interval_tier_names) + point_tier_names.index(tn)
                    if idx < len(label_cols_list):
                        label_text = label_cols_list[idx]

                if not label_text.strip():
                    cols.append("")
                    continue

                # For point tiers that may have "; " joined labels, classify first
                labels_to_classify = [l.strip() for l in label_text.split(";")
                                      if l.strip()]
                if not labels_to_classify:
                    cols.append("")
                    continue

                # Classify the primary (first) label
                lbl = labels_to_classify[0]
                props, mods, matched = _classify_label(
                    lbl, cat_chart[cat_notation], cat_notation,
                    known_vowels_set)

                if not matched:
                    if unmatched_labels is not None:
                        unmatched_labels.add((tn, lbl))
                    cols.append("")
                    continue

                if col_type == "type":
                    cols.append(props.get("type", ""))
                elif col_type == "vowel":
                    if props.get("type") == "vowel":
                        cols.append(props.get(prop, ""))
                    else:
                        cols.append("")
                elif col_type == "consonant":
                    if props.get("type") == "consonant":
                        cols.append(props.get(prop, ""))
                    else:
                        cols.append("")
                elif col_type == "modifier":
                    if prop in mods:
                        cols.append(prop)
                    else:
                        cols.append("")

                # Track all diphthong labels for review
                if (auto_diphthong_candidates is not None
                        and props.get("subtype") == "diphthong"):
                    auto_diphthong_candidates[lbl] = (
                        auto_diphthong_candidates.get(lbl, 0) + 1)

            return cols

        # --- Row generation ---
        if do_at_points and at_pt_tier:
            # When extracting at points, one row per point in each segment
            for iv in lowest_tier.intervals:
                if not iv.text.strip():
                    continue

                eps = 1e-6
                points_in_iv = [
                    p for p in at_pt_tier.points
                    if iv.xmin - eps <= p.time <= iv.xmax + eps
                ]

                if not points_in_iv:
                    # No target points — still emit a row if doing segments too
                    if do_for_segments:
                        lc_np = _label_cols(iv)
                        row = [audio_file] + lc_np
                        if include_point_times:
                            row.append("")  # empty time
                        row.extend(["", "", ""])  # empty at-point F1/F2/F3
                        dur = iv.xmax - iv.xmin
                        for pct in percentage_markers:
                            t = iv.xmin + dur * pct / 100.0
                            f1, f2, f3 = _get_formant_at_time(fd, t)
                            for v in (f1, f2, f3):
                                row.append(f"{v:.1f}" if not np.isnan(v) else "")
                        if extract_durations:
                            row.extend(_dur_cols(iv))
                        row.extend(_cat_cols(lc_np))
                        rows.append(row)
                    continue

                for pt in points_in_iv:
                    row = [audio_file]
                    # Interval tier labels
                    for t_name in interval_tier_names:
                        t = tier_map.get(t_name)
                        if t is None:
                            row.append("")
                        elif t.name == lowest_tier.name:
                            row.append(iv.text)
                        else:
                            civ = _find_containing_interval(t, iv.xmin, iv.xmax)
                            row.append(civ.text if civ else "")
                    # Point tier labels — use parent interval tier bounds
                    # Build interval cache for parent lookups
                    _iv_cache = {}
                    for t_name in interval_tier_names:
                        t = tier_map.get(t_name)
                        if t is None:
                            continue
                        if t.name == lowest_tier.name:
                            _iv_cache[t_name] = iv
                        else:
                            civ = _find_containing_interval(t, iv.xmin, iv.xmax)
                            if civ:
                                _iv_cache[t_name] = civ
                    for pt_name in point_tier_names:
                        if pt_name == point_tier_name:
                            row.append(pt.mark)
                        else:
                            pt_tier = tier_map.get(pt_name)
                            if pt_tier is None or pt_tier.tier_class != "TextTier":
                                row.append("")
                                continue
                            parent_name = point_tier_parents.get(pt_name)
                            parent_iv = _iv_cache.get(parent_name) if parent_name else None
                            if parent_iv is not None:
                                smin, smax = parent_iv.xmin, parent_iv.xmax
                            else:
                                smin, smax = iv.xmin, iv.xmax
                            marks = [p.mark for p in pt_tier.points
                                     if smin - eps <= p.time <= smax + eps]
                            row.append("; ".join(marks) if marks else "")

                    # At-point time and formant values
                    if include_point_times:
                        row.append(f"{pt.time:.4f}")
                    f1, f2, f3 = _get_formant_at_time(fd, pt.time)
                    for v in (f1, f2, f3):
                        row.append(f"{v:.1f}" if not np.isnan(v) else "")

                    # Segment percentage formant values (if also requested)
                    if do_for_segments:
                        dur = iv.xmax - iv.xmin
                        for pct in percentage_markers:
                            t = iv.xmin + dur * pct / 100.0
                            f1, f2, f3 = _get_formant_at_time(fd, t)
                            for v in (f1, f2, f3):
                                row.append(f"{v:.1f}" if not np.isnan(v) else "")

                    if extract_durations:
                        row.extend(_dur_cols(iv))

                    # Categorisation: label cols are row[1 : 1+n_labels]
                    n_labels = len(interval_tier_names) + len(point_tier_names)
                    row.extend(_cat_cols(row[1:1 + n_labels]))

                    rows.append(row)
        else:
            # Row per lowest-tier segment (no at-points expansion)
            for iv in lowest_tier.intervals:
                if not iv.text.strip():
                    continue

                lc = _label_cols(iv)
                row = [audio_file] + lc

                # Segment percentage formant values
                if do_for_segments:
                    dur = iv.xmax - iv.xmin
                    for pct in percentage_markers:
                        t = iv.xmin + dur * pct / 100.0
                        f1, f2, f3 = _get_formant_at_time(fd, t)
                        for v in (f1, f2, f3):
                            row.append(f"{v:.1f}" if not np.isnan(v) else "")

                if extract_durations:
                    row.extend(_dur_cols(iv))

                row.extend(_cat_cols(lc))

                rows.append(row)

    return headers, rows


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
                font-family: "Segoe UI", "Arial Unicode MS", "Noto Sans", sans-serif;
                font-size: 16px;
                font-weight: bold;
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
        # Delete key with empty label or no selection: clear focus and let
        # the main window handle it (e.g. delete a TextTier point).
        if key == Qt.Key.Key_Delete and not self.text():
            self.clearFocus()
            event.ignore()
            return
        super().keyPressEvent(event)


# ---------------------------------------------------------------------------
# Spectrogram Canvas (pyqtgraph embedded in Qt)
# ---------------------------------------------------------------------------

class _GLW(pg.GraphicsLayoutWidget):
    """Inner GraphicsLayoutWidget that forwards mouse events to the canvas."""

    def __init__(self, canvas, parent=None):
        super().__init__(parent)
        self._canvas = canvas
        self.setBackground("#1e1e1e")
        self.setMouseTracking(True)

    def mousePressEvent(self, event):
        self._canvas._on_mouse_press(event)

    def mouseReleaseEvent(self, event):
        self._canvas._on_mouse_release(event)

    def mouseMoveEvent(self, event):
        self._canvas._on_mouse_move(event)

    def wheelEvent(self, event):
        self._canvas._on_scroll(event)

    def mouseDoubleClickEvent(self, event):
        self._canvas._on_mouse_double_click(event)


class SpectrogramCanvas(QWidget):
    """
    pyqtgraph-based canvas displaying:
      - Spectrogram (with adjustable display settings)
      - Formant overlay (color-coded dots/lines)
      - Interactive editing — no blitting needed (scene graph handles repaints)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._glw = _GLW(self)
        layout.addWidget(self._glw)

        # Plot items (created in _setup_axes)
        self._spec_plot = None   # PlotItem for spectrogram
        self._wave_plot = None   # PlotItem for waveform
        self._tier_plots = []    # list of PlotItem for visible TextGrid tiers
        self._tier_plot_indices = []  # maps _tier_plots[i] → tg.tiers index

        # pyqtgraph items (persistent, updated in-place)
        self._spec_image = None       # pg.ImageItem for spectrogram
        self._wave_line = None        # pg.PlotDataItem for waveform
        self._wave_zero = None        # pg.InfiniteLine for zero line
        self._wave_fill_pos = None    # pg.PlotDataItem for envelope
        self._wave_fill_neg = None    # pg.PlotDataItem for envelope
        self._formant_scatters = {}   # {f_idx: pg.ScatterPlotItem}
        self._transient_items = []    # items cleared each render
        self._overlay_items = []      # selection/boundary overlay items

        # Crosshair items
        self._crosshair_v = None      # InfiniteLine on spec
        self._crosshair_h = None      # InfiniteLine on spec
        self._crosshair_wave_v = None # InfiniteLine on wave
        self._crosshair_visible = False

        # Playback cursor items
        self._playback_cursors = []   # list of InfiniteLine on all plots

        # Greyscale LUT (reversed: dark=high intensity)
        self._gray_r_lut = np.zeros((256, 4), dtype=np.ubyte)
        for i in range(256):
            v = 255 - i
            self._gray_r_lut[i] = [v, v, v, 255]

        # Title label
        self._title_item = None

        # "Zoom in" text for when spectrogram is hidden
        self._zoom_text = None

        # Data
        self.sound = None
        self.spectrogram_data = None
        self.spec_times = None
        self.spec_freqs = None
        self.formant_data = None
        self.textgrid_data = None   # TextGrid object or None
        self.hidden_tiers = set()   # indices of tiers to hide
        self.tier_axes = []         # kept for API compat (list, same length as _tier_plots)
        self.wave_ax = None         # kept for API compat

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
        self._last_frame_freq = None
        self._stroke_scatter = None
        self._stroke_times = []
        self._stroke_freqs = []

        # Eraser state (right-click drag in edit mode)
        self._is_erasing = False

        # Undo/redo stacks
        self._undo_stack = []
        self._redo_stack = []
        self._stroke_changes = []  # accumulates per-frame changes during a stroke

        # Playback state
        self._playback_playing = False
        self._playback_start_time = 0.0
        self._playback_start_wall = 0.0
        self._playback_end_time = 0.0
        self._playback_audio_sink = None
        self._playback_audio_buf = None
        self._playback_timer = None
        self._click_time = None
        self._hover_time = None

        # TextGrid editing state
        self._selected_boundary = None
        self._dragging_boundary = False
        self._drag_tier_index = None
        self._drag_original_time = None
        self._drag_min_time = 0.0
        self._drag_max_time = 0.0
        self._drag_aligned = []
        self._drag_lines = []         # InfiniteLine items for drag feedback

        # Interval/point selection state
        self._selected_interval = None
        self._label_edit = None

        # Spectrogram drag selection state
        self._selection_start = None
        self._selection_end = None
        self._spec_dragging = False
        self._spec_drag_start = None
        self._spec_sel_regions = []   # LinearRegionItem items for drag selection

        # Active tier state
        self._active_tier = None

        # Debounce timers
        self._render_timer = QTimer()
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(30)
        self._render_timer.timeout.connect(self._debounced_render)
        self._label_render_timer = QTimer()
        self._label_render_timer.setSingleShot(True)
        self._label_render_timer.setInterval(100)
        self._label_render_timer.timeout.connect(self._debounced_render)

        # Callbacks
        self._status_callback = None
        self._on_formant_edited = None
        self._on_textgrid_edited = None
        self._on_view_changed_callback = None

        # Snap-to-boundary settings (set by MainWindow)
        self._snap_enabled_cb = None       # QCheckBox reference
        self._snap_tolerance_spin = None   # QDoubleSpinBox reference

        self._setup_empty_axes()

    # -------------------------------------------------------------------
    # Axes layout
    # -------------------------------------------------------------------

    def _setup_empty_axes(self):
        """Show an empty canvas with a message."""
        self._glw.clear()
        self._spec_plot = self._glw.addPlot(row=0, col=0)
        self._spec_plot.setMouseEnabled(x=False, y=False)
        self._spec_plot.hideButtons()
        self._spec_plot.setMenuEnabled(False)
        self._spec_plot.getAxis('bottom').setPen('#cccccc')
        self._spec_plot.getAxis('left').setPen('#cccccc')
        self._spec_plot.getAxis('bottom').setTextPen('#cccccc')
        self._spec_plot.getAxis('left').setTextPen('#cccccc')
        self._spec_plot.setLabel('bottom', 'Time (s)')
        self._spec_plot.setLabel('left', 'Frequency (Hz)')
        self._spec_plot.getViewBox().setBackgroundColor('#1a1a2e')
        t = pg.TextItem("Open a WAV file to begin", color='#eeeeee', anchor=(0.5, 0.5))
        t.setFont(QFont("Segoe UI", 11))
        self._spec_plot.addItem(t)
        t.setPos(0.5, 0.5)
        self._spec_plot.setXRange(0, 1)
        self._spec_plot.setYRange(0, 1)
        self._wave_plot = None
        self._tier_plots = []
        self._tier_plot_indices = []
        self.tier_axes = []
        self.wave_ax = None

    def _setup_axes(self):
        """Recreate plot layout based on loaded data."""
        self._glw.clear()
        self._tier_plots = []
        self._tier_plot_indices = []
        self.tier_axes = []
        self._wave_plot = None
        self.wave_ax = None
        self._spec_image = None
        self._wave_line = None
        self._wave_zero = None
        self._wave_fill_pos = None
        self._wave_fill_neg = None
        self._formant_scatters = {}
        self._transient_items = []
        self._overlay_items = []
        self._crosshair_v = None
        self._crosshair_h = None
        self._crosshair_wave_v = None
        self._playback_cursors = []
        self._title_item = None
        self._zoom_text = None
        self._stroke_scatter = None

        has_wave = self.sound is not None
        tg = self.textgrid_data
        has_tiers = tg is not None and len(tg.tiers) > 0

        if not has_wave and not has_tiers:
            self._spec_plot = self._glw.addPlot(row=0, col=0)
            self._configure_plot(self._spec_plot, '#1a1a2e')
            return

        # Build layout with row stretch factors
        row = 0

        if has_wave:
            # Waveform on top
            self._wave_plot = self._glw.addPlot(row=row, col=0)
            self._configure_plot(self._wave_plot, '#1a1a2e')
            self._wave_plot.getAxis('bottom').setStyle(showValues=False)
            self._wave_plot.getAxis('bottom').setHeight(0)
            self._wave_plot.setLabel('left', 'Amp', **{'font-size': '8pt', 'color': '#eeeeee'})
            self._glw.ci.layout.setRowStretchFactor(row, 25)
            self.wave_ax = self._wave_plot  # API compat
            row += 1

        # Spectrogram
        has_visible_tiers = (has_tiers and
                             any(i not in self.hidden_tiers for i in range(len(tg.tiers))))
        if has_visible_tiers:
            spec_stretch = 45
        else:
            spec_stretch = 75 if has_wave else 100
        self._spec_plot = self._glw.addPlot(row=row, col=0)
        self._configure_plot(self._spec_plot, '#1a1a2e')
        self._spec_plot.setLabel('left', 'Frequency (Hz)', **{'font-size': '9pt', 'color': '#eeeeee'})
        self._glw.ci.layout.setRowStretchFactor(row, spec_stretch)
        row += 1

        # Link wave X to spec
        if self._wave_plot is not None:
            self._wave_plot.setXLink(self._spec_plot)

        # Tier plots (skip hidden tiers)
        if has_tiers:
            visible_indices = [i for i in range(len(tg.tiers))
                               if i not in self.hidden_tiers]
            n_visible = len(visible_indices)
            if n_visible > 0:
                # Compute left axis width from longest visible tier name
                from PyQt6.QtGui import QFontMetrics
                measure_font = QFont("Segoe UI", 10)
                fm = QFontMetrics(measure_font)
                max_name_w = max(
                    fm.horizontalAdvance(tg.tiers[i].name)
                    for i in visible_indices
                )
                axis_width = max(60, max_name_w + 20)  # padding for offset

                tier_share = max(3, 30 // n_visible)
                for vi, tier_i in enumerate(visible_indices):
                    tp = self._glw.addPlot(row=row + vi, col=0)
                    self._configure_plot(tp, '#ffffff')
                    tp.setXLink(self._spec_plot)
                    tp.setYRange(0, 1, padding=0)
                    # Left axis: tier name shown as a centered tick label
                    left_ax = tp.getAxis('left')
                    left_ax.setWidth(axis_width)
                    left_ax.setStyle(tickTextOffset=8)
                    # Only show x-axis on bottom visible tier
                    if vi < n_visible - 1:
                        tp.getAxis('bottom').setStyle(showValues=False)
                        tp.getAxis('bottom').setHeight(0)
                    else:
                        tp.setLabel('bottom', 'Time (s)', **{'font-size': '9pt', 'color': '#eeeeee'})
                    self._glw.ci.layout.setRowStretchFactor(row + vi, tier_share)
                    self._tier_plots.append(tp)
                    self._tier_plot_indices.append(tier_i)
                # Hide x-axis on spectrogram when visible tiers present
                self._spec_plot.getAxis('bottom').setStyle(showValues=False)
                self._spec_plot.getAxis('bottom').setHeight(0)
            else:
                # All tiers hidden — spec shows its own x-axis
                self._spec_plot.setLabel('bottom', 'Time (s)', **{'font-size': '9pt', 'color': '#eeeeee'})
        else:
            # No tiers — spec shows its own x-axis
            self._spec_plot.setLabel('bottom', 'Time (s)', **{'font-size': '9pt', 'color': '#eeeeee'})

        self.tier_axes = list(self._tier_plots)  # API compat

        # Create persistent items
        self._spec_image = pg.ImageItem()
        self._spec_image.setLookupTable(self._gray_r_lut)
        self._spec_plot.addItem(self._spec_image)

        # Crosshair
        self._setup_crosshair()

    def _configure_plot(self, plot, bg_color):
        """Apply common configuration to a PlotItem."""
        plot.setMouseEnabled(x=False, y=False)
        plot.hideButtons()
        plot.setMenuEnabled(False)
        plot.getViewBox().setBackgroundColor(bg_color)
        for axis_name in ('bottom', 'left'):
            ax = plot.getAxis(axis_name)
            ax.setPen('#cccccc')
            ax.setTextPen('#cccccc')
            ax.setStyle(tickTextOffset=4)

    def _setup_crosshair(self):
        """Create crosshair InfiniteLine items on spec and wave plots."""
        pen = pg.mkPen('#00ffcc', width=0.7, style=Qt.PenStyle.SolidLine)
        self._crosshair_v = pg.InfiniteLine(angle=90, pen=pen)
        self._crosshair_h = pg.InfiniteLine(angle=0, pen=pen)
        self._crosshair_v.setVisible(False)
        self._crosshair_h.setVisible(False)
        self._spec_plot.addItem(self._crosshair_v)
        self._spec_plot.addItem(self._crosshair_h)
        self._crosshair_wave_v = None
        if self._wave_plot is not None:
            self._crosshair_wave_v = pg.InfiniteLine(angle=90, pen=pen)
            self._crosshair_wave_v.setVisible(False)
            self._wave_plot.addItem(self._crosshair_wave_v)
        self._crosshair_visible = False

    # -------------------------------------------------------------------
    # Data loading
    # -------------------------------------------------------------------

    def load_sound(self, filepath):
        """Load audio and compute spectrogram."""
        self.sound = parselmouth.Sound(filepath)
        self.total_duration = self.sound.duration
        self.view_start = 0.0
        self.view_end = self.total_duration
        self._compute_spectrogram()
        self._setup_axes()

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
        if old_width > 0:
            ratio = (center_time - self.view_start) / old_width
        else:
            ratio = 0.5
        ratio = max(0.0, min(1.0, ratio))
        new_start = center_time - ratio * new_width
        self.set_view(new_start, new_start + new_width)

    # -------------------------------------------------------------------
    # Render
    # -------------------------------------------------------------------

    def _debounced_render(self):
        """Slot for debounce timers — just calls render()."""
        self.render()

    def render(self):
        """Full re-render of spectrogram + formants + TextGrid tiers."""
        if self._playback_playing:
            self.stop_playback()

        if self.spectrogram_data is None:
            self._setup_empty_axes()
            return

        # Clear transient items from previous render
        self._clear_transient_items()
        self._clear_overlay_items()

        # Set view range (propagates via X link)
        self._spec_plot.setXRange(self.view_start, self.view_end, padding=0)
        self._spec_plot.setYRange(0, self.max_freq, padding=0)

        # Draw components
        self._draw_spectrogram()
        if self._wave_plot is not None:
            self._draw_waveform()

        # Remove old formant scatters (must happen even when hiding formants)
        for key, sc in self._formant_scatters.items():
            try:
                self._spec_plot.removeItem(sc)
            except Exception:
                pass
        self._formant_scatters = {}

        if self.show_formants and self.formant_data is not None:
            self._draw_formants()
        if self.textgrid_data is not None and len(self._tier_plots) > 0:
            self._draw_textgrid()

        # Title
        self._update_title()

        # Selection/boundary overlays
        self._draw_selection_overlay()

    def _clear_transient_items(self):
        """Remove transient items from all plots."""
        for item, plot in self._transient_items:
            try:
                plot.removeItem(item)
            except Exception:
                pass
        self._transient_items = []

    def _clear_overlay_items(self):
        """Remove overlay items (selection/boundary highlights)."""
        for item, plot in self._overlay_items:
            try:
                plot.removeItem(item)
            except Exception:
                pass
        self._overlay_items = []

    def _add_transient(self, item, plot):
        """Add a transient item to a plot, tracked for cleanup."""
        plot.addItem(item)
        self._transient_items.append((item, plot))

    def _add_overlay(self, item, plot):
        """Add an overlay item to a plot, tracked for cleanup."""
        plot.addItem(item)
        self._overlay_items.append((item, plot))

    # -------------------------------------------------------------------
    # Spectrogram drawing
    # -------------------------------------------------------------------

    def _draw_spectrogram(self):
        """Render spectrogram image into the spec plot."""
        if self._spec_image is None:
            return

        view_width = self.view_width
        show_spectrogram = view_width <= MAX_SPECTROGRAM_VIEW

        # Remove old zoom text if any
        if self._zoom_text is not None:
            try:
                self._spec_plot.removeItem(self._zoom_text)
            except Exception:
                pass
            self._zoom_text = None

        if show_spectrogram:
            # Slice spectrogram to visible time range
            t_mask = ((self.spec_times >= self.view_start) &
                      (self.spec_times <= self.view_end))
            vis_times = self.spec_times[t_mask]
            vis_data = self.spectrogram_data[:, t_mask]

            if len(vis_times) > 0:
                power_db = 10 * np.log10(vis_data + 1e-20)
                peak_db = np.max(power_db)
                vmax = peak_db + self.brightness
                vmin = vmax - self.dynamic_range

                # Normalize to 0-255 uint8 for LUT
                img = np.clip((power_db - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)

                # ImageItem expects (width, height) = (n_time, n_freq)
                self._spec_image.setImage(img.T, autoLevels=False)
                self._spec_image.setLookupTable(self._gray_r_lut)

                # Position: translate to (t0, f0), scale to fill time/freq range
                t0 = vis_times[0]
                f0 = self.spec_freqs[0] if len(self.spec_freqs) > 0 else 0
                n_time = img.shape[1]
                n_freq = img.shape[0]
                dt = (vis_times[-1] - vis_times[0]) if n_time > 1 else 1.0
                df = (self.spec_freqs[-1] - self.spec_freqs[0]) if n_freq > 1 else self.max_freq

                tr = QTransform()
                tr.translate(t0, f0)
                tr.scale(dt / max(n_time, 1), df / max(n_freq, 1))
                self._spec_image.setTransform(tr)
                self._spec_image.setVisible(True)
            else:
                self._spec_image.setVisible(False)
        else:
            self._spec_image.setVisible(False)
            # Show zoom message
            mid_t = (self.view_start + self.view_end) / 2
            mid_f = self.max_freq / 2
            self._zoom_text = pg.TextItem(
                f"Zoom in to view spectrogram\n"
                f"(current view: {view_width:.1f}s, max: {MAX_SPECTROGRAM_VIEW:.0f}s)\n\n"
                f"Use Ctrl+I to zoom in",
                color='#666688', anchor=(0.5, 0.5),
            )
            self._zoom_text.setFont(QFont("Segoe UI", 13))
            self._spec_plot.addItem(self._zoom_text)
            self._zoom_text.setPos(mid_t, mid_f)

    # -------------------------------------------------------------------
    # Waveform drawing
    # -------------------------------------------------------------------

    def _draw_waveform(self):
        """Draw waveform into the wave plot."""
        if self._wave_plot is None or self.sound is None:
            return

        # Remove old waveform items
        if self._wave_line is not None:
            try:
                self._wave_plot.removeItem(self._wave_line)
            except Exception:
                pass
            self._wave_line = None
        if self._wave_fill_pos is not None:
            try:
                self._wave_plot.removeItem(self._wave_fill_pos)
            except Exception:
                pass
            self._wave_fill_pos = None
        if self._wave_fill_neg is not None:
            try:
                self._wave_plot.removeItem(self._wave_fill_neg)
            except Exception:
                pass
            self._wave_fill_neg = None
        if self._wave_zero is not None:
            try:
                self._wave_plot.removeItem(self._wave_zero)
            except Exception:
                pass
            self._wave_zero = None

        samples = self.sound.values[0]
        sr = self.sound.sampling_frequency
        total_samples = len(samples)

        i_start = max(0, int(self.view_start * sr))
        i_end = min(total_samples, int(self.view_end * sr))
        if i_end <= i_start:
            return

        vis_samples = samples[i_start:i_end]
        n_vis = len(vis_samples)

        # Target ~2000 points for performance
        target_chunks = 1000

        if n_vis <= target_chunks * 2:
            times = (i_start + np.arange(n_vis)) / sr
            self._wave_line = self._wave_plot.plot(
                times, vis_samples,
                pen=pg.mkPen('#66ccff', width=0.5),
            )
        else:
            # Min/max envelope downsampling
            chunk_size = n_vis // target_chunks
            n_usable = chunk_size * target_chunks
            reshaped = vis_samples[:n_usable].reshape(target_chunks, chunk_size)
            env_min = reshaped.min(axis=1)
            env_max = reshaped.max(axis=1)
            chunk_times = (i_start + (np.arange(target_chunks) + 0.5) * chunk_size) / sr

            # Fill between using two curves
            self._wave_fill_pos = pg.PlotDataItem(chunk_times, env_max, pen=pg.mkPen(None))
            self._wave_fill_neg = pg.PlotDataItem(chunk_times, env_min, pen=pg.mkPen(None))
            fill = pg.FillBetweenItem(self._wave_fill_pos, self._wave_fill_neg,
                                       brush=pg.mkBrush(102, 204, 255, 180))
            self._wave_plot.addItem(self._wave_fill_pos)
            self._wave_plot.addItem(self._wave_fill_neg)
            self._wave_plot.addItem(fill)
            # Track fill for cleanup
            self._wave_line = fill  # reuse for removal

        # Zero line
        self._wave_zero = pg.InfiniteLine(
            pos=0, angle=0,
            pen=pg.mkPen('#555577', width=0.5, style=Qt.PenStyle.SolidLine),
        )
        self._wave_plot.addItem(self._wave_zero)

        # Y limits
        vis_max = max(abs(vis_samples.min()), abs(vis_samples.max()), 0.01)
        self._wave_plot.setYRange(-vis_max, vis_max, padding=0)

    # -------------------------------------------------------------------
    # Formant drawing
    # -------------------------------------------------------------------

    def _draw_formants(self):
        """Draw formant overlay on the spectrogram."""
        # Old scatters already removed by render() before this call

        fd = self.formant_data
        t_mask = ((fd.times >= self.view_start) & (fd.times <= self.view_end))

        for f_idx in range(self.display_n_formants):
            fn = f_idx + 1
            color = FORMANT_COLORS.get(fn, "#ffffff")
            qcolor = QColor(color)

            vals = fd.values[f_idx]
            edited = fd.edited_mask[f_idx]
            valid = ~np.isnan(vals) & t_mask

            if not np.any(valid):
                continue

            is_active = (self.edit_mode and f_idx == self.active_formant)

            # Unedited points: small
            unedited_mask = valid & ~edited
            if np.any(unedited_mask):
                sc = pg.ScatterPlotItem(
                    x=fd.times[unedited_mask], y=vals[unedited_mask],
                    size=3, pen=pg.mkPen(None),
                    brush=pg.mkBrush(qcolor.red(), qcolor.green(), qcolor.blue(), 180),
                )
                self._spec_plot.addItem(sc)
                self._formant_scatters[(f_idx, 'unedited')] = sc

            # Edited points: larger with white edge
            edited_valid = valid & edited
            if np.any(edited_valid):
                sc = pg.ScatterPlotItem(
                    x=fd.times[edited_valid], y=vals[edited_valid],
                    size=4,
                    pen=pg.mkPen('white', width=0.3),
                    brush=pg.mkBrush(qcolor.red(), qcolor.green(), qcolor.blue(), 255),
                )
                self._spec_plot.addItem(sc)
                self._formant_scatters[(f_idx, 'edited')] = sc

    # -------------------------------------------------------------------
    # TextGrid drawing
    # -------------------------------------------------------------------

    def _draw_textgrid(self):
        """Render TextGrid tiers into their plots."""
        tg = self.textgrid_data
        if tg is None or len(self._tier_plots) == 0:
            return

        view_start = self.view_start
        view_end = self.view_end
        view_width = view_end - view_start

        # Pixel width for text LOD
        vb = self._tier_plots[0].getViewBox()
        ax_width_px = vb.width() if vb is not None else 0

        spec_interval_boundaries = set()
        spec_point_times = []

        for plot_i, tier_plot in enumerate(self._tier_plots):
            tier_idx = self._tier_plot_indices[plot_i]
            tier = tg.tiers[tier_idx]
            is_active = (tier_idx == self._active_tier)
            tier_plot.getViewBox().setBackgroundColor('#fff8c4' if is_active else '#ffffff')
            tier_plot.setYRange(0, 1, padding=0)

            # Tier name as centered tick label on left axis
            left_ax = tier_plot.getAxis('left')
            left_ax.setTicks([[(0.5, tier.name)]])
            label_color = '#cc2200' if is_active else '#5577aa'
            font = QFont("Segoe UI", 10)
            if is_active:
                font.setBold(True)
            left_ax.setTickFont(font)
            left_ax.setTextPen(pg.mkPen(label_color))

            if tier.tier_class == "IntervalTier":
                vis = [iv for iv in tier.intervals
                       if iv.xmax > view_start and iv.xmin < view_end]

                if vis:
                    boundaries = set()
                    for iv in vis:
                        boundaries.add(iv.xmin)
                        boundaries.add(iv.xmax)
                        spec_interval_boundaries.add(iv.xmin)
                        spec_interval_boundaries.add(iv.xmax)

                    # Boundary lines on tier
                    for bt in sorted(boundaries):
                        line = pg.InfiniteLine(
                            pos=bt, angle=90,
                            pen=pg.mkPen('#4444aa', width=0.8),
                        )
                        self._add_transient(line, tier_plot)

                    # Labels
                    for iv in vis:
                        if iv.text:
                            if ax_width_px > 0 and view_width > 0:
                                px_w = (iv.xmax - iv.xmin) / view_width * ax_width_px
                                if px_w < 20:
                                    continue
                            mid_t = (iv.xmin + iv.xmax) / 2.0
                            if view_start <= mid_t <= view_end:
                                t = pg.TextItem(iv.text, color='#000000', anchor=(0.5, 0.5))
                                t.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
                                t.setPos(mid_t, 0.5)
                                self._add_transient(t, tier_plot)

            elif tier.tier_class == "TextTier":
                vis = [pt for pt in tier.points
                       if view_start <= pt.time <= view_end]

                if vis:
                    times = [pt.time for pt in vis]
                    spec_point_times.extend(times)

                    for pt_time in times:
                        line = pg.InfiniteLine(
                            pos=pt_time, angle=90,
                            pen=pg.mkPen('#cc4400', width=1.0),
                        )
                        self._add_transient(line, tier_plot)

                    # Diamond markers
                    sc = pg.ScatterPlotItem(
                        x=times, y=[0.5] * len(vis),
                        symbol='d', size=10,
                        pen=pg.mkPen('#cc4400'), brush=pg.mkBrush('#cc4400'),
                    )
                    self._add_transient(sc, tier_plot)

                    # Labels
                    for pt in vis:
                        if pt.mark:
                            t = pg.TextItem("  " + pt.mark, color='#000000', anchor=(0.0, 0.5))
                            t.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
                            t.setPos(pt.time, 0.5)
                            self._add_transient(t, tier_plot)

        # Boundary lines on spectrogram + waveform
        bdry_pen = pg.mkPen('#4488ff', width=1.5, style=Qt.PenStyle.DashLine)
        for bt in sorted(spec_interval_boundaries):
            line = pg.InfiniteLine(pos=bt, angle=90, pen=bdry_pen)
            self._add_transient(line, self._spec_plot)
            if self._wave_plot is not None:
                line2 = pg.InfiniteLine(pos=bt, angle=90, pen=bdry_pen)
                self._add_transient(line2, self._wave_plot)

        # Point markers on spectrogram + waveform
        pt_pen = pg.mkPen('#ff6622', width=0.7, style=Qt.PenStyle.DashLine)
        for pt_time in spec_point_times:
            line = pg.InfiniteLine(pos=pt_time, angle=90, pen=pt_pen)
            self._add_transient(line, self._spec_plot)
            if self._wave_plot is not None:
                line2 = pg.InfiniteLine(pos=pt_time, angle=90, pen=pt_pen)
                self._add_transient(line2, self._wave_plot)

    # -------------------------------------------------------------------
    # Title
    # -------------------------------------------------------------------

    def _update_title(self):
        """Update title text on the topmost plot."""
        if self._title_item is not None:
            target = self._wave_plot if self._wave_plot is not None else self._spec_plot
            try:
                target.removeItem(self._title_item)
            except Exception:
                pass
            self._title_item = None

        title = os.path.basename(self._filepath) if hasattr(self, "_filepath") else ""
        if self.edit_mode:
            fn = self.active_formant + 1
            color = FORMANT_COLORS.get(fn, "#ffffff")
            title += f"  |  EDIT MODE — Drawing {FORMANT_LABELS[fn]}"
        else:
            color = "#eeeeee"

        target = self._wave_plot if self._wave_plot is not None else self._spec_plot
        self._title_item = pg.TextItem(title, color=color, anchor=(0.5, 1.0))
        font = QFont("Segoe UI", 11)
        if self.edit_mode:
            font.setBold(True)
        self._title_item.setFont(font)
        target.addItem(self._title_item)
        # Position at top-center of view
        vr = target.viewRange()
        mid_x = (vr[0][0] + vr[0][1]) / 2
        top_y = vr[1][1]
        self._title_item.setPos(mid_x, top_y)

    # -------------------------------------------------------------------
    # Selection overlay
    # -------------------------------------------------------------------

    def _draw_selection_overlay(self):
        """Draw selection/boundary highlights."""
        self._clear_overlay_items()

        # Time selection highlight
        if self._selection_start is not None and self._selection_end is not None:
            for plot in [self._spec_plot, self._wave_plot] + self._tier_plots:
                if plot is not None:
                    alpha = 40 if plot in self._tier_plots else 65
                    region = pg.LinearRegionItem(
                        values=[self._selection_start, self._selection_end],
                        movable=False,
                        brush=pg.mkBrush(51, 102, 204, alpha),
                        pen=pg.mkPen(None),
                    )
                    self._add_overlay(region, plot)

        # Selected interval / point highlight
        if self._selected_interval is not None:
            sel_tier_idx, sel_i = self._selected_interval
            if (self.textgrid_data is not None
                    and 0 <= sel_tier_idx < len(self.textgrid_data.tiers)):
                tier = self.textgrid_data.tiers[sel_tier_idx]
                if (tier.tier_class == "IntervalTier"
                        and sel_i < len(tier.intervals)):
                    iv = tier.intervals[sel_i]
                    sel_plot = self._plot_for_tier(sel_tier_idx)
                    if sel_plot is not None:
                        region = pg.LinearRegionItem(
                            values=[iv.xmin, iv.xmax],
                            movable=False,
                            brush=pg.mkBrush(51, 102, 204, 65),
                            pen=pg.mkPen(None),
                        )
                        self._add_overlay(region, sel_plot)
                elif (tier.tier_class == "TextTier"
                      and sel_i < len(tier.points)):
                    pt = tier.points[sel_i]
                    sel_plot = self._plot_for_tier(sel_tier_idx)
                    if sel_plot is not None:
                        red_pen = pg.mkPen('#ff0000', width=2.5)
                        line = pg.InfiniteLine(
                            pos=pt.time, angle=90, pen=red_pen)
                        self._add_overlay(line, sel_plot)
                    # Also show on spec and wave plots
                    for p in [self._spec_plot, self._wave_plot]:
                        if p is not None:
                            red_pen = pg.mkPen('#ff0000', width=2.5)
                            line = pg.InfiniteLine(
                                pos=pt.time, angle=90, pen=red_pen)
                            self._add_overlay(line, p)

        # Selected boundary highlight + shadows
        if self._selected_boundary is not None:
            sel_tier, sel_time = self._selected_boundary
            if self.view_start <= sel_time <= self.view_end:
                red_pen = pg.mkPen('#ff0000', width=2.5)
                # On spectrogram
                line = pg.InfiniteLine(pos=sel_time, angle=90, pen=red_pen)
                self._add_overlay(line, self._spec_plot)
                # On waveform
                if self._wave_plot is not None:
                    line2 = pg.InfiniteLine(pos=sel_time, angle=90, pen=red_pen)
                    self._add_overlay(line2, self._wave_plot)
                # On selected tier
                sel_tier_plot = self._plot_for_tier(sel_tier)
                if sel_tier_plot is not None:
                    line3 = pg.InfiniteLine(pos=sel_time, angle=90, pen=red_pen)
                    self._add_overlay(line3, sel_tier_plot)

                # Shadow boundaries on OTHER tiers
                for other_idx, other_plot in enumerate(self._tier_plots):
                    if self._tier_plot_indices[other_idx] == sel_tier:
                        continue
                    dash_pen = pg.mkPen('#888888', width=1.0, style=Qt.PenStyle.DashLine)
                    line = pg.InfiniteLine(pos=sel_time, angle=90, pen=dash_pen)
                    self._add_overlay(line, other_plot)
                    # Clickable circle
                    sc = pg.ScatterPlotItem(
                        x=[sel_time], y=[0.5], size=10,
                        pen=pg.mkPen('white', width=1.0),
                        brush=pg.mkBrush('#6688cc'),
                    )
                    self._add_overlay(sc, other_plot)

    # -------------------------------------------------------------------
    # TextGrid editing helpers
    # -------------------------------------------------------------------

    def _tier_index_for_plot(self, plot):
        """Return the actual tier index for a given PlotItem, or None."""
        for i, tp in enumerate(self._tier_plots):
            if tp is plot:
                return self._tier_plot_indices[i]
        return None

    def _plot_for_tier(self, tier_idx):
        """Return the PlotItem for a given tier index, or None if hidden."""
        try:
            pi = self._tier_plot_indices.index(tier_idx)
            return self._tier_plots[pi]
        except ValueError:
            return None

    # Keep the old name for backward compat with any internal usage
    def _tier_index_for_axes(self, axes):
        return self._tier_index_for_plot(axes)

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
        vb = self._spec_plot.getViewBox()
        r = vb.viewRange()
        view_w = r[0][1] - r[0][0]
        px_w = vb.width()
        return pixels * view_w / px_w if px_w > 0 else 0.01

    def _add_boundary(self, tier_idx, time):
        """Add a boundary at the given time in the specified tier."""
        tier = self.textgrid_data.tiers[tier_idx]
        if tier.tier_class == "IntervalTier":
            for i, iv in enumerate(tier.intervals):
                if iv.xmin < time < iv.xmax:
                    old_text = iv.text
                    tier.intervals[i] = Interval(iv.xmin, time, old_text)
                    tier.intervals.insert(i + 1, Interval(time, iv.xmax, ""))
                    return True
        elif tier.tier_class == "TextTier":
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
        """Delete a boundary, merging adjacent intervals."""
        tier = self.textgrid_data.tiers[tier_idx]
        if tier.tier_class == "IntervalTier":
            if boundary_time == tier.xmin or boundary_time == tier.xmax:
                return False
            left_idx = None
            right_idx = None
            for i, iv in enumerate(tier.intervals):
                if iv.xmax == boundary_time:
                    left_idx = i
                if iv.xmin == boundary_time:
                    right_idx = i
            if left_idx is not None and right_idx is not None:
                # Merge labels from both sides of the deleted boundary
                left_text = tier.intervals[left_idx].text.strip()
                right_text = tier.intervals[right_idx].text.strip()
                if left_text and right_text:
                    merged = f"{left_text} {right_text}"
                else:
                    merged = left_text or right_text
                tier.intervals[left_idx] = Interval(
                    tier.intervals[left_idx].xmin,
                    tier.intervals[right_idx].xmax,
                    merged,
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
            prev_time = tier.xmin
            next_time = tier.xmax
            for pt in tier.points:
                if pt.time < boundary_time and pt.time > prev_time:
                    prev_time = pt.time
                if pt.time > boundary_time and pt.time < next_time:
                    next_time = pt.time
            self._drag_min_time = prev_time + epsilon
            self._drag_max_time = next_time - epsilon

    def _find_aligned_boundaries(self, source_tier_idx, boundary_time):
        """Find boundaries on other tiers at exactly the same time."""
        aligned = []
        for i, tier in enumerate(self.textgrid_data.tiers):
            if i == source_tier_idx:
                continue
            if tier.tier_class == "IntervalTier":
                for iv in tier.intervals:
                    if iv.xmin == boundary_time and iv.xmin != tier.xmin:
                        aligned.append((i, boundary_time))
                        break
                    if iv.xmax == boundary_time and iv.xmax != tier.xmax:
                        aligned.append((i, boundary_time))
                        break
            elif tier.tier_class == "TextTier":
                for pt in tier.points:
                    if pt.time == boundary_time:
                        aligned.append((i, boundary_time))
                        break
        return aligned

    def _compute_multi_drag_constraints(self, all_tiers_times):
        """Compute drag constraints as intersection across multiple tiers."""
        overall_min = -float('inf')
        overall_max = float('inf')
        for tier_idx, bt in all_tiers_times:
            self._compute_drag_constraints(tier_idx, bt)
            overall_min = max(overall_min, self._drag_min_time)
            overall_max = min(overall_max, self._drag_max_time)
        self._drag_min_time = overall_min
        self._drag_max_time = overall_max

    # -------------------------------------------------------------------
    # Interval/point selection
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
                    if self._label_edit is not None:
                        self._label_edit.setEnabled(True)
                        self._label_edit.setText(iv.text)
                    return True
        elif tier.tier_class == "TextTier":
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

    # -------------------------------------------------------------------
    # Mouse event coordinate mapping
    # -------------------------------------------------------------------

    def _map_event_to_data(self, event):
        """Map a QMouseEvent to data coordinates.

        Returns (plot_name, plot, time, y_value, tier_idx) or
        (None, None, None, None, None) if outside all plots.

        plot_name: 'spec', 'wave', or 'tier'
        """
        pos = event.position() if hasattr(event, 'position') else event.pos()
        scene_pos = self._glw.mapToScene(pos.toPoint())

        # Check each plot
        for name, plot in self._iter_plots():
            vb = plot.getViewBox()
            if vb is None:
                continue
            scene_rect = vb.sceneBoundingRect()
            if scene_rect.contains(scene_pos):
                mouse_point = vb.mapSceneToView(scene_pos)
                tier_idx = self._tier_index_for_plot(plot) if name == 'tier' else None
                return name, plot, mouse_point.x(), mouse_point.y(), tier_idx

        return None, None, None, None, None

    def _iter_plots(self):
        """Yield (name, plot) for all active plots."""
        if self._wave_plot is not None:
            yield ('wave', self._wave_plot)
        if self._spec_plot is not None:
            yield ('spec', self._spec_plot)
        for tp in self._tier_plots:
            yield ('tier', tp)

    # -------------------------------------------------------------------
    # Mouse scroll — zoom
    # -------------------------------------------------------------------

    def _on_scroll(self, event):
        """Mouse wheel zoom — centered on cursor position."""
        if self.sound is None:
            return
        _, _, time, _, _ = self._map_event_to_data(event)
        if time is None:
            return
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom(1.0 / ZOOM_FACTOR, center_time=time)
        elif delta < 0:
            self.zoom(ZOOM_FACTOR, center_time=time)
        else:
            return
        if self._on_view_changed_callback:
            self._on_view_changed_callback()
        self._render_timer.start()

    # -------------------------------------------------------------------
    # Mouse double click
    # -------------------------------------------------------------------

    def _on_mouse_double_click(self, event):
        """Handle double-click for label editing."""
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self.textgrid_data is None:
            return
        name, plot, time, y, tier_idx = self._map_event_to_data(event)
        if name == 'tier' and tier_idx is not None and time is not None:
            old_active = self._active_tier
            self._active_tier = tier_idx
            self._select_interval(tier_idx, time)
            if self._label_edit is not None:
                self._label_edit.setFocus()
                self._label_edit.selectAll()
            if old_active != tier_idx:
                self.render()
            else:
                self._draw_selection_overlay()

    # -------------------------------------------------------------------
    # Mouse press
    # -------------------------------------------------------------------

    def _on_mouse_press(self, event):
        is_right = (event.button() == Qt.MouseButton.RightButton)
        is_left = (event.button() == Qt.MouseButton.LeftButton)

        if is_right:
            if not self.edit_mode:
                return
        elif not is_left:
            return

        name, plot, time, y, tier_idx = self._map_event_to_data(event)

        # --- TextGrid tier click (left-click only) ---
        if is_left and self.textgrid_data is not None and name == 'tier' and tier_idx is not None and time is not None:
            old_active = self._active_tier
            self._active_tier = tier_idx

            # Check if clicking on a shadow boundary circle
            if self._selected_boundary is not None:
                sel_tier, sel_time = self._selected_boundary
                if tier_idx != sel_tier:
                    threshold = self._time_threshold_for_pixels(8)
                    if abs(time - sel_time) <= threshold:
                        if self._add_boundary(tier_idx, sel_time):
                            if self._on_textgrid_edited is not None:
                                self._on_textgrid_edited()
                            self._selected_boundary = (tier_idx, sel_time)
                            self._selected_interval = None
                            self.render()
                        return

            # Check if near a boundary
            tier = self.textgrid_data.tiers[tier_idx]
            bt, dist = self._find_nearest_boundary(tier_idx, time)
            threshold = self._time_threshold_for_pixels(5)

            if tier.tier_class == "TextTier":
                if bt is not None and dist <= threshold:
                    self._select_interval(tier_idx, time)
                    self._selected_boundary = None
                    # Don't auto-focus label edit for TextTier points —
                    # this allows Delete key to reach the main window for
                    # point deletion.  User can click the label field to
                    # edit if needed.
                    self._start_boundary_drag(tier_idx, bt, event)
                    if old_active != tier_idx:
                        self.render()
                    return
                else:
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

            # IntervalTier
            if bt is not None and dist <= threshold:
                self._selected_interval = None
                if self._label_edit is not None:
                    self._label_edit.setEnabled(False)
                    self._label_edit.clear()
                if tier.tier_class == "IntervalTier" and (bt == tier.xmin or bt == tier.xmax):
                    self._selected_boundary = (tier_idx, bt)
                    if old_active != tier_idx:
                        self.render()
                    else:
                        self._draw_selection_overlay()
                    return
                # Draggable boundary
                self._selected_boundary = None
                self._start_boundary_drag(tier_idx, bt, event)
                self.render()
                return
            else:
                self._select_interval(tier_idx, time)
                # Give focus to label edit so user can type immediately
                if self._label_edit is not None and self._label_edit.isEnabled():
                    self._label_edit.setFocus()
                if old_active != tier_idx:
                    self.render()
                else:
                    self._draw_selection_overlay()
                return

        # --- Spectrogram/waveform click (non-edit mode) ---
        if not self.edit_mode:
            if name in ('spec', 'wave') and time is not None:
                self._click_time = time
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
                self._spec_drag_start = time
                if self._status_callback:
                    if name == 'spec':
                        self._status_callback(
                            f"Time: {time:.4f} s  |  Frequency: {y:.1f} Hz")
                    else:
                        self._status_callback(f"Time: {time:.4f} s")
            return

        # --- Edit mode: formant drawing/erasing (only on spectrogram) ---
        if name != 'spec' or time is None:
            return
        self.is_drawing = True
        self._is_erasing = is_right
        self._last_frame_idx = -1
        self._last_frame_freq = None
        self._stroke_times = []
        self._stroke_freqs = []
        self._stroke_changes = []

        # Create stroke scatter
        fn = self.active_formant + 1
        color = "#888888" if self._is_erasing else FORMANT_COLORS.get(fn, "#ffffff")
        qc = QColor(color)
        self._stroke_scatter = pg.ScatterPlotItem(
            size=5, pen=pg.mkPen('white', width=0.3),
            brush=pg.mkBrush(qc.red(), qc.green(), qc.blue(), 255),
        )
        self._spec_plot.addItem(self._stroke_scatter)
        self._apply_edit(time, y)

    def _snap_to_boundary(self, time, exclude_tier_indices, exclude_time):
        """If snap is enabled, return the nearest boundary on any tier NOT
        in *exclude_tier_indices* within tolerance, else return *time*."""
        if (self._snap_enabled_cb is None
                or not self._snap_enabled_cb.isChecked()):
            return time
        if isinstance(exclude_tier_indices, int):
            exclude_tier_indices = {exclude_tier_indices}
        tolerance = self._snap_tolerance_spin.value()
        best_t = time
        best_dist = tolerance
        for ti, tier in enumerate(self.textgrid_data.tiers):
            if ti in exclude_tier_indices:
                continue
            if tier.tier_class == "IntervalTier":
                for iv in tier.intervals:
                    for bt in (iv.xmin, iv.xmax):
                        d = abs(bt - time)
                        if d < best_dist:
                            best_dist = d
                            best_t = bt
            elif tier.tier_class == "TextTier":
                for pt in tier.points:
                    d = abs(pt.time - time)
                    if d < best_dist:
                        best_dist = d
                        best_t = pt.time
        return best_t

    def _start_boundary_drag(self, tier_idx, bt, event):
        """Start a boundary drag operation."""
        self._dragging_boundary = True
        self._drag_tier_index = tier_idx
        self._drag_original_time = bt

        shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if shift:
            aligned = self._find_aligned_boundaries(tier_idx, bt)
            self._drag_aligned = aligned
            self._compute_multi_drag_constraints([(tier_idx, bt)] + aligned)
        else:
            self._drag_aligned = []
            self._compute_drag_constraints(tier_idx, bt)

        # Create drag feedback lines
        self._drag_lines = []
        red_pen = pg.mkPen('#ff0000', width=2.5)
        for p in [self._spec_plot, self._wave_plot]:
            if p is not None:
                line = pg.InfiniteLine(pos=bt, angle=90, pen=red_pen, movable=False)
                p.addItem(line)
                self._drag_lines.append((p, line))
        drag_tier_plot = self._plot_for_tier(tier_idx)
        if drag_tier_plot is not None:
            line = pg.InfiniteLine(pos=bt, angle=90, pen=red_pen, movable=False)
            drag_tier_plot.addItem(line)
            self._drag_lines.append((drag_tier_plot, line))
        for a_tier_idx, _ in self._drag_aligned:
            a_plot = self._plot_for_tier(a_tier_idx)
            if a_plot is not None:
                line = pg.InfiniteLine(pos=bt, angle=90, pen=red_pen, movable=False)
                a_plot.addItem(line)
                self._drag_lines.append((a_plot, line))

    # -------------------------------------------------------------------
    # Mouse release
    # -------------------------------------------------------------------

    def _on_mouse_release(self, event):
        # --- End boundary drag ---
        if self._dragging_boundary:
            _, _, time, _, _ = self._map_event_to_data(event)
            final_time = self._drag_original_time
            if time is not None:
                final_time = max(self._drag_min_time,
                                 min(self._drag_max_time, time))
                # Snap to nearest boundary on tiers NOT in the drag group
                exclude = {self._drag_tier_index}
                for a_ti, _ in self._drag_aligned:
                    exclude.add(a_ti)
                snapped = self._snap_to_boundary(
                    final_time, exclude,
                    self._drag_original_time)
                # Only snap if result is still within drag constraints
                if self._drag_min_time <= snapped <= self._drag_max_time:
                    final_time = snapped
            self._move_boundary(self._drag_tier_index,
                                self._drag_original_time, final_time)
            for a_tier_idx, a_time in self._drag_aligned:
                self._move_boundary(a_tier_idx, a_time, final_time)
            # For TextTier points, keep _selected_interval (set during
            # mouse press) so Delete key can find the point.  Only set
            # _selected_boundary for IntervalTier boundaries.
            tier = self.textgrid_data.tiers[self._drag_tier_index]
            if tier.tier_class == "TextTier":
                # Re-select the moved point so _selected_interval is up
                # to date with the (possibly dragged) new time.
                self._select_interval(self._drag_tier_index, final_time)
            else:
                self._selected_boundary = (self._drag_tier_index, final_time)
            if self._on_textgrid_edited is not None:
                self._on_textgrid_edited()
            # Clean up drag lines
            for p, line in self._drag_lines:
                try:
                    p.removeItem(line)
                except Exception:
                    pass
            self._drag_lines = []
            self._drag_aligned = []
            self._dragging_boundary = False
            self._drag_tier_index = None
            self._drag_original_time = None
            self.render()
            return

        # --- End spectrogram drag selection ---
        if self._spec_dragging:
            self._spec_dragging = False
            # Remove drag selection regions
            for p, region in self._spec_sel_regions:
                try:
                    p.removeItem(region)
                except Exception:
                    pass
            self._spec_sel_regions = []
            # Determine if real drag or just click
            _, _, time, _, _ = self._map_event_to_data(event)
            if time is not None and self._spec_drag_start is not None:
                drag_dist = abs(time - self._spec_drag_start)
                threshold = self._time_threshold_for_pixels(5)
                if drag_dist > threshold:
                    self._selection_start = min(self._spec_drag_start, time)
                    self._selection_end = max(self._spec_drag_start, time)
                else:
                    self._selection_start = None
                    self._selection_end = None
            self._spec_drag_start = None
            self._draw_selection_overlay()
            return

        # --- End formant edit stroke ---
        if not self.is_drawing:
            return
        self.is_drawing = False
        self._last_frame_idx = -1
        self._last_frame_freq = None

        if self._stroke_scatter is not None:
            try:
                self._spec_plot.removeItem(self._stroke_scatter)
            except Exception:
                pass
            self._stroke_scatter = None
        self._stroke_times = []
        self._stroke_freqs = []

        if self.edit_mode:
            if self._stroke_changes:
                desc = "Erase" if self._is_erasing else "Draw"
                entry = UndoEntry(desc, list(self._stroke_changes))
                self._undo_stack.append(entry)
                self._redo_stack.clear()
                if len(self._undo_stack) > MAX_UNDO_STEPS:
                    self._undo_stack.pop(0)
                self._stroke_changes = []
            self._is_erasing = False
            if self._on_formant_edited is not None:
                self._on_formant_edited()
            self.render()

    # -------------------------------------------------------------------
    # Mouse move
    # -------------------------------------------------------------------

    def _on_mouse_move(self, event):
        # --- Boundary drag ---
        if self._dragging_boundary:
            _, _, time, _, _ = self._map_event_to_data(event)
            if time is not None:
                new_time = max(self._drag_min_time,
                               min(self._drag_max_time, time))
                for p, line in self._drag_lines:
                    line.setValue(new_time)
            return

        # --- Spectrogram drag selection ---
        if self._spec_dragging:
            _, _, time, _, _ = self._map_event_to_data(event)
            if time is not None:
                # Lazy init regions on first move
                if not self._spec_sel_regions:
                    x0 = min(self._spec_drag_start, time)
                    x1 = max(self._spec_drag_start, time)
                    for plot in [self._spec_plot, self._wave_plot] + self._tier_plots:
                        if plot is None:
                            continue
                        alpha = 40 if plot in self._tier_plots else 65
                        region = pg.LinearRegionItem(
                            values=[x0, x1], movable=False,
                            brush=pg.mkBrush(51, 102, 204, alpha),
                            pen=pg.mkPen(None),
                        )
                        plot.addItem(region)
                        self._spec_sel_regions.append((plot, region))
                else:
                    x0 = min(self._spec_drag_start, time)
                    x1 = max(self._spec_drag_start, time)
                    for _, region in self._spec_sel_regions:
                        region.setRegion([x0, x1])
            return

        # --- Crosshair + readout ---
        if self._playback_playing:
            return
        name, plot, time, y, tier_idx = self._map_event_to_data(event)

        if name == 'spec' and time is not None:
            self._hover_time = time
            self._update_crosshair(time, y, on_spectrogram=True)
            if self._status_callback and not self.is_drawing:
                self._status_callback(
                    f"Time: {time:.4f} s  |  Frequency: {y:.1f} Hz")
        elif name == 'wave' and time is not None:
            self._hover_time = time
            self._update_crosshair(time, y, on_spectrogram=False)
            if self._status_callback and not self.is_drawing:
                rms = self._get_rms_at_time(time)
                rms_db = 20 * np.log10(rms + 1e-20)
                self._status_callback(
                    f"Time: {time:.4f} s  |  RMS: {rms:.4f} ({rms_db:.1f} dB)")
        elif self._crosshair_visible:
            self._hide_crosshair()

        # Formant editing
        if not self.is_drawing or not self.edit_mode:
            return
        if name != 'spec' or time is None:
            return
        self._apply_edit(time, y)

    def _update_crosshair(self, xdata, ydata, on_spectrogram=True):
        """Move crosshair lines to cursor position."""
        if self._crosshair_v is None:
            return
        self._crosshair_v.setValue(xdata)
        if on_spectrogram and ydata is not None:
            self._crosshair_h.setValue(ydata)

        if not self._crosshair_visible:
            self._crosshair_v.setVisible(True)
            self._crosshair_h.setVisible(on_spectrogram)
            if self._crosshair_wave_v is not None:
                self._crosshair_wave_v.setVisible(True)
            self._crosshair_visible = True
        else:
            self._crosshair_h.setVisible(on_spectrogram)

        if self._crosshair_wave_v is not None:
            self._crosshair_wave_v.setValue(xdata)

    def _get_rms_at_time(self, time_s, window=0.01):
        """Compute RMS amplitude in a short window centered on time_s."""
        if self.sound is None:
            return 0.0
        sr = self.sound.sampling_frequency
        samples = self.sound.values[0]
        half_win = int(window * 0.5 * sr)
        center = int(time_s * sr)
        i0 = max(0, center - half_win)
        i1 = min(len(samples), center + half_win)
        if i1 <= i0:
            return 0.0
        chunk = samples[i0:i1]
        return float(np.sqrt(np.mean(chunk ** 2)))

    def _hide_crosshair(self):
        """Hide crosshair when cursor leaves."""
        if self._crosshair_h is None:
            return
        self._crosshair_h.setVisible(False)
        self._crosshair_v.setVisible(False)
        if self._crosshair_wave_v is not None:
            self._crosshair_wave_v.setVisible(False)
        self._crosshair_visible = False

    # -------------------------------------------------------------------
    # Audio playback
    # -------------------------------------------------------------------

    def play_audio(self, start_time, end_time):
        """Play audio from start_time to end_time with animated cursor."""
        if self.sound is None:
            return
        if self._playback_playing:
            self.stop_playback()

        sr = int(self.sound.sampling_frequency)
        samples = self.sound.values[0]
        start_sample = max(0, int(start_time * sr))
        end_sample = min(len(samples), int(end_time * sr))
        if end_sample <= start_sample:
            return

        audio_chunk = samples[start_sample:end_sample].astype('float32')
        self._playback_start_time = start_time
        self._playback_end_time = end_time
        self._playback_playing = True

        # Use Qt's native QAudioSink instead of sounddevice/PortAudio.
        # PortAudio's bundled DLL in the PyInstaller exe defaults to the
        # high-latency MME backend, causing short segments to be clipped.
        # QAudioSink uses the Windows audio stack directly and works
        # identically in both the Python script and the frozen exe.
        fmt = QAudioFormat()
        fmt.setSampleRate(sr)
        fmt.setChannelCount(1)
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Float)

        # Pad with ~50ms of silence so the audio sink fully flushes the
        # last real samples before transitioning to IdleState.  Without
        # this, short segments can appear to cut off early because the
        # sink stops before the tail of the buffer reaches the DAC.
        pad_samples = int(sr * 0.05)
        silence = np.zeros(pad_samples, dtype='float32')
        padded = np.concatenate([audio_chunk, silence])

        self._playback_audio_buf = QBuffer()
        self._playback_audio_buf.setData(QByteArray(padded.tobytes()))
        self._playback_audio_buf.open(QIODevice.OpenModeFlag.ReadOnly)

        self._playback_audio_sink = QAudioSink(fmt)
        self._playback_audio_sink.stateChanged.connect(self._on_audio_state_changed)
        self._playback_audio_sink.start(self._playback_audio_buf)
        self._playback_start_wall = _time.monotonic()

        # Create green cursor lines on all plots
        green_pen = pg.mkPen('#00ff44', width=1.5)
        self._playback_cursors = []
        for _, plot in self._iter_plots():
            line = pg.InfiniteLine(pos=start_time, angle=90, pen=green_pen, movable=False)
            plot.addItem(line)
            self._playback_cursors.append((plot, line))

        self._playback_timer = QTimer()
        self._playback_timer.timeout.connect(self._update_playback_cursor)
        self._playback_timer.start(30)

    def _on_audio_state_changed(self, state):
        """Called when QAudioSink transitions state (e.g. idle = done)."""
        from PyQt6.QtMultimedia import QAudio
        if state == QAudio.State.IdleState and self._playback_playing:
            self._cleanup_playback_visuals()

    def _cleanup_playback_visuals(self):
        """Remove cursor lines, timer, and audio resources."""
        self._playback_playing = False
        if self._playback_timer is not None:
            self._playback_timer.stop()
            self._playback_timer = None
        for p, line in self._playback_cursors:
            try:
                p.removeItem(line)
            except Exception:
                pass
        self._playback_cursors = []
        if self._playback_audio_sink is not None:
            self._playback_audio_sink.stop()
            self._playback_audio_sink = None
        if self._playback_audio_buf is not None:
            self._playback_audio_buf.close()
            self._playback_audio_buf = None
        self.render()

    def stop_playback(self):
        """Stop audio playback immediately (user-initiated)."""
        if not self._playback_playing:
            return
        self._cleanup_playback_visuals()

    def _update_playback_cursor(self):
        """Timer callback: update playback cursor position."""
        if not self._playback_playing:
            return
        elapsed = _time.monotonic() - self._playback_start_wall
        current_time = self._playback_start_time + elapsed
        current_time = min(current_time, self._playback_end_time)
        for _, line in self._playback_cursors:
            line.setValue(current_time)

    # -------------------------------------------------------------------
    # Formant editing
    # -------------------------------------------------------------------

    def _apply_edit(self, time_s, freq_hz):
        """Apply an edit at the given time/frequency position."""
        if self.formant_data is None:
            return
        fd = self.formant_data
        f_idx = self.active_formant
        frame_idx = np.argmin(np.abs(fd.times - time_s))
        if frame_idx == self._last_frame_idx:
            return

        if self._is_erasing:
            if (self._last_frame_idx >= 0
                    and abs(frame_idx - self._last_frame_idx) > 1):
                prev_fi = self._last_frame_idx
                step = 1 if frame_idx > prev_fi else -1
                n_steps = abs(frame_idx - prev_fi)
                for k in range(1, n_steps):
                    mid_fi = prev_fi + k * step
                    old_val = float(fd.values[f_idx, mid_fi])
                    old_mask = bool(fd.edited_mask[f_idx, mid_fi])
                    new_val = float(fd.original_values[f_idx, mid_fi])
                    fd.values[f_idx, mid_fi] = new_val
                    fd.edited_mask[f_idx, mid_fi] = False
                    self._stroke_changes.append(
                        (f_idx, mid_fi, old_val, old_mask, new_val, False))
                    self._quick_draw_edit_point(mid_fi, new_val)

            old_val = float(fd.values[f_idx, frame_idx])
            old_mask = bool(fd.edited_mask[f_idx, frame_idx])
            new_val = float(fd.original_values[f_idx, frame_idx])
            fd.values[f_idx, frame_idx] = new_val
            fd.edited_mask[f_idx, frame_idx] = False
            self._stroke_changes.append(
                (f_idx, frame_idx, old_val, old_mask, new_val, False))
            self._last_frame_idx = frame_idx
            self._last_frame_freq = new_val
            self._quick_draw_edit_point(frame_idx, new_val)
            return

        # Normal drawing
        if (self._last_frame_idx >= 0
                and abs(frame_idx - self._last_frame_idx) > 1
                and self._last_frame_freq is not None):
            prev_fi = self._last_frame_idx
            prev_freq = self._last_frame_freq
            step = 1 if frame_idx > prev_fi else -1
            n_steps = abs(frame_idx - prev_fi)
            for k in range(1, n_steps):
                mid_fi = prev_fi + k * step
                t = k / n_steps
                mid_freq = prev_freq + t * (freq_hz - prev_freq)
                old_val = float(fd.values[f_idx, mid_fi])
                old_mask = bool(fd.edited_mask[f_idx, mid_fi])
                fd.set_value(f_idx, mid_fi, mid_freq)
                self._stroke_changes.append(
                    (f_idx, mid_fi, old_val, old_mask, mid_freq, True))
                self._quick_draw_edit_point(mid_fi, mid_freq)

        old_val = float(fd.values[f_idx, frame_idx])
        old_mask = bool(fd.edited_mask[f_idx, frame_idx])
        fd.set_value(f_idx, frame_idx, freq_hz)
        self._stroke_changes.append(
            (f_idx, frame_idx, old_val, old_mask, freq_hz, True))
        self._last_frame_idx = frame_idx
        self._last_frame_freq = freq_hz
        self._quick_draw_edit_point(frame_idx, freq_hz)

    def _quick_draw_edit_point(self, frame_idx, freq_hz):
        """Draw edited point using scatter update."""
        if self._stroke_scatter is None:
            return
        t = self.formant_data.times[frame_idx]
        self._stroke_times.append(t)
        self._stroke_freqs.append(freq_hz)
        self._stroke_scatter.setData(
            x=self._stroke_times, y=self._stroke_freqs)

    # -------------------------------------------------------------------
    # Undo / Redo
    # -------------------------------------------------------------------

    def undo(self):
        """Undo the last formant edit."""
        if not self._undo_stack or self.formant_data is None:
            return False
        entry = self._undo_stack.pop()
        fd = self.formant_data
        for f_idx, frame_idx, old_val, old_mask, new_val, new_mask in entry.changes:
            fd.values[f_idx, frame_idx] = old_val
            fd.edited_mask[f_idx, frame_idx] = old_mask
        self._redo_stack.append(entry)
        self.render()
        return True

    def redo(self):
        """Redo the last undone formant edit."""
        if not self._redo_stack or self.formant_data is None:
            return False
        entry = self._redo_stack.pop()
        fd = self.formant_data
        for f_idx, frame_idx, old_val, old_mask, new_val, new_mask in entry.changes:
            fd.values[f_idx, frame_idx] = new_val
            fd.edited_mask[f_idx, frame_idx] = new_mask
        self._undo_stack.append(entry)
        self.render()
        return True




# ---------------------------------------------------------------------------
# Jump-to-click slider
# ---------------------------------------------------------------------------

class _JumpSlider(QSlider):
    """QSlider subclass that jumps directly to the clicked position."""

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # Calculate the value at the click position
            opt_width = self.width()
            click_x = event.position().x()
            val = self.minimum() + (self.maximum() - self.minimum()) * click_x / opt_width
            val = max(self.minimum(), min(self.maximum(), round(val)))
            self.setValue(int(val))
            event.accept()
            self.sliderReleased.emit()
        else:
            super().mousePressEvent(event)


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
            QCheckBox::indicator {
                width: 14px; height: 14px;
                border: 2px solid #888; border-radius: 3px;
                background-color: transparent;
            }
            QCheckBox::indicator:checked {
                border-color: #6699cc;
                background-color: #4477aa;
            }
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

        self.edit_btn = QPushButton("✏  EDIT MODE  (Ctrl+E)")
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

        # Undo button
        self.undo_btn = QPushButton("Undo Last Edit  (Ctrl+Z)")
        edit_layout.addWidget(self.undo_btn)

        # Reset buttons
        self.reset_current_btn = QPushButton("Reset Current Formant")
        self.reset_all_btn = QPushButton("Reset All Edits")
        edit_layout.addWidget(self.reset_current_btn)
        edit_layout.addWidget(self.reset_all_btn)

        layout.addWidget(edit_group)

        # --- TextGrid Tiers ---
        self._tier_group = QGroupBox("TextGrid Tiers")
        self._tier_group_layout = QVBoxLayout(self._tier_group)
        self._tier_checkboxes = []  # list of QCheckBox
        self._tier_group.setVisible(False)  # hidden until TextGrid loaded
        layout.addWidget(self._tier_group)

        # --- Boundary Snap ---
        snap_group = QGroupBox("Boundary Snap")
        snap_layout = QVBoxLayout(snap_group)

        self.snap_enabled_cb = QCheckBox("Snap to nearest boundary")
        self.snap_enabled_cb.setChecked(False)
        snap_layout.addWidget(self.snap_enabled_cb)

        tol_row = QHBoxLayout()
        tol_row.addWidget(QLabel("Tolerance (s):"))
        self.snap_tolerance_spin = QDoubleSpinBox()
        self.snap_tolerance_spin.setRange(0.001, 0.200)
        self.snap_tolerance_spin.setValue(0.010)
        self.snap_tolerance_spin.setSingleStep(0.005)
        self.snap_tolerance_spin.setDecimals(3)
        tol_row.addWidget(self.snap_tolerance_spin)
        snap_layout.addLayout(tol_row)

        layout.addWidget(snap_group)

        layout.addStretch()

    def _make_slider(self, label_text, min_val, max_val, default, parent_layout):
        lbl = QLabel(f"{label_text}: {default}")
        parent_layout.addWidget(lbl)
        slider = _JumpSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default)
        slider.setPageStep(1)
        slider.valueChanged.connect(lambda v, l=lbl, t=label_text: l.setText(f"{t}: {v}"))
        parent_layout.addWidget(slider)
        return slider

    def populate_tier_checkboxes(self, tier_names, hidden_set):
        """Populate tier visibility checkboxes from tier names."""
        # Clear old checkboxes
        for cb in self._tier_checkboxes:
            self._tier_group_layout.removeWidget(cb)
            cb.deleteLater()
        self._tier_checkboxes = []

        if not tier_names:
            self._tier_group.setVisible(False)
            return

        for i, name in enumerate(tier_names):
            cb = QCheckBox(name)
            cb.setChecked(i not in hidden_set)
            self._tier_checkboxes.append(cb)
            self._tier_group_layout.addWidget(cb)
        self._tier_group.setVisible(True)

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
    QLineEdit:disabled, QComboBox:disabled {
        background-color: #3a3a4a; color: #666666;
    }
    QComboBox QAbstractItemView {
        background-color: #ffffff; color: #000000;
        selection-background-color: #d0d0d0; selection-color: #000000;
    }
    QLabel:disabled {
        color: #555555;
    }
    QCheckBox, QRadioButton {
        color: #cccccc;
        spacing: 6px;
    }
    QCheckBox::indicator, QRadioButton::indicator {
        width: 16px; height: 16px;
        border: 2px solid #8899aa;
        background-color: #2a2a3a;
    }
    QCheckBox::indicator { border-radius: 3px; }
    QRadioButton::indicator { border-radius: 9px; }
    QCheckBox::indicator:checked, QRadioButton::indicator:checked {
        background-color: #6699cc;
        border-color: #88bbee;
    }
    QListWidget {
        background-color: #252535; color: #cccccc;
        border: 1px solid #556677;
    }
    QListWidget::item { padding: 4px; }
    QListWidget::item:selected {
        background-color: #3a4a5a;
    }
    QListWidget::indicator {
        width: 16px; height: 16px;
        border: 2px solid #8899aa;
        border-radius: 3px;
        background-color: #2a2a3a;
    }
    QListWidget::indicator:checked {
        background-color: #6699cc;
        border-color: #88bbee;
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


# ---------------------------------------------------------------------------
# Build CSV Wizard
# ---------------------------------------------------------------------------


class _PathsPage(QWizardPage):
    """Page 1: Select audio, TextGrid, and formants directories."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Select Directories")
        self.setSubTitle(
            "Choose the directories containing your audio files, "
            "TextGrids, and (optionally) formant files."
        )

        layout = QFormLayout(self)
        layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        # Audio directory
        audio_row = QHBoxLayout()
        self._audio_edit = QLineEdit()
        self._audio_edit.setPlaceholderText("Path to audio files...")
        self._audio_edit.setReadOnly(True)
        audio_btn = QPushButton("Browse...")
        audio_btn.clicked.connect(self._browse_audio)
        audio_row.addWidget(self._audio_edit, 1)
        audio_row.addWidget(audio_btn)
        layout.addRow("Audio directory:", audio_row)

        # TextGrid separate?
        self._tg_separate_cb = QCheckBox("TextGrids in a different directory")
        layout.addRow("", self._tg_separate_cb)

        tg_row = QHBoxLayout()
        self._tg_edit = QLineEdit()
        self._tg_edit.setPlaceholderText("Path to TextGrid files...")
        self._tg_edit.setReadOnly(True)
        self._tg_btn = QPushButton("Browse...")
        self._tg_btn.clicked.connect(self._browse_tg)
        tg_row.addWidget(self._tg_edit, 1)
        tg_row.addWidget(self._tg_btn)
        self._tg_edit.setVisible(False)
        self._tg_btn.setVisible(False)
        layout.addRow("TextGrid directory:", tg_row)
        self._tg_label = layout.labelForField(tg_row)
        self._tg_label.setVisible(False)

        self._tg_separate_cb.toggled.connect(self._toggle_tg)

        # Formants separate?
        self._fmt_separate_cb = QCheckBox("Formants in a different directory")
        layout.addRow("", self._fmt_separate_cb)

        fmt_row = QHBoxLayout()
        self._fmt_edit = QLineEdit()
        self._fmt_edit.setPlaceholderText("Path to .formants files...")
        self._fmt_edit.setReadOnly(True)
        self._fmt_btn = QPushButton("Browse...")
        self._fmt_btn.clicked.connect(self._browse_fmt)
        fmt_row.addWidget(self._fmt_edit, 1)
        fmt_row.addWidget(self._fmt_btn)
        self._fmt_edit.setVisible(False)
        self._fmt_btn.setVisible(False)
        layout.addRow("Formants directory:", fmt_row)
        self._fmt_label = layout.labelForField(fmt_row)
        self._fmt_label.setVisible(False)

        self._fmt_separate_cb.toggled.connect(self._toggle_fmt)

        self.setStyleSheet(_DIALOG_FIELD_STYLE)

    # --- Browsing ---
    def _browse_audio(self):
        d = QFileDialog.getExistingDirectory(self, "Select Audio Directory")
        if d:
            self._audio_edit.setText(d)
            self.completeChanged.emit()

    def _browse_tg(self):
        d = QFileDialog.getExistingDirectory(self, "Select TextGrid Directory")
        if d:
            self._tg_edit.setText(d)

    def _browse_fmt(self):
        d = QFileDialog.getExistingDirectory(self, "Select Formants Directory")
        if d:
            self._fmt_edit.setText(d)

    def _toggle_tg(self, checked):
        self._tg_edit.setVisible(checked)
        self._tg_btn.setVisible(checked)
        self._tg_label.setVisible(checked)

    def _toggle_fmt(self, checked):
        self._fmt_edit.setVisible(checked)
        self._fmt_btn.setVisible(checked)
        self._fmt_label.setVisible(checked)

    def isComplete(self):
        return bool(self._audio_edit.text().strip())

    def validatePage(self):
        audio_dir = self._audio_edit.text().strip()
        if not os.path.isdir(audio_dir):
            QMessageBox.warning(self, "Error", "Audio directory does not exist.")
            return False

        tg_dir = (self._tg_edit.text().strip()
                  if self._tg_separate_cb.isChecked() else audio_dir)
        if not os.path.isdir(tg_dir):
            QMessageBox.warning(self, "Error", "TextGrid directory does not exist.")
            return False

        # Check for at least one TextGrid
        has_tg = any(f.lower().endswith(".textgrid")
                     for f in os.listdir(tg_dir))
        if not has_tg:
            QMessageBox.warning(
                self, "Error",
                f"No .TextGrid files found in:\n{tg_dir}")
            return False

        fmt_dir = (self._fmt_edit.text().strip()
                   if self._fmt_separate_cb.isChecked() else audio_dir)

        # Store on wizard
        wiz = self.wizard()
        wiz.audio_dir = audio_dir
        wiz.textgrid_dir = tg_dir
        wiz.formants_dir = fmt_dir
        return True


class _TierSelectionPage(QWizardPage):
    """Page 2: Select and order TextGrid tiers."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Select & Order Tiers")
        self.setSubTitle(
            "Tiers are auto-sorted by hierarchy (widest segments first). "
            "Use Up/Down to adjust. Uncheck tiers you don't need."
        )
        layout = QVBoxLayout(self)

        self._list = QListWidget()
        layout.addWidget(self._list)

        btn_row = QHBoxLayout()
        up_btn = QPushButton("Move Up")
        down_btn = QPushButton("Move Down")
        up_btn.clicked.connect(self._move_up)
        down_btn.clicked.connect(self._move_down)
        btn_row.addWidget(up_btn)
        btn_row.addWidget(down_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._example_label = QLabel()
        self._example_label.setWordWrap(True)
        layout.addWidget(self._example_label)

    def initializePage(self):
        self._list.clear()
        wiz = self.wizard()
        tg_dir = wiz.textgrid_dir

        # Find first TextGrid
        tg_file = None
        for f in sorted(os.listdir(tg_dir)):
            if f.lower().endswith(".textgrid"):
                tg_file = os.path.join(tg_dir, f)
                break

        if tg_file is None:
            return

        try:
            tg = TextGrid.from_file(tg_file)
        except Exception as e:
            self._example_label.setText(f"Error reading TextGrid: {e}")
            return

        wiz.example_textgrid = tg
        self._example_label.setText(
            f"Example TextGrid: {os.path.basename(tg_file)} "
            f"({len(tg.tiers)} tiers)")

        # Auto-detect hierarchy
        ordered = _detect_tier_hierarchy(tg.tiers)

        for tier in ordered:
            kind = "Interval" if tier.tier_class == "IntervalTier" else "Point"
            n = (len([iv for iv in tier.intervals if iv.text.strip()])
                 if tier.tier_class == "IntervalTier"
                 else len(tier.points))
            label = f"{tier.name}  [{kind}, {n} items]"
            item = QListWidgetItem(label)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            item.setData(Qt.ItemDataRole.UserRole, tier.name)
            item.setData(Qt.ItemDataRole.UserRole + 1, tier.tier_class)
            self._list.addItem(item)

    def _move_up(self):
        row = self._list.currentRow()
        if row <= 0:
            return
        item = self._list.takeItem(row)
        self._list.insertItem(row - 1, item)
        self._list.setCurrentRow(row - 1)

    def _move_down(self):
        row = self._list.currentRow()
        if row < 0 or row >= self._list.count() - 1:
            return
        item = self._list.takeItem(row)
        self._list.insertItem(row + 1, item)
        self._list.setCurrentRow(row + 1)

    def validatePage(self):
        selected = self._get_selected_tiers()
        if not selected:
            QMessageBox.warning(self, "Error", "Select at least one tier.")
            return False
        wiz = self.wizard()
        wiz.selected_tier_info = selected  # list of (name, tier_class)

        # Compute point-tier-to-parent-interval-tier mapping based on order.
        # Each point tier belongs to the nearest interval tier above it.
        wiz.point_tier_parents = {}  # point_tier_name -> interval_tier_name
        last_interval = None
        for name, tc in selected:
            if tc == "IntervalTier":
                last_interval = name
            elif tc == "TextTier" and last_interval is not None:
                wiz.point_tier_parents[name] = last_interval

        return True

    def _get_selected_tiers(self):
        result = []
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                name = item.data(Qt.ItemDataRole.UserRole)
                tc = item.data(Qt.ItemDataRole.UserRole + 1)
                result.append((name, tc))
        return result


class _DataOptionsPage(QWizardPage):
    """Page 3: Choose data extraction options."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Data Extraction Options")
        self.setSubTitle("Choose what data to extract for each segment.")
        layout = QVBoxLayout(self)

        # --- Formants ---
        self._fmt_cb = QCheckBox("Extract formant values (F1–F3)")
        layout.addWidget(self._fmt_cb)

        self._fmt_group = QGroupBox()
        fmt_layout = QVBoxLayout(self._fmt_group)

        # At points (checkbox, not radio)
        self._at_points_cb = QCheckBox("At points (from a point tier)")
        fmt_layout.addWidget(self._at_points_cb)

        pt_row = QHBoxLayout()
        self._pt_label = QLabel("Point tier:")
        pt_row.addWidget(self._pt_label)
        self._point_tier_combo = QComboBox()
        pt_row.addWidget(self._point_tier_combo, 1)
        fmt_layout.addLayout(pt_row)

        self._pt_time_cb = QCheckBox("Include point times in output")
        fmt_layout.addWidget(self._pt_time_cb)

        # For segments (checkbox, not radio)
        self._for_segments_cb = QCheckBox("For segments (at percentage markers)")
        self._for_segments_cb.setChecked(True)
        fmt_layout.addWidget(self._for_segments_cb)

        seg_row = QHBoxLayout()
        self._seg_label = QLabel("Segment tier:")
        seg_row.addWidget(self._seg_label)
        self._seg_tier_combo = QComboBox()
        seg_row.addWidget(self._seg_tier_combo, 1)
        fmt_layout.addLayout(seg_row)

        pct_row = QHBoxLayout()
        self._pct_label = QLabel("Percentage markers:")
        pct_row.addWidget(self._pct_label)
        self._pct_edit = QLineEdit("0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100")
        pct_row.addWidget(self._pct_edit, 1)
        fmt_layout.addLayout(pct_row)

        layout.addWidget(self._fmt_group)
        self._fmt_group.setVisible(False)
        self._fmt_cb.toggled.connect(self._fmt_group.setVisible)
        self._at_points_cb.toggled.connect(self._update_fmt_ui)
        self._for_segments_cb.toggled.connect(self._update_fmt_ui)

        # --- Durations ---
        self._dur_cb = QCheckBox("Extract durations")
        layout.addWidget(self._dur_cb)

        self._dur_group = QGroupBox("Duration tiers")
        self._dur_layout = QVBoxLayout(self._dur_group)
        self._dur_checks = []  # list of (QCheckBox, tier_name)
        layout.addWidget(self._dur_group)
        self._dur_group.setVisible(False)
        self._dur_cb.toggled.connect(self._dur_group.setVisible)

        layout.addStretch()
        self.setStyleSheet(_DIALOG_FIELD_STYLE)

    def initializePage(self):
        wiz = self.wizard()
        selected = wiz.selected_tier_info  # list of (name, tier_class)

        # Populate combos
        self._point_tier_combo.clear()
        self._seg_tier_combo.clear()
        # Clear duration checkboxes
        for cb, _ in self._dur_checks:
            self._dur_layout.removeWidget(cb)
            cb.deleteLater()
        self._dur_checks = []

        for name, tc in selected:
            if tc == "TextTier":
                self._point_tier_combo.addItem(name)
            else:
                self._seg_tier_combo.addItem(name)
                cb = QCheckBox(name)
                cb.setChecked(True)
                self._dur_layout.addWidget(cb)
                self._dur_checks.append((cb, name))

        self._update_fmt_ui()

    def _update_fmt_ui(self):
        at_pts = self._at_points_cb.isChecked()
        for_seg = self._for_segments_cb.isChecked()

        # Enable/disable + visual feedback for dark theme
        for widget in (self._point_tier_combo, self._pt_label,
                       self._pt_time_cb):
            widget.setEnabled(at_pts)
            widget.setStyleSheet("" if at_pts else "color: #555555;")
        for widget in (self._seg_tier_combo, self._seg_label,
                       self._pct_edit, self._pct_label):
            widget.setEnabled(for_seg)
            widget.setStyleSheet("" if for_seg else "color: #555555;")

    def validatePage(self):
        wiz = self.wizard()

        wiz.extract_formants = self._fmt_cb.isChecked()
        wiz.extract_durations = self._dur_cb.isChecked()

        if not wiz.extract_formants and not wiz.extract_durations:
            QMessageBox.warning(
                self, "Error",
                "Select at least one extraction type "
                "(formant values or durations).")
            return False

        if wiz.extract_formants:
            at_pts = self._at_points_cb.isChecked()
            for_seg = self._for_segments_cb.isChecked()

            if not at_pts and not for_seg:
                QMessageBox.warning(
                    self, "Error",
                    "Select at least one formant extraction mode "
                    "(at points and/or for segments).")
                return False

            # Build formant_mode
            if at_pts and for_seg:
                wiz.formant_mode = "both"
            elif at_pts:
                wiz.formant_mode = "at_points"
            else:
                wiz.formant_mode = "for_segments"

            # Point tier
            if at_pts:
                wiz.point_tier_name = self._point_tier_combo.currentText()
                if not wiz.point_tier_name:
                    QMessageBox.warning(
                        self, "Error", "No point tier available.")
                    return False
                wiz.include_point_times = self._pt_time_cb.isChecked()
            else:
                wiz.point_tier_name = None
                wiz.include_point_times = False

            # Segment tier + percentages
            if for_seg:
                wiz.segment_tier_name = self._seg_tier_combo.currentText()
                if not wiz.segment_tier_name:
                    QMessageBox.warning(
                        self, "Error", "No segment tier available.")
                    return False
                try:
                    pcts = [float(x.strip())
                            for x in self._pct_edit.text().split(",")
                            if x.strip()]
                    if not pcts:
                        raise ValueError("empty")
                    for p in pcts:
                        if p < 0 or p > 100:
                            raise ValueError(f"{p} out of range")
                    wiz.percentage_markers = pcts
                except ValueError as e:
                    QMessageBox.warning(
                        self, "Error",
                        f"Invalid percentage markers: {e}\n"
                        "Enter comma-separated numbers 0–100.")
                    return False
            else:
                wiz.segment_tier_name = None
                wiz.percentage_markers = []
        else:
            wiz.formant_mode = None
            wiz.point_tier_name = None
            wiz.include_point_times = False
            wiz.segment_tier_name = None
            wiz.percentage_markers = []

        if wiz.extract_durations:
            dur_names = [name for cb, name in self._dur_checks
                         if cb.isChecked()]
            if not dur_names:
                QMessageBox.warning(
                    self, "Error",
                    "Select at least one tier for duration extraction.")
                return False
            wiz.duration_tier_names = dur_names
        else:
            wiz.duration_tier_names = []

        return True


class _DiphthongReviewDialog(QDialog):
    """Review diphthong classifications for confirmation."""

    def __init__(self, candidates, parent=None):
        """*candidates* is a dict {label_str: count_int}."""
        super().__init__(parent)
        self.setWindowTitle("Review Diphthongs")
        self.setMinimumSize(420, 380)
        self.setStyleSheet(_DIALOG_FIELD_STYLE)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "The following labels were classified as diphthongs.\n"
            "Uncheck any that should NOT be treated as diphthongs:"))

        self._list = QListWidget()
        for label in sorted(candidates, key=lambda x: -candidates[x]):
            item = QListWidgetItem(f"{label}  ({candidates[label]}×)")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            item.setData(Qt.ItemDataRole.UserRole, label)
            self._list.addItem(item)
        layout.addWidget(self._list)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def confirmed(self):
        """Return set of labels the user confirmed as diphthongs."""
        result = set()
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                result.add(item.data(Qt.ItemDataRole.UserRole))
        return result


class _CheckablePopup(QWidget):
    """Frameless popup containing checkboxes. Auto-closes on outside click."""

    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.Popup)
        self.setStyleSheet(
            "_CheckablePopup { background: #2a2a3a; border: 1px solid #555; }")
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(6, 6, 6, 6)
        self._layout.setSpacing(0)
        self._checkboxes = []

    def add_item(self, key, display_text):
        """Add a checkable item. Returns the QCheckBox."""
        cb = QCheckBox(display_text)
        cb.setChecked(True)
        cb.setProperty("item_key", key)
        cb.setStyleSheet("""
            QCheckBox {
                color: #cccccc; font-size: 13px; padding: 5px 8px;
            }
            QCheckBox:hover { background-color: #3a3a4a; }
            QCheckBox::indicator { width: 15px; height: 15px; }
            QCheckBox::indicator:checked {
                border: 2px solid #6699cc; border-radius: 3px;
                background-color: #4477aa;
            }
            QCheckBox::indicator:unchecked {
                border: 2px solid #666; border-radius: 3px;
                background-color: transparent;
            }
        """)
        self._layout.addWidget(cb)
        self._checkboxes.append(cb)
        return cb

    def checked_keys(self):
        return [cb.property("item_key") for cb in self._checkboxes
                if cb.isChecked()]

    def popup(self, pos, min_width=200):
        self.setMinimumWidth(min_width)
        self.adjustSize()
        self.move(pos)
        self.show()


class _MultiSelectDropdown(QWidget):
    """A button that opens a checkable dropdown. Shows selected count.

    Uses a custom ``_CheckablePopup`` (``Qt.WindowType.Popup``) which
    auto-closes on any click outside the popup — including clicking the
    button itself, clicking elsewhere in the wizard, or clicking another
    application window.
    """

    def __init__(self, label, items, parent=None):
        """*items* is a list of (key, display_text) tuples."""
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._label = QLabel(label)
        self._label.setMinimumWidth(130)
        layout.addWidget(self._label)

        self._button = QPushButton("All selected  \u25BC")
        self._button.setMinimumWidth(250)
        self._button.setStyleSheet(
            "QPushButton { text-align: left; padding: 4px 8px; }")
        self._button.clicked.connect(self._toggle_popup)
        layout.addWidget(self._button, 1)

        self._popup = _CheckablePopup()
        for key, display in items:
            cb = self._popup.add_item(key, display)
            cb.toggled.connect(self._update_button_text)

        self._update_button_text()

    def _toggle_popup(self):
        if self._popup.isVisible():
            self._popup.hide()
            return
        pos = self._button.mapToGlobal(self._button.rect().bottomLeft())
        self._popup.popup(pos, self._button.width())

    def _update_button_text(self):
        checked = self.selected_keys()
        total = len(self._popup._checkboxes)
        if len(checked) == total:
            self._button.setText(f"All selected ({total})  \u25BC")
        elif len(checked) == 0:
            self._button.setText("None selected  \u25BC")
        else:
            self._button.setText(f"{len(checked)} of {total} selected  \u25BC")

    def selected_keys(self):
        return self._popup.checked_keys()


class _IPAChartDialog(QDialog):
    """Floating IPA / SAMPA symbol reference panel.

    Displays the standard IPA vowel trapezoid and consonant table with
    clickable symbol buttons.  Clicking a symbol copies it to the clipboard
    and inserts it into the active LabelEdit field (if one is focused).
    A notation dropdown switches all symbols between IPA and SAMPA.
    """

    # Ordered lists matching the IPA chart layout
    _VOWEL_HEIGHTS = ["high", "mid-high", "mid", "mid-low", "low"]
    _VOWEL_FRONTINGS = ["front", "centre", "back"]

    _CONSONANT_MANNERS = [
        "plosive", "nasal", "trill", "tap", "fricative",
        "lateral-fricative", "approximant", "lateral-approximant",
        "affricate",
    ]
    _CONSONANT_PLACES = [
        "bilabial", "labiodental", "dental", "alveolar",
        "postalveolar", "retroflex", "alveolo-palatal", "palatal",
        "velar", "uvular", "pharyngeal", "glottal",
    ]
    _PLACE_ABBREV = {
        "bilabial": "Bilab", "labiodental": "Labdn", "dental": "Dent",
        "alveolar": "Alv", "postalveolar": "PAlv", "retroflex": "Retr",
        "alveolo-palatal": "AlvPl", "palatal": "Pal", "velar": "Vel",
        "uvular": "Uvu", "pharyngeal": "Phar", "glottal": "Glot",
    }
    _MANNER_LABELS = {
        "plosive": "Plosive", "nasal": "Nasal", "trill": "Trill",
        "tap": "Tap/Flap", "fricative": "Fricative",
        "lateral-fricative": "Lat. Fric.", "approximant": "Approximant",
        "lateral-approximant": "Lat. Approx.", "affricate": "Affricate",
    }

    _BTN_STYLE = """
        QPushButton {
            background: #2a2a3a; color: #ddd; border: 1px solid #444;
            border-radius: 3px; padding: 2px 4px; font-size: 14px;
            min-width: 30px; min-height: 22px;
        }
        QPushButton:hover { background: #3a3a5a; border-color: #6699cc; }
        QPushButton:pressed { background: #4a4a6a; }
    """
    _HDR_STYLE = "color: #999; font-weight: bold; font-size: 12px;"

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self._main_window = main_window
        self.setWindowTitle("IPA Symbol Chart")
        self.setWindowFlags(
            Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint)
        self.setMinimumSize(700, 480)
        self.setStyleSheet(
            "QDialog { background: #1e1e2e; }"
            "QLabel { color: #cccccc; }"
            "QTabWidget::pane { border: 1px solid #444; background: #1e1e2e; }"
            "QTabBar::tab { background: #2a2a3a; color: #ccc; padding: 6px 14px;"
            "  border: 1px solid #444; border-bottom: none; border-radius: 4px 4px 0 0; }"
            "QTabBar::tab:selected { background: #1e1e2e; color: #fff; }"
            "QComboBox { background: #2a2a3a; color: #ddd; border: 1px solid #555;"
            "  padding: 3px 8px; min-width: 80px; }"
            "QComboBox QAbstractItemView { background: #2a2a3a; color: #ddd; }"
        )

        self._symbol_buttons = []  # list of QPushButton (for notation switch)

        # Load chart data
        self._chart = _load_ipa_chart(_IPA_CHART_PATH)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Top bar — notation dropdown
        top = QHBoxLayout()
        top.addWidget(QLabel("Notation:"))
        self._notation_combo = QComboBox()
        self._notation_combo.addItems(["IPA", "SAMPA"])
        self._notation_combo.currentIndexChanged.connect(self._switch_notation)
        top.addWidget(self._notation_combo)
        top.addStretch()
        hint = QLabel("Click a symbol to copy / insert")
        hint.setStyleSheet("color: #777; font-style: italic;")
        top.addWidget(hint)
        layout.addLayout(top)

        # Tab widget
        tabs = QTabWidget()
        tabs.addTab(self._build_vowel_tab(), "Vowels")
        tabs.addTab(self._build_consonant_tab(), "Consonants")
        layout.addWidget(tabs)

    # Diacritics: (display_suffix_ipa, display_suffix_sampa, description)
    # Shown on a dotted circle (◌) base character.
    _DIACRITICS = [
        ("\u02B0",  "_h",  "aspirated"),
        ("\u0325",  "_0",  "voiceless"),
        ("\u032A",  "_t",  "dental"),
        ("\u0334",  "_k",  "velarized"),
        ("\u02B7",  "_w",  "labialized"),
        ("\u02B2",  "_j",  "palatalized"),
        ("\u0303",  "_~",  "nasalized"),
        ("\u0324",  "_d",  "breathy voice"),
        ("\u0330",  "_c",  "creaky voice"),
        ("\u02D0",  ":",   "long"),
        ("\u0329",  "=",   "syllabic"),
        ("\u032F",  "_^",  "non-syllabic"),
        ("\u031A",  "_}",  "no audible release"),
        ("\u0339",  "_O",  "more rounded"),
        ("\u031C",  "_c",  "less rounded"),
        ("\u031D",  "_r",  "raised"),
        ("\u031E",  "_o",  "lowered"),
        ("\u0318",  "_A",  "advanced tongue root"),
        ("\u0319",  "_q",  "retracted tongue root"),
    ]

    # ------------------------------------------------------------------
    # Vowel tab
    # ------------------------------------------------------------------
    def _build_vowel_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: #1e1e2e; }")
        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(8, 8, 8, 8)

        # --- Monophthong grid ---
        mono_label = QLabel("Monophthongs (click two vowels for a diphthong)")
        mono_label.setStyleSheet("font-weight: bold; font-size: 13px; color: #aaa;")
        outer.addWidget(mono_label)

        grid = QGridLayout()
        grid.setSpacing(2)
        grid.setContentsMargins(0, 0, 0, 0)

        # Two-level column headers:  fronting on top, unrounded / rounded below
        #  Col layout:  0=row-label, then for each fronting: unrounded, rounded
        col_base = 1
        for fi, fronting in enumerate(self._VOWEL_FRONTINGS):
            c = col_base + fi * 2
            lbl = QLabel(fronting.title())
            lbl.setStyleSheet(self._HDR_STYLE)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(lbl, 0, c, 1, 2)  # span 2 cols
            for si, sub in enumerate(["unrnd", "rnd"]):
                sl = QLabel(sub)
                sl.setStyleSheet("color: #777; font-size: 10px;")
                sl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                grid.addWidget(sl, 1, c + si)

        # Build lookup: (height, fronting, rounding) -> (ipa, sampa)
        vowel_lookup = {}
        for ipa_sym, props in self._chart["ipa"].items():
            if props["type"] != "vowel":
                continue
            if props["subtype"] != "monophthong" or props["length"] != "short":
                continue
            sampa_sym = ""
            for s, p in self._chart["sampa"].items():
                if p is props:
                    sampa_sym = s
                    break
            key = (props["height"], props["fronting"], props["rounding"])
            vowel_lookup[key] = (ipa_sym, sampa_sym)

        # Rows (start at row 2 — after fronting + rounding headers)
        for row, height in enumerate(self._VOWEL_HEIGHTS):
            lbl = QLabel(height.title())
            lbl.setStyleSheet(self._HDR_STYLE)
            lbl.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            grid.addWidget(lbl, row + 2, 0)

            for fi, fronting in enumerate(self._VOWEL_FRONTINGS):
                c = col_base + fi * 2
                for ri, rounding in enumerate(["unrounded", "rounded"]):
                    entry = vowel_lookup.get((height, fronting, rounding))
                    if entry:
                        btn = self._make_btn(entry[0], entry[1])
                        grid.addWidget(btn, row + 2, c + ri,
                                       alignment=Qt.AlignmentFlag.AlignCenter)

        outer.addLayout(grid)

        # --- Diacritics ---
        outer.addSpacing(14)
        dia_label = QLabel("Diacritics (combining marks)")
        dia_label.setStyleSheet(
            "font-weight: bold; font-size: 13px; color: #aaa;")
        outer.addWidget(dia_label)
        dia_grid = QGridLayout()
        dia_grid.setSpacing(3)
        cols = 4  # diacritics per row
        for i, (ipa_suf, sampa_suf, desc) in enumerate(self._DIACRITICS):
            r, c = divmod(i, cols)
            # Show on dotted circle base (◌)
            ipa_display = "\u25CC" + ipa_suf
            sampa_display = sampa_suf
            btn = self._make_btn(ipa_suf, sampa_suf)
            # Override display to show on dotted circle
            btn.setText(ipa_display)
            btn.setProperty("ipa_display", ipa_display)
            btn.setProperty("sampa_display", sampa_suf)
            btn.setToolTip(f"{desc}\nIPA: {ipa_display}   SAMPA: {sampa_suf}")
            btn.setMinimumWidth(40)
            cell = QHBoxLayout()
            cell.setSpacing(4)
            cell.setContentsMargins(0, 0, 0, 0)
            cell.addWidget(btn)
            lbl = QLabel(desc)
            lbl.setStyleSheet("color: #888; font-size: 11px;")
            cell.addWidget(lbl)
            cell.addStretch()
            w = QWidget()
            w.setLayout(cell)
            dia_grid.addWidget(w, r, c)
        outer.addLayout(dia_grid)

        outer.addStretch()
        scroll.setWidget(container)
        return scroll

    # ------------------------------------------------------------------
    # Consonant tab
    # ------------------------------------------------------------------
    def _build_consonant_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: #1e1e2e; }")
        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(8, 8, 8, 8)

        pul_label = QLabel("Pulmonic Consonants")
        pul_label.setStyleSheet("font-weight: bold; font-size: 13px; color: #aaa;")
        outer.addWidget(pul_label)

        grid = QGridLayout()
        grid.setSpacing(3)

        # Build lookup: (manner, place, voicing) -> (ipa, sampa)
        cons_lookup = {}
        non_pulmonic = []  # (ipa, sampa, props)
        for ipa_sym, props in self._chart["ipa"].items():
            if props["type"] != "consonant":
                continue
            sampa_sym = ""
            for s, p in self._chart["sampa"].items():
                if p is props:
                    sampa_sym = s
                    break
            manner = props["manner"]
            place = props["place"]
            if manner in ("click", "implosive", "ejective"):
                non_pulmonic.append((ipa_sym, sampa_sym, props))
                continue
            if place in ("labio-velar", "labio-palatal"):
                non_pulmonic.append((ipa_sym, sampa_sym, props))
                continue
            key = (manner, place, props["voicing"])
            cons_lookup[key] = (ipa_sym, sampa_sym)

        # Column headers
        for col, place in enumerate(self._CONSONANT_PLACES):
            lbl = QLabel(self._PLACE_ABBREV.get(place, place[:4].title()))
            lbl.setStyleSheet(self._HDR_STYLE + " font-size: 10px;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(lbl, 0, col + 1)

        # Rows
        for row, manner in enumerate(self._CONSONANT_MANNERS):
            lbl = QLabel(self._MANNER_LABELS.get(manner, manner.title()))
            lbl.setStyleSheet(self._HDR_STYLE)
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            grid.addWidget(lbl, row + 1, 0)

            for col, place in enumerate(self._CONSONANT_PLACES):
                vl = cons_lookup.get((manner, place, "voiceless"))
                vd = cons_lookup.get((manner, place, "voiced"))
                cell = QHBoxLayout()
                cell.setSpacing(1)
                cell.setContentsMargins(0, 0, 0, 0)
                if vl:
                    cell.addWidget(self._make_btn(vl[0], vl[1]))
                if vl and vd:
                    dot = QLabel("\u00B7")
                    dot.setStyleSheet("color: #555; font-size: 10px;")
                    dot.setFixedWidth(6)
                    cell.addWidget(dot)
                if vd:
                    cell.addWidget(self._make_btn(vd[0], vd[1]))
                if not vl and not vd:
                    cell.addStretch()
                w = QWidget()
                w.setLayout(cell)
                grid.addWidget(w, row + 1, col + 1)

        outer.addLayout(grid)

        # --- Non-pulmonic ---
        if non_pulmonic:
            outer.addSpacing(12)
            np_label = QLabel("Other (clicks, implosives, ejectives, labio-velars)")
            np_label.setStyleSheet(
                "font-weight: bold; font-size: 13px; color: #aaa;")
            outer.addWidget(np_label)
            flow = QHBoxLayout()
            flow.setSpacing(4)
            non_pulmonic.sort(key=lambda x: (x[2]["manner"], x[0]))
            for ipa_sym, sampa_sym, _ in non_pulmonic:
                flow.addWidget(self._make_btn(ipa_sym, sampa_sym))
            flow.addStretch()
            np_widget = QWidget()
            np_widget.setLayout(flow)
            outer.addWidget(np_widget)

        outer.addStretch()
        scroll.setWidget(container)
        return scroll

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _make_btn(self, ipa, sampa):
        """Create a symbol button storing both notations."""
        btn = QPushButton(ipa)  # default to IPA
        btn.setProperty("ipa", ipa)
        btn.setProperty("sampa", sampa)
        btn.setToolTip(f"IPA: {ipa}   SAMPA: {sampa}")
        btn.setStyleSheet(self._BTN_STYLE)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.clicked.connect(lambda checked, b=btn: self._on_symbol_click(b))
        self._symbol_buttons.append(btn)
        return btn

    def _on_symbol_click(self, btn):
        """Copy symbol to clipboard and insert into active label edit."""
        key = "ipa" if self._notation_combo.currentIndex() == 0 else "sampa"
        symbol = btn.property(key)

        # Copy to clipboard
        QApplication.clipboard().setText(symbol)

        # Insert into label edit if active
        le = self._main_window.label_edit
        if le.isEnabled():
            pos = le.cursorPosition()
            text = le.text()
            le.setText(text[:pos] + symbol + text[pos:])
            le.setCursorPosition(pos + len(symbol))
            le.setFocus()

        self._main_window.status.showMessage(f"Copied: {symbol}", 2000)

    def _switch_notation(self):
        """Update all buttons to show the selected notation."""
        is_ipa = self._notation_combo.currentIndex() == 0
        for btn in self._symbol_buttons:
            # Diacritic buttons have special display properties
            disp = btn.property("ipa_display" if is_ipa else "sampa_display")
            if disp is not None:
                btn.setText(disp)
            else:
                btn.setText(btn.property("ipa" if is_ipa else "sampa"))


class _CategorisationPage(QWizardPage):
    """Page 4: Phonetic categorisation options for CSV export."""

    _VOWEL_ITEMS = [
        ("height",   "Height  —  e.g. high, mid-high, mid-low, low"),
        ("fronting",  "Fronting  —  e.g. front, centre, back"),
        ("rounding",  "Rounding  —  e.g. rounded, unrounded"),
        ("length",    "Length  —  e.g. short, long"),
        ("voicing",   "Voicing  —  e.g. voiced"),
        ("subtype",   "Vowel type  —  e.g. monophthong, diphthong"),
    ]

    _CONSONANT_ITEMS = [
        ("place",   "Place  —  e.g. bilabial, alveolar, velar, glottal"),
        ("manner",  "Manner  —  e.g. plosive, fricative, nasal, trill"),
        ("voicing", "Voicing  —  e.g. voiced, voiceless"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Sound Categorisation")
        self.setSubTitle(
            "Optionally add phonetic classification columns to the CSV.")

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Master enable checkbox
        self._enable_cb = QCheckBox("Add phonetic categorisation columns")
        self._enable_cb.setStyleSheet("font-size: 14px; padding: 4px;")
        layout.addWidget(self._enable_cb)

        # Container widget for all options — hidden until master checkbox is on
        self._options_widget = QWidget()
        opts_layout = QVBoxLayout(self._options_widget)
        opts_layout.setContentsMargins(10, 8, 0, 0)
        opts_layout.setSpacing(10)

        # Notation selector
        notation_row = QHBoxLayout()
        lbl = QLabel("Notation system:")
        lbl.setMinimumWidth(130)
        notation_row.addWidget(lbl)
        self._notation_combo = QComboBox()
        self._notation_combo.addItems(["IPA", "SAMPA", "X-SAMPA"])
        self._notation_combo.setMinimumWidth(150)
        notation_row.addWidget(self._notation_combo)
        notation_row.addStretch()
        opts_layout.addLayout(notation_row)

        # Tier selection
        opts_layout.addWidget(QLabel("Tiers to categorise:"))
        self._tier_container = QVBoxLayout()
        self._tier_container.setContentsMargins(10, 0, 0, 0)
        self._tier_cbs = []
        opts_layout.addLayout(self._tier_container)

        # Vowel columns multi-select dropdown
        self._vowel_dropdown = _MultiSelectDropdown(
            "Vowel columns:", self._VOWEL_ITEMS)
        opts_layout.addWidget(self._vowel_dropdown)

        # Consonant columns multi-select dropdown
        self._consonant_dropdown = _MultiSelectDropdown(
            "Consonant columns:", self._CONSONANT_ITEMS)
        opts_layout.addWidget(self._consonant_dropdown)

        opts_layout.addStretch()
        layout.addWidget(self._options_widget)
        layout.addStretch()

        # Start hidden; show when checkbox is toggled
        self._options_widget.setVisible(False)
        self._enable_cb.toggled.connect(self._options_widget.setVisible)

    def initializePage(self):
        """Populate tier checkboxes from wizard state."""
        # Clear old tier checkboxes
        for cb in self._tier_cbs:
            self._tier_container.removeWidget(cb)
            cb.deleteLater()
        self._tier_cbs = []

        wiz = self.wizard()
        for name, tier_class in wiz.selected_tier_info:
            cb = QCheckBox(f"{name}  ({tier_class})")
            cb.setChecked(tier_class == "IntervalTier")
            cb.setProperty("tier_name", name)
            self._tier_cbs.append(cb)
            self._tier_container.addWidget(cb)

    def validatePage(self):
        wiz = self.wizard()

        if not self._enable_cb.isChecked():
            wiz.categorise = False
            return True

        # Collect selected tiers
        cat_tiers = [cb.property("tier_name")
                     for cb in self._tier_cbs if cb.isChecked()]
        if not cat_tiers:
            QMessageBox.warning(
                self, "Error",
                "Select at least one tier to categorise.")
            return False

        # Collect selected properties
        v_props = self._vowel_dropdown.selected_keys()
        c_props = self._consonant_dropdown.selected_keys()
        if not v_props and not c_props:
            QMessageBox.warning(
                self, "Error",
                "Select at least one vowel or consonant property.")
            return False

        notation_map = {"IPA": "ipa", "SAMPA": "sampa", "X-SAMPA": "xsampa"}
        wiz.categorise = True
        wiz.cat_notation = notation_map[self._notation_combo.currentText()]
        wiz.cat_tier_names = cat_tiers
        wiz.cat_vowel_props = v_props
        wiz.cat_consonant_props = c_props
        return True


class BuildCSVWizard(QWizard):
    """Four-page wizard for batch CSV export."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Build CSV — Batch Export")
        self.setMinimumSize(580, 480)
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)

        # Wizard-level state (populated by pages)
        self.audio_dir = ""
        self.textgrid_dir = ""
        self.formants_dir = ""
        self.example_textgrid = None
        self.selected_tier_info = []  # [(name, tier_class), ...]
        self.point_tier_parents = {}  # point_name -> parent_interval_name
        self.extract_formants = False
        self.formant_mode = None
        self.point_tier_name = None
        self.include_point_times = False
        self.segment_tier_name = None
        self.percentage_markers = []
        self.extract_durations = False
        self.duration_tier_names = []

        # Categorisation state (Page 4)
        self.categorise = False
        self.cat_notation = "ipa"
        self.cat_tier_names = []
        self.cat_vowel_props = []
        self.cat_consonant_props = []

        self.addPage(_PathsPage())
        self.addPage(_TierSelectionPage())
        self.addPage(_DataOptionsPage())
        self.addPage(_CategorisationPage())

        self.setStyleSheet(_DIALOG_FIELD_STYLE)


class MainWindow(QMainWindow):
    """FormantStudio main application window."""

    def __init__(self):
        super().__init__()
        self._app_title = "FormantStudio — Manual Formant Editor"
        self.setWindowTitle(self._app_title)
        self.setMinimumSize(1200, 900)
        self.resize(2100, 1425)

        # Dark theme
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e2e; }
            QMenuBar { background-color: #252535; color: #cccccc; }
            QMenuBar::item:selected { background-color: #445566; }
            QStatusBar { background-color: #252535; color: #dddddd; font-size: 13px; }
        """)

        self._filepath = None
        self._textgrid_path = None
        self._formants_path = None
        self._scrollbar_updating = False
        self._formants_dirty = False
        self._textgrid_dirty = False

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
        self.canvas._on_formant_edited = self._on_formant_edited
        self.canvas._on_textgrid_edited = self._on_textgrid_edited

        # Inline label editor
        self.label_edit = LabelEdit()
        self.label_edit._play_callback = self._play_selection
        self.label_edit.textChanged.connect(self._on_label_text_changed)
        self.label_edit.escape_pressed.connect(self._on_label_escape)
        self.canvas._label_edit = self.label_edit

        self.scrollbar = QScrollBar(Qt.Orientation.Horizontal)
        self.scrollbar.setStyleSheet("""
            QScrollBar:horizontal {
                background: #3a3a4e; height: 18px;
                border: none;
            }
            QScrollBar::handle:horizontal {
                background: #667788; border-radius: 4px;
                min-width: 30px;
            }
            QScrollBar::add-line, QScrollBar::sub-line { width: 0; }
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
                background: #3a3a4e;
            }
        """)

        _arrow_style = """
            QPushButton {
                background: #3a3a4e; color: #aabbcc;
                border: 1px solid #556677; border-radius: 3px;
                font-size: 14px; font-weight: bold;
                padding: 0px;
            }
            QPushButton:hover { background: #4a4a5e; }
            QPushButton:pressed { background: #556677; }
        """
        self._scroll_back_btn = QPushButton("\u25C0")
        self._scroll_back_btn.setFixedSize(22, 18)
        self._scroll_back_btn.setStyleSheet(_arrow_style)
        self._scroll_back_btn.setAutoRepeat(True)
        self._scroll_back_btn.setAutoRepeatInterval(150)
        self._scroll_fwd_btn = QPushButton("\u25B6")
        self._scroll_fwd_btn.setFixedSize(22, 18)
        self._scroll_fwd_btn.setStyleSheet(_arrow_style)
        self._scroll_fwd_btn.setAutoRepeat(True)
        self._scroll_fwd_btn.setAutoRepeatInterval(150)

        canvas_layout.addWidget(self.canvas, 1)
        label_row = QHBoxLayout()
        label_row.addStretch(1)
        label_row.addWidget(self.label_edit)
        label_row.addStretch(1)
        canvas_layout.addLayout(label_row)
        scroll_row = QHBoxLayout()
        scroll_row.setContentsMargins(0, 0, 0, 0)
        scroll_row.setSpacing(0)
        scroll_row.addWidget(self._scroll_back_btn)
        scroll_row.addWidget(self.scrollbar, 1)
        scroll_row.addWidget(self._scroll_fwd_btn)
        canvas_layout.addLayout(scroll_row)

        # Control panel
        self.controls = ControlPanel()
        self.canvas._snap_enabled_cb = self.controls.snap_enabled_cb
        self.canvas._snap_tolerance_spin = self.controls.snap_tolerance_spin

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
        self.status.showMessage("Ready — Open a WAV file to begin (Ctrl+Shift+O)")

    def _setup_menu(self):
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        open_action = QAction("&Open WAV...", self)
        open_action.setShortcut(QKeySequence("Ctrl+Shift+O"))
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

        # View menu — keyboard shortcuts reference
        # Shortcuts are displayed via \t text (not bound via setShortcut)
        # so they don't conflict with keyPressEvent handling.
        view_menu = menubar.addMenu("&View")

        zoom_in_action = QAction("Zoom &In\tCtrl+I", self)
        zoom_in_action.triggered.connect(lambda: self._menu_action(
            lambda: self.canvas.zoom(1.0 / ZOOM_FACTOR)))
        view_menu.addAction(zoom_in_action)

        zoom_out_action = QAction("Zoom &Out\tCtrl+O", self)
        zoom_out_action.triggered.connect(lambda: self._menu_action(
            lambda: self.canvas.zoom(ZOOM_FACTOR)))
        view_menu.addAction(zoom_out_action)

        zoom_sel_action = QAction("Zoom to &Selection\tCtrl+N", self)
        zoom_sel_action.triggered.connect(self._menu_zoom_to_selection)
        view_menu.addAction(zoom_sel_action)

        zoom_all_action = QAction("Zoom &All\tCtrl+A", self)
        zoom_all_action.triggered.connect(lambda: self._menu_action(
            lambda: self.canvas.set_view(0, self.canvas.total_duration)))
        view_menu.addAction(zoom_all_action)

        view_menu.addSeparator()

        edit_mode_action = QAction("Toggle &Edit Mode\tCtrl+E", self)
        edit_mode_action.triggered.connect(self.controls.edit_btn.toggle)
        view_menu.addAction(edit_mode_action)

        toggle_formants_action = QAction("Toggle &Formant Display\tF", self)
        toggle_formants_action.triggered.connect(
            lambda: self.controls.show_formants_cb.setChecked(
                not self.controls.show_formants_cb.isChecked()))
        view_menu.addAction(toggle_formants_action)

        view_menu.addSeparator()

        undo_ref = QAction("Undo\tCtrl+Z", self)
        undo_ref.setEnabled(False)
        view_menu.addAction(undo_ref)

        redo_ref = QAction("Redo\tCtrl+Y", self)
        redo_ref.setEnabled(False)
        view_menu.addAction(redo_ref)

        view_menu.addSeparator()

        for text, shortcut in [
            ("Play Selection", "Tab"),
            ("Add Boundary", "Enter"),
            ("Delete Boundary", "Del"),
            ("Stop / Clear Selection", "Esc"),
        ]:
            a = QAction(f"{text}\t{shortcut}", self)
            a.setEnabled(False)
            view_menu.addAction(a)

        view_menu.addSeparator()

        ref_fkeys = QAction("Select F1\u2013F5 (edit mode)\tF1\u2013F5", self)
        ref_fkeys.setEnabled(False)
        view_menu.addAction(ref_fkeys)

        ref_shift = QAction("Drag aligned boundaries\tShift+Drag", self)
        ref_shift.setEnabled(False)
        view_menu.addAction(ref_shift)

        # Tools menu
        tools_menu = menubar.addMenu("&Tools")
        build_csv_action = QAction("&Build CSV...", self)
        build_csv_action.triggered.connect(self._build_csv)
        tools_menu.addAction(build_csv_action)
        symbol_chart_action = QAction("&IPA Symbol Chart", self)
        symbol_chart_action.triggered.connect(self._show_symbol_chart)
        tools_menu.addAction(symbol_chart_action)

    def _menu_action(self, fn):
        """Run fn then re-render + update scrollbar (for View menu actions)."""
        if self.canvas.sound is not None:
            fn()
            self.canvas.render()
            self._update_scrollbar()

    def _menu_zoom_to_selection(self):
        """Zoom to selection triggered from View menu."""
        c = self.canvas
        if c.sound is not None and c._selection_start is not None and c._selection_end is not None:
            sel_width = c._selection_end - c._selection_start
            pad = sel_width * 0.05
            c.set_view(c._selection_start - pad, c._selection_end + pad)
            c.render()
            self._update_scrollbar()

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
        ctrl.undo_btn.clicked.connect(self._undo_last_edit)
        ctrl.reset_current_btn.clicked.connect(self._reset_current_formant)
        ctrl.reset_all_btn.clicked.connect(self._reset_all_formants)
        # Scrollbar + arrow buttons
        self.scrollbar.valueChanged.connect(self._on_scrollbar_changed)
        self._scroll_back_btn.clicked.connect(self._scroll_backward)
        self._scroll_fwd_btn.clicked.connect(self._scroll_forward)

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
        self._textgrid_dirty = True
        self.canvas._label_render_timer.start()  # debounced render

    def _on_formant_edited(self):
        """Callback from canvas when formant edit stroke completes."""
        self._formants_dirty = True

    def _on_textgrid_edited(self):
        """Callback from canvas when textgrid data changes."""
        self._textgrid_dirty = True

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
                # Auto-expand display range if needed
                if f_idx >= n_display:
                    new_n = f_idx + 1
                    combo_idx = new_n - 1  # combo entries: 0→F1, 1→F1-F2, ...
                    self.controls.num_formants_combo.setCurrentIndex(combo_idx)
                    # Signal fires _on_num_formants_display_changed automatically
                self.canvas.active_formant = f_idx
                self.controls.update_active_formant_display(f_idx)
                self.canvas.render()
                self.status.showMessage(
                    f"Now editing {FORMANT_LABELS[f_idx + 1]}"
                )
                event.accept()
                return

        # Tab is handled by _TabPlayFilter (application event filter)

        # --- Keyboard zoom shortcuts (Ctrl+I/O/N/A) ---
        mods = event.modifiers()
        ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)

        if ctrl and self.canvas.sound is not None:
            if key == Qt.Key.Key_I:
                # Zoom in
                self.canvas.zoom(1.0 / ZOOM_FACTOR)
                self.canvas.render()
                self._update_scrollbar()
                event.accept()
                return
            if key == Qt.Key.Key_O:
                # Zoom out
                self.canvas.zoom(ZOOM_FACTOR)
                self.canvas.render()
                self._update_scrollbar()
                event.accept()
                return
            if key == Qt.Key.Key_N:
                # Zoom to selection
                c = self.canvas
                if c._selection_start is not None and c._selection_end is not None:
                    sel_width = c._selection_end - c._selection_start
                    pad = sel_width * 0.05
                    c.set_view(c._selection_start - pad, c._selection_end + pad)
                    c.render()
                    self._update_scrollbar()
                event.accept()
                return
            if key == Qt.Key.Key_A:
                # Zoom all — but let Select All work in active label editor
                if self.label_edit.hasFocus() and self.label_edit.isEnabled():
                    pass  # let Qt handle Select All
                else:
                    c = self.canvas
                    c.set_view(0, c.total_duration)
                    c.render()
                    self._update_scrollbar()
                    event.accept()
                    return
            if key == Qt.Key.Key_E:
                # Toggle formant edit mode
                self.controls.edit_btn.toggle()
                event.accept()
                return
            if key == Qt.Key.Key_Z:
                # Undo
                if self.canvas.undo():
                    self.status.showMessage("Undo")
                    self._formants_dirty = True
                else:
                    self.status.showMessage("Nothing to undo")
                event.accept()
                return
            if key == Qt.Key.Key_Y:
                # Redo
                if self.canvas.redo():
                    self.status.showMessage("Redo")
                    self._formants_dirty = True
                else:
                    self.status.showMessage("Nothing to redo")
                event.accept()
                return

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
        # If a selection region exists, create boundaries at both ends.
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if not self.label_edit.hasFocus():
                c = self.canvas
                if (c._active_tier is not None
                        and c.textgrid_data is not None):
                    # Selection region → two boundaries
                    if (c._selection_start is not None
                            and c._selection_end is not None
                            and c._selection_start != c._selection_end):
                        added = 0
                        for bt in (c._selection_start, c._selection_end):
                            if c._add_boundary(c._active_tier, bt):
                                added += 1
                        if added:
                            self._textgrid_dirty = True
                            mid = (c._selection_start + c._selection_end) / 2
                            c._select_interval(c._active_tier, mid)
                            c.render()
                            self.status.showMessage(
                                f"{added} boundar{'ies' if added > 1 else 'y'} added"
                            )
                        else:
                            self.status.showMessage("Cannot add boundaries here")
                    else:
                        # Single cursor position
                        boundary_time = c._hover_time or c._click_time
                        if boundary_time is not None:
                            if c._add_boundary(c._active_tier, boundary_time):
                                self._textgrid_dirty = True
                                c._select_interval(c._active_tier, boundary_time)
                                c.render()
                                self.status.showMessage("Boundary added")
                            else:
                                self.status.showMessage("Cannot add boundary here")
                event.accept()
                return

        # Delete — remove selected boundary or point
        if key == Qt.Key.Key_Delete:
            if self.canvas._selected_boundary is not None:
                tier_idx, bt = self.canvas._selected_boundary
                if self.canvas._delete_boundary(tier_idx, bt):
                    self._textgrid_dirty = True
                    self.canvas._selected_boundary = None
                    self.canvas.render()
                    self.status.showMessage("Boundary deleted")
                else:
                    self.status.showMessage("Cannot delete tier start/end boundary")
                event.accept()
                return
            # Delete selected point on a TextTier
            if self.canvas._selected_interval is not None:
                tier_idx, pt_idx = self.canvas._selected_interval
                tier = self.canvas.textgrid_data.tiers[tier_idx]
                if tier.tier_class == "TextTier" and 0 <= pt_idx < len(tier.points):
                    tier.points.pop(pt_idx)
                    self._textgrid_dirty = True
                    self.canvas._selected_interval = None
                    if self.canvas._label_edit is not None:
                        self.canvas._label_edit.setEnabled(False)
                        self.canvas._label_edit.clear()
                    self.canvas.render()
                    self.status.showMessage("Point deleted")
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

    def _check_unsaved_changes(self):
        """Check for unsaved formant/TextGrid edits. Returns True if OK to proceed."""
        unsaved = []
        if self._formants_dirty:
            unsaved.append("formants")
        if self._textgrid_dirty:
            unsaved.append("TextGrid")
        if not unsaved:
            return True
        reply = QMessageBox.question(
            self,
            "Unsaved Changes",
            f"You have unsaved {' and '.join(unsaved)} changes.\n\n"
            "Do you want to save before opening a new file?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
        )
        if reply == QMessageBox.StandardButton.Save:
            if self._formants_dirty:
                self.save_formants()
            if self._textgrid_dirty:
                self.save_textgrid()
            return True
        elif reply == QMessageBox.StandardButton.Discard:
            return True
        else:  # Cancel
            return False

    def open_file(self):
        if not self._check_unsaved_changes():
            return

        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Audio File", "",
            "Audio Files (*.wav *.WAV *.aiff *.AIFF *.mp3);;All Files (*)"
        )
        if not filepath:
            return

        self._filepath = filepath
        self.canvas._filepath = filepath
        self._textgrid_path = None
        self._formants_path = None
        self._formants_dirty = False
        self._textgrid_dirty = False
        self.status.showMessage(f"Loading {os.path.basename(filepath)}...")
        QApplication.processEvents()

        try:
            self.canvas.load_sound(filepath)
            self._run_formant_analysis()
            self._update_scrollbar()
            self.canvas.render()
            self.setWindowTitle(
                f"{self._app_title}  —  {os.path.basename(filepath)}"
            )
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
                    self._formants_path = fmt_path
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
                self._prompt_textgrid_choice()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file:\n{e}")
            self.status.showMessage("Error loading file")

    def save_formants(self):
        if self.canvas.formant_data is None:
            self.status.showMessage("Nothing to save")
            return

        if self._formants_path:
            fmt_path = self._formants_path
        else:
            # Default to audio directory if available, otherwise prompt
            if self._filepath:
                default = os.path.splitext(self._filepath)[0] + ".formants"
            else:
                default = ""
            fmt_path, _ = QFileDialog.getSaveFileName(
                self, "Save Formants", default,
                "Formant Files (*.formants);;All Files (*)"
            )
            if not fmt_path:
                return

        try:
            self.canvas.formant_data.save(fmt_path)
            self._formants_path = fmt_path
            self._formants_dirty = False
            self.status.showMessage(f"Saved: {os.path.basename(fmt_path)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save:\n{e}")

    def load_formants(self):
        start_dir = os.path.dirname(self._formants_path or self._filepath or "")
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Formant Data", start_dir,
            "Formant Files (*.formants);;All Files (*)"
        )
        if not filepath:
            return
        try:
            self.canvas.formant_data = FormantData.load(filepath)
            self._formants_path = filepath
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
            self._textgrid_dirty = False
            self._setup_tier_checkboxes()
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
            # Suggest audio directory with matching basename, but let user choose
            if self._filepath:
                default = os.path.splitext(self._filepath)[0] + ".TextGrid"
            else:
                default = ""
            save_path, _ = QFileDialog.getSaveFileName(
                self, "Save TextGrid", default,
                "TextGrid Files (*.TextGrid);;All Files (*)"
            )
            if not save_path:
                return

        try:
            self.canvas.textgrid_data.save(save_path)
            self._textgrid_path = save_path
            self._textgrid_dirty = False
            self.status.showMessage(f"Saved TextGrid: {os.path.basename(save_path)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save TextGrid:\n{e}")

    def _show_symbol_chart(self):
        """Show (or raise) the floating IPA / SAMPA symbol chart."""
        if not hasattr(self, '_symbol_chart_dialog') or self._symbol_chart_dialog is None:
            self._symbol_chart_dialog = _IPAChartDialog(self)
        self._symbol_chart_dialog.show()
        self._symbol_chart_dialog.raise_()
        self._symbol_chart_dialog.activateWindow()

    def _build_csv(self):
        """Launch the Build CSV wizard and run batch export."""
        wizard = BuildCSVWizard(self)
        if wizard.exec() != QDialog.DialogCode.Accepted:
            return

        # Resolve selected tiers to Tier objects from the example TextGrid
        # (used only for hierarchy info; each file's tiers are looked up fresh)
        tg = wizard.example_textgrid
        selected_tiers = []
        for name, tc in wizard.selected_tier_info:
            for t in tg.tiers:
                if t.name == name and t.tier_class == tc:
                    selected_tiers.append(t)
                    break

        if not selected_tiers:
            QMessageBox.warning(self, "Error", "No valid tiers selected.")
            return

        # Count audio files
        audio_exts = {".wav", ".aiff", ".mp3"}
        audio_files = [
            f for f in os.listdir(wizard.audio_dir)
            if os.path.splitext(f)[1].lower() in audio_exts
        ]
        if not audio_files:
            QMessageBox.warning(
                self, "Error",
                f"No audio files found in:\n{wizard.audio_dir}")
            return

        # Progress dialog
        progress = QProgressDialog(
            "Building CSV...", "Cancel", 0, len(audio_files), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)

        cancelled = False

        def on_progress(current, total, filename):
            nonlocal cancelled
            progress.setValue(current)
            progress.setLabelText(
                f"Processing {current + 1}/{total}: {filename}")
            QApplication.processEvents()
            if progress.wasCanceled():
                cancelled = True

        # Load IPA chart if categorisation is enabled
        cat_chart = None
        if wizard.categorise:
            try:
                cat_chart = _load_ipa_chart(_IPA_CHART_PATH)
            except Exception as e:
                QMessageBox.critical(
                    self, "Error",
                    f"Failed to load IPA chart:\n{e}")
                return

        auto_diphthong_candidates = {} if wizard.categorise else None
        unmatched_labels = set() if wizard.categorise else None

        try:
            headers, rows = _build_csv_data(
                audio_dir=wizard.audio_dir,
                textgrid_dir=wizard.textgrid_dir,
                formants_dir=wizard.formants_dir,
                selected_tiers=selected_tiers,
                extract_formants=wizard.extract_formants,
                formant_mode=wizard.formant_mode,
                point_tier_name=wizard.point_tier_name,
                segment_tier_name=wizard.segment_tier_name,
                percentage_markers=wizard.percentage_markers,
                extract_durations=wizard.extract_durations,
                duration_tier_names=wizard.duration_tier_names,
                point_tier_parents=wizard.point_tier_parents,
                progress_callback=on_progress,
                include_point_times=wizard.include_point_times,
                categorise=wizard.categorise,
                cat_chart=cat_chart,
                cat_notation=wizard.cat_notation,
                cat_tier_names=wizard.cat_tier_names,
                cat_vowel_props=wizard.cat_vowel_props,
                cat_consonant_props=wizard.cat_consonant_props,
                auto_diphthong_candidates=auto_diphthong_candidates,
                unmatched_labels=unmatched_labels,
            )
        except Exception as e:
            progress.close()
            QMessageBox.critical(self, "Error", f"CSV build failed:\n{e}")
            return

        progress.close()

        if cancelled:
            self.status.showMessage("CSV export cancelled")
            return

        # Diphthong review dialog
        if auto_diphthong_candidates:
            dlg = _DiphthongReviewDialog(auto_diphthong_candidates, self)
            if dlg.exec() == QDialog.DialogCode.Accepted:
                confirmed = dlg.confirmed()
                # Rejected diphthong labels → mark their cells as unmatched
                rejected = set(auto_diphthong_candidates) - confirmed
                if rejected:
                    # Find cat column indices for type/subtype
                    # Clear cells for rejected diphthongs
                    n_labels = len([t for t in selected_tiers
                                    if t.tier_class == "IntervalTier"])
                    n_labels += len([t for t in selected_tiers
                                     if t.tier_class == "TextTier"])
                    for row in rows:
                        for tn in wizard.cat_tier_names:
                            # Find the label for this tier
                            label_text = ""
                            all_tnames = ([t.name for t in selected_tiers
                                           if t.tier_class == "IntervalTier"] +
                                          [t.name for t in selected_tiers
                                           if t.tier_class == "TextTier"])
                            if tn in all_tnames:
                                li = 1 + all_tnames.index(tn)
                                if li < len(row):
                                    label_text = row[li]
                            lbl = label_text.split(";")[0].strip() if label_text else ""
                            if lbl in rejected:
                                # Reclassify as vowel-combination
                                for hi, h in enumerate(headers):
                                    if hi >= len(row):
                                        continue
                                    if h == f"{tn}_type":
                                        row[hi] = "vowel"
                                    elif h == f"{tn}_V_subtype":
                                        row[hi] = "vowel-combination"
                                    elif h.startswith(f"{tn}_V_") or h.startswith(f"{tn}_C_"):
                                        # Clear vowel/consonant detail cols
                                        row[hi] = ""
                                    elif h.startswith(f"{tn}_"):
                                        # Keep diacritic modifier cols as-is
                                        pass

        # Filter empty columns (categorisation columns only)
        if wizard.categorise and rows:
            non_empty_cols = set()
            # Always keep non-categorisation columns
            cat_start = None
            for i, h in enumerate(headers):
                for tn in wizard.cat_tier_names:
                    if h.startswith(f"{tn}_"):
                        if cat_start is None:
                            cat_start = i
                        break
                else:
                    non_empty_cols.add(i)
            # Check which cat columns have data
            if cat_start is not None:
                for row in rows:
                    for i in range(cat_start, len(headers)):
                        if i < len(row) and row[i]:
                            non_empty_cols.add(i)
                keep = sorted(non_empty_cols)
                headers = [headers[i] for i in keep]
                rows = [[row[i] if i < len(row) else "" for i in keep]
                        for row in rows]

        # Report unmatched labels
        if unmatched_labels:
            items = sorted(unmatched_labels)
            msg = "The following labels could not be classified:\n\n"
            for tn, lbl in items[:50]:  # cap at 50 to avoid huge dialog
                msg += f"  [{tn}] {lbl}\n"
            if len(items) > 50:
                msg += f"\n  ... and {len(items) - 50} more."
            QMessageBox.information(self, "Unmatched Labels", msg)

        # Save dialog
        default_name = os.path.join(wizard.audio_dir, "formant_data.csv")
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV", default_name,
            "CSV Files (*.csv);;All Files (*)")
        if not save_path:
            return

        try:
            with open(save_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(rows)
            self.status.showMessage(
                f"CSV exported: {len(rows)} rows → "
                f"{os.path.basename(save_path)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to write CSV:\n{e}")

    def _setup_tier_checkboxes(self):
        """Populate tier visibility checkboxes for the current TextGrid."""
        tg = self.canvas.textgrid_data
        if tg is None:
            self.controls.populate_tier_checkboxes([], set())
            return
        names = [t.name for t in tg.tiers]
        self.canvas.hidden_tiers = set()  # reset when loading new TextGrid
        self.controls.populate_tier_checkboxes(names, self.canvas.hidden_tiers)
        # Connect signals
        for i, cb in enumerate(self.controls._tier_checkboxes):
            cb.toggled.connect(lambda checked, idx=i: self._on_tier_visibility_toggled(idx, checked))

    def _on_tier_visibility_toggled(self, tier_idx, checked):
        """Toggle tier visibility and rebuild axes."""
        if checked:
            self.canvas.hidden_tiers.discard(tier_idx)
        else:
            self.canvas.hidden_tiers.add(tier_idx)
        self.canvas._setup_axes()
        self.canvas.render()

    def _prompt_textgrid_choice(self):
        """When no matching TextGrid is found, let the user create or load one."""
        msg = QMessageBox(self)
        msg.setWindowTitle("No TextGrid Found")
        msg.setText(
            "No TextGrid was found for this audio file.\n\n"
            "Would you like to create a new TextGrid or load an existing one?"
        )
        btn_create = msg.addButton("Create New", QMessageBox.ButtonRole.AcceptRole)
        btn_load = msg.addButton("Load Existing…", QMessageBox.ButtonRole.ActionRole)
        msg.addButton("Skip", QMessageBox.ButtonRole.RejectRole)
        msg.exec()

        clicked = msg.clickedButton()
        if clicked is btn_create:
            tg = self._create_textgrid_from_dialog()
            if tg is not None:
                self.canvas.textgrid_data = tg
                self._setup_tier_checkboxes()
                self.canvas._setup_axes()
                self.canvas.render()
                tier_desc = ", ".join(t.name for t in tg.tiers)
                self.status.showMessage(
                    f"Created TextGrid "
                    f"({len(tg.tiers)} tier{'s' if len(tg.tiers) != 1 else ''}: {tier_desc})"
                )
        elif clicked is btn_load:
            self.load_textgrid()

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
            # Adjust active tier and hidden tiers if they shifted
            if (self.canvas._active_tier is not None
                    and pos <= self.canvas._active_tier):
                self.canvas._active_tier += 1
            # Shift hidden tier indices
            new_hidden = set()
            for h in self.canvas.hidden_tiers:
                new_hidden.add(h + 1 if h >= pos else h)
            self.canvas.hidden_tiers = new_hidden
            self._setup_tier_checkboxes()
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
        self.scrollbar.setSingleStep(max(1, int(view_ms * 2 / 3)))
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

    def _scroll_backward(self):
        """Arrow button: scroll backward by 2/3 of visible window."""
        c = self.canvas
        if c.sound is None:
            return
        shift = c.view_width * 2 / 3
        new_start = max(0, c.view_start - shift)
        c.set_view(new_start, new_start + c.view_width)
        c.render()
        self._update_scrollbar()

    def _scroll_forward(self):
        """Arrow button: scroll forward by 2/3 of visible window."""
        c = self.canvas
        if c.sound is None:
            return
        shift = c.view_width * 2 / 3
        new_start = min(c.total_duration - c.view_width, c.view_start + shift)
        c.set_view(new_start, new_start + c.view_width)
        c.render()
        self._update_scrollbar()

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

    def _undo_last_edit(self):
        """Undo last formant edit via button click."""
        if self.canvas.undo():
            self.status.showMessage("Undo")
            self._formants_dirty = True
        else:
            self.status.showMessage("Nothing to undo")

    def _reset_current_formant(self):
        if self.canvas.formant_data is None:
            return
        fd = self.canvas.formant_data
        f_idx = self.canvas.active_formant
        # Capture changes for undo
        changes = []
        for fi in range(fd.n_frames):
            old_val = float(fd.values[f_idx, fi])
            old_mask = bool(fd.edited_mask[f_idx, fi])
            new_val = float(fd.original_values[f_idx, fi])
            if old_val != new_val or old_mask:
                changes.append((f_idx, fi, old_val, old_mask, new_val, False))
        if changes:
            self.canvas._undo_stack.append(
                UndoEntry("Reset formant", changes))
            self.canvas._redo_stack.clear()
            if len(self.canvas._undo_stack) > MAX_UNDO_STEPS:
                self.canvas._undo_stack.pop(0)
        fd.reset_to_original(f_idx)
        self.canvas.render()
        self._formants_dirty = True
        self.status.showMessage(f"Reset {FORMANT_LABELS[f_idx + 1]} to original")

    def _reset_all_formants(self):
        if self.canvas.formant_data is None:
            return
        fd = self.canvas.formant_data
        # Capture changes for undo
        changes = []
        for f_idx in range(fd.n_formants):
            for fi in range(fd.n_frames):
                old_val = float(fd.values[f_idx, fi])
                old_mask = bool(fd.edited_mask[f_idx, fi])
                new_val = float(fd.original_values[f_idx, fi])
                if old_val != new_val or old_mask:
                    changes.append((f_idx, fi, old_val, old_mask, new_val, False))
        if changes:
            self.canvas._undo_stack.append(
                UndoEntry("Reset all formants", changes))
            self.canvas._redo_stack.clear()
            if len(self.canvas._undo_stack) > MAX_UNDO_STEPS:
                self.canvas._undo_stack.pop(0)
        fd.reset_to_original()
        self.canvas.render()
        self._formants_dirty = True
        self.status.showMessage("Reset all formants to original")

    # -------------------------------------------------------------------
    # Close event — unsaved changes prompt
    # -------------------------------------------------------------------

    def closeEvent(self, event):
        unsaved = []
        if self._formants_dirty:
            unsaved.append("formants")
        if self._textgrid_dirty:
            unsaved.append("TextGrid")
        if unsaved:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                f"You have unsaved changes in your {' and '.join(unsaved)}.\n"
                "Do you want to save before closing?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel)
            if reply == QMessageBox.StandardButton.Yes:
                if self._formants_dirty:
                    self.save_formants()
                if self._textgrid_dirty:
                    self.save_textgrid()
                event.accept()
            elif reply == QMessageBox.StandardButton.No:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # Set Windows taskbar icon identity (must be before QApplication)
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "FormantStudio.FormantStudio.1"
        )

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Application icon (title bar + taskbar)
    from PyQt6.QtGui import QIcon
    icon_path = os.path.join(
        getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__))),
        "formant_studio.ico",
    )
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

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
