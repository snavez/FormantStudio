# FormantStudio — Software Specification

**Manual Formant Editor**

| Field | Value |
|-------|-------|
| Version | 0.3.0 (Milestone 3) |
| Date | February 2026 |
| Status | In Development |
| Platform | Windows / macOS / Linux |
| Analysis Engine | Praat (via Parselmouth) |

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Functional Specification](#2-functional-specification)
3. [Implementation Status](#3-implementation-status)
4. [Roadmap](#4-roadmap)
5. [Architecture](#5-architecture)
6. [Testing](#6-testing)
7. [Citation & Attribution](#7-citation--attribution)

---

## 1. Project Overview

### 1.1 Problem Statement

Praat is the established standard for acoustic phonetic analysis, widely used in linguistics research. Its formant tracking algorithm (Burg method) is generally reliable, but produces tracking errors in challenging conditions: non-modal phonation, nasalised vowels, breathy voice, creaky voice, and noisy recordings. While plugins like FastTrack can reduce gross errors by optimising analysis parameters, there is currently no way for a researcher to manually correct the calculated formant values when they are visibly wrong on the spectrogram.

Praat displays formant tracks overlaid on the spectrogram, and the user can clearly see when a tracked formant deviates from the actual spectral energy concentration. However, Praat provides no mechanism to adjust these values, nor to save corrected formant data alongside the TextGrid annotation files that are central to the phonetic analysis workflow.

### 1.2 Solution

FormantStudio is a standalone desktop application that uses Praat's actual analysis engine (via the Parselmouth Python library) to perform formant analysis, then provides a visual editing interface where researchers can manually correct mistracked formants by drawing directly on the spectrogram. Corrected formant data is saved as a companion file (.formants) alongside the existing TextGrid files, preserving the established workflow while adding the missing correction capability.

### 1.3 Design Principles

- **Praat-native analysis:** All acoustic analysis is performed by Praat's own code (compiled into Parselmouth). Results are identical to Praat and can be cited as such in publications.
- **Non-destructive editing:** Original Praat-calculated values are always preserved. Edits are stored separately with an edited_mask, allowing reset and comparison.
- **Workflow compatibility:** Formant files are saved alongside TextGrids in the same directory structure. Time alignment uses the same frame-based system as Praat.
- **Cross-platform:** Runs on Windows, macOS, and Linux from the same codebase (PyQt6 + Parselmouth).

### 1.4 Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Analysis Engine | Parselmouth 0.4.x | Python bindings to Praat's C++ source code |
| GUI Framework | PyQt6 | Cross-platform desktop UI |
| Visualisation | Matplotlib | Spectrogram rendering, formant overlay |
| Data Handling | NumPy | Array operations for formant/spectrogram data |
| File Format | JSON (.formants) | Editable, human-readable, version-trackable |

---

## 2. Functional Specification

### 2.1 Spectrogram Display

The main area of the application displays a wideband spectrogram computed by Praat's spectrogram algorithm. The spectrogram is rendered using Matplotlib's `imshow` with bilinear interpolation and the `gray_r` colourmap (greyscale, dark = high intensity), limited to ≤10 seconds of visible audio for performance. A waveform display sits above the spectrogram showing the audio amplitude.

#### 2.1.1 Display Controls

| Control | Range | Default | Effect |
|---------|-------|---------|--------|
| Dynamic Range | 10–100 dB | 50 dB | Controls the visible dB range from peak. Lower values show only the strongest spectral energy; higher values reveal quieter detail. |
| Brightness | -30 to +30 | 0 | Offsets the peak reference point in dB. Positive values brighten the overall display. |
| Pre-emphasis | 1–200 Hz | 50 Hz | Frequency above which a 6 dB/octave boost is applied before formant analysis. Compensates for the natural spectral tilt of speech. |
| Max Frequency | 1000–12000 Hz | 5500 Hz | Upper frequency limit for both the spectrogram display and the vertical axis. |

These controls are located in the right-hand control panel and update the display in real time.

### 2.2 Formant Analysis

Formant analysis uses Praat's Burg method (Linear Predictive Coding), the same algorithm invoked by Praat's "To Formant (Burg...)" command. The analysis is performed by Parselmouth, which compiles and executes Praat's actual C++ source code.

#### 2.2.1 Analysis Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Number of formants | 3 (F1–F3) | How many formants to calculate. Selectable from 1 to 5 via dropdown. Default changed from Praat's 5 to 3 for typical use. |
| Maximum formant (Hz) | 5500 | Expected ceiling frequency for the highest formant. Standard value for adult female speakers; may need to be lowered to ~5000 for adult males. |
| Window length (s) | 0.025 | Analysis window duration. Praat default. Controls the time resolution of formant frames; shorter windows give finer temporal detail but noisier estimates. |
| Time step | Auto | Distance between analysis frames. When set to auto (0), Praat uses 25% of the window length (default: 6.25 ms). |

#### 2.2.2 Colour Coding

Formant tracks are overlaid on the spectrogram as scatter plots with the following colour assignments:

| Formant | Colour | Hex |
|---------|--------|-----|
| F1 | Blue | #3399FF |
| F2 | Yellow | #ffe600 |
| F3 | Red | #FF4444 |
| F4 | Green | #44CC44 |
| F5 | Magenta | #CC44CC |

Unedited formant points are rendered as small white dots (current default Praat size) with a thin black outline to distinguish them from the background. Manually edited points are rendered in the same size, but with the colouring described above.

### 2.3 Formant Editing

#### 2.3.1 Activating Edit Mode

The user activates edit mode by clicking the "EDIT MODE" button in the control panel. When active, the button turns red and the title bar displays the currently selected formant with its colour. Edit mode can be toggled off by clicking the button again.

#### 2.3.2 Drawing Formants

While edit mode is active, the user holds the left mouse button and drags across the spectrogram. The cursor position is mapped to the nearest analysis frame (by time) and the frequency axis (by vertical position). The formant value at each frame under the cursor is overwritten with the cursor's frequency. Fast mouse movement fills skipped frames with linear interpolation between consecutive edit positions. A visual marker is drawn immediately for responsiveness (using blitting), with a full re-render on mouse release.

#### 2.3.2a Eraser Tool (Right-Click Drag)

While edit mode is active, right-click dragging across the spectrogram reverts formant points to their original Praat-calculated values and clears the edited mask. Grey scatter dots provide visual feedback during erasing. Skipped frames during fast mouse movement are also reverted. Eraser strokes are undoable.

#### 2.3.2b Undo/Redo (Ctrl+Z / Ctrl+Y)

All formant editing operations (draw strokes, erase strokes, resets, interpolation) are undoable. The undo stack holds up to 100 entries. Each entry stores per-frame diffs (old/new value and mask). Ctrl+Z undoes, Ctrl+Y redoes.

#### 2.3.2c Interpolate Between Edited Points

The "Interpolate Between Points" button in the control panel finds consecutive pairs of edited points for the active formant and fills intermediate frames with linear interpolation. If a time selection is active, only points within the selection are considered. The operation is undoable.

#### 2.3.3 Formant Selection

The F1 through F5 keyboard keys select which formant is being edited, but only while edit mode is active. When edit mode is not active, these keys retain their default Praat-like behaviour (if applicable). The currently active formant is displayed in the control panel with its assigned colour. Only formants within the currently configured display range can be selected (e.g., if showing F1–F3, pressing F4 has no effect).

#### 2.3.4 Reset

Two reset options are provided: "Reset Current Formant" reverts only the active formant to its Praat-calculated values; "Reset All Edits" reverts all formants. Reset operations clear the edited_mask for the affected formant(s) and restore the original_values array.

### 2.4 File Format (.formants)

Edited formant data is saved as a JSON file with the .formants extension, placed in the same directory as the source audio file with a matching base filename (e.g., `recording.wav` → `recording.formants`).

#### 2.4.1 Schema

| Field | Type | Description |
|-------|------|-------------|
| format_version | int | Schema version (currently 1) |
| generator | string | Always "FormantStudio" |
| analysis_engine | string | "Praat (via Parselmouth)" |
| n_formants | int | Number of formant tracks stored (always 5 slots) |
| n_frames | int | Number of analysis frames |
| time_step_seconds | float | Time distance between consecutive frames in seconds |
| times | float[] | Centre time of each frame in seconds (length = n_frames) |
| values | float[][] | 2D array [n_formants][n_frames]. Each value is frequency in Hz, or null where no formant was detected. |
| edited_mask | bool[][] | 2D array [n_formants][n_frames]. true where the value has been manually edited. |

#### 2.4.2 Time Alignment

The times array provides the exact centre time of each analysis frame. Combined with time_step_seconds, this allows any downstream tool to precisely align formant values with TextGrid interval or point tier boundaries. When extracting formant values for a labelled segment, the standard approach is to find all frames whose times fall within the segment's start and end times.

#### 2.4.3 Null Handling

Where Praat's algorithm does not detect a formant at a given frame (or where the user has not drawn a value), the corresponding entry in the values array is null (JSON) / NaN (internal NumPy representation). This ensures that all formant tracks have the same number of entries, and that alignment is maintained even when data is sparse.

#### 2.4.4 Save/Load Behaviour

Ctrl+S saves the .formants file. On opening a WAV file, FormantStudio checks for an existing .formants file with a matching name and offers to load it. The loaded data replaces the Praat-calculated formants (the loaded values become the new "original" baseline for reset purposes). The user can also manually load a .formants file via Ctrl+L.

---

## 3. Implementation Status

### 3.1 Milestone 1 — Core Prototype ✅ COMPLETE

| Feature | Status | Notes |
|---------|--------|-------|
| Open WAV files | ✅ COMPLETE | Also supports AIFF, MP3 |
| Spectrogram display (Praat engine) | ✅ COMPLETE | Greyscale (`gray_r`), bilinear interpolation, ≤10s view limit |
| Waveform display | ✅ COMPLETE | Amplitude waveform above spectrogram |
| Dynamic range / brightness sliders | ✅ COMPLETE | Real-time update |
| Pre-emphasis control | ✅ COMPLETE | Affects formant re-analysis |
| Max frequency control | ✅ COMPLETE | Recomputes spectrogram |
| Spectrogram window length slider | ✅ COMPLETE | Adjustable analysis window |
| Formant analysis (Praat Burg method) | ✅ COMPLETE | Configurable parameters |
| Colour-coded formant overlay | ✅ COMPLETE | F1–F5, edited points highlighted |
| Click-drag formant editing | ✅ COMPLETE | Per-frame overwrite with interpolation |
| F1–F5 key selection | ✅ COMPLETE | Active only in edit mode |
| Save .formants file (Ctrl+S) | ✅ COMPLETE | JSON format, alongside WAV |
| Load .formants file (Ctrl+L / auto) | ✅ COMPLETE | Auto-detect on WAV open |
| Reset edits (per-formant / all) | ✅ COMPLETE | Restores original values, undoable |
| Re-analyse with changed parameters | ✅ COMPLETE | Button in control panel |
| Dark theme UI | ✅ COMPLETE | Fusion style + custom palette |

### 3.2 Milestone 2 — TextGrid Integration & Navigation ✅ COMPLETE

| Feature | Status | Notes |
|---------|--------|-------|
| Display TextGrid tiers below spectrogram | ✅ COMPLETE | IntervalTier + TextTier, batch rendering |
| Load/save TextGrid files | ✅ COMPLETE | Normal and short format parsing, normal format save |
| TextGrid boundary editing | ✅ COMPLETE | Add (Enter), delete (Del), drag boundaries |
| Shift+drag aligned boundaries | ✅ COMPLETE | Multi-tier aligned boundary movement |
| Interval/point label editing | ✅ COMPLETE | Inline label editor, live update |
| Create new TextGrid | ✅ COMPLETE | Dialog for creating from scratch |
| Add tier to existing TextGrid | ✅ COMPLETE | IntervalTier or TextTier |
| Zoom in/out (time axis) | ✅ COMPLETE | Mouse scroll wheel + Ctrl+I/O |
| Scroll / pan through audio | ✅ COMPLETE | Scrollbar with ◀/▶ arrow buttons (2/3 window step) |
| Click-to-play audio selection | ✅ COMPLETE | Tab to play, selection/click/view range |
| Drag time selection on spectrogram | ✅ COMPLETE | Click-drag in non-edit mode |
| Crosshair cursor | ✅ COMPLETE | Time/frequency readout in status bar |
| Boundary lines on spectrogram + waveform | ✅ COMPLETE | Interval (thick) and point (thin) dashed lines |

### 3.3 Milestone 3 — Editing Refinements (Current)

| Feature | Status | Notes |
|---------|--------|-------|
| Undo/redo (Ctrl+Z / Ctrl+Y) | ✅ COMPLETE | Command-based, per-frame diffs, 100-entry stack |
| Frame interpolation during drawing | ✅ COMPLETE | Fills skipped frames with linear interpolation |
| Eraser tool (right-click drag) | ✅ COMPLETE | Reverts to original Praat values, grey feedback |
| Interpolate between edited points | ✅ COMPLETE | Button in control panel, respects selection range |
| Drag-to-adjust existing formant points | 🔲 DEFERRED | Removed due to conflict with freehand draw; needs modifier key |
| Adjustable formant time resolution | 🔵 FUTURE | Low priority |

### 3.4 Known Issues

- Parselmouth's Spectrogram object does not expose `get_frequency_from_frequency_bin_number`; resolved by using the `.xs()` and `.ys()` methods instead.
- The pre-emphasis slider minimum must be 1 Hz (not 0), as Praat's Burg method requires a positive value.
- Drag-to-adjust existing formant points was implemented but removed — left-click detection near existing points conflicted with freehand drawing. Needs a modifier key (e.g. Ctrl+click) approach.

---

## 4. Roadmap

### 4.1 Milestone 4 — Export & Distribution

| Feature | Status | Priority |
|---------|--------|----------|
| Export to Praat Formant object format | 🔵 FUTURE | Medium |
| Export formant values as CSV | 🔵 FUTURE | Medium |
| Batch processing (directory of files) | 🔵 FUTURE | Low |
| PyInstaller packaging (.exe) | 🔵 FUTURE | High |
| pip-installable package | 🔵 FUTURE | Medium |

---

## 5. Architecture

### 5.1 Application Structure

The application is a single Python file (`formant_editor.py`) organised into the following classes:

| Class | Responsibility |
|-------|---------------|
| FormantData | Data container for formant values (original + edited), with save/load to JSON .formants format, edit tracking via edited_mask, and per-formant reset. |
| Interval, Point, Tier, TextGrid | TextGrid data model — parsing (normal + short format), serialisation, and in-memory representation. |
| LabelEdit | Inline QLineEdit for editing interval/point labels with Tab-play and Escape support. |
| SpectrogramCanvas | Matplotlib FigureCanvas embedded in Qt. Handles spectrogram/waveform/TextGrid rendering, formant overlay, mouse interaction (editing, boundary drag, selection), audio playback with animated cursor, zoom/scroll, crosshair, undo/redo. |
| ControlPanel | Qt widget containing all sliders, spin boxes, buttons, and the edit mode controls. |
| MainWindow | QMainWindow subclass. Orchestrates file operations, formant analysis, keyboard events, menu bar, signal routing, and scrollbar management. |

### 5.2 Data Flow

1. User opens a WAV file → Parselmouth loads it as a Sound object.
2. Sound → `to_spectrogram()` → spectrogram data (power values, frequency bins, time bins) for display.
3. Sound → `to_formant_burg()` → Formant object → extracted into FormantData (NumPy arrays of times and values).
4. SpectrogramCanvas renders spectrogram via `imshow`, waveform, formant scatter points, and TextGrid tiers using a GridSpec layout with shared x-axis.
5. Mouse editing writes directly to `FormantData.values` and sets `edited_mask`. Changes are recorded in the undo stack as per-frame diffs.
6. Save serialises FormantData to JSON; load deserialises and replaces the current FormantData.
7. TextGrid files are parsed into the Tier/Interval/Point data model, displayed as synchronised axes, and saved back in Praat normal text format.

### 5.3 Key Design Decisions

- **JSON file format:** Chosen over binary formats for human readability, version control compatibility, and ease of integration with analysis scripts (Python, R). The slight file size increase is negligible for formant data.
- **5 formant slots always stored:** Even when only F1–F3 are calculated, the data structure holds 5 rows (with NaN for unused formants). This prevents schema changes if the user later increases the analysis to F1–F5.
- **Separate original_values:** The original Praat-calculated values are stored alongside edited values, enabling per-point reset without re-running the analysis.
- **Matplotlib for rendering:** While a custom OpenGL canvas could be faster for large files, matplotlib provides well-tested spectrogram rendering, scientific colour maps, and a familiar API. Performance is adequate for typical phonetics files (up to several minutes of audio). `imshow` with bilinear interpolation is used instead of `pcolormesh` (Gouraud shading was too slow for large spectrograms). Blitting is used for responsive mouse interactions (drawing, boundary drag, selection, crosshair).

---

## 6. Testing

### 6.1 Testing Philosophy

FormantStudio handles scientific data — researchers will make analytical decisions based on the formant values this tool produces and saves. A bug that silently corrupts formant data, misaligns values with the time axis, or loses edits could invalidate research findings. The test suite must therefore prioritise **data integrity** above all else, with secondary focus on analysis correctness and UI behaviour.

All tests use **pytest** as the test framework. Tests are organised into three tiers: unit tests (fast, no GUI), integration tests (data pipeline end-to-end), and GUI tests (Qt event simulation). The full suite should run in under 60 seconds on a standard machine.

### 6.2 Test Infrastructure

#### 6.2.1 Synthetic Test Signals

Tests should avoid depending on external audio files. Instead, test signals are generated programmatically using Parselmouth or NumPy:

| Signal | Construction | Purpose |
|--------|-------------|---------|
| Steady-state vowel | Sum of harmonics (f0 = 120 Hz, 30 harmonics) with formant-shaped spectral envelope | Predictable F1/F2/F3 values for extraction validation |
| Silence | Zero-valued array | Edge case: no formants detectable |
| Pure tone | Single sinusoid at known frequency | Verify spectrogram frequency axis accuracy |
| White noise | `np.random.randn()` | Stress test: formant tracker should return mostly NaN or unreliable values |
| Chirp | Linear frequency sweep 100–5000 Hz | Verify time-frequency alignment in spectrogram |
| Long file | 5+ minutes of generated speech-like signal | Performance and memory regression testing |

#### 6.2.2 Reference Data

For validation against Praat, generate "ground truth" formant values by running Praat's analysis directly via Parselmouth and storing the results. Tests then compare FormantStudio's extracted values against this reference. Since FormantStudio uses the same Parselmouth calls, these should match exactly — any discrepancy indicates a bug in the extraction or storage pipeline.

#### 6.2.3 Temporary File Fixtures

All tests that write files should use pytest's `tmp_path` fixture. No test should write to the working directory or leave artefacts.

### 6.3 Unit Tests — FormantData Class

The FormantData class is the core data structure. Every operation on it must be tested rigorously.

#### 6.3.1 Construction & Properties

| Test | Assertion |
|------|-----------|
| `test_empty_construction` | Default FormantData has 0 frames, empty arrays, correct n_formants |
| `test_construction_with_data` | Provided times and values are stored correctly, shapes match |
| `test_original_values_are_copy` | Modifying `values` does not modify `original_values` (deep copy verification) |
| `test_n_frames_property` | Returns `len(times)` accurately |
| `test_nan_initialization` | Unused formant slots (e.g., F4/F5 when only 3 calculated) are NaN |

#### 6.3.2 Editing

| Test | Assertion |
|------|-----------|
| `test_set_value_updates_values` | `set_value(f, t, hz)` writes to correct position in values array |
| `test_set_value_sets_edited_mask` | `edited_mask[f, t]` is True after `set_value` |
| `test_set_value_preserves_original` | `original_values[f, t]` is unchanged after editing |
| `test_set_value_boundary_formant_0` | Editing formant index 0 works correctly |
| `test_set_value_boundary_formant_4` | Editing formant index 4 (F5) works correctly |
| `test_set_value_boundary_frame_first` | Editing frame index 0 works |
| `test_set_value_boundary_frame_last` | Editing the last frame works |
| `test_set_value_out_of_range_formant` | Indices outside 0–4 are silently ignored (no crash) |
| `test_set_value_out_of_range_frame` | Negative or too-large frame indices are silently ignored |
| `test_set_value_nan` | Setting a value to NaN is valid (represents "no formant here") |
| `test_set_value_zero` | Setting a value to 0.0 is stored as 0.0, not converted to NaN |
| `test_multiple_edits_same_point` | Last write wins; edited_mask remains True |
| `test_edit_does_not_affect_other_formants` | Editing F1 does not change F2–F5 |

#### 6.3.3 Reset

| Test | Assertion |
|------|-----------|
| `test_reset_single_formant` | `reset_to_original(f)` restores values[f] and clears edited_mask[f] |
| `test_reset_single_preserves_others` | Resetting F1 does not affect F2's edits or mask |
| `test_reset_all` | `reset_to_original()` restores all values and clears all masks |
| `test_reset_idempotent` | Calling reset twice has no additional effect |
| `test_reset_after_no_edits` | Reset on unedited data is a no-op (no crash, values unchanged) |

#### 6.3.4 Save/Load Round-Trip

| Test | Assertion |
|------|-----------|
| `test_save_creates_file` | File exists at expected path after save |
| `test_save_valid_json` | File parses as valid JSON |
| `test_save_schema_fields` | All required fields present: format_version, generator, analysis_engine, n_formants, n_frames, time_step_seconds, times, values, edited_mask |
| `test_save_format_version` | format_version == 1 |
| `test_save_generator` | generator == "FormantStudio" |
| `test_roundtrip_times` | Saved and reloaded times arrays match within float64 precision |
| `test_roundtrip_values` | Saved and reloaded values match (including NaN positions) |
| `test_roundtrip_edited_mask` | Saved and reloaded edited_mask arrays match exactly |
| `test_roundtrip_nan_as_null` | NaN values are serialised as JSON null and deserialised back to NaN |
| `test_roundtrip_time_step` | time_step_seconds is preserved |
| `test_roundtrip_n_formants` | n_formants is preserved |
| `test_roundtrip_with_edits` | Edited values and masks survive save/load |
| `test_roundtrip_no_edits` | Unedited data survives save/load |
| `test_roundtrip_all_nan` | A formant track that is entirely NaN survives round-trip |
| `test_roundtrip_precision` | Values are stored with 2 decimal places (as per the round() call in save) |
| `test_load_nonexistent_file` | Raises appropriate exception |
| `test_load_malformed_json` | Raises appropriate exception |
| `test_load_missing_fields` | Raises KeyError or similar for incomplete files |
| `test_save_overwrites_existing` | Saving to an existing path replaces the file |

### 6.4 Unit Tests — Formant Extraction

#### 6.4.1 Praat Equivalence

| Test | Assertion |
|------|-----------|
| `test_extraction_returns_formant_data` | `extract_formants_from_praat()` returns a FormantData instance |
| `test_extraction_frame_count` | Number of frames matches Praat's formant object `n_frames` |
| `test_extraction_times_match_praat` | Times array matches `formant_obj.xs()` exactly |
| `test_extraction_values_match_praat` | Each F1–Fn value matches `formant_obj.get_value_at_time()` |
| `test_extraction_default_3_formants` | With n_formants=3, F1–F3 are populated, F4–F5 are NaN |
| `test_extraction_5_formants` | With n_formants=5, all five tracks have data where Praat detects them |
| `test_extraction_time_step_computed` | time_step matches the actual difference between consecutive frame times |

#### 6.4.2 Parameter Sensitivity

| Test | Assertion |
|------|-----------|
| `test_max_formant_affects_values` | Changing max_formant from 5500 to 5000 produces different F values |
| `test_window_length_affects_frame_count` | Shorter window → more frames |
| `test_pre_emphasis_affects_values` | Different pre-emphasis values produce different formant estimates |
| `test_time_step_none_uses_auto` | Passing time_step=0 uses Praat's auto (25% of window) |

#### 6.4.3 Edge Cases

| Test | Assertion |
|------|-----------|
| `test_extraction_silence` | Returns FormantData with mostly NaN values (no crash) |
| `test_extraction_very_short_audio` | Audio shorter than one window still returns a result (possibly 0 or 1 frame) |
| `test_extraction_high_sample_rate` | 48 kHz audio works correctly |
| `test_extraction_low_sample_rate` | 8 kHz audio works (max_formant should be adjusted) |
| `test_extraction_mono_only` | Stereo files should be handled (Parselmouth may auto-convert) |

### 6.5 Unit Tests — Spectrogram Computation

| Test | Assertion |
|------|-----------|
| `test_spectrogram_shape` | `values.shape == (n_freq_bins, n_time_bins)` |
| `test_spectrogram_freq_axis` | `ys()` range covers 0 to approximately max_frequency |
| `test_spectrogram_time_axis` | `xs()` range covers approximately 0 to sound duration |
| `test_spectrogram_pure_tone` | A 1000 Hz pure tone produces peak energy at ~1000 Hz in the spectrogram |
| `test_spectrogram_silence` | Silent audio produces uniformly low power values |
| `test_spectrogram_values_non_negative` | All spectrogram power values ≥ 0 (power spectral density) |

### 6.6 Integration Tests — Data Pipeline

These tests exercise the full pipeline from audio to saved file and back.

| Test | Assertion |
|------|-----------|
| `test_full_pipeline_open_analyse_save_load` | Generate audio → extract formants → edit some values → save → load → verify all values and masks match |
| `test_pipeline_edit_then_reanalyse` | After editing, re-analysing replaces FormantData with fresh Praat values (edits are lost, as expected) |
| `test_pipeline_save_location_matches_wav` | Saving formants for `/path/to/recording.wav` creates `/path/to/recording.formants` |
| `test_pipeline_auto_detect_formants_file` | If `recording.formants` exists when `recording.wav` is opened, it is detected |
| `test_pipeline_time_alignment_with_textgrid` | Formant frame times can be used to extract values within TextGrid interval boundaries (using textgrid library to parse a test TextGrid) |
| `test_pipeline_large_file_performance` | A 5-minute audio file completes analysis in under 10 seconds |

### 6.7 GUI Tests

GUI tests use `pytest-qt` and operate on the actual Qt widgets with simulated events. All GUI tests must set `QT_QPA_PLATFORM=offscreen` for headless CI execution.

#### 6.7.1 Application Lifecycle

| Test | Assertion |
|------|-----------|
| `test_window_creates` | MainWindow instantiates without error |
| `test_initial_state` | Canvas has no sound, no formant data; edit mode is off; sliders at defaults |
| `test_status_bar_initial` | Status bar shows "Ready" message |

#### 6.7.2 Control Panel Interactions

| Test | Assertion |
|------|-----------|
| `test_edit_button_toggles` | Clicking edit button toggles `canvas.edit_mode` |
| `test_edit_button_visual_state` | Button is checked/red when active, unchecked when inactive |
| `test_dynamic_range_slider` | Moving slider updates `canvas.dynamic_range` |
| `test_brightness_slider` | Moving slider updates `canvas.brightness` |
| `test_formant_count_dropdown` | Changing dropdown value updates the display range |
| `test_f_key_in_edit_mode` | Pressing F2 while edit mode is on sets `canvas.active_formant` to 1 |
| `test_f_key_outside_edit_mode` | Pressing F2 while edit mode is off does NOT change active_formant |
| `test_f_key_beyond_display_range` | Pressing F4 while showing F1–F3 has no effect |
| `test_reset_current_formant_button` | Clicking reset reverts the active formant |
| `test_reset_all_button` | Clicking reset all reverts everything |

#### 6.7.3 Mouse Editing

These tests simulate mouse events on the matplotlib canvas. They require a loaded audio file.

| Test | Assertion |
|------|-----------|
| `test_click_in_edit_mode_sets_value` | Mouse press at known (time, freq) position writes the correct value to FormantData |
| `test_drag_sets_multiple_values` | Mouse press + move + release writes values at multiple frames |
| `test_click_outside_edit_mode_no_effect` | Mouse interaction without edit mode does not modify FormantData |
| `test_click_outside_axes_no_effect` | Clicking outside the spectrogram area does not crash |
| `test_edit_respects_active_formant` | Editing writes to the correct formant index |

### 6.8 Regression Tests

These tests are added as bugs are discovered and fixed, to prevent recurrence.

| Test | Relates To | Assertion |
|------|-----------|-----------|
| `test_spectrogram_uses_xs_ys` | Bug: `get_frequency_from_frequency_bin_number` not available | Spectrogram computation uses `.xs()` and `.ys()` methods, not bin-number methods |
| `test_pre_emphasis_minimum_1` | Bug: Pre-emphasis of 0 crashes Praat | Pre-emphasis value passed to Praat is always ≥ 1.0 |
| `test_time_step_none_not_zero` | Bug: `time_step=0.0` rejected by Parselmouth | Auto time step passes `None`, not `0.0` |
| `test_n_formants_is_float` | Bug: Parselmouth expects float for max_number_of_formants | The value passed is `float(n_formants)` |

### 6.9 Test Execution

#### 6.9.1 Running the Tests

```bash
# Run all tests
pytest tests/ -v

# Run only unit tests (fast, no GUI)
pytest tests/ -v -k "not gui"

# Run with coverage report
pytest tests/ -v --cov=formant_editor --cov-report=term-missing

# Run GUI tests (requires display or offscreen platform)
QT_QPA_PLATFORM=offscreen pytest tests/ -v -k "gui"
```

#### 6.9.2 Coverage Targets

| Component | Target | Rationale |
|-----------|--------|-----------|
| FormantData | 100% | Core data structure; every code path must be verified |
| extract_formants_from_praat | 95%+ | Critical analysis pipeline; edge cases may be hard to trigger |
| SpectrogramCanvas (data logic) | 90%+ | Rendering code is hard to test, but data manipulation should be covered |
| ControlPanel | 80%+ | Widget wiring; some visual-only code excluded |
| MainWindow | 80%+ | Orchestration logic; file dialogs are mocked |

#### 6.9.3 CI Integration

Tests should be runnable in a headless CI environment (GitHub Actions, etc.) with the following setup:

```yaml
# Example GitHub Actions step
- name: Run tests
  env:
    QT_QPA_PLATFORM: offscreen
  run: |
    pip install pytest pytest-qt pytest-cov
    pip install -r requirements.txt
    pytest tests/ -v --cov=formant_editor
```

### 6.10 Test Dependencies

| Package | Purpose |
|---------|---------|
| pytest | Test framework |
| pytest-qt | Qt widget testing (simulated clicks, key presses) |
| pytest-cov | Coverage reporting |
| numpy (already installed) | Array comparisons with `np.testing.assert_array_equal` and `assert_array_almost_equal` |

---

## 7. Citation & Attribution

FormantStudio uses Praat's acoustic analysis algorithms. When publishing research that uses FormantStudio for formant analysis, the underlying engine should be cited as:

> Boersma, Paul & Weenink, David (2024). Praat: doing phonetics by computer [Computer program]. Version 6.4.xx. Retrieved from http://www.praat.org/

The Parselmouth Python library should be cited as:

> Jadoul, Y., Thompson, B., & de Boer, B. (2018). Introducing Parselmouth: A Python interface to Praat. *Journal of Phonetics*, 71, 1–15.
