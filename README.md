# FormantStudio — Manual Formant Editor

A standalone tool for viewing, analysing, and **manually correcting** formant tracks in speech audio. Built on Praat's acoustic analysis engine (Boersma & Weenink) via [Parselmouth](https://github.com/YannickJadoul/Parselmouth), with a modern PyQt6 editing interface.

**[Download the latest Windows executable](https://github.com/snavez/FormantStudio/releases/latest)** — no Python installation required.

## Why?

Praat's automatic formant tracking is excellent but not infallible — particularly for non-modal phonation, nasalised vowels, or noisy recordings. Existing tools like FastTrack help reduce gross errors, but sometimes you just need to **draw the formant where you can see it**. FormantStudio gives you that ability.

## Features

### Display & Navigation
- **Greyscale spectrogram** with adjustable dynamic range, brightness, window length, and max frequency
- **Waveform display** above the spectrogram with RMS readout
- **TextGrid overlay** — IntervalTier and TextTier display below the spectrogram
- **Zoom** via mouse scroll wheel (cursor-centred) or Ctrl+I / Ctrl+O
- **Scroll/pan** via scrollbar with arrow buttons (2/3 window overlap)
- **Crosshair cursor** with time/frequency readout in status bar
- **Drag time selection** on spectrogram (click-drag in non-edit mode)
- **Audio playback** (Tab to play selection/view, Esc to stop) — reliable for both short and long segments

### Formant Editing
- **Praat-powered analysis**: Formant extraction uses Praat's Burg algorithm (identical results to Praat)
- **Color-coded formant overlay**: F1 (blue), F2 (yellow), F3 (red), F4 (green), F5 (magenta)
- **Click-and-drag drawing**: Enter edit mode, select a formant (F1–F5 keys), draw corrections
- **Frame interpolation**: Fast mouse movement fills skipped frames automatically
- **Eraser tool**: Right-click drag reverts points to original Praat values
- **Undo/redo**: Ctrl+Z / Ctrl+Y (100-step history)
- **Unsaved changes warning**: Prompts before closing or opening a new file with unsaved edits
- **Reset**: Per-formant or all-formant reset (undoable)

### TextGrid Editing
- **Create or load TextGrid** — on opening a WAV, choose to create a new TextGrid or load an existing one
- **Load/save TextGrid files** (Praat normal + short format)
- **Add boundaries** (Enter key at cursor/hover position)
- **Delete boundaries and points** (Del key on selected boundary or point)
- **Drag boundaries** to adjust timing (blit-based visual feedback)
- **Shift+drag** for aligned multi-tier boundary movement
- **Snap-to-boundary** with adjustable tolerance for precise alignment
- **Inline label editing** with live update
- **Add/remove tiers** (IntervalTier and TextTier)

### Batch CSV Export
- **Build CSV wizard** (Tools > Build CSV) for batch extraction of formant values across multiple files
- **Two extraction modes**: percentage-based (e.g. 25%, 50%, 75% of each interval) or absolute time points
- **Point tier support**: Extract labels and times from TextTier point tiers
- **Phonetic categorisation**: Optionally classify labels by manner, place, voicing, height, backness, and rounding using an IPA/SAMPA chart
- **Diphthong handling**: Automatic detection and classification of diphthongs and vowel combinations

### IPA/SAMPA Reference
- **Built-in IPA/SAMPA chart** (View menu) — quick reference panel for phonetic symbols
- **Compact vowel layout** with diacritics section
- **Click-to-insert** labels directly into TextGrid tiers

### File Management
- **Save/load** edited formant data as `.formants` files alongside your TextGrids
- **Auto-detect** existing `.formants` and `.TextGrid` files when opening a WAV
- **Filename in title bar** — always shows the currently loaded audio file

## Installation

### Windows Executable (recommended)

Download `FormantStudio.exe` from the [latest release](https://github.com/snavez/FormantStudio/releases/latest) — no installation needed, just run it.

### From Source

```bash
# Requires Python 3.9+
pip install -r requirements.txt

# Run
python formant_editor.py
```

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl+Shift+O | Open WAV file |
| Ctrl+S | Save formants |
| Ctrl+Shift+S | Save TextGrid |
| Ctrl+L | Load formants |
| Ctrl+T | Load TextGrid |
| Ctrl+E | Toggle edit mode |
| F1–F5 | Select formant (in edit mode) |
| Ctrl+Z | Undo |
| Ctrl+Y | Redo |
| Ctrl+I | Zoom in |
| Ctrl+O | Zoom out |
| Ctrl+N | Zoom to selection |
| Ctrl+A | Zoom all |
| Tab | Play selection/view |
| Esc | Stop playback / clear selection |
| Enter | Add boundary at cursor |
| Del | Delete selected boundary/point |
| F | Toggle formant visibility |
| Scroll wheel | Zoom in/out (centred on cursor) |

## .formants File Format

The saved file is JSON with this structure:
```json
{
  "format_version": 1,
  "generator": "FormantStudio",
  "analysis_engine": "Praat (via Parselmouth)",
  "n_formants": 5,
  "n_frames": 1234,
  "time_step_seconds": 0.00625,
  "times": [0.003125, 0.009375, ...],
  "values": [
    [500.0, 502.3, null, ...],
    [1500.2, 1498.7, ...],
    ...
  ],
  "edited_mask": [
    [false, false, true, ...],
    ...
  ]
}
```

- `null` values indicate no formant detected at that frame
- `edited_mask` tracks which values were manually corrected
- `time_step_seconds` + `times` array ensure frame-accurate alignment with TextGrids

## Citing

If you use FormantStudio in research, the formant analysis itself should be cited as:

> Boersma, Paul & Weenink, David (2024). Praat: doing phonetics by computer [Computer program]. Retrieved from http://www.praat.org/

> Jadoul, Y., Thompson, B., & de Boer, B. (2018). Introducing Parselmouth: A Python interface to Praat. *Journal of Phonetics*, 71, 1–15.

## Roadmap

- [ ] Export to Praat Formant object format
- [ ] Adjustable formant time resolution
- [ ] Batch processing across a directory of files (beyond CSV export)

## License

MIT
