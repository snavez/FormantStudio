# FormantStudio — Manual Formant Editor

A standalone tool for viewing, analysing, and **manually correcting** formant tracks in speech audio. Built on Praat's acoustic analysis engine (Boersma & Weenink) via [Parselmouth](https://github.com/YannickJadoul/Parselmouth), with a modern PyQt6 editing interface.

## Why?

Praat's automatic formant tracking is excellent but not infallible — particularly for non-modal phonation, nasalised vowels, or noisy recordings. Existing tools like FastTrack help reduce gross errors, but sometimes you just need to **draw the formant where you can see it**. FormantStudio gives you that ability.

## Features

- **Praat-powered analysis**: Formant extraction uses Praat's Burg algorithm (identical results to Praat itself)
- **Spectrogram display** with adjustable dynamic range, brightness, and pre-emphasis
- **Color-coded formant overlay**: F1 (blue), F2 (yellow), F3 (red), F4 (green), F5 (magenta)
- **Click-and-drag formant editing**: Enter edit mode, select a formant (F1–F5 keys), draw corrections
- **Save/load** edited formant data as `.formants` files alongside your TextGrids
- **Cross-platform**: Windows, macOS, Linux

## Installation

```bash
# Requires Python 3.9+
pip install -r requirements.txt

# Run
python formant_editor.py
```

## Quick Start

1. **Open a WAV file** (Ctrl+O)
2. Adjust spectrogram display using the sliders (Dynamic Range, Brightness)
3. Click **EDIT MODE** (or press the Edit button)
4. Press **F1**, **F2**, or **F3** to select which formant to draw
5. **Click and drag** on the spectrogram to correct formant values
6. **Ctrl+S** to save (creates a `.formants` file next to your WAV)

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

## Roadmap

- [ ] TextGrid display overlay
- [ ] Zoom and scroll (time selection)
- [ ] Interpolation tools for edited regions
- [ ] Export to Praat Formant object format
- [ ] Adjustable formant time resolution
- [ ] PyInstaller packaging for non-Python users

## License

MIT
