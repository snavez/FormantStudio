# FormantStudio — Methods & User Guide

**Audience:** users and reviewers who want to know *what FormantStudio measures, how it
measures it, what they can change, and why the defaults are what they are.*

This document is the **methods and usage reference**. Its companion,
`FormantStudio_Specification.md`, is the implementation contract (what the code must do);
where the two overlap, the Specification governs behaviour and this document explains
intent, rationale, and use. Nothing here is a hidden setting — every parameter named as
tweakable is exposed in the UI, and every parameter named as fixed is a deliberate design
choice recorded with its reason.

---

## 1. What FormantStudio is

FormantStudio is a Praat-adjacent tool for **manual formant editing** and **acoustic data
extraction** from labelled speech. It displays a spectrogram and waveform, overlays
Praat-computed formant tracks that you can hand-correct, lets you build and edit TextGrid
tier annotations, and exports per-segment acoustic measurements to CSV for downstream
statistics and plotting (e.g. in FRED).

It computes two families of acoustic data:

- **Formants** (F1–F5): resonances of the vocal tract, primarily for vowels.
- **Spectral moments** (COG, SD, skewness, kurtosis): the shape of the frication/aspiration
  spectrum, primarily for consonants (fricatives, stop releases).

It is built on **parselmouth**, the Python binding to **Praat** (Boersma & Weenink), so the
underlying numerical methods are Praat's.

---

## 2. Display vs. analysis — an important distinction

Two things on screen look like "the spectrogram settings" but do **not** affect any exported
number:

- The **spectrogram image** (greyscale, reversed so dark = high energy) and its controls —
  Dynamic Range (default 70 dB), Brightness, display Window (default 5 ms), Max Frequency
  (default 8000 Hz). These change only how the picture looks. They are a **display** of a
  Praat spectrogram and never enter the formant or spectral-moment calculations.
- The ≤ 10 s spectrogram-view limit is a rendering performance guard, not an analysis limit.

Everything that lands in a CSV comes from the **formant analysis** (§3) and the
**spectral-moment analysis** (§5), each of which has its own, separate window and parameters.
Changing the on-screen spectrogram never changes your data.

_Why this matters:_ users often assume tightening the display window sharpens their formants.
It does not — formant analysis has its own window length (§3.1). Keeping display and analysis
independent means you can make the picture legible without perturbing the measurements.

---

## 3. Formant analysis

### 3.1 Method

Formants are computed by **Praat's Burg LPC algorithm** (`Sound: To Formant (burg)`, via
parselmouth `to_formant_burg`). LPC models the spectrum as an all-pole filter; the Burg method
estimates those poles, and the pole frequencies are the formants.

FormantStudio **always extracts five formants** and stores all five, regardless of how many
you choose to display. The "Show formants" control (F1–F1 … F1–F5, default **F1–F3**) only
changes how many tracks are drawn and exported — switching it never re-runs the analysis. This
is deliberate: it lets you reveal F4/F5 instantly without recomputation, and keeps the stored
data complete.

### 3.2 Parameters you can change

| Parameter | Default | Range | Effect |
|---|---|---|---|
| Max formant (Hz) | 5500 | 3000–8000 | Upper frequency Praat fits poles within. ~5500 suits adult male/most speech; raise for higher/child voices. |
| Analysis window length (s) | 0.025 | 0.005–0.1 | LPC analysis frame. Longer = smoother/more stable but less time-local. |
| Pre-emphasis (Hz) | 50 | 1–200 | High-frequency boost before LPC, standard for lifting upper formants. |
| Time step (s) | auto (0) | — | Frame hop. `0` = Praat default (25% of window length). |
| Show formants | F1–F3 | F1–F1 … F1–F5 | Display/export count only; all 5 are always computed. |

These are set in the control panel and applied via **Re-analyse Formants**.

### 3.3 Manual editing

Praat's automatic tracks are a starting point, not the last word. In **Edit Mode** you can:

- **Draw** a corrected formant by dragging on the spectrogram (left-drag), one formant track
  at a time (select F1–F5 with the number keys).
- **Erase** points with right-drag.
- **Undo/Redo** every stroke (Ctrl+Z / Ctrl+Y).
- **Reset** the current formant or all formants back to the Praat estimate.

Edited points are marked distinctly (larger, white-edged) and flagged in the stored data so a
reset knows what to restore and exports can distinguish hand-corrected from automatic values.

### 3.4 Storage — the `.formants` file

Formant data saves as a JSON `.formants` file next to the audio: it records the time step, the
frame times, all five formant value tracks (with missing frames stored as JSON `null`), and an
`edited_mask` marking which points you corrected. Re-opening restores your edits exactly.

### 3.5 Missing values

Where Praat produces no formant at a frame (or after erasing), the value is `NaN` internally
and is written as an **empty cell** in the CSV. Downstream tools (FRED) treat formant-less
tokens as simply absent for that measure.

---

## 4. TextGrid annotation

FormantStudio reads and writes Praat TextGrids (interval tiers and point tiers) and supports
enough editing to prepare extraction targets:

- **Boundaries and points**: click a marker to **select** it (it never moves on a click);
  it moves only once you drag past a small threshold — so aligning a new tier's boundary to an
  existing one by clicking it will not nudge the original.
- **Snapping**: an optional snap-to-nearest-boundary aids cross-tier time alignment.
- **Labels**: click an interval/point to edit its label inline.
- **Tiers**: add, delete, **rename** (click the tier's name in the left margin), and
  **duplicate** (File → Duplicate Tier) — the last is for seeding a tier that mostly mirrors
  another (e.g. an allophone tier from the phoneme tier) so you edit only the differences.

The tier structure you build is what the CSV builder draws on; nothing about the annotation is
hard-coded to particular tier names.

---

## 5. Spectral-moment analysis (consonants)

### 5.1 What it measures

The **spectral moments** treat the (power) spectrum of a short window as a distribution over
frequency and summarise its shape:

- **COG** (centre of gravity, Hz) — the mean frequency; the single most-reported fricative
  measure (separates /s/ from /ʃ/ robustly).
- **SD** (standard deviation, Hz) — spread of energy around the COG.
- **skewness** (dimensionless) — tilt toward low vs high frequencies.
- **kurtosis** (dimensionless) — peakedness/concentration of the spectrum.

They are computed by Praat's `Spectrum` moment functions (via parselmouth
`get_centre_of_gravity` / `get_standard_deviation` / `get_skewness` / `get_kurtosis`) on the
windowed segment's power spectrum, with moment **power = 2** (Praat's default). Skewness and
kurtosis follow Praat's standardized definitions.

_Method provenance:_ Forrest, Weismer, Milenkovic & Dougall (1988) established spectral moments
for obstruents; COG/SD are the most robust, with skew/kurtosis noisier — which is why the
provenance columns (§5.5) exist and why a dynamic/DCT extension is on the roadmap (§8).

### 5.2 Window geometry — the core design

A naïve "always 25 ms" window is a trap: on a 40 ms release it covers most of the segment, on a
160 ms one it covers a sliver, so **window artefacts get confounded with segment duration**.
FormantStudio removes that confound with **proportional windowing** and a strict,
never-lie geometry.

**Two window modes:**

- **Proportional** (default): window width = a chosen fraction of the segment's duration
  (default **30%**), so every segment is treated identically in its own terms.
- **Fixed**: the same width in milliseconds everywhere (default 25 ms), if you specifically
  want that.

**Two clamps (both configurable in the wizard):**

- **Min window** (default 5 ms): the floor. Frequency resolution ≈ 1/window, so 5 ms ≈ 200 Hz —
  fine for COG/SD, marginal for skew/kurtosis. Windows are not allowed narrower than this.
- **Max window** (default 30 ms, proportional mode only): the ceiling. Caps the window on long
  segments so each marker stays a **local** estimate rather than averaging a big chunk of the
  segment. (It does not apply in fixed mode — there, the width is exactly what you typed.)

**Invariants the geometry guarantees (it never lies about location):**

1. The window is always **centred exactly on its marker time**. The reported location and the
   analysed location are identical — a marker is never shifted to make a window fit.
2. A window **never crosses a segment edge** and is **never one-sidedly truncated** — no reading
   of neighbouring-segment audio, no shift of the window's centre of mass near boundaries.
3. If the requested width won't fit centred inside the segment, the window falls back to the
   **min floor**; if even the floor won't fit, the marker is **`too_short`** and its moments are
   left blank rather than fudged.

_Why proportional + these clamps:_ see §2's confound argument, plus Reidy (2015) on estimator
choice for sibilants. The min floor guards frequency resolution / estimation variance; the max
ceiling guards time-locality. They are mirror-image guards at opposite ends.

### 5.3 High-pass filtering

Low-frequency voicing energy drags COG downward and makes it unreliable for voiced or noisy
fricatives. In the Build CSV wizard the **high-pass filter defaults to ON at 300 Hz** (a Praat
Hann-band filter applied once per file before analysis). It is configurable and can be turned
off. (The library-level default constant is off; the wizard turns it on because it is standard
practice for fricative moments.)

### 5.4 Sampling locations

Moments are measured at **percentage markers** through the segment (default 20/50/80%). With the
default 20/50/80 markers and a 30% proportional window, the three windows tile the segment into
non-overlapping thirds for segments up to ~100 ms; beyond that the max-window cap makes them
three local snapshots with gaps, which is the intended trade for long segments.

### 5.5 Output columns and provenance

Per percentage marker `p`, the CSV carries:

- `COG_p%`, `SD_p%`, `skew_p%`, `kurt_p%` — the four moments (blank when unmeasurable).
- `winms_p%` — the **effective** window width actually used (ms), so you can see what was
  measured (blank when `too_short`).
- `winsource_p%` — which rule set the width: `proportional`, `clamped_max`, `fixed`,
  `clamped_min`, or `too_short`.
- `nsamples_p%` — signal samples in the window, for hard reliability thresholds.

These transparency columns are FRED-friendly: you can **filter or group** on `winsource`
(e.g. keep only `proportional`, or drop `clamped_min`/`too_short` before analysing skewness) or
threshold on `nsamples`, without any of them affecting the moments themselves.

### 5.6 Missing values

Consistent with formants, an unmeasurable moment is an **empty cell** — never `0`, never a
guess. FRED ignores spectrally-blank tokens exactly as it ignores formant-blank ones.

---

## 6. Build CSV — the data model

The CSV builder is deliberately generic: it imposes no fixed tier names or hierarchy.

- **Primary tier (the row anchor):** you explicitly choose which tier defines the rows. An
  interval tier gives **one row per labelled segment**; a point tier gives **one row per point**.
- **Time-based label matching (both directions):** every other selected tier contributes a
  column matched to the row token purely by time — the label of the interval *encapsulating* the
  token, the `;`-joined labels of intervals *subdividing* it (e.g. an `allophone` column reading
  `t0; tH` under a `/t/` row), or the marks of points falling inside it. There is no
  parent/child hierarchy to configure; it is all time overlap.
- **Formant sampling:** *at points* (wide numbered columns `F1_Target1`, `F1_Target2`, … sized
  to the corpus maximum, filled in time order) or *for segments* at percentage markers,
  percentage steps, or fixed time steps (ms).
- **Durations:** per interval tier, opt in to `<tier>_dur`, and optionally `<tier>_start` /
  `<tier>_end`. A point primary contributes a single `<tier>_time` column.
- **Segment context:** optionally add `<primary>_prev` / `<primary>_next` — the label of the
  immediately adjacent unit on the primary tier (by time). Blank at a recording edge or an
  unlabelled/pause neighbour. Available to every analysis, not just spectral, so you can group
  by phonetic environment (e.g. release-by-following-vowel) directly in the CSV. (Note: the
  segment *immediately* preceding a stop release is usually its own closure; recovering the
  sound before the closure is a separate, label-scheme-dependent problem left for later.)
- **Spectral moments:** §5, sampled on a chosen segment tier (defaults to the primary tier).
- **Categorisation (optional):** IPA/SAMPA property columns (place, manner, height, etc.)
  derived from labels via the built-in chart.
- **Missing-value convention:** empty cell throughout, for every measure.
- **Unmatched files:** an audio file with no matching TextGrid is reported (not silently blank);
  TextGrids must share the audio file's exact base name.

See `FormantStudio_Specification.md` §2.7 for the full column catalogue and ordering rules.

---

## 7. Parameter reference — tweakable vs fixed

**User-tweakable (exposed in the UI):**

- Formants: max formant, analysis window length, pre-emphasis, time step, display count;
  plus full manual editing.
- Spectrogram *display* only: dynamic range, brightness, display window, max frequency.
- Spectral: window mode (proportional/fixed), width (% or ms), min window, max window,
  window shape (Hamming/Hann), high-pass on/off + cutoff, percentage markers, segment tier.
- Build CSV: primary tier, tier inclusion, formant sampling mode/markers, duration & bounds
  per tier, categorisation options.

**Fixed by design (with reason):**

- Formant analysis always extracts 5 tracks (display count is separate) — so F4/F5 are available
  without re-analysis and stored data is complete.
- Moment `power = 2` — Praat's default, for comparability with existing moment literature.
- Spectral windows are always centred and never truncated — the whole point of the geometry
  (§5.2); this is not negotiable because relaxing it reintroduces the duration confound.
- Missing values are always empty cells — one consistent, machine-detectable convention.

---

## 8. Roadmap for spectral analysis (not yet implemented)

Recorded so they are added deliberately, as new columns, rather than invented ad hoc:

- **Multitaper estimation** (DPSS/Slepian tapers) to stabilise moments from short windows
  (Thomson 1982; Reidy 2015; Shadle et al.).
- **Trajectory + DCT pass**: a dense sliding-window moment track reduced to a few DCT
  coefficients per moment, capturing sibilant *dynamics* rather than three static points
  (Reidy 2016; JASA 2017 moments-vs-DCT comparison), plus the normalized track for smooth-arc
  plotting.
- A **provenance sidecar** (resolved config next to each CSV) for full reproducibility.
- A configurable **`NA` token** — held until confirmed compatible with the downstream grapher.

These are analytical upgrades; the current point-moment CSV is complete and valid without them.

---

## 9. Practical guidance

**Vowel formants.** Set the primary tier to your phoneme/phonetic tier (or a vowel-target point
tier). Sample *at points* if you have hand-placed formant targets (diphthongs get side-by-side
`Target1`/`Target2` columns), or *for segments* at 20/50/80% for a monophthong trajectory. Check
the max-formant setting suits your speakers.

**Consonant spectra.** Put the segment tier on the release/frication interval (e.g. an allophone
tier's `tH`). Keep the high-pass on (~300 Hz). Proportional 30% windows at 20/50/80% are a good
default; watch the `winsource` columns — a lot of `clamped_min`/`too_short` means your segments
are short and you may want to lean on COG/SD rather than skew/kurtosis.

**Combined analyses.** Vowel-formant and consonant-spectral data with *different* row
populations (e.g. one row per vowel vs one row per release) belong in **separate runs / CSVs**;
the tool structures each correctly rather than forcing incompatible rows together. Use the
`_start`/`_end` columns if you need to align two runs on the same audio afterward.

**Reliability filtering (FRED-side).** Prefer categorical filtering on `winsource` and/or a
`nsamples` threshold over ad-hoc COG cutoffs — it is principled and reproducible.

---

## 10. References

- Boersma, P. & Weenink, D. *Praat: doing phonetics by computer.* (The analysis engine; Burg LPC
  formants and the Spectrum moment functions.) https://www.praat.org
- Forrest, K., Weismer, G., Milenkovic, P. & Dougall, R. N. (1988). *Statistical analysis of
  word-initial voiceless obstruents.* JASA 84(1), 115–123.
  https://pubmed.ncbi.nlm.nih.gov/3411039/
- Thomson, D. J. (1982). *Spectrum estimation and harmonic analysis.* Proc. IEEE 70(9),
  1055–1096.
- Reidy, P. F. (2015). *A comparison of spectral estimation methods for the analysis of sibilant
  fricatives.* JASA 137(4), EL248. https://pubs.aip.org/asa/jasa/article/137/4/EL248/940516/
- Reidy, P. F. (2016). *Spectral dynamics of sibilant fricatives are contrastive and language
  specific.* JASA 140(4), 2518. https://pubs.aip.org/asa/jasa/article-abstract/140/4/2518/919684/
- *Spectral moments vs discrete cosine transformation coefficients: evaluation of acoustic
  measures distinguishing two merging German fricatives.* JASA 142(1), 395 (2017).
  https://pubs.aip.org/asa/jasa/article/142/1/395/662504/
- Jongman, A. (2024). *Phonetics of Fricatives* (overview).
  https://kuppl.ku.edu/sites/kuppl/files/documents/publications/Jongman%20OREL%202024%20Phonetics%20of%20Fricatives.pdf
- Shadle, C. H., et al. *Multitaper harmonic analysis of fricatives.*
  https://eprints.soton.ac.uk/259162/1/SSP6mtapr.pdf

_Formant method:_ the Burg LPC estimator is Praat's standard formant algorithm; parameters
(max formant, window length, pre-emphasis) follow Praat conventions. _Spectral method:_ moments
per Forrest et al. (1988); proportional windowing and the min/max clamps are FormantStudio's
design response to the duration-confound and short-segment reliability problems those references
document.
```
