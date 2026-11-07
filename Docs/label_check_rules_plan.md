# Label Check: corpus profiles and configurable rules

Label Check (Tools → Check Label Files) runs three fixed checks over a folder of TextGrids:
unlabelled intervals, unknown symbols, and boundaries that nearly meet a reference tier. This
plan makes it configurable per corpus. The settings and rules live in a **corpus profile**, a
JSON file saved in the TextGrid folder (`formantstudio-profile.json`) and loaded automatically
whenever that folder is checked or built into a CSV. Users can also save and load profiles
elsewhere, to share them between corpora.

Decisions already taken with the user (2026-11-07):

- **Non-speech labels** (e.g. `-`, `<p:>`, `<noise>`, `<speech noise>`) are configurable. Every
  label in the list is "nothing to analyse": Label Check never treats it as an unknown symbol,
  and Build CSV leaves its intervals out. The configuration UI must say so plainly, because a
  prosody study that measures pauses has to keep its pause label *out* of the list.
- **Containment** (e.g. "every phone has exactly one `phone_mark` point"): marks attach to speech
  segments only, never to non-speech intervals. A point sitting exactly on a boundary cannot be
  assigned to either side, so it is flagged for checking rather than guessed.
- **Sequence rules** are patterns to flag, specified by the user, e.g. "`>` followed by a
  non-speech interval", "`>` at the end of the tier", "`<` preceded by a non-speech interval".
  Adjacent arrows (`> >`) are fine unless a rule says otherwise. Anything not flagged by a rule or
  by the built-in checks passes.
- A label error with no obvious repair shows a clear placeholder, not `—` (which reads as a
  pause marker).
- Label Check stays open while issues are opened in the main window.
- (Revised after trying Stage 2) **Profiles are saved explicitly**, with a Save to Corpus button
  and a status line saying whether there is a profile and whether the screen has unsaved
  changes. Running a check saves nothing. Non-speech labels are one list for all tiers, and the
  UI says so.
- (Revised after trying Stage 2) **Mislabelled segments are configured per tier**: each tier
  carries any label, the symbols of a notation (IPA, SAMPA or X-SAMPA, optionally only an
  inventory of them), or exactly a list of its own (case-sensitive, e.g. `T, T1, T2`).

1. Stage 1: Working alongside the editor

1. [x] Show "(no suggestion — double-click to type)" in grey italics in an empty Proposed cell, and
   "(cannot snap safely)" for an unsafe boundary, instead of `—`. Display only: editing starts
   from an empty cell.
2. [x] Make Label Check non-modal: the main window stays usable while it is open, and reopening
   it from the Tools menu raises the existing window with its results intact.
3. [x] Split `MainWindow.open_file()` into the file picker and `open_audio(path)`, so recordings
   can be opened by path. Behaviour of File → Open is unchanged.
4. [x] Double-clicking an issue (or pressing "Open in Editor") opens that recording and its
   TextGrid in the main window, makes the issue's tier active, selects the interval or point and
   zooms to it. The audio is looked for in the dialog's optional Audio folder, then beside the
   TextGrid, then in the folder a recording was last opened from; if neither has it, the TextGrid opens on its own (no prompt) and
   File → Open Recording for TextGrid… adds the recording later, keeping any edits. Unsaved
   edits in the main window are protected by the usual prompt.
6. [x] A TextGrid can be opened without its recording: its own span gives the time axis, and
   labels can be navigated, edited and saved. Load TextGrid with no recording open does this.
5. [x] Applying fixes to the file that is open in the main window reloads it there if it has no
   unsaved edits, and warns instead if it does (saving would overwrite the fixes).

2. Stage 2: Corpus profile and non-speech labels

1. [x] New Qt-free module `corpus_profile.py`: a `CorpusProfile` dataclass holding the
   non-speech labels and every Label Check setting (audio folder, tiers per check, notation,
   inventory, reference tier, tolerance), with `load(path)` / `save(path)` to JSON (versioned) and
   `for_folder(dir)` returning the folder's profile or the defaults. Unit tests for round-trip,
   missing file, unknown version, and malformed JSON (reported, not raised).
2. [x] `label_check`: replace the hard-coded `PAUSE_MARKER` with the profile's non-speech labels
   (skipped by the symbol check, accepted by the unlabelled check; the unlabelled fix proposes
   the first label in the list).
3. [x] Label Check dialog: a "Non-speech labels" field with the explanation above as visible
   text, not just a tooltip; the folder's profile loaded automatically when the folder is
   chosen. Saving was later made explicit (see 3.2).
4. [x] Build CSV: replace the `EMPTY_LABEL_MARKERS` constant with the profile's non-speech list,
   passed down as a parameter (no globals). The wizard shows the list it will use, with the same
   warning, and saves any change to the folder's profile.
5. [x] `divergence.py`: treat every non-speech label as `EMPTY` (an insertion against a label;
   nothing to analyse against another non-speech label), with the list passed in. Check against
   the existing divergence tests before changing semantics.

3. Stage 3: Labels per tier, and explicit profiles

1. [x] Profile: `label_check.label_rules`, one `LabelRule(tier, kind, labels)` per checked tier,
   replacing the single notation, inventory and symbol-tier selection. `kind` is `ipa`, `sampa`,
   `xsampa` (labels, when given, are the inventory) or `list` (exactly these labels,
   case-sensitive). Rules for tiers a corpus lacks are kept, so a shared profile loses nothing.
2. [x] `label_check`: `check_labels(tg, name, policies)` judges each tier by its own
   `LabelPolicy(accept, candidates, kind)`; a listed label that is wrong is reported as
   `NOT_ALLOWED`, with repairs from `suggest_symbols` (`t2` → `T2`).
3. [x] Dialog: the Mislabelled check is a table, one row per tier, with the rule and its labels.
   Profiles: Save to Corpus / Load… / Save As…, a status line tracking unsaved changes, nothing
   saved as a side effect of running.

4. Stage 4: Containment rules

1. [ ] Rule: on *child* tier (points or intervals), every speech interval of *parent* tier holds
   between *min* and *max* items (e.g. exactly 1). Flags: too few, too many, a child item inside
   a non-speech interval or outside any interval, and a point exactly on a parent boundary
   ("cannot tell which segment it belongs to"). No automatic fixes.
2. [ ] Dialog: add / remove / edit containment rules in a small table.

5. Stage 5: Sequence rules

1. [ ] Rule: on *tier*, flag an interval labelled *label* whose *previous* or *next* neighbour
   matches a condition: a specific label, any non-speech label, a blank interval, or the edge of
   the tier. No automatic fixes.
2. [ ] A "Deletion arrows" preset adding the four rules agreed above (`>` before non-speech,
   `>` at the end, `<` after non-speech, `<` at the start) plus `>` immediately followed by `<`.
3. [ ] Dialog: add / remove / edit sequence rules, and insert the preset.

6. Stage 6: Documentation

1. [ ] Methods and Guide: a Label Check section covering profiles, non-speech labels (with the
   prosody caveat), and each rule type with an example.
