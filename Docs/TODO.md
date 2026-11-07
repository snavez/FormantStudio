# FormantStudio — deferred ideas

Measures and features that have been discussed and deliberately left for later. Each entry says
what it is, why we might want it, and what to watch out for, so it can be picked up with no other
context. The spectral roadmap in `FormantStudio_Methods_and_Guide.md` §8 lists the smaller
analytical upgrades.

1. Formant-derived indices

1. [ ] **Bark difference, Z2 − Z1, at each formant marker.** Convert F1 and F2 to the Bark scale
   (Traunmüller 1990: `Z = 26.81·f / (1960 + f) − 0.53`) and output their difference as a column
   beside the formants (e.g. `Z2-Z1_50%`).
   - *What it is:* Syrdal & Gopal's (1986) front/back dimension. Large for high front vowels (low
     F1, high F2), small for low and back ones — a single per-token "front-high" index.
   - *Why:* it collapses F1 and F2 into one value that can be plotted against a third variable
     such as a release's COG. The hypothesis that prompted it: the lower F1 and the higher F2 of
     the following vowel, the higher the release COG.
   - *Why it is deferred:* the cleaner version of the same index is `F2z − F1z` after
     Lobanov (per-speaker z-score) normalisation, which needs speaker grouping and so belongs in
     FRED or the statistics. The Bark difference is the one variant FormantStudio can compute
     alone, because it is per-token and needs no speaker information; add it if a
     FormantStudio-only index proves useful.
   - *Watch for:* in raw Hz, F2 − F1 is dominated by F2 (roughly three times F1's range), which is
     why the Bark or z-scored form is used. Model F1 and F2 separately as well: a combined index
     assumes they matter equally and oppositely, which the regression
     `COG ~ F1z + F2z + (1|speaker)` can check.

2. Level measures

1. [ ] **Broadband segment level (RMS) for stress and vowel loudness.** Mean level (energy-averaged,
   then converted to dB), peak level and its time as a % of the segment, level at the percentage
   markers, and the energy integral (level combined with duration).
   - *Why:* intensity is a correlate of stress; the energy integral tracks prominence better than
     level or duration alone.
   - *Design notes already worked out:* the window must span 2–3 glottal periods for vowels
     (Praat's intensity rule is 3.2 / pitch floor, ≈ 32 ms at 100 Hz) or the level ripples with
     each pulse, and 5–10 ms for bursts. Reuse the centred, never-crosses-an-edge window geometry.
     No pre-emphasis on this signal. An absolute level measures microphone gain and distance, so
     the useful outputs are relative: to the file's speech level (`speech_level_db`, already
     computed), to the containing interval on a chosen tier (word, phrase), or to an adjacent
     segment.
   - *Watch for:* never normalise a consonant against its following vowel when the question
     involves vowel quality. Vowels differ in intrinsic intensity (low vowels are a few dB louder
     than high ones), so "release minus following vowel" builds a vowel-height effect into the
     measure.

3. Opening files

1. [ ] **Load TextGrid over an open recording should offer to change the recording too.** Today
   File → Load TextGrid swaps the TextGrid and leaves the recording and formants untouched, which
   is right when deliberately pairing a TextGrid with some other recording, but surprising
   otherwise.
   - *Proposal (from the user, 2026-11-07):* when a recording is open and the chosen TextGrid
     does not share its base name, ask: open the name-matched recording from the TextGrid's
     folder if there is one, choose a recording, keep the current one, or close it (TextGrid
     alone, see `open_textgrid_alone`). Formants follow the recording (`_offer_saved_formants`).
   - *Watch for:* unsaved formant or TextGrid edits must still go through
     `_check_unsaved_changes`; keep the existing behaviour one click away for the deliberate
     mismatch.

4. Tools

1. [ ] **Forced alignment as a FormantStudio tool.** The user has a working aligner built for one
   workflow (`D:\Research\AIPBC\LabelledData\aligner\`, launched by `Textgrid Tool.bat` there).
   - *Goal:* make it general — specify tiers by name and type, enter the text for each tier, and
     align a batch of files in which every speaker repeats the same phrase or list in the same
     order.
   - *Related, already in FormantStudio:* `phone_aligner.py` (phone-class aligner, 12.5 ms
     median boundary error) and template alignment (`_align_template`, pause-anchored, measured
     11× better than linear stretching). Survey the external aligner against these before
     designing, so the tool reuses rather than duplicates them.
