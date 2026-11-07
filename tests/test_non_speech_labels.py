"""
Tests for a corpus's own non-speech labels reaching every reader of them.

Intervals labelled with any of a corpus's non-speech labels have nothing to
analyse: Build CSV leaves them out, the divergence reading treats them as
EMPTY, and the subdivision scan the wizard runs ignores them.  A label left
off the list is ordinary data -- the escape hatch for measuring pauses.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import divergence  # noqa: E402
from formant_editor import (  # noqa: E402
    Interval, TextGrid, Tier, _build_csv_data, _tiers_subdividing_primary,
)

DUR = 1.0
NOISY = ("-", "<noise>", "<p:>")
LABELS = ["a", "<noise>", "t", "-", "<p:>", "i"]


@pytest.fixture
def corpus(tmp_path):
    import parselmouth
    sr = 16000
    samples = np.random.default_rng(0).standard_normal(int(sr * DUR)) * 0.05
    parselmouth.Sound(samples, sampling_frequency=sr).save(
        str(tmp_path / "rec.wav"), "WAV")
    edges = np.linspace(0.0, DUR, len(LABELS) + 1)
    phones = Tier("phones", "IntervalTier", 0.0, DUR, intervals=[
        Interval(float(a), float(b), lab)
        for a, b, lab in zip(edges[:-1], edges[1:], LABELS)])
    TextGrid(0.0, DUR, [phones]).save(str(tmp_path / "rec.TextGrid"))
    return str(tmp_path)


def _rows(corpus, **extra):
    headers, rows = _build_csv_data(
        audio_dir=corpus, textgrid_dir=corpus, formants_dir=None,
        selected_tiers=[Tier("phones", "IntervalTier", 0.0, DUR)],
        extract_formants=False, formant_mode=None,
        point_tier_name=None, segment_tier_name="phones",
        percentage_markers=[], extract_durations=True,
        duration_tier_names=["phones"], **extra)
    col = headers.index("phones")
    return [r[col] for r in rows]


class TestBuildCsv:

    def test_by_default_only_the_dash_is_left_out(self, corpus):
        assert _rows(corpus) == ["a", "<noise>", "t", "<p:>", "i"]

    def test_every_non_speech_label_is_left_out(self, corpus):
        assert _rows(corpus, non_speech_labels=NOISY) == ["a", "t", "i"]

    def test_a_pause_label_kept_off_the_list_is_measured(self, corpus):
        """The prosody case: <p:> stays data when it is not on the list."""
        rows = _rows(corpus, non_speech_labels=("-", "<noise>"))
        assert rows == ["a", "t", "<p:>", "i"]


class TestDivergence:

    def _tiers(self, pairs):
        ph = [(i * 0.1, (i + 1) * 0.1, p) for i, (p, _) in enumerate(pairs)]
        al = [(i * 0.1, (i + 1) * 0.1, a) for i, (_, a) in enumerate(pairs)]
        return ph, al

    def test_any_non_speech_label_against_another_is_not_analysed(self):
        kinds = [d.kind for d in divergence.resolve(
            *self._tiers([("<p:>", "<noise>")]), empty=NOISY)]
        assert kinds == [divergence.NOT_ANALYSED]

    def test_a_sound_realised_as_noise_is_a_deletion(self):
        kinds = [d.kind for d in divergence.resolve(
            *self._tiers([("a", "<noise>")]), empty=NOISY)]
        assert kinds == [divergence.DELETION]

    def test_by_default_noise_is_an_ordinary_label(self):
        kinds = [d.kind for d in divergence.resolve(
            *self._tiers([("a", "<noise>")]))]
        assert kinds == [divergence.SUBSTITUTION]

    def test_an_arrow_cannot_reach_through_non_speech(self):
        problems = divergence.validate(
            *self._tiers([("a", ">"), ("<p:>", "<p:>"), ("t", "t")]),
            empty=NOISY)
        assert any("cannot reach a sound" in p for p in problems)


class TestSubdivisionScan:

    def test_non_speech_subdivisions_are_not_labels(self, tmp_path):
        """A finer tier's pause pieces do not become CSV columns."""
        phones = Tier("phones", "IntervalTier", 0.0, DUR,
                      intervals=[Interval(0.0, DUR, "t")])
        parts = Tier("parts", "IntervalTier", 0.0, DUR, intervals=[
            Interval(0.0, 0.3, "closure"), Interval(0.3, 0.5, "<noise>"),
            Interval(0.5, DUR, "release")])
        TextGrid(0.0, DUR, [phones, parts]).save(str(tmp_path / "x.TextGrid"))
        found = _tiers_subdividing_primary(
            str(tmp_path), "phones", ["parts"], non_speech=NOISY)
        assert found == [("parts", ["closure", "release"])]


class TestWizardPathsPage:
    """The wizard's first page shows the corpus's list and saves edits."""

    @pytest.fixture
    def page(self, qapp, corpus):
        from formant_editor import BuildCSVWizard
        wiz = BuildCSVWizard()
        page = wiz.page(wiz.startId())
        page._audio_edit.setText(corpus)
        yield page
        wiz.deleteLater()

    @pytest.fixture(scope="session")
    def qapp(self):
        from PyQt6.QtWidgets import QApplication
        yield QApplication.instance() or QApplication(sys.argv)

    def test_the_folder_profile_fills_the_field(self, qapp, corpus):
        from corpus_profile import CorpusProfile
        from formant_editor import BuildCSVWizard
        CorpusProfile(non_speech_labels=NOISY).save_to_folder(corpus)
        wiz = BuildCSVWizard()
        page = wiz.page(wiz.startId())
        page._audio_edit.setText(corpus)
        page._load_profile()
        assert page._non_speech_edit.text() == "-, <noise>, <p:>"

    def test_the_labels_reach_the_wizard(self, page):
        page._non_speech_edit.setText("-, <noise>")
        assert page.validatePage()
        assert page.wizard().non_speech_labels == ("-", "<noise>")

    def test_an_edit_is_saved_when_asked(self, page, corpus):
        from corpus_profile import CorpusProfile
        page._non_speech_edit.setText("<sil>")
        page._save_ns_cb.setChecked(True)
        page.validatePage()
        assert CorpusProfile.for_folder(corpus).non_speech_labels == ("<sil>",)

    def test_saving_keeps_label_checks_settings(self, page, corpus):
        from corpus_profile import CorpusProfile
        CorpusProfile().with_label_check(tolerance_ms=3.0).save_to_folder(
            corpus)
        page._load_profile()
        page._non_speech_edit.setText("<sil>")
        page._save_ns_cb.setChecked(True)
        page.validatePage()
        saved = CorpusProfile.for_folder(corpus)
        assert saved.label_check.tolerance_ms == 3.0
        assert saved.non_speech_labels == ("<sil>",)

    def test_an_untouched_default_writes_nothing(self, page, corpus):
        from corpus_profile import profile_path
        page._load_profile()
        page.validatePage()
        assert not os.path.exists(profile_path(corpus))

    def test_unasked_an_edit_applies_to_this_build_only(self, page, corpus):
        from corpus_profile import profile_path
        page._non_speech_edit.setText("<sil>")
        assert page.validatePage()
        assert page.wizard().non_speech_labels == ("<sil>",)
        assert not os.path.exists(profile_path(corpus))
