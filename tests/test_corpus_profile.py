"""
Tests for corpus profiles: the per-corpus settings file kept beside the
TextGrids.

A profile that silently fell back to defaults would quietly change what is
analysed, so a broken file must be reported, while a missing one is simply
a corpus nobody has configured yet.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import corpus_profile as cp  # noqa: E402
from corpus_profile import CorpusProfile, ProfileError  # noqa: E402


def _custom():
    return CorpusProfile(
        non_speech_labels=("-", "<p:>", "<speech noise>"),
        audio_dir="D:/corpus/audio",
    ).with_label_check(
        label_rules=(cp.LabelRule("phones", "sampa", ("p", "t", "a")),
                     cp.LabelRule("targets", cp.ONLY_LISTED,
                                  ("T", "T1", "T2"))),
        tolerance_ms=12.5, boundaries=False)


class TestRoundTrip:

    def test_everything_survives_save_and_load(self, tmp_path):
        path = str(tmp_path / "p.json")
        _custom().save(path)
        assert CorpusProfile.load(path) == _custom()

    def test_all_tiers_and_no_tiers_stay_distinct(self, tmp_path):
        """None means every tier; an empty selection means none."""
        path = str(tmp_path / "p.json")
        CorpusProfile().with_label_check(unlabelled_tiers=()).save(path)
        lc = CorpusProfile.load(path).label_check
        assert lc.unlabelled_tiers == ()
        assert lc.boundary_tiers is None

    def test_labels_with_spaces_and_unicode_survive(self, tmp_path):
        path = str(tmp_path / "p.json")
        CorpusProfile(non_speech_labels=("<speech noise>", "\u0294?")).save(path)
        assert CorpusProfile.load(path).non_speech_labels == (
            "<speech noise>", "\u0294?")

    def test_the_file_says_what_it_is(self, tmp_path):
        path = str(tmp_path / "p.json")
        CorpusProfile().save(path)
        data = json.loads(open(path, encoding="utf-8").read())
        assert data["format"] == cp.FORMAT
        assert data["version"] == cp.VERSION


class TestFolder:

    def test_a_folder_without_a_profile_gets_the_defaults(self, tmp_path):
        assert CorpusProfile.for_folder(str(tmp_path)) == CorpusProfile()

    def test_the_default_non_speech_label_is_the_pause_dash(self):
        assert CorpusProfile().non_speech_labels == ("-",)

    def test_saved_to_a_folder_it_is_found_there(self, tmp_path):
        _custom().save_to_folder(str(tmp_path))
        assert os.path.exists(tmp_path / cp.PROFILE_NAME)
        assert CorpusProfile.for_folder(str(tmp_path)) == _custom()


class TestBrokenFiles:

    def _write(self, tmp_path, text):
        path = tmp_path / cp.PROFILE_NAME
        path.write_text(text, encoding="utf-8")
        return str(tmp_path)

    def test_malformed_json_is_reported(self, tmp_path):
        folder = self._write(tmp_path, "{not json")
        with pytest.raises(ProfileError):
            CorpusProfile.for_folder(folder)

    def test_some_other_json_is_not_a_profile(self, tmp_path):
        folder = self._write(tmp_path, '{"hello": 1}')
        with pytest.raises(ProfileError, match="not a FormantStudio"):
            CorpusProfile.for_folder(folder)

    def test_a_newer_version_is_refused(self, tmp_path):
        folder = self._write(tmp_path, json.dumps(
            {"format": cp.FORMAT, "version": cp.VERSION + 1}))
        with pytest.raises(ProfileError, match="newer"):
            CorpusProfile.for_folder(folder)

    def test_a_malformed_setting_is_reported(self, tmp_path):
        folder = self._write(tmp_path, json.dumps(
            {"format": cp.FORMAT, "version": 1, "non_speech_labels": 5}))
        with pytest.raises(ProfileError, match="malformed"):
            CorpusProfile.for_folder(folder)

    def test_missing_settings_take_their_defaults(self, tmp_path):
        folder = self._write(tmp_path, json.dumps(
            {"format": cp.FORMAT, "version": 1}))
        assert CorpusProfile.for_folder(folder) == CorpusProfile()

    def test_unknown_settings_are_ignored(self, tmp_path):
        """A profile from a version with more settings still loads."""
        folder = self._write(tmp_path, json.dumps(
            {"format": cp.FORMAT, "version": 1, "future": True,
             "label_check": {"tolerance_ms": 5, "future_rule": []}}))
        assert CorpusProfile.for_folder(folder).label_check.tolerance_ms == 5.0


class TestLabelText:

    def test_commas_separate_and_spaces_stay_inside_labels(self):
        assert cp.parse_labels("-, <p:>,<speech noise> ") == (
            "-", "<p:>", "<speech noise>")

    def test_blanks_and_repeats_are_dropped(self):
        assert cp.parse_labels(" , -, -,, ") == ("-",)

    def test_format_round_trips(self):
        labels = ("-", "<speech noise>")
        assert cp.parse_labels(cp.format_labels(labels)) == labels


class TestLabelRules:

    def _load(self, tmp_path, rules):
        path = tmp_path / cp.PROFILE_NAME
        path.write_text(json.dumps({"format": cp.FORMAT, "version": 1,
                                    "label_check": {"label_rules": rules}}),
                        encoding="utf-8")
        return CorpusProfile.for_folder(str(tmp_path)).label_check.label_rules

    def test_rules_load_in_order(self, tmp_path):
        rules = self._load(tmp_path, [
            {"tier": "phones", "kind": "ipa"},
            {"tier": "targets", "kind": "list", "labels": ["T", "T1"]}])
        assert rules == (cp.LabelRule("phones", "ipa"),
                         cp.LabelRule("targets", "list", ("T", "T1")))

    def test_an_unknown_kind_is_reported(self, tmp_path):
        with pytest.raises(ProfileError, match="malformed"):
            self._load(tmp_path, [{"tier": "phones", "kind": "klingon"}])

    def test_a_rule_without_a_tier_is_reported(self, tmp_path):
        with pytest.raises(ProfileError, match="malformed"):
            self._load(tmp_path, [{"kind": "ipa"}])

    def test_an_inventory_may_be_spaced_or_commaed(self):
        assert cp.parse_inventory("p t,k  m") == ("p", "t", "k", "m")

    def test_a_list_keeps_spaces_inside_labels(self):
        assert cp.parse_labels("T1, speech noise") == ("T1", "speech noise")
