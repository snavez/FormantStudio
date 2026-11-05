"""
Tests for the IPA symbol chart and the diacritic table behind it.

The chart types symbols into the label editor, which writes only the user's
own edits through to the TextGrid -- so a chart click has to arrive as an
edit, not as the programmatic refill the editor gets on every selection.
The chart must also show every symbol the CSV knows, and say the same thing
about each diacritic as the classifier that reads labels back.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication

from formant_editor import (
    _DIACRITICS,
    _DIACRITIC_SUFFIXES,
    _IPA_CHART_PATH,
    _IPA_COMBINING_DIACRITICS,
    _IPAChartDialog,
    _SUPRASEGMENTALS,
    _build_known_vowels,
    _classify_label,
    _load_ipa_chart,
    Interval,
    MainWindow,
    TextGrid,
    Tier,
)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


@pytest.fixture(scope="module")
def chart():
    return _load_ipa_chart(_IPA_CHART_PATH)


@pytest.fixture
def window(qapp):
    mw = MainWindow()
    c = mw.canvas
    c.textgrid_data = TextGrid(0, 1.0, [
        Tier("phones", "IntervalTier", 0, 1.0,
             intervals=[Interval(0, 0.5, "h"), Interval(0.5, 1.0, "a")]),
    ])
    c.total_duration = 1.0
    c.view_start, c.view_end = 0.0, 1.0
    c._active_tier = 0
    yield mw
    # Closing a dirty window asks to save, which would block the run.
    mw._textgrid_dirty = mw._formants_dirty = False
    mw.close()


@pytest.fixture
def dialog(window):
    d = _IPAChartDialog(window)
    yield d
    d.close()


def _button(dialog, ipa):
    return next(b for b in dialog._symbol_buttons if b.property("ipa") == ipa)


def _set_notation(dialog, name):
    dialog._notation_combo.setCurrentText(name)


def _labels(window):
    return [iv.text for iv in window.canvas.textgrid_data.tiers[0].intervals]


class TestClickWritesThrough:
    """The reported bug: the symbol showed in the editor, never in the tier."""

    @pytest.mark.parametrize("notation, ipa, expected", [
        ("IPA", "\u027D", "\u027D"),     # retroflex tap
        ("IPA", "s", "s"),
        ("SAMPA", "\u027D", "r`"),
        ("SAMPA", "s", "s"),
    ])
    def test_click_replaces_the_selected_label(
            self, window, dialog, notation, ipa, expected):
        window.canvas._select_interval(0, 0.25)
        window.canvas.focus_label_for_selection(0)   # selects the label
        _set_notation(dialog, notation)
        _button(dialog, ipa).click()
        assert _labels(window) == [expected, "a"]
        assert window._textgrid_dirty is True

    def test_label_survives_moving_away_and_back(self, window, dialog):
        window.canvas._select_interval(0, 0.25)
        window.canvas.focus_label_for_selection(0)
        _button(dialog, "\u027D").click()
        window.canvas._select_interval(0, 0.75)
        window.canvas._select_interval(0, 0.25)
        assert window.label_edit.text() == "\u027D"

    def test_clicks_append_at_the_cursor(self, window, dialog):
        """Two vowels in a row build a diphthong."""
        window.canvas._select_interval(0, 0.25)
        window.label_edit.setText("")
        _button(dialog, "a").click()
        _button(dialog, "\u026A").click()
        assert _labels(window)[0] == "a\u026A"

    def test_diacritic_follows_its_base(self, window, dialog):
        window.canvas._select_interval(0, 0.25)
        window.label_edit.deselect()
        window.label_edit.end(False)
        _set_notation(dialog, "SAMPA")
        _button(dialog, "\u02B0").click()
        assert _labels(window)[0] == "h_h"

    def test_without_a_selection_only_the_clipboard_changes(
            self, window, dialog):
        assert not window.label_edit.isEnabled()
        _button(dialog, "\u027D").click()
        assert QApplication.clipboard().text() == "\u027D"
        assert _labels(window) == ["h", "a"]
        assert window._textgrid_dirty is False


class TestEverySymbolHasAButton:

    def test_every_chart_symbol_is_shown(self, dialog, chart):
        """Long vowels and diphthongs are built by clicking, not listed."""
        shown = {b.property("ipa") for b in dialog._symbol_buttons}
        expected = {ipa for ipa, p in chart["ipa"].items()
                    if p["type"] == "consonant"
                    or (p["subtype"] == "monophthong"
                        and p["length"] == "short")}
        assert expected - shown == set()

    def test_every_status_has_a_section(self, chart):
        """A status the Extended tab has no section for would hide rows."""
        sections = {s for s, _ in _IPAChartDialog._EXTENDED_SECTIONS}
        statuses = {p["status"] for p in chart["ipa"].values()}
        assert statuses - sections == {""}

    @pytest.mark.parametrize("ipa", [
        "\u026A", "\u028A", "\u028F",    # ɪ ʊ ʏ, once hidden behind e o ø
        "\u025B", "\u025C",              # ɛ ɜ, once hidden behind æ ɐ
        "p\u0361f",                      # place with no grid column
        "\u02A1", "\u029C", "\u02A2", "\u027A", "\u0267",
    ])
    def test_formerly_missing_symbols(self, dialog, ipa):
        assert _button(dialog, ipa) is not None

    def test_every_mark_is_shown(self, dialog):
        shown = {b.property("ipa") for b in dialog._symbol_buttons}
        for mark in _DIACRITICS + _SUPRASEGMENTALS:
            assert mark.ipa in shown

    def test_no_symbol_is_shown_twice(self, dialog):
        ipas = [b.property("ipa") for b in dialog._symbol_buttons]
        assert len(ipas) == len(set(ipas))

    def test_notation_switch_relabels_buttons(self, dialog):
        btn = _button(dialog, "\u026A")
        _set_notation(dialog, "SAMPA")
        assert btn.text() == "I"
        _set_notation(dialog, "IPA")
        assert btn.text() == "\u026A"

    def test_every_official_symbol_has_a_sampa_form(self, chart):
        """Only symbols beyond the IPA chart may lack one."""
        sampa_ids = {id(p) for p in chart["sampa"].values()}
        missing = [ipa for ipa, p in chart["ipa"].items()
                   if id(p) not in sampa_ids and not p["status"]]
        assert missing == []

    def test_sampa_ampersand_is_visible(self, dialog):
        """A lone & is a button's accelerator marker and draws nothing."""
        btn = _button(dialog, "\u0276")
        _set_notation(dialog, "SAMPA")
        assert btn.text() == "&&"
        _set_notation(dialog, "IPA")
        assert btn.text() == "\u0276"


class TestExtendedSymbols:
    """Symbols beyond the official chart: extensions, extIPA, obsolete."""

    @pytest.mark.parametrize("ipa", [
        "\u025A", "\u025D", "\u026B",        # rhotic schwas, dark l
        "\u1D7B", "\u1D7F",                     # barred I and U
        "k\u0361p",                               # doubly articulated
        "\uA78E", "\U0001DF0A",                 # extIPA
        "\u02A7", "\u0287", "\u01A5", "g",    # obsolete, variants
    ])
    def test_shown(self, dialog, ipa):
        assert _button(dialog, ipa) is not None

    def test_extensions_stay_off_the_official_grids(self, dialog):
        official = {s[0] for s in dialog._symbols}
        assert "\u025A" not in official
        assert "\u02A7" not in official

    def test_ipa_only_symbol_is_disabled_under_sampa(self, dialog):
        btn = _button(dialog, "\u02A7")
        assert btn.isEnabled()
        _set_notation(dialog, "SAMPA")
        assert not btn.isEnabled()
        assert btn.text() == "\u02A7"
        _set_notation(dialog, "IPA")
        assert btn.isEnabled()

    def test_extended_symbol_with_sampa_types_it(self, window, dialog):
        window.canvas._select_interval(0, 0.25)
        window.canvas.focus_label_for_selection(0)
        _set_notation(dialog, "SAMPA")
        _button(dialog, "\u025A").click()
        assert _labels(window)[0] == "@`"

    def test_tooltip_explains_an_obsolete_symbol(self, dialog):
        assert "ligature" in _button(dialog, "\u02A7").toolTip()

    def test_loader_reads_rows_without_the_trailing_columns(self, tmp_path):
        csv_file = tmp_path / "chart.csv"
        csv_file.write_text(
            "ipa,sampa,xsampa,type,subtype,height,fronting,rounding,"
            "length,voicing,place,manner,status,note\n"
            "p,p,p,consonant,,,,,,voiceless,bilabial,plosive\n",
            encoding="utf-8")
        props = _load_ipa_chart(str(csv_file))["ipa"]["p"]
        assert props["status"] == "" and props["note"] == ""


class TestDiacriticTable:
    """X-SAMPA codes, and the classifier agreeing with the chart."""

    def test_sampa_codes_are_unique(self):
        codes = [d.sampa for d in _DIACRITICS + _SUPRASEGMENTALS]
        assert len(codes) == len(set(codes))

    def test_ipa_marks_are_unique(self):
        marks = [d.ipa for d in _DIACRITICS + _SUPRASEGMENTALS]
        assert len(marks) == len(set(marks))

    @pytest.mark.parametrize("description, sampa", [
        ("dental", "_d"),
        ("breathy voice", "_t"),
        ("creaky voice", "_k"),
        ("less rounded", "_c"),
        ("velarized", "_G"),
    ])
    def test_xsampa_codes(self, description, sampa):
        mark = next(d for d in _DIACRITICS if d.description == description)
        assert mark.sampa == sampa

    def test_suffixes_are_longest_first(self):
        lengths = [len(s) for s, _ in _DIACRITIC_SUFFIXES]
        assert lengths == sorted(lengths, reverse=True)

    def test_classifier_tables_come_from_the_chart(self):
        featured = [d for d in _DIACRITICS + _SUPRASEGMENTALS if d.feature]
        assert _IPA_COMBINING_DIACRITICS == [(d.ipa, d.feature)
                                             for d in featured]
        assert set(_DIACRITIC_SUFFIXES) == (
            {(d.sampa, d.feature) for d in featured} | {("_:", "long")})


class TestClassifierReadsDiacritics:

    @pytest.fixture(scope="class")
    def sampa(self):
        c = _load_ipa_chart(_IPA_CHART_PATH)["sampa"]
        return c, _build_known_vowels(c)

    @pytest.fixture(scope="class")
    def ipa(self):
        c = _load_ipa_chart(_IPA_CHART_PATH)["ipa"]
        return c, _build_known_vowels(c)

    @pytest.mark.parametrize("label, modifier", [
        ("t_d", "dental_diac"),
        ("a_t", "breathy"),
        ("a_k", "creaky"),
        ("t_h", "aspirated"),
        ("l_G", "velarized"),
        ("l_e", "velarized"),
        ("a_:", "long_diac"),
    ])
    def test_sampa_suffix(self, sampa, label, modifier):
        _, mods, ok = _classify_label(label, sampa[0], "sampa", sampa[1])
        assert ok
        assert mods == [modifier]

    @pytest.mark.parametrize("label, modifier", [
        ("t\u032A", "dental_diac"),
        ("a\u0324", "breathy"),
        ("a\u0330", "creaky"),
        ("l\u02E0", "velarized"),
    ])
    def test_ipa_mark(self, ipa, label, modifier):
        _, mods, ok = _classify_label(label, ipa[0], "ipa", ipa[1])
        assert ok
        assert mods == [modifier]

    def test_chart_only_mark_is_not_classified(self, sampa):
        """Less rounded is offered for labelling but has no CSV column."""
        _, _, ok = _classify_label("o_c", sampa[0], "sampa", sampa[1])
        assert not ok

    def test_ligature_classifies_as_its_affricate(self, ipa):
        props, _, ok = _classify_label("\u02A7", ipa[0], "ipa", ipa[1])
        assert ok
        assert (props["place"], props["manner"]) == ("postalveolar",
                                                     "affricate")

    def test_ipa_only_symbol_is_unknown_in_sampa(self, sampa):
        _, _, ok = _classify_label("\u02A7", sampa[0], "sampa", sampa[1])
        assert not ok

    def test_rhotic_schwa_in_sampa(self, sampa):
        props, _, ok = _classify_label("@`", sampa[0], "sampa", sampa[1])
        assert ok
        assert props["type"] == "vowel"

    def test_new_consonants_classify(self, sampa):
        props, _, ok = _classify_label("H\\", sampa[0], "sampa", sampa[1])
        assert ok
        assert props["place"] == "epiglottal"


class TestShortcutReference:

    def _view_texts(self, window):
        menu = next(a.menu() for a in window.menuBar().actions()
                    if a.text() == "&View")
        return [a.text() for a in menu.actions()]

    def test_every_reference_entry_is_in_the_view_menu(self, window):
        texts = self._view_texts(window)
        for group in MainWindow._SHORTCUT_REFERENCE:
            for text, shortcut in group:
                assert f"{text}\t{shortcut}" in texts

    def test_interval_navigation_is_listed(self, window):
        assert any("Ctrl+\u2192" in t for t in self._view_texts(window))
