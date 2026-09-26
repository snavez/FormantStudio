"""
Tests for the two guards that stand between a wizard run and an unusable CSV:
the warning raised when the spectral segment tier is the primary tier while a
finer tier sits unused, and the provenance sidecar that records what a run
actually did.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import formant_editor as fe
from formant_editor import (
    Interval,
    Point,
    TextGrid,
    Tier,
    _cat_label_columns,
    _reclassify_rejected_diphthongs,
    _tiers_subdividing_primary,
)


DUR = 0.5


@pytest.fixture(scope="session")
def qapp():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


def _grid(with_allophone=True):
    """'tata' > phones 't','a'; the allophone tier splits 't' into t0/tH."""
    words = Tier("words", "IntervalTier", 0.0, DUR,
                 intervals=[Interval(0.05, 0.45, "tata")])
    phones = Tier("phones", "IntervalTier", 0.0, DUR,
                  intervals=[Interval(0.05, 0.2, "t"),
                             Interval(0.2, 0.45, "a")])
    tiers = [words, phones]
    if with_allophone:
        tiers.append(Tier("allophone", "IntervalTier", 0.0, DUR,
                          intervals=[Interval(0.0, 0.05, ""),
                                     Interval(0.05, 0.12, "t0"),
                                     Interval(0.12, 0.2, "tH"),
                                     Interval(0.2, DUR, "")]))
    return TextGrid(0.0, DUR, tiers)


@pytest.fixture
def tg_dir(tmp_path):
    d = tmp_path / "tg"
    d.mkdir()
    _grid().save(str(d / "test.TextGrid"))
    return str(d)


class TestSubdividingTierDetection:
    """What the wizard consults before it lets a whole-interval run through."""

    def test_a_finer_tier_is_reported_with_its_labels(self, tg_dir):
        found = _tiers_subdividing_primary(
            tg_dir, "phones", ["words", "phones", "allophone"])
        assert found == [("allophone", ["t0", "tH"])]

    def test_coarser_and_matching_tiers_are_not_reported(self, tg_dir):
        """One 'words' interval holds both phones, so it does not subdivide."""
        assert _tiers_subdividing_primary(tg_dir, "phones", ["words"]) == []
        assert _tiers_subdividing_primary(tg_dir, "phones", ["phones"]) == []

    def test_nothing_to_subdivide_by_is_no_answer(self, tmp_path):
        d = tmp_path / "plain"
        d.mkdir()
        _grid(with_allophone=False).save(str(d / "test.TextGrid"))
        assert _tiers_subdividing_primary(
            str(d), "phones", ["words", "phones"]) == []

    def test_a_primary_tier_the_grids_do_not_carry_is_no_answer(
            self, tg_dir):
        assert _tiers_subdividing_primary(
            tg_dir, "no_such_tier", ["allophone"]) == []

    def test_a_point_primary_tier_has_no_intervals_to_split(
            self, tmp_path):
        """Rows anchored on points have no span for a tier to cut."""
        d = tmp_path / "points"
        d.mkdir()
        g = _grid()
        g.tiers.append(Tier("Target", "TextTier", 0.0, DUR,
                            points=[Point(0.10, "rel"),
                                    Point(0.25, "T1")]))
        g.save(str(d / "test.TextGrid"))
        assert _tiers_subdividing_primary(
            str(d), "Target", ["phones", "allophone"]) == []

    def test_an_unreadable_directory_is_no_answer(self, tmp_path):
        assert _tiers_subdividing_primary(
            str(tmp_path / "gone"), "phones", ["allophone"]) == []

    def test_only_the_first_few_grids_are_read(self, tmp_path):
        """Validation samples the corpus; it does not walk all of it."""
        d = tmp_path / "many"
        d.mkdir()
        for i in range(10):
            _grid(with_allophone=(i == 8)).save(str(d / f"f{i:02d}.TextGrid"))
        assert _tiers_subdividing_primary(
            str(d), "phones", ["allophone"]) == []
        assert _tiers_subdividing_primary(
            str(d), "phones", ["allophone"], max_files=10) == [
                ("allophone", ["t0", "tH"])]


def _data_page(wiz):
    for pid in wiz.pageIds():
        page = wiz.page(pid)
        if type(page).__name__ == "_DataOptionsPage":
            return page
    raise AssertionError("data options page not found")


def _armed_wizard(tg_dir, spectral_tier, with_allophone=True):
    """A wizard on the spectral page, set up the way a real run arrives."""
    wiz = fe.BuildCSVWizard()
    wiz.textgrid_dir = tg_dir
    wiz.primary_tier_name = "phones"
    wiz.selected_tier_info = [("words", "IntervalTier"),
                              ("phones", "IntervalTier")]
    if with_allophone:
        wiz.selected_tier_info.append(("allophone", "IntervalTier"))
    page = _data_page(wiz)
    page.initializePage()
    page._fmt_cb.setChecked(False)
    page._dur_cb.setChecked(True)
    page._spectral_data_cb.setChecked(True)
    page._spectral_cb.setChecked(True)
    page._traj_cb.setChecked(False)
    idx = page._spectral_tier_combo.findText(spectral_tier)
    assert idx >= 0
    page._spectral_tier_combo.setCurrentIndex(idx)
    return wiz, page


class TestPreEmphasisControls:
    """Off unless asked for, and collected as a corner in Hz."""

    def test_off_by_default(self, qapp, tg_dir):
        wiz, page = _armed_wizard(tg_dir, "allophone")
        assert not page._spectral_pe_cb.isChecked()
        assert not page._spectral_pe_spin.isEnabled()
        assert page.validatePage() is True
        assert wiz.spectral_preemph_hz == 0.0

    def test_ticked_collects_the_corner(self, qapp, tg_dir):
        wiz, page = _armed_wizard(tg_dir, "allophone")
        page._spectral_pe_cb.setChecked(True)
        page._spectral_pe_spin.setValue(141.8)
        assert page._spectral_pe_spin.isEnabled()
        assert page.validatePage() is True
        assert wiz.spectral_preemph_hz == pytest.approx(141.8)

    def test_shows_the_equivalent_alpha(self, qapp, tg_dir):
        _, page = _armed_wizard(tg_dir, "allophone")
        page._spectral_pe_spin.setValue(141.8)
        assert "0.9800" in page._spectral_pe_alpha.text()


class TestTrajectoryMeasureChoices:
    def test_every_tracked_measure_can_be_chosen(self, qapp, tg_dir):
        wiz, page = _armed_wizard(tg_dir, "allophone")
        assert [n for _, n in page._traj_moment_cbs] == list(
            fe.SPECTRAL_TRACK_NAMES)
        page._traj_cb.setChecked(True)
        for cb, name in page._traj_moment_cbs:
            cb.setChecked(name == "flatness")
        assert page.validatePage() is True
        assert wiz.traj_moments == ["flatness"]


class TestWholeIntervalSpectraWarning:
    """The guard fires only when a finer tier would really have split things."""

    def _spy(self, page, monkeypatch, answer):
        seen = []

        def fake(primary, finer):
            seen.append((primary, finer))
            return answer

        monkeypatch.setattr(page, "_confirm_whole_interval_spectra", fake)
        return seen

    def test_warns_when_the_spectral_tier_is_the_primary_tier(
            self, qapp, tg_dir, monkeypatch):
        wiz, page = _armed_wizard(tg_dir, "phones")
        seen = self._spy(page, monkeypatch, False)
        assert page.validatePage() is False       # "Go back" holds the page
        assert seen == [("phones", [("allophone", ["t0", "tH"])])]

    def test_continue_lets_the_run_through(self, qapp, tg_dir, monkeypatch):
        wiz, page = _armed_wizard(tg_dir, "phones")
        seen = self._spy(page, monkeypatch, True)
        assert page.validatePage() is True
        assert len(seen) == 1
        assert wiz.spectral_tier_name == "phones"

    def test_silent_when_the_finer_tier_is_the_one_chosen(
            self, qapp, tg_dir, monkeypatch):
        wiz, page = _armed_wizard(tg_dir, "allophone")
        seen = self._spy(page, monkeypatch, False)
        assert page.validatePage() is True
        assert seen == []

    def test_silent_when_no_tier_subdivides_the_primary_one(
            self, qapp, tmp_path, monkeypatch):
        d = tmp_path / "plain"
        d.mkdir()
        _grid(with_allophone=False).save(str(d / "test.TextGrid"))
        wiz, page = _armed_wizard(str(d), "phones", with_allophone=False)
        seen = self._spy(page, monkeypatch, False)
        assert page.validatePage() is True
        assert seen == []

    def test_the_warning_names_the_tier_and_its_labels(self, qapp, tg_dir,
                                                       monkeypatch):
        """Dismissing the box without a choice holds the page."""
        shown = {}

        def fake_exec(box):
            shown["text"] = box.text()
            shown["detail"] = box.informativeText()
            shown["buttons"] = [b.text() for b in box.buttons()]
            return 0

        monkeypatch.setattr(fe.QMessageBox, "exec", fake_exec)
        _wiz, page = _armed_wizard(tg_dir, "phones")
        assert page._confirm_whole_interval_spectra(
            "phones", [("allophone", ["t0", "tH"])]) is False
        assert "phones" in shown["text"]
        assert "allophone" in shown["detail"]
        assert "t0, tH" in shown["detail"]
        assert "COG_t0_20%" in shown["detail"]
        assert shown["buttons"] == ["Continue", "Go back"]

    def test_silent_when_no_spectral_analysis_is_asked_for(
            self, qapp, tg_dir, monkeypatch):
        wiz, page = _armed_wizard(tg_dir, "phones")
        page._spectral_data_cb.setChecked(False)
        page._dur_cb.setChecked(True)
        seen = self._spy(page, monkeypatch, False)
        assert page.validatePage() is True
        assert seen == []


def _full_run_wizard(qapp):
    """A wizard configured like a real full-featured export."""
    wiz = fe.BuildCSVWizard()
    wiz.audio_dir = "."
    wiz.primary_tier_name = "phones"
    wiz.selected_tier_info = [("words", "IntervalTier"),
                              ("phones", "IntervalTier"),
                              ("allophone", "IntervalTier")]
    wiz.extract_formants = True
    wiz.formant_mode = "for_segments"
    wiz.segment_tier_name = "phones"
    wiz.sampling_mode = "pct_list"
    wiz.percentage_markers = [20.0, 50.0, 80.0]
    wiz.extract_durations = True
    wiz.duration_tier_names = ["phones", "allophone"]
    wiz.include_next_segment = True
    wiz.context_tier_name = "phones"
    wiz.extract_spectral = True
    wiz.extract_trajectory = True
    wiz.spectral_tier_name = "allophone"
    wiz.spectral_markers = [20.0, 50.0, 80.0]
    wiz.spectral_estimator = "multitaper"
    wiz.band_low_hz = 300.0
    wiz.band_high_hz = 10000.0
    wiz.ratio_low_band = (500.0, 1500.0)
    wiz.ratio_high_band = (4000.0, 5500.0)
    wiz.categorise = True
    wiz.cat_tier_names = ["phones"]
    wiz.cat_next_segment = True
    return wiz


class TestProvenanceSidecar:
    """Every export leaves a record of the settings that produced it."""

    def test_a_full_run_writes_the_sidecar_beside_the_csv(self, qapp,
                                                          tmp_path):
        wiz = _full_run_wizard(qapp)
        csv_path = tmp_path / "acoustic_data.csv"
        csv_path.write_text("a,b\n1,2\n", encoding="utf-8")
        side = fe.MainWindow._write_provenance_sidecar(
            None, str(csv_path), wiz, 9915)
        assert side == str(tmp_path / "acoustic_data.provenance.json")
        rec = json.loads(open(side, encoding="utf-8").read())

        assert rec["tool"] == "FormantStudio"
        assert rec["csv_file"] == "acoustic_data.csv"
        assert rec["n_rows"] == 9915
        assert rec["primary_tier"] == "phones"
        assert rec["formants"]["segment_tier"] == "phones"
        assert rec["durations"]["tiers"] == ["phones", "allophone"]
        assert rec["categorisation"]["tiers"] == ["phones"]
        assert rec["trajectory"]["moments"]

        spec = rec["spectral"]
        assert spec["segment_tier"] == "allophone"
        assert spec["markers_pct"] == [20.0, 50.0, 80.0]
        assert spec["estimator"] == "multitaper"
        assert spec["analysis_band_hz"] == [300.0, 10000.0]
        assert spec["band_ratio_low_hz"] == [500.0, 1500.0]
        assert spec["band_ratio_high_hz"] == [4000.0, 5500.0]
        assert "10*log10" in spec["band_ratio_units"]

    def test_the_record_says_how_the_signal_was_filtered(self, qapp,
                                                          tmp_path):
        wiz = _full_run_wizard(qapp)
        wiz.spectral_highpass_hz = 300.0
        wiz.spectral_preemph_hz = 141.8
        csv_path = tmp_path / "pe.csv"
        csv_path.write_text("a\n1\n", encoding="utf-8")
        rec = json.loads(open(fe.MainWindow._write_provenance_sidecar(
            None, str(csv_path), wiz, 1), encoding="utf-8").read())
        sig = rec["signal"]
        assert sig["highpass_hz"] == 300.0
        assert sig["pre_emphasis_from_hz"] == 141.8
        assert sig["pre_emphasis_alpha_column"] == "pre_emphasis_alpha"
        assert "highpass_hz" not in rec["spectral"]

    def test_the_record_defines_the_noisiness_columns(self, qapp, tmp_path):
        wiz = _full_run_wizard(qapp)
        csv_path = tmp_path / "n.csv"
        csv_path.write_text("a\n1\n", encoding="utf-8")
        rec = json.loads(open(fe.MainWindow._write_provenance_sidecar(
            None, str(csv_path), wiz, 1), encoding="utf-8").read())
        assert rec["file_signal"]["columns"] == list(fe.FILE_SIGNAL_COLUMNS)
        assert "clipped_pct" in rec["file_signal"]
        for key in ("flatness", "hblevel", "hbrel"):
            assert rec["spectral"][key]

    def test_the_record_says_which_tier_the_spectra_came_from(self, qapp,
                                                              tmp_path):
        """The one field that tells a whole-interval run from a split one."""
        wiz = _full_run_wizard(qapp)
        wiz.spectral_tier_name = wiz.primary_tier_name
        csv_path = tmp_path / "whole.csv"
        csv_path.write_text("a\n1\n", encoding="utf-8")
        side = fe.MainWindow._write_provenance_sidecar(
            None, str(csv_path), wiz, 10)
        rec = json.loads(open(side, encoding="utf-8").read())
        assert rec["spectral"]["segment_tier"] == rec["primary_tier"]

    def test_a_failed_write_is_reported_and_never_fatal(self, qapp, tmp_path,
                                                        monkeypatch):
        """A sidecar that cannot be written says so; the export still stands."""
        warned = []
        monkeypatch.setattr(
            fe.QMessageBox, "warning",
            lambda *a, **k: warned.append(a[1:3]))
        wiz = _full_run_wizard(qapp)
        missing = tmp_path / "no_such_dir" / "acoustic_data.csv"
        assert fe.MainWindow._write_provenance_sidecar(
            None, str(missing), wiz, 10) is None
        assert warned and "Sidecar" in warned[0][0]


def _categorisation_page(wiz):
    for pid in wiz.pageIds():
        page = wiz.page(pid)
        if type(page).__name__ == "_CategorisationPage":
            return page
    raise AssertionError("categorisation page not found")


def _categorisation_wizard(include_prev=False, include_next=False,
                           allophone_tier="MAU"):
    """Wizard state produced by the preceding CSV-builder pages."""
    wiz = fe.BuildCSVWizard()
    wiz.primary_tier_name = "ORT_MAU"
    wiz.reconcile_allophone_tier = allophone_tier
    wiz.context_tier_name = "MAU"
    wiz.selected_tier_info = [
        ("KAN-MAU", "IntervalTier"),
        ("ORT_MAU", "IntervalTier"),
        ("SYLLABLE", "IntervalTier"),
        ("MAU", "IntervalTier"),
        ("TARGETS", "TextTier"),
    ]
    wiz.include_prev_segment = include_prev
    wiz.include_next_segment = include_next
    page = _categorisation_page(wiz)
    page.initializePage()
    return wiz, page


class TestCategorisationPageChoices:
    """The page names and limits exactly what the classifier will inspect."""

    @staticmethod
    def _checked(page):
        return {cb.property("tier_name") for cb in page._tier_cbs
                if cb.isChecked()}

    def test_produced_allophone_tier_is_the_only_default(self, qapp):
        _wiz, page = _categorisation_wizard()
        assert self._checked(page) == {"MAU"}
        assert page._tier_summary.text() == "Categorising tier: MAU"
        # Other included tiers remain available as explicit opt-ins.
        assert {cb.property("tier_name") for cb in page._tier_cbs} == {
            "KAN-MAU", "ORT_MAU", "SYLLABLE", "MAU", "TARGETS"}

    def test_primary_tier_is_the_fallback_default(self, qapp):
        _wiz, page = _categorisation_wizard(allophone_tier=None)
        assert self._checked(page) == {"ORT_MAU"}
        assert "selected primary tier" in page._tier_note.text()

    def test_manual_tier_choices_survive_revisiting_the_page(self, qapp):
        _wiz, page = _categorisation_wizard()
        next(cb for cb in page._tier_cbs
             if cb.property("tier_name") == "SYLLABLE").setChecked(True)
        page.initializePage()
        assert self._checked(page) == {"MAU", "SYLLABLE"}
        assert page._tier_summary.text() == (
            "Categorising tiers: SYLLABLE, MAU")

    def test_following_only_does_not_offer_preceding(self, qapp):
        _wiz, page = _categorisation_wizard(include_next=True)
        assert not page._flank_group.isHidden()
        assert page._cat_prev_cb.isHidden()
        assert not page._cat_next_cb.isHidden()
        assert "MAU_next" in page._cat_next_cb.text()
        assert "MAU" in page._flank_group.title()

    def test_preceding_only_does_not_offer_following(self, qapp):
        _wiz, page = _categorisation_wizard(include_prev=True)
        assert not page._flank_group.isHidden()
        assert not page._cat_prev_cb.isHidden()
        assert page._cat_next_cb.isHidden()
        assert "MAU_prev" in page._cat_prev_cb.text()

    def test_no_context_columns_hides_the_whole_section(self, qapp):
        _wiz, page = _categorisation_wizard()
        assert page._flank_group.isHidden()


# A row layout of the shape the export builds: the filename, the label
# columns, the flanking context labels, then a block of categorisation
# columns per label column that was classified.
_HEADERS = [
    "filename", "MAU", "TARGETS", "MAU_prev", "MAU_next",
    "MAU_type", "MAU_V_height", "MAU_V_subtype", "MAU_C_place",
    "MAU_aspirated",
    "MAU_prev_type", "MAU_prev_V_height", "MAU_prev_V_subtype",
    "MAU_prev_C_place",
    "MAU_next_type", "MAU_next_V_height", "MAU_next_V_subtype",
    "MAU_next_C_place",
]


def _row(main="ai", prev="ai", nxt="t"):
    """One row whose three label columns are all classified as diphthongs."""
    return [
        "rec01", main, "", prev, nxt,
        "diphthong", "low", "closing", "", "yes",
        "diphthong", "low", "closing", "",
        "diphthong", "low", "closing", "",
    ]


class TestCatLabelColumns:
    """Where each block of categorisation columns reads its label from."""

    def test_a_categorised_tier_points_at_its_own_label_column(self):
        assert _cat_label_columns(_HEADERS, ["MAU"], "MAU", False,
                                  False) == [("MAU", 1)]

    def test_flanking_prefixes_are_their_own_label_columns(self):
        assert _cat_label_columns(_HEADERS, ["MAU"], "MAU", True, True) == [
            ("MAU", 1), ("MAU_prev", 3), ("MAU_next", 4)]

    def test_each_side_is_added_on_its_own(self):
        assert _cat_label_columns(_HEADERS, [], "MAU", False, True) == [
            ("MAU_next", 4)]
        assert _cat_label_columns(_HEADERS, [], "MAU", True, False) == [
            ("MAU_prev", 3)]

    def test_a_side_with_no_label_column_is_left_out(self):
        """Asking to categorise a flank whose label column was not included."""
        headers = [h for h in _HEADERS if h != "MAU_prev"]
        assert _cat_label_columns(headers, ["MAU"], "MAU", True, True) == [
            ("MAU", 1), ("MAU_next", 3)]

    def test_a_tier_that_produced_no_label_column_is_left_out(self):
        assert _cat_label_columns(_HEADERS, ["MAU", "ORT_MAU"], "MAU",
                                  False, False) == [("MAU", 1)]

    def test_the_context_tier_names_the_flanking_columns(self):
        """Context read from another tier: that tier's columns are the ones."""
        headers = _HEADERS + ["allophone_next"]
        assert _cat_label_columns(headers, [], "allophone", False, True) == [
            ("allophone_next", len(headers) - 1)]


class TestRejectedDiphthongs:
    """A candidate the user rejected is a vowel combination, everywhere."""

    def _cols(self, prev=True, nxt=True):
        return _cat_label_columns(_HEADERS, ["MAU"], "MAU", prev, nxt)

    def test_the_main_block_is_rewritten(self):
        rows = [_row()]
        _reclassify_rejected_diphthongs(_HEADERS, rows, self._cols(), {"ai"})
        row = rows[0]
        assert row[_HEADERS.index("MAU_type")] == "vowel"
        assert row[_HEADERS.index("MAU_V_subtype")] == "vowel-combination"
        assert row[_HEADERS.index("MAU_V_height")] == ""

    def test_diacritic_columns_still_describe_the_label(self):
        """Rejecting the diphthong reading does not unsay the diacritics."""
        rows = [_row()]
        _reclassify_rejected_diphthongs(_HEADERS, rows, self._cols(), {"ai"})
        assert rows[0][_HEADERS.index("MAU_aspirated")] == "yes"

    def test_a_rejected_label_reads_the_same_way_in_every_block(self):
        rows = [_row()]
        _reclassify_rejected_diphthongs(_HEADERS, rows, self._cols(), {"ai"})
        row = rows[0]
        for pfx in ("MAU", "MAU_prev"):
            assert row[_HEADERS.index(f"{pfx}_type")] == "vowel"
            assert row[_HEADERS.index(f"{pfx}_V_subtype")] == (
                "vowel-combination")

    def test_a_block_whose_label_was_not_rejected_is_untouched(self):
        """'t' fills the next column and was never a candidate."""
        rows = [_row()]
        _reclassify_rejected_diphthongs(_HEADERS, rows, self._cols(), {"ai"})
        row = rows[0]
        assert row[_HEADERS.index("MAU_next_type")] == "diphthong"
        assert row[_HEADERS.index("MAU_next_V_height")] == "low"

    def test_nothing_rejected_changes_nothing(self):
        rows = [_row()]
        _reclassify_rejected_diphthongs(_HEADERS, rows, self._cols(), {"au"})
        assert rows[0] == _row()

    def test_a_flank_not_asked_for_is_left_alone(self):
        """Its columns are only rewritten when its block was built."""
        rows = [_row()]
        _reclassify_rejected_diphthongs(
            _HEADERS, rows, self._cols(prev=False), {"ai"})
        row = rows[0]
        assert row[_HEADERS.index("MAU_type")] == "vowel"
        assert row[_HEADERS.index("MAU_prev_type")] == "diphthong"

    def test_a_subdivided_cell_is_judged_on_its_first_label(self):
        """A column joining sub-segment labels: the row token is the first."""
        rows = [_row(main="ai; u")]
        _reclassify_rejected_diphthongs(_HEADERS, rows, self._cols(), {"ai"})
        assert rows[0][_HEADERS.index("MAU_type")] == "vowel"

    def test_an_empty_label_matches_nothing(self):
        rows = [_row(main="")]
        _reclassify_rejected_diphthongs(_HEADERS, rows, self._cols(), {"ai"})
        assert rows[0][_HEADERS.index("MAU_type")] == "diphthong"

    def test_a_short_row_is_survived(self):
        """A deletion row is padded from the tier columns and stops early."""
        rows = [_row()[:7]]
        _reclassify_rejected_diphthongs(_HEADERS, rows, self._cols(), {"ai"})
        assert rows[0][_HEADERS.index("MAU_type")] == "vowel"
        assert len(rows[0]) == 7
