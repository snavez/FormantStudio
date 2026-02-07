"""
Tests for FormantData class — spec section 6.3.

Covers construction, editing, reset, and save/load round-trip.
"""

import json
import numpy as np
import numpy.testing as npt
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from formant_editor import FormantData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_formant_data(n_frames=10, n_formants=5):
    """Create a FormantData with known values for testing."""
    times = np.linspace(0.0, 0.1, n_frames)
    values = np.full((n_formants, n_frames), np.nan)
    # Fill F1–F3 with plausible frequencies
    for f in range(min(3, n_formants)):
        values[f] = np.linspace(300 * (f + 1), 350 * (f + 1), n_frames)
    return FormantData(times=times, values=values, n_formants=n_formants,
                       time_step=times[1] - times[0] if n_frames >= 2 else 0.01)


# ===================================================================
# 6.3.1 Construction & Properties
# ===================================================================

class TestConstruction:

    def test_empty_construction(self):
        fd = FormantData()
        assert fd.n_frames == 0
        assert fd.times.shape == (0,)
        assert fd.values.shape == (5, 0)
        assert fd.n_formants == 5

    def test_construction_with_data(self):
        fd = _make_formant_data(n_frames=20, n_formants=5)
        assert fd.n_frames == 20
        assert fd.values.shape == (5, 20)
        assert fd.times.shape == (20,)

    def test_original_values_are_copy(self):
        fd = _make_formant_data()
        fd.values[0, 0] = 9999.0
        assert fd.original_values[0, 0] != 9999.0

    def test_n_frames_property(self):
        fd = _make_formant_data(n_frames=15)
        assert fd.n_frames == len(fd.times) == 15

    def test_nan_initialization(self):
        fd = _make_formant_data(n_frames=10, n_formants=5)
        # F4 and F5 (indices 3,4) should be all NaN
        assert np.all(np.isnan(fd.values[3]))
        assert np.all(np.isnan(fd.values[4]))


# ===================================================================
# 6.3.2 Editing
# ===================================================================

class TestEditing:

    def test_set_value_updates_values(self):
        fd = _make_formant_data()
        fd.set_value(0, 5, 555.0)
        assert fd.values[0, 5] == 555.0

    def test_set_value_sets_edited_mask(self):
        fd = _make_formant_data()
        assert not fd.edited_mask[0, 5]
        fd.set_value(0, 5, 555.0)
        assert fd.edited_mask[0, 5]

    def test_set_value_preserves_original(self):
        fd = _make_formant_data()
        orig = fd.original_values[0, 5]
        fd.set_value(0, 5, 9999.0)
        assert fd.original_values[0, 5] == orig

    def test_set_value_boundary_formant_0(self):
        fd = _make_formant_data()
        fd.set_value(0, 0, 123.0)
        assert fd.values[0, 0] == 123.0
        assert fd.edited_mask[0, 0]

    def test_set_value_boundary_formant_4(self):
        fd = _make_formant_data()
        fd.set_value(4, 0, 4000.0)
        assert fd.values[4, 0] == 4000.0
        assert fd.edited_mask[4, 0]

    def test_set_value_boundary_frame_first(self):
        fd = _make_formant_data()
        fd.set_value(0, 0, 111.0)
        assert fd.values[0, 0] == 111.0

    def test_set_value_boundary_frame_last(self):
        fd = _make_formant_data(n_frames=10)
        fd.set_value(0, 9, 999.0)
        assert fd.values[0, 9] == 999.0

    def test_set_value_out_of_range_formant(self):
        fd = _make_formant_data()
        # Should not crash
        fd.set_value(5, 0, 100.0)
        fd.set_value(-1, 0, 100.0)

    def test_set_value_out_of_range_frame(self):
        fd = _make_formant_data(n_frames=10)
        fd.set_value(0, 10, 100.0)
        fd.set_value(0, -1, 100.0)

    def test_set_value_nan(self):
        fd = _make_formant_data()
        fd.set_value(0, 0, np.nan)
        assert np.isnan(fd.values[0, 0])
        assert fd.edited_mask[0, 0]

    def test_set_value_zero(self):
        fd = _make_formant_data()
        fd.set_value(0, 0, 0.0)
        assert fd.values[0, 0] == 0.0
        assert not np.isnan(fd.values[0, 0])

    def test_multiple_edits_same_point(self):
        fd = _make_formant_data()
        fd.set_value(0, 0, 100.0)
        fd.set_value(0, 0, 200.0)
        assert fd.values[0, 0] == 200.0
        assert fd.edited_mask[0, 0]

    def test_edit_does_not_affect_other_formants(self):
        fd = _make_formant_data()
        orig_f2 = fd.values[1].copy()
        fd.set_value(0, 5, 9999.0)
        npt.assert_array_equal(fd.values[1], orig_f2)


# ===================================================================
# 6.3.3 Reset
# ===================================================================

class TestReset:

    def test_reset_single_formant(self):
        fd = _make_formant_data()
        orig = fd.original_values[0].copy()
        fd.set_value(0, 5, 9999.0)
        fd.reset_to_original(0)
        npt.assert_array_equal(fd.values[0], orig)
        assert not np.any(fd.edited_mask[0])

    def test_reset_single_preserves_others(self):
        fd = _make_formant_data()
        fd.set_value(0, 5, 9999.0)
        fd.set_value(1, 3, 8888.0)
        fd.reset_to_original(0)
        assert fd.values[1, 3] == 8888.0
        assert fd.edited_mask[1, 3]

    def test_reset_all(self):
        fd = _make_formant_data()
        fd.set_value(0, 5, 9999.0)
        fd.set_value(1, 3, 8888.0)
        fd.reset_to_original()
        npt.assert_array_equal(fd.values, fd.original_values)
        assert not np.any(fd.edited_mask)

    def test_reset_idempotent(self):
        fd = _make_formant_data()
        fd.set_value(0, 0, 9999.0)
        fd.reset_to_original(0)
        vals_after_first = fd.values[0].copy()
        fd.reset_to_original(0)
        npt.assert_array_equal(fd.values[0], vals_after_first)

    def test_reset_after_no_edits(self):
        fd = _make_formant_data()
        orig = fd.values.copy()
        fd.reset_to_original()
        npt.assert_array_equal(fd.values, orig)


# ===================================================================
# 6.3.4 Save/Load Round-Trip
# ===================================================================

class TestSaveLoad:

    def test_save_creates_file(self, tmp_path):
        fd = _make_formant_data()
        path = tmp_path / "test.formants"
        fd.save(str(path))
        assert path.exists()

    def test_save_valid_json(self, tmp_path):
        fd = _make_formant_data()
        path = tmp_path / "test.formants"
        fd.save(str(path))
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_save_schema_fields(self, tmp_path):
        fd = _make_formant_data()
        path = tmp_path / "test.formants"
        fd.save(str(path))
        with open(path) as f:
            data = json.load(f)
        required = {"format_version", "generator", "analysis_engine",
                     "n_formants", "n_frames", "time_step_seconds",
                     "times", "values", "edited_mask"}
        assert required.issubset(data.keys())

    def test_save_format_version(self, tmp_path):
        fd = _make_formant_data()
        path = tmp_path / "test.formants"
        fd.save(str(path))
        with open(path) as f:
            data = json.load(f)
        assert data["format_version"] == 1

    def test_save_generator(self, tmp_path):
        fd = _make_formant_data()
        path = tmp_path / "test.formants"
        fd.save(str(path))
        with open(path) as f:
            data = json.load(f)
        assert data["generator"] == "FormantStudio"

    def test_roundtrip_times(self, tmp_path):
        fd = _make_formant_data()
        path = tmp_path / "test.formants"
        fd.save(str(path))
        loaded = FormantData.load(str(path))
        npt.assert_array_almost_equal(loaded.times, fd.times)

    def test_roundtrip_values(self, tmp_path):
        fd = _make_formant_data()
        path = tmp_path / "test.formants"
        fd.save(str(path))
        loaded = FormantData.load(str(path))
        # NaN positions should match
        nan_orig = np.isnan(fd.values)
        nan_load = np.isnan(loaded.values)
        npt.assert_array_equal(nan_orig, nan_load)
        # Non-NaN values should be close (within rounding)
        mask = ~nan_orig
        npt.assert_array_almost_equal(loaded.values[mask], fd.values[mask], decimal=2)

    def test_roundtrip_edited_mask(self, tmp_path):
        fd = _make_formant_data()
        fd.set_value(0, 3, 555.0)
        path = tmp_path / "test.formants"
        fd.save(str(path))
        loaded = FormantData.load(str(path))
        npt.assert_array_equal(loaded.edited_mask, fd.edited_mask)

    def test_roundtrip_nan_as_null(self, tmp_path):
        fd = _make_formant_data()
        # F4 is all NaN
        path = tmp_path / "test.formants"
        fd.save(str(path))
        with open(path) as f:
            data = json.load(f)
        # JSON should have null for NaN
        assert all(v is None for v in data["values"][3])
        # Reload should be NaN
        loaded = FormantData.load(str(path))
        assert np.all(np.isnan(loaded.values[3]))

    def test_roundtrip_time_step(self, tmp_path):
        fd = _make_formant_data()
        path = tmp_path / "test.formants"
        fd.save(str(path))
        loaded = FormantData.load(str(path))
        assert loaded.time_step == pytest.approx(fd.time_step)

    def test_roundtrip_n_formants(self, tmp_path):
        fd = _make_formant_data()
        path = tmp_path / "test.formants"
        fd.save(str(path))
        loaded = FormantData.load(str(path))
        assert loaded.n_formants == fd.n_formants

    def test_roundtrip_with_edits(self, tmp_path):
        fd = _make_formant_data()
        fd.set_value(0, 0, 111.0)
        fd.set_value(1, 5, 222.0)
        fd.set_value(2, 9, 333.0)
        path = tmp_path / "test.formants"
        fd.save(str(path))
        loaded = FormantData.load(str(path))
        assert loaded.values[0, 0] == pytest.approx(111.0, abs=0.01)
        assert loaded.values[1, 5] == pytest.approx(222.0, abs=0.01)
        assert loaded.values[2, 9] == pytest.approx(333.0, abs=0.01)
        assert loaded.edited_mask[0, 0]
        assert loaded.edited_mask[1, 5]
        assert loaded.edited_mask[2, 9]

    def test_roundtrip_no_edits(self, tmp_path):
        fd = _make_formant_data()
        path = tmp_path / "test.formants"
        fd.save(str(path))
        loaded = FormantData.load(str(path))
        assert not np.any(loaded.edited_mask)

    def test_roundtrip_all_nan(self, tmp_path):
        times = np.linspace(0, 0.1, 10)
        values = np.full((5, 10), np.nan)
        fd = FormantData(times=times, values=values, time_step=0.01)
        path = tmp_path / "test.formants"
        fd.save(str(path))
        loaded = FormantData.load(str(path))
        assert np.all(np.isnan(loaded.values))

    def test_roundtrip_precision(self, tmp_path):
        fd = _make_formant_data()
        fd.set_value(0, 0, 123.456789)
        path = tmp_path / "test.formants"
        fd.save(str(path))
        with open(path) as f:
            data = json.load(f)
        # Should be rounded to 2 decimal places
        assert data["values"][0][0] == 123.46

    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            FormantData.load("nonexistent_file.formants")

    def test_load_malformed_json(self, tmp_path):
        path = tmp_path / "bad.formants"
        path.write_text("not valid json {{{")
        with pytest.raises(json.JSONDecodeError):
            FormantData.load(str(path))

    def test_load_missing_fields(self, tmp_path):
        path = tmp_path / "incomplete.formants"
        path.write_text('{"format_version": 1}')
        with pytest.raises(KeyError):
            FormantData.load(str(path))

    def test_save_overwrites_existing(self, tmp_path):
        fd1 = _make_formant_data(n_frames=5)
        fd2 = _make_formant_data(n_frames=20)
        path = tmp_path / "test.formants"
        fd1.save(str(path))
        fd2.save(str(path))
        loaded = FormantData.load(str(path))
        assert loaded.n_frames == 20
