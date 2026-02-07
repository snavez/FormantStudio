"""
Tests for TextGrid editing operations — add/move/delete boundaries,
interval selection, label editing, and cross-tier aligned boundaries.

These test the data manipulation logic directly on TextGrid/Tier objects
using the same algorithms as SpectrogramCanvas methods, without needing
a running Qt application.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from formant_editor import Interval, Point, Tier, TextGrid


# ---------------------------------------------------------------------------
# Helper: build a simple TextGrid for testing
# ---------------------------------------------------------------------------

def _make_interval_tier():
    """IntervalTier 'words' [0, 2.0] with 3 intervals."""
    return Tier("words", "IntervalTier", 0, 2.0, intervals=[
        Interval(0.0, 0.5, "the"),
        Interval(0.5, 1.0, "cat"),
        Interval(1.0, 2.0, ""),
    ])


def _make_text_tier():
    """TextTier 'events' [0, 2.0] with 2 points."""
    return Tier("events", "TextTier", 0, 2.0, points=[
        Point(0.5, "click"),
        Point(1.5, "beep"),
    ])


def _make_textgrid():
    return TextGrid(0, 2.0, [_make_interval_tier(), _make_text_tier()])


# ---------------------------------------------------------------------------
# We test the data manipulation logic directly on TextGrid/Tier objects
# using the same algorithms as SpectrogramCanvas methods.
# ---------------------------------------------------------------------------

def _add_boundary(tier, time):
    """Add a boundary (mirrors SpectrogramCanvas._add_boundary logic)."""
    if tier.tier_class == "IntervalTier":
        for i, iv in enumerate(tier.intervals):
            if iv.xmin < time < iv.xmax:
                old_text = iv.text
                tier.intervals[i] = Interval(iv.xmin, time, old_text)
                tier.intervals.insert(i + 1, Interval(time, iv.xmax, ""))
                return True
    elif tier.tier_class == "TextTier":
        pt = Point(time, "")
        for i, existing in enumerate(tier.points):
            if time < existing.time:
                tier.points.insert(i, pt)
                return True
        tier.points.append(pt)
        return True
    return False


def _move_boundary(tier, old_time, new_time):
    """Move a boundary (mirrors SpectrogramCanvas._move_boundary logic)."""
    if tier.tier_class == "IntervalTier":
        for iv in tier.intervals:
            if iv.xmax == old_time:
                iv.xmax = new_time
            elif iv.xmin == old_time:
                iv.xmin = new_time
    elif tier.tier_class == "TextTier":
        for pt in tier.points:
            if pt.time == old_time:
                pt.time = new_time
                break


def _delete_boundary(tier, boundary_time):
    """Delete a boundary (mirrors SpectrogramCanvas._delete_boundary logic)."""
    if tier.tier_class == "IntervalTier":
        if boundary_time == tier.xmin or boundary_time == tier.xmax:
            return False
        left_idx = None
        right_idx = None
        for i, iv in enumerate(tier.intervals):
            if iv.xmax == boundary_time:
                left_idx = i
            if iv.xmin == boundary_time:
                right_idx = i
        if left_idx is not None and right_idx is not None:
            tier.intervals[left_idx] = Interval(
                tier.intervals[left_idx].xmin,
                tier.intervals[right_idx].xmax,
                tier.intervals[left_idx].text,
            )
            tier.intervals.pop(right_idx)
            return True
    elif tier.tier_class == "TextTier":
        for i, pt in enumerate(tier.points):
            if pt.time == boundary_time:
                tier.points.pop(i)
                return True
    return False


# ===================================================================
# Tests — Add boundary
# ===================================================================

class TestAddBoundary:

    def test_add_boundary_splits_interval(self):
        tier = _make_interval_tier()
        assert _add_boundary(tier, 0.75)
        assert len(tier.intervals) == 4
        assert tier.intervals[1] == Interval(0.5, 0.75, "cat")
        assert tier.intervals[2] == Interval(0.75, 1.0, "")

    def test_add_boundary_left_keeps_text(self):
        tier = _make_interval_tier()
        _add_boundary(tier, 1.5)
        assert tier.intervals[2] == Interval(1.0, 1.5, "")
        assert tier.intervals[3] == Interval(1.5, 2.0, "")

    def test_add_boundary_at_existing_returns_false(self):
        """Clicking on an existing boundary doesn't duplicate it."""
        tier = _make_interval_tier()
        # 0.5 is already a boundary — it's at the edge of intervals,
        # so no interval has xmin < 0.5 < xmax with 0.5 strictly inside
        assert not _add_boundary(tier, 0.5)
        assert len(tier.intervals) == 3

    def test_add_point_to_text_tier(self):
        tier = _make_text_tier()
        assert _add_boundary(tier, 1.0)
        assert len(tier.points) == 3
        assert tier.points[1] == Point(1.0, "")

    def test_add_point_sorted_order(self):
        tier = _make_text_tier()
        _add_boundary(tier, 0.2)
        assert tier.points[0] == Point(0.2, "")
        assert tier.points[1] == Point(0.5, "click")


# ===================================================================
# Tests — Move boundary
# ===================================================================

class TestMoveBoundary:

    def test_move_interval_boundary(self):
        tier = _make_interval_tier()
        _move_boundary(tier, 0.5, 0.6)
        assert tier.intervals[0] == Interval(0.0, 0.6, "the")
        assert tier.intervals[1] == Interval(0.6, 1.0, "cat")

    def test_move_point(self):
        tier = _make_text_tier()
        _move_boundary(tier, 0.5, 0.7)
        assert tier.points[0] == Point(0.7, "click")


# ===================================================================
# Tests — Delete boundary
# ===================================================================

class TestDeleteBoundary:

    def test_delete_merges_intervals(self):
        tier = _make_interval_tier()
        assert _delete_boundary(tier, 0.5)
        assert len(tier.intervals) == 2
        assert tier.intervals[0] == Interval(0.0, 1.0, "the")

    def test_delete_keeps_left_text(self):
        tier = _make_interval_tier()
        _delete_boundary(tier, 1.0)
        assert tier.intervals[1] == Interval(0.5, 2.0, "cat")

    def test_cannot_delete_tier_start(self):
        tier = _make_interval_tier()
        assert not _delete_boundary(tier, 0.0)
        assert len(tier.intervals) == 3

    def test_cannot_delete_tier_end(self):
        tier = _make_interval_tier()
        assert not _delete_boundary(tier, 2.0)
        assert len(tier.intervals) == 3

    def test_delete_point(self):
        tier = _make_text_tier()
        assert _delete_boundary(tier, 0.5)
        assert len(tier.points) == 1
        assert tier.points[0] == Point(1.5, "beep")


# ===================================================================
# Helpers — Interval selection (mirrors SpectrogramCanvas._select_interval)
# ===================================================================

def _select_interval_tier(tier, click_time):
    """Find which interval contains click_time. Returns (index, interval) or None."""
    if tier.tier_class == "IntervalTier":
        for i, iv in enumerate(tier.intervals):
            if iv.xmin <= click_time <= iv.xmax:
                return (i, iv)
    return None


def _select_point_near(tier, click_time, threshold):
    """Find nearest point within threshold. Returns (index, point) or None."""
    if tier.tier_class != "TextTier":
        return None
    best_idx = None
    best_dist = float('inf')
    for i, pt in enumerate(tier.points):
        d = abs(pt.time - click_time)
        if d < best_dist:
            best_dist = d
            best_idx = i
    if best_idx is not None and best_dist <= threshold:
        return (best_idx, tier.points[best_idx])
    return None


# ===================================================================
# Tests — Interval selection
# ===================================================================

class TestIntervalSelection:

    def test_select_first_interval(self):
        tier = _make_interval_tier()
        result = _select_interval_tier(tier, 0.25)
        assert result is not None
        idx, iv = result
        assert idx == 0
        assert iv.text == "the"

    def test_select_middle_interval(self):
        tier = _make_interval_tier()
        result = _select_interval_tier(tier, 0.75)
        assert result is not None
        idx, iv = result
        assert idx == 1
        assert iv.text == "cat"

    def test_select_last_interval(self):
        tier = _make_interval_tier()
        result = _select_interval_tier(tier, 1.5)
        assert result is not None
        idx, iv = result
        assert idx == 2
        assert iv.text == ""

    def test_select_at_boundary_hits_interval(self):
        """Click exactly on a boundary should still select an interval."""
        tier = _make_interval_tier()
        result = _select_interval_tier(tier, 0.5)
        assert result is not None

    def test_select_outside_returns_none(self):
        """Click outside tier range returns None (no interval at time -1)."""
        tier = _make_interval_tier()
        result = _select_interval_tier(tier, -1.0)
        assert result is None

    def test_select_point_near(self):
        tier = _make_text_tier()
        result = _select_point_near(tier, 0.52, threshold=0.1)
        assert result is not None
        idx, pt = result
        assert idx == 0
        assert pt.mark == "click"

    def test_select_point_too_far(self):
        tier = _make_text_tier()
        result = _select_point_near(tier, 1.0, threshold=0.1)
        assert result is None


# ===================================================================
# Tests — Label editing (commit)
# ===================================================================

class TestLabelEdit:

    def test_commit_interval_label(self):
        """Committing a label changes the interval's text."""
        tier = _make_interval_tier()
        tier.intervals[1].text = "dog"
        assert tier.intervals[1].text == "dog"
        assert tier.intervals[1].xmin == 0.5
        assert tier.intervals[1].xmax == 1.0

    def test_commit_point_mark(self):
        """Committing a label changes a point's mark."""
        tier = _make_text_tier()
        tier.points[0].mark = "snap"
        assert tier.points[0].mark == "snap"

    def test_commit_empty_label(self):
        """Can set a label to empty string."""
        tier = _make_interval_tier()
        tier.intervals[0].text = ""
        assert tier.intervals[0].text == ""

    def test_commit_unicode_label(self):
        """Labels can contain Unicode (IPA, accented chars)."""
        tier = _make_interval_tier()
        tier.intervals[0].text = "\u0259\u0301"  # schwa + accent
        assert tier.intervals[0].text == "\u0259\u0301"


# ===================================================================
# Tests — Aligned (shadow) boundary creation
# ===================================================================

class TestAlignedBoundary:

    def test_aligned_boundary_interval_tier(self):
        """Creating aligned boundary at exact time from another tier."""
        tg = _make_textgrid()
        interval_tier = tg.tiers[0]
        # Add a boundary at 0.75 (simulating aligned creation from tier 1)
        assert _add_boundary(interval_tier, 0.75)
        assert len(interval_tier.intervals) == 4
        assert interval_tier.intervals[1].xmax == 0.75
        assert interval_tier.intervals[2].xmin == 0.75

    def test_aligned_boundary_text_tier(self):
        """Creating aligned point on a TextTier at exact time from IntervalTier."""
        tg = _make_textgrid()
        text_tier = tg.tiers[1]
        # Align to the 0.5 boundary from tier 0
        assert _add_boundary(text_tier, 1.0)
        assert len(text_tier.points) == 3
        assert text_tier.points[1].time == 1.0

    def test_aligned_boundary_preserves_existing(self):
        """Aligned boundary doesn't disturb existing data."""
        tg = _make_textgrid()
        interval_tier = tg.tiers[0]
        text_tier = tg.tiers[1]
        _add_boundary(interval_tier, 1.5)
        # Original intervals still intact around the split
        assert interval_tier.intervals[0] == Interval(0.0, 0.5, "the")
        assert interval_tier.intervals[1] == Interval(0.5, 1.0, "cat")
        # Text tier unchanged
        assert len(text_tier.points) == 2


# ===================================================================
# Tests — Boundary drag (single-move semantics)
# ===================================================================

class TestBoundaryDragOnRelease:
    """Test that boundary drag commits the move only once (original → final),
    matching the blit-based drag behavior where data isn't mutated per-frame."""

    def test_single_move_from_original_to_final(self):
        """Moving directly from original to final position is correct."""
        tier = _make_interval_tier()
        original = 0.5
        final = 0.7
        _move_boundary(tier, original, final)
        assert tier.intervals[0] == Interval(0.0, 0.7, "the")
        assert tier.intervals[1] == Interval(0.7, 1.0, "cat")

    def test_large_drag_within_constraints(self):
        tier = _make_interval_tier()
        # Drag the 1.0 boundary to 1.8 (between 0.5 and 2.0)
        _move_boundary(tier, 1.0, 1.8)
        assert tier.intervals[1] == Interval(0.5, 1.8, "cat")
        assert tier.intervals[2] == Interval(1.8, 2.0, "")

    def test_point_drag_single_move(self):
        tier = _make_text_tier()
        _move_boundary(tier, 1.5, 1.8)
        assert tier.points[1] == Point(1.8, "beep")
