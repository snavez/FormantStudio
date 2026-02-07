"""
Tests for TextGrid parser — covers normal format, short format, multiple tiers,
unicode, escaped quotes, edge cases, and format consistency.
"""

import os
import sys
import tempfile
import pytest

# Ensure the project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from formant_editor import (
    Interval, Point, Tier, TextGrid,
    _tokenize_textgrid, _parse_textgrid_tokens,
)


# ---------------------------------------------------------------------------
# Helper to write a temp TextGrid file and parse it
# ---------------------------------------------------------------------------

def _parse_text(text):
    """Parse TextGrid from raw text (convenience for tests)."""
    tokens = _tokenize_textgrid(text)
    return _parse_textgrid_tokens(tokens)


def _write_and_parse(text):
    """Write text to a temp file and parse via TextGrid.from_file."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".TextGrid", delete=False, encoding="utf-8"
    ) as f:
        f.write(text)
        path = f.name
    try:
        return TextGrid.from_file(path)
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Sample TextGrid data — normal format
# ---------------------------------------------------------------------------

NORMAL_SINGLE_INTERVAL = """\
File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 1.5
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 1.5
        intervals: size = 3
        intervals [1]:
            xmin = 0
            xmax = 0.5
            text = "the"
        intervals [2]:
            xmin = 0.5
            xmax = 1.0
            text = "cat"
        intervals [3]:
            xmin = 1.0
            xmax = 1.5
            text = ""
"""

NORMAL_TEXT_TIER = """\
File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 2.0
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "TextTier"
        name = "events"
        xmin = 0
        xmax = 2.0
        points: size = 2
        points [1]:
            number = 0.5
            mark = "click"
        points [2]:
            number = 1.5
            mark = "beep"
"""

NORMAL_MULTI_TIER = """\
File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 2.0
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 2.0
        intervals: size = 2
        intervals [1]:
            xmin = 0
            xmax = 1.0
            text = "hello"
        intervals [2]:
            xmin = 1.0
            xmax = 2.0
            text = "world"
    item [2]:
        class = "TextTier"
        name = "tones"
        xmin = 0
        xmax = 2.0
        points: size = 1
        points [1]:
            number = 0.5
            mark = "H*"
"""

# ---------------------------------------------------------------------------
# Sample TextGrid data — short format
# ---------------------------------------------------------------------------

SHORT_SINGLE_INTERVAL = """\
File type = "ooTextFile"
Object class = "TextGrid"

0
1.5
<exists>
1
"IntervalTier"
"words"
0
1.5
3
0
0.5
"the"
0.5
1.0
"cat"
1.0
1.5
""
"""

SHORT_TEXT_TIER = """\
File type = "ooTextFile"
Object class = "TextGrid"

0
2.0
<exists>
1
"TextTier"
"events"
0
2.0
2
0.5
"click"
1.5
"beep"
"""

SHORT_MULTI_TIER = """\
File type = "ooTextFile"
Object class = "TextGrid"

0
2.0
<exists>
2
"IntervalTier"
"words"
0
2.0
2
0
1.0
"hello"
1.0
2.0
"world"
"TextTier"
"tones"
0
2.0
1
0.5
"H*"
"""


# ===================================================================
# Tests — Normal format
# ===================================================================

class TestNormalFormat:

    def test_single_interval_tier_header(self):
        tg = _parse_text(NORMAL_SINGLE_INTERVAL)
        assert tg.xmin == 0.0
        assert tg.xmax == 1.5
        assert len(tg.tiers) == 1

    def test_single_interval_tier_properties(self):
        tg = _parse_text(NORMAL_SINGLE_INTERVAL)
        tier = tg.tiers[0]
        assert tier.name == "words"
        assert tier.tier_class == "IntervalTier"
        assert tier.xmin == 0.0
        assert tier.xmax == 1.5

    def test_single_interval_tier_intervals(self):
        tg = _parse_text(NORMAL_SINGLE_INTERVAL)
        tier = tg.tiers[0]
        assert len(tier.intervals) == 3
        assert tier.intervals[0] == Interval(0.0, 0.5, "the")
        assert tier.intervals[1] == Interval(0.5, 1.0, "cat")
        assert tier.intervals[2] == Interval(1.0, 1.5, "")

    def test_text_tier_header(self):
        tg = _parse_text(NORMAL_TEXT_TIER)
        assert tg.xmin == 0.0
        assert tg.xmax == 2.0
        assert len(tg.tiers) == 1

    def test_text_tier_points(self):
        tg = _parse_text(NORMAL_TEXT_TIER)
        tier = tg.tiers[0]
        assert tier.name == "events"
        assert tier.tier_class == "TextTier"
        assert len(tier.points) == 2
        assert tier.points[0] == Point(0.5, "click")
        assert tier.points[1] == Point(1.5, "beep")

    def test_multi_tier_count(self):
        tg = _parse_text(NORMAL_MULTI_TIER)
        assert len(tg.tiers) == 2

    def test_multi_tier_order(self):
        tg = _parse_text(NORMAL_MULTI_TIER)
        assert tg.tiers[0].name == "words"
        assert tg.tiers[0].tier_class == "IntervalTier"
        assert tg.tiers[1].name == "tones"
        assert tg.tiers[1].tier_class == "TextTier"

    def test_multi_tier_interval_values(self):
        tg = _parse_text(NORMAL_MULTI_TIER)
        assert tg.tiers[0].intervals[0] == Interval(0.0, 1.0, "hello")
        assert tg.tiers[0].intervals[1] == Interval(1.0, 2.0, "world")

    def test_multi_tier_point_values(self):
        tg = _parse_text(NORMAL_MULTI_TIER)
        assert tg.tiers[1].points[0] == Point(0.5, "H*")


# ===================================================================
# Tests — Short format
# ===================================================================

class TestShortFormat:

    def test_short_interval_tier(self):
        tg = _parse_text(SHORT_SINGLE_INTERVAL)
        assert tg.xmin == 0.0
        assert tg.xmax == 1.5
        assert len(tg.tiers) == 1
        tier = tg.tiers[0]
        assert tier.name == "words"
        assert len(tier.intervals) == 3
        assert tier.intervals[0] == Interval(0.0, 0.5, "the")
        assert tier.intervals[1] == Interval(0.5, 1.0, "cat")
        assert tier.intervals[2] == Interval(1.0, 1.5, "")

    def test_short_text_tier(self):
        tg = _parse_text(SHORT_TEXT_TIER)
        assert len(tg.tiers) == 1
        tier = tg.tiers[0]
        assert tier.name == "events"
        assert tier.tier_class == "TextTier"
        assert len(tier.points) == 2
        assert tier.points[0] == Point(0.5, "click")
        assert tier.points[1] == Point(1.5, "beep")

    def test_short_multi_tier(self):
        tg = _parse_text(SHORT_MULTI_TIER)
        assert len(tg.tiers) == 2
        assert tg.tiers[0].name == "words"
        assert tg.tiers[1].name == "tones"
        assert tg.tiers[0].intervals[0] == Interval(0.0, 1.0, "hello")
        assert tg.tiers[1].points[0] == Point(0.5, "H*")


# ===================================================================
# Tests — Format consistency (normal and short produce identical results)
# ===================================================================

class TestFormatConsistency:

    def test_interval_tier_consistency(self):
        normal = _parse_text(NORMAL_SINGLE_INTERVAL)
        short = _parse_text(SHORT_SINGLE_INTERVAL)
        assert normal == short

    def test_text_tier_consistency(self):
        normal = _parse_text(NORMAL_TEXT_TIER)
        short = _parse_text(SHORT_TEXT_TIER)
        assert normal == short

    def test_multi_tier_consistency(self):
        normal = _parse_text(NORMAL_MULTI_TIER)
        short = _parse_text(SHORT_MULTI_TIER)
        assert normal == short


# ===================================================================
# Tests — Empty intervals and edge cases
# ===================================================================

class TestEdgeCases:

    def test_empty_label_stored_as_empty_string(self):
        tg = _parse_text(NORMAL_SINGLE_INTERVAL)
        assert tg.tiers[0].intervals[2].text == ""

    def test_zero_intervals(self):
        text = """\
File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 1.0
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "empty"
        xmin = 0
        xmax = 1.0
        intervals: size = 0
"""
        tg = _parse_text(text)
        assert len(tg.tiers[0].intervals) == 0

    def test_zero_points(self):
        text = """\
File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 1.0
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "TextTier"
        name = "empty"
        xmin = 0
        xmax = 1.0
        points: size = 0
"""
        tg = _parse_text(text)
        assert len(tg.tiers[0].points) == 0

    def test_single_interval(self):
        text = """\
File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 0.1
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "word"
        xmin = 0
        xmax = 0.1
        intervals: size = 1
        intervals [1]:
            xmin = 0
            xmax = 0.1
            text = "a"
"""
        tg = _parse_text(text)
        assert len(tg.tiers[0].intervals) == 1
        assert tg.tiers[0].intervals[0] == Interval(0.0, 0.1, "a")

    def test_tiny_intervals(self):
        """Intervals with very small durations should parse correctly."""
        text = """\
File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 0.001
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "micro"
        xmin = 0
        xmax = 0.001
        intervals: size = 1
        intervals [1]:
            xmin = 0
            xmax = 0.001
            text = "x"
"""
        tg = _parse_text(text)
        assert tg.tiers[0].intervals[0].xmax == 0.001

    def test_scientific_notation(self):
        """Numbers in scientific notation should parse correctly."""
        text = """\
File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 1.5e1
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "sci"
        xmin = 0
        xmax = 1.5e1
        intervals: size = 1
        intervals [1]:
            xmin = 0
            xmax = 1.5e1
            text = "long"
"""
        tg = _parse_text(text)
        assert tg.xmax == 15.0
        assert tg.tiers[0].intervals[0].xmax == 15.0

    def test_two_interval_tiers(self):
        text = """\
File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 1.0
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 1.0
        intervals: size = 1
        intervals [1]:
            xmin = 0
            xmax = 1.0
            text = "hi"
    item [2]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 1.0
        intervals: size = 2
        intervals [1]:
            xmin = 0
            xmax = 0.5
            text = "h"
        intervals [2]:
            xmin = 0.5
            xmax = 1.0
            text = "ay"
"""
        tg = _parse_text(text)
        assert len(tg.tiers) == 2
        assert tg.tiers[0].name == "words"
        assert tg.tiers[1].name == "phones"
        assert len(tg.tiers[1].intervals) == 2


# ===================================================================
# Tests — Unicode and special characters
# ===================================================================

class TestUnicode:

    def test_ipa_vowels(self):
        text = """\
File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 1.0
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 1.0
        intervals: size = 2
        intervals [1]:
            xmin = 0
            xmax = 0.5
            text = "\u0251"
        intervals [2]:
            xmin = 0.5
            xmax = 1.0
            text = "\u0259"
"""
        tg = _parse_text(text)
        assert tg.tiers[0].intervals[0].text == "\u0251"
        assert tg.tiers[0].intervals[1].text == "\u0259"

    def test_accented_characters(self):
        text = """\
File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 0.5
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 0.5
        intervals: size = 1
        intervals [1]:
            xmin = 0
            xmax = 0.5
            text = "caf\u00e9"
"""
        tg = _parse_text(text)
        assert tg.tiers[0].intervals[0].text == "caf\u00e9"

    def test_cjk_characters(self):
        text = """\
File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 0.5
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 0.5
        intervals: size = 1
        intervals [1]:
            xmin = 0
            xmax = 0.5
            text = "\u4f60\u597d"
"""
        tg = _parse_text(text)
        assert tg.tiers[0].intervals[0].text == "\u4f60\u597d"


# ===================================================================
# Tests — Escaped quotes
# ===================================================================

class TestEscapedQuotes:

    def test_escaped_double_quotes(self):
        """Praat escapes " as "" inside quoted strings."""
        text = '''\
File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 1.0
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 1.0
        intervals: size = 1
        intervals [1]:
            xmin = 0
            xmax = 1.0
            text = "she said ""hello"""
'''
        tg = _parse_text(text)
        assert tg.tiers[0].intervals[0].text == 'she said "hello"'

    def test_escaped_quotes_in_tier_name(self):
        text = '''\
File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 1.0
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "my ""tier"""
        xmin = 0
        xmax = 1.0
        intervals: size = 0
'''
        tg = _parse_text(text)
        assert tg.tiers[0].name == 'my "tier"'


# ===================================================================
# Tests — File I/O
# ===================================================================

class TestFileIO:

    def test_from_file_normal(self):
        tg = _write_and_parse(NORMAL_SINGLE_INTERVAL)
        assert tg.xmin == 0.0
        assert tg.xmax == 1.5
        assert len(tg.tiers) == 1
        assert len(tg.tiers[0].intervals) == 3

    def test_from_file_short(self):
        tg = _write_and_parse(SHORT_SINGLE_INTERVAL)
        assert len(tg.tiers[0].intervals) == 3

    def test_from_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            TextGrid.from_file("/nonexistent/path.TextGrid")

    def test_from_file_invalid_header(self):
        with pytest.raises(ValueError, match="Not a TextGrid"):
            _parse_text('"notATextFile" "TextGrid" 0 1.0 1')


# ===================================================================
# Tests — Tokenizer
# ===================================================================

class TestTokenizer:

    def test_tokenizer_extracts_strings(self):
        tokens = _tokenize_textgrid('"hello" "world"')
        assert tokens == ["hello", "world"]

    def test_tokenizer_extracts_numbers(self):
        tokens = _tokenize_textgrid("42 3.14 -1.5")
        assert tokens == [42.0, 3.14, -1.5]

    def test_tokenizer_extracts_flags(self):
        tokens = _tokenize_textgrid("<exists>")
        assert tokens == ["<exists>"]

    def test_tokenizer_ignores_labels(self):
        tokens = _tokenize_textgrid('class = "IntervalTier"')
        assert tokens == ["IntervalTier"]

    def test_tokenizer_handles_escaped_quotes(self):
        tokens = _tokenize_textgrid('"say ""hi"""')
        assert tokens == ['say "hi"']

    def test_tokenizer_scientific_notation(self):
        tokens = _tokenize_textgrid("1.5e2 -3.0E+1")
        assert tokens == [150.0, -30.0]


# ===================================================================
# Tests — Data class equality and repr
# ===================================================================

class TestDataClasses:

    def test_interval_equality(self):
        a = Interval(0.0, 1.0, "x")
        b = Interval(0.0, 1.0, "x")
        assert a == b

    def test_interval_inequality(self):
        a = Interval(0.0, 1.0, "x")
        b = Interval(0.0, 1.0, "y")
        assert a != b

    def test_point_equality(self):
        a = Point(0.5, "click")
        b = Point(0.5, "click")
        assert a == b

    def test_tier_equality(self):
        a = Tier("w", "IntervalTier", 0, 1, intervals=[Interval(0, 1, "x")])
        b = Tier("w", "IntervalTier", 0, 1, intervals=[Interval(0, 1, "x")])
        assert a == b

    def test_textgrid_equality(self):
        tier = Tier("w", "IntervalTier", 0, 1, intervals=[Interval(0, 1, "x")])
        a = TextGrid(0, 1, [tier])
        b = TextGrid(0, 1, [tier])
        assert a == b

    def test_interval_repr(self):
        iv = Interval(0.0, 1.0, "hi")
        assert "Interval" in repr(iv)
        assert "hi" in repr(iv)

    def test_point_repr(self):
        pt = Point(0.5, "click")
        assert "Point" in repr(pt)

    def test_tier_repr(self):
        tier = Tier("w", "IntervalTier", 0, 1, intervals=[Interval(0, 1, "x")])
        assert "IntervalTier" in repr(tier)
        assert "1 intervals" in repr(tier)


# ===================================================================
# Tests — TextGrid save (round-trip)
# ===================================================================

class TestTextGridSave:

    def _round_trip(self, tg):
        """Save a TextGrid to a temp file and parse it back."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".TextGrid", delete=False, encoding="utf-8"
        ) as f:
            path = f.name
        try:
            tg.save(path)
            return TextGrid.from_file(path)
        finally:
            os.unlink(path)

    def test_round_trip_single_interval_tier(self):
        tg = _parse_text(NORMAL_SINGLE_INTERVAL)
        tg2 = self._round_trip(tg)
        assert tg == tg2

    def test_round_trip_text_tier(self):
        tg = _parse_text(NORMAL_TEXT_TIER)
        tg2 = self._round_trip(tg)
        assert tg == tg2

    def test_round_trip_multi_tier(self):
        tg = _parse_text(NORMAL_MULTI_TIER)
        tg2 = self._round_trip(tg)
        assert tg == tg2

    def test_round_trip_empty_interval_tier(self):
        tg = TextGrid(0, 1.0, [
            Tier("empty", "IntervalTier", 0, 1.0, intervals=[])
        ])
        tg2 = self._round_trip(tg)
        assert tg == tg2

    def test_round_trip_empty_text_tier(self):
        tg = TextGrid(0, 1.0, [
            Tier("empty", "TextTier", 0, 1.0, points=[])
        ])
        tg2 = self._round_trip(tg)
        assert tg == tg2

    def test_round_trip_unicode_ipa(self):
        tg = TextGrid(0, 1.0, [
            Tier("phones", "IntervalTier", 0, 1.0, intervals=[
                Interval(0, 0.5, "\u0251"),
                Interval(0.5, 1.0, "\u0259"),
            ])
        ])
        tg2 = self._round_trip(tg)
        assert tg == tg2

    def test_round_trip_unicode_cjk(self):
        tg = TextGrid(0, 1.0, [
            Tier("words", "IntervalTier", 0, 1.0, intervals=[
                Interval(0, 1.0, "\u4f60\u597d"),
            ])
        ])
        tg2 = self._round_trip(tg)
        assert tg == tg2

    def test_round_trip_escaped_quotes(self):
        tg = TextGrid(0, 1.0, [
            Tier("words", "IntervalTier", 0, 1.0, intervals=[
                Interval(0, 1.0, 'she said "hello"'),
            ])
        ])
        tg2 = self._round_trip(tg)
        assert tg2.tiers[0].intervals[0].text == 'she said "hello"'

    def test_round_trip_escaped_quotes_in_tier_name(self):
        tg = TextGrid(0, 1.0, [
            Tier('my "tier"', "IntervalTier", 0, 1.0, intervals=[
                Interval(0, 1.0, "x"),
            ])
        ])
        tg2 = self._round_trip(tg)
        assert tg2.tiers[0].name == 'my "tier"'

    def test_round_trip_escaped_quotes_in_point_mark(self):
        tg = TextGrid(0, 1.0, [
            Tier("events", "TextTier", 0, 1.0, points=[
                Point(0.5, 'a "quote" mark'),
            ])
        ])
        tg2 = self._round_trip(tg)
        assert tg2.tiers[0].points[0].mark == 'a "quote" mark'

    def test_round_trip_mixed_tiers(self):
        tg = TextGrid(0, 2.0, [
            Tier("words", "IntervalTier", 0, 2.0, intervals=[
                Interval(0, 1.0, "hello"),
                Interval(1.0, 2.0, "world"),
            ]),
            Tier("tones", "TextTier", 0, 2.0, points=[
                Point(0.5, "H*"),
                Point(1.5, "L%"),
            ]),
        ])
        tg2 = self._round_trip(tg)
        assert tg == tg2
