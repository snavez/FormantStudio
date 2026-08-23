"""Read a phoneme tier against an allophone tier.

A phoneme tier records what should have been said; an allophone tier what was.
Most sounds correspond one to one, but the interesting cases do not: a speaker
substitutes one sound for another, adds one that was never expected, or leaves
one out entirely.

A sound that was left out is the awkward case, because it has no segment of its
own — its span belongs to whatever *was* produced there. It is written as an
arrow pointing at the neighbour that took the time, `<` or `>`, which records
both that the sound was absent and which of its neighbours survived. That second
part is a judgement the acoustics cannot settle, so it is stated rather than
inferred.

This module turns such a pair of tiers into one record per expected sound, and
checks that the annotation is structurally sound before anything downstream
relies on it.
"""
from collections import namedtuple

# Deliberately empty, as opposed to not yet annotated. Meaning comes from the
# combination: "-" against a label is an insertion, "-" against "-" is a stretch
# with nothing to analyse.
EMPTY = "-"

ABSORB_RIGHT = ">"
ABSORB_LEFT = "<"
ARROWS = (ABSORB_LEFT, ABSORB_RIGHT)

MATCH = "match"
SUBSTITUTION = "substitution"
INSERTION = "insertion"
DELETION = "deletion"
NOT_ANALYSED = "not-analysed"
UNANNOTATED = "unannotated"

# One expected sound. *xmin* and *xmax* give the span it should be measured
# over, which for a sound that survived a coalescence is wider than its own
# interval; both are None for a sound that was not produced at all.
Divergence = namedtuple(
    "Divergence", "index phoneme realised kind xmin xmax")


class AnnotationError(Exception):
    """Raised when two tiers cannot be read against each other at all."""


def _text(interval):
    return interval[2].strip()


def is_empty(label):
    """True if *label* deliberately marks an interval as having no content."""
    return label.strip() == EMPTY


def is_arrow(label):
    return label.strip() in ARROWS


def is_anchor(label):
    """True if *label* is a realised sound rather than a marker or a blank."""
    t = label.strip()
    return bool(t) and t not in ARROWS and t != EMPTY


def validate(phonemes, allophones):
    """Return a list of problems with a phoneme/allophone tier pair.

    An empty list means the annotation is structurally sound. Problems are
    reported rather than raised so a caller can show all of them at once.
    """
    problems = []

    if len(phonemes) != len(allophones):
        problems.append(
            f"tiers have different numbers of intervals "
            f"({len(phonemes)} and {len(allophones)}); their boundaries must match")
        return problems

    for i, (p, a) in enumerate(zip(phonemes, allophones)):
        if abs(p[0] - a[0]) > 1e-6 or abs(p[1] - a[1]) > 1e-6:
            problems.append(
                f"interval {i + 1}: boundaries differ between the tiers "
                f"({p[0]:.4f}-{p[1]:.4f} against {a[0]:.4f}-{a[1]:.4f})")

    labels = [_text(a) for a in allophones]
    for i, lab in enumerate(labels):
        if lab == ABSORB_LEFT and i == 0:
            problems.append(
                f"interval 1: '{ABSORB_LEFT}' cannot open a tier — "
                "there is nothing to its left to absorb it")
            continue
        if lab == ABSORB_RIGHT and i == len(labels) - 1:
            problems.append(
                f"interval {i + 1}: '{ABSORB_RIGHT}' cannot close a tier — "
                "there is nothing to its right to absorb it")
            continue
        if lab in ARROWS:
            step = 1 if lab == ABSORB_RIGHT else -1
            j = i + step
            while 0 <= j < len(labels):
                other = labels[j]
                if is_anchor(other):
                    break
                if other == EMPTY or not other:
                    problems.append(
                        f"interval {i + 1}: '{lab}' cannot reach a sound — "
                        f"interval {j + 1} has nothing in it")
                    break
                if other in ARROWS and other != lab:
                    if lab == ABSORB_RIGHT:      # report the pair once
                        problems.append(
                            f"interval {i + 1}: '{lab}' and interval {j + 1}'s "
                            f"'{other}' point at each other with no sound between")
                    break
                j += step
            else:
                problems.append(
                    f"interval {i + 1}: '{lab}' runs off the end of the tier "
                    "without reaching a sound")
    return problems


def _extent(allophones, index):
    """Span of the realised segment anchored at *index*.

    Arrows immediately to the left pointing right, and immediately to the right
    pointing left, all belong to this segment.
    """
    start = index
    while start > 0 and _text(allophones[start - 1]) == ABSORB_RIGHT:
        start -= 1
    end = index
    while (end < len(allophones) - 1
           and _text(allophones[end + 1]) == ABSORB_LEFT):
        end += 1
    return allophones[start][0], allophones[end][1]


def resolve(phonemes, allophones):
    """One :class:`Divergence` per expected sound.

    *phonemes* and *allophones* are sequences of ``(xmin, xmax, label)``. Their
    boundaries must match; call :func:`validate` first if that is not certain.
    """
    if len(phonemes) != len(allophones):
        raise AnnotationError(
            "phoneme and allophone tiers have different numbers of intervals")

    out = []
    for i, (p, a) in enumerate(zip(phonemes, allophones)):
        expected, realised = _text(p), _text(a)

        if is_empty(expected) and is_empty(realised):
            # Nothing was expected and nothing produced: a pause or a stretch
            # deliberately left out of the analysis.
            kind, xmin, xmax = NOT_ANALYSED, p[0], p[1]
        elif is_arrow(realised) or (expected and is_empty(realised)):
            # Expected but not produced. An arrow says a neighbour took its
            # time; a "-" says nothing analysable is there. Either way the
            # sound has no span of its own, so there is nothing to measure.
            kind, xmin, xmax = DELETION, None, None
        elif is_empty(expected) and is_anchor(realised):
            kind, (xmin, xmax) = INSERTION, _extent(allophones, i)
        elif not expected or not realised:
            kind, xmin, xmax = UNANNOTATED, p[0], p[1]
        else:
            kind = MATCH if expected == realised else SUBSTITUTION
            xmin, xmax = _extent(allophones, i)

        out.append(Divergence(i, expected, realised, kind, xmin, xmax))
    return out


def summarise(divergences):
    """Count each kind of divergence, for a per-file report."""
    counts = {}
    for d in divergences:
        counts[d.kind] = counts.get(d.kind, 0) + 1
    return counts
