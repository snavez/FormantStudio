"""Integrity checks over a directory of TextGrid label files.

Run before the CSV builder, this answers a narrow question: is the annotation
self-consistent enough to measure?  Three faults matter, because each one
silently corrupts an export rather than stopping it — an interval nobody
labelled contributes an empty row, a mistyped symbol splits one phone into
two categories, and a boundary that missed its neighbour by three milliseconds
makes a segment that is measured slightly wrong everywhere it appears.

Every check reports :class:`Issue` records and, where a fix can be inferred
rather than guessed, proposes one.  Nothing here edits anything: applying the
fixes is a separate, explicit step (:func:`apply_fixes`), so the caller can put
the proposals in front of a person first.

The module is deliberately free of Qt and of ``formant_editor`` — the TextGrid
loader is injected — so the checks can be tested on their own.
"""

import difflib
import os
import shutil
from dataclasses import dataclass, field

from corpus_profile import DEFAULT_NON_SPEECH


# Boundaries agreeing to within this are already aligned, not drifted; it
# absorbs the float round-trip through a TextGrid file.
TIME_EPS = 1e-6

# Allophone tiers use these to say a phoneme was absorbed into its neighbour
# rather than produced. They are annotation syntax, not symbols to validate.
ARROWS = ("<", ">")

UNLABELLED = "unlabelled"
UNKNOWN_SYMBOL = "unknown-symbol"
NOT_IN_INVENTORY = "not-in-inventory"
NOT_ALLOWED = "not-allowed"
BOUNDARY_DRIFT = "boundary-drift"

# Appended to a grid's name to keep what was there before a fix.
BACKUP_SUFFIX = ".bak"


@dataclass(frozen=True)
class Issue:
    """One problem found in one file, with a fix when one can be inferred.

    *index* addresses the thing to change on *tier*: the interval or point
    carrying a bad label, or — for :data:`BOUNDARY_DRIFT` — the interval whose
    ``xmin`` is the drifted boundary.  *proposed* is empty when the fault is
    real but the repair is not ours to guess.
    """

    file: str
    kind: str
    tier: str
    index: int
    time: float
    current: str
    proposed: str
    detail: str

    @property
    def fixable(self):
        """Whether this issue carries a fix that can be applied."""
        return bool(self.proposed)


@dataclass(frozen=True)
class LabelPolicy:
    """How one tier's labels are judged.

    *accept* says whether a label is allowed; *candidates* are the labels
    repairs are drawn from; *kind* is the issue kind a rejected label is
    reported as (:data:`UNKNOWN_SYMBOL`, :data:`NOT_IN_INVENTORY` or
    :data:`NOT_ALLOWED`).
    """

    accept: object
    candidates: frozenset
    kind: str

    @classmethod
    def only(cls, labels, kind=NOT_ALLOWED):
        """Accept exactly *labels*, case and all."""
        allowed = frozenset(labels)
        return cls(allowed.__contains__, allowed, kind)


@dataclass
class CheckOptions:
    """Which checks to run, over which tiers, and how strictly.

    An empty tier tuple disables its check outright, so the caller enables a
    check simply by naming the tiers it applies to; *label_policies* maps
    each tier whose labels are judged to its :class:`LabelPolicy`.
    *non_speech* are the corpus's labels for "nothing to analyse here" (see
    ``corpus_profile``), accepted on every tier.
    """

    non_speech: tuple = DEFAULT_NON_SPEECH
    unlabelled_tiers: tuple = ()
    label_policies: dict = field(default_factory=dict)
    boundary_tiers: tuple = ()
    reference_tier: str = ""
    tolerance_s: float = 0.020


def _tier(tg, name):
    """The tier called *name*, or None when this file has no such tier."""
    return next((t for t in tg.tiers if t.name == name), None)


def _is_blank(text):
    """True when nothing at all has been written here."""
    return not text.strip()


def _skip_symbol(text, non_speech):
    """Whether *text* is annotation syntax rather than a symbol to validate."""
    stripped = text.strip()
    return (_is_blank(stripped) or stripped in non_speech
            or stripped in ARROWS)


def _annotated_extent(intervals):
    """Index range ``[first, last]`` of the tier's annotated region.

    Bounded by the outermost intervals carrying any text at all, non-speech
    labels included: a marked pause is part of what was annotated, whereas the
    silence before the speaker started is not.  Returns None when the tier is
    entirely blank.
    """
    written = [i for i, iv in enumerate(intervals) if not _is_blank(iv.text)]
    if not written:
        return None
    return written[0], written[-1]


def check_unlabelled(tg, name, tier_names, non_speech=DEFAULT_NON_SPEECH):
    """Blank intervals and points that look like omissions rather than silence.

    Praat interval tiers tile, so most of a file is legitimately blank; only a
    hole *inside* the annotated extent is evidence of something missed.  The
    proposed fix is the first of the *non_speech* labels, which is the one
    repair that can be made without knowing what the annotator meant to
    write — accepting it says "nothing to analyse here", and declining it
    leaves the interval for them.  With no non-speech labels configured
    nothing is proposed.
    """
    pause = non_speech[0] if non_speech else ""
    issues = []
    for tier_name in tier_names:
        tier = _tier(tg, tier_name)
        if tier is None:
            continue
        if tier.tier_class == "IntervalTier":
            extent = _annotated_extent(tier.intervals)
            if extent is None:
                continue
            first, last = extent
            for i in range(first, last + 1):
                iv = tier.intervals[i]
                if not _is_blank(iv.text):
                    continue
                issues.append(Issue(
                    file=name, kind=UNLABELLED, tier=tier_name, index=i,
                    time=iv.xmin, current="", proposed=pause,
                    detail="blank interval inside the annotated region"))
        else:
            for i, pt in enumerate(tier.points):
                if _is_blank(pt.mark):
                    issues.append(Issue(
                        file=name, kind=UNLABELLED, tier=tier_name, index=i,
                        time=pt.time, current="", proposed="",
                        detail="point carries no mark"))
    return issues


def _one_keystroke_away(label, alphabet):
    """Labels one slip away from *label*, grouped likeliest slip first.

    A doubled key is the commonest typing slip, then two keys swapped, then
    a wrong key, then a missed one -- so deletions are tried before
    transpositions, substitutions and insertions.
    """
    n = len(label)
    return [
        [label[:i] + label[i + 1:] for i in range(n)],
        [label[:i] + label[i + 1] + label[i] + label[i + 2:]
         for i in range(n - 1)],
        [label[:i] + c + label[i + 1:] for i in range(n) for c in alphabet],
        [label[:i] + c + label[i:] for i in range(n + 1) for c in alphabet],
    ]


def suggest_symbols(label, accept, candidates,
                    non_speech=DEFAULT_NON_SPEECH):
    """Repairs for a rejected *label*, best first; empty when none is likely.

    Spelling-corrector style: the labels one keystroke away that *accept*
    takes, from the likeliest kind of slip that yields any.  That reaches
    composed labels a candidate list never holds -- ``p__h`` gives ``p_h``,
    a base symbol plus a diacritic.  Only when no single slip explains the
    label does it fall back to the closest whole symbol in *candidates*.
    A non-speech label is never suggested: it would hide the fault rather
    than repair it.
    """
    alphabet = sorted({c for s in candidates for c in s} | set(label))
    for group in _one_keystroke_away(label, alphabet):
        hits = []
        for c in group:
            if (c and c != label and c not in hits
                    and not _skip_symbol(c, non_speech) and accept(c)):
                hits.append(c)
        if hits:
            return sorted(hits, key=lambda c: -difflib.SequenceMatcher(
                None, label, c).ratio())
    symbols = sorted(c for c in candidates if not _skip_symbol(c, non_speech))
    return difflib.get_close_matches(label, symbols, n=3, cutoff=0.6)


_REJECTION = {
    UNKNOWN_SYMBOL: "not a recognised symbol",
    NOT_IN_INVENTORY: "not in the inventory",
    NOT_ALLOWED: "not one of the allowed labels",
}


def check_labels(tg, name, policies, non_speech=DEFAULT_NON_SPEECH):
    """Labels their tier's policy rejects, with the likeliest repair proposed.

    *policies* maps tier names to :class:`LabelPolicy`: a chart lookup
    catches symbols that are not valid at all, an inventory catches symbols
    that are valid but impossible for the language, and a plain list holds a
    tier such as targets to its own few labels.  Blanks, *non_speech* labels
    and divergence arrows are annotation syntax and are never validated.

    No suggestion is proposed when nothing is close enough, which is the honest
    answer for a label that is simply wrong rather than mistyped.  Other
    plausible repairs are named in the detail, since the first is a guess.
    """
    issues = []
    for tier_name, policy in policies.items():
        tier = _tier(tg, tier_name)
        if tier is None:
            continue
        if tier.tier_class == "IntervalTier":
            items = [(i, iv.text, iv.xmin) for i, iv in enumerate(tier.intervals)]
        else:
            items = [(i, pt.mark, pt.time) for i, pt in enumerate(tier.points)]
        for i, text, when in items:
            label = text.strip()
            if _skip_symbol(label, non_speech) or policy.accept(label):
                continue
            near = suggest_symbols(label, policy.accept, policy.candidates,
                                   non_speech)
            detail = _REJECTION[policy.kind]
            if len(near) > 1:
                detail += " — or perhaps " + ", ".join(near[1:4])
            issues.append(Issue(
                file=name, kind=policy.kind, tier=tier_name, index=i,
                time=when, current=label, proposed=near[0] if near else "",
                detail=detail))
    return issues


def _boundaries(tier):
    """Movable boundaries on *tier* as ``(index, time)``.

    An interval tier's boundaries are the starts of its intervals; index 0 is
    the tier's own beginning and is not movable, so it is left out.  A point
    tier's points are all movable.
    """
    if tier.tier_class == "IntervalTier":
        return [(i, iv.xmin) for i, iv in enumerate(tier.intervals) if i > 0]
    return [(i, pt.time) for i, pt in enumerate(tier.points)]


def _reference_times(tier):
    """Every boundary time on the reference tier, including its far edge."""
    if tier.tier_class == "IntervalTier":
        if not tier.intervals:
            return []
        return ([iv.xmin for iv in tier.intervals]
                + [tier.intervals[-1].xmax])
    return [pt.time for pt in tier.points]


def _move_is_safe(tier, index, new_time):
    """Whether moving boundary *index* to *new_time* keeps the tier ordered.

    A drifted boundary can sit close to a reference time that lies the far side
    of its own neighbour; snapping to it would invert the interval.  Such a
    case is a real fault but not one to repair automatically.
    """
    if tier.tier_class == "IntervalTier":
        before = tier.intervals[index - 1]
        here = tier.intervals[index]
        return before.xmin < new_time < here.xmax
    lo = tier.points[index - 1].time if index > 0 else float("-inf")
    hi = (tier.points[index + 1].time if index + 1 < len(tier.points)
          else float("inf"))
    return lo < new_time < hi


def check_boundaries(tg, name, reference_tier, tier_names, tolerance_s):
    """Boundaries that nearly, but not quite, meet the reference tier.

    Only near misses are reported.  A boundary sitting exactly on a reference
    time is already aligned, and one further away than *tolerance_s* is a
    genuine subdivision — an allophone tier splitting a phone in two — not a
    slip.  The band between those is where a dragged boundary hides.
    """
    ref = _tier(tg, reference_tier)
    if ref is None:
        return []
    ref_times = _reference_times(ref)
    if not ref_times:
        return []

    issues = []
    for tier_name in tier_names:
        if tier_name == reference_tier:
            continue
        tier = _tier(tg, tier_name)
        if tier is None:
            continue
        for index, t in _boundaries(tier):
            nearest = min(ref_times, key=lambda r: abs(r - t))
            gap = abs(nearest - t)
            # The tolerance is inclusive, with TIME_EPS of slack: a boundary
            # exactly at the limit differs from it only by float noise, and
            # without the slack it would fall in or out unpredictably.
            if gap <= TIME_EPS or gap > tolerance_s + TIME_EPS:
                continue
            safe = _move_is_safe(tier, index, nearest)
            issues.append(Issue(
                file=name, kind=BOUNDARY_DRIFT, tier=tier_name, index=index,
                time=t, current=f"{t:.4f}",
                proposed=f"{nearest:.6f}" if safe else "",
                detail=(f"{gap * 1000:.1f} ms from '{reference_tier}'"
                        + ("" if safe else " — snapping would cross a neighbour"))))
    return issues


def check_textgrid(tg, name, options):
    """Every enabled check over one loaded TextGrid, in report order."""
    issues = []
    if options.unlabelled_tiers:
        issues += check_unlabelled(tg, name, options.unlabelled_tiers,
                                   options.non_speech)
    if options.label_policies:
        issues += check_labels(tg, name, options.label_policies,
                               options.non_speech)
    if options.boundary_tiers and options.reference_tier:
        issues += check_boundaries(tg, name, options.reference_tier,
                                   options.boundary_tiers, options.tolerance_s)
    return issues


def apply_fixes(tg, issues):
    """Apply the fixable *issues* to a loaded TextGrid, in place.

    Boundary moves are re-validated here rather than trusted from the report:
    an earlier fix may have moved the neighbour a later one was checked
    against, and applying a stale proposal would corrupt the tier it was meant
    to repair.  Returns ``(applied, skipped)``.
    """
    applied = skipped = 0
    for issue in issues:
        if not issue.fixable:
            skipped += 1
            continue
        tier = _tier(tg, issue.tier)
        if tier is None:
            skipped += 1
            continue

        if issue.kind == BOUNDARY_DRIFT:
            t = float(issue.proposed)
            if not _move_is_safe(tier, issue.index, t):
                skipped += 1
                continue
            if tier.tier_class == "IntervalTier":
                tier.intervals[issue.index - 1].xmax = t
                tier.intervals[issue.index].xmin = t
            else:
                tier.points[issue.index].time = t
        elif tier.tier_class == "IntervalTier":
            tier.intervals[issue.index].text = issue.proposed
        else:
            tier.points[issue.index].mark = issue.proposed
        applied += 1
    return applied, skipped


def textgrid_paths(textgrid_dir):
    """Every TextGrid in *textgrid_dir*, sorted, whatever the extension case."""
    return sorted(
        os.path.join(textgrid_dir, f)
        for f in os.listdir(textgrid_dir)
        if os.path.splitext(f)[1].lower() == ".textgrid"
    )


def paths_by_name(textgrid_dir):
    """Each grid's base name mapped to its path.

    Issues carry the name rather than the path, so applying a fix has to find
    the file again — and the extension's case varies across corpora.
    """
    return {os.path.splitext(os.path.basename(p))[0]: p
            for p in textgrid_paths(textgrid_dir)}


def save_with_backup(tg, path, suffix=BACKUP_SUFFIX):
    """Write *tg* over *path*, keeping what was there beside it.

    Rewriting someone's annotation in batch is not obviously reversible, so
    the file being replaced is copied first.  An existing backup is left
    alone: it holds the state before this tool ever ran, which is the version
    worth being able to return to — a second run would otherwise overwrite the
    only copy of the annotator's own work.
    """
    backup = path + suffix
    if os.path.exists(path) and not os.path.exists(backup):
        shutil.copy2(path, backup)
    tg.save(path)


def check_directory(textgrid_dir, options, loader, progress=None):
    """Run the enabled checks over every TextGrid in *textgrid_dir*.

    *loader* maps a path to a TextGrid, injected so this module needs nothing
    from the editor.  A file that cannot be read is reported rather than
    raised: one corrupt grid should not hide the faults in all the others.

    Returns ``(issues, unreadable)`` where *unreadable* is a list of
    ``(filename, reason)``.
    """
    paths = textgrid_paths(textgrid_dir)
    issues, unreadable = [], []
    for n, path in enumerate(paths):
        name = os.path.splitext(os.path.basename(path))[0]
        if progress:
            progress(n, len(paths), name)
        try:
            tg = loader(path)
        except Exception as e:
            unreadable.append((os.path.basename(path), str(e)))
            continue
        issues += check_textgrid(tg, name, options)
    return issues, unreadable


def tier_names_in(textgrid_dir, loader, max_files=None):
    """Tier names across the directory, as ``(name, tier_class)``, in file order.

    Corpora are not always uniform, so this unions the tiers rather than
    trusting the first file — a tier missing from one grid should still be
    offered for checking in the rest.
    """
    seen = {}
    for path in textgrid_paths(textgrid_dir)[:max_files]:
        try:
            tg = loader(path)
        except Exception:
            continue
        for tier in tg.tiers:
            seen.setdefault(tier.name, tier.tier_class)
    return list(seen.items())
