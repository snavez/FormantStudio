"""Per-corpus settings, kept beside the TextGrids they describe.

A corpus profile records the conventions one body of annotation follows --
which labels mean "nothing to analyse here", which tiers to check and how --
so they are set once rather than re-entered on every run.  It is saved as
:data:`PROFILE_NAME` in the TextGrid folder and found there automatically by
Label Check and by Build CSV; it can also be saved elsewhere and loaded into
another corpus.

Like ``label_check``, this module is free of Qt and of ``formant_editor``.
"""

import json
import os
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Optional, Tuple

PROFILE_NAME = "formantstudio-profile.json"
FORMAT = "FormantStudio corpus profile"
VERSION = 1

# Labels that mark an interval as deliberately empty rather than unannotated.
# A bare "-" is one keypress, so a pause or a gap can be marked without the
# typo risk of a longer convention.
DEFAULT_NON_SPEECH = ("-",)


class ProfileError(Exception):
    """A profile file that exists but cannot be used, with the reason."""


# What a tier's labels may be: symbols of a notation, or a list of its own.
NOTATIONS = ("ipa", "sampa", "xsampa")
ONLY_LISTED = "list"
RULE_KINDS = NOTATIONS + (ONLY_LISTED,)


@dataclass(frozen=True)
class LabelRule:
    """What one tier's labels are allowed to be.

    *kind* is a notation (``"ipa"``, ``"sampa"``, ``"xsampa"``), accepting
    that notation's symbols -- only *labels* among them when given, as the
    corpus's inventory -- or :data:`ONLY_LISTED`, accepting exactly *labels*,
    case and all.
    """

    tier: str
    kind: str
    labels: Tuple[str, ...] = ()


@dataclass(frozen=True)
class LabelCheckSettings:
    """What Label Check runs, as the user last set it up for this corpus.

    A tier list of None means every tier, which is also what a fresh corpus
    starts with; an empty tuple means none.  *label_rules* holds one
    :class:`LabelRule` per tier whose labels are checked; a tier without
    one may carry any label.
    """

    unlabelled: bool = True
    unlabelled_tiers: Optional[Tuple[str, ...]] = None
    labels: bool = True
    label_rules: Tuple[LabelRule, ...] = ()
    boundaries: bool = True
    boundary_tiers: Optional[Tuple[str, ...]] = None
    reference_tier: str = ""
    tolerance_ms: float = 20.0


@dataclass(frozen=True)
class CorpusProfile:
    """One corpus's conventions.

    *non_speech_labels* are the labels that leave an interval with nothing
    to analyse: Label Check accepts them as deliberate, and Build CSV leaves
    their intervals out.  *audio_dir* is where the recordings live when not
    beside the TextGrids ("" when they are).
    """

    non_speech_labels: Tuple[str, ...] = DEFAULT_NON_SPEECH
    audio_dir: str = ""
    label_check: LabelCheckSettings = field(default_factory=LabelCheckSettings)

    def with_label_check(self, **changes):
        """A copy with the given Label Check settings changed."""
        return replace(self, label_check=replace(self.label_check, **changes))

    # ------------------------------------------------------------------
    # Files
    # ------------------------------------------------------------------

    def save(self, path):
        """Write the profile to *path* as JSON."""
        data = {"format": FORMAT, "version": VERSION, **asdict(self)}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
            fh.write("\n")

    @classmethod
    def load(cls, path):
        """Read a profile from *path*.

        Raises :class:`ProfileError` when the file is unreadable, is not a
        profile, or was written by a newer FormantStudio.  Settings missing
        from the file take their defaults, and unknown ones are ignored, so
        a profile from a version with more settings still loads.
        """
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError) as e:
            raise ProfileError(f"{os.path.basename(path)}: {e}") from e
        if not isinstance(data, dict) or data.get("format") != FORMAT:
            raise ProfileError(
                f"{os.path.basename(path)} is not a FormantStudio corpus "
                "profile")
        if not isinstance(data.get("version"), int) or data["version"] > VERSION:
            raise ProfileError(
                f"{os.path.basename(path)} was written by a newer version of "
                "FormantStudio")
        try:
            return cls(
                non_speech_labels=_labels(data.get(
                    "non_speech_labels", DEFAULT_NON_SPEECH)),
                audio_dir=str(data.get("audio_dir", "")),
                label_check=_settings(data.get("label_check", {})))
        except (TypeError, ValueError) as e:
            raise ProfileError(
                f"{os.path.basename(path)} has a malformed setting: {e}") from e

    @classmethod
    def for_folder(cls, folder):
        """The profile saved in *folder*, or the defaults when there is none.

        Raises :class:`ProfileError` for a profile that exists but is broken:
        silently falling back to the defaults would quietly change what gets
        analysed.
        """
        path = profile_path(folder)
        return cls.load(path) if os.path.exists(path) else cls()

    def save_to_folder(self, folder):
        """Save as *folder*'s own profile, where it will be found again."""
        self.save(profile_path(folder))


def profile_path(folder):
    """Where *folder*'s profile lives."""
    return os.path.join(folder, PROFILE_NAME)


def parse_labels(text):
    """Labels from comma-separated *text*, blanks dropped, order kept.

    Commas rather than spaces, because labels such as ``<speech noise>``
    contain spaces.
    """
    seen = []
    for part in text.split(","):
        label = part.strip()
        if label and label not in seen:
            seen.append(label)
    return tuple(seen)


def format_labels(labels):
    """*labels* as the comma-separated text :func:`parse_labels` reads."""
    return ", ".join(labels)


def _labels(value):
    if not isinstance(value, (list, tuple)):
        raise TypeError("expected a list of labels")
    return tuple(str(v).strip() for v in value if str(v).strip())


def _settings(data):
    if not isinstance(data, dict):
        raise TypeError("label_check must be an object")
    known = {f.name: f for f in fields(LabelCheckSettings)}
    values = {}
    for key, value in data.items():
        if key not in known:
            continue
        if key.endswith("_tiers"):
            value = None if value is None else _labels(value)
        elif key == "label_rules":
            value = _rules(value)
        elif key == "tolerance_ms":
            value = float(value)
        elif isinstance(known[key].default, bool):
            value = bool(value)
        else:
            value = str(value)
        values[key] = value
    return LabelCheckSettings(**values)


def parse_inventory(text):
    """Symbols from *text*, separated by commas or spaces, order kept.

    Symbols never contain spaces, so either separator will do -- unlike
    :func:`parse_labels`, whose labels may.
    """
    return parse_labels(text.replace(" ", ",").replace("\t", ","))


def _rules(value):
    if not isinstance(value, (list, tuple)):
        raise TypeError("label_rules must be a list")
    rules = []
    for item in value:
        if not isinstance(item, dict) or "tier" not in item:
            raise TypeError("each label rule needs a tier")
        kind = str(item.get("kind", ""))
        if kind not in RULE_KINDS:
            raise ValueError(f"unknown label rule kind {kind!r}")
        rules.append(LabelRule(tier=str(item["tier"]), kind=kind,
                               labels=_labels(item.get("labels", []))))
    return tuple(rules)
