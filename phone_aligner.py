"""First-pass forced alignment of TextGrid phone labels to a recording.

A template TextGrid carries the phones that *should* be present; this module
decides where in a new recording each of them falls. It works by Viterbi
decoding the phone sequence against per-frame acoustic likelihoods drawn from
a model over broad manner classes — silence, vowel, nasal, approximant,
fricative, stop, affricate — rather than individual phones. Nothing here knows
about any particular language: the only language-facing part is a SAMPA symbol
table, and SAMPA is shared across languages.

The result is expressed as a monotonic time warp rather than a set of
boundaries, so every tier in a TextGrid can be mapped through the same
function. Boundaries that coincide in the template still coincide afterwards,
because identical inputs produce identical outputs.
"""
import io
import os
import sys

import numpy as np

# --------------------------------------------------------------------------
# Broad phonetic classes
# --------------------------------------------------------------------------

SIL = "SIL"
VOWEL = "VOWEL"
NASAL = "NASAL"
APPROX = "APPROX"
FRIC = "FRIC"
STOP = "STOP"
AFFR = "AFFR"

# Broad manner classes, used for reporting and for deciding what a label is.
CLASSES = [SIL, VOWEL, NASAL, APPROX, FRIC, STOP, AFFR]
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}

# The acoustic model splits VOWEL by quality. With a single vowel class nothing
# changes across an /i/-/e/ boundary, leaving the decoder only the duration
# prior to place it by; separating qualities makes such a boundary a real class
# transition. Height and fronting are relative descriptions, so the split
# carries no assumption about any speaker's formant frequencies.
VOWEL_HEIGHTS = ("high", "mid-high", "mid", "mid-low", "low")
VOWEL_FRONTINGS = ("front", "centre", "back")
UNKNOWN_QUALITY = "unknown"


def _vowel_class(height, fronting):
    return f"{VOWEL}_{height}_{fronting}"


# Fixed and exhaustive, so the class list never depends on what a particular
# corpus happened to contain and a stored model always lines up with the code.
MODEL_CLASSES = (
    [c for c in CLASSES if c != VOWEL]
    + [_vowel_class(h, f) for h in VOWEL_HEIGHTS for f in VOWEL_FRONTINGS]
    + [_vowel_class(UNKNOWN_QUALITY, UNKNOWN_QUALITY)]
)
MODEL_CLASS_INDEX = {c: i for i, c in enumerate(MODEL_CLASSES)}

# SAMPA base symbols, plus their IPA equivalents, so a template may be
# transcribed in either. Nothing here is specific to one language.
_VOWELS = set("aeiouy26@3{AEIOUV}8&"
              "\u0251\u0252\u0254\u0259\u025b\u025c\u026a\u0268\u028a"
              "\u0289\u00f8\u0153\u0250\u028c\u00e6\u0264\u026f\u025e")
_NASALS = {"m", "n", "N", "J", "F",
           "\u014b", "\u0272", "\u0274", "\u0273", "\u0271"}
_APPROX = {"w", "j", "l", "r", "4", "L", "5", "R",
           "\u0279", "\u027b", "\u0270", "\u028b", "\u026d", "\u028e",
           "\u026b", "\u027e", "\u027d", "\u0280", "\u0281"}
_FRICS = {"f", "v", "s", "z", "S", "Z", "h", "C", "x", "T", "D", "G", "B", "H",
          "\u0283", "\u0292", "\u03b8", "\u00f0", "\u0278", "\u03b2",
          "\u00e7", "\u029d", "\u0263", "\u03c7", "\u0127", "\u0295",
          "\u0255", "\u0291", "\u0282", "\u0290", "\u0266"}
_STOPS = {"p", "b", "t", "d", "k", "g", "q", "?",
          "\u0294", "\u0261", "\u0288", "\u0256", "\u025f", "\u0262"}
_AFFRS = {"ts", "dz", "tS", "dZ", "kC", "pf",
          "t\u0283", "d\u0292", "t\u0255", "d\u0291"}

# Markers for silence. Matched case-insensitively, and an empty label always
# counts, since that is the one convention every annotation tool shares.
# Markers for silence, matched case-insensitively. A bare "-" is the app's own
# empty-interval marker; the rest cover conventions from other tools.
PAUSE_LABELS = {"", "-", "<p:>", "<p>", "sil", "sp", "sp:", "#", "_", "pau",
                "pause", "silence", "<sil>", "<silence>", "<pause>", "sil.",
                "<eps>", "spn", "<unk>"}

_DIACRITICS = ("_h", "_0", "_j", "_w", "_t", "_d", "_n", "_G")
_COMBINING = "\u0361\u035c\u02b0\u02b2\u02b7\u0325\u030a\u0329\u02d0\u031f\u0320"


def _base_symbol(label):
    """Strip length, voicing, aspiration and tie-bar marks from *label*."""
    base = label
    for mark in _DIACRITICS:
        base = base.replace(mark, "")
    for mark in (":", "=", "~", "\\"):
        base = base.replace(mark, "")
    return "".join(c for c in base if c not in _COMBINING)


def classify_known(label):
    """Return the class of *label*, or None if its symbols are unrecognised.

    Callers use the None case to warn that a template's alphabet is not
    understood, rather than letting a silent guess degrade the alignment.
    """
    if is_pause(label):
        return SIL

    base = _base_symbol(label.strip())
    if not base:
        return SIL
    if base in _AFFRS:
        return AFFR
    if len(base) == 1:
        for group, cls in ((_NASALS, NASAL), (_STOPS, STOP), (_FRICS, FRIC),
                           (_APPROX, APPROX), (_VOWELS, VOWEL)):
            if base in group:
                return cls
        return None
    if all(c in _VOWELS for c in base):
        return VOWEL
    for group, cls in ((_NASALS, NASAL), (_STOPS, STOP), (_FRICS, FRIC),
                       (_APPROX, APPROX)):
        if base[0] in group:
            return cls
    return VOWEL if base[0] in _VOWELS else None


def classify(label):
    """Return the broad acoustic class of a phone *label*.

    Symbols that are not recognised fall back to VOWEL, the commonest class,
    so an unfamiliar transcription degrades the alignment rather than failing
    outright. Use :func:`classify_known` to detect that case.
    """
    cls = classify_known(label)
    return VOWEL if cls is None else cls


def unrecognised_labels(labels):
    """The distinct labels in *labels* whose symbols were not understood."""
    return sorted({l for l in labels if classify_known(l) is None})


_CHART_FILENAME = os.path.join("Docs", "ipa_symbol_chart.csv")
_vowel_quality_cache = None


def _load_vowel_qualities():
    """Map every vowel symbol in the bundled IPA chart to (height, fronting).

    The chart already backs the CSV export's feature columns, so vowel quality
    has one definition in the app rather than a second hand-written one here.
    A missing or unreadable chart is not fatal: vowels simply fall back to the
    unknown-quality class, which the model treats as uninformative.
    """
    global _vowel_quality_cache
    if _vowel_quality_cache is not None:
        return _vowel_quality_cache

    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, _CHART_FILENAME)
    table = {}
    try:
        import csv
        with io.open(path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if (not row.get("ipa") or row["ipa"].startswith("#")
                        or row.get("type") != "vowel"):
                    continue
                height = (row.get("height") or "").strip()
                fronting = (row.get("fronting") or "").strip()
                if height not in VOWEL_HEIGHTS or fronting not in VOWEL_FRONTINGS:
                    continue
                for key in ("ipa", "sampa", "xsampa"):
                    sym = (row.get(key) or "").strip()
                    if sym:
                        table.setdefault(sym, (height, fronting))
    except (OSError, ImportError, csv.Error):
        table = {}
    _vowel_quality_cache = table
    return table


def vowel_quality(label):
    """(height, fronting) for a vowel *label*, or None if it is not known.

    A diphthong is one segment, not two, so it takes the quality of its onset:
    that is where the segment begins, which is what the boundary before it has
    to find.
    """
    base = _base_symbol(label.strip())
    if not base:
        return None
    table = _load_vowel_qualities()
    for candidate in (base, base[:1]):
        if candidate in table:
            return table[candidate]
    return None


def model_class(label):
    """The acoustic model's class for *label*: manner, or vowel quality."""
    manner = classify(label)
    if manner != VOWEL:
        return manner
    quality = vowel_quality(label)
    if quality is None:
        return _vowel_class(UNKNOWN_QUALITY, UNKNOWN_QUALITY)
    return _vowel_class(*quality)


def is_pause(label):
    """True if *label* marks silence rather than a spoken phone.

    Matched case-insensitively, since pause conventions vary between tools.
    """
    return label.strip().lower() in PAUSE_LABELS


# --------------------------------------------------------------------------
# Frame features
# --------------------------------------------------------------------------

HOP = 0.010      # s
WIN = 0.025      # s

FEATURE_NAMES = [
    "energy", "energy_range", "d_energy", "zcr", "centroid", "rolloff85",
    "hf_ratio", "lf_ratio", "flux", "voicing", "autocorr_f0",
    "f1", "f2", "f3", "d_f1", "d_f2", "d_f3",
    "bark_f1_f0", "bark_f2_f1", "bark_f3_f2",
]

# Formant search ceiling, high enough to cover female speakers.
FORMANT_CEILING = 5500.0


def _frame_signal(x, sr):
    win = int(round(WIN * sr))
    hop = int(round(HOP * sr))
    if len(x) < win:
        x = np.pad(x, (0, win - len(x)))
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return x[idx], hop, win


def _voicing(frames, sr):
    """Normalised autocorrelation peak over the 60-400 Hz lag range.

    A cheap stand-in for a pitch tracker: it separates periodic from aperiodic
    frames, which is all the class models need, at a fraction of the cost of
    running a full pitch algorithm.
    """
    n_frames, win = frames.shape
    centred = frames - frames.mean(axis=1, keepdims=True)
    energy = (centred ** 2).sum(axis=1) + 1e-12

    nfft = 1 << int(np.ceil(np.log2(2 * win)))
    spec = np.fft.rfft(centred, n=nfft, axis=1)
    ac = np.fft.irfft(spec * np.conj(spec), n=nfft, axis=1)[:, :win]

    lo = max(1, int(sr / 400.0))
    hi = min(win - 1, int(sr / 60.0))
    if hi <= lo:
        return np.zeros(n_frames), np.zeros(n_frames)
    seg = ac[:, lo:hi + 1]
    k = np.argmax(seg, axis=1)
    return np.clip(seg[np.arange(n_frames), k] / energy, 0.0, 1.0), \
        sr / (k + lo).astype(float)


def _bark(hz):
    """Convert Hz to the Bark scale, on which formant distances are comparable
    across speakers of different vocal tract length."""
    hz = np.maximum(np.asarray(hz, dtype=float), 1.0)
    return 13 * np.arctan(0.00076 * hz) + 3.5 * np.arctan((hz / 7500.0) ** 2)


def _formant_track(sound, times):
    """F1-F3 in kHz at each frame centre, interpolated across undefined frames.

    Formant motion is what distinguishes a vowel from a neighbouring vowel or
    approximant: /w/ has a low F2, /j/ a high one, /r/ a low F3, and none of
    them differ much from a vowel in energy or spectral shape. Undefined
    frames are bridged rather than zero-filled so the tracks stay smooth
    enough for a Gaussian to model.
    """
    fmt = sound.to_formant_burg(time_step=HOP, max_number_of_formants=5,
                                maximum_formant=FORMANT_CEILING)
    tracks = np.empty((len(times), 3))
    for i in range(3):
        vals = np.array([fmt.get_value_at_time(i + 1, float(t)) for t in times])
        good = np.isfinite(vals)
        if not good.any():
            vals = np.full(len(times), (i + 1) * 800.0)
        elif not good.all():
            vals = np.interp(np.arange(len(times)), np.flatnonzero(good), vals[good])
        tracks[:, i] = vals
    return tracks / 1000.0


def extract_features(sound):
    """Return (features [n_frames, n_feat], frame centre times [n_frames]).

    *sound* is a parselmouth Sound, which the caller already holds and which
    the formant tracker needs.
    """
    sample_rate = sound.sampling_frequency
    samples = sound.values[0]
    x = np.asarray(samples, dtype=np.float64)
    peak = np.max(np.abs(x)) if x.size else 0.0
    if peak > 0:
        x = x / peak

    frames, hop, win = _frame_signal(x, sample_rate)
    n = frames.shape[0]
    times = (np.arange(n) * hop + win / 2.0) / sample_rate

    windowed = frames * np.hanning(win)[None, :]
    mag = np.abs(np.fft.rfft(windowed, axis=1))
    freqs = np.fft.rfftfreq(win, 1.0 / sample_rate)
    total = mag.sum(axis=1) + 1e-12

    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    energy = 20 * np.log10(rms)
    floor, ceiling = np.percentile(energy, [5, 95])
    # Peak-relative energy alone makes a noisy recording's silence look loud,
    # so pair it with a level spanning the file's own noise floor to peak.
    energy_range = np.clip((energy - floor) / max(ceiling - floor, 1e-6), -0.5, 1.5)
    energy = energy - ceiling

    zcr = np.diff(np.signbit(frames), axis=1).sum(axis=1) / float(win)
    centroid = (mag * freqs[None, :]).sum(axis=1) / total
    cumulative = np.cumsum(mag, axis=1) / total[:, None]
    rolloff = freqs[np.argmax(cumulative >= 0.85, axis=1)]
    hf = mag[:, freqs >= 4000].sum(axis=1) / total
    lf = mag[:, freqs <= 500].sum(axis=1) / total

    norm = mag / total[:, None]
    flux = np.zeros(n)
    flux[1:] = np.sqrt(((norm[1:] - norm[:-1]) ** 2).sum(axis=1))

    voicing, f0 = _voicing(frames, sample_rate)

    d_energy = np.zeros(n)
    d_energy[1:-1] = (energy[2:] - energy[:-2]) / 2.0

    formants = _formant_track(sound, times)
    d_formants = np.zeros_like(formants)
    d_formants[1:-1] = np.abs(formants[2:] - formants[:-2]) / 2.0

    # Distances between formants on the Bark scale largely cancel vocal tract
    # length, so they describe vowel quality in the relative terms it is
    # actually defined by rather than in a particular speaker's frequencies.
    bark_f = _bark(formants * 1000.0)
    bark_0 = _bark(f0)
    bark_diffs = np.column_stack([
        bark_f[:, 0] - bark_0, bark_f[:, 1] - bark_f[:, 0],
        bark_f[:, 2] - bark_f[:, 1],
    ])

    feats = np.column_stack([
        energy, energy_range, d_energy, zcr, centroid / 1000.0,
        rolloff / 1000.0, hf, lf, flux, voicing, f0 / 100.0,
        formants, d_formants, bark_diffs,
    ])
    return feats.astype(np.float32), times.astype(np.float32)


# --------------------------------------------------------------------------
# Acoustic model
# --------------------------------------------------------------------------

MODEL_FILENAME = "phone_class_model.npz"


def default_model_path():
    """Where the bundled model lives, inside a PyInstaller bundle or beside us."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, MODEL_FILENAME)


class ClassModel:
    """Diagonal-covariance Gaussians over frame features, one per class."""

    def __init__(self, mu, sd, means, variances, log_prior, log_dur, min_frames):
        self.mu = mu
        self.sd = sd
        self.means = means
        self.variances = variances
        self.log_prior = log_prior
        self.log_dur = log_dur          # mean log-duration in frames, per class
        self.min_frames = min_frames.astype(int)

    @classmethod
    def train(cls, frames, labels, durations):
        """Fit from stacked frames, their class indices, and observed per-class
        phone durations in frames."""
        mu = frames.mean(axis=0)
        sd = frames.std(axis=0) + 1e-8
        z = (frames - mu) / sd

        n_feat = frames.shape[1]
        n_cls = len(MODEL_CLASSES)
        means = np.zeros((n_cls, n_feat))
        variances = np.ones((n_cls, n_feat))
        log_prior = np.full(n_cls, -np.log(n_cls))
        log_dur = np.zeros(n_cls)
        min_frames = np.ones(n_cls)
        for c in range(n_cls):
            m = labels == c
            if m.sum() > 10:
                means[c] = z[m].mean(axis=0)
                variances[c] = z[m].var(axis=0) + 1e-3
                log_prior[c] = np.log(m.mean())
            d = durations.get(c)
            if d:
                log_dur[c] = np.log(np.mean(d))
                min_frames[c] = max(1, int(np.percentile(d, 5)))
        return cls(mu, sd, means, variances, log_prior, log_dur, min_frames)

    def log_likelihood(self, frames):
        """[n_frames, n_classes] log-likelihood of each frame under each class."""
        z = (frames - self.mu) / self.sd
        diff = z[:, None, :] - self.means[None, :, :]
        return -0.5 * ((diff ** 2) / self.variances[None] +
                       np.log(self.variances)[None]).sum(axis=-1) + self.log_prior[None]

    def save(self, path):
        np.savez(path, mu=self.mu, sd=self.sd, means=self.means,
                 variances=self.variances, log_prior=self.log_prior,
                 log_dur=self.log_dur, min_frames=self.min_frames)

    @classmethod
    def load(cls, path=None):
        d = np.load(path or default_model_path())
        return cls(d["mu"], d["sd"], d["means"], d["variances"],
                   d["log_prior"], d["log_dur"], d["min_frames"])


# --------------------------------------------------------------------------
# Alignment
# --------------------------------------------------------------------------

class AlignmentError(Exception):
    """Raised when a recording or label sequence cannot be aligned."""


def _viterbi(classes, frames, model, duration_weight=1.0):
    """Frame index at which each phone ends.

    Each phone becomes a chain of sub-states so it must occupy at least its
    class's minimum duration, with a self-loop on the last one so it may run
    as long as the acoustics justify.
    """
    n_phones = len(classes)
    n_frames = frames.shape[0]

    widths = np.maximum(1, model.min_frames[classes])
    if widths.sum() > n_frames:
        widths = np.ones(n_phones, dtype=int)
    if widths.sum() > n_frames:
        raise AlignmentError(
            f"{n_phones} phones will not fit in {n_frames} frames of audio")

    state_phone = np.repeat(np.arange(n_phones), widths)
    n_states = len(state_phone)
    is_last = np.zeros(n_states, dtype=bool)
    is_last[np.cumsum(widths) - 1] = True

    mean_frames = np.exp(model.log_dur[classes])
    tail = np.maximum(1.0, mean_frames - widths + 1.0)
    p_loop = np.clip(1.0 - 1.0 / tail, 1e-6, 1.0 - 1e-6)
    loop_cost = np.where(is_last,
                         duration_weight * np.log(p_loop)[state_phone], -np.inf)
    adv_cost = np.zeros(n_states)
    adv_cost[1:] = np.where(is_last[:-1],
                            duration_weight * np.log1p(-p_loop)[state_phone[:-1]], 0.0)

    emit = model.log_likelihood(frames)[:, classes[state_phone]]

    dp = np.full(n_states, -1e30)
    dp[0] = emit[0, 0]
    back = np.zeros((n_frames, n_states), dtype=bool)
    for t in range(1, n_frames):
        stay = dp + loop_cost
        adv = np.full(n_states, -1e30)
        adv[1:] = dp[:-1] + adv_cost[1:]
        take_adv = adv > stay
        dp = np.where(take_adv, adv, stay) + emit[t]
        back[t] = take_adv

    ends = np.zeros(n_phones, dtype=int)
    s = n_states - 1
    for t in range(n_frames - 1, 0, -1):
        if back[t, s]:
            if s > 0 and state_phone[s] != state_phone[s - 1]:
                ends[state_phone[s - 1]] = t - 1
            s -= 1
    ends[-1] = n_frames - 1
    for k in range(n_phones - 2, -1, -1):
        ends[k] = min(ends[k], ends[k + 1])
    return ends


class TimeWarp:
    """A monotonic map from template time to target time.

    Applying one warp to every tier is what keeps boundaries that coincide in
    the template coinciding afterwards.
    """

    def __init__(self, knots):
        xs, ys = [], []
        for x, y in knots:
            if xs and x <= xs[-1] + 1e-9:
                continue
            if ys and y < ys[-1]:
                y = ys[-1]
            xs.append(x)
            ys.append(y)
        self._xs = np.asarray(xs, dtype=float)
        self._ys = np.asarray(ys, dtype=float)
        self.n_knots = len(xs)

    def __call__(self, t):
        return float(np.interp(t, self._xs, self._ys))

    @classmethod
    def linear(cls, src_duration, dst_duration):
        """The proportional stretch used when no acoustic evidence is sought."""
        return cls([(0.0, 0.0), (src_duration, dst_duration)])


def align_tier(intervals, sound, model=None, duration_weight=1.0):
    """Build a TimeWarp from a template tier's (xmin, xmax, label) intervals.

    *intervals* must cover the template's time domain in order; the warp maps
    template time onto the parselmouth Sound *sound*.
    """
    if not intervals:
        raise AlignmentError("the chosen tier has no intervals to align")
    model = model or ClassModel.load()

    labels = [lab for _, _, lab in intervals]
    classes = np.array([MODEL_CLASS_INDEX[model_class(l)] for l in labels])

    frames, times = extract_features(sound)
    if frames.shape[0] < 2:
        raise AlignmentError("the recording is too short to align")

    ends = _viterbi(classes, frames, model, duration_weight)
    duration = sound.duration
    hop = float(times[1] - times[0])

    knots = [(intervals[0][0], 0.0)]
    for k in range(len(intervals) - 1):
        knots.append((intervals[k][1], min(float(times[ends[k]]) + hop / 2.0, duration)))
    knots.append((intervals[-1][1], duration))
    return TimeWarp(knots)


def warp_textgrid(textgrid, warp, duration):
    """Return a copy of *textgrid* with every tier mapped through *warp*.

    All tiers pass through the one function, so their boundaries keep exactly
    the relationships they had in the template.
    """
    from formant_editor import Interval, Point, Tier, TextGrid

    tiers = []
    for src in textgrid.tiers:
        tier = Tier(src.name, src.tier_class, 0, duration)
        if src.tier_class == "IntervalTier":
            tier.intervals = [Interval(warp(iv.xmin), warp(iv.xmax), iv.text)
                              for iv in src.intervals]
            if tier.intervals:
                tier.intervals[0].xmin = 0
                tier.intervals[-1].xmax = duration
            else:
                tier.intervals = [Interval(0, duration, "")]
        else:
            tier.points = [Point(warp(p.time), p.mark) for p in src.points]
        tiers.append(tier)
    return TextGrid(0, duration, tiers)


def alignable_tiers(textgrid):
    """Every interval tier that could drive an alignment, best candidate first.

    Tier naming is a matter of local convention, so nothing is selected by
    name: a tier simply ranks higher the more of it is short, phone-like
    intervals. A word tier will still align, with fewer anchors to work from.
    Ordering is only a default — the choice belongs to the user.
    """
    scored = []
    for i, tier in enumerate(textgrid.tiers):
        if tier.tier_class != "IntervalTier" or len(tier.intervals) < 2:
            continue
        spoken = [iv for iv in tier.intervals if not is_pause(iv.text)]
        mean_len = (sum(len(iv.text.strip()) for iv in spoken) / len(spoken)
                    if spoken else 99.0)
        scored.append((len(spoken) > 0, len(tier.intervals), -mean_len, -i,
                       tier.name))
    scored.sort(reverse=True)
    return [name for _, _, _, _, name in scored]
