"""Cleaning transforms. Each one is optional and counted, and the whole pass is idempotent.

Order: strip ANSI escape codes, redact, shorten observations (redacting again after a
cut), normalise whitespace (these act on text), then drop failed steps, then collapse
consecutive identical steps (these act on the step list). Running the pass twice gives the
same result as running it once; the tests check this on random inputs.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace

from .model import Step, ToolCall, Trajectory

# CSI sequences, OSC sequences (ended by BEL or ESC \), charset designations, then the
# remaining two-character escapes.
ANSI = re.compile(
    r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?|\x1b[()*+][0-9A-Za-z]"
    r"|\x1b[@-Z\\-_]"
)
_WS = re.compile(r"\s+")

_SEP = r"[\\/]"
# a home directory followed by a path separator, not inside a URL (no ! # : / % before it)
_HOME = re.compile(
    r"(?<![\w.~!#:/%-])(?:/home|/" + "Users" + r"|[A-Za-z]:\\" + "Users" + r")" + _SEP
    + r"[A-Za-z0-9._-]+(?=[\\/])"
)
# Key-like strings, built from pieces so this file holds no literal secret-shaped text.
_KEY_PARTS = [
    "s" + r"k-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}",
    "gh" + r"[pousr]_[A-Za-z0-9]{30,}",
    "github" + r"_pat_[A-Za-z0-9_]{30,}",
    "AK" + r"IA[0-9A-Z]{16}",
    "h" + r"f_[A-Za-z0-9]{30,}",
    "xo" + r"x[abprs]-[A-Za-z0-9-]{10,}",
]
_KEY = re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(_KEY_PARTS) + r")")
_PEM = re.compile(
    "-----BEGIN [A-Z ]*PRIV" + "ATE KEY-----.*?(?:-----END [A-Z ]*PRIV" + "ATE KEY-----|\\Z)",
    re.DOTALL,
)
REDACTED = "[REDACTED]"
HOME_PLACEHOLDER = "~"

_TRAILING = re.compile(r"[ \t]+$", re.MULTILINE)
_BLANKS = re.compile(r"\n{3,}")

DEFAULT_FAILURE_PATTERNS = (
    r"Your proposed edit has introduced new syntax error",
    r"command not found",
)


@dataclass
class CleanConfig:
    """Which transforms to run. Everything is off by default."""

    strip_ansi: bool = False
    redact: bool = False
    max_observation_chars: int | None = None
    head_fraction: float = 0.5
    normalise_whitespace: bool = False
    drop_failed: bool = False
    failure_patterns: tuple[str, ...] = DEFAULT_FAILURE_PATTERNS
    collapse_repeats: bool = False

    def __post_init__(self) -> None:
        if self.max_observation_chars is not None and self.max_observation_chars < 1:
            raise ValueError("max_observation_chars must be >= 1")
        if not 0.0 <= self.head_fraction <= 1.0:
            raise ValueError("head_fraction must be between 0 and 1")
        self.failure_patterns = tuple(self.failure_patterns)
        for p in self.failure_patterns:
            try:
                re.compile(p)
            except re.error as exc:
                raise ValueError(f"bad failure pattern {p!r}: {exc}") from None

    @property
    def active(self) -> bool:
        return any((self.strip_ansi, self.redact, self.max_observation_chars is not None,
                    self.normalise_whitespace, self.drop_failed, self.collapse_repeats))


@dataclass
class CleanStats:
    """Counts of what the cleaning pass changed, summed over trajectories."""

    counts: Counter[str] = field(default_factory=Counter)

    def add(self, other: CleanStats) -> None:
        self.counts.update(other.counts)

    def to_dict(self) -> dict[str, int]:
        keys = ("trajectories_changed", "ansi_texts", "redactions", "observations_shortened",
                "chars_elided", "whitespace_texts", "failed_steps_dropped", "steps_collapsed")
        return {k: int(self.counts.get(k, 0)) for k in keys}


def strip_ansi(text: str) -> str:
    return ANSI.sub("", text)


def redact(text: str) -> tuple[str, int]:
    """Replace private-key blocks and key-like strings with ``[REDACTED]`` and home
    directories (``/home/<name>``, the macOS and Windows user folders) with ``~``."""
    text, a = _PEM.subn(REDACTED, text)
    text, b = _KEY.subn(REDACTED, text)
    text, c = _HOME.subn(HOME_PLACEHOLDER, text)
    return text, a + b + c


def elision_marker(n: int) -> str:
    return f"\n[... {n} characters elided ...]\n"


def shorten(text: str, budget: int, head_fraction: float = 0.5) -> str:
    """Keep the head and tail of ``text`` around an elision marker so the result has at most
    ``budget`` characters. Text already within budget is returned unchanged."""
    if budget < 1:
        raise ValueError("budget must be >= 1")
    if len(text) <= budget:
        return text
    widest = len(elision_marker(len(text)))
    keep = budget - widest
    if keep <= 0:
        return text[:budget]
    head = int(keep * head_fraction)
    tail = keep - head
    elided = len(text) - head - tail
    return text[:head] + elision_marker(elided) + (text[len(text) - tail:] if tail else "")


def normalise_whitespace(text: str) -> str:
    """Unify line endings, drop trailing spaces, squeeze 3+ newlines to 2, trim the ends."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TRAILING.sub("", text)
    text = _BLANKS.sub("\n\n", text)
    return text.strip("\n")


def is_failed(step: Step, patterns: Iterable[re.Pattern[str]]) -> bool:
    if step.failed:
        return True
    obs = step.observation
    return any(p.search(obs) for p in patterns)


def _text_pipeline(cfg: CleanConfig, stats: Counter[str]) -> Callable[[str, bool], str]:
    def run(text: str, is_observation: bool) -> str:
        if cfg.strip_ansi:
            new = strip_ansi(text)
            stats["ansi_texts"] += new != text
            text = new
        if cfg.redact:
            text, n = redact(text)
            stats["redactions"] += n
        if is_observation and cfg.max_observation_chars is not None:
            new = shorten(text, cfg.max_observation_chars, cfg.head_fraction)
            if new != text:
                stats["observations_shortened"] += 1
                stats["chars_elided"] += len(text) - len(new)
                if cfg.redact:
                    # a cut can expose a fragment the first pass skipped; replacements
                    # are never longer than what they replace, so the budget still holds
                    new, n = redact(new)
                    stats["redactions"] += n
            text = new
        if cfg.normalise_whitespace:
            new = normalise_whitespace(text)
            stats["whitespace_texts"] += new != text
            text = new
        return text

    return run


def clean(traj: Trajectory, cfg: CleanConfig) -> tuple[Trajectory, CleanStats]:
    """Return a cleaned copy of ``traj`` and the counts of what changed."""
    stats: Counter[str] = Counter()
    if not cfg.active:
        return traj, CleanStats(stats)
    run = _text_pipeline(cfg, stats)
    steps = []
    for s in traj.steps:
        calls = []
        for c in s.tool_calls:
            args = c.arguments
            if cfg.redact:
                args, n = redact(args)
                stats["redactions"] += n
            calls.append(ToolCall(c.name, args, c.id))
        steps.append(Step(
            content=run(s.content, False),
            observations=[run(o, True) for o in s.observations],
            tool_calls=calls,
            action=run(s.action, False),
            failed=s.failed,
            reply_roles=list(s.reply_roles),
            reply_ids=list(s.reply_ids),
        ))
    if cfg.drop_failed:
        compiled = [re.compile(p) for p in cfg.failure_patterns]
        kept = [s for s in steps if not is_failed(s, compiled)]
        stats["failed_steps_dropped"] += len(steps) - len(kept)
        steps = kept
    if cfg.collapse_repeats:
        collapsed: list[Step] = []
        for s in steps:
            if collapsed and _same_step(collapsed[-1], s):
                stats["steps_collapsed"] += 1
                continue
            collapsed.append(s)
        steps = collapsed
    patch = traj.patch
    if cfg.redact:
        patch, n = redact(patch)
        stats["redactions"] += n
    out = replace(traj, steps=steps, system=run(traj.system, False), task=run(traj.task, False),
                  patch=patch, meta=dict(traj.meta))
    if out != traj:
        stats["trajectories_changed"] += 1
    return out, CleanStats(stats)


def _same_step(a: Step, b: Step) -> bool:
    """Same action (whitespace-normalised), same observations and same tool calls apart from
    their ids, which differ on every call. The thought text may differ."""
    return (_WS.sub(" ", a.action).strip() == _WS.sub(" ", b.action).strip()
            and a.observations == b.observations
            and [(c.name, c.arguments) for c in a.tool_calls]
            == [(c.name, c.arguments) for c in b.tool_calls])


def clean_all(trajs: Iterable[Trajectory], cfg: CleanConfig) -> tuple[list[Trajectory],
                                                                       CleanStats]:
    total = CleanStats()
    out = []
    for t in trajs:
        c, s = clean(t, cfg)
        out.append(c)
        total.add(s)
    return out, total
