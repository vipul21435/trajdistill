"""The funnel report: cleaning counts, rows removed by each filter, and the composition of the
data before and after (by repository, outcome, steps and size), with a concentration index.

Concentration is the Herfindahl-Hirschman index over repository shares, ``HHI = sum(s_i^2)``
where ``s_i`` is repository i's share of the rows. It is 1 when one repository holds every
row and ``1/R`` when R repositories hold equal shares; ``1/HHI`` reads as an effective number
of repositories.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .clean import CleanConfig, CleanStats, clean_all
from .model import Trajectory, content_hash
from .select import SelectConfig, Selection, select
from .tokens import TokenCounter


def nearest_rank(values: Sequence[float], q: float) -> float:
    """The q-quantile by the nearest-rank rule (q in [0, 1]); 0 for an empty list."""
    if not values:
        return 0
    s = sorted(values)
    return s[max(0, math.ceil(q * len(s)) - 1)]


def _summary(values: list[int]) -> dict[str, float]:
    if not values:
        return {"min": 0, "median": 0, "mean": 0.0, "p90": 0, "max": 0}
    return {"min": min(values), "median": statistics.median(values),
            "mean": round(statistics.fmean(values), 2), "p90": nearest_rank(values, 0.9),
            "max": max(values)}


def hhi(counts: Iterable[int]) -> float:
    counts = list(counts)
    total = sum(counts)
    if total == 0:
        return 0.0
    return sum((c / total) ** 2 for c in counts)


def composition(trajs: Sequence[Trajectory], counter: TokenCounter, top: int = 5
                ) -> dict[str, Any]:
    """Who and what is in ``trajs``: repositories, outcomes, steps, tokens and duplicates."""
    n = len(trajs)
    repos = Counter(t.repo for t in trajs)
    ranked = sorted(repos.items(), key=lambda kv: (-kv[1], kv[0]))
    runs = Counter(t.instance_id for t in trajs)
    tokens = [counter.count_trajectory(t) for t in trajs]
    conc = hhi(repos.values())
    return {
        "rows": n,
        "instances": len(runs),
        "instances_with_several_runs": sum(1 for c in runs.values() if c > 1),
        "repositories": len(repos),
        "top_repositories": [{"repo": r, "rows": c, "share": round(c / n, 4)}
                             for r, c in ranked[:top]],
        "largest_repo_share": round(ranked[0][1] / n, 4) if n else 0.0,
        "hhi": round(conc, 4),
        "effective_repositories": round(1 / conc, 1) if conc else 0.0,
        "outcome": {
            "resolved": sum(t.resolved is True for t in trajs),
            "unresolved": sum(t.resolved is False for t in trajs),
            "unknown": sum(t.resolved is None for t in trajs),
        },
        "steps": _summary([t.n_steps for t in trajs]),
        "tokens": dict(_summary(tokens), total=sum(tokens)),
        "exact_duplicates": n - len({content_hash(t) for t in trajs}),
    }


@dataclass
class Report:
    selection: Selection
    cleaning: CleanStats
    before: dict[str, Any]
    tokens_after_cleaning: int
    after: dict[str, Any]

    @property
    def kept(self) -> list[Trajectory]:
        return self.selection.kept

    def to_dict(self) -> dict[str, Any]:
        return {
            "cleaning": self.cleaning.to_dict(),
            "tokens_after_cleaning": self.tokens_after_cleaning,
            "funnel": self.selection.to_dict(),
            "before": self.before,
            "after": self.after,
        }


def run_pipeline(trajs: Iterable[Trajectory], clean_cfg: CleanConfig | None = None,
                 select_cfg: SelectConfig | None = None, counter: TokenCounter | None = None
                 ) -> Report:
    """Read -> composition before -> clean -> select -> composition after."""
    counter = counter or TokenCounter()
    raw = list(trajs)
    before = composition(raw, counter)
    cleaned, stats = clean_all(raw, clean_cfg or CleanConfig())
    tokens_after_cleaning = sum(counter.count_trajectory(t) for t in cleaned)
    sel = select(cleaned, select_cfg, counter)
    return Report(selection=sel, cleaning=stats, before=before,
                  tokens_after_cleaning=tokens_after_cleaning,
                  after=composition(sel.kept, counter))


_HEAD = ("filter", "setting", "before", "removed", "changed", "after")


def funnel_table(sel: Selection) -> str:
    """Plain-text funnel: one row per filter in the order it ran; ``off`` rows removed nothing."""
    rows = [_HEAD]
    for s in sel.stages:
        setting = s.setting if s.enabled else "off"
        rows.append((s.name, setting, str(s.rows_before), str(s.removed), str(s.changed),
                     str(s.rows_after)))
    lines = [f"rows in: {sel.rows_in}   rows out: {sel.rows_out}"]
    lines += _table(rows, left=2)
    lines.append(f"patches touching test files (flagged, before filtering): "
                 f"{sel.flagged_test_patches}")
    return "\n".join(lines)


def _table(rows: list[tuple[str, ...]], left: int) -> list[str]:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    out = []
    for r in rows:
        cells = [c.ljust(w) for c, w in zip(r[:left], widths[:left], strict=True)]
        cells += [c.rjust(w) for c, w in zip(r[left:], widths[left:], strict=True)]
        out.append("  ".join(cells).rstrip())
    return out


def _pct(x: float) -> str:
    return f"{100 * x:.1f} %"


def composition_table(before: dict[str, Any], after: dict[str, Any]) -> str:
    def top(c: dict[str, Any]) -> str:
        if not c["top_repositories"]:
            return "-"
        t = c["top_repositories"][0]
        return f"{t['repo']} {_pct(t['share'])}"

    def outcome(c: dict[str, Any]) -> str:
        o = c["outcome"]
        return f"{o['resolved']} / {o['unresolved']} / {o['unknown']}"

    def spread(d: dict[str, Any]) -> str:
        return f"{d['median']:g} / {d['p90']:g} / {d['max']:g}"

    rows = [("composition", "before", "after")]
    for label, fn in [
        ("rows", lambda c: str(c["rows"])),
        ("instances", lambda c: str(c["instances"])),
        ("instances with several runs", lambda c: str(c["instances_with_several_runs"])),
        ("repositories", lambda c: str(c["repositories"])),
        ("largest repository", top),
        ("concentration (HHI)", lambda c: f"{c['hhi']:.4f}"),
        ("effective repositories", lambda c: f"{c['effective_repositories']:g}"),
        ("resolved / unresolved / unknown", outcome),
        ("steps median / p90 / max", lambda c: spread(c["steps"])),
        ("tokens median / p90 / max", lambda c: spread(c["tokens"])),
        ("tokens total", lambda c: str(c["tokens"]["total"])),
        ("exact duplicates", lambda c: str(c["exact_duplicates"])),
    ]:
        rows.append((label, fn(before), fn(after)))
    return "\n".join(_table(rows, left=1))


def report_text(rep: Report) -> str:
    parts = []
    changed = {k: v for k, v in rep.cleaning.to_dict().items() if v}
    if changed:
        parts.append("cleaning: " + ", ".join(f"{k} {v}" for k, v in changed.items()))
        parts.append(f"tokens after cleaning (all rows): {rep.tokens_after_cleaning}")
    parts.append(funnel_table(rep.selection))
    parts.append(composition_table(rep.before, rep.after))
    return "\n\n".join(parts)
