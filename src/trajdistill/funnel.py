"""Render a :class:`~trajdistill.select.Selection` as a funnel table."""

from __future__ import annotations

from .select import Selection

_HEAD = ("filter", "setting", "before", "removed", "changed", "after")


def funnel_table(sel: Selection) -> str:
    """Plain-text funnel: one row per filter in the order it ran; ``off`` rows removed nothing."""
    rows = [_HEAD]
    for s in sel.stages:
        setting = s.setting if s.enabled else "off"
        rows.append((s.name, setting, str(s.rows_before), str(s.removed), str(s.changed),
                     str(s.rows_after)))
    widths = [max(len(r[i]) for r in rows) for i in range(len(_HEAD))]
    lines = [f"rows in: {sel.rows_in}   rows out: {sel.rows_out}"]
    for r in rows:
        cells = [r[0].ljust(widths[0]), r[1].ljust(widths[1])]
        cells += [c.rjust(w) for c, w in zip(r[2:], widths[2:], strict=True)]
        lines.append("  ".join(cells).rstrip())
    lines.append(f"patches touching test files (flagged, before filtering): "
                 f"{sel.flagged_test_patches}")
    return "\n".join(lines)
