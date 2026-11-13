"""Command line: ``trajdistill select``, ``funnel``, ``convert`` and ``split``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .clean import DEFAULT_FAILURE_PATTERNS, CleanConfig, clean_all
from .funnel import report_text, run_pipeline
from .readers import (
    FORMATS,
    FormatError,
    ReadStats,
    iter_jsonl,
    read_path,
    write_jsonl,
    write_native,
)
from .samples import GRANULARITIES, STYLES, ConvertStats, build_samples
from .select import (
    DEDUPE_KEYS,
    LOOP_KEYS,
    LOOP_MODES,
    OUTCOMES,
    PER_INSTANCE_MODES,
    TEST_PATCH_MODES,
    SelectConfig,
)
from .split import SPLIT_BY, split_records
from .tokens import DEFAULT_CHARS_PER_TOKEN, TokenCounter


def _add_input(p: argparse.ArgumentParser) -> None:
    p.add_argument("input", help="a JSONL/JSON file (optionally .gz), a .traj file, or a "
                                 "directory of .traj files")
    p.add_argument("--format", default="auto", choices=FORMATS, help="input format (default: auto)")


def _add_tokens(p: argparse.ArgumentParser) -> None:
    t = p.add_argument_group("token counting")
    t.add_argument("--tokenizer", choices=("chars", "tiktoken"), default="chars",
                   help="character estimate (default) or tiktoken (needs the tiktoken extra)")
    t.add_argument("--chars-per-token", type=float, default=DEFAULT_CHARS_PER_TOKEN)


def _add_cleaning(p: argparse.ArgumentParser) -> None:
    c = p.add_argument_group("cleaning (applied before the filters, all off by default)")
    c.add_argument("--clean", action="store_true",
                   help="shorthand for --strip-ansi --redact --normalise-whitespace "
                        "--collapse-repeats")
    c.add_argument("--strip-ansi", action="store_true", help="remove ANSI escape codes")
    c.add_argument("--redact", action="store_true",
                   help="replace key-like strings and home directories")
    c.add_argument("--max-observation-chars", type=int, default=None, metavar="N",
                   help="keep the head and tail of longer observations around an elision marker")
    c.add_argument("--head-fraction", type=float, default=0.5,
                   help="share of a shortened observation taken from its start (default: 0.5)")
    c.add_argument("--normalise-whitespace", action="store_true")
    c.add_argument("--collapse-repeats", action="store_true",
                   help="merge consecutive steps with the same action and observation")
    c.add_argument("--drop-failed", action="store_true",
                   help="drop steps whose call failed (the retry that follows is kept)")
    c.add_argument("--failure-pattern", action="append", default=None, metavar="REGEX",
                   help="what marks a failed step (repeatable; replaces the defaults)")


def _add_filters(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("filters (applied in this order)")
    g.add_argument("--no-dedupe", action="store_true", help="keep duplicate trajectories")
    g.add_argument("--dedupe-key", choices=DEDUPE_KEYS, default="content",
                   help="duplicates are identical content, or the same action sequence on the "
                        "same instance (default: content)")
    g.add_argument("--outcome", choices=OUTCOMES, default="resolved",
                   help="keep resolved trajectories only, or any outcome (default: resolved)")
    g.add_argument("--exit-status", action="append", default=[], metavar="PATTERN",
                   help="keep only these exit statuses (glob, repeatable), e.g. 'submitted*'")
    g.add_argument("--allow-empty-patch", action="store_true",
                   help="keep trajectories whose final patch is empty")
    g.add_argument("--test-patches", choices=TEST_PATCH_MODES, default="keep",
                   help="patches touching test files are always flagged; 'drop' removes them")
    g.add_argument("--loops", choices=LOOP_MODES, default="keep",
                   help="on a repeated identical step: keep, drop the trajectory, or cut it there")
    g.add_argument("--loop-key", choices=LOOP_KEYS, default="action+observation",
                   help="what counts as identical (default: action and observation)")
    g.add_argument("--loop-window", type=int, default=1,
                   help="compare each step with this many steps before it (default: 1)")
    g.add_argument("--max-steps", type=int, default=None)
    g.add_argument("--max-tokens", type=int, default=None,
                   help="maximum estimated tokens of the whole trajectory")
    g.add_argument("--per-instance", choices=PER_INSTANCE_MODES, default="all",
                   help="several runs of one instance: keep all, the shortest, the best or random")
    g.add_argument("-k", "--per-instance-k", type=int, default=1,
                   help="how many runs per instance to keep (default: 1)")
    g.add_argument("--max-per-repo", type=int, default=None,
                   help="cap on trajectories per repository (spread across instances)")
    g.add_argument("--seed", type=int, default=0)


def _select_config(a: argparse.Namespace) -> SelectConfig:
    return SelectConfig(
        dedupe=not a.no_dedupe,
        dedupe_key=a.dedupe_key,
        outcome=a.outcome,
        exit_status=tuple(a.exit_status),
        require_patch=not a.allow_empty_patch,
        test_patches=a.test_patches,
        loops=a.loops,
        loop_key=a.loop_key,
        loop_window=a.loop_window,
        max_steps=a.max_steps,
        max_tokens=a.max_tokens,
        per_instance=a.per_instance,
        per_instance_k=a.per_instance_k,
        max_per_repo=a.max_per_repo,
        seed=a.seed,
    )


def _clean_config(a: argparse.Namespace) -> CleanConfig:
    return CleanConfig(
        strip_ansi=a.strip_ansi or a.clean,
        redact=a.redact or a.clean,
        max_observation_chars=a.max_observation_chars,
        head_fraction=a.head_fraction,
        normalise_whitespace=a.normalise_whitespace or a.clean,
        drop_failed=a.drop_failed,
        failure_patterns=tuple(a.failure_pattern or DEFAULT_FAILURE_PATTERNS),
        collapse_repeats=a.collapse_repeats or a.clean,
    )


def _counter(a: argparse.Namespace) -> TokenCounter:
    return TokenCounter(a.tokenizer, chars_per_token=a.chars_per_token)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trajdistill",
        description="Select, clean and convert coding-agent trajectories, counting every filter.")
    parser.add_argument("--version", action="version", version=f"trajdistill {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("select", help="clean and filter trajectories, write the survivors")
    _add_input(s)
    s.add_argument("-o", "--output", required=True, help="output JSONL of trajectories")
    s.add_argument("--report", help="also write the full funnel report as JSON to this path")
    _add_cleaning(s)
    _add_filters(s)
    _add_tokens(s)

    f = sub.add_parser("funnel", help="print what cleaning and each filter would change")
    _add_input(f)
    f.add_argument("--json", action="store_true", help="print the report as JSON")
    _add_cleaning(f)
    _add_filters(f)
    _add_tokens(f)

    c = sub.add_parser("convert", help="turn trajectories into chat training samples")
    _add_input(c)
    c.add_argument("-o", "--output", required=True, help="output JSONL of samples")
    c.add_argument("--to", choices=STYLES, default="openai", help="sample style (default: openai)")
    c.add_argument("--granularity", choices=GRANULARITIES, default="trajectory",
                   help="one sample per trajectory, or one per step (default: trajectory)")
    c.add_argument("--context-budget", type=int, default=None, metavar="TOKENS",
                   help="step samples only: drop the oldest steps until the sample fits")
    c.add_argument("--bare", action="store_true",
                   help="write only messages/conversations, without id, instance and repo")
    _add_cleaning(c)
    _add_tokens(c)

    p = sub.add_parser("split", help="split a JSONL of trajectories or samples into train/val")
    p.add_argument("input", help="JSONL (optionally .gz) whose records carry instance_id")
    p.add_argument("--train", required=True, help="output JSONL for training")
    p.add_argument("--val", required=True, help="output JSONL for validation")
    p.add_argument("--by", choices=SPLIT_BY, default="instance",
                   help="keep each instance or each repository on one side (default: instance)")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    return parser


def _warn_skipped(rs: ReadStats) -> None:
    if rs.skipped_unencodable:
        print(f"trajdistill: warning: skipped {rs.skipped_unencodable} of {rs.rows} records "
              "holding text that cannot be encoded as UTF-8", file=sys.stderr)


def _cmd_select(a: argparse.Namespace) -> int:
    rs = ReadStats()
    rep = run_pipeline(read_path(a.input, a.format, rs), _clean_config(a), _select_config(a),
                       _counter(a))
    _warn_skipped(rs)
    if a.command == "funnel":
        print(json.dumps(rep.to_dict(), indent=2) if a.json else report_text(rep))
        return 0
    write_native(rep.kept, a.output)
    if a.report:
        Path(a.report).write_text(json.dumps(rep.to_dict(), indent=2) + "\n", encoding="utf-8")
    print(report_text(rep), file=sys.stderr)
    return 0


def _cmd_convert(a: argparse.Namespace) -> int:
    rs = ReadStats()
    trajs, cstats = clean_all(read_path(a.input, a.format, rs), _clean_config(a))
    _warn_skipped(rs)
    stats = ConvertStats()
    write_jsonl(build_samples(trajs, style=a.to, granularity=a.granularity,
                              context_budget=a.context_budget, counter=_counter(a),
                              bare=a.bare, stats=stats), a.output)
    out = stats.to_dict()
    changed = {k: v for k, v in cstats.to_dict().items() if v}
    if changed:
        out["cleaning"] = changed
    print(json.dumps(out), file=sys.stderr)
    return 0


def _cmd_split(a: argparse.Namespace) -> int:
    rs = ReadStats()
    records = list(iter_jsonl(a.input, rs))
    _warn_skipped(rs)
    train, val, info = split_records(records, by=a.by, val_fraction=a.val_fraction,
                                     seed=a.seed)
    write_jsonl(train, a.train)
    write_jsonl(val, a.val)
    print(json.dumps(info.to_dict()), file=sys.stderr)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {"select": _cmd_select, "funnel": _cmd_select, "convert": _cmd_convert,
                "split": _cmd_split}
    try:
        return handlers[args.command](args)
    except (FormatError, ValueError, OSError, RuntimeError) as exc:
        print(f"trajdistill: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
