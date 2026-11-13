"""Command line: ``trajdistill select`` and ``trajdistill funnel``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .funnel import funnel_table
from .readers import FORMATS, FormatError, read_path, write_native
from .select import (
    LOOP_KEYS,
    LOOP_MODES,
    OUTCOMES,
    PER_INSTANCE_MODES,
    TEST_PATCH_MODES,
    SelectConfig,
    Selection,
    select,
)
from .tokens import DEFAULT_CHARS_PER_TOKEN, TokenCounter


def _add_input(p: argparse.ArgumentParser) -> None:
    p.add_argument("input", help="a JSONL/JSON file, a .traj file, or a directory of .traj files")
    p.add_argument("--format", default="auto", choices=FORMATS, help="input format (default: auto)")


def _add_filters(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("filters (applied in this order)")
    g.add_argument("--no-dedupe", action="store_true", help="keep exact duplicate trajectories")
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
    t = p.add_argument_group("token counting")
    t.add_argument("--tokenizer", choices=("chars", "tiktoken"), default="chars")
    t.add_argument("--chars-per-token", type=float, default=DEFAULT_CHARS_PER_TOKEN)


def _config(a: argparse.Namespace) -> SelectConfig:
    return SelectConfig(
        dedupe=not a.no_dedupe,
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


def _run_select(a: argparse.Namespace) -> Selection:
    counter = TokenCounter(a.tokenizer, chars_per_token=a.chars_per_token)
    return select(read_path(a.input, a.format), _config(a), counter)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trajdistill",
        description="Select and convert coding-agent trajectories, counting every filter.")
    parser.add_argument("--version", action="version", version=f"trajdistill {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("select", help="filter trajectories and write the survivors as JSONL")
    _add_input(s)
    s.add_argument("-o", "--output", required=True, help="output JSONL (native format)")
    s.add_argument("--report", help="also write the funnel as JSON to this path")
    _add_filters(s)

    f = sub.add_parser("funnel", help="print how many trajectories each filter removes")
    _add_input(f)
    f.add_argument("--json", action="store_true", help="print the funnel as JSON")
    _add_filters(f)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        sel = _run_select(args)
    except (FormatError, ValueError, OSError, RuntimeError) as exc:
        print(f"trajdistill: error: {exc}", file=sys.stderr)
        return 2
    if args.command == "select":
        write_native(sel.kept, args.output)
        if args.report:
            Path(args.report).write_text(json.dumps(sel.to_dict(), indent=2) + "\n",
                                         encoding="utf-8")
        print(funnel_table(sel), file=sys.stderr)
        return 0
    if args.json:
        print(json.dumps(sel.to_dict(), indent=2))
    else:
        print(funnel_table(sel))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
