"""Selection filters, applied in a fixed order, each switchable and counted.

Order: duplicates, outcome, exit status, empty patch, test-file patches, loops, maximum
steps, maximum tokens, trajectories per instance, cap per repository.

The input is sorted by ``(instance_id, id, content hash)`` first and every random choice
uses a seeded hash of the trajectory id, so the result does not depend on input order.
"""

from __future__ import annotations

import fnmatch
import hashlib
import re
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from .model import Step, Trajectory, content_hash
from .patches import touches_tests
from .tokens import TokenCounter

OUTCOMES = ("resolved", "any")
TEST_PATCH_MODES = ("keep", "drop")
LOOP_MODES = ("keep", "drop", "cut")
LOOP_KEYS = ("action+observation", "action")
PER_INSTANCE_MODES = ("all", "shortest", "best", "random")

_WS = re.compile(r"\s+")


@dataclass
class SelectConfig:
    """Settings for :func:`select`. ``None`` or the neutral choice turns a filter off."""

    dedupe: bool = True
    outcome: str = "resolved"
    exit_status: tuple[str, ...] = ()
    require_patch: bool = True
    test_patches: str = "keep"
    loops: str = "keep"
    loop_key: str = "action+observation"
    loop_window: int = 1
    max_steps: int | None = None
    max_tokens: int | None = None
    per_instance: str = "all"
    per_instance_k: int = 1
    max_per_repo: int | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        _check("outcome", self.outcome, OUTCOMES)
        _check("test_patches", self.test_patches, TEST_PATCH_MODES)
        _check("loops", self.loops, LOOP_MODES)
        _check("loop_key", self.loop_key, LOOP_KEYS)
        _check("per_instance", self.per_instance, PER_INSTANCE_MODES)
        for name in ("max_steps", "max_tokens", "max_per_repo"):
            v = getattr(self, name)
            if v is not None and v < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.per_instance_k < 1:
            raise ValueError("per_instance_k must be >= 1")
        if self.loop_window < 1:
            raise ValueError("loop_window must be >= 1")
        self.exit_status = tuple(self.exit_status)


def _check(name: str, value: str, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {', '.join(allowed)}; got {value!r}")


@dataclass
class StageCount:
    """What one filter did: rows it saw, rows it removed, rows it changed in place."""

    name: str
    setting: str
    enabled: bool
    rows_before: int
    removed: int = 0
    changed: int = 0

    @property
    def rows_after(self) -> int:
        return self.rows_before - self.removed


@dataclass
class Selection:
    kept: list[Trajectory]
    rows_in: int
    stages: list[StageCount]
    flagged_test_patches: int = 0
    config: SelectConfig = field(default_factory=SelectConfig)

    @property
    def rows_out(self) -> int:
        return len(self.kept)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "flagged_test_patches": self.flagged_test_patches,
            "stages": [dict(asdict(s), rows_after=s.rows_after) for s in self.stages],
            "config": asdict(self.config),
        }


def seeded_rank(seed: int, key: str) -> int:
    """A deterministic pseudo-random rank for ``key`` under ``seed``."""
    return int.from_bytes(hashlib.sha256(f"{seed}\x00{key}".encode()).digest()[:8], "big")


def _norm(text: str) -> str:
    return _WS.sub(" ", text).strip()


def _step_key(step: Step, loop_key: str) -> tuple[str, ...]:
    if loop_key == "action":
        return (_norm(step.action),)
    return (_norm(step.action), step.observation)


def find_loop(steps: list[Step], loop_key: str = "action+observation", window: int = 1
              ) -> int | None:
    """Index of the first step whose key equals one of the ``window`` steps before it."""
    keys = [_step_key(s, loop_key) for s in steps]
    for i in range(1, len(keys)):
        if keys[i] in keys[max(0, i - window):i]:
            return i
    return None


Filter = Callable[[list[Trajectory]], tuple[list[Trajectory], int]]


def select(trajs: Iterable[Trajectory], config: SelectConfig | None = None,
           counter: TokenCounter | None = None) -> Selection:
    """Run every filter in order and count what each one removed."""
    cfg = config or SelectConfig()
    counter = counter or TokenCounter()
    hashes: dict[int, str] = {}
    # shallow copies: filters reassign ``steps`` and write ``meta`` but never mutate a step
    rows = [replace(t, meta=dict(t.meta)) for t in trajs]
    for t in rows:
        hashes[id(t)] = content_hash(t)
    rows.sort(key=lambda t: (t.instance_id, t.id, hashes[id(t)]))
    rows_in = len(rows)

    token_cache: dict[int, int] = {}

    def tokens(t: Trajectory) -> int:
        if id(t) not in token_cache:
            token_cache[id(t)] = counter.count_trajectory(t)
        return token_cache[id(t)]

    flagged = 0
    for t in rows:
        t.meta["touches_tests"] = touches_tests(t.patch)
        flagged += t.meta["touches_tests"]

    def dedupe(ts: list[Trajectory]) -> tuple[list[Trajectory], int]:
        seen: set[str] = set()
        out = []
        for t in ts:
            h = hashes[id(t)]
            if h not in seen:
                seen.add(h)
                out.append(t)
        return out, 0

    def outcome(ts: list[Trajectory]) -> tuple[list[Trajectory], int]:
        return [t for t in ts if t.resolved is True], 0

    def exit_status(ts: list[Trajectory]) -> tuple[list[Trajectory], int]:
        return [t for t in ts
                if any(fnmatch.fnmatchcase(t.exit_status, p) for p in cfg.exit_status)], 0

    def patch(ts: list[Trajectory]) -> tuple[list[Trajectory], int]:
        return [t for t in ts if t.patch.strip()], 0

    def test_patches(ts: list[Trajectory]) -> tuple[list[Trajectory], int]:
        return [t for t in ts if not t.meta["touches_tests"]], 0

    def loops(ts: list[Trajectory]) -> tuple[list[Trajectory], int]:
        out, changed = [], 0
        for t in ts:
            i = find_loop(t.steps, cfg.loop_key, cfg.loop_window)
            if i is None:
                out.append(t)
            elif cfg.loops == "cut":
                t.steps = t.steps[:i]
                t.meta["loop_cut_at"] = i
                token_cache.pop(id(t), None)
                changed += 1
                out.append(t)
        return out, changed

    def max_steps(ts: list[Trajectory]) -> tuple[list[Trajectory], int]:
        return [t for t in ts if t.n_steps <= (cfg.max_steps or 0)], 0

    def max_tokens(ts: list[Trajectory]) -> tuple[list[Trajectory], int]:
        return [t for t in ts if tokens(t) <= (cfg.max_tokens or 0)], 0

    def per_instance(ts: list[Trajectory]) -> tuple[list[Trajectory], int]:
        groups: dict[str, list[Trajectory]] = defaultdict(list)
        for t in ts:
            groups[t.instance_id].append(t)
        keep: set[int] = set()
        for group in groups.values():
            group.sort(key=lambda t: _instance_rank(t, cfg, tokens))
            keep.update(id(t) for t in group[: cfg.per_instance_k])
        return [t for t in ts if id(t) in keep], 0

    def per_repo(ts: list[Trajectory]) -> tuple[list[Trajectory], int]:
        keep: set[int] = set()
        by_repo: dict[str, list[Trajectory]] = defaultdict(list)
        for t in ts:
            by_repo[t.repo].append(t)
        for group in by_repo.values():
            keep.update(id(t) for t in _spread_by_instance(group, cfg.seed)[: cfg.max_per_repo])
        return [t for t in ts if id(t) in keep], 0

    plan: list[tuple[str, str, bool, Filter]] = [
        ("duplicates", "exact content", cfg.dedupe, dedupe),
        ("outcome", cfg.outcome, cfg.outcome == "resolved", outcome),
        ("exit_status", ",".join(cfg.exit_status) or "any", bool(cfg.exit_status), exit_status),
        ("empty_patch", "required" if cfg.require_patch else "off", cfg.require_patch, patch),
        ("test_patches", cfg.test_patches, cfg.test_patches == "drop", test_patches),
        ("loops", f"{cfg.loops} ({cfg.loop_key}, window {cfg.loop_window})",
         cfg.loops != "keep", loops),
        ("max_steps", str(cfg.max_steps) if cfg.max_steps is not None else "off",
         cfg.max_steps is not None, max_steps),
        ("max_tokens", str(cfg.max_tokens) if cfg.max_tokens is not None else "off",
         cfg.max_tokens is not None, max_tokens),
        ("per_instance", cfg.per_instance if cfg.per_instance == "all"
         else f"{cfg.per_instance} k={cfg.per_instance_k}", cfg.per_instance != "all",
         per_instance),
        ("per_repo", str(cfg.max_per_repo) if cfg.max_per_repo is not None else "off",
         cfg.max_per_repo is not None, per_repo),
    ]

    stages = []
    for name, setting, enabled, fn in plan:
        stage = StageCount(name=name, setting=setting, enabled=enabled, rows_before=len(rows))
        if enabled:
            after, changed = fn(rows)
            stage.removed = len(rows) - len(after)
            stage.changed = changed
            rows = after
        stages.append(stage)
    return Selection(kept=rows, rows_in=rows_in, stages=stages, flagged_test_patches=flagged,
                     config=cfg)


def _instance_rank(t: Trajectory, cfg: SelectConfig, tokens: Callable[[Trajectory], int]
                   ) -> tuple[Any, ...]:
    tie = seeded_rank(cfg.seed, t.id)
    if cfg.per_instance == "shortest":
        return (t.n_steps, tokens(t), tie)
    if cfg.per_instance == "best":
        return (t.resolved is not True, t.n_steps, tokens(t), tie)
    return (tie,)


def _spread_by_instance(group: list[Trajectory], seed: int) -> list[Trajectory]:
    """Order a repository's trajectories so a cap keeps as many distinct instances as it can:
    every instance's first pick (in seeded order) comes before any instance's second pick."""
    by_inst: dict[str, list[Trajectory]] = defaultdict(list)
    for t in group:
        by_inst[t.instance_id].append(t)
    keyed = []
    for members in by_inst.values():
        members.sort(key=lambda t: seeded_rank(seed, t.id))
        for depth, t in enumerate(members):
            keyed.append(((depth, seeded_rank(seed, t.id)), t))
    keyed.sort(key=lambda kv: kv[0])
    return [t for _, t in keyed]
