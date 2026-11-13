"""Train and validation splits grouped by instance or repository.

Whole groups go to one side, so no instance (and, with ``by="repo"``, no repository) is on
both. With ``by="repo"`` every instance is first given one repository, the most common
``repo`` value on its rows (else the one derived from its id), so rows of one instance that
disagree cannot end up apart. Groups are visited in a seeded pseudo-random order and a group
joins validation when that brings validation closer to ``val_fraction`` of the rows. A
warning is recorded when a side is empty or the share misses the target by more than 10
points. Records are sorted before splitting, so the result does not depend on input order.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .model import repo_from_instance
from .select import seeded_rank

SPLIT_BY = ("instance", "repo")


def instance_of(rec: dict[str, Any]) -> str:
    inst = rec.get("instance_id")
    if not inst:
        raise ValueError("record has no instance_id (write samples without --bare to split)")
    return str(inst)


def instance_repos(records: list[dict[str, Any]]) -> dict[str, str]:
    """One repository per instance: its most common ``repo`` value (ties by name), else the
    repository derived from the instance id."""
    votes: dict[str, Counter[str]] = defaultdict(Counter)
    for r in records:
        inst = instance_of(r)
        votes[inst]  # noqa: B018 - make sure every instance has an entry
        if r.get("repo"):
            votes[inst][str(r["repo"])] += 1
    out = {}
    for inst, c in votes.items():
        out[inst] = min(c.items(), key=lambda kv: (-kv[1], kv[0]))[0] if c else \
            repo_from_instance(inst)
    return out


@dataclass
class SplitInfo:
    by: str
    rows_train: int
    rows_val: int
    groups_train: int
    groups_val: int
    instances_train: int
    instances_val: int
    shared_instances: int
    target_val_fraction: float
    warnings: list[str] = field(default_factory=list)

    @property
    def val_fraction(self) -> float:
        total = self.rows_train + self.rows_val
        return self.rows_val / total if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["warnings"] = list(self.warnings)
        d["val_fraction"] = round(self.val_fraction, 4)
        return d


def split_records(records: Iterable[dict[str, Any]], by: str = "instance",
                  val_fraction: float = 0.1, seed: int = 0
                  ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], SplitInfo]:
    if by not in SPLIT_BY:
        raise ValueError(f"by must be one of {', '.join(SPLIT_BY)}")
    if not 0.0 <= val_fraction <= 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    records = list(records)
    repos = instance_repos(records) if by == "repo" else {}

    def key_of(r: dict[str, Any]) -> str:
        inst = instance_of(r)
        return repos[inst] if by == "repo" else inst

    keyed = sorted(((key_of(r), json.dumps(r, sort_keys=True), r) for r in records),
                   key=lambda x: (x[0], x[1]))
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key, _, rec in keyed:
        groups[key].append(rec)
    total = len(keyed)
    target = val_fraction * total
    val_keys: set[str] = set()
    val_rows = 0
    for key in sorted(groups, key=lambda k: (seeded_rank(seed, k), k)):
        size = len(groups[key])
        if abs(val_rows + size - target) < abs(val_rows - target):
            val_keys.add(key)
            val_rows += size
    train = [r for k, _, r in keyed if k not in val_keys]
    val = [r for k, _, r in keyed if k in val_keys]
    inst_train = {str(r["instance_id"]) for r in train}
    inst_val = {str(r["instance_id"]) for r in val}
    info = SplitInfo(by=by, rows_train=len(train), rows_val=len(val),
                     groups_train=len(groups) - len(val_keys), groups_val=len(val_keys),
                     instances_train=len(inst_train), instances_val=len(inst_val),
                     shared_instances=len(inst_train & inst_val),
                     target_val_fraction=val_fraction)
    if total and 0.0 < val_fraction < 1.0:
        if not val or not train:
            side = "validation" if not val else "training"
            info.warnings.append(f"{side} is empty: the {by} groups are too large for a "
                                 f"{val_fraction:g} split of {total} rows")
        elif abs(info.val_fraction - val_fraction) > 0.1:
            info.warnings.append(f"validation holds {info.val_fraction:.1%} of rows against a "
                                 f"target of {val_fraction:.1%}: the {by} groups are uneven")
    return train, val, info
