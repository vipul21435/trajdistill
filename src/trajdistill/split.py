"""Train and validation splits grouped by instance or repository.

Whole groups go to one side, so no instance (and, with ``by="repo"``, no repository) is on
both. Groups are visited in a seeded pseudo-random order and added to validation while it
holds fewer rows than ``val_fraction`` of the total, so validation can overshoot the target by
at most one group. Records are sorted before splitting, so the result does not depend on
input order.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .model import repo_from_instance
from .select import seeded_rank

SPLIT_BY = ("instance", "repo")


def group_key(rec: dict[str, Any], by: str) -> str:
    inst = rec.get("instance_id")
    if not inst:
        raise ValueError("record has no instance_id (write samples without --bare to split)")
    if by == "instance":
        return str(inst)
    return str(rec.get("repo") or repo_from_instance(str(inst)))


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

    @property
    def val_fraction(self) -> float:
        total = self.rows_train + self.rows_val
        return self.rows_val / total if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["val_fraction"] = round(self.val_fraction, 4)
        return d


def split_records(records: Iterable[dict[str, Any]], by: str = "instance",
                  val_fraction: float = 0.1, seed: int = 0
                  ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], SplitInfo]:
    if by not in SPLIT_BY:
        raise ValueError(f"by must be one of {', '.join(SPLIT_BY)}")
    if not 0.0 <= val_fraction <= 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    keyed = sorted(((group_key(r, by), json.dumps(r, sort_keys=True), r) for r in records),
                   key=lambda x: (x[0], x[1]))
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key, _, rec in keyed:
        groups[key].append(rec)
    total = len(keyed)
    target = val_fraction * total
    val_keys: set[str] = set()
    val_rows = 0
    for key in sorted(groups, key=lambda k: (seeded_rank(seed, k), k)):
        if val_rows >= target or val_fraction == 0.0:
            break
        val_keys.add(key)
        val_rows += len(groups[key])
    train = [r for k, _, r in keyed if k not in val_keys]
    val = [r for k, _, r in keyed if k in val_keys]
    inst_train = {str(r["instance_id"]) for r in train}
    inst_val = {str(r["instance_id"]) for r in val}
    info = SplitInfo(by=by, rows_train=len(train), rows_val=len(val),
                     groups_train=len(groups) - len(val_keys), groups_val=len(val_keys),
                     instances_train=len(inst_train), instances_val=len(inst_val),
                     shared_instances=len(inst_train & inst_val),
                     target_val_fraction=val_fraction)
    return train, val, info
