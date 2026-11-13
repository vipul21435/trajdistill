from __future__ import annotations

import random

import pytest

from trajdistill.split import split_records


def _records(rng: random.Random, n: int) -> list[dict]:
    return [{"id": f"r{k}", "instance_id": f"o{rng.randint(0, 6)}__p-{rng.randint(0, 9)}",
             "x": rng.random()} for k in range(n)]


@pytest.mark.parametrize("by", ["instance", "repo"])
def test_split_property_no_group_on_both_sides_and_order_free(by: str) -> None:
    rng = random.Random(4)
    for _ in range(60):
        recs = _records(rng, rng.randint(0, 80))
        frac = rng.choice([0.0, 0.1, 0.3, 0.5, 1.0])
        seed = rng.randint(0, 5)
        train, val, info = split_records(recs, by=by, val_fraction=frac, seed=seed)
        assert len(train) + len(val) == len(recs)
        inst_t = {r["instance_id"] for r in train}
        inst_v = {r["instance_id"] for r in val}
        assert not inst_t & inst_v and info.shared_instances == 0
        if by == "repo":
            assert not {r["instance_id"].split("-")[0] for r in train} & \
                {r["instance_id"].split("-")[0] for r in val}
        if frac == 0.0:
            assert val == []
        shuffled = recs[:]
        rng.shuffle(shuffled)
        t2, v2, _ = split_records(shuffled, by=by, val_fraction=frac, seed=seed)
        assert (t2, v2) == (train, val)


def test_split_hits_target_within_one_group() -> None:
    recs = [{"instance_id": f"o__p-{i}"} for i in range(100)]
    _, val, info = split_records(recs, val_fraction=0.2, seed=1)
    assert len(val) == 20 and info.val_fraction == 0.2
    with pytest.raises(ValueError):
        split_records(recs, by="model")
    with pytest.raises(ValueError):
        split_records(recs, val_fraction=1.5)


def test_rows_of_one_instance_with_different_repo_values_stay_together() -> None:
    recs = [{"instance_id": "django-101", "repo": "django/django"}, {"instance_id": "django-101"},
            {"instance_id": "flask-7"}, {"instance_id": "numpy-9"}]
    for seed in range(10):
        train, val, info = split_records(recs, by="repo", val_fraction=0.3, seed=seed)
        assert info.shared_instances == 0
        sides = {r["instance_id"] for r in train} & {r["instance_id"] for r in val}
        assert not sides


def test_one_large_group_cannot_take_validation() -> None:
    recs = [{"instance_id": "big__r-1", "k": i} for i in range(90)]
    recs += [{"instance_id": f"small__r-{i}"} for i in range(10)]
    for seed in range(20):
        _, val, info = split_records(recs, val_fraction=0.1, seed=seed)
        assert len(val) == 10 and not info.warnings


def test_empty_side_and_miss_are_warned() -> None:
    recs = [{"instance_id": "o__r-1", "k": i} for i in range(10)]
    train, val, info = split_records(recs, val_fraction=0.1)
    assert (len(train), len(val)) == (10, 0)
    assert info.warnings and "validation is empty" in info.warnings[0]
    uneven = [{"instance_id": "a__r-1", "k": i} for i in range(6)]
    uneven += [{"instance_id": "b__r-1", "k": i} for i in range(4)]
    _, val, info = split_records(uneven, val_fraction=0.25)
    assert len(val) == 4 and "against a target of 25.0%" in info.warnings[0]
    assert split_records(recs, val_fraction=0.0)[2].warnings == []
