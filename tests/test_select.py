from __future__ import annotations

import random

import pytest

from trajdistill.model import Step
from trajdistill.select import SelectConfig, find_loop, seeded_rank, select

from .helpers import PATCH_TEST, make_traj

ANY = dict(outcome="any", require_patch=False)


def stage(sel, name):
    return next(s for s in sel.stages if s.name == name)


def ids(sel):
    return [t.id for t in sel.kept]


def test_defaults_keep_resolved_with_patch() -> None:
    rows = [make_traj(tid="a"), make_traj(tid="b", resolved=False), make_traj(tid="c", patch=" "),
            make_traj(tid="d", resolved=None)]
    sel = select(rows)
    assert ids(sel) == ["a"]
    assert stage(sel, "outcome").removed == 2
    assert stage(sel, "empty_patch").removed == 1


def test_duplicates_removed_by_content() -> None:
    a = make_traj(tid="x1")
    b = make_traj(tid="x2")  # same content, different id
    c = make_traj(tid="x3", actions=["pwd", "submit"])
    sel = select([a, b, c], SelectConfig(**ANY))
    assert stage(sel, "duplicates").removed == 1
    assert ids(sel) == ["x1", "x3"]
    sel_off = select([a, b, c], SelectConfig(dedupe=False, **ANY))
    assert sel_off.rows_out == 3


def test_exit_status_patterns() -> None:
    rows = [make_traj(tid="a", exit_status="submitted"),
            make_traj(tid="b", exit_status="submitted (exit_context)", actions=["x"]),
            make_traj(tid="c", exit_status="exit_cost", actions=["y"])]
    assert ids(select(rows, SelectConfig(exit_status=("submitted",), **ANY))) == ["a"]
    assert ids(select(rows, SelectConfig(exit_status=("submitted*",), **ANY))) == ["a", "b"]


def test_test_patches_flagged_and_optionally_dropped() -> None:
    rows = [make_traj(tid="a"), make_traj(tid="b", patch=PATCH_TEST, actions=["q"])]
    keep = select(rows, SelectConfig(**ANY))
    assert keep.flagged_test_patches == 1 and keep.rows_out == 2
    assert {t.id: t.meta["touches_tests"] for t in keep.kept} == {"a": False, "b": True}
    drop = select(rows, SelectConfig(test_patches="drop", **ANY))
    assert ids(drop) == ["a"] and stage(drop, "test_patches").removed == 1


def test_find_loop_keys_and_window() -> None:
    steps = [Step(content="```\nls\n```", observations=["a"]),
             Step(content="```\nls \n```", observations=["b"]),
             Step(content="```\nls\n```", observations=["b"])]
    assert find_loop(steps) == 2
    assert find_loop(steps, loop_key="action") == 1
    abab = [Step(action=a, observations=["o"]) for a in ["a", "b", "a", "b"]]
    assert find_loop(abab) is None
    assert find_loop(abab, window=2) == 2


def test_loops_drop_and_cut() -> None:
    looping = make_traj(tid="L", actions=["ls", "pytest", "pytest", "submit"],
                        observations=["a", "F", "F", "ok"])
    clean = make_traj(tid="C")
    drop = select([looping, clean], SelectConfig(loops="drop", **ANY))
    assert ids(drop) == ["C"] and stage(drop, "loops").removed == 1
    cut = select([looping, clean], SelectConfig(loops="cut", **ANY))
    assert stage(cut, "loops").changed == 1 and stage(cut, "loops").removed == 0
    kept = {t.id: t for t in cut.kept}
    assert [s.action for s in kept["L"].steps] == ["ls", "pytest"]
    assert kept["L"].meta["loop_cut_at"] == 2
    assert looping.n_steps == 4  # the caller's object is untouched


def test_max_steps_and_max_tokens() -> None:
    short = make_traj(tid="s", actions=["a"])
    long = make_traj(tid="l", actions=[f"cmd {i}" for i in range(10)])
    sel = select([short, long], SelectConfig(max_steps=5, **ANY))
    assert ids(sel) == ["s"]
    sel = select([short, long], SelectConfig(max_tokens=40, **ANY))
    assert ids(sel) == ["s"] and stage(sel, "max_tokens").removed == 1


def test_per_instance_modes() -> None:
    rows = [make_traj(tid="r3", actions=["a", "b", "c"]),
            make_traj(tid="r1", actions=["a"], resolved=False),
            make_traj(tid="r2", actions=["a", "b"]),
            make_traj("acme__widgets-2", tid="other")]
    assert ids(select(rows, SelectConfig(per_instance="shortest", **ANY))) == ["r1", "other"]
    assert ids(select(rows, SelectConfig(per_instance="best", **ANY))) == ["r2", "other"]
    two = select(rows, SelectConfig(per_instance="best", per_instance_k=2, **ANY))
    assert sorted(ids(two)) == ["other", "r2", "r3"]
    rnd = select(rows, SelectConfig(per_instance="random", seed=3, **ANY))
    assert rnd.rows_out == 2


def test_per_repo_cap_prefers_distinct_instances() -> None:
    rows = [make_traj(f"big__repo-{i}", tid=f"b{i}-{j}", actions=[f"x{j}"])
            for i in range(3) for j in range(3)]
    rows.append(make_traj("small__repo-1", tid="s"))
    sel = select(rows, SelectConfig(max_per_repo=3, **ANY))
    kept_big = [t for t in sel.kept if t.repo == "big/repo"]
    assert len(kept_big) == 3
    assert len({t.instance_id for t in kept_big}) == 3
    assert "s" in ids(sel)
    assert stage(sel, "per_repo").removed == 6


def test_disabled_filters_are_reported_as_off() -> None:
    sel = select([make_traj()], SelectConfig(**ANY))
    off = {s.name for s in sel.stages if not s.enabled}
    assert {"outcome", "exit_status", "empty_patch", "loops", "max_steps", "max_tokens",
            "per_instance", "per_repo", "test_patches"} <= off
    assert all(s.removed == 0 for s in sel.stages if not s.enabled)


@pytest.mark.parametrize(
    "kwargs",
    [dict(outcome="won"), dict(loops="maybe"), dict(max_steps=-1), dict(per_instance_k=0),
     dict(loop_window=0), dict(per_instance="worst"), dict(test_patches="x")],
)
def test_config_validation(kwargs) -> None:
    with pytest.raises(ValueError):
        SelectConfig(**kwargs)


def _random_rows(rng: random.Random, n: int) -> list:
    rows = []
    for k in range(n):
        acts = [rng.choice(["ls", "pytest", "cat a", "edit"]) for _ in range(rng.randint(1, 8))]
        obs = [rng.choice(["ok", "F", ""]) for _ in acts]
        rows.append(make_traj(
            f"{rng.choice(['o1__r1', 'o2__r2', 'o3__r3'])}-{rng.randint(1, 4)}",
            actions=acts, observations=obs, tid=f"t{k}",
            resolved=rng.choice([True, False, None]),
            exit_status=rng.choice(["submitted", "exit_cost"]),
            patch=rng.choice(["", PATCH_TEST, "diff --git a/m.py b/m.py\n"])))
    return rows


def _random_config(rng: random.Random) -> SelectConfig:
    return SelectConfig(
        outcome=rng.choice(["any", "resolved"]),
        exit_status=rng.choice([(), ("submitted",)]),
        require_patch=rng.random() < 0.5,
        test_patches=rng.choice(["keep", "drop"]),
        loops=rng.choice(["keep", "drop", "cut"]),
        max_steps=rng.choice([None, 4]),
        max_tokens=rng.choice([None, 80]),
        per_instance=rng.choice(["all", "shortest", "best", "random"]),
        per_instance_k=rng.randint(1, 2),
        max_per_repo=rng.choice([None, 2, 5]),
        seed=rng.randint(0, 9),
    )


def test_property_counts_add_up_and_order_does_not_matter() -> None:
    rng = random.Random(11)
    for _ in range(60):
        rows = _random_rows(rng, rng.randint(0, 25))
        cfg = _random_config(rng)
        sel = select(rows, cfg)
        assert sel.rows_in == len(rows)
        assert sel.rows_in - sum(s.removed for s in sel.stages) == sel.rows_out
        for a, b in zip(sel.stages, sel.stages[1:], strict=False):
            assert a.rows_after == b.rows_before
        shuffled = rows[:]
        rng.shuffle(shuffled)
        again = select(shuffled, cfg)
        assert [t.to_dict() for t in again.kept] == [t.to_dict() for t in sel.kept]
        assert again.to_dict() == sel.to_dict()


def test_seeded_rank_is_stable_and_seed_dependent() -> None:
    assert seeded_rank(0, "a") == seeded_rank(0, "a")
    assert seeded_rank(0, "a") != seeded_rank(1, "a")
