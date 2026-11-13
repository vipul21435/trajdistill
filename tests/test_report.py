from __future__ import annotations

import pytest

from trajdistill.clean import CleanConfig
from trajdistill.funnel import composition, hhi, nearest_rank, run_pipeline
from trajdistill.select import SelectConfig
from trajdistill.tokens import TokenCounter

from .helpers import make_traj


def test_hhi_closed_forms() -> None:
    assert hhi([10]) == 1.0
    assert hhi([5, 5, 5, 5]) == pytest.approx(0.25)
    assert hhi([3, 1]) == pytest.approx(0.75 ** 2 + 0.25 ** 2)
    assert hhi([]) == 0.0


def test_nearest_rank() -> None:
    v = list(range(1, 11))
    assert nearest_rank(v, 0.9) == 9
    assert nearest_rank(v, 1.0) == 10
    assert nearest_rank(v, 0.0) == 1
    assert nearest_rank([], 0.5) == 0


def test_composition_fields() -> None:
    rows = [make_traj("big__a-1", tid="1"), make_traj("big__a-1", tid="2", actions=["x"]),
            make_traj("big__a-2", tid="3", resolved=False),
            make_traj("small__b-1", tid="4", resolved=None)]
    rows.append(make_traj("small__b-1", tid="5", resolved=None))  # duplicate content of "4"
    c = composition(rows, TokenCounter())
    assert c["rows"] == 5 and c["instances"] == 3 and c["repositories"] == 2
    assert c["top_repositories"][0] == {"repo": "big/a", "rows": 3, "share": 0.6}
    assert c["hhi"] == pytest.approx(0.6 ** 2 + 0.4 ** 2)
    assert c["effective_repositories"] == pytest.approx(1 / 0.52, abs=0.05)
    assert c["outcome"] == {"resolved": 2, "unresolved": 1, "unknown": 2}
    assert c["exact_duplicates"] == 1 and c["instances_with_several_runs"] == 2
    assert c["tokens"]["total"] == sum(TokenCounter().count_trajectory(t) for t in rows)
    empty = composition([], TokenCounter())
    assert empty["rows"] == 0 and empty["hhi"] == 0.0


def test_pipeline_counts_cleaning_and_shift() -> None:
    rows = [make_traj(f"big__a-{i}", tid=f"b{i}", observations=["x" * 300, "y", "z"])
            for i in range(8)]
    rows += [make_traj("small__b-1", tid="s")]
    rep = run_pipeline(rows, CleanConfig(max_observation_chars=100),
                       SelectConfig(max_per_repo=2))
    assert rep.cleaning.to_dict()["observations_shortened"] == 8
    assert rep.tokens_after_cleaning < rep.before["tokens"]["total"]
    assert rep.before["largest_repo_share"] == pytest.approx(8 / 9, abs=1e-4)
    assert rep.after["largest_repo_share"] == pytest.approx(2 / 3, abs=1e-4)
    assert rep.to_dict()["funnel"]["rows_out"] == 3
