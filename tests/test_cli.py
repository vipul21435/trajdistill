from __future__ import annotations

import json
from pathlib import Path

import pytest

from trajdistill.cli import main
from trajdistill.funnel import funnel_table
from trajdistill.readers import iter_jsonl, read_path, write_native
from trajdistill.select import SelectConfig, select

from .helpers import PATCH_TEST, make_traj

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "sample.jsonl"


@pytest.fixture()
def sample(tmp_path: Path) -> Path:
    rows = [make_traj(tid="a"), make_traj(tid="b", resolved=False),
            make_traj(tid="c", patch=PATCH_TEST, actions=["x"]),
            make_traj("acme__widgets-2", tid="d", actions=["a", "b", "c", "d", "e"])]
    p = tmp_path / "in.jsonl"
    write_native(rows, p)
    return p


def test_select_writes_survivors_and_report(sample: Path, tmp_path: Path, capsys) -> None:
    out, rep = tmp_path / "out.jsonl", tmp_path / "rep.json"
    code = main(["select", str(sample), "-o", str(out), "--report", str(rep),
                 "--test-patches", "drop", "--max-steps", "4"])
    assert code == 0
    assert [t.id for t in read_path(out)] == ["a"]
    report = json.loads(rep.read_text())
    f = report["funnel"]
    assert f["rows_in"] == 4 and f["rows_out"] == 1
    assert f["rows_in"] - sum(s["removed"] for s in f["stages"]) == 1
    assert report["before"]["rows"] == 4 and report["after"]["rows"] == 1
    err = capsys.readouterr().err
    assert "rows in: 4" in err and "concentration (HHI)" in err


def test_funnel_json_and_text(sample: Path, capsys) -> None:
    assert main(["funnel", str(sample), "--outcome", "any", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["funnel"]["rows_out"] == 4 and data["funnel"]["flagged_test_patches"] == 1
    assert main(["funnel", str(sample), "--clean", "--max-observation-chars", "3"]) == 0
    text = capsys.readouterr().out
    assert "outcome" in text and "rows out: 3" in text
    assert "cleaning: trajectories_changed" in text and "observations_shortened" in text


def test_convert_trajectory_and_step(sample: Path, tmp_path: Path, capsys) -> None:
    out = tmp_path / "s.jsonl"
    assert main(["convert", str(sample), "-o", str(out)]) == 0
    recs = list(iter_jsonl(out))
    assert len(recs) == 4 and all("messages" in r for r in recs)
    stats = json.loads(capsys.readouterr().err)
    assert stats["samples"] == 4
    assert main(["convert", str(sample), "-o", str(out), "--to", "sharegpt", "--granularity",
                 "step", "--context-budget", "60", "--bare", "--redact"]) == 0
    recs = list(iter_jsonl(out))
    assert recs and all(list(r) == ["conversations"] for r in recs)
    stats = json.loads(capsys.readouterr().err)
    assert stats["samples"] + stats["skipped_over_budget"] == 3 + 3 + 1 + 5


def test_split_command(tmp_path: Path, capsys) -> None:
    rows = [make_traj(f"o{i % 3}__r-{i}", tid=f"t{i}{j}", actions=[f"a{j}"])
            for i in range(10) for j in range(2)]
    src = tmp_path / "in.jsonl"
    write_native(rows, src)
    tr, va = tmp_path / "train.jsonl", tmp_path / "val.jsonl"
    assert main(["split", str(src), "--train", str(tr), "--val", str(va), "--val-fraction",
                 "0.3", "--by", "repo"]) == 0
    info = json.loads(capsys.readouterr().err)
    assert info["rows_train"] + info["rows_val"] == 20 and info["shared_instances"] == 0
    train_repos = {r["repo"] for r in iter_jsonl(tr)}
    val_repos = {r["repo"] for r in iter_jsonl(va)}
    assert val_repos and not train_repos & val_repos


def test_errors_exit_2(tmp_path: Path, capsys) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{nope\n")
    assert main(["funnel", str(bad)]) == 2
    assert "invalid JSON" in capsys.readouterr().err
    assert main(["funnel", str(tmp_path / "missing.jsonl")]) == 2
    assert main(["funnel", str(bad), "--max-steps", "-3"]) == 2
    bare = tmp_path / "bare.jsonl"
    bare.write_text('{"messages": []}\n')
    assert main(["split", str(bare), "--train", str(tmp_path / "a"), "--val",
                 str(tmp_path / "b")]) == 2
    assert "instance_id" in capsys.readouterr().err


def test_funnel_table_marks_disabled_filters() -> None:
    sel = select([make_traj()], SelectConfig(outcome="any"))
    lines = funnel_table(sel).splitlines()
    outcome_line = next(line for line in lines if line.startswith("outcome"))
    assert " off " in outcome_line


def test_example_file_runs(capsys) -> None:
    assert main(["funnel", str(EXAMPLE), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["funnel"]["rows_in"] == 21
    assert data["before"]["exact_duplicates"] == 1
