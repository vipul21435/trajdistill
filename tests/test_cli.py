from __future__ import annotations

import json
from pathlib import Path

import pytest

from trajdistill.cli import main
from trajdistill.funnel import funnel_table
from trajdistill.readers import read_path, write_native
from trajdistill.select import SelectConfig, select

from .helpers import PATCH_TEST, make_traj


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
    assert report["rows_in"] == 4 and report["rows_out"] == 1
    assert report["rows_in"] - sum(s["removed"] for s in report["stages"]) == 1
    assert "rows in: 4" in capsys.readouterr().err


def test_funnel_json_and_text(sample: Path, capsys) -> None:
    assert main(["funnel", str(sample), "--outcome", "any", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["rows_out"] == 4 and data["flagged_test_patches"] == 1
    assert main(["funnel", str(sample)]) == 0
    text = capsys.readouterr().out
    assert "outcome" in text and "rows out: 3" in text


def test_errors_exit_2(tmp_path: Path, capsys) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{nope\n")
    assert main(["funnel", str(bad)]) == 2
    assert "invalid JSON" in capsys.readouterr().err
    assert main(["funnel", str(tmp_path / "missing.jsonl")]) == 2
    assert main(["funnel", str(bad), "--max-steps", "-3"]) == 2


def test_funnel_table_marks_disabled_filters() -> None:
    sel = select([make_traj()], SelectConfig(outcome="any"))
    lines = funnel_table(sel).splitlines()
    outcome_line = next(line for line in lines if line.startswith("outcome"))
    assert " off " in outcome_line


def test_example_file_runs(capsys) -> None:
    example = Path(__file__).resolve().parents[1] / "examples" / "sample.jsonl"
    assert main(["funnel", str(example), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["rows_in"] == 21
