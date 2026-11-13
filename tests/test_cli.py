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
    info = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
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


def test_pipeline_example_runs(capsys) -> None:
    import runpy

    runpy.run_path(str(EXAMPLE.with_name("pipeline.py")), run_name="__main__")
    out = capsys.readouterr().out
    assert "rows in: 21" in out and "'shared_instances': 0" in out


@pytest.mark.parametrize(
    "content,args,needle",
    [
        ('{"id": "x", "steps": [1, 2]}\n', [], "in.jsonl:1: cannot read record"),
        ('{"steps": []}\n', ["--format", "native"], "in.jsonl:1: cannot read record"),
        ('{"messages": [{"role": "assistant", "content": "", "tool_calls": 5}]}\n', [],
         "in.jsonl:1: cannot read record"),
        ('{"a": 1}\n{"foo": 2}\n', [], "in.jsonl:1: cannot tell the format"),
        ('[{"messages": "x"}]', ["--format", "messages"], "in.jsonl item 1"),
    ],
)
def test_malformed_records_give_an_error_line(tmp_path: Path, capsys, content, args,
                                              needle) -> None:
    p = tmp_path / "in.jsonl"
    p.write_text(content, encoding="utf-8")
    assert main(["funnel", str(p), *args]) == 2
    err = capsys.readouterr().err
    assert err.startswith("trajdistill: error:") and needle in err and "Traceback" not in err


def test_traj_file_holding_a_list_and_empty_directory(tmp_path: Path, capsys) -> None:
    (tmp_path / "x.traj").write_text("[1, 2]", encoding="utf-8")
    assert main(["funnel", str(tmp_path / "x.traj"), "--outcome", "any"]) == 2
    assert "x.traj: cannot read record" in capsys.readouterr().err
    only_jsonl = tmp_path / "d"
    only_jsonl.mkdir()
    (only_jsonl / "a.jsonl").write_text("{}\n", encoding="utf-8")
    assert main(["funnel", str(only_jsonl)]) == 2
    assert "no .traj files" in capsys.readouterr().err


def test_bad_failure_pattern_is_an_error(sample: Path, capsys) -> None:
    assert main(["funnel", str(sample), "--drop-failed", "--failure-pattern", "("]) == 2
    assert "bad failure pattern" in capsys.readouterr().err


def test_unencodable_rows_are_skipped_and_counted(tmp_path: Path, capsys) -> None:
    good = make_traj(tid="g").to_dict()
    p = tmp_path / "in.jsonl"
    bad = json.dumps(make_traj(tid="b", actions=["x"]).to_dict()).replace("x", "\\ud800", 1)
    p.write_text(json.dumps(good) + "\n" + bad + "\n", encoding="utf-8")
    out = tmp_path / "out.jsonl"
    assert main(["select", str(p), "-o", str(out)]) == 0
    assert [t.id for t in read_path(out)] == ["g"]
    assert "skipped 1 of 2 records" in capsys.readouterr().err
    samples = tmp_path / "s.jsonl"
    assert main(["convert", str(p), "-o", str(samples)]) == 0
    assert "skipped 1 of 2" in capsys.readouterr().err
    split_in = tmp_path / "split.jsonl"
    split_in.write_text('{"instance_id": "a-1"}\n{"instance_id": "\\ud800"}\n', encoding="utf-8")
    assert main(["split", str(split_in), "--train", str(tmp_path / "t"), "--val",
                 str(tmp_path / "v")]) == 0
    assert "skipped 1 of 2" in capsys.readouterr().err


def test_split_prints_warnings(tmp_path: Path, capsys) -> None:
    src = tmp_path / "one.jsonl"
    src.write_text("".join(json.dumps({"instance_id": "o__r-1", "k": i}) + "\n"
                           for i in range(10)), encoding="utf-8")
    assert main(["split", str(src), "--train", str(tmp_path / "t"), "--val",
                 str(tmp_path / "v")]) == 0
    err = capsys.readouterr().err
    assert "trajdistill: warning: validation is empty" in err
