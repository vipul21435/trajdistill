from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from trajdistill.model import Step, ToolCall, Trajectory, derive_action, repo_from_instance
from trajdistill.readers import (
    FormatError,
    detect_format,
    from_messages_row,
    from_nebius_row,
    read_path,
    text_of,
    write_native,
)

NEBIUS_ROW = {
    "instance_id": "acme__widgets-12",
    "model_name": "agent-7b",
    "target": True,
    "exit_status": "submitted",
    "generated_patch": "diff --git a/w.py b/w.py\n",
    "eval_logs": "",
    "trajectory": [
        {"role": "system", "system_prompt": "SETTING: shell", "text": None, "mask": False,
         "cutoff_date": "01.01.2023"},
        {"role": "user", "text": "ISSUE: widgets break", "system_prompt": None, "mask": False,
         "cutoff_date": None},
        {"role": "ai", "text": "Look first.\n\n```\nls -F\n```", "system_prompt": None,
         "mask": True, "cutoff_date": None},
        {"role": "user", "text": "w.py\n(Open file: n/a)", "system_prompt": None, "mask": False,
         "cutoff_date": None},
        {"role": "ai", "text": "Done.\n\n```\nsubmit\n```", "system_prompt": None, "mask": True,
         "cutoff_date": None},
    ],
}

MESSAGES_ROW = {
    "instance_id": "acme__widgets-3",
    "resolved": "true",
    "patch": "diff --git a/x b/x\n",
    "messages": [
        {"role": "system", "content": "You can call tools."},
        {"role": "user", "content": [{"type": "text", "text": "Fix the "}, {"type": "text",
                                                                          "text": "parser."}]},
        {"role": "assistant", "content": "Listing.", "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "bash", "arguments": {"cmd": "ls"}}},
            {"id": "call_2", "type": "function",
             "function": {"name": "bash", "arguments": "{\"cmd\": \"pwd\"}"}},
        ]},
        {"role": "tool", "tool_call_id": "call_1", "content": "a.py"},
        {"role": "tool", "tool_call_id": "call_2", "content": "/repo"},
        {"role": "assistant", "content": "Finished."},
    ],
}


def test_repo_from_instance() -> None:
    assert repo_from_instance("django__django-11099") == "django/django"
    assert repo_from_instance("AnalogJ__lexicon-336") == "AnalogJ/lexicon"
    assert repo_from_instance("owner__name-with-dash-7") == "owner/name-with-dash"
    assert repo_from_instance("plainproject-4") == "plainproject"
    assert repo_from_instance("noversion") == "noversion"


def test_derive_action_prefers_tool_calls_then_last_fence() -> None:
    assert derive_action("a\n```\nls\n```\nthen\n```bash\npwd\n```", []) == "pwd"
    assert derive_action("just text  ", []) == "just text"
    assert derive_action("x", [ToolCall("bash", '{"cmd":"ls"}')]) == 'bash({"cmd":"ls"})'


def test_text_of_handles_parts_and_none() -> None:
    assert text_of(None) == ""
    assert text_of([{"type": "text", "text": "a"}, "b", {"type": "image"}]) == "ab"
    assert text_of(3) == "3"


def test_nebius_row() -> None:
    t = from_nebius_row(NEBIUS_ROW)
    assert t.instance_id == "acme__widgets-12"
    assert t.repo == "acme/widgets"
    assert t.resolved is True
    assert t.system == "SETTING: shell"
    assert t.task == "ISSUE: widgets break"
    assert [s.action for s in t.steps] == ["ls -F", "submit"]
    assert t.steps[0].observations == ["w.py\n(Open file: n/a)"]
    assert t.steps[1].observations == []
    assert t.model == "agent-7b"
    assert t.id.startswith("acme__widgets-12#")


def test_messages_row_with_tool_calls() -> None:
    t = from_messages_row(MESSAGES_ROW)
    assert t.task == "Fix the parser."
    assert t.resolved is True
    assert len(t.steps) == 2
    s = t.steps[0]
    assert [c.name for c in s.tool_calls] == ["bash", "bash"]
    assert s.tool_calls[0].arguments == '{"cmd": "ls"}'
    assert s.observations == ["a.py", "/repo"]
    assert s.action.startswith("bash(")


def test_messages_row_rejects_unknown_role() -> None:
    with pytest.raises(FormatError):
        from_messages_row({"messages": [{"role": "narrator", "content": "x"}]})


def test_ids_do_not_depend_on_position() -> None:
    a = from_nebius_row(NEBIUS_ROW)
    b = from_nebius_row(json.loads(json.dumps(NEBIUS_ROW)))
    assert a.id == b.id
    changed = json.loads(json.dumps(NEBIUS_ROW))
    changed["trajectory"][2]["text"] = "Other."
    assert from_nebius_row(changed).id != a.id


def test_detect_format() -> None:
    assert detect_format(NEBIUS_ROW) == "nebius"
    assert detect_format(MESSAGES_ROW) == "messages"
    assert detect_format({"id": "x", "steps": []}) == "native"
    with pytest.raises(FormatError):
        detect_format({"foo": 1})


def test_swe_agent_traj_file(tmp_path: Path) -> None:
    doc = {
        "environment": "swe_main",
        "history": [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "demo", "is_demo": True},
            {"role": "user", "content": [{"type": "text", "text": "ISSUE text"}]},
        ],
        "trajectory": [
            {"action": "ls\n", "observation": "a.py", "response": "", "thought": "Look."},
            {"action": "submit", "observation": "", "response": "Done.\n```\nsubmit\n```",
             "thought": "Done."},
        ],
        "info": {"exit_status": "submitted", "submission": "diff --git a/a b/a\n"},
    }
    p = tmp_path / "acme__widgets-9.traj"
    p.write_text(json.dumps(doc), encoding="utf-8")
    (t,) = list(read_path(p))
    assert t.instance_id == "acme__widgets-9"
    assert t.system == "SYS" and t.task == "ISSUE text"
    assert [s.action for s in t.steps] == ["ls", "submit"]
    assert "```\nls\n```" in t.steps[0].content
    assert t.resolved is None
    assert t.exit_status == "submitted"
    (t2,) = list(read_path(tmp_path))
    assert t2.id == t.id


def test_jsonl_and_json_list_and_native_roundtrip(tmp_path: Path) -> None:
    jl = tmp_path / "rows.jsonl"
    jl.write_text(json.dumps(NEBIUS_ROW) + "\n\n" + json.dumps(MESSAGES_ROW) + "\n",
                  encoding="utf-8")
    trajs = list(read_path(jl))
    assert [t.source for t in trajs] == ["nebius", "messages"]
    js = tmp_path / "rows.json"
    js.write_text(json.dumps([NEBIUS_ROW]), encoding="utf-8")
    assert list(read_path(js))[0].id == trajs[0].id
    out = tmp_path / "native.jsonl"
    assert write_native(trajs, out) == 2
    back = list(read_path(out))
    assert [t.to_dict() for t in back] == [t.to_dict() for t in trajs]


def test_bad_json_line_reports_position(tmp_path: Path) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps(NEBIUS_ROW) + "\n{not json\n", encoding="utf-8")
    with pytest.raises(FormatError, match="bad.jsonl:2"):
        list(read_path(p))


def test_native_roundtrip_fuzz() -> None:
    rng = random.Random(7)
    for _ in range(50):
        steps = [
            Step(content=rng.choice(["a", "```\nls\n```", ""]),
                 observations=[rng.choice(["x", "", "y\nz"])] * rng.randint(0, 2),
                 tool_calls=[ToolCall("f", "{}", "c1")] * rng.randint(0, 1),
                 failed=rng.random() < 0.3)
            for _ in range(rng.randint(0, 4))
        ]
        t = Trajectory(id="i", instance_id="o__r-1", steps=steps,
                       resolved=rng.choice([True, False, None]))
        assert Trajectory.from_dict(json.loads(json.dumps(t.to_dict()))) == t


def test_gzip_input_and_output(tmp_path: Path) -> None:
    from trajdistill.readers import iter_jsonl, write_jsonl

    gz = tmp_path / "rows.jsonl.gz"
    assert write_jsonl([NEBIUS_ROW, MESSAGES_ROW], gz) == 2
    assert gz.read_bytes()[:2] == b"\x1f\x8b"
    assert [t.source for t in read_path(gz)] == ["nebius", "messages"]
    assert [r["instance_id"] for r in iter_jsonl(gz)] == ["acme__widgets-12", "acme__widgets-3"]
    lst = tmp_path / "list.json"
    lst.write_text("[1]", encoding="utf-8")
    with pytest.raises(FormatError):
        list(iter_jsonl(lst))
