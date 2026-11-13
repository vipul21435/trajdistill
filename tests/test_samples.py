from __future__ import annotations

import json
import random

import pytest

from trajdistill.model import Step, ToolCall, Trajectory
from trajdistill.samples import ConvertStats, build_samples, conversation, omitted_note
from trajdistill.tokens import TokenCounter

from .helpers import make_traj


def tool_traj() -> Trajectory:
    steps = [
        Step(content="Look around.", observations=["a.py", "/repo"],
             tool_calls=[ToolCall("bash", '{"cmd": "ls"}', "c1"),
                         ToolCall("bash", '{"cmd": "pwd"}', "")]),
        Step(content="", observations=["x", "y"], tool_calls=[ToolCall("view", "{}", "c3")]),
        Step(content="Done.", observations=["bye"]),
    ]
    return Trajectory(id="tt", instance_id="o__r-1", steps=steps, system="S", task="T")


def roles(msgs):
    return [m.get("role", m.get("from")) for m in msgs]


def test_openai_text_agent_sample() -> None:
    t = make_traj(actions=["ls", "submit"], observations=["a", "final reply"])
    (rec,) = build_samples([t])
    assert roles(rec["messages"]) == ["system", "user", "assistant", "user", "assistant"]
    assert rec["messages"][3]["content"] == "a"
    assert "final reply" not in json.dumps(rec)  # samples end on the last model turn
    assert rec["id"] == t.id and rec["repo"] == "acme/widgets" and "step" not in rec


def test_openai_tool_calls() -> None:
    (rec,) = build_samples([tool_traj()])
    m = rec["messages"]
    assert roles(m) == ["system", "user", "assistant", "tool", "tool", "assistant", "tool",
                        "assistant"]
    assert [c["id"] for c in m[2]["tool_calls"]] == ["c1", "call_0_1"]
    assert m[3] == {"role": "tool", "tool_call_id": "c1", "content": "a.py"}
    assert m[4]["tool_call_id"] == "call_0_1"
    assert m[6]["content"] == "x\ny"  # extra replies join the last call


def test_missing_tool_reply_is_empty() -> None:
    t = Trajectory(id="m", instance_id="o__r-1", task="T", steps=[
        Step(content="", tool_calls=[ToolCall("a", "{}", "1"), ToolCall("b", "{}", "2")],
             observations=["only one"]),
        Step(content="end")])
    (rec,) = build_samples([t])
    assert [x.get("content") for x in rec["messages"][2:4]] == ["only one", ""]


def test_sharegpt_alternates_and_counts_dropped_text() -> None:
    stats = ConvertStats()
    (rec,) = build_samples([tool_traj()], style="sharegpt", stats=stats)
    conv = rec["conversations"]
    assert roles(conv) == ["system", "human", "function_call", "observation", "function_call",
                           "observation", "gpt"]
    assert json.loads(conv[2]["value"])[0] == {"name": "bash", "arguments": {"cmd": "ls"}}
    assert json.loads(conv[4]["value"]) == {"name": "view", "arguments": {}}
    assert stats.tool_call_text_dropped == 1
    t = Trajectory(id="g", instance_id="o__r-1", task="T",
                   steps=[Step(content="a"), Step(content="b")])
    (rec,) = build_samples([t], style="sharegpt")
    assert roles(rec["conversations"]) == ["human", "gpt", "human", "gpt"]


def test_step_level_samples() -> None:
    t = make_traj(actions=["ls", "cat a", "submit"])
    recs = list(build_samples([t], granularity="step"))
    assert [r["step"] for r in recs] == [0, 1, 2]
    for r in recs:
        last = r["messages"][-1]
        assert last["role"] == "assistant"
        assert last["content"] == t.steps[r["step"]].content
        assert sum(m["role"] == "assistant" for m in r["messages"]) == r["step"] + 1
    assert recs[2]["id"] == f"{t.id}:2"


def test_budget_drops_oldest_steps_with_a_note() -> None:
    t = make_traj(actions=[f"cmd {i}" for i in range(8)],
                  observations=["o" * 200 for _ in range(8)])
    c = TokenCounter()
    stats = ConvertStats()
    recs = list(build_samples([t], granularity="step", context_budget=200, counter=c,
                              stats=stats))
    assert len(recs) == 8
    late = recs[-1]["messages"]
    assert any(m["content"].startswith("[") and "omitted" in m["content"] for m in late)
    assert stats.steps_omitted > 0
    for r in recs:
        assert c.count_many(m["content"] for m in r["messages"]) <= 200
    tight = list(build_samples([t], granularity="step", context_budget=10, counter=c))
    assert tight == []


def test_budget_property_both_styles() -> None:
    rng = random.Random(3)
    c = TokenCounter(chars_per_token=rng.choice([1, 3, 4]))
    for k in range(80):
        n = rng.randint(1, 10)
        steps = []
        for _step in range(n):
            calls = [ToolCall("f", json.dumps({"x": "y" * rng.randint(0, 40)}), "")] \
                if rng.random() < 0.4 else []
            steps.append(Step(content="c" * rng.randint(0, 80), tool_calls=calls,
                              observations=["o" * rng.randint(0, 150)] * rng.randint(0, 2)))
        t = Trajectory(id=f"r{k}", instance_id="o__r-1", steps=steps, system="s" * 20,
                       task="t" * rng.randint(1, 60))
        budget = rng.randint(20, 400)
        for style in ("openai", "sharegpt"):
            stats = ConvertStats()
            recs = list(build_samples([t], style=style, granularity="step",
                                      context_budget=budget, counter=c, stats=stats))
            assert stats.samples + stats.skipped_over_budget == n
            key = "messages" if style == "openai" else "conversations"
            for r in recs:
                texts = [str(m.get("content", m.get("value", ""))) for m in r[key]]
                texts += [x["function"]["name"] + x["function"]["arguments"]
                          for m in r[key] for x in m.get("tool_calls", [])]
                assert c.count_many(texts) <= budget


def test_output_does_not_depend_on_input_order() -> None:
    rows = [make_traj(f"o__r-{i}", tid=f"id{i}") for i in range(6)]
    a = list(build_samples(rows, granularity="step"))
    rng = random.Random(1)
    rng.shuffle(rows)
    assert list(build_samples(rows, granularity="step")) == a


def test_bare_records_and_empty_trajectories() -> None:
    stats = ConvertStats()
    t = make_traj()
    empty = Trajectory(id="e", instance_id="o__r-1", steps=[])
    recs = list(build_samples([t, empty], style="sharegpt", bare=True, stats=stats))
    assert list(recs[0]) == ["conversations"]
    assert stats.skipped_empty == 1 and stats.samples == 1 and stats.tokens > 0


def test_conversation_note_and_validation() -> None:
    t = make_traj(actions=["a", "b", "c"])
    msgs = conversation(t, "openai", target=2, start=2)
    assert msgs[2] == {"role": "user", "content": omitted_note(2)}
    assert omitted_note(1) == "[1 earlier step omitted]"
    sg = conversation(t, "sharegpt", target=1, start=1)
    assert sg[1]["value"].endswith(omitted_note(1))
    for kwargs in (dict(style="xml"), dict(granularity="token"), dict(context_budget=0)):
        with pytest.raises(ValueError):
            list(build_samples([t], **kwargs))
