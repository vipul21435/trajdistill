"""Build chat training samples from trajectories.

Two output styles:

- ``openai``: ``{"messages": [...]}`` with ``system``, ``user``, ``assistant`` and, when the
  source has tool calls, ``assistant.tool_calls`` and ``tool`` replies carrying
  ``tool_call_id``. Observations of text-command agents become ``user`` turns.
- ``sharegpt``: ``{"conversations": [{"from": ..., "value": ...}]}`` with the roles
  ``system``, ``human``, ``gpt``, ``function_call`` and ``observation``. Model turns and
  environment turns alternate, so a step with tool calls becomes one ``function_call`` turn
  (its accompanying text is not kept; the count is reported) and a missing reply becomes an
  empty ``human`` turn.

Two granularities: one sample per trajectory, or one per step (context up to step t, target
step t) under a token budget that drops the oldest steps first and says so in a short note.
Every sample ends on a model turn: the last step's reply is not included.
"""

from __future__ import annotations

import bisect
import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from typing import Any

from .model import Step, Trajectory
from .tokens import TokenCounter

STYLES = ("openai", "sharegpt")
GRANULARITIES = ("trajectory", "step")

Msg = dict[str, Any]


def omitted_note(n: int) -> str:
    return f"[{n} earlier step{'s' if n != 1 else ''} omitted]"


def _call_id(step_index: int, j: int, given: str) -> str:
    return given or f"call_{step_index}_{j}"


def _maybe_json(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        return text


def _openai_step(step: Step, i: int, with_reply: bool) -> list[Msg]:
    msg: Msg = {"role": "assistant", "content": step.content}
    if step.tool_calls:
        msg["tool_calls"] = [
            {"id": _call_id(i, j, c.id), "type": "function",
             "function": {"name": c.name, "arguments": c.arguments}}
            for j, c in enumerate(step.tool_calls)
        ]
    out = [msg]
    if not with_reply:
        return out
    if step.tool_calls:
        obs = list(step.observations)
        n = len(step.tool_calls)
        if len(obs) > n:  # extra replies go with the last call
            obs = obs[: n - 1] + ["\n".join(obs[n - 1:])]
        obs += [""] * (n - len(obs))
        for j, c in enumerate(step.tool_calls):
            out.append({"role": "tool", "tool_call_id": _call_id(i, j, c.id), "content": obs[j]})
    elif step.observations:
        out.append({"role": "user", "content": step.observation})
    return out


def _sharegpt_step(step: Step, with_reply: bool) -> list[Msg]:
    if step.tool_calls:
        calls = [{"name": c.name, "arguments": _maybe_json(c.arguments)} for c in step.tool_calls]
        value = json.dumps(calls[0] if len(calls) == 1 else calls, ensure_ascii=False)
        out = [{"from": "function_call", "value": value}]
        reply = {"from": "observation", "value": step.observation}
    else:
        out = [{"from": "gpt", "value": step.content}]
        reply = {"from": "human", "value": step.observation}
    if with_reply:
        # keep model and environment turns alternating even when the reply is missing
        out.append(reply if step.observations else {"from": "human", "value": ""})
    return out


def _step_msgs(style: str, step: Step, i: int, with_reply: bool) -> list[Msg]:
    if style == "openai":
        return _openai_step(step, i, with_reply)
    return _sharegpt_step(step, with_reply)


def _header(traj: Trajectory, style: str, omitted: int) -> list[Msg]:
    if style == "openai":
        head: list[Msg] = [{"role": "system", "content": traj.system}] if traj.system else []
        head.append({"role": "user", "content": traj.task})
        if omitted:
            head.append({"role": "user", "content": omitted_note(omitted)})
        return head
    head = [{"from": "system", "value": traj.system}] if traj.system else []
    task = traj.task + ("\n\n" + omitted_note(omitted) if omitted else "")
    head.append({"from": "human", "value": task})
    return head


def _texts(msgs: list[Msg]) -> list[str]:
    out = []
    for m in msgs:
        out.append(str(m.get("content", m.get("value", "")) or ""))
        for c in m.get("tool_calls", []) or []:
            out.append(c["function"]["name"] + c["function"]["arguments"])
    return out


def conversation(traj: Trajectory, style: str, target: int, start: int = 0) -> list[Msg]:
    """Header, a note if ``start`` steps were left out, steps ``start..target-1`` with their
    replies, then the model turn of step ``target``."""
    msgs = _header(traj, style, start)
    for k in range(start, target):
        msgs.extend(_step_msgs(style, traj.steps[k], k, True))
    msgs.extend(_step_msgs(style, traj.steps[target], target, False))
    return msgs


@dataclass
class ConvertStats:
    trajectories: int = 0
    samples: int = 0
    skipped_empty: int = 0
    skipped_over_budget: int = 0
    steps_omitted: int = 0
    tool_call_text_dropped: int = 0
    tokens: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def _record(traj: Trajectory, msgs: list[Msg], style: str, bare: bool, step: int | None
            ) -> dict[str, Any]:
    key = "messages" if style == "openai" else "conversations"
    if bare:
        return {key: msgs}
    rec: dict[str, Any] = {
        "id": traj.id if step is None else f"{traj.id}:{step}",
        "instance_id": traj.instance_id,
        "repo": traj.repo,
        "resolved": traj.resolved,
    }
    if step is not None:
        rec["step"] = step
    rec[key] = msgs
    return rec


def _fit_start(traj: Trajectory, style: str, budget: int, counter: TokenCounter
               ) -> Iterator[tuple[int, int | None]]:
    """For each target step, the smallest number of leading steps to leave out so the sample
    fits ``budget`` (``None`` if it cannot fit even with no earlier steps).

    Per-step costs are counted separately and summed; with the character estimate a sum of
    rounded-up parts is never below the count of the joined sample, so the sum is an upper
    bound and the exact count of the built sample is checked afterwards anyway.
    """
    n = len(traj.steps)
    step_cost = [counter.count_many(_texts(_step_msgs(style, traj.steps[k], k, True)))
                 for k in range(n)]
    prefix = [0]
    for c in step_cost:
        prefix.append(prefix[-1] + c)
    head0 = counter.count_many(_texts(_header(traj, style, 0)))
    for t in range(n):
        target_cost = counter.count_many(_texts(_step_msgs(style, traj.steps[t], t, False)))
        if head0 + prefix[t] + target_cost <= budget:
            yield t, 0
            continue
        head = counter.count_many(_texts(_header(traj, style, t)))
        room = budget - head - target_cost
        if room < 0:
            yield t, None
            continue
        # smallest s in [1, t] with prefix[t] - prefix[s] <= room
        s = bisect.bisect_left(prefix, prefix[t] - room, lo=1, hi=t + 1)
        yield t, s


def build_samples(trajs: Iterable[Trajectory], style: str = "openai",
                  granularity: str = "trajectory", context_budget: int | None = None,
                  counter: TokenCounter | None = None, bare: bool = False,
                  stats: ConvertStats | None = None) -> Iterator[dict[str, Any]]:
    """Yield sample records in id order (then step order), independent of input order."""
    if style not in STYLES:
        raise ValueError(f"style must be one of {', '.join(STYLES)}")
    if granularity not in GRANULARITIES:
        raise ValueError(f"granularity must be one of {', '.join(GRANULARITIES)}")
    if context_budget is not None and context_budget < 1:
        raise ValueError("context_budget must be >= 1")
    counter = counter or TokenCounter()
    st = stats if stats is not None else ConvertStats()
    for traj in sorted(trajs, key=lambda t: (t.id, t.instance_id)):
        st.trajectories += 1
        if not traj.steps:
            st.skipped_empty += 1
            continue
        if granularity == "trajectory":
            plan: Iterable[tuple[int, int | None]] = [(len(traj.steps) - 1, 0)]
            budget = None
        else:
            budget = context_budget
            plan = ((t, 0) for t in range(len(traj.steps))) if budget is None else \
                _fit_start(traj, style, budget, counter)
        for t, start in plan:
            msgs: list[Msg] = []
            n_tok = 0
            while start is not None and start <= t:
                msgs = conversation(traj, style, t, start)
                n_tok = counter.count_many(_texts(msgs))
                if budget is None or n_tok <= budget:
                    break
                start += 1  # only reached when a tokenizer is not additive across texts
            if start is None or start > t:
                st.skipped_over_budget += 1
                continue
            st.samples += 1
            st.steps_omitted += start
            st.tokens += n_tok
            if style == "sharegpt":
                st.tool_call_text_dropped += sum(
                    1 for s in traj.steps[start:t + 1] if s.tool_calls and s.content.strip())
            yield _record(traj, msgs, style, bare, None if granularity == "trajectory" else t)
