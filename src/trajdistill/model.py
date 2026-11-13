"""In-memory trajectory model shared by readers, filters, cleaners and writers."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

_FENCE = re.compile(r"```[^\n`]*\n(.*?)```", re.DOTALL)
_INSTANCE_SUFFIX = re.compile(r"^(?P<repo>.+)-(?P<num>\d+)$")


@dataclass
class ToolCall:
    """One function call made by the assistant; ``arguments`` is kept as JSON text."""

    name: str
    arguments: str = "{}"
    id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ToolCall:
        return cls(name=str(d.get("name", "")), arguments=str(d.get("arguments", "{}")),
                   id=str(d.get("id", "")))


@dataclass
class Step:
    """One assistant turn and what the environment answered.

    ``content`` is the assistant text as produced (for text-command agents it includes the
    fenced command). ``observations`` holds one entry per tool call when the source has tool
    calls, otherwise a single entry. ``action`` is the canonical action string used for loop
    detection; when empty it is derived from the tool calls or the last fenced block.
    ``reply_roles`` and ``reply_ids`` run alongside ``observations`` when the source says who
    replied (``tool``, ``user``, ``function``) and to which call; they are empty otherwise.
    """

    content: str = ""
    observations: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    action: str = ""
    failed: bool = False
    reply_roles: list[str] = field(default_factory=list)
    reply_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.action:
            self.action = derive_action(self.content, self.tool_calls)

    @property
    def observation(self) -> str:
        return "\n".join(self.observations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "action": self.action,
            "observations": list(self.observations),
            "tool_calls": [c.to_dict() for c in self.tool_calls],
            "failed": self.failed,
            "reply_roles": list(self.reply_roles),
            "reply_ids": list(self.reply_ids),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Step:
        return cls(
            content=str(d.get("content", "")),
            observations=[str(o) for o in d.get("observations", [])],
            tool_calls=[ToolCall.from_dict(c) for c in d.get("tool_calls", [])],
            action=str(d.get("action", "")),
            failed=bool(d.get("failed", False)),
            reply_roles=[str(r) for r in d.get("reply_roles", [])],
            reply_ids=[str(r) for r in d.get("reply_ids", [])],
        )

    def reply(self, k: int) -> tuple[str, str, str]:
        """``(role, tool_call_id, text)`` of reply ``k``; role and id are "" when unknown."""
        role = self.reply_roles[k] if k < len(self.reply_roles) else ""
        rid = self.reply_ids[k] if k < len(self.reply_ids) else ""
        return role, rid, self.observations[k]


@dataclass
class Trajectory:
    """A full agent run on one instance."""

    id: str
    instance_id: str
    steps: list[Step]
    system: str = ""
    task: str = ""
    repo: str = ""
    resolved: bool | None = None
    exit_status: str = ""
    patch: str = ""
    model: str = ""
    source: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.repo:
            self.repo = repo_from_instance(self.instance_id)

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    def texts(self) -> list[str]:
        """Every piece of text a training sample built from this trajectory would contain."""
        out = [self.system, self.task]
        for s in self.steps:
            out.append(s.content)
            out.extend(s.observations)
            out.extend(c.name + c.arguments for c in s.tool_calls)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "instance_id": self.instance_id,
            "repo": self.repo,
            "model": self.model,
            "resolved": self.resolved,
            "exit_status": self.exit_status,
            "patch": self.patch,
            "system": self.system,
            "task": self.task,
            "steps": [s.to_dict() for s in self.steps],
            "source": self.source,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Trajectory:
        return cls(
            id=str(d["id"]),
            instance_id=str(d.get("instance_id", "")),
            steps=[Step.from_dict(s) for s in d.get("steps", [])],
            system=str(d.get("system", "")),
            task=str(d.get("task", "")),
            repo=str(d.get("repo", "")),
            resolved=d.get("resolved"),
            exit_status=str(d.get("exit_status", "") or ""),
            patch=str(d.get("patch", "") or ""),
            model=str(d.get("model", "") or ""),
            source=str(d.get("source", "")),
            meta=dict(d.get("meta", {})),
        )


def derive_action(content: str, tool_calls: list[ToolCall]) -> str:
    """Canonical action: rendered tool calls, else the last fenced block, else the stripped text."""
    if tool_calls:
        return "\n".join(f"{c.name}({c.arguments})" for c in tool_calls)
    blocks = _FENCE.findall(content)
    if blocks:
        return blocks[-1].strip()
    return content.strip()


def repo_from_instance(instance_id: str) -> str:
    """Map a SWE-bench style id such as ``owner__name-123`` to ``owner/name``.

    Ids without the ``__`` separator keep everything before a trailing ``-<number>``.
    """
    head = instance_id
    m = _INSTANCE_SUFFIX.match(instance_id)
    if m:
        head = m.group("repo")
    if "__" in head:
        owner, _, name = head.partition("__")
        return f"{owner}/{name}"
    return head


def _step_payload(s: Step) -> list[Any]:
    out: list[Any] = [s.content, s.observations, [c.to_dict() for c in s.tool_calls]]
    if any(s.reply_roles) or any(s.reply_ids):  # only then, so other formats keep their ids
        out += [s.reply_roles, s.reply_ids]
    return out


def content_hash(traj: Trajectory) -> str:
    """Hash of everything that defines the run (not its id or source format)."""
    payload = {
        "instance_id": traj.instance_id,
        "model": traj.model,
        "resolved": traj.resolved,
        "exit_status": traj.exit_status,
        "patch": traj.patch,
        "system": traj.system,
        "task": traj.task,
        "steps": [_step_payload(s) for s in traj.steps],
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
