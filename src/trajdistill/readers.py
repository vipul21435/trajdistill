"""Small readers for open trajectory formats.

Formats:

- ``native``: the JSONL this package writes (one :class:`Trajectory` per line).
- ``messages``: JSONL rows with a ``messages`` list in OpenAI chat form (tool calls allowed).
- ``nebius``: rows shaped like the nebius/SWE-agent-trajectories dataset
  (``instance_id``, ``model_name``, ``target``, ``trajectory`` of ``{role, text}``,
  ``exit_status``, ``generated_patch``).
- ``swe-agent``: SWE-agent ``.traj`` JSON files (one file, or a directory of them).

``auto`` picks the format from the path and the first record.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from .model import Step, ToolCall, Trajectory, content_hash

FORMATS = ("auto", "native", "messages", "nebius", "swe-agent")

_PATCH_KEYS = ("patch", "model_patch", "generated_patch", "submission")
_RESOLVED_KEYS = ("resolved", "target", "success")
_MODEL_KEYS = ("model", "model_name")


class FormatError(ValueError):
    """Raised when a record cannot be read in the requested format."""


def text_of(content: Any) -> str:
    """Flatten message content (a string, None, or a list of typed parts) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and isinstance(p.get("text"), str):
                parts.append(p["text"])
        return "".join(parts)
    return str(content)


def _first(row: dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return default


def _as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in ("true", "1", "yes", "resolved"):
        return True
    if s in ("false", "0", "no", "unresolved"):
        return False
    return None


def _finish(traj: Trajectory, explicit_id: Any) -> Trajectory:
    """Give a trajectory a stable id that does not depend on its position in the input."""
    if explicit_id not in (None, ""):
        traj.id = str(explicit_id)
    else:
        traj.id = f"{traj.instance_id}#{content_hash(traj)[:12]}"
    return traj


def _arguments_text(args: Any) -> str:
    if isinstance(args, str):
        return args
    return json.dumps(args, sort_keys=True, ensure_ascii=False)


def _tool_calls(raw: Any) -> list[ToolCall]:
    calls = []
    for c in raw or []:
        if not isinstance(c, dict):
            continue
        fn = c.get("function") if isinstance(c.get("function"), dict) else c
        calls.append(ToolCall(name=str(fn.get("name", "")),
                              arguments=_arguments_text(fn.get("arguments", {})),
                              id=str(c.get("id", "") or "")))
    return calls


def from_messages_row(row: dict[str, Any], source: str = "messages") -> Trajectory:
    """Build a trajectory from a row with an OpenAI-style ``messages`` list."""
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise FormatError("row has no 'messages' list")
    system_parts: list[str] = []
    task_parts: list[str] = []
    steps: list[Step] = []
    for m in messages:
        if not isinstance(m, dict):
            raise FormatError("message is not an object")
        role = m.get("role")
        text = text_of(m.get("content"))
        if role == "system":
            system_parts.append(text)
        elif role == "assistant":
            calls = _tool_calls(m.get("tool_calls"))
            legacy = m.get("function_call")
            if isinstance(legacy, dict):  # the older single-call field
                calls.append(ToolCall(name=str(legacy.get("name", "")),
                                      arguments=_arguments_text(legacy.get("arguments", {}))))
            steps.append(Step(content=text, tool_calls=calls))
        elif role in ("user", "tool", "function"):
            if not steps:
                task_parts.append(text)
            else:
                step = steps[-1]
                step.observations.append(text)
                step.reply_roles.append(role)
                step.reply_ids.append(str(m.get("tool_call_id", "") or ""))
        else:
            raise FormatError(f"unknown role {role!r}")
    instance_id = str(_first(row, ("instance_id",), "") or _first(row, ("id",), ""))
    traj = Trajectory(
        id="",
        instance_id=instance_id,
        steps=steps,
        system="\n\n".join(system_parts),
        task="\n\n".join(task_parts),
        repo=str(row.get("repo", "") or ""),
        resolved=_as_bool(_first(row, _RESOLVED_KEYS)),
        exit_status=str(row.get("exit_status", "") or ""),
        patch=str(_first(row, _PATCH_KEYS, "") or ""),
        model=str(_first(row, _MODEL_KEYS, "") or ""),
        source=source,
    )
    return _finish(traj, row.get("traj_id"))


def from_nebius_row(row: dict[str, Any]) -> Trajectory:
    """Build a trajectory from a nebius/SWE-agent-trajectories row."""
    turns = row.get("trajectory")
    if not isinstance(turns, list):
        raise FormatError("row has no 'trajectory' list")
    system_parts: list[str] = []
    task_parts: list[str] = []
    steps: list[Step] = []
    for t in turns:
        if not isinstance(t, dict):
            raise FormatError("trajectory entry is not an object")
        role = t.get("role")
        text = text_of(t.get("text"))
        if role == "system":
            system_parts.append(text_of(t.get("system_prompt")) or text)
        elif role == "ai":
            steps.append(Step(content=text))
        elif role == "user":
            if not steps:
                task_parts.append(text)
            else:
                steps[-1].observations.append(text)
        else:
            raise FormatError(f"unknown role {role!r}")
    traj = Trajectory(
        id="",
        instance_id=str(row.get("instance_id", "")),
        steps=steps,
        system="\n\n".join(p for p in system_parts if p),
        task="\n\n".join(task_parts),
        resolved=_as_bool(row.get("target")),
        exit_status=str(row.get("exit_status", "") or ""),
        patch=str(row.get("generated_patch", "") or ""),
        model=str(row.get("model_name", "") or ""),
        source="nebius",
    )
    return _finish(traj, None)


def from_swe_agent(data: dict[str, Any], instance_id: str) -> Trajectory:
    """Build a trajectory from a parsed SWE-agent ``.traj`` document."""
    items = data.get("trajectory")
    if not isinstance(items, list):
        raise FormatError("document has no 'trajectory' list")
    system, task = "", ""
    for h in data.get("history", []) or []:
        if not isinstance(h, dict):
            continue
        if h.get("role") == "system" and not system:
            system = text_of(h.get("content"))
        elif h.get("role") == "user" and not h.get("is_demo") and not task:
            task = text_of(h.get("content"))
    steps = []
    for it in items:
        if not isinstance(it, dict):
            raise FormatError("trajectory entry is not an object")
        action = text_of(it.get("action"))
        content = text_of(it.get("response"))
        if not content:
            thought = text_of(it.get("thought"))
            content = f"{thought}\n\n```\n{action.strip()}\n```" if action else thought
        steps.append(Step(content=content, observations=[text_of(it.get("observation"))],
                          action=action.strip()))
    info = data.get("info", {}) or {}
    traj = Trajectory(
        id="",
        instance_id=instance_id,
        steps=steps,
        system=system,
        task=task,
        resolved=_as_bool(info.get("resolved")),
        exit_status=str(info.get("exit_status", "") or ""),
        patch=str(info.get("submission", "") or ""),
        source="swe-agent",
    )
    return _finish(traj, None)


def _open_text(path: Path, mode: str = "r") -> IO[str]:
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def _iter_records(path: Path) -> Iterator[tuple[str, Any]]:
    """Yield ``(position, record)`` from a JSONL file or a JSON list, optionally gzipped."""
    with _open_text(path) as fh:
        head = fh.read(1)
        while head and head.isspace():
            head = fh.read(1)
        fh.seek(0)
        if head == "[":
            try:
                data = json.load(fh)
            except json.JSONDecodeError as exc:
                raise FormatError(f"{path}: invalid JSON ({exc.msg})") from None
            for i, row in enumerate(data, 1):
                yield f"{path} item {i}", row
            return
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                yield f"{path}:{lineno}", json.loads(line)
            except json.JSONDecodeError as exc:
                raise FormatError(f"{path}:{lineno}: invalid JSON ({exc.msg})") from None


@dataclass
class ReadStats:
    """What a read saw: records read and records skipped because they cannot be written."""

    rows: int = 0
    skipped_unencodable: int = 0


def _encodable(obj: Any) -> bool:
    """False when the record holds text (a lone surrogate) that UTF-8 cannot encode."""
    try:
        json.dumps(obj, ensure_ascii=False).encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _parse(where: str, fn: Callable[[], Trajectory]) -> Trajectory:
    try:
        return fn()
    except FormatError as exc:
        raise FormatError(f"{where}: {exc}") from None
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        raise FormatError(f"{where}: cannot read record ({type(exc).__name__}: {exc})") from None


def detect_format(row: dict[str, Any]) -> str:
    """Guess the format of one JSON record."""
    if isinstance(row.get("steps"), list) and "id" in row:
        return "native"
    if isinstance(row.get("trajectory"), list):
        first = next((t for t in row["trajectory"] if isinstance(t, dict)), {})
        if "text" in first or first.get("role") == "ai":
            return "nebius"
        if "action" in first or "observation" in first:
            return "swe-agent"
    if isinstance(row.get("messages"), list):
        return "messages"
    raise FormatError("cannot tell the format of this record; pass --format")


def _load_traj_file(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        raise FormatError(f"{path}: invalid JSON ({exc.msg})") from None


def read_path(path: str | Path, fmt: str = "auto", stats: ReadStats | None = None
              ) -> Iterator[Trajectory]:
    """Read every trajectory under ``path`` (a file, or a directory of ``.traj`` files).

    Malformed records raise :class:`FormatError` naming the file and line. Records that
    cannot be encoded as UTF-8 are skipped and counted in ``stats``.
    """
    p = Path(path)
    st = stats if stats is not None else ReadStats()
    if fmt not in FORMATS:
        raise FormatError(f"unknown format {fmt!r}; choose from {', '.join(FORMATS)}")
    if p.is_dir() or p.suffix == ".traj" or (fmt == "swe-agent"
                                              and p.suffix not in (".jsonl", ".gz")):
        files = sorted(p.rglob("*.traj")) if p.is_dir() else [p]
        if not files:
            raise FormatError(f"{p}: directory holds no .traj files")
        for f in files:
            data = _load_traj_file(f)
            st.rows += 1
            if not _encodable(data):
                st.skipped_unencodable += 1
                continue
            yield _parse(str(f), lambda d=data, f=f: from_swe_agent(
                d, instance_id=f.name.removesuffix(".traj")))
        return
    for where, row in _iter_records(p):
        if not isinstance(row, dict):
            raise FormatError(f"{where}: record is not an object")
        st.rows += 1
        if not _encodable(row):
            st.skipped_unencodable += 1
            continue
        try:
            kind = detect_format(row) if fmt == "auto" else fmt
        except FormatError as exc:
            raise FormatError(f"{where}: {exc}") from None
        if kind == "native":
            yield _parse(where, lambda r=row: Trajectory.from_dict(r))
        elif kind == "nebius":
            yield _parse(where, lambda r=row: from_nebius_row(r))
        elif kind == "messages":
            yield _parse(where, lambda r=row: from_messages_row(r))
        elif kind == "swe-agent":
            yield _parse(where, lambda r=row: from_swe_agent(
                r, instance_id=str(r.get("instance_id", ""))))


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> int:
    """Write JSON objects one per line (gzipped when the name ends in ``.gz``)."""
    p = Path(path)
    n = 0
    with _open_text(p, "w") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def iter_jsonl(path: str | Path, stats: ReadStats | None = None) -> Iterator[dict[str, Any]]:
    """Yield the JSON objects of a JSONL (or JSON list) file, gzipped or not; records that
    cannot be encoded as UTF-8 are skipped and counted in ``stats``."""
    st = stats if stats is not None else ReadStats()
    for where, row in _iter_records(Path(path)):
        if not isinstance(row, dict):
            raise FormatError(f"{where}: record is not an object")
        st.rows += 1
        if not _encodable(row):
            st.skipped_unencodable += 1
            continue
        yield row


def write_native(trajs: Iterable[Trajectory], path: str | Path) -> int:
    """Write trajectories as native JSONL; returns the number written."""
    return write_jsonl((t.to_dict() for t in trajs), path)
