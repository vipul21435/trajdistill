"""Write examples/sample.jsonl, synthetic trajectories in the row shape of a public dataset.

The shape follows nebius/SWE-agent-trajectories; every row is made up. The mix is chosen so
each filter has something to remove: unresolved runs, an empty patch, a patch that edits
tests, a stuck loop, an exact duplicate, several runs of one instance, and one repository with
far more rows than the others.

    python examples/make_sample.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

SYSTEM = "SETTING: You are an autonomous programmer working in a shell with a file viewer."
REPOS = ["acme__widgets"] * 6 + ["lab__parser"] * 2 + ["tiny__cli"]


def patch_for(path: str) -> str:
    return (f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n"
            "@@ -1,2 +1,2 @@\n-old = 1\n+old = 2\n")


def row(rng: random.Random, instance: str, n_steps: int, resolved: bool, *,
        patch: str | None = None, loop: bool = False, exit_status: str = "submitted") -> dict:
    turns = [{"role": "system", "system_prompt": SYSTEM, "text": None},
             {"role": "user", "text": f"ISSUE:\nThe {instance.split('-')[0]} module returns the "
                                      "wrong value for empty input.\nINSTRUCTIONS: fix it."}]
    commands = ["ls -F", "find_file core.py", "open core.py", "goto 40", "edit 41:41",
                "python reproduce.py", "scroll_down", "search_file empty"]
    for i in range(n_steps - 1):
        cmd = "python reproduce.py" if loop and i >= 2 else rng.choice(commands)
        out = "AssertionError: expected 0" if loop and i >= 2 else f"(output of {cmd}, line {i})"
        turns.append({"role": "ai", "text": f"Next I will run a command.\n\n```\n{cmd}\n```"})
        turns.append({"role": "user", "text": f"{out}\n(Open file: core.py)\nbash-$"})
    turns.append({"role": "ai", "text": "The fix is in place.\n\n```\nsubmit\n```"})
    return {
        "instance_id": instance,
        "model_name": "synthetic-agent",
        "target": resolved,
        "trajectory": turns,
        "exit_status": exit_status,
        "generated_patch": patch_for("src/core.py") if patch is None else patch,
        "eval_logs": "",
    }


def main() -> None:
    rng = random.Random(0)
    rows = []
    for k, repo in enumerate(REPOS):
        instance = f"{repo}-{100 + k}"
        for _ in range(rng.randint(1, 3)):
            rows.append(row(rng, instance, rng.randint(3, 12), rng.random() < 0.6))
    rows.append(row(rng, "lab__parser-200", 4, True, patch=""))
    rows.append(row(rng, "lab__parser-201", 5, True, patch=patch_for("tests/test_core.py")))
    rows.append(row(rng, "tiny__cli-300", 8, True, loop=True))
    rows.append(row(rng, "tiny__cli-301", 30, True))
    rows.append(row(rng, "tiny__cli-302", 4, False, exit_status="exit_cost"))
    rows.append(json.loads(json.dumps(rows[0])))
    out = Path(__file__).with_name("sample.jsonl")
    out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(f"wrote {len(rows)} rows to {out.name}")


if __name__ == "__main__":
    main()
