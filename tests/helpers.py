"""Synthetic trajectory builders shared by the tests."""

from __future__ import annotations

from trajdistill.model import Step, Trajectory

PATCH_SRC = (
    "diff --git a/pkg/core.py b/pkg/core.py\n--- a/pkg/core.py\n+++ b/pkg/core.py\n"
    "@@ -1 +1 @@\n-x\n+y\n"
)
PATCH_TEST = (
    "diff --git a/tests/test_core.py b/tests/test_core.py\n--- a/tests/test_core.py\n"
    "+++ b/tests/test_core.py\n@@ -1 +1 @@\n-x\n+y\n"
)


def make_traj(
    instance_id: str = "acme__widgets-1",
    actions: list[str] | None = None,
    observations: list[str] | None = None,
    resolved: bool | None = True,
    exit_status: str = "submitted",
    patch: str = PATCH_SRC,
    tid: str | None = None,
    model: str = "m1",
) -> Trajectory:
    actions = actions if actions is not None else ["ls", "cat a.py", "submit"]
    if observations is None:
        observations = [f"out {i}" for i in range(len(actions))]
    steps = [
        Step(content=f"Thinking about step {i}.\n\n```\n{a}\n```", observations=[o])
        for i, (a, o) in enumerate(zip(actions, observations, strict=True))
    ]
    return Trajectory(
        id=tid or f"{instance_id}#{len(actions)}-{resolved}-{model}",
        instance_id=instance_id,
        steps=steps,
        system="You are a coding agent.",
        task=f"Fix the bug in {instance_id}.",
        resolved=resolved,
        exit_status=exit_status,
        patch=patch,
        model=model,
        source="test",
    )
