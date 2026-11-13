from __future__ import annotations

import pytest

from trajdistill.patches import is_test_path, patch_files, touches_tests
from trajdistill.tokens import TokenCounter

from .helpers import PATCH_SRC, PATCH_TEST, make_traj


def test_char_estimate_rounds_up_once_over_all_pieces() -> None:
    c = TokenCounter()
    assert c.count("") == 0
    assert c.count("abcd") == 1
    assert c.count("abcde") == 2
    assert c.count_many(["ab", "cd"]) == 1
    assert TokenCounter(chars_per_token=2).count("abcde") == 3


def test_trajectory_count_covers_every_text() -> None:
    t = make_traj()
    total = sum(len(x) for x in t.texts())
    assert TokenCounter(chars_per_token=1).count_trajectory(t) == total


def test_bad_counter_settings() -> None:
    with pytest.raises(ValueError):
        TokenCounter(method="words")
    with pytest.raises(ValueError):
        TokenCounter(chars_per_token=0)


def test_patch_files() -> None:
    assert patch_files(PATCH_SRC) == ["pkg/core.py"]
    new_file = "diff --git a/n.py b/n.py\nnew file mode 100644\n--- /dev/null\n+++ b/n.py\n"
    assert patch_files(new_file) == ["n.py"]
    assert patch_files("") == []


@pytest.mark.parametrize(
    "path,expected",
    [
        ("tests/test_core.py", True),
        ("pkg/tests/helpers.py", True),
        ("src/test_utils.py", True),
        ("lib/core_test.go", True),
        ("web/app.spec.ts", True),
        ("conftest.py", True),
        ("src/FooTest.java", True),
        ("pkg/core.py", False),
        ("docs/testing.md", False),
        ("pkg/contest.py", False),
    ],
)
def test_is_test_path(path: str, expected: bool) -> None:
    assert is_test_path(path) is expected


def test_touches_tests() -> None:
    assert touches_tests(PATCH_TEST)
    assert not touches_tests(PATCH_SRC)
    assert touches_tests(PATCH_SRC + PATCH_TEST)


def test_missing_tiktoken_names_the_git_install(monkeypatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "tiktoken", None)  # makes the import fail
    with pytest.raises(RuntimeError, match=r"trajdistill\[tiktoken\] @ git\+https://github.com/"):
        TokenCounter(method="tiktoken")
