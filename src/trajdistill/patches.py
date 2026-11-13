"""Read file paths out of unified diffs and tell whether a patch touches test files."""

from __future__ import annotations

import fnmatch
import re
from pathlib import PurePosixPath

_DIFF_GIT = re.compile(r"^diff --git a/(\S+) b/(\S+)", re.MULTILINE)
_HEADER = re.compile(r"^(?:\+\+\+|---) (?:[ab]/)?(\S+)", re.MULTILINE)

TEST_DIRS = frozenset({"test", "tests", "testing", "__tests__", "spec", "specs"})
TEST_NAMES = (
    "test_*.py", "*_test.py", "*_tests.py", "conftest.py", "tests.py",
    "*.test.*", "*.spec.*", "*_test.go", "*Test.java", "*Tests.java", "*_spec.rb",
)


def patch_files(patch: str) -> list[str]:
    """Sorted unique paths named in a unified diff (``/dev/null`` excluded)."""
    paths: set[str] = set()
    for a, b in _DIFF_GIT.findall(patch):
        paths.update((a, b))
    for p in _HEADER.findall(patch):
        paths.add(p)
    paths.discard("/dev/null")
    paths.discard("dev/null")
    return sorted(paths)


def is_test_path(path: str) -> bool:
    """True when a path looks like a test file by directory or file name."""
    p = PurePosixPath(path)
    if any(part.lower() in TEST_DIRS for part in p.parts[:-1]):
        return True
    return any(fnmatch.fnmatchcase(p.name, pat) for pat in TEST_NAMES)


def touches_tests(patch: str) -> bool:
    return any(is_test_path(p) for p in patch_files(patch))
