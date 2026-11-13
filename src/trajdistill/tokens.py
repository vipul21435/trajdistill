"""Token counting: a character-based estimate by default, tiktoken when installed and asked for."""

from __future__ import annotations

import math
from collections.abc import Iterable

from .model import Trajectory

DEFAULT_CHARS_PER_TOKEN = 4.0


class TokenCounter:
    """Count tokens with ``method="chars"`` (length / chars_per_token, rounded up) or
    ``method="tiktoken"`` (exact for the named encoding; needs the ``tiktoken`` extra)."""

    def __init__(self, method: str = "chars", chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
                 encoding: str = "cl100k_base") -> None:
        if method not in ("chars", "tiktoken"):
            raise ValueError(f"unknown token method {method!r}")
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be positive")
        self.method = method
        self.chars_per_token = chars_per_token
        self._enc = None
        if method == "tiktoken":
            try:
                import tiktoken
            except ImportError as exc:
                raise RuntimeError(
                    "tiktoken is not installed; pip install "
                    '"trajdistill[tiktoken] @ git+https://github.com/vipul21435/trajdistill"'
                ) from exc
            self._enc = tiktoken.get_encoding(encoding)

    def count(self, text: str) -> int:
        return self.count_many([text])

    def count_many(self, texts: Iterable[str]) -> int:
        if self._enc is None:
            return math.ceil(sum(len(t) for t in texts) / self.chars_per_token)
        return sum(len(self._enc.encode(t, disallowed_special=())) for t in texts)

    def count_trajectory(self, traj: Trajectory) -> int:
        return self.count_many(traj.texts())
