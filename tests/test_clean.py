from __future__ import annotations

import random

import pytest

from trajdistill.clean import (
    CleanConfig,
    clean,
    clean_all,
    normalise_whitespace,
    redact,
    shorten,
    strip_ansi,
)
from trajdistill.model import Step, ToolCall, Trajectory

from .helpers import make_traj

ESC = "\x1b"
# secret-shaped and home-path strings are assembled at runtime
OPENAI_LIKE = "s" + "k-" + "Ab12" * 8
AWS_LIKE = "AK" + "IA" + "ABCDEFGHIJKLMNOP"
GH_LIKE = "gh" + "p_" + "x9" * 18
HF_LIKE = "h" + "f_" + "Q" * 34
PEM = ("-----BEGIN RSA PRIV" + "ATE KEY-----\nMIIabc\n-----END RSA PRIV" + "ATE KEY-----")
MAC_HOME = "/" + "Users" + "/alice"
LINUX_HOME = "/home" + "/bob"
WIN_HOME = "C:\\" + "Users" + "\\carol"


def test_strip_ansi() -> None:
    s = f"{ESC}[31mred{ESC}[0m {ESC}[1;32mbold{ESC}[K {ESC}]0;title\x07done {ESC}(Bx"
    assert strip_ansi(s) == "red bold done x"
    assert strip_ansi("plain") == "plain"


def test_redact_keys_and_homes() -> None:
    text = (f"key={OPENAI_LIKE} aws {AWS_LIKE} tok {GH_LIKE} hf {HF_LIKE}\n{PEM}\n"
            f"cd {MAC_HOME}/proj && ls {LINUX_HOME}/src 'C {WIN_HOME}\\x'")
    out, n = redact(text)
    assert n == 8
    for secret in (OPENAI_LIKE, AWS_LIKE, GH_LIKE, HF_LIKE, "MIIabc", "alice", "bob", "carol"):
        assert secret not in out
    assert "cd ~/proj" in out and "ls ~/src" in out
    assert redact(out) == (out, 0)


def test_redact_leaves_ordinary_text() -> None:
    # false positives seen on real rows: a file directly under /home, a URL fragment
    for s in ("task-skip", "risk-assessment-is-long-enough-here", "/homepage/x", "my/home/dir",
              "/usr/local/bin", "AKIAshort", "(Open file: /home/temp.py)",
              "https://example.org/api#!/" + "Users" + "/get_users/x", "cd " + LINUX_HOME):
        assert redact(s) == (s, 0)


def test_shorten_head_tail_and_budget() -> None:
    text = "".join(chr(97 + i % 26) for i in range(1000))
    out = shorten(text, 200)
    assert len(out) <= 200
    assert out.startswith(text[:50]) and out.endswith(text[-50:])
    assert "characters elided" in out
    assert shorten("short", 200) == "short"
    assert shorten(text, 5) == text[:5]
    head_only = shorten(text, 100, head_fraction=1.0)
    assert head_only.startswith(text[:60]) and head_only.endswith("elided ...]\n")
    with pytest.raises(ValueError):
        shorten(text, 0)


def test_shorten_never_exceeds_budget_property() -> None:
    rng = random.Random(5)
    for _ in range(500):
        text = "x" * rng.randint(0, 5000)
        budget = rng.randint(1, 400)
        out = shorten(text, budget, rng.random())
        assert len(out) <= max(budget, 0)
        assert shorten(out, budget) == out


def test_normalise_whitespace() -> None:
    s = "\n\na  \r\nb\t\r\n\n\n\nc \n\n"
    assert normalise_whitespace(s) == "a\nb\n\nc"
    assert normalise_whitespace("  indented\n    code") == "  indented\n    code"


def test_drop_failed_and_collapse() -> None:
    t = make_traj(actions=["edit 1:1", "edit 1:1", "ls", "ls", "ls", "submit"],
                  observations=["Your proposed edit has introduced new syntax error(s).", "ok",
                                "a", "a", "b", "done"])
    t.steps[5].failed = True
    out, stats = clean(t, CleanConfig(drop_failed=True))
    assert [s.action for s in out.steps] == ["edit 1:1", "ls", "ls", "ls"]
    assert stats.counts["failed_steps_dropped"] == 2
    out, stats = clean(t, CleanConfig(collapse_repeats=True))
    assert [s.observation for s in out.steps] == [
        "Your proposed edit has introduced new syntax error(s).", "ok", "a", "b", "done"]
    assert stats.counts["steps_collapsed"] == 1
    custom = CleanConfig(drop_failed=True, failure_patterns=(r"^b$",))
    assert [s.observation for s in clean(t, custom)[0].steps] == [
        "Your proposed edit has introduced new syntax error(s).", "ok", "a", "a"]


def test_collapse_ignores_tool_call_ids() -> None:
    steps = [Step(content="", observations=["x"], tool_calls=[ToolCall("bash", "{}", f"c{i}")])
             for i in range(3)]
    t = Trajectory(id="t", instance_id="o__r-1", steps=steps)
    out, _ = clean(t, CleanConfig(collapse_repeats=True))
    assert out.n_steps == 1


def test_clean_touches_every_text_and_counts() -> None:
    t = make_traj(observations=[f"{ESC}[31mE{ESC}[0m", "y" * 500, f"{MAC_HOME}/x"])
    t.system = f"sys  \n\n\n\n{OPENAI_LIKE}"
    t.steps[0].tool_calls = [ToolCall("bash", f'{{"cmd": "cat {LINUX_HOME}/k"}}', "c1")]
    t.patch = f"+token = '{AWS_LIKE}'\n"
    cfg = CleanConfig(strip_ansi=True, redact=True, max_observation_chars=100,
                      normalise_whitespace=True)
    out, stats = clean(t, cfg)
    assert out.steps[0].observations == ["E"]
    assert len(out.steps[1].observations[0]) <= 100
    assert out.steps[2].observations == ["~/x"]
    assert out.system == "sys\n\n[REDACTED]"
    assert "bob" not in out.steps[0].tool_calls[0].arguments
    assert AWS_LIKE not in out.patch
    d = stats.to_dict()
    assert d["trajectories_changed"] == 1 and d["ansi_texts"] == 1
    assert d["observations_shortened"] == 1 and d["redactions"] == 4
    assert t.steps[0].observations[0].startswith(ESC)  # input untouched


def test_inactive_config_is_identity() -> None:
    t = make_traj()
    out, stats = clean(t, CleanConfig())
    assert out is t and stats.to_dict()["trajectories_changed"] == 0


@pytest.mark.parametrize("kwargs", [dict(max_observation_chars=0), dict(head_fraction=1.5),
                                    dict(failure_patterns=("(",))])
def test_clean_config_validation(kwargs) -> None:
    with pytest.raises((ValueError, Exception)):
        CleanConfig(**kwargs)


PIECES = ["ok", " ", "\t", "\n", "\r\n", "\n\n\n", f"{ESC}[1m", f"{ESC}[0m", OPENAI_LIKE,
          MAC_HOME + "/w", "x" + LINUX_HOME, "command not found", "Traceback", "y" * 120,
          "xs" + "k-" + "Z" * 30]


def _noise(rng: random.Random) -> str:
    return "".join(rng.choice(PIECES) for _ in range(rng.randint(0, 12)))


def test_cleaning_is_idempotent_property() -> None:
    rng = random.Random(9)
    for _ in range(300):
        n = rng.randint(0, 6)
        steps = []
        for _ in range(n):
            obs = _noise(rng) if rng.random() < 0.7 else "same"
            steps.append(Step(content=rng.choice([_noise(rng), "same"]), observations=[obs],
                              failed=rng.random() < 0.1))
        t = Trajectory(id="t", instance_id="o__r-1", steps=steps, system=_noise(rng),
                       task=_noise(rng), patch=_noise(rng))
        cfg = CleanConfig(strip_ansi=rng.random() < 0.7, redact=rng.random() < 0.7,
                          max_observation_chars=rng.choice([None, 1, 20, 60, 200]),
                          head_fraction=rng.random(),
                          normalise_whitespace=rng.random() < 0.7,
                          drop_failed=rng.random() < 0.5, collapse_repeats=rng.random() < 0.5)
        once, _ = clean(t, cfg)
        twice, stats = clean(once, cfg)
        assert twice == once, cfg
        assert stats.to_dict()["trajectories_changed"] == 0
        if cfg.max_observation_chars:
            assert all(len(o) <= cfg.max_observation_chars
                       for s in once.steps for o in s.observations)


def test_clean_all_sums_stats() -> None:
    rows = [make_traj(observations=["a" * 50, "b", "c"]) for _ in range(3)]
    out, stats = clean_all(rows, CleanConfig(max_observation_chars=40))
    assert len(out) == 3 and stats.to_dict()["observations_shortened"] == 3


def test_unterminated_key_block_stops_at_blank_line_or_diff_header() -> None:
    begin = "-----BEGIN RSA PRIV" + "ATE KEY-----"
    patch = (f"diff --git a/k.py b/k.py\n--- a/k.py\n+++ b/k.py\n@@ -0,0 +1,3 @@\n+K = '''{begin}\n"
             "+MIIabc\n"
             "diff --git a/test_numeric.py b/test_numeric.py\n--- a/test_numeric.py\n"
             "+++ b/test_numeric.py\n@@ -1 +1 @@\n-a\n+b\n")
    out, n = redact(patch)
    assert n == 1 and "MIIabc" not in out
    assert "diff --git a/test_numeric.py" in out and out.endswith("+b\n")
    prose = f"{begin}\nMIIabc\n\nThe next paragraph stays."
    assert redact(prose)[0] == "[REDACTED]\n\nThe next paragraph stays."


def test_test_file_flag_comes_from_the_raw_patch() -> None:
    from trajdistill.select import SelectConfig, select

    begin = "-----BEGIN RSA PRIV" + "ATE KEY-----"
    patch = (f"diff --git a/k.py b/k.py\n+K = '{begin}\n+MII\n"
             "diff --git a/test_x.py b/test_x.py\n+++ b/test_x.py\n+b\n")
    t = make_traj(patch=patch)
    cleaned, _ = clean(t, CleanConfig(redact=True))
    assert cleaned.meta["touches_tests"] is True
    sel = select([cleaned], SelectConfig(test_patches="drop"))
    assert sel.flagged_test_patches == 1 and sel.rows_out == 0


def test_key_pattern_leaves_identifiers_of_the_same_shape() -> None:
    css = "s" + "k-navigation-header-wrapper-item"
    assert redact(f'<div class="{css}">')[1] == 0
    assert redact("my-" + "s" + "k-" + "Ab12" * 8)[1] == 0  # part of a longer name
    assert redact("h" + "f_" + "AutoModelForCausalLMWithValueHead")[1] == 0
    assert redact(OPENAI_LIKE)[1] == 1


ALICE = "/home" + "/alice"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("PATH=/usr/bin:" + ALICE + "/bin", "PATH=/usr/bin:~/bin"),
        ("file://" + ALICE + "/proj/x.py", "file://~/proj/x.py"),
        ("gcc -I" + ALICE + "/include", "gcc -I~/include"),
        ("scp alice@host:" + ALICE + "/x", "scp alice@host:~/x"),
        ("C:/" + "Users" + "/alice/x.txt", "~/x.txt"),
        ("/home" + "/jos\u00e9/proj/x.py", "~/proj/x.py"),
        ("C:\\" + "Users" + "\\John Doe\\x.txt", "~\\x.txt"),
        ('open("' + ALICE + '/a")', 'open("~/a")'),
    ],
)
def test_home_directories_in_common_shapes(text: str, expected: str) -> None:
    assert redact(text) == (expected, 1)
    assert redact(expected) == (expected, 0)


@pytest.mark.parametrize(
    "text",
    [
        "https://example.org" + ALICE + "/x",
        "http://localhost:8000" + ALICE + "/x",
        "see docs#!/" + "Users" + "/get_users/list",
        "(Open file: /home/temp.py)",
        "x" + ALICE + "/y",
        "~" + ALICE + "/y",
    ],
)
def test_home_candidates_inside_urls_or_names_are_kept(text: str) -> None:
    assert redact(text) == (text, 0)
