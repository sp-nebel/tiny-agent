"""The `a` answer at a confirmation prompt: what prefix a command is
approved under, and when approval is never offered."""
import builtins
import io

import pytest
from rich.console import Console

import config
import tools
from tools import confirm, command_prefix, command_allow, EDIT_ALLOW, run_cmd, edit_file


@pytest.fixture
def answers(monkeypatch):
    out = io.StringIO()
    monkeypatch.setattr(config, "AUTO_YES", False)
    monkeypatch.setattr(config, "console", Console(file=out, force_terminal=False, width=200))
    monkeypatch.setattr(tools, "_always_allowed", set())
    queue = []

    def fake_input(*a, **kw):
        if not queue:
            pytest.fail("prompted when it should have been auto-approved")
        return queue.pop(0)
    monkeypatch.setattr(builtins, "input", fake_input)
    return queue, out


@pytest.mark.parametrize("cmd, prefix", [
    ("git checkout main", "git checkout"),
    ("git checkout -b x", "git checkout"),
    ("npm run dev -- --port 3000", "npm run dev"),
    ("python3 -m pytest -q", "python3 -m pytest"),
    ("pytest -q tests/", "pytest"),
    ("ls", "ls"),
])
def test_prefixes(cmd, prefix):
    assert command_prefix(cmd) == prefix


@pytest.mark.parametrize("cmd", [
    "git status; rm -rf x", "git log | head", "echo $(whoami)", "ls > out",
    "a && b", "FOO=1 pytest", "echo `id`", "git log\nrm x", "echo 'unclosed",
])
def test_never_approvable_by_prefix(cmd):
    assert command_prefix(cmd) is None
    assert command_allow(cmd) is None


def test_a_approves_the_prefix_for_the_session(answers, tmp_path, monkeypatch):
    queue, out = answers
    monkeypatch.chdir(tmp_path)
    queue.append("a")
    assert run_cmd("git init -q") == "[exit 0, no output]"
    assert "[y/N/a/reason]" in out.getvalue()
    assert "a = always allow commands starting with 'git init'" in out.getvalue()
    # Same prefix: no prompt (fake_input would fail the test).
    assert run_cmd("git init --quiet") == "[exit 0, no output]"
    # Different prefix: asked again.
    queue.append("n")
    assert run_cmd("git status").startswith("[user declined")


def test_chained_command_still_asks_and_has_no_a(answers):
    queue, out = answers
    tools._always_allowed.add("cmd:echo")
    queue.append("n")
    assert run_cmd("echo hi; echo there").startswith("[user declined")
    assert "[y/N/reason]" in out.getvalue()
    assert "a/reason" not in out.getvalue()


def test_a_on_an_edit_approves_all_edits(answers, tmp_path):
    queue, _ = answers
    queue.append("a")
    edit_file(str(tmp_path / "one.txt"), "", "1\n")
    assert edit_file(str(tmp_path / "one.txt"), "1", "2").startswith("[edited")
    assert EDIT_ALLOW[0] in tools._always_allowed


def test_a_without_allow_is_a_reason(answers):
    queue, _ = answers
    queue.append("a")
    assert confirm("do it?") == (False, "a")
