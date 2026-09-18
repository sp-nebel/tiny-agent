import builtins
import io

import pytest
from rich.console import Console

import config
from tools import confirm, declined


@pytest.fixture
def prompt(monkeypatch):
    """Route confirm()'s prompt through a captured console and feed it an
    answer. Returns (set_answer, rendered_output)."""
    out = io.StringIO()
    monkeypatch.setattr(config, "AUTO_YES", False)
    monkeypatch.setattr(config, "console",
                        Console(file=out, force_terminal=False, width=200))
    answer = {"value": ""}

    def fake_input(*a, **kw):
        if isinstance(answer["value"], BaseException):
            raise answer["value"]
        return answer["value"]

    monkeypatch.setattr(builtins, "input", fake_input)

    def set_answer(v):
        answer["value"] = v

    return set_answer, out


@pytest.mark.parametrize("ans, expected", [
    ("y", (True, "")),
    ("YES", (True, "")),
    ("", (False, "")),
    ("n", (False, "")),
    ("No", (False, "")),
    ("  use pytest, not python  ", (False, "use pytest, not python")),
])
def test_answers(prompt, ans, expected):
    set_answer, _ = prompt
    set_answer(ans)
    assert confirm("run: ls") == expected


@pytest.mark.parametrize("exc", [EOFError(), KeyboardInterrupt()])
def test_abandoned_prompt_declines(prompt, exc):
    set_answer, _ = prompt
    set_answer(exc)
    assert confirm("run: ls") == (False, "")


def test_auto_yes_skips_the_prompt(monkeypatch):
    monkeypatch.setattr(config, "AUTO_YES", True)
    monkeypatch.setattr(builtins, "input", lambda *a: pytest.fail("prompted under --yes"))
    assert confirm("run: rm -rf /tmp/x") == (True, "")


def test_hint_and_bracketed_message_are_displayed(prompt):
    # Rich silently drops an unknown "[lowercase …]" tag, so both the literal
    # hint and brackets inside the message must survive escaping.
    set_answer, out = prompt
    set_answer("n")
    confirm("run: test [weird] -f x")
    shown = out.getvalue()
    assert "[y/N/reason]" in shown
    assert "run: test [weird] -f x" in shown


def test_declined_without_reason():
    assert declined("write", "") == "[user declined write]"


def test_declined_with_reason_tells_model_to_change_course():
    msg = declined("command", "use the test runner")
    assert msg.startswith("[user declined command. Their reason: use the test runner.")
    assert "do not repeat the same call" in msg
