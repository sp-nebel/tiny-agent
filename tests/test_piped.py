"""Piped stdin: one turn on the input, then exit; the answer alone on a
piped stdout; confirmations refused without a terminal."""
import io

import pytest

import agent
import config
import tools
from agent import PIPED_HEAD
from test_repl import Repl, FakeTTY


@pytest.fixture
def piped(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "INTERACTIVE", True)
    monkeypatch.setattr(config, "ANSWER_TO_STDOUT", False)
    monkeypatch.setattr(config, "console", config.console)

    def run(stdin_text, argv, stdout_tty=True, answer="the answer"):
        r = Repl(monkeypatch, [])
        monkeypatch.setattr("sys.stdin", FakeTTY(stdin_text, tty=False))
        out = io.StringIO()
        out.isatty = lambda: stdout_tty
        monkeypatch.setattr("sys.stdout", out)
        monkeypatch.setattr("sys.argv", ["local_agent.py"] + argv)

        def fake_run_turn(messages, max_steps=None, check_every=None):
            r.turns.append(messages[-1]["content"])
            agent._print_answer(answer)
        monkeypatch.setattr(agent, "run_turn", fake_run_turn)
        with pytest.raises(SystemExit) as exc:
            agent.main()
        return r, out, exc.value.code
    return run


def test_piped_input_rides_ahead_of_the_prompt(piped):
    r, out, code = piped("diff --git a b\n@decorator\n", ["review this"])
    assert code == 0
    assert len(r.turns) == 1
    assert f"{PIPED_HEAD}\ndiff --git a b\n@decorator\n\nreview this" in r.turns[0]
    assert "[contents of" not in r.turns[0]


def test_piped_text_alone_is_the_prompt_and_never_a_command(piped):
    r, out, code = piped("!echo pwned\n", [])
    assert r.turns[0].endswith("!echo pwned")


def test_answer_alone_on_piped_stdout(piped):
    r, out, code = piped("x", ["summarize"], stdout_tty=False)
    assert out.getvalue() == "the answer\n"
    assert config.console.stderr


def test_nothing_to_do(piped):
    r, out, code = piped("  \n", [])
    assert code == 2 and r.turns == []


def test_confirm_refuses_without_a_terminal(monkeypatch):
    monkeypatch.setattr(config, "AUTO_YES", False)
    monkeypatch.setattr(config, "INTERACTIVE", False)
    ok, reason = tools.confirm("run: rm x")
    assert not ok and "no terminal" in reason
    monkeypatch.setattr(config, "AUTO_YES", True)
    assert tools.confirm("run: rm x") == (True, "")
