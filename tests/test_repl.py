"""main()'s REPL over scripted prompt input, with run_turn and the warmup
stubbed out — the prompt-level features (`!cmd`, `!!cmd`, `@file`, `/undo`) are
pure bookkeeping on the message list and need no model."""
import subprocess

import pytest

import agent
import config
from agent import SHELL_BLOCK_HEAD, run_shell_escape


class Repl:
    """Feeds `inputs` to main() as typed prompts and records what each turn
    was handed. The seed read_prompt was given for each prompt is recorded
    too, which is how /undo's pre-filled prompt shows up."""

    def __init__(self, monkeypatch, inputs, on_turn=None):
        self.inputs = list(inputs)
        self.seeds  = []
        self.turns  = []                 # user message content per turn
        self.on_turn = on_turn

        def fake_read_prompt(prompt, seed=""):
            self.seeds.append(seed)
            if not self.inputs:
                raise EOFError
            return self.inputs.pop(0)

        def fake_run_turn(messages, max_steps=None, check_every=None):
            self.turns.append(messages[-1]["content"])
            messages.append({"role": "assistant", "content": "ok"})
            if self.on_turn:
                self.on_turn(len(self.turns))
            self.messages = messages

        monkeypatch.setattr(agent, "read_prompt", fake_read_prompt)
        monkeypatch.setattr(agent, "run_turn", fake_run_turn)
        monkeypatch.setattr(agent, "warm_cache", lambda *a, **k: None)
        # Keep main()'s exit hooks (session autosave, readline history) out of
        # the real home directory: they would fire at interpreter exit.
        monkeypatch.setattr(agent.atexit, "register", lambda *a, **k: None)
        monkeypatch.setattr(agent, "readline", None)
        monkeypatch.setattr("sys.argv", ["local_agent.py"])

    def run(self):
        agent.main()
        return self


def test_bang_output_rides_with_the_next_prompt(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    r = Repl(monkeypatch, ["!echo hello", "what did it print?", "next"]).run()
    first = r.turns[0]
    block = SHELL_BLOCK_HEAD.format(cmd="echo hello", status="exit 0") + "\nhello"
    assert first == f"Working directory: {tmp_path}\n\n{block}\n\nwhat did it print?"
    assert r.turns[1] == "next"            # consumed by the turn, not resent


def test_double_bang_is_never_sent(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    r = Repl(monkeypatch, ["!!echo secret", "hi"]).run()
    assert "secret" not in r.turns[0]


def test_clear_drops_pending_shell_output(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    r = Repl(monkeypatch, ["!echo stale", "/clear", "hi"]).run()
    assert "stale" not in r.turns[0]


def test_shell_escape_reports_exit_code_and_empty_output():
    block = run_shell_escape("exit 3")
    assert block == SHELL_BLOCK_HEAD.format(cmd="exit 3", status="exit 3") + "\n[no output]"


def test_shell_escape_timeout(monkeypatch):
    monkeypatch.setattr(config, "CMD_TIMEOUT", 0.2)
    block = run_shell_escape("sleep 2")
    assert "timed out after 0.2s" in block


def test_undo_rewinds_and_prefills_the_prompt(monkeypatch, tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "f.txt").write_text("before\n")
    monkeypatch.chdir(tmp_path)

    def agent_edits(n):
        if n == 2:
            (tmp_path / "f.txt").write_text("after\n")
            (tmp_path / "made.txt").write_text("new\n")

    r = Repl(monkeypatch, ["one", "two", "/undo", "two again"], on_turn=agent_edits).run()

    assert (tmp_path / "f.txt").read_text() == "before\n"
    assert not (tmp_path / "made.txt").exists()
    assert r.seeds[3] == "two"             # the prompt read right after /undo
    assert [m["content"] for m in r.messages[1:]] == [
        f"Working directory: {tmp_path}\n\none", "ok", "two again", "ok"]


def test_undoing_the_first_turn_reinjects_the_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    r = Repl(monkeypatch, ["one", "/undo", "one again"]).run()
    assert r.turns[1] == f"Working directory: {tmp_path}\n\none again"


def test_file_ref_is_attached_but_prompt_seed_stays_raw(monkeypatch, tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "a.py").write_text("x = 1\n")
    monkeypatch.chdir(tmp_path)
    r = Repl(monkeypatch, ["fix @a.py", "/undo"]).run()
    assert "[contents of a.py, attached by the user]" in r.turns[0]
    assert r.seeds[2] == "fix @a.py"
