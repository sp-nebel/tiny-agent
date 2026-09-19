"""The REPL commands added on top of the basics: custom commands, /help,
/history, /undo N, /export, /editor, session titles, @path#A-B and Tab
completion. Driven through main() like test_repl."""
import io
import json
import subprocess

import pytest
from rich.console import Console

import agent
import commands
import config
from agent import expand_file_refs, FILE_BLOCK_HEAD
from test_repl import Repl


@pytest.fixture
def out(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr(config, "console", Console(file=buf, force_terminal=False, width=200))
    return buf


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A project dir as cwd, with config and sessions kept out of $HOME."""
    proj = tmp_path / "proj"
    proj.mkdir()
    monkeypatch.chdir(proj)
    monkeypatch.setattr(config, "CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setattr(config, "SESSION_DIR", str(tmp_path / "sessions"))
    return proj


# ---- custom commands -------------------------------------------------------- #

def test_custom_command_expands_arguments_shell_and_files(home, monkeypatch):
    cmds = home / ".tiny-agent" / "commands"
    cmds.mkdir(parents=True)
    (cmds / "review.md").write_text(
        "---\ndescription: review a file\n---\nReview @$1 for $2. All: $ARGUMENTS\n"
        "Branch: !`echo main`\n")
    (home / "a.py").write_text("x = 1\n")
    r = Repl(monkeypatch, ["/review a.py bugs"]).run()
    sent = r.turns[0]
    assert "Review @a.py for bugs. All: a.py bugs\nBranch: main" in sent
    assert FILE_BLOCK_HEAD.format(path="a.py") in sent


def test_user_command_dir_and_appended_arguments(home, monkeypatch):
    user_cmds = home.parent / "cfg" / "commands"
    user_cmds.mkdir(parents=True)
    (user_cmds / "explain.md").write_text("Explain this code simply.")
    r = Repl(monkeypatch, ["/explain the parser"]).run()
    assert r.turns[0].endswith("Explain this code simply.\n\nthe parser")


def test_project_command_shadows_user_command_and_builtins_win(home):
    (home / ".tiny-agent" / "commands").mkdir(parents=True)
    (home / ".tiny-agent" / "commands" / "x.md").write_text("project")
    (home / ".tiny-agent" / "commands" / "clear.md").write_text("never")
    user_cmds = home.parent / "cfg" / "commands"
    user_cmds.mkdir(parents=True)
    (user_cmds / "x.md").write_text("user")
    found = commands.load_custom_commands(str(home))
    assert found["/x"][2] == "project"
    assert "/clear" not in found


def test_unknown_command_is_not_sent(home, monkeypatch, out):
    r = Repl(monkeypatch, ["/revew a.py", "hi"]).run()
    assert r.turns == [r.turns[0]] and r.turns[0].endswith("hi")
    assert "unknown command /revew" in out.getvalue()
    assert r.seeds[1] == "/revew a.py"


def test_help_lists_builtins_and_custom(home, monkeypatch, out):
    (home / ".tiny-agent" / "commands").mkdir(parents=True)
    (home / ".tiny-agent" / "commands" / "ship.md").write_text("---\ndescription: ship it\n---\nGo")
    Repl(monkeypatch, ["/help"]).run()
    text = out.getvalue()
    assert "/undo [N]" in text and "/export [PATH]" in text
    assert "/ship" in text and "ship it" in text


# ---- /undo N and /history --------------------------------------------------- #

def test_undo_n_rewinds_several_turns(home, monkeypatch, out):
    subprocess.run(["git", "init", "-q"], cwd=home, check=True)
    f = home / "f.txt"
    f.write_text("0\n")

    def edit(n):
        f.write_text(f"{n}\n")

    r = Repl(monkeypatch, ["one", "two", "three", "/history", "/undo 2"], on_turn=edit).run()
    assert f.read_text() == "1\n"
    assert [m["content"] for m in r.messages[1:]][-1] == "ok"
    assert len(r.messages) == 3                 # system, "one", "ok"
    assert r.seeds[-1] == "two"                 # the oldest undone prompt
    text = out.getvalue()
    assert "  1  one" in text and "  3  three" in text


def test_undo_more_than_there_is(home, monkeypatch):
    r = Repl(monkeypatch, ["one", "/undo 5"]).run()
    assert len(r.messages) == 1


# ---- /export ---------------------------------------------------------------- #

def test_export_writes_markdown(home, monkeypatch):
    target = home / "out.md"
    Repl(monkeypatch, ["hello", f"/export {target}"]).run()
    text = target.read_text()
    assert text.startswith("# hello\n")
    assert "## You" in text and "## Assistant\n\nok" in text


def test_export_fences_tool_output_safely():
    msgs = [{"role": "system", "content": "S"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "grep", "arguments": {"pattern": "x"}}}]},
            {"role": "tool", "content": "has ``` inside", "name": "grep"}]
    md = commands.export_markdown(msgs)
    assert '**→ `grep`** `{"pattern": "x"}`' in md
    assert "````text\nhas ``` inside\n````" in md


# ---- /editor ---------------------------------------------------------------- #

def test_editor_text_is_a_prompt_even_if_it_looks_like_a_command(home, monkeypatch):
    def fake_editor(argv):
        with open(argv[-1], "w") as f:
            f.write("!rm -rf nothing\nsecond line\n")
        return 0
    monkeypatch.setattr(commands.subprocess, "call", fake_editor)
    monkeypatch.setenv("EDITOR", "myed --wait")
    r = Repl(monkeypatch, ["/editor"]).run()
    assert r.turns[0].endswith("!rm -rf nothing\nsecond line")


def test_editor_empty_sends_nothing(home, monkeypatch):
    monkeypatch.setattr(commands.subprocess, "call", lambda argv: 0)
    r = Repl(monkeypatch, ["/editor"]).run()
    assert r.turns == []


# ---- session titles --------------------------------------------------------- #

def test_first_prompt_becomes_the_session_title(home, monkeypatch, out):
    Repl(monkeypatch, ["fix the flaky test\nmore detail", "/save t1", "/sessions"]).run()
    data = json.loads((home.parent / "sessions" / "t1.json").read_text())
    assert data["title"] == "fix the flaky test"
    assert "t1 * — fix the flaky test ·" in out.getvalue()


# ---- @path#A-B and completion ------------------------------------------------ #

def test_file_ref_line_range(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "f.py").write_text("".join(f"l{i}\n" for i in range(1, 11)))
    out = expand_file_refs("see @f.py#3-4.")
    assert FILE_BLOCK_HEAD.format(path="f.py lines 3-4") in out
    assert "    3  l3\n    4  l4\n" in out and "l5" not in out


def test_completion(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "setup.py").write_text("")
    assert commands.complete("@s", False, []) == ["@setup.py", "@src/"]
    assert commands.complete("/he", True, []) == ["/help"]
    assert commands.complete("/re", True, ["/review"]) == ["/resume", "/review"]
    assert commands.complete("/he", False, []) == []
