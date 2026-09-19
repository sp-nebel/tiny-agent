"""The first user message's context: cwd, repo/platform/date line, and the
global and project instructions files."""
import config
from agent import context_header, find_instructions, INSTRUCTIONS_HEAD


def test_walks_up_to_the_git_root(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    (tmp_path / "AGENTS.md").write_text("use tabs")
    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)
    assert find_instructions(str(sub)) == str(tmp_path / "AGENTS.md")


def test_nearest_file_wins_and_agents_before_claude(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / "AGENTS.md").write_text("root")
    sub = tmp_path / "pkg"
    sub.mkdir()
    (sub / "CLAUDE.md").write_text("pkg claude")
    assert find_instructions(str(sub)) == str(sub / "CLAUDE.md")
    (sub / "AGENTS.md").write_text("pkg agents")
    assert find_instructions(str(sub)) == str(sub / "AGENTS.md")


def test_outside_a_repo_only_the_cwd_counts(tmp_path):
    (tmp_path / "AGENTS.md").write_text("parent")
    sub = tmp_path / "x"
    sub.mkdir()
    assert find_instructions(str(sub)) is None


def test_header_includes_global_then_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()
    (tmp_path / "AGENTS.md").write_text("project rule")
    glob = tmp_path / "global.md"
    glob.write_text("global rule")
    monkeypatch.setattr(config, "GLOBAL_INSTRUCTIONS", str(glob))
    out = context_header()
    lines = out.split("\n")
    assert lines[0] == f"Working directory: {tmp_path}"
    assert lines[1].startswith(f"Git repo: yes, root {tmp_path} · Platform: ")
    assert out.index("global rule") < out.index("project rule")
    assert INSTRUCTIONS_HEAD.format(path=tmp_path / "AGENTS.md") + "\nproject rule" in out


def test_long_instructions_are_cut_with_a_pointer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "INSTRUCTIONS_MAX_CHARS", 10)
    (tmp_path / "AGENTS.md").write_text("x" * 50)
    out = context_header()
    assert "x" * 11 not in out
    assert f"[… cut at 10 chars; read_file {tmp_path / 'AGENTS.md'} with start=1 for the rest]" in out
