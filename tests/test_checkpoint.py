import os
import subprocess

import pytest

import checkpoint
from agent import undo_last_turn


def git(root, *args):
    return subprocess.run(["git"] + list(args), cwd=root, check=True,
                          capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "t@example.com")
    git(tmp_path, "config", "user.name", "t")
    (tmp_path / "keep.txt").write_text("keep\n")
    (tmp_path / "edit.txt").write_text("before\n")
    (tmp_path / "gone.txt").write_text("gone\n")
    (tmp_path / ".gitignore").write_text("ignored.txt\n")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "init")
    return tmp_path


def test_outside_a_repo_there_is_no_snapshot(tmp_path):
    assert checkpoint.snapshot(str(tmp_path)) is None


def test_restore_undoes_edits_deletes_and_creations(repo):
    (repo / "wip.txt").write_text("untracked user work\n")   # pre-turn, untracked
    snap = checkpoint.snapshot(str(repo))
    assert snap is not None

    (repo / "edit.txt").write_text("after\n")
    (repo / "gone.txt").unlink()
    (repo / "new.txt").write_text("new\n")
    (repo / "pkg" / "sub").mkdir(parents=True)
    (repo / "pkg" / "sub" / "mod.py").write_text("x = 1\n")
    (repo / "wip.txt").write_text("clobbered\n")

    changes, head_moved = checkpoint.restore(snap)

    assert dict((p, s) for s, p in changes) == {
        "edit.txt": "M", "gone.txt": "D", "new.txt": "A",
        "pkg/sub/mod.py": "A", "wip.txt": "M",
    }
    assert not head_moved
    assert (repo / "edit.txt").read_text() == "before\n"
    assert (repo / "gone.txt").read_text() == "gone\n"
    assert (repo / "wip.txt").read_text() == "untracked user work\n"
    assert not (repo / "new.txt").exists()
    assert not (repo / "pkg").exists()        # directories it created go too


def test_real_index_and_head_are_untouched(repo):
    (repo / "edit.txt").write_text("staged by the user\n")
    git(repo, "add", "edit.txt")
    staged = git(repo, "diff", "--cached")
    head   = git(repo, "rev-parse", "HEAD")

    snap = checkpoint.snapshot(str(repo))
    (repo / "keep.txt").write_text("changed\n")
    checkpoint.restore(snap)

    assert git(repo, "diff", "--cached") == staged
    assert git(repo, "rev-parse", "HEAD") == head
    assert git(repo, "stash", "list") == ""


def test_ignored_files_are_out_of_scope(repo):
    snap = checkpoint.snapshot(str(repo))
    (repo / "ignored.txt").write_text("build output\n")
    changes, _ = checkpoint.restore(snap)
    assert changes == []
    assert (repo / "ignored.txt").exists()


def test_commit_during_turn_is_reported_not_reset(repo):
    snap = checkpoint.snapshot(str(repo))
    (repo / "edit.txt").write_text("committed\n")
    git(repo, "commit", "-qam", "agent commit")
    head = git(repo, "rev-parse", "HEAD")

    changes, head_moved = checkpoint.restore(snap)

    assert head_moved
    assert git(repo, "rev-parse", "HEAD") == head
    assert (repo / "edit.txt").read_text() == "before\n"


def test_snapshot_works_from_a_subdirectory(repo):
    (repo / "sub").mkdir()
    snap = checkpoint.snapshot(str(repo / "sub"))
    (repo / "edit.txt").write_text("after\n")
    checkpoint.restore(snap)
    assert (repo / "edit.txt").read_text() == "before\n"


# ---- undo_last_turn -------------------------------------------------------- #

def test_undo_with_nothing_to_undo():
    messages = [{"role": "system", "content": "S"}]
    assert undo_last_turn(messages, []) is None
    assert messages == [{"role": "system", "content": "S"}]


def test_undo_cuts_the_turn_and_restores_files(repo, monkeypatch):
    monkeypatch.chdir(repo)
    messages = [{"role": "system", "content": "S"},
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "done"}]
    turns = [{"msg_index": 1, "prompt": "first", "first": True,
              "cwd": str(repo), "snap": None},
             {"msg_index": 3, "prompt": "second", "first": False,
              "cwd": str(repo), "snap": checkpoint.snapshot(str(repo))}]
    messages += [{"role": "user", "content": "second"},
                 {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "edit_file"}}]},
                 {"role": "tool", "content": "[edited]"},
                 {"role": "assistant", "content": "edited it"}]
    (repo / "edit.txt").write_text("after\n")

    rec = undo_last_turn(messages, turns)

    assert rec["prompt"] == "second" and rec["first"] is False
    assert [m["content"] for m in messages] == ["S", "first", "done"]
    assert (repo / "edit.txt").read_text() == "before\n"
    assert len(turns) == 1

    rec = undo_last_turn(messages, turns)       # no snapshot: conversation only
    assert rec["first"] is True
    assert messages == [{"role": "system", "content": "S"}]
    assert turns == []


def test_undo_returns_to_the_turns_cwd(repo, tmp_path_factory, monkeypatch):
    elsewhere = tmp_path_factory.mktemp("elsewhere")
    monkeypatch.chdir(elsewhere)
    messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "t"}]
    undo_last_turn(messages, [{"msg_index": 1, "prompt": "t", "first": True,
                               "cwd": str(repo), "snap": None}])
    assert os.getcwd() == str(repo)
