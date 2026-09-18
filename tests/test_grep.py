"""grep over both backends: the output shape must not depend on which one
is installed."""
import os
import shutil

import pytest

import config
import tools
from tools import grep

RG_WRAPPER = os.environ.get("TINY_AGENT_TEST_RG")   # optional: path to an rg binary


@pytest.fixture(params=["grep", "rg"])
def backend(request, monkeypatch):
    if request.param == "rg":
        rg = RG_WRAPPER or shutil.which("rg")
        if not rg:
            pytest.skip("rg not installed")
        monkeypatch.setattr(tools.shutil, "which", lambda name: rg)
        real_run = tools.subprocess.run

        def run(cmd, **kw):
            return real_run([rg] + cmd[1:], **kw) if cmd[0] == "rg" else real_run(cmd, **kw)
        monkeypatch.setattr(tools.subprocess, "run", run)
    else:
        monkeypatch.setattr(tools.shutil, "which", lambda name: None)
    return request.param


@pytest.fixture
def tree(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.py").write_text("import os\nx = 1\ny = os.sep\n")
    (tmp_path / "b.txt").write_text("os here\n")
    (tmp_path / "we:ird-1.py").write_text("os\n")
    return tmp_path


def test_grouped_by_file(backend, tree):
    out = grep("os", "a.py")
    assert out == "a.py\n1:import os\n3:y = os.sep"


def test_include_glob(backend, tree):
    out = grep("os", ".", include="*.txt")
    assert out.splitlines() == ["./b.txt", "1:os here"]


def test_paths_with_colons_and_dashes_parse(backend, tree):
    out = grep("os", ".", include="we*")
    assert out.splitlines() == ["./we:ird-1.py", "1:os"]


def test_context_lines_and_separators(backend, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "f.py").write_text("".join(f"line{i}\n" for i in range(1, 21)).replace("line10", "HIT").replace("line2\n", "HIT\n"))
    out = grep("HIT", "f.py", context=1)
    assert out.splitlines() == ["f.py", "1-line1", "2:HIT", "3-line3", "--",
                                "9-line9", "10:HIT", "11-line11"]


def test_cap_counts_matches_and_says_how_to_narrow(backend, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "MAX_GREP_HITS", 3)
    (tmp_path / "f.py").write_text("hit\n" * 10)
    out = grep("hit", "f.py")
    assert out.splitlines()[1:4] == ["1:hit", "2:hit", "3:hit"]
    assert out.endswith("[+7 more matches; narrow the path, the pattern, or use include]")


def test_long_lines_are_cut(backend, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "GREP_MAX_LINE_CHARS", 10)
    (tmp_path / "min.js").write_text("hit" + "x" * 5000 + "\n")
    out = grep("hit", "min.js")
    assert out.splitlines()[1] == "1:hitxxxxxxx [line cut]"


def test_no_matches_and_errors(backend, tree):
    assert grep("zzz", ".") == "[no matches]"
    assert grep("(", ".").startswith("[grep error")
