"""dispatch: tool-name and argument repair, and bad-call errors that name
the tool's real parameters."""
import config
from tools import dispatch, tool_name, _cap_output


def test_aliases_and_prefixes_resolve():
    assert tool_name("bash") == "run_cmd"
    assert tool_name("functions.Read_File") == "read_file"
    assert tool_name("cat") == "read_file"
    assert tool_name("write_file") == "edit_file"
    assert tool_name("nope") is None


def test_unknown_tool_lists_the_real_ones():
    out = dispatch("teleport", {})
    assert out.startswith("[unknown tool: teleport. The tools are: read_file, grep,")


def test_argument_aliases(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AUTO_YES", True)
    assert dispatch("bash", {"command": "echo hi"}) == "hi"
    p = tmp_path / "f.txt"
    p.write_text("a\n")
    assert dispatch("read", {"file_path": str(p)}).startswith("    1  a")


def test_write_file_creates_through_edit_file(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AUTO_YES", True)
    p = tmp_path / "new.txt"
    out = dispatch("write_file", {"path": str(p), "content": "hello\n"})
    assert out == f"[created {p}, 6 chars]"
    assert p.read_text() == "hello\n"


def test_missing_and_unknown_parameters_are_named():
    out = dispatch("edit_file", {"path": "x", "old_string": "a", "lines": 3})
    assert out == ("[bad args for edit_file: missing required 'new_string'; unknown "
                   "'lines'. Its parameters are: path, old_string, new_string, replace_all]")


def test_non_dict_args():
    assert dispatch("read_file", None).startswith("[bad args for read_file: missing required 'path'")


def test_capped_output_is_saved_to_a_file():
    text = "H" * 100 + "M" * 10000 + "T" * 100
    capped = _cap_output(text, max_chars=400)
    path = capped.split("full output saved in ")[1].split(" - ")[0]
    with open(path) as f:
        assert f.read() == text
    assert "grep or read_file it instead of running the command again" in capped
