import config
import tools
from tools import append_file


def _read(path):
    with open(path, encoding="utf-8", newline="") as f:
        return f.read()


def test_appends_to_file_ending_in_newline(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AUTO_YES", True)
    p = tmp_path / "notes.md"
    p.write_text("one\ntwo\n")
    result = append_file(str(p), "three\n")
    assert _read(p) == "one\ntwo\nthree\n"
    # The line count lets the model aim a follow-up read at the new tail.
    assert result == f"[appended 6 chars to {p}; it now has 3 lines]"


def test_inserts_separator_when_trailing_newline_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AUTO_YES", True)
    p = tmp_path / "notes.md"
    p.write_text("one\ntwo")
    append_file(str(p), "three\n")
    assert _read(p) == "one\ntwo\nthree\n"


def test_creates_missing_file_and_parent_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AUTO_YES", True)
    p = tmp_path / "new" / "dir" / "log.txt"
    result = append_file(str(p), "first\n")
    assert result == f"[created {p}, 6 chars]"
    assert _read(p) == "first\n"


def test_empty_existing_file_gets_no_leading_newline(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AUTO_YES", True)
    p = tmp_path / "empty.txt"
    p.write_text("")
    result = append_file(str(p), "hello\n")
    assert _read(p) == "hello\n"
    assert result.startswith("[appended")


def test_preserves_crlf_line_endings(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AUTO_YES", True)
    p = tmp_path / "win.txt"
    p.write_bytes(b"a\r\nb\r\n")
    append_file(str(p), "c\nd\n")
    assert p.read_bytes() == b"a\r\nb\r\nc\r\nd\r\n"


def test_binary_file_refused_and_untouched(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AUTO_YES", True)
    p = tmp_path / "blob.bin"
    p.write_bytes(b"\xff\xfe\x00\x01")
    result = append_file(str(p), "text")
    assert result == f"[binary file, cannot append: {p}]"
    assert p.read_bytes() == b"\xff\xfe\x00\x01"


def test_decline_with_reason_is_passed_back_and_file_untouched(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "confirm", lambda msg: (False, "put it in README instead"))
    p = tmp_path / "notes.md"
    p.write_text("one\n")
    result = append_file(str(p), "two\n")
    assert "put it in README instead" in result
    assert "do not repeat the same call" in result
    assert _read(p) == "one\n"


def test_registered_with_dispatch_and_schema():
    assert tools.TOOLS["append_file"] is append_file
    names = [s["function"]["name"] for s in config.TOOL_SCHEMAS]
    assert "append_file" in names
