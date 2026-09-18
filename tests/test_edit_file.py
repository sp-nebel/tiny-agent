import os

import config
from tools import edit_file, read_file


def _write(path, text):
    with open(path, "w") as f:
        f.write(text)


def test_line_number_prefix_detected_and_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AUTO_YES", True)
    p = str(tmp_path / "f.py")
    _write(p, "def foo():\n    return 1\n")

    # Grab the exact numbered column read_file would have shown, then use it
    # (uncorrected) as old_string — the common small-model mistake.
    numbered = read_file(p)
    line2 = [ln for ln in numbered.splitlines() if "return 1" in ln][0] + "\n"

    result = edit_file(p, line2, "    return 2\n")
    assert "line-number" in result
    assert "Strip it" in result


def test_genuine_numeric_column_content_not_mistaken_for_line_numbers(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AUTO_YES", True)
    p = str(tmp_path / "report.txt")
    # Real content: no leading space before the numeric column, so it's NOT
    # padded to read_file's exact width-5 column format.
    content = "Report Q2\n42  widgets sold\n99  gadgets sold\n"
    _write(p, content)

    # old_string has an unrelated leading-space typo, so it fails to match
    # the file for a reason that has nothing to do with a line-number column.
    # The heuristic must not mistake the file's genuine "42  " column for
    # read_file's metadata and tell the model to strip real data.
    result = edit_file(p, " 42  widgets sold\n", " 42  widgets bought\n")
    assert "line-number" not in result
    assert result == "[old_string not found; it must match the file exactly, whitespace included]"
    # The file must be untouched.
    with open(p) as f:
        assert f.read() == content


def test_normal_edit_replaces_unique_match(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AUTO_YES", True)
    p = str(tmp_path / "f.py")
    _write(p, "x = 1\ny = 2\n")
    result = edit_file(p, "x = 1", "x = 99")
    assert result.startswith("[edited")
    with open(p) as f:
        assert f.read() == "x = 99\ny = 2\n"


def test_create_new_file_with_empty_old_string(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AUTO_YES", True)
    p = str(tmp_path / "new.py")
    result = edit_file(p, "", "print('hi')\n")
    assert result.startswith("[created")
    with open(p) as f:
        assert f.read() == "print('hi')\n"


def test_ambiguous_match_requires_replace_all(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AUTO_YES", True)
    p = str(tmp_path / "f.py")
    _write(p, "x = 1\nx = 1\n")
    result = edit_file(p, "x = 1", "x = 2")
    assert "matches 2 times" in result


def test_empty_old_string_fills_an_existing_empty_file(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AUTO_YES", True)
    p = str(tmp_path / "touched.py")
    _write(p, "")
    result = edit_file(p, "", "print('hi')\n")
    assert result.startswith("[created")
    with open(p) as f:
        assert f.read() == "print('hi')\n"


def test_empty_old_string_refuses_a_file_with_content(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AUTO_YES", True)
    p = str(tmp_path / "f.py")
    _write(p, "x = 1\n")
    result = edit_file(p, "", "y = 2\n")
    assert "already exists and has content" in result
    assert "old_string" in result
    with open(p) as f:
        assert f.read() == "x = 1\n"


def _raw(path):
    with open(path, "rb") as f:
        return f.read()


def test_crlf_file_keeps_its_line_endings(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AUTO_YES", True)
    p = str(tmp_path / "win.txt")
    with open(p, "wb") as f:
        f.write(b"a\r\nb\r\nc\r\n")
    # LF strings, as the model writes them after reading through read_file.
    result = edit_file(p, "b\nc\n", "B\nC\nD\n")
    assert result.startswith("[edited")
    assert _raw(p) == b"a\r\nB\r\nC\r\nD\r\n"


def test_mixed_endings_only_the_replacement_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AUTO_YES", True)
    p = str(tmp_path / "mixed.txt")
    with open(p, "wb") as f:
        f.write(b"crlf1\r\ncrlf2\r\nlf1\nlf2\n")
    edit_file(p, "lf1\nlf2\n", "LF1\nLF2\n")          # the target is an LF region
    assert _raw(p) == b"crlf1\r\ncrlf2\r\nLF1\nLF2\n"
    edit_file(p, "crlf1\ncrlf2\n", "CRLF\n")          # and a CRLF one
    assert _raw(p) == b"CRLF\r\nLF1\nLF2\n"


def test_lf_file_is_unchanged_by_the_crlf_handling(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AUTO_YES", True)
    p = str(tmp_path / "unix.txt")
    with open(p, "wb") as f:
        f.write(b"a\nb\n")
    edit_file(p, "a\n", "A\n")
    assert _raw(p) == b"A\nb\n"


def test_line_number_prefix_detected_in_a_crlf_file(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AUTO_YES", True)
    p = str(tmp_path / "win.py")
    with open(p, "wb") as f:
        f.write(b"def foo():\r\n    return 1\r\n")
    line2 = [ln for ln in read_file(p).splitlines() if "return 1" in ln][0] + "\n"
    assert "line-number" in edit_file(p, line2, "    return 2\n")
