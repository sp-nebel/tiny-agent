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
