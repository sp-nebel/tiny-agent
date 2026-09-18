"""edit_file's fallbacks after an exact match fails, and its smaller guards:
identical strings, "did you mean" paths, non-UTF-8 files, and the syntax
check reported after a write."""
import config
from tools import edit_file


def _write(path, text):
    with open(path, "w") as f:
        f.write(text)


def _read(path):
    with open(path) as f:
        return f.read()


# ---- fuzzy fallback --------------------------------------------------------- #

def test_indentation_shift_carries_into_new_string(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AUTO_YES", True)
    p = str(tmp_path / "f.py")
    _write(p, "class A:\n    def f(self):\n        return 1\n")
    # The model dropped one level of indent from both strings.
    result = edit_file(p, "def f(self):\n    return 1\n", "def f(self):\n    return 2\n")
    assert result == f"[edited {p}: 1 replacement (matched ignoring indentation)]"
    assert _read(p) == "class A:\n    def f(self):\n        return 2\n"


def test_trailing_whitespace(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AUTO_YES", True)
    p = str(tmp_path / "f.txt")
    _write(p, "a = 1   \nb = 2\n")
    result = edit_file(p, "a = 1\nb = 2", "a = 3\nb = 2")
    assert "(matched ignoring trailing whitespace)" in result
    assert _read(p) == "a = 3\nb = 2\n"


def test_curly_quotes(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AUTO_YES", True)
    p = str(tmp_path / "f.py")
    _write(p, 'print("hi")\n')
    result = edit_file(p, "print(\u201chi\u201d)", 'print("bye")')
    assert "quote/dash style" in result
    assert _read(p) == 'print("bye")\n'


def test_splices_at_the_matched_line_not_an_earlier_substring(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AUTO_YES", True)
    p = str(tmp_path / "f.py")
    # "  foo()" is also a substring of the first line; str.replace would hit it.
    _write(p, "    foo()  # x\n  foo()\t\n")
    edit_file(p, "  foo()\n", "  bar()\n")
    assert _read(p) == "    foo()  # x\n  bar()\n"


def test_ambiguous_match_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AUTO_YES", True)
    p = str(tmp_path / "f.py")
    _write(p, "if a:\n    x = 1\nif b:\n  x = 1\n")
    result = edit_file(p, "x = 1 \n", "x = 2\n")
    assert "ignoring indentation it matches 2 places" in result
    assert _read(p) == "if a:\n    x = 1\nif b:\n  x = 1\n"


def test_partial_line_is_not_fuzzy_matched(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AUTO_YES", True)
    p = str(tmp_path / "f.py")
    _write(p, "value = compute(a,  b)\n")
    assert edit_file(p, "compute(a, b)", "compute(b, a)").startswith("[old_string not found")


def test_crlf_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AUTO_YES", True)
    p = tmp_path / "f.txt"
    p.write_bytes(b"top\r\n    a = 1\r\n    b = 2\r\nend\r\n")
    edit_file(str(p), "a = 1\nb = 2\n", "a = 9\nb = 2\n")
    assert p.read_bytes() == b"top\r\n    a = 9\r\n    b = 2\r\nend\r\n"


def test_crlf_single_line_without_trailing_newline(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AUTO_YES", True)
    p = tmp_path / "f.txt"
    p.write_bytes(b"  a = 1\r\nb\r\n")
    edit_file(str(p), "a = 1", "a = 2")
    assert p.read_bytes() == b"  a = 2\r\nb\r\n"


def test_line_number_prefix_still_taught_not_forgiven(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AUTO_YES", True)
    p = str(tmp_path / "f.py")
    _write(p, "x = 1\n")
    assert "line-number" in edit_file(p, "    1  x = 1\n", "x = 2\n")


# ---- small fixes ------------------------------------------------------------ #

def test_identical_strings_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AUTO_YES", True)
    p = str(tmp_path / "f.py")
    _write(p, "x = 1\n")
    assert "identical" in edit_file(p, "x = 1", "x = 1")


def test_missing_file_suggests_close_names(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "lib").mkdir()
    _write(tmp_path / "lib" / "utils.py", "x\n")
    result = edit_file("src/util.py", "x", "y")
    assert result == ("[no such file: src/util.py. Did you mean: lib/utils.py? "
                      "To create it, pass an empty old_string.]")


def test_non_utf8_file_is_not_edited(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AUTO_YES", True)
    p = tmp_path / "l.py"
    p.write_bytes("x = 'caf\u00e9'\n".encode("latin-1"))
    assert "not UTF-8" in edit_file(str(p), "x", "y")
    assert p.read_bytes() == "x = 'caf\u00e9'\n".encode("latin-1")


# ---- syntax check ----------------------------------------------------------- #

def test_syntax_error_reported_after_edit(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AUTO_YES", True)
    p = str(tmp_path / "f.py")
    _write(p, "def f():\n    return 1\n")
    result = edit_file(p, "return 1", "return (1")
    assert result.startswith(f"[edited {p}: 1 replacement]\n[warning: {p} now has a "
                             f"syntax error, line 2")
    assert result.endswith("Fix it before moving on.]")


def test_preexisting_syntax_error_not_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AUTO_YES", True)
    p = str(tmp_path / "c.json")
    _write(p, '{\n  // comment\n  "a": 1\n}\n')
    assert "warning" not in edit_file(p, '"a": 1', '"a": 2')


def test_syntax_error_on_create(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AUTO_YES", True)
    assert "syntax error, line 1" in edit_file(str(tmp_path / "n.json"), "", '{"a": }')
    assert "warning" not in edit_file(str(tmp_path / "ok.py"), "", "x = 1\n")


def test_custom_syntax_check_command(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AUTO_YES", True)
    monkeypatch.setattr(config, "SYNTAX_CHECK_CMDS", {".sh": "bash -n {path}"})
    assert "syntax error" in edit_file(str(tmp_path / "s.sh"), "", "if true; then\n")
    assert "warning" not in edit_file(str(tmp_path / "ok.sh"), "", "echo hi\n")


def test_append_reports_syntax_error(tmp_path, monkeypatch):
    from tools import append_file
    monkeypatch.setattr(config, "AUTO_YES", True)
    p = str(tmp_path / "f.py")
    _write(p, "x = 1\n")
    assert "syntax error" in append_file(p, "def broken(:\n")
