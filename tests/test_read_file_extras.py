"""read_file's footers and the cases it used to refuse or answer tersely:
directories, latin-1 files, missing paths."""
from tools import read_file


def test_end_of_file_footer(tmp_path):
    p = tmp_path / "f.py"
    p.write_text("a\nb\nc\n")
    assert read_file(str(p), start=2) == "    2  b\n    3  c\n[end of file, 3 lines]"


def test_directory_returns_listing(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "x.py").write_text("")
    out = read_file(str(tmp_path))
    assert out.splitlines() == [f"[{tmp_path} is a directory, not a file. Its entries:]",
                                "sub/", "x.py"]


def test_latin1_is_shown_not_refused(tmp_path):
    p = tmp_path / "l.py"
    p.write_bytes("s = 'café'\n".encode("latin-1"))
    out = read_file(str(p))
    assert "not UTF-8" in out.splitlines()[0]
    assert "    1  s = 'café'" in out


def test_nul_byte_means_binary(tmp_path):
    p = tmp_path / "b.bin"
    p.write_bytes(b"abc\0def")
    assert read_file(str(p)) == f"[binary file, cannot display: {p}]"


def test_missing_file_did_you_mean(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent.py").write_text("")
    assert read_file("agnet.py") == "[no such file: agnet.py. Did you mean: agent.py?]"
    assert read_file("zzzz.qq") == "[no such file: zzzz.qq.]"


def test_line_numbers_follow_universal_newlines(tmp_path):
    # \r\n and a lone \r end a line; a form feed does not.
    p = tmp_path / "f.txt"
    p.write_bytes(b"a\r\nb\rc\x0cd\n")
    assert read_file(str(p)) == "    1  a\n    2  b\n    3  c\x0cd\n[end of file, 3 lines]"
