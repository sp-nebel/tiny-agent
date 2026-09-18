import config
from agent import expand_file_refs, FILE_BLOCK_HEAD


def block(path, body):
    return FILE_BLOCK_HEAD.format(path=path) + "\n" + body


def test_existing_file_is_attached(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.py").write_text("x = 1\n")
    out = expand_file_refs("explain @a.py")
    assert out == "explain @a.py\n\n" + block("a.py", "    1  x = 1\n")


def test_non_files_are_left_alone(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    text = "mail me@example.com, ask @alice, see @missing.py"
    assert expand_file_refs(text) == text


def test_sentence_punctuation_is_stripped(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.py").write_text("x\n")
    out = expand_file_refs("look at @a.py.")
    assert FILE_BLOCK_HEAD.format(path="a.py") in out


def test_each_file_is_attached_once(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.py").write_text("x\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.py").write_text("y\n")
    out = expand_file_refs("@a.py vs @sub/b.py, then @a.py again")
    assert out.count(FILE_BLOCK_HEAD.format(path="a.py")) == 1
    assert out.index("[contents of a.py") < out.index("[contents of sub/b.py")


def test_directories_are_not_attached(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pkg").mkdir()
    assert expand_file_refs("@pkg") == "@pkg"


def test_long_file_carries_read_files_continuation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "MAX_READ_LINES", 3)
    (tmp_path / "big.txt").write_text("".join(f"l{i}\n" for i in range(10)))
    out = expand_file_refs("@big.txt")
    assert "[TRUNCATED. To continue reading, call read_file with start=4.]" in out
    assert "l3" not in out
