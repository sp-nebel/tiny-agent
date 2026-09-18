import os
import glob

import config
from tools import find_files


def _expected(base, pattern):
    """Independently recompute what find_files should return, straight from
    glob.glob(recursive=True) + the SKIP_DIRS filter — the same semantics
    find_files is supposed to have. A regression back to a hand-rolled
    matcher (like the one this suite was written to catch) would diverge
    from this.
    """
    matches = glob.glob(os.path.join(base, pattern), recursive=True)
    out = []
    for m in sorted(matches):
        if config.SKIP_DIRS.intersection(m.split(os.sep)):
            continue
        out.append(m + ("/" if os.path.isdir(m) else ""))
    return out


def _actual(base, pattern, path_arg=None):
    result = find_files(pattern, path=path_arg if path_arg is not None else base)
    if result == "[no matches]":
        return []
    return result.split("\n")


def _make_tree(root):
    files = [
        "src/a.py",
        "src/test_foo.py",
        "src/sub/test_foo.py",
        "src/test_utils/helper.py",
        ".hidden.py",
        ".tox/lib/x.py",
        "node_modules/dep.py",
        "README.md",
    ]
    for rel in files:
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write("# x\n")
    return root


def test_recursive_star_star_py(tmp_path):
    root = _make_tree(str(tmp_path))
    assert set(_actual(root, "**/*.py")) == set(_expected(root, "**/*.py"))


def test_star_does_not_cross_slash_under_star_star(tmp_path):
    root = _make_tree(str(tmp_path))
    actual = set(_actual(root, "src/**/test_*.py"))
    expected = set(_expected(root, "src/**/test_*.py"))
    assert actual == expected
    # The hand-rolled matcher this suite replaced let 'test_*' eat 'utils/'
    # via an embedded '.*' — guard against that regression explicitly.
    assert not any("test_utils" in p for p in actual)


def test_non_recursive_pattern_stays_shallow(tmp_path):
    root = _make_tree(str(tmp_path))
    src = os.path.join(root, "src")
    actual = set(_actual(src, "*.py"))
    expected = set(_expected(src, "*.py"))
    assert actual == expected
    assert not any("sub" in p for p in actual)


def test_star_star_alone(tmp_path):
    root = _make_tree(str(tmp_path))
    src = os.path.join(root, "src")
    assert set(_actual(src, "**")) == set(_expected(src, "**"))


def test_star_star_embedded_in_path_argument(tmp_path):
    root = _make_tree(str(tmp_path))
    src_glob = os.path.join(root, "src", "**")
    actual = _actual(root, "*.py", path_arg=src_glob)
    expected = _expected(src_glob, "*.py")
    assert set(actual) == set(expected)
    # Depth beyond one level must still be reached (this is what a
    # pattern-only '**' check would miss, since the '**' lives in `path`).
    assert any("sub" in p for p in actual)


def test_absolute_pattern_with_star_star(tmp_path):
    root = _make_tree(str(tmp_path))
    abs_pattern = os.path.join(root, "**", "*.py")
    actual = _actual(".", abs_pattern)
    expected = _expected(".", abs_pattern)
    assert set(actual) == set(expected)
    assert any(root in p for p in actual)


def test_hidden_dirs_and_files_excluded(tmp_path):
    root = _make_tree(str(tmp_path))
    actual = _actual(root, "**/*.py")
    assert not any(".hidden.py" in p for p in actual)
    assert not any(".tox" in p for p in actual)


def test_skip_dirs_pruned(tmp_path):
    root = _make_tree(str(tmp_path))
    actual = _actual(root, "**/*.py")
    assert not any("node_modules" in p for p in actual)
