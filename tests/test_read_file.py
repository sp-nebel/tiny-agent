import config
from tools import read_file


def _parse_notice(body):
    """Extract (start, end, total) from the '[lines S-E of T ...]' notice."""
    first = body.splitlines()[0]
    assert first.startswith("[lines ")
    rng, rest = first[len("[lines "):].split(" of ", 1)
    start, end = (int(x) for x in rng.split("-"))
    total = int(rest.split(" ")[0])
    return start, end, total


def test_notice_is_truthful_for_long_lines(tmp_path, monkeypatch):
    # 40 lines of ~400 chars each — comfortably over MAX_TOOL_OUTPUT_CHARS
    # (set small here) well before MAX_READ_LINES would kick in, mimicking
    # minified/generated code.
    monkeypatch.setattr(config, "MAX_TOOL_OUTPUT_CHARS", 2000)
    lines = [("X" * 400) + "\n" for _ in range(40)]
    p = tmp_path / "big.txt"
    p.write_text("".join(lines))

    result = read_file(str(p))
    start, end, total = _parse_notice(result)
    assert start == 1
    assert total == 40
    assert end < total  # genuinely truncated, not a false claim of completeness

    # Every line the notice claims is present (1..end) must actually be
    # fully intact in the body — no mid-line elision marker anywhere.
    assert "chars elided" not in result
    body_lines = set(result.splitlines())
    for n in range(start, end + 1):
        assert (f"{n:5}  " + "X" * 400) in body_lines


def test_continuation_recovers_the_rest_of_the_file(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MAX_TOOL_OUTPUT_CHARS", 2000)
    lines = [("X" * 400) + "\n" for _ in range(40)]
    p = tmp_path / "big.txt"
    p.write_text("".join(lines))

    seen = set()
    start = 1
    for _ in range(100):  # generous cap against an infinite loop on a bug
        result = read_file(str(p), start=start)
        first_line = result.splitlines()[0]
        if first_line.startswith("[lines "):
            s, e, total = _parse_notice(result)
            seen.update(range(s, e + 1))
            start = e + 1
        else:
            # Last call: no notice, remaining lines returned in full.
            for ln in result.splitlines():
                stripped = ln.strip()
                if stripped:
                    n = int(ln[:5])
                    seen.add(n)
            break
    assert seen == set(range(1, 41))


def test_single_line_exceeding_budget_gets_a_truncation_note(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MAX_TOOL_OUTPUT_CHARS", 100)
    p = tmp_path / "one_huge_line.txt"
    p.write_text("Y" * 5000 + "\n" + "second line\n")

    result = read_file(str(p))
    assert "[line 1 truncated at 100 chars]" in result


def test_short_file_has_no_notice(tmp_path):
    p = tmp_path / "small.py"
    p.write_text("x = 1\ny = 2\n")
    result = read_file(str(p))
    assert not result.startswith("[lines ")
    assert "x = 1" in result and "y = 2" in result
