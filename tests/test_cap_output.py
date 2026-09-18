import config
from tools import _cap_output, run_cmd


def test_under_limit_is_unchanged():
    text = "hello world"
    assert _cap_output(text, max_chars=1000) == text


def test_exactly_at_limit_is_unchanged():
    text = "x" * 500
    assert _cap_output(text, max_chars=500) == text


def test_over_limit_keeps_head_and_tail():
    text = "H" * 100 + "M" * 10000 + "T" * 100
    capped = _cap_output(text, max_chars=400)
    assert capped.startswith("H" * 100)
    assert capped.endswith("T" * 100)
    assert "chars elided" in capped
    assert len(capped) < len(text)


def test_default_max_chars_from_config(monkeypatch):
    monkeypatch.setattr(config, "MAX_TOOL_OUTPUT_CHARS", 50)
    text = "Z" * 200
    capped = _cap_output(text)
    assert "chars elided" in capped
    assert len(capped) < len(text)


def test_run_cmd_uses_cap_output_for_oversized_commands(monkeypatch):
    monkeypatch.setattr(config, "AUTO_YES", True)
    monkeypatch.setattr(config, "MAX_CMD_CHARS", 200)
    result = run_cmd("python3 -c \"print('A'*1000 + 'END')\"")
    assert "chars elided" in result
    assert result.rstrip().endswith("END")
