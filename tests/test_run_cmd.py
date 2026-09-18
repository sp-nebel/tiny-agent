import config
from tools import run_cmd


def test_failure_with_output_reports_the_exit_code(monkeypatch):
    monkeypatch.setattr(config, "AUTO_YES", True)
    assert run_cmd("echo boom; exit 2") == "boom\n[exit 2]"


def test_success_with_output_adds_nothing(monkeypatch):
    monkeypatch.setattr(config, "AUTO_YES", True)
    assert run_cmd("echo fine") == "fine"


def test_no_output_still_reports_the_code(monkeypatch):
    monkeypatch.setattr(config, "AUTO_YES", True)
    assert run_cmd("true") == "[exit 0, no output]"
    assert run_cmd("false") == "[exit 1, no output]"


def test_exit_code_survives_the_output_cap(monkeypatch):
    monkeypatch.setattr(config, "AUTO_YES", True)
    monkeypatch.setattr(config, "MAX_CMD_CHARS", 100)
    result = run_cmd("python3 -c \"print('A'*1000)\"; exit 3")
    assert result.endswith("\n[exit 3]")
    assert "chars elided" in result
