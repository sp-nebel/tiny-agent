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


def test_timeout_keeps_partial_output_and_does_not_wait_for_grandchildren(monkeypatch):
    # `sleep 30 &` holds the pipe open: subprocess.run's timeout killed only
    # the shell and then blocked on the grandchild for the full 30s.
    import time
    monkeypatch.setattr(config, "AUTO_YES", True)
    monkeypatch.setattr(config, "CMD_TIMEOUT", 1)
    t0 = time.monotonic()
    result = run_cmd("echo started; sleep 30 & sleep 30")
    assert time.monotonic() - t0 < 10
    assert result.startswith("started\n[timed out after 1s")
    assert "timeout=SECONDS" in result


def test_timeout_parameter_is_clamped(monkeypatch):
    monkeypatch.setattr(config, "AUTO_YES", True)
    monkeypatch.setattr(config, "MAX_CMD_TIMEOUT", 1)
    assert "[timed out after 1s" in run_cmd("sleep 5", timeout=9999)


def test_command_reading_stdin_gets_eof_instead_of_hanging(monkeypatch):
    monkeypatch.setattr(config, "AUTO_YES", True)
    monkeypatch.setattr(config, "CMD_TIMEOUT", 5)
    assert run_cmd("cat; echo done") == "done"


def test_ctrl_c_kills_the_command_and_carries_its_output(monkeypatch):
    import os, signal, threading, time
    import pytest
    from tools import CommandInterrupted
    monkeypatch.setattr(config, "AUTO_YES", True)
    threading.Timer(0.5, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
    t0 = time.monotonic()
    with pytest.raises(CommandInterrupted) as exc:
        run_cmd("echo partial; sleep 30")
    assert time.monotonic() - t0 < 10
    assert exc.value.output == "partial"
