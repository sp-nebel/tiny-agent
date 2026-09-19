"""Display-only helpers: the stats line, one-line tool summaries, the
notification, and the live view's /thinking and queued-message notices."""
import io

import pytest

import config
import ui
from ui import fmt_stats, tool_call_label, tool_outcome, tool_failed, _render_stream


def test_stats_line_has_total_time_and_context_share():
    stats = {"prompt_eval_count": 100, "prompt_eval_duration": 2e9,
             "eval_count": 10, "eval_duration": 1e9}
    assert fmt_stats(stats, 3, elapsed=75, ctx_pct=41) == (
        "3 steps · prefill 100 tok in 2.0s · gen 10 tok @ 10.0 tok/s · 1m15s total · ctx ~41%")
    assert fmt_stats(stats, 1) == "prefill 100 tok in 2.0s · gen 10 tok @ 10.0 tok/s"


@pytest.mark.parametrize("name, args, label", [
    ("read_file", {"path": "a.py", "start": 10, "end": 40}, "read a.py:10-40"),
    ("read_file", {"path": "a.py"}, "read a.py"),
    ("grep", {"pattern": "foo", "path": "src", "include": "*.py"}, 'grep "foo" in src (*.py)'),
    ("grep", {"pattern": "foo"}, 'grep "foo"'),
    ("run_cmd", {"cmd": "pytest -q"}, "run_cmd pytest -q"),
])
def test_call_labels(name, args, label):
    assert tool_call_label(name, args) == label


@pytest.mark.parametrize("name, result, outcome, failed", [
    ("grep", "a.py\n1:x\n2:y\n[+5 more matches; narrow]", "7 matches", False),
    ("grep", "[no matches]", "no matches", False),
    ("read_file", "[lines 1-100 of 543 - file continues]\n...", "1-100 of 543", False),
    ("read_file", "    1  x\n[end of file, 1 line]", "1 line to the end", False),
    ("read_file", "[no such file: x]", "", True),
    ("run_cmd", "boom\n[exit 2]", "exit 2", True),
    ("run_cmd", "ok", "exit 0", False),
    ("run_cmd", "[exit 0, no output]", "exit 0", False),
    ("run_cmd", "part\n[timed out after 5s and was killed. …]", "timed out", True),
    ("list_dir", "a/\nb.py\n[+3 more]", "5 entries", False),
    ("edit_file", "[edited x: 1 replacement]\n[warning: x now has a syntax error, …]", "", True),
])
def test_outcomes_and_failures(name, result, outcome, failed):
    assert tool_outcome(name, result) == outcome
    assert tool_failed(name, result) is failed


def test_notify_only_after_a_long_turn(monkeypatch):
    out = io.StringIO()
    out.isatty = lambda: True
    monkeypatch.setattr(ui.sys, "stdout", out)
    monkeypatch.setattr(config, "NOTIFY", True)
    monkeypatch.setattr(config, "NOTIFY_AFTER", 20)
    monkeypatch.delenv("VTE_VERSION", raising=False)
    ui.mark_turn_start()
    ui.notify("done")
    assert out.getvalue() == ""
    monkeypatch.setattr(ui, "_turn_started", ui.time.monotonic() - 30)
    ui.notify("turn; finished")
    assert out.getvalue() == "\a\x1b]9;tiny-agent: turn, finished\x07"
    monkeypatch.setenv("VTE_VERSION", "7600")
    ui.notify("x")
    assert out.getvalue().endswith("\a\x1b]777;notify;tiny-agent;x\x07")


def test_notify_off(monkeypatch):
    out = io.StringIO()
    out.isatty = lambda: True
    monkeypatch.setattr(ui.sys, "stdout", out)
    monkeypatch.setattr(config, "NOTIFY", False)
    monkeypatch.setattr(ui, "_turn_started", ui.time.monotonic() - 999)
    ui.notify("x")
    assert out.getvalue() == ""


def test_hidden_thinking_and_queued_count(monkeypatch):
    monkeypatch.setattr(config, "SHOW_THINKING", False)
    ui.take_interjections()
    ui._interjections.extend(["a", "b"])
    try:
        text = _render_stream("secret plan here", "answer").plain
    finally:
        ui.take_interjections()
    assert "secret" not in text
    assert "thinking hidden — 3 words so far" in text
    assert "(2 messages queued for the model after this step)" in text
