"""run_turn's handling of individual tool calls: the repeated-call note, the
alias name shown and recorded, and a run_cmd stopped with Ctrl-C."""
import pytest

import agent
import config
from agent import run_turn, REPEAT_CALL_NOTE, USER_STOPPED_CMD
from tools import CommandInterrupted
from test_run_turn import env, reply, call, fresh, assert_well_formed   # noqa: F401


def test_identical_call_and_result_three_times_gets_the_note(env, monkeypatch):
    monkeypatch.setattr(config, "REPEAT_CALL_LIMIT", 3)
    model = env["install"]([reply(tool_calls=[call()]) for _ in range(3)]
                           + [reply(content="done")])
    messages = fresh()
    run_turn(messages)
    results = [m["content"] for m in model.calls[-1] if m["role"] == "tool"]
    assert results[:2] == ["result", "result"]
    assert results[2] == "result\n" + REPEAT_CALL_NOTE.format(name="grep", n=3)


def test_changing_result_resets_the_count(env, monkeypatch):
    monkeypatch.setattr(config, "REPEAT_CALL_LIMIT", 3)
    outputs = iter(["fail 1", "fail 2", "fail 2", "pass"])

    def fake_dispatch(name, args):
        return next(outputs)
    monkeypatch.setattr(agent, "dispatch", fake_dispatch)
    model = env["install"]([reply(tool_calls=[call("run_cmd", cmd="pytest")]) for _ in range(4)]
                           + [reply(content="done")])
    run_turn(fresh())
    results = [m["content"] for m in model.calls[-1] if m["role"] == "tool"]
    assert not any("exact run_cmd call" in r for r in results)


def test_alias_is_shown_and_recorded_as_the_real_tool(env):
    model = env["install"]([reply(tool_calls=[call("bash", command="ls")]),
                            reply(content="done")])
    messages = fresh()
    run_turn(messages)
    assert env["dispatched"] == [("bash", {"command": "ls"})]
    assert [m.get("name") for m in messages if m["role"] == "tool"] == ["run_cmd"]


def test_ctrl_c_in_run_cmd_reports_partial_output(env, monkeypatch):
    def interrupted(name, args):
        raise CommandInterrupted("half done")
    monkeypatch.setattr(agent, "dispatch", interrupted)
    env["install"]([reply(tool_calls=[call("run_cmd", cmd="make"), call()])])
    messages = fresh()
    with pytest.raises(KeyboardInterrupt):
        run_turn(messages)
    tools = [m["content"] for m in messages if m["role"] == "tool"]
    assert tools == ["half done\n" + USER_STOPPED_CMD, "[interrupted before this tool ran]"]
    assert_well_formed(messages)


def test_repeat_note_does_not_hide_a_failed_command(env, monkeypatch):
    # The note goes after "[exit 1]"; the display must still see the failure.
    import io
    from rich.console import Console
    buf = io.StringIO()
    monkeypatch.setattr(config, "console", Console(file=buf, force_terminal=False, width=200))
    monkeypatch.setattr(config, "REPEAT_CALL_LIMIT", 2)
    env["tool_result"] = "FAILED test_x\n[exit 1]"
    env["install"]([reply(tool_calls=[call("run_cmd", cmd="pytest")]) for _ in range(2)]
                   + [reply(content="done")])
    run_turn(fresh())
    shown = buf.getvalue()
    assert "exit 0" not in shown
    assert shown.count("FAILED test_x") == 2      # body printed both times
    assert "exact run_cmd call 2 times" in shown
