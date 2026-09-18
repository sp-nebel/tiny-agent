"""run_turn over a scripted stand-in for call_ollama.

Covers the mid-turn features: queued interjections, Tab stop-and-steer,
the --check-every loop probe, and the step-limit / empty-retry nudges — and,
through all of them, the message-history invariants (every tool_calls
answered, no thinking or nudges left once the turn ends).

Nudges are stripped in run_turn's finally, so anything transient (a nudge's
presence, where an interjection landed) is asserted on the per-call
snapshots the fake model records, and the end state on `messages` itself.
"""
import pytest

import agent
import config
import ui
from agent import (run_turn, strip_nudges, _is_stray_nudge, _says_stuck,
                   STEP_LIMIT_NUDGE, EMPTY_RETRY_NUDGE, LOOP_CHECK_PREFIX,
                   INTERJECTION_PREFIX, STOP_NOTE_PREFIX, STOP_NOTE_DEFAULT)


def reply(content="", thinking="", tool_calls=None, cancelled=False, interrupted=False):
    return content, thinking, list(tool_calls or []), cancelled, interrupted, {}


def call(name="grep", **args):
    return {"function": {"name": name, "arguments": args or {"pattern": "x"}}}


class FakeModel:
    """Plays back a script of replies, one per call_ollama request.

    A script entry is either a reply tuple or a function of the messages the
    call received, for replies that must act mid-stream (queue an
    interjection, as a user typing during the reply would)."""

    def __init__(self, script):
        self.script = list(script)
        self.calls  = []          # snapshot of the message list per call
        self.lists  = []          # the list objects themselves, for identity checks

    def __call__(self, messages, timeout=None, retry_stall=False):
        self.calls.append([dict(m) for m in messages])
        self.lists.append(messages)
        assert self.script, "model called more times than the script allows"
        r = self.script.pop(0)
        return r(messages) if callable(r) else r


@pytest.fixture
def env(monkeypatch):
    """Patch the model, the tools and the steer prompt; keep the module-level
    interjection queue from bleeding between tests."""
    ui.take_interjections()
    state = {"dispatched": [], "tool_result": "result", "note": "", "on_dispatch": None}

    def fake_dispatch(name, args):
        state["dispatched"].append((name, args))
        if state["on_dispatch"]:
            state["on_dispatch"]()
        return state["tool_result"]

    def fake_read_prompt(prompt, seed=""):
        note = state["note"]
        if isinstance(note, BaseException):
            raise note
        return note

    monkeypatch.setattr(agent, "dispatch", fake_dispatch)
    monkeypatch.setattr(agent, "read_prompt", fake_read_prompt)

    def install(script):
        model = FakeModel(script)
        monkeypatch.setattr(agent, "call_ollama", model)
        return model

    state["install"] = install
    yield state
    ui.take_interjections()


def fresh():
    return [{"role": "system", "content": config.SYSTEM},
            {"role": "user", "content": "do the task"}]


def assert_well_formed(messages):
    """The invariants every turn must leave behind."""
    assert messages[0] == {"role": "system", "content": config.SYSTEM}
    for i, m in enumerate(messages):
        assert "thinking" not in m
        assert not _is_stray_nudge(m), m
        if m.get("tool_calls"):
            n = len(m["tool_calls"])
            follow = messages[i + 1:i + 1 + n]
            assert len(follow) == n and all(f["role"] == "tool" for f in follow)


def contents(messages):
    return [m.get("content") for m in messages]


# ---- basic round-trip ----------------------------------------------------- #

def test_tool_round_trip_then_answer(env):
    model = env["install"]([
        reply(thinking="look first", tool_calls=[call()]),
        reply(content="done"),
    ])
    messages = fresh()
    run_turn(messages)

    assert env["dispatched"] == [("grep", {"pattern": "x"})]
    # Thinking rode along to the second call so the model could build on it...
    assert model.calls[1][2].get("thinking") == "look first"
    # ...and is gone once the turn ended.
    assert_well_formed(messages)
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool", "assistant"]
    assert messages[-1]["content"] == "done"


# ---- interjections -------------------------------------------------------- #

def test_interjection_lands_after_the_tool_results(env):
    # Typed while the tool step ran: must not split tool_calls from results.
    def type_once():
        if not ui.interjections_pending() and len(env["dispatched"]) == 1:
            ui._interjections.append("also check tests/")
    env["on_dispatch"] = type_once
    model = env["install"]([
        reply(tool_calls=[call(), call("read_file", path="a.py")]),
        reply(content="done"),
    ])
    messages = fresh()
    run_turn(messages)

    second = model.calls[1]
    assert [m["role"] for m in second[-4:]] == ["assistant", "tool", "tool", "user"]
    assert second[-1]["content"] == INTERJECTION_PREFIX + "also check tests/"
    # Real user input, not a nudge: it survives the turn-end cleanup.
    assert INTERJECTION_PREFIX + "also check tests/" in contents(messages)
    assert_well_formed(messages)


def test_empty_reply_gets_retry_nudge(env):
    model = env["install"]([reply(), reply(content="ok")])
    messages = fresh()
    run_turn(messages)

    assert model.calls[1][-1]["content"] == EMPTY_RETRY_NUDGE
    # The nudge and the empty reply that prompted it are stripped at turn end.
    assert contents(messages) == [config.SYSTEM, "do the task", "ok"]


def test_empty_retry_nudge_suppressed_when_interjection_pending(env):
    def empty_while_user_types(messages):
        ui._interjections.append("try the other file")
        return reply()

    model = env["install"]([empty_while_user_types, reply(content="ok")])
    messages = fresh()
    run_turn(messages)

    second = contents(model.calls[1])
    assert EMPTY_RETRY_NUDGE not in second
    assert second[-1] == INTERJECTION_PREFIX + "try the other file"
    assert_well_formed(messages)


def test_cancel_leaves_queued_interjection_for_main(env):
    def cancelled_while_user_types(messages):
        ui._interjections.append("next thing")
        return reply(cancelled=True)

    env["install"]([cancelled_while_user_types])
    messages = fresh()
    run_turn(messages)

    assert ui.take_interjections() == ["next thing"]
    assert contents(messages) == [config.SYSTEM, "do the task"]


# ---- Tab: stop and steer -------------------------------------------------- #

def test_stop_commits_partial_without_its_tool_call(env):
    env["note"] = "use grep instead"
    model = env["install"]([
        reply(content="I will delete it", thinking="rm is fastest",
              tool_calls=[call("run_cmd", cmd="rm -rf build")], interrupted=True),
        reply(content="grepping instead"),
    ])
    messages = fresh()
    run_turn(messages)

    # The call the user stopped is exactly the one they didn't want run.
    assert env["dispatched"] == []
    second = model.calls[1]
    assert second[-2] == {"role": "assistant", "content": "I will delete it",
                          "thinking": "rm is fastest"}
    assert second[-1] == {"role": "user", "content": STOP_NOTE_PREFIX + "use grep instead"}
    # The partial reply and the note are kept; only the thinking goes.
    assert contents(messages) == [config.SYSTEM, "do the task", "I will delete it",
                                  STOP_NOTE_PREFIX + "use grep instead", "grepping instead"]
    assert_well_formed(messages)


@pytest.mark.parametrize("note", ["", "   ", EOFError()])
def test_empty_or_abandoned_note_falls_back_to_default(env, note):
    env["note"] = note
    model = env["install"]([reply(content="partial", interrupted=True), reply(content="ok")])
    run_turn(fresh())
    assert model.calls[1][-1]["content"] == STOP_NOTE_PREFIX + STOP_NOTE_DEFAULT


def test_stop_before_first_token_commits_only_the_note(env):
    env["note"] = "wait"
    model = env["install"]([reply(interrupted=True), reply(content="ok")])
    run_turn(fresh())
    assert [m["role"] for m in model.calls[1]] == ["system", "user", "user"]


def test_thinking_only_partial_is_removed_at_turn_end(env):
    env["note"] = "wait"
    env["install"]([reply(thinking="hmm", interrupted=True), reply(content="ok")])
    messages = fresh()
    run_turn(messages)
    # drop_thinking empties the shell, strip_nudges removes it; the note stays.
    assert contents(messages) == [config.SYSTEM, "do the task",
                                  STOP_NOTE_PREFIX + "wait", "ok"]


def test_stop_does_not_consume_a_step_or_stack_the_limit_nudge(env):
    env["note"] = "shorter please"
    model = env["install"]([
        reply(content="long answer…", interrupted=True),
        reply(content="short answer"),
    ])
    messages = fresh()
    run_turn(messages, max_steps=1)

    # Still on the single allowed step after the stop, so the model got a
    # second call, carrying the one step-limit nudge already sent.
    assert len(model.calls) == 2
    assert contents(model.calls[1]).count(STEP_LIMIT_NUDGE) == 1
    assert messages[-1]["content"] == "short answer"
    assert_well_formed(messages)


# ---- step limit ----------------------------------------------------------- #

def test_forced_last_step_drops_unanswered_tool_calls(env):
    model = env["install"]([
        reply(tool_calls=[call()]),
        reply(content="best effort", tool_calls=[call()]),
    ])
    messages = fresh()
    run_turn(messages, max_steps=2)

    assert model.calls[1][-1]["content"] == STEP_LIMIT_NUDGE
    assert len(env["dispatched"]) == 1
    assert "tool_calls" not in messages[-1]
    assert messages[-1]["content"] == "best effort"
    assert_well_formed(messages)


# ---- --check-every loop probe --------------------------------------------- #

def _is_probe(msgs):
    return (msgs[-1].get("content") or "").startswith(LOOP_CHECK_PREFIX)


def test_loop_check_continue_leaves_no_trace(env):
    model = env["install"]([
        reply(tool_calls=[call()]),
        reply(tool_calls=[call()]),
        reply(content="CONTINUE"),          # the probe
        reply(content="done"),
    ])
    messages = fresh()
    run_turn(messages, max_steps=0, check_every=2)

    probe_idx = [i for i, c in enumerate(model.calls) if _is_probe(c)]
    assert probe_idx == [2]
    assert "2 tool steps" in model.calls[2][-1]["content"]
    # The probe went out on a copy; the real list never held it.
    assert model.lists[2] is not messages
    assert not any(LOOP_CHECK_PREFIX in (m.get("content") or "") for m in model.calls[3])
    assert messages[-1]["content"] == "done"
    assert_well_formed(messages)


def test_loop_check_fires_again_after_a_fresh_budget(env):
    model = env["install"]([
        reply(tool_calls=[call()]),
        reply(content="CONTINUE"),
        reply(tool_calls=[call()]),
        reply(content="CONTINUE"),
        reply(content="done"),
    ])
    run_turn(fresh(), max_steps=0, check_every=1)
    assert [i for i, c in enumerate(model.calls) if _is_probe(c)] == [1, 3]


def test_loop_check_stuck_ends_turn_on_the_explanation(env):
    model = env["install"]([
        reply(tool_calls=[call()]),
        reply(tool_calls=[call()]),
        reply(content="STUCK: I kept grepping for the same symbol."),
    ])
    messages = fresh()
    run_turn(messages, max_steps=0, check_every=2)

    assert len(model.calls) == 3 and len(env["dispatched"]) == 2
    assert messages[-1] == {"role": "assistant",
                            "content": "STUCK: I kept grepping for the same symbol."}
    # The probe question itself is a nudge and is stripped at turn end.
    assert not any((m.get("content") or "").startswith(LOOP_CHECK_PREFIX) for m in messages)
    assert_well_formed(messages)


def test_loop_check_stuck_with_tool_calls_is_not_a_verdict(env):
    model = env["install"]([
        reply(tool_calls=[call()]),
        reply(content="STUCK", tool_calls=[call()]),   # probe answered with a tool call
        reply(content="done"),
    ])
    messages = fresh()
    run_turn(messages, max_steps=0, check_every=1)
    assert len(model.calls) == 3
    assert messages[-1]["content"] == "done"
    assert len(env["dispatched"]) == 1                 # the probe's call never ran


def test_loop_check_inert_when_it_equals_max_steps(env):
    # Documented: under --max-steps N a check every N steps can never fire.
    model = env["install"]([reply(tool_calls=[call()])] * 3)
    run_turn(fresh(), max_steps=3, check_every=3)
    assert not any(_is_probe(c) for c in model.calls)


def test_loop_check_disabled_with_zero(env):
    model = env["install"]([reply(tool_calls=[call()])] * 3 + [reply(content="done")])
    run_turn(fresh(), max_steps=0, check_every=0)
    assert not any(_is_probe(c) for c in model.calls)


def test_stop_during_probe_commits_only_the_note(env):
    env["note"] = "focus on the parser"
    model = env["install"]([
        reply(tool_calls=[call()]),
        reply(content="CONT", interrupted=True),   # Tab during the probe
        reply(content="done"),
    ])
    messages = fresh()
    run_turn(messages, max_steps=0, check_every=1)

    third = model.calls[2]
    assert third[-1]["content"] == STOP_NOTE_PREFIX + "focus on the parser"
    assert third[-2]["role"] == "tool"            # no partial probe reply committed
    assert "CONT" not in contents(messages)
    assert_well_formed(messages)


@pytest.mark.parametrize("text, stuck", [
    ("STUCK", True),
    ("stuck: kept reading", True),
    ("**STUCK** I looped", True),
    ("  > STUCK", True),
    ("CONTINUE", False),
    ("CONTINUE, I'm not stuck", False),
    ("", False),
    (None, False),
])
def test_says_stuck(text, stuck):
    assert _says_stuck(text) is stuck


# ---- turn-end cleanup ----------------------------------------------------- #

def test_turn_end_stubs_all_but_the_most_recent_outputs(env, monkeypatch):
    monkeypatch.setattr(config, "KEEP_FULL_TOOL_RESULTS", 1)
    monkeypatch.setattr(config, "TRIM_MIN_CHARS", 10)
    env["tool_result"] = "line\n" * 50
    env["install"]([reply(tool_calls=[call()])] * 3 + [reply(content="done")])
    messages = fresh()
    run_turn(messages)

    tools = [m for m in messages if m["role"] == "tool"]
    assert [t["content"].startswith(config.TRIM_PREFIX) for t in tools] == [True, True, False]
    assert "grep" in tools[0]["content"] and "250 chars" in tools[0]["content"]


def test_strip_nudges_keeps_real_user_input(env):
    messages = fresh() + [
        {"role": "user", "content": INTERJECTION_PREFIX + "hi"},
        {"role": "user", "content": STOP_NOTE_PREFIX + "stop"},
        {"role": "user", "content": LOOP_CHECK_PREFIX + "you have taken 3 tool steps…]"},
        {"role": "user", "content": STEP_LIMIT_NUDGE},
        {"role": "user", "content": EMPTY_RETRY_NUDGE},
        {"role": "assistant", "content": "  "},
        {"role": "assistant", "content": "", "tool_calls": [call()]},
        {"role": "tool", "content": "r", "name": "grep"},
    ]
    strip_nudges(messages, 2)
    assert contents(messages) == [config.SYSTEM, "do the task",
                                  INTERJECTION_PREFIX + "hi", STOP_NOTE_PREFIX + "stop",
                                  "", "r"]
