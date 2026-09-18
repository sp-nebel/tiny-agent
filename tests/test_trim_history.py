import json

import config
from agent import trim_history, _total_tokens


def _set_thresholds(monkeypatch, *, trim_at=0, hard_at, keep_recent=2, trim_min=50):
    # trim_at=0 means "always past the soft-trim gate", so every call reaches
    # the hard-truncate backstop under test without needing a giant history.
    monkeypatch.setattr(config, "TRIM_AT_TOKENS", trim_at)
    monkeypatch.setattr(config, "HARD_TRUNCATE_AT_TOKENS", hard_at)
    monkeypatch.setattr(config, "KEEP_RECENT_MESSAGES", keep_recent)
    monkeypatch.setattr(config, "TRIM_MIN_CHARS", trim_min)
    monkeypatch.setattr(config, "KEEP_FULL_TOOL_RESULTS", 0)
    # The soft pass runs first on every call here (trim_at=0); keep its
    # thinking fallback out of the way so these tests exercise the backstop.
    monkeypatch.setattr(config, "TRIM_KEEP_THINKING", 4)


def test_backstop_truncates_oversized_non_protected_message(monkeypatch):
    _set_thresholds(monkeypatch, hard_at=10)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "A" * 5000},
        {"role": "user", "content": "pad1"},
        {"role": "user", "content": "pad2"},
    ]
    trim_history(messages)
    assert messages[1]["content"].startswith(config.TRIM_PREFIX)
    assert "was 5000 chars" in messages[1]["content"]


def test_backstop_is_idempotent_and_preserves_original_size(monkeypatch):
    _set_thresholds(monkeypatch, hard_at=10)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "A" * 5000},
        {"role": "user", "content": "pad1"},
        {"role": "user", "content": "pad2"},
    ]
    trim_history(messages)
    first_pass = messages[1]["content"]
    assert "was 5000 chars" in first_pass

    # Push the total back over the hard-truncate threshold with a *new*
    # oversized message, forcing a second real pass through the backstop
    # loop. The already-truncated message must be recognized (via the
    # startswith(TRIM_PREFIX) guard) and left untouched — not re-edited,
    # which would both bust the KV cache again for nothing and overwrite the
    # "was 5000 chars" figure with the now-much-smaller current length.
    messages.insert(2, {"role": "user", "content": "B" * 5000})
    trim_history(messages)

    assert messages[1]["content"] == first_pass
    assert "was 5000 chars" in messages[1]["content"]
    assert messages[2]["content"].startswith(config.TRIM_PREFIX)
    assert "was 5000 chars" in messages[2]["content"]


def test_thinking_dropped_on_non_protected_message_regardless_of_content_size(monkeypatch):
    _set_thresholds(monkeypatch, hard_at=10)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "short", "thinking": "T" * 5000},
        {"role": "user", "content": "pad1"},
        {"role": "user", "content": "pad2"},
    ]
    trim_history(messages)
    assert "thinking" not in messages[1]


def test_thinking_kept_on_most_recent_assistant_message_in_protected_tail(monkeypatch):
    _set_thresholds(monkeypatch, hard_at=10, keep_recent=3)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "", "thinking": "T" * 4000},
        {"role": "tool", "content": "result", "name": "grep"},
        {"role": "assistant", "content": "", "thinking": "T" * 4000},
    ]
    trim_history(messages)
    # The earlier protected assistant message loses its thinking (step 2)...
    assert "thinking" not in messages[2]
    # ...but the LAST assistant message keeps it, since that's the reasoning
    # the model is about to build on for its next step.
    assert messages[4].get("thinking") == "T" * 4000


def test_tool_calls_are_never_modified(monkeypatch):
    _set_thresholds(monkeypatch, hard_at=1_000_000, keep_recent=1)
    big_tool_calls = [{"function": {"name": "f", "arguments": {"x": "Z" * 5000}}}]
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "short", "tool_calls": list(big_tool_calls)},
        {"role": "user", "content": "pad"},
    ]
    before = json.dumps(messages[1]["tool_calls"])
    trim_history(messages)
    assert json.dumps(messages[1]["tool_calls"]) == before


def test_warns_when_still_over_budget_after_full_backstop(monkeypatch, capsys):
    # An impossible-to-satisfy threshold forces the backstop through all
    # three steps and out the other side still over budget.
    _set_thresholds(monkeypatch, hard_at=1, keep_recent=1)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "x", "thinking": "T" * 5000},
    ]
    trim_history(messages)
    out = capsys.readouterr().out
    assert "still exceeds" in out


# ---- soft pass ------------------------------------------------------------ #

def _soft(monkeypatch, *, trim_at, keep_thinking=4, trim_min=50):
    # Backstop out of reach, so only the soft pass acts.
    monkeypatch.setattr(config, "TRIM_AT_TOKENS", trim_at)
    monkeypatch.setattr(config, "HARD_TRUNCATE_AT_TOKENS", 10_000_000)
    monkeypatch.setattr(config, "TRIM_MIN_CHARS", trim_min)
    monkeypatch.setattr(config, "TRIM_KEEP_THINKING", keep_thinking)


def _tool(text, name="read_file"):
    return {"role": "tool", "content": text, "name": name}


def test_under_threshold_is_a_no_op(monkeypatch):
    _soft(monkeypatch, trim_at=1_000_000)
    messages = [{"role": "system", "content": "sys"}, _tool("R" * 5000)]
    before = json.dumps(messages)
    assert trim_history(messages) is False
    assert json.dumps(messages) == before


def test_processed_outputs_stubbed_unseen_ones_kept(monkeypatch):
    _soft(monkeypatch, trim_at=1000)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read_file"}}]},
        _tool("A" * 3000),
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "grep"}},
                                                             {"function": {"name": "read_file"}}]},
        _tool("B\n" * 1500, name="grep"),
        _tool("C" * 3000),
    ]
    assert trim_history(messages) is True
    # Replied to already: collapsed to a one-line stub naming the tool.
    assert messages[3]["content"] == f"{config.TRIM_PREFIX}read_file — was 1 lines, 3000 chars]"
    # Not yet seen by the model: stubbing these would have it act on a placeholder.
    assert messages[5]["content"] == "B\n" * 1500
    assert messages[6]["content"] == "C" * 3000


def test_small_outputs_are_not_stubbed(monkeypatch):
    _soft(monkeypatch, trim_at=0, trim_min=100)
    messages = [{"role": "system", "content": "sys"}, _tool("short"),
                {"role": "assistant", "content": "ok"}]
    trim_history(messages)
    assert messages[1]["content"] == "short"


def test_thinking_survives_when_stubbing_is_enough(monkeypatch):
    _soft(monkeypatch, trim_at=2000, keep_thinking=1)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "", "thinking": "T" * 400},
        _tool("A" * 10000),
        {"role": "assistant", "content": "", "thinking": "T" * 400},
    ]
    assert trim_history(messages) is True
    assert messages[2]["content"].startswith(config.TRIM_PREFIX)
    assert messages[1]["thinking"] == "T" * 400
    assert messages[3]["thinking"] == "T" * 400


def test_thinking_fallback_keeps_only_the_most_recent(monkeypatch):
    _soft(monkeypatch, trim_at=500, keep_thinking=2)
    messages = [{"role": "system", "content": "sys"}]
    for n in range(5):
        messages.append({"role": "assistant", "content": f"step {n}", "thinking": "T" * 1000})
    assert trim_history(messages) is True
    has = ["thinking" in m for m in messages[1:]]
    assert has == [False, False, False, True, True]


def test_second_pass_reports_no_edit(monkeypatch):
    # A False must be reliable: it keeps the next call on the short timeout.
    _soft(monkeypatch, trim_at=0)
    messages = [{"role": "system", "content": "sys"}, _tool("A" * 3000),
                {"role": "assistant", "content": "ok"}]
    assert trim_history(messages) is True
    stubbed = messages[1]["content"]
    assert trim_history(messages) is False
    assert messages[1]["content"] == stubbed
