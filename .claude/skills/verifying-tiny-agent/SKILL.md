---
name: verifying-tiny-agent
description: How to run and verify tiny-agent after a change — there is no test suite or CI. Covers the static compile pass, in-process harnesses for logic that doesn't need Ollama, the live smoke test and how to read the prefill stats line to prove the KV cache survived, the interactive-path checklist, and the docs that must stay in sync. Use when testing, running, launching, or verifying this project.
---

# Verifying tiny-agent changes

There is no test suite and no CI. Verification is: compile check → in-process harness for the
changed logic → live smoke test when behavior warrants it → doc sync. Do the first two always;
they need nothing but Python.

## 1. Static pass (always)

```bash
python3 -m py_compile local_agent.py agent.py config.py ollama.py tools.py session.py ui.py
```

Silence means success. Also confirm nothing imported a new third-party package — the
dependency policy is stdlib + `rich` only, Python 3.8+.

## 2. In-process harness (always, for the logic you changed)

Most of the interesting logic is pure functions over a `messages` list or strings, and the
tools run without any server. Exercise the changed function directly with `python3 -c` or a
short throwaway script. Examples to adapt:

```bash
# trim_history on a synthetic conversation (no Ollama needed)
python3 -c "
import config
from agent import trim_history, _total_tokens
msgs = [{'role': 'system', 'content': 'S'}]
msgs += [{'role': 'user', 'content': 'task'}]
msgs += [{'role': 'tool', 'content': 'x' * 30000, 'name': 'grep'} for _ in range(5)]
trim_history(msgs)
assert all(m['content'].startswith(config.TRIM_PREFIX) for m in msgs[2:2+5-config.KEEP_FULL_TOOL_RESULTS])
trim_history(msgs)   # idempotent: second pass must not re-edit
print('ok', _total_tokens(msgs))
"

# tools against the repo itself
python3 -c "from tools import read_file, grep; print(read_file('config.py', start=1, end=5)); print(grep('TRIM_PREFIX', '.'))"

# a specific helper
python3 -c "from tools import _strip_line_number_prefix; print(repr(_strip_line_number_prefix('   12  foo\n   13  bar\n')))"
```

When you changed history handling (`run_turn`, `drop_thinking`, `strip_nudges`), assert the
invariants from `working-on-tiny-agent` on the resulting list: system prompt still at index 0,
every assistant `tool_calls` answered by tool messages, no `thinking` or nudges left after the
turn.

## 3. Live smoke test (when behavior toward Ollama changed)

Needs a running Ollama ≥ 0.20.2 with a tool-capable model pulled (default
`gemma4:12b-it-qat`; override with `--model` or `AGENT_MODEL`; server via `OLLAMA_URL`).

```bash
python3 local_agent.py "what does trim_history in agent.py do?" --yes --max-steps 5
```

`--yes` skips confirmation prompts so the run is unattended. Expect: cyan `→ tool(...)` lines,
a Markdown answer, then a dim stats line.

**Reading the stats line is the cache verification.** It looks like
`3 steps · prefill 2841 tok in 41.2s · gen 190 tok @ 4.1 tok/s`. `prefill` counts only tokens
NOT served from the KV prefix cache. So in an interactive session, ask a second question and
check its stats: turn 2+ prefill should be roughly the size of the new messages only. A
full-conversation-sized prefill on a later turn means your change busted the prefix cache —
go back to the `cache-discipline` checklist. (Exception: the first call after `trim_history`
fires, and the first call after any multi-step turn — the turn-boundary cleanup strips
thinking and stubs the turn's old tool outputs — legitimately re-prefill; those are the
designed once-per-pass/turn busts.)

If the model isn't available, `ollama pull gemma4:12b-it-qat` or use any tool-capable model
you have. Do not silently skip this step when you changed `ollama.py` payloads or streaming —
report that it wasn't run and why.

## 4. Interactive-path checklist (when you touched `main()`, session.py, or ui.py)

Run `python3 local_agent.py` and exercise:

- `/clear` — next stats line should still show a small prefill (system prompt stayed cached).
- `/save foo` then `/resume foo` — round-trips via `~/.tiny_agent_sessions/foo.json`;
  `/sessions` lists it with a `*` on the active one.
- Esc during a streaming reply — prints `cancelled`, returns to the prompt, and the
  partial reply is NOT in history (ask a follow-up to confirm the model never saw it).
- Ctrl-C while a tool is running — the turn aborts but history stays well-formed
  (stub `[interrupted before this tool ran]` results pair up any pending tool_calls).
- A multi-line paste arrives as one prompt, not one prompt per line.
- Ask for a long file edit: while the tool call is composed silently, the live region shows
  `generating… Ns without visible output` after ~2s and Esc still cancels. The next prompt
  must answer promptly — if it hangs for minutes, the abandoned generation wasn't torn down
  (`ollama._abort_stream`).

## 5. Doc sync (always, for user-visible changes)

The README is the spec. Update, as applicable: the flags / env vars / interactive commands /
tools tables, the Design notes section, and the `local_agent.py` module docstring (usage and
env vars). Grep the README for the old behavior's wording to catch drift.
