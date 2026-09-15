---
name: working-on-tiny-agent
description: Start-here map for any code change in this repo (tiny-agent). Read before editing any .py file here — covers the master design constraint, module responsibilities, message-history invariants, and code conventions. Applies to agent.py, config.py, ollama.py, tools.py, session.py, ui.py, local_agent.py.
---

# Working on tiny-agent

tiny-agent is a minimal local coding agent (~1,600 lines, stdlib + `rich` only) that drives a
CPU-bound Ollama model. One constraint shapes everything: **no GPU, so prefill is expensive.
Every design decision minimizes prefilled tokens and reuses them across turns via Ollama's
KV prefix cache.** Before changing anything that touches the request payload, message history,
system prompt, tool schemas, or trimming, read the `cache-discipline` skill and vet your change
against it. When touching tools or any text the model reads, read `model-facing-text`. To test,
read `verifying-tiny-agent`.

## Module map

- `local_agent.py` — entry shim. Its module docstring states the four core design decisions
  (static system prompt, native tool calling, dynamic context in user messages, lazy context)
  and documents usage/env vars. **Keep it in sync when behavior changes.**
- `config.py` — every tunable, in one place. Contains `SYSTEM` (the system prompt, marked
  KEEP BYTE-IDENTICAL — it is the cached prefix) and `TOOL_SCHEMAS` (also part of the cached
  prefix), plus all caps/thresholds. New knobs go here, env-var overridable where users might
  need them (`AGENT_*` naming).
- `agent.py` — the loop. `run_turn` (one user task: repeated model calls + tool round-trips),
  `trim_history` (lazy mid-turn context shedding: stubs every already-processed tool output,
  sheds old thinking only as a fallback when that isn't enough, then the hard-truncate
  backstop; returns True when it edited history, which gives the next call a
  prefill-sized stall timeout and one retry), `_stub_tool_outputs` (shared stub helper),
  `drop_thinking` and `strip_nudges` (turn-boundary cleanup, invoked from `run_turn`'s
  `finally`, which also stubs the finished turn's oversized tool outputs), `main()` (CLI +
  REPL with `/clear`, `/save`, `/resume`, `/sessions`).
- `ollama.py` — HTTP layer. `_build_payload` is the **single source of truth for request
  bodies**; both call sites (`call_ollama`, `warm_cache`) go through it so
  model/keep_alive/num_ctx can't drift. `call_ollama` streams and returns
  `(content, thinking, tool_calls, cancelled, stats)`.
- `tools.py` — tool implementations + `dispatch`. Tools return strings and never raise.
- `session.py` — save/resume persistence (`~/.tiny_agent_sessions/`). `apply_session` mutates
  the `messages` list in place (`messages[:] = ...`) so the autosave closure keeps seeing it —
  never rebind that list.
- `ui.py` — terminal helpers: cbreak/cancel-key handling, bracketed paste, live stream
  rendering, stats formatting.

## Message-history invariants

The `messages` list is the agent's entire state. Corrupting it breaks the *next* request in
non-obvious ways. Any change must preserve:

1. `messages[0]` is always the static system prompt. `/clear` deletes `messages[1:]` only.
2. An assistant message carrying `tool_calls` must be followed by exactly one `role: "tool"`
   message per call. `run_turn`'s inner `finally` appends `[interrupted before this tool ran]`
   stubs for calls that never executed (e.g. Ctrl-C mid-tool) to keep this true.
3. `thinking` fields exist on assistant messages only *while a turn is live*. Every exit from
   `run_turn` — return, cancel, or exception — passes through its `finally`, which runs
   `drop_thinking` + `strip_nudges`. Don't add an exit path that bypasses it.
4. Nudges (`STEP_LIMIT_NUDGE`, `EMPTY_RETRY_NUDGE`) and the empty assistant replies that
   prompt them are transient: stripped at turn end by `strip_nudges`, and filtered again on
   session restore via `_is_stray_nudge` (a crash can persist one into an autosaved session).
5. Trimming is idempotent via the `TRIM_PREFIX` sentinel: every compacted/truncated message
   starts with it and is skipped on later passes. Any new compaction mechanism must use it too.
6. The forced final step (`last`) drops unanswered `tool_calls` from the stored assistant
   message — storing them without results would corrupt history for the next turn.

## Code conventions

- **Comments explain why, never what.** Every non-obvious decision gets a comment or docstring
  stating the rationale and the failure mode it prevents (read `trim_history`'s docstring for
  the house style). If you make a judgment call, write down why. Match this density — it is
  the project's substitute for tests and design docs.
- Dependencies: stdlib + `rich`. HTTP via `urllib`, not `requests`. Python 3.8+ compatible —
  no `match`, no `X | Y` type syntax. No type annotations are used; don't introduce them.
- Narrow exception tuples (`except (OSError, ValueError):`), never bare `except` — the sole
  broad catch is `dispatch`'s tool-error boundary.
- Visual style: aligned assignment blocks (`content    = ""`), section-divider banners
  (`# ---- #`), Rich markup for all console output (`[dim]`, `[yellow]`, `[red]`).
- No test suite, no CI. **The README is the spec**: any user-visible change must update its
  tables (flags, env vars, interactive commands, tools) and Design notes section, plus the
  `local_agent.py` docstring. Git history shows doc drift gets flagged and fixed in review;
  don't create it.
