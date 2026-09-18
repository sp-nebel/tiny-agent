---
name: verifying-tiny-agent
description: How to run and verify tiny-agent after a change — there is a pytest suite under tests/ but no CI. Covers the static compile pass, the test suite, in-process harnesses for logic that doesn't need Ollama, the live smoke test and how to read the prefill stats line to prove the KV cache survived, the interactive-path checklist, and the docs that must stay in sync. Use when testing, running, launching, or verifying this project.
---

# Verifying tiny-agent changes

There is a pytest suite under `tests/` but no CI, so nothing runs it for you. Verification is:
compile check → test suite → in-process harness for the changed logic → live smoke test when
behavior warrants it → doc sync. Do the first three always; they need nothing but Python.

## 1. Static pass (always)

```bash
python3 -m py_compile local_agent.py agent.py config.py ollama.py tools.py session.py checkpoint.py ui.py commands.py
```

Silence means success. Also confirm nothing imported a new third-party package — the
dependency policy is stdlib + `rich` only, Python 3.8+.

## 2. Test suite (always)

```bash
python3 -m venv venv && venv/bin/pip install rich pytest   # once; venv/ is gitignored
venv/bin/python -m pytest -q
```

No Ollama needed: `tests/test_run_turn.py` drives `run_turn` with a scripted fake in place of
`agent.call_ollama` (plus `agent.dispatch` and `agent.read_prompt`), and the tool tests run
against `tmp_path`. Every test must pass before you commit. pytest is a dev-only dependency;
it stays out of `requirements.txt`, which lists what the agent itself needs.

When you change behavior, change or add the test that pins it in the same commit. A failing
test that asserts the *old* behavior means the test is stale (fix the test), not that the
change is wrong. A new feature gets tests too:

- History handling (`run_turn`, nudges, interjections, trimming): script the fake model in
  `test_run_turn.py`. Nudges are stripped in `run_turn`'s `finally`, so assert anything
  transient on the per-call snapshots `FakeModel.calls` records, and the end state with
  `assert_well_formed`.
- Tools: one file per tool (`test_<tool>.py`). Set `config.AUTO_YES`, or monkeypatch
  `tools.confirm` to test a decline.
- Tunables: monkeypatch `config.*` instead of relying on defaults, so retuning a threshold
  doesn't break unrelated tests.

## 3. In-process harness (always, for the logic you changed)

Most of the interesting logic is pure functions over a `messages` list or strings, and the
tools run without any server. The suite covers the common paths; for a quick look at a changed
function, or an edge case not worth a permanent test, call it directly with `python3 -c` or a
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

## 4. Live smoke test (when behavior toward Ollama changed)

Needs a running Ollama ≥ 0.20.2 with a tool-capable model pulled (default
`gemma4:12b-it-qat`; override with `--model` or `AGENT_MODEL`; server via `OLLAMA_URL`).

```bash
python3 local_agent.py "what does trim_history in agent.py do?" --yes --max-steps 5
```

`--yes` skips confirmation prompts so the run is unattended. To see the loop check fire, run
with `--max-steps 0 --check-every 3`: a dim `loop check after 3 steps…` line appears, and a
CONTINUE reply leaves no trace in history (under the default `--max-steps 20` it never fires). Expect: cyan `→ tool(...)` lines,
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

## 5. Interactive-path checklist (when you touched `main()`, session.py, or ui.py)

Run `python3 local_agent.py` and exercise:

- `/clear` — next stats line should still show a small prefill (system prompt stayed cached).
- `/save foo` then `/resume foo` — round-trips via `~/.tiny_agent_sessions/foo.json`;
  `/sessions` lists it with a `*` on the active one.
- Esc during a streaming reply — prints `cancelled`, returns to the prompt, and the
  partial reply is NOT in history (ask a follow-up to confirm the model never saw it).
- Typing during a streaming reply opens `interject`; the message is echoed as `you …` after
  the current step's tool results. Tab during a reply stops it at once and opens `steer`; the
  pending tool call must NOT run, and the model's next step follows the note. Tab inside the
  `interject` prompt is readline completion, not a stop.
- The `[y/N/a/reason]` hint is visible on a confirmation prompt (`[y/N/reason]` for a
  chained command, which can't be always-allowed); `a` stops later prompts for the same
  command prefix or for all edits, and bracketed tool results
  (`[lines 1-100 of 543 …]`) appear under their `→ tool …` line with `/details` on, or
  when the call failed (Rich would swallow them unescaped).
- `/details` and `/thinking` toggle the display only; a turn that runs 20s+ ends with a
  bell (and a desktop notification on terminals that support OSC 9/777).
- Ctrl-C while a tool is running — the turn aborts but history stays well-formed
  (stub `[interrupted before this tool ran]` results pair up any pending tool_calls).
  Ctrl-C during `run_cmd` answers that call with its partial output plus
  `[the user stopped this command …]`, and the command's process group is gone
  (`ps` shows no leftover `sleep`).
- `!git status` then a question about it — the model answers from the output; `!!ls` shows
  output but the model never sees it.
- `@README.md summarize` prints `attached README.md` and the answer draws on the file;
  `me@example.com` in a prompt is left alone.
- Typing `foo \`, Enter, `bar` shows a `...` continuation prompt and sends one message.
- `/undo` after a turn that edited a file in a git repo — the file is back, a file the turn
  created is gone, the prompt is pre-filled in the input line, and `git status` / `git diff
  --cached` look exactly as before the turn. Outside a repo it warns and rewinds only the
  conversation.
- A multi-line paste arrives as one prompt, not one prompt per line.
- `/help` lists built-ins and any `.tiny-agent/commands/*.md`; `/NAME args` sends the filled-in
  template; a mistyped `/word` comes back in the input line instead of going to the model.
- Tab at the prompt completes `@pa…` to a path and `/he…` to `/help`.
- `/editor` opens `$EDITOR`; the saved text is sent as a prompt even if it starts with `!`.
- `/history` then `/undo 2` rewinds two turns (files too, in a git repo).
- `/export` writes a readable Markdown transcript; `/sessions` shows first-prompt titles.
- `echo 'what is 2+2' | python3 local_agent.py > out.md` runs one turn: `out.md` holds only the
  answer, everything else went to stderr, and an edit attempt without `--yes` is refused.
- Ask for a long file edit: while the tool call is composed silently, the live region shows
  `generating… Ns without visible output` after ~2s and Esc still cancels. The next prompt
  must answer promptly — if it hangs for minutes, the abandoned generation wasn't torn down
  (`ollama._abort_stream`).

## 6. Doc sync (always, for user-visible changes)

The README is the spec. Update, as applicable: the flags / env vars / interactive commands /
tools tables, the Design notes section, and the `local_agent.py` module docstring (usage and
env vars). Grep the README for the old behavior's wording to catch drift.
