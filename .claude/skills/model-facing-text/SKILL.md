---
name: model-facing-text
description: Rules for anything the local Ollama model reads or calls in tiny-agent — adding or changing a tool, tool schema, tool result format, error message, truncation notice, system prompt text, or nudge. Use when editing tools.py, config.TOOL_SCHEMAS, config.SYSTEM, or any string a tool returns. The audience is a small (7–12B) model; text must be written for that audience.
---

# Model-facing text and tools

The consumer of every tool result, schema description, and error message is a **small local
model** (default `gemma4:12b-it-qat`), not a human and not a frontier model. Small models
attend poorly to long-result tails, take hints worse than imperatives, and repeat a mistake
harder when told only "that was wrong". Write for that audience.

## Tool contract (tools.py)

- A tool implementation **returns a string and never raises**. Errors are bracketed strings
  that name the problem: `[no such file: X]`, `[grep timed out]`, `[user declined write]`.
  `dispatch` is the last-resort boundary (`TypeError` → bad args, `Exception` → tool error);
  don't rely on it for anticipated failures.
- **Every output is size-capped.** Item-count caps (`MAX_GREP_HITS`, `MAX_READ_LINES`, …) are
  not enough — a single minified line or grep context block can blow past them — so results
  also pass through `_cap_output` (`MAX_TOOL_OUTPUT_CHARS` byte backstop). `_cap_output`
  keeps **head AND tail**, because the useful part (a match, a test failure summary) can land
  at either end. A new tool must apply both kinds of cap. Caps are constants in `config.py`.
- Metadata inside a result is always **bracketed** — `[lines 1-100 of 543 - file continues]`,
  `[+12 more matches]` — and `config.SYSTEM` tells the model bracketed lines are tool
  metadata, not file content. Keep both sides of that convention in sync.
- Destructive tools (`edit_file`, `run_cmd`) gate on `confirm()`, which honors
  `config.AUTO_YES` (`--yes`). Edits call `show_diff` *even under `--yes`* — it's the only
  record of what the agent changed. A new destructive tool must do both.

## Adding or changing a tool touches four places

1. Schema in `config.TOOL_SCHEMAS` (name, description, JSON-schema parameters).
2. Implementation in `tools.py` following the contract above.
3. Registration in the `TOOLS` dict at the bottom of `tools.py`.
4. Row in the README "Tools available to the model" table.

The schemas are part of the KV-cached prefix (see `cache-discipline`): any schema edit
re-prefills nearly the whole prompt for every user after upgrading. That's acceptable but not
free — batch schema wording tweaks rather than churning them one word at a time.

## Small-model ergonomics rules

These come from observed failures (see commit 6c352d8) and are the quality bar:

- **Truncation notices go at BOTH ends of a long result, and the tail one is an imperative
  with exact parameters.** `read_file` ends with
  `[TRUNCATED. To continue reading, call read_file with start=101.]` — not "output may be
  incomplete". Small models follow instructions; they don't infer next steps from hints.
- **Error messages teach the fix for the *specific* mistake.** Canonical example: when
  `edit_file`'s `old_string` fails to match, it runs `_strip_line_number_prefix` to detect
  the model having copied `read_file`'s line-number column (`"   12  "`) into it, and returns
  an error saying exactly that and how to fix it. The generic "must match exactly" message
  made small models retry the same mistake with *more* context lines. When you see a repeated
  model failure, detect it and return a targeted message — don't just restate the rule.
- **Defend against a known failure mode in three layers:** a warning in `config.SYSTEM`, a
  warning in the tool's schema description, and a targeted runtime error message. The
  line-number-prefix mistake has all three; follow that pattern for new failure modes.
- Tool descriptions state what the tool does NOT do when models confuse it
  (`find_files`: "Searches file names/paths, not contents (use grep for contents)").
- Keep `config.SYSTEM` about *working style* only (lazy grep-then-read, one tool call at a
  time, bracketed-metadata convention). Tool specifics belong in the schemas — the format
  the model was trained to read tools in. Never describe tools in prose in the system prompt.
- Uniform interface over implementation detail: `grep` accepts identical flags whether `rg`
  or `grep` backs it; the model must never need to know which is installed.
