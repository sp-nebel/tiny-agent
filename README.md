>[!WARNING]
>Only vibecoding ahead

# tiny-agent

A minimal local coding agent for CPU-bound machines, built around [Ollama](https://ollama.com).

The whole design is shaped by one constraint: **no GPU**. Prefill is expensive on CPU, so every architectural decision is about keeping the prefilled token count small and reusing it across turns via Ollama's KV-cache prefix caching.

## Requirements

- Python 3.8+
- [Ollama](https://ollama.com) ≥ 0.20.2
- `pip install rich`
- A model with tool-call support pulled in Ollama (default: `gemma4:12b-it-qat`)

## Setup

```bash
pip install rich
ollama pull gemma4:12b-it-qat   # or any tool-capable model
```

## Usage

```bash
# One-shot task
python local_agent.py "review the null handling in AuthService"

# Interactive mode
python local_agent.py

# Pipe input in: one turn, then exit
git diff | python local_agent.py "review this diff" > review.md
```

When stdin is not a terminal, the piped text goes to the model ahead of your prompt, marked `[input piped to the agent by the user]` (with no prompt, the piped text *is* the prompt). The agent runs one turn and exits, with code 1 if the turn failed. If stdout is not a terminal either, only the final answer is written there, as plain Markdown, and everything else goes to stderr. A piped run is not autosaved as a session. Nobody can answer a confirmation in this mode, so edits and commands are refused unless you pass `--yes`, and the model is told why. Piped input over 16000 chars is cut in the middle, like an oversized tool result, and saved in full to a file the model can read.

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model TAG` | `gemma4:12b-it-qat` | Ollama model tag |
| `--yes` | off | Auto-approve all writes and shell commands |
| `--max-steps N` | 20 | Max tool round-trips before giving up on a task — one round can include several tool calls if the model requests them together, so this isn't a raw tool-call count (`0` = unlimited) |
| `--check-every N` | 20 | Every N tool round-trips, ask the model whether its recent steps are repeating without new information; `STUCK` ends the turn on its explanation and hands control back, anything else lets it continue (`0` = never). Inert under the default `--max-steps 20` — the check could only fire as the cap ends the turn anyway — so it is meant for `--max-steps 0` or a higher cap |
| `--resume NAME` | off | Resume the saved session `NAME` |
| `-c`, `--continue` | off | Resume the most recent saved session. Takes no value, so `-c "fix the test"` resumes and then runs the prompt |

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_MODEL` | `gemma4:12b-it-qat` | Ollama model tag (overridden by `--model`) |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API base URL |
| `AGENT_THINK` | `1` | Set to `0` to disable reasoning output |
| `AGENT_NUM_CTX` | `24576` | Ollama context window size; lower on memory-constrained machines |
| `AGENT_SYNTAX_CHECKS` | none | JSON object mapping file extensions to a syntax-check command run after an edit, with `{path}` as the placeholder, e.g. `{".js": "node --check {path}", ".sh": "bash -n {path}"}`. `.py` and `.json` are checked without it |
| `AGENT_NOTIFY` | `1` | Set to `0` to turn off the bell and desktop notification (OSC 777 on VTE terminals, OSC 9 elsewhere) sent when a turn that ran 20s or more finishes, or when a confirmation prompt is waiting in such a turn |
| `AGENT_STREAM_TIMEOUT` | `300` | Per-read socket timeout (s) on streaming calls; fires only when nothing arrives at all. The call right after a trim pass instead gets a timeout sized to its full re-prefill (estimated tokens ÷ measured prefill rate) |

### Interactive commands

| Input | Effect |
|-------|--------|
| `exit` / `quit` | Quit the agent |
| `/help` | List the built-in commands and your custom commands |
| `/clear` or `clear` | Reset conversation history (system prompt and tool schemas stay cached) |
| `/save [NAME]` | Save the current conversation as a session (defaults to a timestamp) |
| `/resume [NAME]` | Resume a saved session; bare `/resume` resumes the most recent |
| `/sessions` | List saved sessions, each with its first prompt as a title |
| `/history` | List this conversation's turns, numbered oldest first, as far back as `/undo N` can reach (the list starts over after `/clear` and `/resume`) |
| `/export [PATH]` | Write the conversation to a Markdown file: prompts, answers, and each tool call with its result (default `tiny-agent-<session>.md` in the working directory) |
| `/editor` | Write the next prompt in `$VISUAL` or `$EDITOR` (default `vi`). What you save is sent as a prompt, even if it starts with `!` or `/` |
| `/NAME [ARGS]` | Run a custom command (see below). An unknown `/word` is not sent to the model; it goes back into the input line |
| `/details` | Toggle full tool results under each call. Off by default: each call is one line (`→ read agent.py:10-40 (10-40 of 543)`, `→ grep "foo" (7 matches)`, `→ run_cmd pytest` then `exit 0`), and a result's body is shown only when the call failed |
| `/thinking` | Toggle the live view of the model's reasoning while it streams. Display only: the model still thinks, and the request is unchanged |
| `!cmd` | Run a shell command yourself; its output (and exit code) is shown and goes to the model ahead of your next message. It runs in a subshell, so `!cd` has no lasting effect |
| `!!cmd` | Same, but the output is only shown to you, never sent to the model |
| Line ending in `\` | Continue the prompt on the next line (a dim `...` prompt); the lines are sent as one message. A multi-line paste needs no backslashes — it already arrives as one message |
| `@path` or `@path#A-B` in a prompt | Attach that file, or only its lines A to B (`@agent.py#10-40`, `@agent.py#12`): its contents go to the model with the message, exactly as the model's own first `read_file` of it would return them (numbered lines, first 100, with the usual instruction for reading on). Only tokens that name an existing file count, so `me@example.com` and `@alice` stay plain text; trailing sentence punctuation is ignored. Paths with spaces aren't supported |
| `/undo [N]` | Take back the last turn (or the last N): its messages are cut from the conversation, the files it changed are restored from a git snapshot taken when it started, and its prompt is put back in the input line to edit and resend. Repeat to walk further back, or give a count: `/undo 3` takes back the last three turns and puts the oldest one's prompt back. Outside a git repo only the conversation is rewound. `!cmd` output that rode along with the undone prompt is not queued again, and a multi-line prompt is printed rather than pre-filled |
| Any other key during a reply | Pauses the stream and opens an `interject` prompt for a message to the model; the key you typed becomes the first character of the line. Delivered at the next step boundary (after the current tool round-trip finishes), or as the next prompt if the turn ends first. Submit an empty line to think better of it |
| Tab during a reply | Stop the reply now and steer it: the text and reasoning so far stay in context, any tool call it had started is dropped unrun, and a `steer` prompt asks for a note the model must follow before continuing (empty = "stop and reconsider"). Costs no step. Inside the `interject` prompt Tab is ordinary line-editing completion, not a stop |
| Esc during a reply | Cancel the in-flight response |
| Up / Down arrows | Recall previous prompts (history persists in `~/.tiny_agent_history`) |
| Tab at the prompt | Complete an `@path` or a `/command` |

### Custom commands

A Markdown file in `.tiny-agent/commands/` (at the git root, or the working directory outside a repo) or `~/.config/tiny-agent/commands/` becomes a command named after the file: `review.md` is `/review`. When both define the same name, the project's file wins, and neither can replace a built-in command. Typing the command sends the file's text as your prompt, filled in first:

- `$ARGUMENTS` becomes everything after the command name, and `$1` … `$9` the individual words. If the file uses neither, what you typed after the name is added at the end.
- `` !`cmd` `` is replaced by that shell command's output. It runs without asking, like `!cmd` at the prompt, and each one is printed as it runs.
- `@path` attaches a file, as in a typed prompt.

An optional front-matter block sets the description `/help` shows:

```markdown
---
description: review a file for bugs
---
Review @$1 for bugs, especially $2. Current branch: !`git branch --show-current`
```

Esc is the only key that cancels, and only on its own: arrow keys and other escape sequences are drained rather than read as a cancel. Once a reply has started streaming, keys are polled every quarter second whether or not anything arrives. Before that — during the silent re-prefill after a trim pass, when Ollama hasn't sent even the response headers yet — nothing is seen until generation starts. A multi-line paste into a stream arrives as one interjection per line.

While the model composes a tool call nothing streams — Ollama sends the call as one piece when it is complete — so after a couple of silent seconds the live region shows an elapsed-time notice instead of freezing, and the keys keep working throughout. Cancelling in that phase shuts the connection down outright, so Ollama stops generating the abandoned call instead of making your next request wait behind it.

## Tools available to the model

| Tool | Description |
|------|-------------|
| `read_file` | Read a file with optional line range (capped at 100 lines per call). A read that reaches the end says `[end of file, N lines]`. A directory returns its listing, a missing path suggests the closest existing file names, and a non-UTF-8 text file (latin-1 source) is shown instead of refused; only a NUL byte in the first 4 KB marks a file as binary |
| `grep` | Search files by regex (extended syntax), with optional context lines and an `include` glob (`*.py`). Hits are grouped under their file, lines are cut at 300 chars, and the overflow notice says to narrow the search. rg or grep is auto-detected; the output is the same either way |
| `find_files` | Glob-pattern file search |
| `list_dir` | List directory contents |
| `cd` | Change the working directory |
| `edit_file` | Exact-string replacement edit, or create a new file (an empty existing file counts as new); CRLF files keep their line endings, and bytes outside the replacement are never rewritten. When the exact text isn't found, whole lines are matched again ignoring trailing whitespace, then indentation (`new_string` is shifted by the same amount), then curly quotes and dashes. The match must be unique, and the result says which fallback matched. `old_string == new_string` is refused, and a missing path gets "did you mean" suggestions |
| `append_file` | Append text to the end of a file (creating it if missing); inserts a separating newline if needed and keeps the file's line endings |

After `edit_file` or `append_file` writes a `.py` or `.json` file, it is parsed, and if the write *introduced* a syntax error the tool result says where, in one line. More extensions can be checked with shell commands via `AGENT_SYNTAX_CHECKS`.
| `run_cmd` | Run a shell command; a non-zero exit code is appended to the output as `[exit N]`. Optional `timeout` (default 120s, max 600s). On a timeout the whole process group is killed, background children included, and the output so far is kept with a note on what to do next. stdin is `/dev/null`, so a command that waits for input ends at once |

Every tool's result is size-capped. When a result is cut in the middle (a long test run, a big grep), the full text is saved under the system temp dir (`tiny-agent/output-*.txt`), and the cut notice gives the path so the model can grep or read it instead of running the command again. These files are not deleted by the agent; the OS clears the temp dir as usual.

A model that calls a tool by another agent's name (`bash`, `cat`, `write_file`, `functions.read_file`) or with its argument names (`command`, `file_path`, `content`) is mapped onto the real tool. An unknown tool name gets the list of real ones, and a call with missing or unknown parameters is told which, along with the tool's parameter list. When the same call returns the same result three times in a row within one task, a note telling the model to stop repeating it is added to the result.

`edit_file`, `append_file` and `run_cmd` ask for confirmation before executing unless `--yes` is passed. Edits show a colored unified diff before the confirmation prompt (and under `--yes`, as a record of what changed). The prompt is `[y/N/a/reason]`: `y` approves, empty/`n` declines, `a` approves and stops asking for the rest of the session, and anything else declines *and* is passed back to the model as the reason — `use the test runner, not python directly` redirects it, where a bare refusal tends to make a small model re-issue the identical call. For an edit, `a` allows all file edits. For a command, it allows every command with the same prefix, shown in the prompt: `git checkout main` allows `git checkout …`, `npm run dev` allows `npm run dev …`, and an unlisted command allows its first word (`pytest …`). A command containing `;`, `&`, `|`, `<`, `>`, `$`, a backtick, parentheses, braces or a newline, or starting with a `VAR=value` assignment, is never approved by prefix and always asks, with no `a` option, since a prefix says nothing about what else it runs. Approvals last until the process exits and are never saved.

**Security note:** `--yes` auto-approves every write and shell command with no confirmation, and `run_cmd` executes with `shell=True`, so the model can run anything a real shell command can. Only use `--yes` in a repo/directory you trust the agent with.

After each model turn a dim stats line is printed, e.g. `3 steps · prefill 142 tok in 3.2s · gen 56 tok @ 8.4 tok/s · 1m12s total · ctx ~41%`. The prefill count covers only tokens *not* served from the KV prefix cache, so a small number on a long conversation means the prefix caching is working. `total` is the turn's wall time, tools and confirmation waits included, and `ctx` is the estimated share of the context window the conversation now fills (trimming starts at ~70%). While a reply streams, the live region also shows how many typed messages are queued for the model.

## Design notes

**Static system prompt** — the system prompt and tool schemas are byte-identical on every call so Ollama's KV-cache prefix caching fires: the expensive prefill of the *static prefix* happens once per session, not once per turn. Conversation history still grows every turn and re-prefills incrementally as it grows — see "reasoning feedback" below for the one place that's paid more than once.

**Dynamic context in user messages** — the working directory, whether it is a git repo, the platform and the date go in the first user message, keeping the system prompt unchanged across projects. So do instructions files: `~/.config/tiny-agent/AGENTS.md` (honours `XDG_CONFIG_HOME`) for every project, then the first `AGENTS.md` or `CLAUDE.md` found walking up from the working directory to the git root (outside a repo, only the working directory is checked). Each is cut at 3000 chars with a pointer to read the rest. They are read when a conversation starts and again after `/clear`; a running conversation keeps what it was sent, because rewriting its first message would bust the whole cache.

**Lazy context** — the model is not front-loaded with files. It greps to locate code and then reads a tight line range. This keeps per-turn prefill small.

**History trimming** — editing history busts the KV cache from the edit point on, and on a sliding-window-attention model (gemma) the next call then re-prefills the whole prompt from token 0 — so the cost of any trim is the *post-trim prompt size*, and edits are only ever made at two already-paid moments. First, at every turn boundary: the finished turn's tool outputs (all but the most recent few) collapse to one-line stubs, piggybacked on the same cache bust the thinking strip below already causes, so completed turns stay lean and multi-turn sessions reach each new task with a nearly-empty window. Second, lazily mid-turn: when a single long turn pushes the estimate past ~70% of the context window, one pass collapses every tool output the model has already replied to — its kept thinking is the distilled record of them, and raw file reads are the biggest single contributor — while the trailing results it hasn't seen yet stay verbatim. Thinking itself is preferentially kept: only if stubbing alone can't get back under the trigger does the pass also shed all but the last few steps' thinking, since otherwise every later step would bust the cache for its one new output. If that still leaves the estimate over ~90%, a backstop hard-truncates other oversized messages outside the most recent few, so the prompt can't silently grow past `NUM_CTX` and push the system prompt itself out of Ollama's context. The model call right after a mid-turn trim re-prefills the whole trimmed prompt in silence (no tokens stream during prefill), so that one call runs with a stall timeout sized to the actual re-prefill — estimated tokens over the prefill rate measured from live stats — and gets one retry if it still times out, since the server keeps its prefill progress and the retry resumes nearly free.

**Mid-turn updates stay on screen** — while a reply streams, the reasoning and the text-so-far share a transient live region that is wiped when the stream ends. The reasoning is meant to go: it is scratch work, and it is dropped from history too. Text the model writes *for the user* is not, so whenever a step produces prose alongside its tool calls — "let me check the config first" — that prose is re-rendered as Markdown above the `→ tool(...)` lines it explains, exactly as a final answer is. A whole turn's narration therefore reads back in order once the turn ends, instead of only its last message surviving. This is display only: the text was always carried on the assistant message and fed back to the model either way, so nothing about the request payload or the prefix cache changes.

**Reasoning feedback** — a model's `thinking` output is carried on its assistant message and fed back on later tool round-trips *within* a turn, so it doesn't have to re-derive its chain of thought after every tool result. It's stripped once the turn produces a final answer (and, under mid-turn context pressure, trimming may shed all but the last few steps' thinking early — though only as a fallback when stubbing tool outputs alone isn't enough, see above). On a thinking model, that strip busts the KV cache back to the start of the turn, so each multi-step turn re-prefills its own tool round-trips on the next turn — the "once per session" prefill claim above applies to the static system prompt and schemas, not to every token exchanged.

**Surviving an Ollama restart mid-turn** — on a memory-constrained machine, a long session can OOM-kill the Ollama runner (host-memory prompt-cache state scales with resident context tokens). systemd typically restarts Ollama within seconds. A request that fails before anything streams (connection refused or reset, or HTTP 429/500/502/503/504) is retried with the identical payload up to 5 times, backing off 2s, 4s, 8s … (capped at 30s, with jitter) and printing `retry N/5 in Ns`. A context-overflow error or any 4xx is never retried, since resending the same prompt can only fail again. Lower `AGENT_NUM_CTX` if OOM kills recur.

**Interjections** — there are two ways to talk to the model mid-turn, and the difference is *when* it hears you. Tab means **now**: the stream stops, the pending tool call is dropped, and your note is the next thing the model reads. Typing means **after this step**: the reply keeps going, its tool still runs, and your message waits for the step boundary. Both are tail appends, so the prefix cache is untouched either way, and both are permanent history rather than nudges — later turns still see why the model changed course.

Tab commits the stopped reply as an assistant message (so the model keeps the thought it was in the middle of) minus any tool calls it had emitted, then appends your note marked `[the user interrupted you with this note; …]`. A reply stopped before its first token commits nothing, leaving two user messages in a row, which chat templates render in sequence. The partial's reasoning is dropped at turn end like any other, and if nothing but reasoning was there, the empty shell goes with it.

For typing, any keypress other than Esc or Tab opens a prompt for a message to the model. What you type is queued, not injected: the model is already generating, and an assistant message carrying `tool_calls` must be followed immediately by its tool results, so slipping a user message in between would corrupt history. The queue is drained at the next step boundary — after the current round-trip's tool results, before the next model call — and appended there as an ordinary user message marked `[user message sent mid-task]`, the same tail-append shape as the step-limit and empty-reply nudges, so only its own tokens prefill and the cached prefix is untouched. If the turn ends before the next boundary (final answer, cancel, step limit, or a connection error), whatever is still queued becomes the next prompt instead of being swallowed. Because `cbreak` echo is off, the key that opened the prompt would otherwise be lost, so it is pre-filled into the line — typing straight into a stream keeps every character, which is what made `q` safe to drop as a second cancel key.

**Loop check** — a hard step cap stops a model going in circles, but it also stops long useful work. `--max-steps 0` removes the cap, and `--check-every` puts a soft budget in its place: every N round-trips the model is asked, in a question appended to the tail of a *copy* of the history, whether its recent steps repeat without producing new information. On `CONTINUE` nothing is committed and the next real request simply diverges from the probe at the tail, so the prefix cache is intact. On `STUCK` the exchange is committed, the model's one-sentence account becomes the turn's final message, and you get control back with the whole turn still in context to steer from. A tool call or any other reply counts as `CONTINUE`, so a model that ignores the question cannot be stopped by it — the check catches tight loops (re-reading the same range five times); slow drift is what Tab and typing are for. The probe runs after any trim pass (so it never goes out over the threshold, and pays the re-prefill in place of the next real call) and never on the forced last step of a capped turn.

**Prompt expansions** — `!cmd` output is not sent on its own: it is held until your next message and prepended to it, in the same user message, marked `[the user ran this shell command: … — exit N]` so the model doesn't mistake it for a result of its own. That keeps the history free of back-to-back user messages and costs nothing until you ask about it. The output goes through the same size cap as `run_cmd`. `@path` attachments follow the same route: blocks marked `[contents of PATH, attached by the user]` after your text in the same message. They reuse `read_file`, so a large file costs one read's worth of prefill, not its full size, and the model already knows how to continue reading it. Like everything else you type, both are appended at the tail, so the cached prefix is untouched.

**Undo** — before every turn the working tree of the enclosing git repo is snapshotted as a git tree object, written through a *temporary* index (`GIT_INDEX_FILE`) seeded from the real one so only changed files are hashed. `/undo` diffs that tree against the current one and rewrites just the differing paths — edited and deleted files come back, files the turn created are removed — again through a temporary index, so the user's real index, staged changes, HEAD and stash are never touched. Because it is git and not a journal kept by the edit tools, changes made through `run_cmd` (a formatter, codegen, an `rm`) are undone too. What it can't undo: gitignored files, anything outside the repo, commits made during the turn (HEAD is left where it is, with a warning), and side effects that aren't file contents. The conversation side cuts the turn off the tail of history, so what remains is exactly a prefix of the last request: a full-attention model keeps it cached. On an SWA model (gemma) the rollback is expected to cost at most one re-prefill of what remains — the same kind of bust as a trim — and possibly only a checkpoint restore, like the loop-check probe's divergence; not yet measured. Either way it is paid once, at a moment you chose, and runs as a background warmup while you edit the restored prompt. The undo stack lives for the process only; `/clear` and `/resume` empty it.

**Native tool calling** — tools are passed via Ollama's `tools` parameter as JSON schemas, not described in the system prompt, so the model uses the format it was actually trained on.

## Changelog

Newest first. Every commit adds its entry here (see `CLAUDE.md`).

### 2026-09-23

- **Cleanup after #5**: Tab completion no longer re-reads every custom command file and re-globs the directory once per candidate. It builds the list once per Tab press and reads commands only when completing a `/word`. The "did you mean" path search now stops at its file cap even inside one huge directory. Fuzzy `edit_file` matching is cheaper on long files. The rest is internal tidying with no visible change (one retry path in `ollama.py`, one tool-name normalizer, one turn clock).

### 2026-09-18

- **Stdin piping** (`5af5b9c`, `37662ba`, merged in #5): `git diff | local_agent.py "review this"` sends the piped text with the prompt, runs one turn and exits (code 1 if it failed); piped runs aren't autosaved as sessions. With stdout piped as well, only the final answer goes to stdout, as plain Markdown. Without a terminal, edits and commands are refused unless `--yes` is given.
- **Custom commands, `/help`, `/history`, `/undo N`, `/export`, `/editor`, line ranges, Tab completion, session titles** (`fc7a609`, merged in #5): Markdown files in `.tiny-agent/commands/` or `~/.config/tiny-agent/commands/` become `/name` commands, with `$ARGUMENTS`, `$1`…`$9`, `` !`cmd` `` and `@file` filled in; `/help` lists them with the built-ins. `/history` numbers the conversation's turns, and `/undo N` takes back several at once. `/export` writes the conversation to Markdown, and `/editor` composes a prompt in `$EDITOR`. `@path#10-40` attaches just those lines, and Tab completes `@paths` and `/commands`. `/sessions` shows each session's first prompt as its title. A mistyped `/command` is no longer sent to the model as a prompt.
- **`a` = always allow at the confirmation prompt** (`60c4efc`, merged in #5): the prompt is now `[y/N/a/reason]`. `a` on an edit allows all edits for the rest of the session. On a command, it allows commands with the same prefix (`git checkout …`, `npm run dev …`, `pytest …`). Commands that chain, pipe, redirect or substitute always ask.
- **Quieter tool display, `/details`, `/thinking`, richer stats, notifications** (`19648c6`, merged in #5): each tool call now prints one line with its outcome (`→ grep "foo" (7 matches)`), and a result's body appears only when the call failed or `/details` is on. `/thinking` hides the live reasoning view without changing the request. The stats line adds the turn's total time and the share of the context window in use, and the live region counts queued messages. A turn that ran 20s or more rings the bell and sends a desktop notification when it ends or waits for a confirmation (`AGENT_NOTIFY=0` turns this off).
- **Retry with backoff** (`67adeff`, merged in #5): a request to Ollama that fails before anything streams (connection refused or reset, HTTP 429/500/502/503/504) is retried up to 5 times with exponential backoff (2s doubling to 30s, with jitter), instead of a single retry on a refused connection. A context overflow and any 4xx are never retried.
- **`AGENTS.md` / `CLAUDE.md` instructions** (`f88b792`, merged in #5): the first message of a conversation now carries `~/.config/tiny-agent/AGENTS.md` and the nearest project `AGENTS.md` or `CLAUDE.md` (walking up to the git root), each cut at 3000 chars. The environment line also says whether the directory is a git repo, the platform, and the date. The system prompt is unchanged, so the cached prefix is unaffected.
- **Saved full output, tool-name repair, repeat detection** (`fa098a9`, merged in #5): a tool result cut in the middle is also saved to a temp file, and the notice tells the model to grep or read that file instead of re-running a slow command. Tool names and argument names from other toolsets (`bash`, `cat`, `write_file`, `command`, `file_path`) are mapped onto the real tools, an unknown tool name gets the list of valid ones, and bad arguments are named along with the tool's parameters. The third identical call in a row that returns an identical result gets a note telling the model to stop repeating it.
- **Forgiving `edit_file`, syntax check after writes, friendlier `read_file`** (`0f67103`, merged in #5): when `old_string` isn't found exactly, `edit_file` tries whole-line matches ignoring trailing whitespace, then indentation (re-indenting `new_string` to fit), then curly quotes and dashes. The match must be unique, and the result names the fallback used. After an edit or append to a `.py` or `.json` file, a syntax error the write introduced is reported in one line (more languages via `AGENT_SYNTAX_CHECKS`). `read_file` ends a complete read with `[end of file, N lines]`, lists a directory instead of failing, shows latin-1 files instead of calling them binary, and a missing path in `read_file` or `edit_file` comes with "did you mean" suggestions. An edit whose two strings are identical is refused.
- **`run_cmd` no longer hangs, Ctrl-C tells the truth, `-c/--continue`, grep grouped by file** (`ee580a4`, merged in #5): a command that starts a background process (`cmd &`, a dev server) used to block `run_cmd` long past its timeout, and the output gathered before a timeout was thrown away. The command now runs in its own process group, which is killed as a whole on a timeout or Ctrl-C, and its partial output is kept. A timed-out command's result explains how to pass the new `timeout` parameter (max 600s). A command stopped with Ctrl-C is reported to the model as partly run instead of "interrupted before this tool ran", and Ctrl-C during a `!cmd` no longer crashes the agent. `--resume` now needs a name; `-c/--continue` resumes the most recent session, so `--resume "fix the test"` can't swallow the prompt anymore. `grep` gained an `include` glob, prints each file's path once above its hits, cuts lines at 300 chars and uses extended regex on both backends. The two schema additions cost one re-prefill of the static prefix after upgrading.
- **`edit_file` keeps CRLF line endings; `run_cmd` reports failures** (`366c860`): editing a Windows-style file no longer rewrites all its line endings to LF. Only the replaced text changes, even in files with mixed endings. A command that fails with output now ends with `[exit N]`; before, the exit code was only shown when there was no output, so a failing test run looked like a pass.
- **`/undo` fixes** (`a7e94e8`): a multi-line prompt is printed on `/undo` instead of being put back into the single-line input buffer, where it garbled the display. Restore and remove messages now show only the path, and the README's cost note for rollbacks on SWA models now says the cost hasn't been measured.
- **Multi-line prompts with a trailing `\`** (`1eeb6dc`): a line ending in `\` opens a `...` continuation prompt, and the lines are sent as one message. This works in the main prompt only; `interject` and `steer` stay single-line.
- **`@path` attachments** (`1b84ee4`): any `@token` in a prompt that names an existing file attaches that file's contents to the message. They're rendered by `read_file`, so the model gets the same line cap and read-on instruction as its own reads.
- **`!cmd` and `!!cmd` prompt escapes** (`392971c`): `!cmd` runs a shell command from the prompt, and its output goes to the model along with your next message. `!!cmd` only shows the output to you. `run_cmd`'s execution moved into a shared `run_shell`, so both use one output cap and timeout.
- **`/undo`** (`8110c11`): takes back the last turn. Files are restored from a git working-tree snapshot taken before the turn, through a temporary index. Files the turn created are removed, and the turn is cut off the history. Its prompt goes back into the input line. Your real index, HEAD and stash are never touched. Outside a git repo only the conversation is rewound. New module `checkpoint.py` with tests.
- **Test suite** (`3a3ef62`, `5da78dd`): a pytest suite under `tests/` (86 tests at the time; 95 with the `/undo` tests) covering `run_turn` over a scripted fake model (interjections, Tab steering, the loop check), `append_file`, `confirm`, `edit_file`, `find_files`, `read_file`, output capping and history trimming. Six stale trim tests were fixed on the way. The project skills now document how to run and extend the suite.
- **Loop check, `--check-every N`** (`bbd4eeb`): every N tool round-trips (default 20, `0` = never), the model is asked on a copy of the history whether it is going in circles. `STUCK` ends the turn on its explanation; `CONTINUE` leaves no trace. It does nothing under the default `--max-steps 20`. Use it with a higher cap or `--max-steps 0`.
- **Tab to stop and steer** (`ab90903`): Tab stops a streaming reply immediately. The partial text and reasoning are kept, any pending tool call is dropped without running, and a `steer` prompt asks for a note the model must follow. It costs no step.
- **Live view and keys stay responsive while a tool call is composed** (`80b1aef`): the response is read on a pump thread, so Esc and typing work during the silent phase and an elapsed-time notice replaces the frozen display. Cancelling shuts the socket down, so Ollama actually stops generating the abandoned call.
- **`append_file` tool** (`16c2e1b`): appends text to a file or creates it. It inserts a separating newline if needed, keeps line endings, and shows the same diff and confirmation as `edit_file`.
- **`edit_file` can fill an empty file** (`56b89a3`): an empty `old_string` on an existing empty file now writes the contents instead of being refused.
- **Rich markup escaping** (`cce49de`): tool results like `[lines 1-100 of 543]` and the `[y/N/reason]` hint were silently swallowed as markup tags. Interpolated text is now escaped at every console site.
- Docs: the verification checklist covers the new key paths (`ea5a327`), and the key table says "any other key" (`e72da6a`).

### 2026-09-15

- **Interjections and denial reasons** (`0ba6421`): any key during a reply opens an `interject` prompt. The message is queued and delivered at the next step boundary, or as the next prompt if the turn ends first. The confirmation prompt is now `[y/N/reason]`, and a typed reason is passed back to the model so it changes course instead of repeating the call.
- **Trimming prefers stubbing tool outputs over shedding thinking** (`41d5b1c`): a mid-turn trim stubs every tool result the model has already replied to. Old thinking is dropped only when stubbing alone isn't enough.
- **Mid-turn updates stay on screen** (`c6efa18`, merged in #4 as `352d7f8`): prose that a step writes alongside its tool calls is re-rendered above the `→ tool(...)` lines instead of vanishing with the live region.
