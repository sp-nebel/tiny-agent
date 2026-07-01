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
```

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model TAG` | `gemma4:12b-it-qat` | Ollama model tag |
| `--yes` | off | Auto-approve all writes and shell commands |
| `--max-steps N` | 20 | Max tool round-trips before giving up on a task — one round can include several tool calls if the model requests them together, so this isn't a raw tool-call count (`0` = unlimited) |
| `--resume [NAME]` | off | Resume a saved session by name; bare `--resume` resumes the most recent |

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_MODEL` | `gemma4:12b-it-qat` | Ollama model tag (overridden by `--model`) |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API base URL |
| `AGENT_THINK` | `1` | Set to `0` to disable reasoning output |
| `AGENT_SUMMARIZE_TRIM` | `1` | Set to `0` to elide trimmed tool outputs instead of summarizing them |

### Interactive commands

| Input | Effect |
|-------|--------|
| `exit` / `quit` | Quit the agent |
| `/clear` or `clear` | Reset conversation history (system prompt and tool schemas stay cached) |
| `/save [NAME]` | Save the current conversation as a session (defaults to a timestamp) |
| `/resume [NAME]` | Resume a saved session; bare `/resume` resumes the most recent |
| `/sessions` | List saved sessions |
| Esc / `q` / `Q` during a reply | Cancel the in-flight response |
| Up / Down arrows | Recall previous prompts (history persists in `~/.tiny_agent_history`) |

## Tools available to the model

| Tool | Description |
|------|-------------|
| `read_file` | Read a file with optional line range (capped at 100 lines per call) |
| `grep` | Search files by regex (with optional context lines), rg/grep auto-detection |
| `find_files` | Glob-pattern file search |
| `list_dir` | List directory contents |
| `cd` | Change the working directory |
| `edit_file` | Exact-string replacement edit, or create a new file |
| `run_cmd` | Run a shell command |

`edit_file` and `run_cmd` ask for confirmation before executing unless `--yes` is passed. Edits show a colored unified diff before the confirmation prompt (and under `--yes`, as a record of what changed).

**Security note:** `--yes` auto-approves every write and shell command with no confirmation, and `run_cmd` executes with `shell=True`, so the model can run anything a real shell command can. Only use `--yes` in a repo/directory you trust the agent with.

After each model turn a dim stats line is printed, e.g. `prefill 142 tok in 3.2s · gen 56 tok @ 8.4 tok/s`. The prefill count covers only tokens *not* served from the KV prefix cache, so a small number on a long conversation means the prefix caching is working.

## Design notes

**Static system prompt** — the system prompt and tool schemas are byte-identical on every call so Ollama's KV-cache prefix caching fires: the expensive prefill of the *static prefix* happens once per session, not once per turn. Conversation history still grows every turn and re-prefills incrementally as it grows — see "reasoning feedback" below for the one place that's paid more than once.

**Dynamic context in user messages** — the working directory and task go in the first user message, keeping the system prompt unchanged across projects.

**Lazy context** — the model is not front-loaded with files. It greps to locate code and then reads a tight line range. This keeps per-turn prefill small.

**History trimming** — when the conversation crosses ~70% of the context window, old tool outputs (all but the last few) are collapsed in one pass, bounding context-window usage. Trimming is lazy because editing history busts the KV cache from the edit point on, so it happens once per long session rather than every turn. If collapsing tool outputs alone still leaves the estimate over ~90% of the context window, a backstop hard-truncates other oversized messages outside the most recent few, so the prompt can't silently grow past `NUM_CTX` and push the system prompt itself out of Ollama's context.

**Reasoning feedback** — a model's `thinking` output is carried on its assistant message and fed back on later tool round-trips *within* a turn, so it doesn't have to re-derive its chain of thought after every tool result. It's stripped once the turn produces a final answer. On a thinking model, that strip busts the KV cache back to the start of the turn, so each multi-step turn re-prefills its own tool round-trips on the next turn — the "once per session" prefill claim above applies to the static system prompt and schemas, not to every token exchanged.

**Summarize-on-trim** — rather than discarding a collapsed output to a bare `[elided]` stub, the agent spawns a fresh, empty Ollama session on the *same* resident model to digest it down to the task-relevant facts (paths, line numbers, names, errors) and keeps that digest. This only ever runs at the trim moment — when the window is already under pressure and the cache is being busted anyway — so short sessions pay nothing for it. Failures fall back to the plain stub. Disable with `AGENT_SUMMARIZE_TRIM=0`.

**Native tool calling** — tools are passed via Ollama's `tools` parameter as JSON schemas, not described in the system prompt, so the model uses the format it was actually trained on.
