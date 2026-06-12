>[!WARNING]
>Only vibecoding ahead

# mini-agent

A minimal local coding agent for CPU-bound machines, built around [Ollama](https://ollama.com).

The whole design is shaped by one constraint: **no GPU**. Prefill is expensive on CPU, so every architectural decision is about keeping the prefilled token count small and reusing it across turns via Ollama's KV-cache prefix caching.

## Requirements

- Python 3.8+
- [Ollama](https://ollama.com) ≥ 0.20.2
- `pip install rich`
- A model with tool-call support pulled in Ollama (default: `gemma4:latest`)

## Setup

```bash
pip install rich
ollama pull gemma4:latest   # or any tool-capable model
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
| `--model TAG` | `gemma4:latest` | Ollama model tag |
| `--yes` | off | Auto-approve all writes and shell commands |
| `--max-steps N` | 20 | Max tool calls before giving up on a task |

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_MODEL` | `gemma4:latest` | Ollama model tag (overridden by `--model`) |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API base URL |
| `AGENT_THINK` | `1` | Set to `0` to disable reasoning output |

### Interactive commands

| Input | Effect |
|-------|--------|
| `exit` / `quit` | Quit the agent |
| `/clear` or `clear` | Reset conversation history (system prompt and tool schemas stay cached) |
| Esc / `q` / `Q` during a reply | Cancel the in-flight response |
| Up / Down arrows | Recall previous prompts (history persists in `~/.mini_agent_history`) |

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

After each model turn a dim stats line is printed, e.g. `prefill 142 tok in 3.2s · gen 56 tok @ 8.4 tok/s`. The prefill count covers only tokens *not* served from the KV prefix cache, so a small number on a long conversation means the prefix caching is working.

## Design notes

**Static system prompt** — the system prompt and tool schemas are byte-identical on every call so Ollama's KV-cache prefix caching fires: the expensive prefill happens once per session, not once per turn.

**Dynamic context in user messages** — the working directory and task go in the first user message, keeping the system prompt unchanged across projects.

**Lazy context** — the model is not front-loaded with files. It greps to locate code and then reads a tight line range. This keeps per-turn prefill small.

**History trimming** — old tool outputs are collapsed to stubs after a few turns, bounding both context-window usage and per-turn prefill cost.

**Native tool calling** — tools are passed via Ollama's `tools` parameter as JSON schemas, not described in the system prompt, so the model uses the format it was actually trained on.
