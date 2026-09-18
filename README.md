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
| `AGENT_NUM_CTX` | `24576` | Ollama context window size; lower on memory-constrained machines |
| `AGENT_STREAM_TIMEOUT` | `300` | Per-read socket timeout (s) on streaming calls; fires only when nothing arrives at all. The call right after a trim pass instead gets a timeout sized to its full re-prefill (estimated tokens ÷ measured prefill rate) |

### Interactive commands

| Input | Effect |
|-------|--------|
| `exit` / `quit` | Quit the agent |
| `/clear` or `clear` | Reset conversation history (system prompt and tool schemas stay cached) |
| `/save [NAME]` | Save the current conversation as a session (defaults to a timestamp) |
| `/resume [NAME]` | Resume a saved session; bare `/resume` resumes the most recent |
| `/sessions` | List saved sessions |
| Any key during a reply | Pauses the stream and opens an `interject` prompt for a message to the model; the key you typed becomes the first character of the line. Delivered at the next step boundary (after the current tool round-trip finishes), or as the next prompt if the turn ends first. Submit an empty line to think better of it |
| Esc during a reply | Cancel the in-flight response |
| Up / Down arrows | Recall previous prompts (history persists in `~/.tiny_agent_history`) |

Esc is the only key that cancels, and only on its own: arrow keys and other escape sequences are drained rather than read as a cancel. Once a reply has started streaming, keys are polled every quarter second whether or not anything arrives. Before that — during the silent re-prefill after a trim pass, when Ollama hasn't sent even the response headers yet — nothing is seen until generation starts. A multi-line paste into a stream arrives as one interjection per line.

While the model composes a tool call nothing streams — Ollama sends the call as one piece when it is complete — so after a couple of silent seconds the live region shows an elapsed-time notice instead of freezing, and the keys keep working throughout. Cancelling in that phase shuts the connection down outright, so Ollama stops generating the abandoned call instead of making your next request wait behind it.

## Tools available to the model

| Tool | Description |
|------|-------------|
| `read_file` | Read a file with optional line range (capped at 100 lines per call) |
| `grep` | Search files by regex (with optional context lines), rg/grep auto-detection |
| `find_files` | Glob-pattern file search |
| `list_dir` | List directory contents |
| `cd` | Change the working directory |
| `edit_file` | Exact-string replacement edit, or create a new file (an empty existing file counts as new) |
| `append_file` | Append text to the end of a file (creating it if missing); inserts a separating newline if needed and keeps the file's line endings |
| `run_cmd` | Run a shell command |

`edit_file`, `append_file` and `run_cmd` ask for confirmation before executing unless `--yes` is passed. Edits show a colored unified diff before the confirmation prompt (and under `--yes`, as a record of what changed). The prompt is `[y/N/reason]`: `y` approves, empty/`n` declines, and anything else declines *and* is passed back to the model as the reason — `use the test runner, not python directly` redirects it, where a bare refusal tends to make a small model re-issue the identical call.

**Security note:** `--yes` auto-approves every write and shell command with no confirmation, and `run_cmd` executes with `shell=True`, so the model can run anything a real shell command can. Only use `--yes` in a repo/directory you trust the agent with.

After each model turn a dim stats line is printed, e.g. `prefill 142 tok in 3.2s · gen 56 tok @ 8.4 tok/s`. The prefill count covers only tokens *not* served from the KV prefix cache, so a small number on a long conversation means the prefix caching is working.

## Design notes

**Static system prompt** — the system prompt and tool schemas are byte-identical on every call so Ollama's KV-cache prefix caching fires: the expensive prefill of the *static prefix* happens once per session, not once per turn. Conversation history still grows every turn and re-prefills incrementally as it grows — see "reasoning feedback" below for the one place that's paid more than once.

**Dynamic context in user messages** — the working directory and task go in the first user message, keeping the system prompt unchanged across projects.

**Lazy context** — the model is not front-loaded with files. It greps to locate code and then reads a tight line range. This keeps per-turn prefill small.

**History trimming** — editing history busts the KV cache from the edit point on, and on a sliding-window-attention model (gemma) the next call then re-prefills the whole prompt from token 0 — so the cost of any trim is the *post-trim prompt size*, and edits are only ever made at two already-paid moments. First, at every turn boundary: the finished turn's tool outputs (all but the most recent few) collapse to one-line stubs, piggybacked on the same cache bust the thinking strip below already causes, so completed turns stay lean and multi-turn sessions reach each new task with a nearly-empty window. Second, lazily mid-turn: when a single long turn pushes the estimate past ~70% of the context window, one pass collapses every tool output the model has already replied to — its kept thinking is the distilled record of them, and raw file reads are the biggest single contributor — while the trailing results it hasn't seen yet stay verbatim. Thinking itself is preferentially kept: only if stubbing alone can't get back under the trigger does the pass also shed all but the last few steps' thinking, since otherwise every later step would bust the cache for its one new output. If that still leaves the estimate over ~90%, a backstop hard-truncates other oversized messages outside the most recent few, so the prompt can't silently grow past `NUM_CTX` and push the system prompt itself out of Ollama's context. The model call right after a mid-turn trim re-prefills the whole trimmed prompt in silence (no tokens stream during prefill), so that one call runs with a stall timeout sized to the actual re-prefill — estimated tokens over the prefill rate measured from live stats — and gets one retry if it still times out, since the server keeps its prefill progress and the retry resumes nearly free.

**Mid-turn updates stay on screen** — while a reply streams, the reasoning and the text-so-far share a transient live region that is wiped when the stream ends. The reasoning is meant to go: it is scratch work, and it is dropped from history too. Text the model writes *for the user* is not, so whenever a step produces prose alongside its tool calls — "let me check the config first" — that prose is re-rendered as Markdown above the `→ tool(...)` lines it explains, exactly as a final answer is. A whole turn's narration therefore reads back in order once the turn ends, instead of only its last message surviving. This is display only: the text was always carried on the assistant message and fed back to the model either way, so nothing about the request payload or the prefix cache changes.

**Reasoning feedback** — a model's `thinking` output is carried on its assistant message and fed back on later tool round-trips *within* a turn, so it doesn't have to re-derive its chain of thought after every tool result. It's stripped once the turn produces a final answer (and, under mid-turn context pressure, trimming may shed all but the last few steps' thinking early — though only as a fallback when stubbing tool outputs alone isn't enough, see above). On a thinking model, that strip busts the KV cache back to the start of the turn, so each multi-step turn re-prefills its own tool round-trips on the next turn — the "once per session" prefill claim above applies to the static system prompt and schemas, not to every token exchanged.

**Surviving an Ollama restart mid-turn** — on a memory-constrained machine, a long session can OOM-kill the Ollama runner (host-memory prompt-cache state scales with resident context tokens). systemd typically restarts Ollama within seconds; if the next request hits that window as connection-refused, the agent waits `RETRY_REFUSED_DELAY` (5s) and retries once with the identical payload before giving up. Lower `AGENT_NUM_CTX` if OOM kills recur.

**Interjections** — any keypress during a reply opens a prompt for a message to the model (Esc alone still cancels). What you type is queued, not injected: the model is already generating, and an assistant message carrying `tool_calls` must be followed immediately by its tool results, so slipping a user message in between would corrupt history. The queue is drained at the next step boundary — after the current round-trip's tool results, before the next model call — and appended there as an ordinary user message marked `[user message sent mid-task]`, the same tail-append shape as the step-limit and empty-reply nudges, so only its own tokens prefill and the cached prefix is untouched. If the turn ends before the next boundary (final answer, cancel, step limit, or a connection error), whatever is still queued becomes the next prompt instead of being swallowed. Because `cbreak` echo is off, the key that opened the prompt would otherwise be lost, so it is pre-filled into the line — typing straight into a stream keeps every character, which is what made `q` safe to drop as a second cancel key.

**Native tool calling** — tools are passed via Ollama's `tools` parameter as JSON schemas, not described in the system prompt, so the model uses the format it was actually trained on.
