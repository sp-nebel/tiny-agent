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

### Flash attention

Two things to know before turning it on, because the second one is counter-intuitive:

**It cannot be requested by this agent.** Ollama resolves flash attention when it loads the model, from the environment of the `ollama serve` process, and there is no per-request API option for it (a `flash_attn` key inside `options` is not a field Ollama parses — it is dropped). Ollama's own default is `auto`, which leaves the decision to llama.cpp; `OLLAMA_FLASH_ATTENTION=1` forces it on, `0` forces it off.

**On CPU it does not speed up prefill — it costs throughput.** Flash attention's win is avoiding memory-bandwidth round-trips, which is a GPU problem; on CPU the fused kernel means the `KQ` and `KQV` matmuls stop going through llama.cpp's optimized GEMM path, and measured prompt processing gets *slower* — roughly 6% in mainline llama.cpp and up to 26% in ik_llama.cpp ([benchmark](https://github.com/ikawrakow/ik_llama.cpp/discussions/25)). Since this agent is built for machines with no GPU, flash attention here is a **memory** trade, not a speed one. Turn it on when the runner is OOM-killing (see "Surviving an Ollama restart mid-turn" below), not when prefill feels slow.

If you do want it — it keeps the attention score matrix (batch × context, per head, which at `AGENT_NUM_CTX=24576` is not small) from being materialized in full, and it is the precondition for quantizing the K/V cache:

```bash
# Linux, Ollama installed as a systemd service
sudo systemctl edit ollama
# in the editor, add:
#   [Service]
#   Environment="OLLAMA_FLASH_ATTENTION=1"
sudo systemctl restart ollama

# macOS (desktop app or brew service) — then restart Ollama
launchctl setenv OLLAMA_FLASH_ATTENTION 1

# server started by hand
OLLAMA_FLASH_ATTENTION=1 ollama serve
```

`OLLAMA_KV_CACHE_TYPE=q8_0` halves KV memory again on top of that, but carries the same warning twice over: quantized K/V costs prompt-processing speed on CPU through per-token dequantization, and Gemma in particular has a [reported](https://github.com/ggml-org/llama.cpp/issues/12352) pathology there. Reach for it only under memory pressure, and measure the stats line before and after.

Neither setting changes the bytes this agent sends, so prefix caching is unaffected by them. Restarting the server to apply them does drop the resident model and its prompt cache once — do that between sessions, not mid-task.

**If prefill is what you actually want to improve**, the lever is the number of tokens prefilled, not the attention kernel: that is what prefix caching, lazy context and the trimming design exist for, and a large `prefill` count mid-session means a cache bust, not a slow kernel. See the Design notes.

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

These two are read by the **Ollama server**, not by the agent — set them in `ollama serve`'s environment and restart it (see [Flash attention](#flash-attention)):

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_FLASH_ATTENTION` | `auto` | `1` forces flash attention on, `0` forces it off. Saves memory; costs prompt-processing speed on CPU |
| `OLLAMA_KV_CACHE_TYPE` | `f16` | K/V cache quantization, e.g. `q8_0` to roughly halve KV memory. Requires flash attention; slower still on CPU |

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

**History trimming** — editing history busts the KV cache from the edit point on, and on a sliding-window-attention model (gemma) the next call then re-prefills the whole prompt from token 0 — so the cost of any trim is the *post-trim prompt size*, and edits are only ever made at two already-paid moments. First, at every turn boundary: the finished turn's tool outputs (all but the most recent few) collapse to one-line stubs, piggybacked on the same cache bust the thinking strip below already causes, so completed turns stay lean and multi-turn sessions reach each new task with a nearly-empty window. Second, lazily mid-turn: when a single long turn pushes the estimate past ~70% of the context window, one pass collapses old tool outputs *and* drops `thinking` from all but the last few steps — on a thinking-heavy turn that old reasoning is the bulk of the prompt, and shedding it is what makes the post-trim re-prefill cheap. If that still leaves the estimate over ~90%, a backstop hard-truncates other oversized messages outside the most recent few, so the prompt can't silently grow past `NUM_CTX` and push the system prompt itself out of Ollama's context. The model call right after a mid-turn trim re-prefills the whole trimmed prompt in silence (no tokens stream during prefill), so that one call runs with a stall timeout sized to the actual re-prefill — estimated tokens over the prefill rate measured from live stats — and gets one retry if it still times out, since the server keeps its prefill progress and the retry resumes nearly free.

**Reasoning feedback** — a model's `thinking` output is carried on its assistant message and fed back on later tool round-trips *within* a turn, so it doesn't have to re-derive its chain of thought after every tool result. It's stripped once the turn produces a final answer (and, under mid-turn context pressure, trimming may shed all but the last few steps' thinking early — see above). On a thinking model, that strip busts the KV cache back to the start of the turn, so each multi-step turn re-prefills its own tool round-trips on the next turn — the "once per session" prefill claim above applies to the static system prompt and schemas, not to every token exchanged.

**Flash attention** — a server-side setting, not a request option: Ollama picks it when it loads the model, so the agent cannot ask for it, and the README's Setup section covers turning it on. Worth being explicit about what it buys on this project's target hardware: it is a *memory* optimization, not a prefill speedup. Its benefit is avoiding memory-bandwidth round-trips, which is a GPU bottleneck; on a CPU-only box the fused kernel bypasses llama.cpp's optimized GEMM path and prompt processing measures slower with it on. So it belongs in the same bucket as lowering `NUM_CTX` — a remedy for a runner that is running out of memory — and not in the bucket with prefix caching and lazy context, which are what actually make prefill cheap here.

**Surviving an Ollama restart mid-turn** — on a memory-constrained machine, a long session can OOM-kill the Ollama runner (host-memory prompt-cache state scales with resident context tokens). systemd typically restarts Ollama within seconds; if the next request hits that window as connection-refused, the agent waits `RETRY_REFUSED_DELAY` (5s) and retries once with the identical payload before giving up. Lower `AGENT_NUM_CTX` if OOM kills recur.

**Native tool calling** — tools are passed via Ollama's `tools` parameter as JSON schemas, not described in the system prompt, so the model uses the format it was actually trained on.
