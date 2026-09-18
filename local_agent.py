#!/usr/bin/env python3
"""
tiny-agent — a minimal local coding agent for CPU-bound Ollama models.

Built around the constraint that matters on a no-GPU laptop: prefill is
expensive, so the whole design is about minimising and reusing prefilled tokens.

Design decisions (the "why" behind the code):

  1. STATIC SYSTEM PROMPT. Byte-identical on every call so Ollama's KV-cache
     prefix caching kicks in: the system prompt is prefilled once per session,
     not once per turn. First call is still slow; every call after skips it.

  2. NATIVE TOOL CALLING. Tools are passed via Ollama's `tools` parameter as
     JSON schemas, not described in the system prompt. The model responds with
     structured `tool_calls` using the format it was actually trained on —
     fighting it with custom XML just produces unreliable plain-text output.
     The schemas are static, so they are also part of the cached prefix.

  3. DYNAMIC CONTEXT LIVES IN USER MESSAGES. Even the working directory goes
     in the first user message so the system prompt stays byte-identical
     across projects and the prefix cache survives.

  4. LAZY CONTEXT. The model is not front-loaded with files. It greps to
     locate code, then reads a tight line range. A 30-line read beats
     dumping a 400-line service class into the context window.

Usage:
    python local_agent.py "review the null handling in AuthService"
    python local_agent.py            # interactive; seed a task at the prompt
    python local_agent.py --max-steps 0 --check-every 10
                                     # no step cap; every 10 steps ask the model
                                     # whether it is looping, stop if it says so

While a reply streams: any keypress opens a prompt for a message to the model
(queued and delivered at the next step boundary, so the cached prefix is
untouched), Tab stops the reply now and asks for a note to steer it (the
pending tool call is dropped), Esc alone cancels the reply. A declined edit
or command can carry a reason, which is handed to the model in place of a
bare refusal.

Env vars:
    AGENT_MODEL           (default: gemma4:12b-it-qat)   — any Ollama model with tool support
    OLLAMA_URL            (default: http://localhost:11434)
    AGENT_THINK           (default: 1) — set to 0 to disable reasoning output
    AGENT_NUM_CTX         (default: 24576) — Ollama context window size; lower on memory-constrained machines
    AGENT_STREAM_TIMEOUT  (default: 300) — per-read socket timeout (s) on streaming calls; the call right
                          after a trim pass instead gets a timeout sized to its full re-prefill

Dependency: pip install rich
Note: ensure Ollama >= 0.20.2 for reliable Gemma 4 tool-call parsing.
"""

from agent import main

if __name__ == "__main__":
    main()
