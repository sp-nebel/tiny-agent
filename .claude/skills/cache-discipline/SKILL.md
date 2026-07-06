---
name: cache-discipline
description: Vet any tiny-agent change that touches the request payload, message list/history, system prompt, tool schemas, token accounting, or trimming against the KV prefix-cache-preservation doctrine. Use before proposing changes to agent.py (run_turn, trim_history), ollama.py (_build_payload, call_ollama, warm_cache), or config.py (SYSTEM, TOOL_SCHEMAS, NUM_CTX and trim thresholds). Always state the cache impact of such a change before making it.
---

# Cache discipline — the load-bearing design rule

This project runs LLMs on CPU, where prefill dominates cost. Ollama's KV prefix cache skips
prefill for any request whose byte prefix matches the previous request. **Changing bytes at
position N forces re-prefill of everything from N onward.** This is the project's central
design constraint; the user vets every change against it and will reject changes that violate
it (a past proposal to drop tool schemas from one request was rejected for exactly this).

**Before proposing any change to payloads, message history, or prompt structure, explicitly
state its cache impact.** If you can't state it, you don't understand the change yet.

## The rules

1. **The static prefix is sacred.** `config.SYSTEM` and `config.TOOL_SCHEMAS` are serialized
   at the front of every request and must be byte-identical across calls, sessions, and
   projects. Never interpolate anything dynamic into them (no cwd, no date, no per-project
   text). Dynamic context goes in **user messages** — the cwd goes in the *first* user message
   only (see `main()` in agent.py). Editing SYSTEM/TOOL_SCHEMAS in a commit is fine (users pay
   one re-prefill after upgrading); making them *vary at runtime* is not.

2. **Appending at the tail is free; editing committed history busts the cache** from the edit
   point on. This is why:
   - The step-limit and empty-reply nudges (`STEP_LIMIT_NUDGE`, `EMPTY_RETRY_NUDGE`) are
     *appended* user messages, not rewrites of earlier ones.
   - The forced final step keeps the tool schemas in the payload (removing them would shift
     every byte after that point) and nudges via the tail instead.
   - `thinking` is *carried forward* on appended assistant messages during a turn rather than
     re-sent some other way.

3. **Cache busts are only allowed at already-expensive, bounded moments:**
   - `trim_history`: once per long session, and only when the estimate crosses
     `TRIM_AT_TOKENS`. It deliberately does nothing before that ("untouched history is free").
   - `drop_thinking` / `strip_nudges`: once per turn boundary, in `run_turn`'s `finally`.
   A feature that edits history (or varies the payload head) *per step* is wrong by
   construction — redesign it to append, or to piggyback on one of these existing moments.

4. **All request bodies go through `_build_payload`** (ollama.py). `model`, `keep_alive`,
   `options.num_ctx`, and `think` must match between `warm_cache` and `call_ollama`: a
   different `num_ctx` makes Ollama reload the model, and a different `think` renders a
   different prompt template — either wastes the background warmup entirely.
   `summarize_output` also matches `num_ctx` for the same reason. Never hand-build a payload.

5. **Token accounting must count everything that rides in the request body.** `_msg_tokens`
   (agent.py) counts `content` + `thinking` + serialized `tool_calls` at ~4 chars/token. If a
   new field is added to messages, add it there. Undercounting lets the prompt silently exceed
   `NUM_CTX`; Ollama then drops the *front* of the prompt — the system prompt — which both
   lobotomizes the agent and kills the prefix cache for the rest of the session. The estimate
   only needs to be accurate within the headroom (`TRIM_AT_TOKENS` = 70% of `NUM_CTX`,
   `HARD_TRUNCATE_AT_TOKENS` = 90%).

## Review checklist

Run this on any diff touching messages/payload/prompt:

1. Do any bytes *before the tail of the message list* change between consecutive calls? Where?
2. If history is edited, does it happen only inside `trim_history` or the turn-boundary
   cleanup? Is it idempotent (guarded by `TRIM_PREFIX` where applicable)?
3. Do `warm_cache`, `call_ollama`, and `summarize_output` still send identical
   model/num_ctx/keep_alive (and matching `think` for the first two)?
4. Does `_msg_tokens` still count every field the change adds to messages?
5. State the verdict in one sentence: "cache impact: none / one bust at <existing moment> /
   NEW bust — needs justification."

Verify empirically after live changes: the per-turn stats line prints `prefill N tok`; on
turn 2+ of a conversation N should be small (roughly the new tokens only). A full-prompt-sized
prefill mid-session means your change busted the cache. See `verifying-tiny-agent`.
