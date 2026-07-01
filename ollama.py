import json
import time
import urllib.request
import urllib.error

from rich.live import Live

import config
from ui import cbreak_stdin, cancel_pressed, _render_stream

# --------------------------------------------------------------------------- #
# Ollama call  (streaming, native tool-call detection)
# --------------------------------------------------------------------------- #

def _build_payload(messages, *, stream, tools=True, think=False, num_predict=None):
    """Assemble an /api/chat body shared by all three call sites.

    Hardcodes model/keep_alive/num_ctx so they can no longer drift between
    callers — warm_cache depends on that to avoid triggering a model reload.
    """
    payload = {
        "model":      config.MODEL,
        "messages":   messages,
        "stream":     stream,
        "keep_alive": "30m",    # keeps model resident → prefix cache stays warm
        "options": {
            "num_ctx": config.NUM_CTX,
        },
    }
    if num_predict is not None:
        payload["options"]["num_predict"] = num_predict
    if tools:
        payload["tools"] = config.TOOL_SCHEMAS
    if think:
        payload["think"] = True
    return payload


def _chat_request(payload):
    return urllib.request.Request(
        f"{config.OLLAMA_URL}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )


def call_ollama(messages):
    """Stream a chat turn.

    Returns (content: str, thinking: str, tool_calls: list, cancelled: bool,
    stats: dict). Exactly one of content/tool_calls is meaningful: a final
    answer has content and no tool_calls; a tool-calling turn has tool_calls.
    Reasoning arrives in a separate `thinking` field; it is shown live and now
    also returned, so run_turn can carry it on the assistant message and feed
    it back on the next step — keeping the model's chain of thought intact
    across tool round-trips until a final answer lands. cancelled is True if
    the user pressed the cancel key (Esc/q) mid-stream; stats is a dict of raw
    token counters from the final chunk, empty ({}) in that case (the final
    chunk never arrived) — run_turn sums it across the turn.
    """
    payload = _build_payload(messages, stream=True, tools=True, think=config.THINK)
    req = _chat_request(payload)

    content    = ""
    thinking   = ""
    tool_calls = []
    cancelled  = False
    stats      = {}
    last_paint = 0.0

    try:
        resp = urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        # Model doesn't support the `think` parameter — disable and retry once
        # so non-thinking models (the default) keep working. Other 400s (no
        # tool support, bad request) must surface, so check the error body.
        if config.THINK and e.code == 400 and "think" in body.lower():
            config.THINK = False
            return call_ollama(messages)
        e.body = body   # already consumed; stash for the handler in main()
        raise

    # cbreak lets us catch a single cancel keypress without blocking the
    # stream; transient=True clears the live region (thinking included) when
    # done, so run_turn re-renders only the final answer.
    with resp, cbreak_stdin():
        with Live(console=config.console, refresh_per_second=8, transient=True) as live:
            for raw in resp:
                if cancel_pressed():
                    cancelled = True
                    break

                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    continue   # malformed chunk (e.g. server hiccup) — skip it
                msg = obj.get("message", {})

                # Reasoning streams in its own field (models that emit it).
                tdelta = msg.get("thinking", "")
                if tdelta:
                    thinking += tdelta

                # The actual answer / final text content.
                delta = msg.get("content", "")
                if delta:
                    content += delta

                # Rebuilding the renderable is O(accumulated text), so doing
                # it per chunk goes quadratic and steals CPU from inference.
                # Live paints at 8 fps anyway; skip updates it would never show.
                if (tdelta or delta) and time.monotonic() - last_paint >= 0.12:
                    live.update(_render_stream(thinking, content))
                    last_paint = time.monotonic()

                # Tool calls appear in a dedicated field, often in a chunk
                # where content is empty.
                tcs = msg.get("tool_calls", [])
                if tcs:
                    tool_calls.extend(tcs)

                if obj.get("done"):
                    # Raw counters; run_turn sums these across the turn and
                    # formats one line at the end via fmt_stats.
                    stats = {
                        "prompt_eval_count":    obj.get("prompt_eval_count", 0),
                        "prompt_eval_duration": obj.get("prompt_eval_duration", 0),
                        "eval_count":           obj.get("eval_count", 0),
                        "eval_duration":        obj.get("eval_duration", 0),
                    }
                    break

    return content, thinking, tool_calls, cancelled, stats


def warm_cache(messages=None):
    """Prefill a message prefix in the background so the model is loaded and
    the KV cache is warm by the time it's needed for a real turn.

    Defaults to just the static system prompt (the cold-start case). Passing
    the full restored history after a resume warms that instead, so the next
    real turn only prefills the newly-typed user message.

    Options must match call_ollama exactly — a different num_ctx would make
    Ollama reload the model and waste the warmup. Failures are ignored; the
    real call will surface them.
    """
    if messages is None:
        messages = [{"role": "system", "content": config.SYSTEM}]
    payload = _build_payload(messages, stream=False, tools=True, num_predict=1)
    req = _chat_request(payload)
    try:
        with urllib.request.urlopen(req) as resp:
            resp.read()
    except OSError:
        pass


def summarize_output(tool_name, content, task):
    """Digest one oversized tool result down to its task-relevant facts.

    Runs on a fresh, empty 2-message session against the same resident model:
    no prior history, no tools (the summarizer must not call any), no thinking.
    num_ctx matches call_ollama exactly so Ollama doesn't reload the model.

    Returns the digest string, or None on any failure (or empty reply) so the
    caller can fall back to the plain elision stub — summarization must never
    break a turn.
    """
    messages = [
        {"role": "system", "content": config.SUMMARIZER_SYSTEM},
        {"role": "user",   "content": f"Task:\n{task}\n\n{tool_name} output:\n{content}"},
    ]
    payload = _build_payload(messages, stream=False, tools=False, num_predict=config.SUMMARY_MAX_TOKENS)
    req = _chat_request(payload)
    try:
        with urllib.request.urlopen(req, timeout=config.CMD_TIMEOUT) as resp:
            obj = json.loads(resp.read())
    except (OSError, ValueError):
        return None
    text = (obj.get("message") or {}).get("content", "").strip()
    return text or None
