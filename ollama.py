import json
import time
import random
import queue
import socket
import threading
import http.client
import urllib.request
import urllib.error

from rich.live import Live

import config
from ui import cbreak_stdin, poll_keypress, read_interjection, _render_stream

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


_EOF = object()


def _iter_with_ticks(resp, tick=0.25):
    """Yield the response's lines as they arrive, and None once per `tick`
    of silence in between.

    Ollama parses a tool call server-side and sends it as one chunk when it
    is complete, so while the model composes a long edit nothing arrives for
    minutes. A plain `for raw in resp` blocks inside the socket read for that
    whole time: the live region freezes with no sign of life, and — since
    keys are only polled per chunk — Esc and typing go dead in exactly the
    phase where the user most wants them. Reading on a helper thread and
    handing lines over a queue lets the caller wake up every tick to poll
    keys and repaint an elapsed-time notice.

    Errors from the read (the socket stall timeout included — urlopen's
    timeout still governs the underlying socket) are re-raised on the
    caller's thread as the same exception object, so call_ollama's
    `except socket.timeout` stall handling is unchanged. The queue is
    unbounded so the pump never blocks on a consumer that has gone away; the
    thread is a daemon, and a caller that stops early must _abort_stream the
    response so the pump's blocked read actually returns (see there).
    """
    q = queue.Queue()

    def pump():
        try:
            for line in resp:
                q.put(line)
        # AttributeError is the abort race, not a bug to surface: after an
        # _abort_stream the caller's `with resp` sets resp.fp to None while
        # this read is still unwinding, and http.client's own cleanup then
        # calls None.close(). Uncaught it would print a thread traceback
        # over the live view.
        except (OSError, ValueError, AttributeError, http.client.HTTPException) as e:
            q.put(e)    # after an _abort_stream nobody reads this; it just drops
        q.put(_EOF)

    threading.Thread(target=pump, daemon=True).start()
    while True:
        try:
            item = q.get(timeout=tick)
        except queue.Empty:
            yield None
            continue
        if item is _EOF:
            return
        if isinstance(item, Exception):
            raise item
        yield item


def _abort_stream(resp):
    """Tear the connection down under a pump thread still blocked in recv().

    Closing the response isn't enough once _iter_with_ticks reads on another
    thread: on Linux, close() on an fd doesn't wake a recv() blocked on it in
    a different thread, and the in-flight syscall keeps the socket alive, so
    no FIN goes out. While tokens stream the next chunk unblocks the pump
    soon enough — but during a silent tool-call composition, exactly when
    Esc is now usable, Ollama would keep generating the discarded call for
    minutes, and with a single slot the next request (the next prompt) would
    queue behind it. shutdown() wakes the blocked recv and sends the FIN.

    Called only where call_ollama abandons a live stream, never on normal
    completion: after the final chunk resp.fp is already None, and a
    shutdown on a finished connection can raise ENOTCONN. `_sock` is private
    to the socket's file wrapper, hence the AttributeError fallthrough to
    the plain close that `with resp` does anyway.
    """
    try:
        resp.fp.raw._sock.shutdown(socket.SHUT_RDWR)
    except (AttributeError, OSError):
        pass


def _chat_request(payload):
    return urllib.request.Request(
        f"{config.OLLAMA_URL}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )


# Error bodies that mean the prompt itself is too big: resending the same
# payload can only fail the same way, however long we wait.
_OVERFLOW_HINTS = ("context length", "context window", "exceeds the context",
                   "too many tokens", "prompt is too long")


def _retry_reason(err, body=""):
    """A short label when `err` is worth retrying with the same payload, else
    None. Covers the transient failures of a local server: overloaded or
    busy (429, 503), the runner crashing or being OOM-killed mid-load (500,
    502, 504), and the connection refused or reset while systemd restarts
    it. A 4xx is the request's own fault and never retried."""
    if isinstance(err, urllib.error.HTTPError):
        if err.code not in (429, 500, 502, 503, 504):
            return None
        if any(h in body.lower() for h in _OVERFLOW_HINTS):
            return None
        return f"HTTP {err.code}"
    reason = err.reason if isinstance(err, urllib.error.URLError) else err
    if isinstance(reason, ConnectionRefusedError):
        return "connection refused"
    if isinstance(reason, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
        return "connection reset"
    return None


def _backoff(why, attempt):
    """Sleep before retry `attempt` (0-based): 2s, 4s, 8s … capped, with
    jitter so a burst of failures doesn't retry in lockstep."""
    delay = min(config.RETRY_MAX_DELAY, config.RETRY_BASE_DELAY * 2 ** attempt)
    delay *= random.uniform(0.8, 1.2)
    config.console.print(f"[dim]Ollama: {why}; retry {attempt + 1}/{config.RETRY_MAX} "
                         f"in {delay:.0f}s…[/dim]")
    time.sleep(delay)


def call_ollama(messages, timeout=None, retry_stall=False, _attempt=0):
    """Stream a chat turn.

    timeout overrides STREAM_TIMEOUT for this call only — run_turn passes a
    prefill-sized value on the call right after a trim pass, whose prefill
    legitimately stays silent far longer than a healthy-cache call. On that
    same call it also sets retry_stall: if the stream still times out, retry
    once with an identical payload before giving up — the server keeps the
    prefill progress it made (slot cache + checkpoints), so the retry resumes
    nearly free instead of losing minutes of CPU work with the finish line
    in sight.

    Returns (content: str, thinking: str, tool_calls: list, cancelled: bool,
    interrupted: bool, stats: dict). Exactly one of content/tool_calls is
    meaningful: a final answer has content and no tool_calls; a tool-calling
    turn has tool_calls.
    Reasoning arrives in a separate `thinking` field; it is shown live and now
    also returned, so run_turn can carry it on the assistant message and feed
    it back on the next step — keeping the model's chain of thought intact
    across tool round-trips until a final answer lands. cancelled is True if
    the user pressed Esc mid-stream. interrupted is True if they pressed Tab
    to stop and steer: content/thinking then hold the reply so far and
    tool_calls whatever had already arrived, which run_turn drops. stats is
    a dict of raw token counters from the final chunk, empty ({}) in both
    cases (the final chunk never arrived) — run_turn sums it across the turn.

    Any other key, polled the same way, opens a prompt for a mid-reply message
    to the model; that text is queued in ui (see read_interjection) rather than
    returned, so the retry paths above can't discard it along with the partial
    stream they throw away.
    """
    payload = _build_payload(messages, stream=True, tools=True, think=config.THINK)
    req = _chat_request(payload)
    if timeout is None:
        timeout = config.STREAM_TIMEOUT

    content     = ""
    thinking    = ""
    tool_calls  = []
    cancelled   = False
    interrupted = False
    stats       = {}
    last_paint  = 0.0
    last_data   = time.monotonic()

    # The outer try exists for the post-trim stall retry: the silent prefill
    # can time out either while waiting for the response headers (Ollama
    # flushes them with the first generated chunk) or mid-stream — both raise
    # a raw socket.timeout, so one enclosing handler covers both sites.
    try:
        try:
            resp = urllib.request.urlopen(req, timeout=timeout)
        except (urllib.error.URLError, ConnectionError) as e:   # HTTPError too
            body = ""
            if isinstance(e, urllib.error.HTTPError):
                body = e.read().decode(errors="replace")
                # Model doesn't support the `think` parameter — disable and
                # retry once so non-thinking models (the default) keep
                # working. Other 400s (no tool support, bad request) must
                # surface, so check the error body.
                if config.THINK and e.code == 400 and "think" in body.lower():
                    config.THINK = False
                    return call_ollama(messages, timeout=timeout,
                                       retry_stall=retry_stall, _attempt=_attempt)
                e.body = body   # already consumed; stash for the handler in main()
            # Refused or reset usually means Ollama is mid-restart (systemd
            # bouncing it back up seconds after an OOM kill). Backing off
            # rides through that window instead of losing the turn. Only
            # here, before any of the reply streamed: the resent payload is
            # byte-identical, so the retry costs no cache. A reset raised by
            # getresponse() arrives unwrapped, hence ConnectionError too.
            why = _retry_reason(e, body)
            if why and _attempt < config.RETRY_MAX:
                _backoff(why, _attempt)
                return call_ollama(messages, timeout=timeout,
                                   retry_stall=retry_stall, _attempt=_attempt + 1)
            raise

        # cbreak lets us catch a single keypress — Esc to cancel, Tab to
        # stop and steer, anything else to interject — without blocking the
        # stream; transient=True
        # clears the live region (thinking included) when done, so run_turn
        # re-renders the content it kept — a final answer or a mid-turn
        # update — while the thinking stays wiped.
        with resp, cbreak_stdin():
            try:
                with Live(console=config.console, refresh_per_second=8, transient=True) as live:
                    for raw in _iter_with_ticks(resp):
                        # Keys first, on idle ticks too: a silent tool-call
                        # composition is when the user most wants them.
                        action, seed = poll_keypress()
                        if action == "cancel":
                            cancelled = True
                            _abort_stream(resp)
                            break
                        if action == "stop":
                            # Stop now, unlike a queued interjection: the
                            # partial reply goes back to run_turn, which reads
                            # the user's note. The generation is abandoned
                            # just as surely as on Esc, so tear it down too.
                            interrupted = True
                            _abort_stream(resp)
                            break
                        if action == "interject":
                            # Queued in ui, not returned: the retry paths above
                            # discard this stream and start over, and the message
                            # must survive that. run_turn drains the queue at the
                            # next step boundary.
                            read_interjection(live, seed,
                                              _render_stream(thinking, content))

                        if raw is None:
                            # Idle tick: nothing arrived. Repaint with how long the
                            # silence has lasted so a long tool call in progress
                            # doesn't look like a hang.
                            live.update(_render_stream(thinking, content,
                                                       quiet_s=time.monotonic() - last_data))
                            continue
                        last_data = time.monotonic()

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
            except KeyboardInterrupt:
                # Ctrl-C abandons the stream like Esc does, and the pump is
                # blocked in recv just the same — without the shutdown Ollama
                # keeps generating and the next request queues behind it.
                _abort_stream(resp)
                raise
    except socket.timeout:
        # Retry once, with retry_stall cleared so the retried request can't
        # keep retrying itself. Any partial content is discarded — a fresh
        # request re-prefills its prompt from the server's retained cache
        # and regenerates.
        if retry_stall:
            config.console.print(
                f"[dim]no data for {timeout}s on the post-trim call; retrying "
                f"once (the server keeps its prefill progress)…[/dim]"
            )
            return call_ollama(messages, timeout=timeout, _attempt=_attempt)
        raise

    return content, thinking, tool_calls, cancelled, interrupted, stats


def warm_cache(messages=None):
    """Prefill a message prefix in the background so the model is loaded and
    the KV cache is warm by the time it's needed for a real turn.

    Defaults to just the static system prompt (the cold-start case). Passing
    the full restored history after a resume warms that instead, so the next
    real turn only prefills the newly-typed user message.

    Options must match call_ollama exactly — a different num_ctx would make
    Ollama reload the model, and a different `think` would make it render a
    different prompt template, both wasting the warmup by prefilling a prefix
    the real call won't reuse. Failures are ignored; the real call will
    surface them.
    """
    if messages is None:
        messages = [{"role": "system", "content": config.SYSTEM}]
    payload = _build_payload(messages, stream=False, tools=True, think=config.THINK, num_predict=1)
    req = _chat_request(payload)
    try:
        with urllib.request.urlopen(req, timeout=config.STREAM_TIMEOUT) as resp:
            resp.read()
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        # Same think-unsupported fallback as call_ollama: retry once with it
        # disabled so the warmup still happens (and the real call downstream
        # skips the same 400 round-trip) instead of losing the warmup outright.
        if config.THINK and e.code == 400 and "think" in body.lower():
            config.THINK = False
            warm_cache(messages)
    except OSError:
        pass
