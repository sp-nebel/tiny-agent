import os
import json
import atexit
import argparse
import threading
import contextlib
import urllib.error

from rich.markdown import Markdown

import config
from tools import dispatch
from ollama import call_ollama, warm_cache, summarize_output
from ui import read_prompt, fmt_args, truncate, fmt_stats
from session import *

try:                      # line editing + history for the interactive prompt
    import readline
except ImportError:
    readline = None

# --------------------------------------------------------------------------- #
# Agent loop
# --------------------------------------------------------------------------- #

# Appended at the tail on the final allowed step to force a closing answer
# instead of yet another tool call. Tail-only, so the prefix cache is untouched.
STEP_LIMIT_NUDGE = (
    "[step limit reached — give your best final answer now using what you've "
    "gathered; do not call any more tools.]"
)

# Appended at the tail when the model returns a wholly empty reply, to give it
# another swing. Deliberately neutral — an empty turn may just mean the model
# needs another step (more thinking, or a tool call), so we don't force a final
# answer; the retry simply re-enters the loop. Tail-only, so the cache is intact.
EMPTY_RETRY_NUDGE = "[your last reply was empty — please continue.]"
MAX_EMPTY_RETRIES = 2


def _msg_tokens(m):
    """Token estimate for one message, including the fields that actually
    ride along in the request body — not just `content`. `thinking` is fed
    back across a turn's tool round-trips (see run_turn) and `tool_calls` is
    serialized JSON, so a content-only count understates the real prefill and
    lets the window overflow silently (Ollama then drops the front of the
    prompt — the system prompt — and every call re-prefills from scratch)."""
    n = len(m.get("content") or "") + len(m.get("thinking") or "")
    tool_calls = m.get("tool_calls")
    if tool_calls:
        n += len(json.dumps(tool_calls))
    return n // 4


def _total_tokens(messages):
    return sum(_msg_tokens(m) for m in messages)


def trim_history(messages):
    """Collapse old tool outputs in place to conserve the context window.

    Untouched history is free: it sits in the KV prefix cache and is never
    re-prefilled. Editing a message, by contrast, invalidates the cache from
    the edit point on. So trimming is lazy — do nothing until the conversation
    approaches the context window, then collapse every tool result outside the
    keep-window in one pass. One cache bust per long session, not one per step.

    The token estimate uses the ~4 chars/token heuristic; it only needs to be
    right to within the headroom left below NUM_CTX.

    When SUMMARIZE_ON_TRIM is set, each collapsed output is first run through
    summarize_output (a fresh session on the same model) so the kept stub is an
    informative digest rather than a bare "[elided]" line. That fires a burst of
    summarizer calls, but only here — at the already-expensive trim moment — so
    short sessions pay nothing. On failure it falls back to the plain stub.

    Idempotent by construction: every collapsed message is prefixed with
    TRIM_PREFIX and skipped on later passes (a length check no longer suffices,
    since a digest can be longer than TRIM_MIN_CHARS).

    After collapsing eligible tool outputs, a backstop hard-truncates any
    remaining oversized message outside a protected recent tail. Tool-output
    collapsing alone can't help when the bloat is in long assistant/user
    messages, or when the keep-window itself is what's oversized — without
    this, those cases silently overflow NUM_CTX.
    """
    if _total_tokens(messages) < config.TRIM_AT_TOKENS:
        return
    task = next((m["content"] for m in reversed(messages)
                 if m.get("role") == "user"
                 and m.get("content") not in (STEP_LIMIT_NUDGE, EMPTY_RETRY_NUDGE)), "")
    tool_idxs = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    old = tool_idxs[:-config.KEEP_FULL_TOOL_RESULTS] if config.KEEP_FULL_TOOL_RESULTS else tool_idxs
    targets = [i for i in old
               if not (messages[i].get("content") or "").startswith(config.TRIM_PREFIX)
               and len(messages[i].get("content") or "") > config.TRIM_MIN_CHARS]
    if targets:
        config.console.print(f"[dim]compacting {len(targets)} old tool output"
                      f"{'s' if len(targets) != 1 else ''}…[/dim]")
        for i in targets:
            m       = messages[i]
            content = m.get("content", "")
            name    = m.get("name", "tool")
            nlines  = content.count("\n") + 1
            marker  = f"{config.TRIM_PREFIX}{name} — was {nlines} lines, {len(content)} chars]"
            summary = summarize_output(name, content, task) if config.SUMMARIZE_ON_TRIM else None
            m["content"] = f"{marker}\n{summary}" if summary else marker

    if _total_tokens(messages) < config.HARD_TRUNCATE_AT_TOKENS:
        return
    protected = set(range(max(0, len(messages) - config.KEEP_RECENT_MESSAGES), len(messages)))
    protected.add(0)   # system prompt

    # Step 1: hard-truncate oversized content outside the protected tail, and
    # drop `thinking` off those same messages — it's being thrown away anyway.
    # The marker is PREPENDED (unlike the tool-collapse marker style above)
    # so the startswith(TRIM_PREFIX) guard actually recognizes an
    # already-truncated message on a later pass instead of re-editing it
    # every time and destroying the original "was N chars" figure.
    for i, m in enumerate(messages):
        if i in protected:
            continue
        m.pop("thinking", None)
        content = m.get("content") or ""
        if len(content) <= config.TRIM_MIN_CHARS or content.startswith(config.TRIM_PREFIX):
            continue
        marker = f"{config.TRIM_PREFIX}hard-truncated, was {len(content)} chars]"
        m["content"] = f"{marker}\n{content[:config.TRIM_MIN_CHARS]}"
        if _total_tokens(messages) < config.HARD_TRUNCATE_AT_TOKENS:
            return

    # Step 2: content alone wasn't enough — the overage is `thinking` fields
    # on protected (recent) messages, which step 1 never touches. Drop it
    # there too, except on the single most recent assistant message, so a
    # thinking model mid-turn doesn't lose the reasoning it's about to build
    # on for its very next step.
    last_assistant_idx = next((i for i in range(len(messages) - 1, -1, -1)
                                if messages[i].get("role") == "assistant"), None)
    for i in sorted(protected):
        if i == last_assistant_idx:
            continue
        messages[i].pop("thinking", None)
        if _total_tokens(messages) < config.HARD_TRUNCATE_AT_TOKENS:
            return

    # Step 3: still over budget — most likely the protected tail itself
    # (short contents, or the one thinking field we deliberately kept) is
    # simply too large. Warn rather than silently exceed NUM_CTX.
    config.console.print(
        "[yellow]warning: context still exceeds the safety margin after "
        "trimming; the model may lose the system prompt[/yellow]"
    )


def drop_thinking(messages, start):
    """Strip the `thinking` field off assistant messages from `start` onward.

    Reasoning is fed back across a turn's tool round-trips (see run_turn) so the
    model keeps its chain of thought while it works. Once the turn ends we drop
    it: it has served its purpose, and leaving it would bloat the context window
    and persist into saved sessions across turns. This edits committed history,
    so it busts the KV prefix cache from the first stripped message on — paid
    once at the turn boundary, the same one-cache-bust-when-it's-worth-it trade
    trim_history makes.
    """
    for m in messages[start:]:
        if m.get("role") == "assistant":
            m.pop("thinking", None)


def _is_stray_nudge(m):
    """A step-limit/empty-retry nudge, or the empty assistant reply that
    prompted one — the pieces strip_nudges removes once a turn ends. Exposed
    separately so a restored session (main()'s apply_session call sites) can
    run the same filter: a turn interrupted before strip_nudges ran (a
    Ctrl-C or a dropped connection propagating out of run_turn) can persist
    one of these into an autosaved session, and it would otherwise sit there
    forever, including being handed to the summarizer as "the task".
    """
    if m.get("role") == "user" and m.get("content") in (STEP_LIMIT_NUDGE, EMPTY_RETRY_NUDGE):
        return True
    if (m.get("role") == "assistant" and not (m.get("content") or "").strip()
            and not m.get("tool_calls")):
        return True
    return False


def strip_nudges(messages, start):
    """Remove stray nudges (see _is_stray_nudge) from `start` onward, once
    the turn has its real answer. They're single-purpose "try again" prompts
    for getting the model unstuck mid-turn; left in place they persist into
    later turns and saved sessions as stray "[your last reply was empty...]"
    filler. Edits committed history, so — like drop_thinking — this busts
    the KV cache, paid once at the same turn boundary drop_thinking already
    busts it at.
    """
    keep = messages[:start]
    keep.extend(m for m in messages[start:] if not _is_stray_nudge(m))
    messages[:] = keep


def run_turn(messages, max_steps=20):
    # max_steps <= 0 means unlimited: no forced final step, no nudge.
    # A turn fans out into several call_ollama requests (one per tool
    # round-trip); sum their stats and print one summary line when the turn
    # finishes, rather than a line per call.
    turn_stats = {}
    calls = 0
    step = 0
    empty_retries = 0
    # Where this turn's messages begin, so drop_thinking/strip_nudges can find
    # them at the end and clean up what was fed back across the turn's tool
    # round-trips.
    turn_start = len(messages)
    try:
        while max_steps <= 0 or step < max_steps:
            trim_history(messages)
            # Last allowed step: force an answer. We keep the tool schemas in the
            # payload (dropping them would shift the cached prefix and bust nearly
            # the whole prefill) and instead append a nudge at the *tail* — only its
            # own tokens prefill, so the prefix cache stays intact.
            last = max_steps > 0 and step == max_steps - 1
            if last:
                messages.append({"role": "user", "content": STEP_LIMIT_NUDGE})

            content, thinking, tool_calls, cancelled, stats = call_ollama(messages)

            if cancelled:
                # User aborted mid-stream. Drop the partial reply (don't commit
                # it to history) and hand control back to the prompt.
                config.console.print("[yellow]cancelled[/yellow]")
                return

            if stats:
                calls += 1
                for k, v in stats.items():
                    turn_stats[k] = turn_stats.get(k, 0) + v

            # On the forced final step we treat the reply as the answer and drop any
            # tool_calls it may still carry: we're out of budget, and storing
            # unanswered tool_calls would corrupt history for the next turn.
            keep_tc = bool(tool_calls) and not last

            # Build the assistant history entry. The Ollama API expects tool_calls
            # to be included in the message when present.
            assistant_msg = {"role": "assistant", "content": content}
            if keep_tc:
                assistant_msg["tool_calls"] = tool_calls
            # Carry the reasoning on the assistant message so the next step gets it
            # back and can build on it instead of re-deriving from scratch after
            # each tool result. Appended, not edited, so the prefix cache is intact;
            # drop_thinking sheds it all once the turn produces a final answer.
            if thinking:
                assistant_msg["thinking"] = thinking
            messages.append(assistant_msg)

            if not keep_tc:
                # A wholly empty reply (no content, no tools) isn't a real finish —
                # give the model another swing, up to MAX_EMPTY_RETRIES times, by
                # appending a neutral nudge and re-entering the loop. Not on the
                # forced last step (out of budget) and not once the cap is hit.
                if not content.strip() and not last and empty_retries < MAX_EMPTY_RETRIES:
                    empty_retries += 1
                    messages.append({"role": "user", "content": EMPTY_RETRY_NUDGE})
                    continue

                # Final answer (normal early finish, or the forced last step).
                if content.strip():
                    config.console.print(Markdown(content))
                elif last:
                    config.console.print("[yellow]hit step limit; model returned no answer[/yellow]")
                else:
                    config.console.print("[yellow]model returned an empty answer[/yellow]")
                if turn_stats:
                    config.console.print(f"[dim]{fmt_stats(turn_stats, calls)}[/dim]")
                return

            # Execute each requested tool and feed results back. The finally
            # block appends stub results for any calls that never ran (e.g.
            # Ctrl-C mid-tool), so the history never carries an assistant
            # message with unanswered tool_calls into the next request.
            answered = 0
            try:
                for tc in tool_calls:
                    fn   = tc.get("function", {})
                    name = fn.get("name", "")
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        # Some model versions return arguments as a JSON string.
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}

                    config.console.print(f"[cyan]→ {name}({fmt_args(args)})[/cyan]")
                    result = dispatch(name, args)
                    config.console.print(f"[dim]{truncate(result)}[/dim]\n")

                    # Tool results use role "tool", one message per call.
                    messages.append({"role": "tool", "content": result, "name": name})
                    answered += 1
            finally:
                for tc in tool_calls[answered:]:
                    name = tc.get("function", {}).get("name", "tool")
                    messages.append({"role": "tool", "content": "[interrupted before this tool ran]", "name": name})

            # A productive tool round-trip clears the empty streak, so rare one-off
            # empties across a long turn don't accumulate toward the cap.
            empty_retries = 0
            step += 1
    finally:
        # Every exit from this function — normal return, a cancelled stream,
        # or an exception (a stalled connection, an HTTP error) propagating
        # out of call_ollama — is a turn boundary. Cleaning up here instead
        # of at each individual exit point means an abort can no longer skip
        # it and leave stale thinking/nudges sitting in committed history.
        drop_thinking(messages, turn_start)
        strip_nudges(messages, turn_start)

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Minimal local coding agent over Ollama.")
    ap.add_argument("prompt",      nargs="*",      help="initial task (optional)")
    ap.add_argument("--model",     default=None,   help=f"Ollama model tag (default: {config.MODEL}, or the saved model when resuming)")
    ap.add_argument("--yes",       action="store_true", help="auto-approve writes and commands")
    ap.add_argument("--max-steps", type=int, default=20, help="max tool calls per task (0 = unlimited)")
    ap.add_argument("--resume",    nargs="?", const="", default=None,
                     help="resume a saved session by name; bare --resume resumes the most recent")
    args = ap.parse_args()

    if args.model:
        config.MODEL = args.model
    config.AUTO_YES = args.yes

    # Persistent prompt history: importing readline upgrades input() in place,
    # so config.console.input gets line editing and up-arrow recall for free.
    if readline:
        histfile = os.path.expanduser("~/.tiny_agent_history")
        with contextlib.suppress(OSError):
            readline.read_history_file(histfile)
        readline.set_history_length(500)
        readline.parse_and_bind("set enable-bracketed-paste on")

        def save_history():
            with contextlib.suppress(OSError):
                readline.write_history_file(histfile)
        atexit.register(save_history)

    messages       = [{"role": "system", "content": config.SYSTEM}]
    first_user_msg = True
    session_name   = None

    def autosave():
        # Skip a system-prompt-only conversation so exits without real work
        # don't litter the session store.
        if len(messages) > 1:
            with contextlib.suppress(OSError):
                save_session(session_name or default_ts_name(), messages)
    atexit.register(autosave)

    if args.resume is not None:
        path = resolve_session(args.resume)
        if path:
            name = os.path.splitext(os.path.basename(path))[0]
            try:
                data = load_session(name)
                apply_session(messages, data, args.model)
                # A session saved by a run that was interrupted mid-turn (before
                # strip_nudges ran) can carry a stray nudge; drop it on restore.
                messages[:] = [m for m in messages if not _is_stray_nudge(m)]
                session_name   = name
                first_user_msg = False
                config.console.print(f"[dim]resumed session '{name}' ({len(messages)} messages)[/dim]")
            except (OSError, ValueError, KeyError, TypeError):
                config.console.print("[yellow]could not read session; starting fresh[/yellow]")
        else:
            config.console.print("[yellow]no matching session; starting fresh[/yellow]")

    initial = " ".join(args.prompt).strip()

    # Interactive start: prefill the static prefix — or, if a session was just
    # restored, the full restored history — while the user types their first
    # prompt. With an argv prompt the real request follows immediately, so a
    # warmup would just queue ahead of it for no gain.
    if not initial:
        # A snapshot, not the live list: the main loop may append the user's
        # first message to `messages` before this thread's request goes out.
        threading.Thread(target=warm_cache, args=(list(messages),), daemon=True).start()

    config.console.print(f"[bold]tiny-agent[/bold] · {config.MODEL} · {os.getcwd()}")
    config.console.print(
        "[dim]model warms up in the background; the first reply is slow if it "
        "hasn't finished. later turns reuse the KV cache. press Esc/q to "
        "cancel a reply, '/clear' to reset context, '/save', '/resume', "
        "'/sessions' to pause/switch conversations (each takes an optional "
        "name), 'exit' to quit.[/dim]\n"
    )

    while True:
        if initial:
            user    = initial
            initial = None
            config.console.print(f"[bold green]you[/bold green] {user}")
        else:
            try:
                user = read_prompt("[bold green]you[/bold green] ").strip()
            except (EOFError, KeyboardInterrupt):
                config.console.print("\nbye")
                return

        if user.lower() in ("exit", "quit"):
            return
        if user.lower() in ("/clear", "clear"):
            # Drop all conversation history but keep messages[0] (the static
            # system prompt). The system prompt + tool schemas are the cached
            # prefix, so this frees the context window without paying to
            # prefill them again. Resetting first_user_msg re-injects the cwd
            # context into the next first user message.
            del messages[1:]
            first_user_msg = True
            config.console.print("[dim]context cleared (system prompt preserved)[/dim]\n")
            continue
        if user.lower() == "/sessions":
            files = list_sessions()
            if not files:
                config.console.print("[dim]no saved sessions[/dim]\n")
                continue
            for f in files:
                name = os.path.splitext(os.path.basename(f))[0]
                try:
                    with open(f, "r", encoding="utf-8") as fh:
                        meta = json.load(fh)
                except (OSError, ValueError):
                    continue
                marker = " [bold]*[/bold]" if name == session_name else ""
                n        = len(meta.get("messages", []))
                saved_at = meta.get("saved_at", "?")
                config.console.print(f"[dim]{name}{marker} — {saved_at} · {n} messages[/dim]")
            config.console.print()
            continue
        if user.lower() == "/save" or user.lower().startswith("/save "):
            arg_name = user.split(maxsplit=1)[1].strip() if " " in user else ""
            name = arg_name or session_name or default_ts_name()
            save_session(name, messages)
            session_name = name
            config.console.print(f"[dim]saved session '{name}'[/dim]\n")
            continue
        if user.lower() == "/resume" or user.lower().startswith("/resume "):
            arg_name = user.split(maxsplit=1)[1].strip() if " " in user else ""
            path = resolve_session(arg_name)
            if not path:
                config.console.print("[yellow]no matching session; conversation unchanged[/yellow]\n")
                continue
            new_name = os.path.splitext(os.path.basename(path))[0]
            # Switching away from a named, non-empty session shouldn't lose it.
            if session_name and len(messages) > 1:
                with contextlib.suppress(OSError):
                    save_session(session_name, messages)
            try:
                data = load_session(new_name)
                apply_session(messages, data, args.model)
                messages[:] = [m for m in messages if not _is_stray_nudge(m)]
            except (OSError, ValueError, KeyError, TypeError):
                config.console.print("[yellow]could not read session; conversation unchanged[/yellow]\n")
                continue
            session_name   = new_name
            first_user_msg = False
            config.console.print(f"[dim]resumed session '{new_name}' ({len(messages)} messages)[/dim]\n")
            threading.Thread(target=warm_cache, args=(list(messages),), daemon=True).start()
            continue
        if not user:
            continue

        # cwd injected into the FIRST user message only — keeps the system
        # prompt byte-identical across projects so the prefix cache hits every
        # session. Computed here, not once at startup, so a `cd` tool call or
        # a `/clear` (which resets first_user_msg) picks up the current
        # directory instead of whatever it was when the process started.
        content        = (f"Working directory: {os.getcwd()}\n\n" + user) if first_user_msg else user
        first_user_msg = False
        messages.append({"role": "user", "content": content})

        try:
            run_turn(messages, max_steps=args.max_steps)
        except urllib.error.HTTPError as e:
            body = getattr(e, "body", "") or e.read().decode(errors="replace")
            config.console.print(f"[red]Ollama error {e.code}: {body.strip() or e.reason}[/red]")
        except urllib.error.URLError as e:
            config.console.print(f"[red]cannot reach Ollama at {config.OLLAMA_URL}: {e}[/red]")
        except (TimeoutError, OSError) as e:
            # A stalled read mid-stream (see STREAM_TIMEOUT in ollama.py) raises
            # a raw socket timeout here rather than a urllib error, since it
            # happens while iterating an already-opened response, not while
            # opening the connection. Without this the process would crash.
            config.console.print(
                f"[red]connection to Ollama stalled (no data for "
                f"{config.STREAM_TIMEOUT}s): {e}[/red]"
            )
        except KeyboardInterrupt:
            config.console.print("\n[yellow]interrupted[/yellow]")
        config.console.print()
