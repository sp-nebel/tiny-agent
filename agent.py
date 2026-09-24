import os
import re
import sys
import json
import time
import atexit
import argparse
import threading
import contextlib
import urllib.error

from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape

import config
import commands
import checkpoint
from tools import (dispatch, read_file, run_shell, normalize_call, _cap_output,
                   CommandInterrupted)
from ollama import call_ollama, warm_cache
from ui import (read_prompt, read_multiline, truncate, fmt_stats,
                take_interjections, interjections_pending, mark_turn_start,
                turn_elapsed, notify, tool_failed, tool_call_label, tool_outcome)
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

# Asked every check_every tool steps, at the tail of a *copy* of the message
# list. On CONTINUE nothing is committed: the next real call simply diverges
# from the probe at the tail, which costs the server those few tokens and no
# cache bust (on an SWA model, one checkpoint restore). On STUCK the probe
# and the reply are committed so the turn can end on the model's own
# explanation; _is_stray_nudge matches the prefix and strips the question at
# turn end, leaving the explanation as the final message. One-word verdict
# first, because small models bury a yes/no in prose.
LOOP_CHECK_PREFIX = "[checkpoint: "
LOOP_CHECK_NUDGE  = (
    LOOP_CHECK_PREFIX + "you have taken {n} tool steps on this task. Look at your "
    "recent steps. If they repeat the same actions without producing new "
    "information, reply with the single word STUCK followed by one sentence on "
    "what you tried. Otherwise reply with the single word CONTINUE and nothing else.]"
)

# Marks a message the user typed while the model was mid-task (see
# ui.read_interjection), so the model reads it as a new instruction arriving
# during the work rather than as a fresh, unrelated task. Same shape as the
# nudges above — a bracketed lead-in on an appended user message.
INTERJECTION_PREFIX = "[user message sent mid-task] "

# Prefixes the note the user types after stopping a reply with Tab. Unlike a
# queued interjection this one arrives *instead of* the step the model was
# taking, so it says so, and in the imperative, because small models take
# hints badly. A permanent user message, not a nudge: later turns must still
# see why the model changed course.
STOP_NOTE_PREFIX = ("[the user interrupted you with this note; follow it, "
                    "then continue the task]\n")
STOP_NOTE_DEFAULT = "Stop and reconsider what you were about to do."

# Added to a tool result when the model has made the identical call and got
# the identical result REPEAT_CALL_LIMIT times in a row this turn — the doom
# loop a small model falls into (re-reading one range, re-running one grep).
# Both call and result must repeat: re-running tests between edits is the
# normal loop, and its output changes. Part of the result, so it lands at the
# tail like any tool output and needs no extra model call, unlike the probe.
REPEAT_CALL_NOTE = ("[you have made this exact {name} call {n} times in a row "
                    "and got the same result each time. Repeating it will not "
                    "give new information: use what you have, try a different "
                    "approach, or give your answer.]")

# Result for a run_cmd the user stopped with Ctrl-C, after whatever it printed.
USER_STOPPED_CMD = ("[the user stopped this command before it finished; it may "
                    "have partly run]")


def deliver_interjections(messages):
    """Append whatever the user typed during the last step.

    Appended at the tail, like the nudges, so only its own tokens prefill and
    the cached prefix is untouched. Called at a step boundary and nowhere
    else: an assistant message carrying tool_calls must be followed straight
    away by one tool result per call, so a user message slipped in between
    would corrupt history for every later request.
    """
    queued = take_interjections()
    for text in queued:
        messages.append({"role": "user", "content": INTERJECTION_PREFIX + text})
        # The echoed `interject` line sits wherever the user happened to type
        # it; print it again here, where it actually enters the conversation —
        # after the tool results of the step it was typed during.
        config.console.print(f"[bold green]you[/bold green] {escape(text)}")


def _stop_and_steer(messages, content, thinking):
    """Commit a Tab-stopped partial reply and read the user's note.

    The user stopped the stream to steer, not to abort: the turn stays live
    and nothing gathered so far is touched. The partial reply is committed as
    an assistant message so the model keeps the reasoning it had (and the
    note can refer to it), minus any tool_calls it had already emitted — a
    call the user interrupted is exactly the one they didn't want run, and
    storing it unanswered would break the tool_calls/results pairing.

    Nothing is committed when the stream stopped before the first token:
    history then ends user(prompt), user(note), which chat templates render
    in sequence (strip_nudges can leave the same shape already). Everything
    is appended at the tail, so the KV prefix cache is intact. At turn end
    drop_thinking strips the partial's thinking, and if that leaves an empty
    shell, strip_nudges removes it; the note itself is kept.
    """
    if content or thinking:
        partial = {"role": "assistant", "content": content}
        if thinking:
            partial["thinking"] = thinking
        messages.append(partial)
    # The live region was transient, so the partial text just vanished from
    # the screen; show it again — it is what the user is steering against.
    if content.strip():
        config.console.print(Markdown(content))
    config.console.print("[yellow]stopped — the reply so far stays in context and "
                         "any tool call it started is dropped. Type a note for "
                         "the model (empty = reconsider)[/yellow]")
    try:
        note = read_prompt("[bold green]steer[/bold green] ").strip()
    except EOFError:
        note = ""
    messages.append({"role": "user", "content": STOP_NOTE_PREFIX + (note or STOP_NOTE_DEFAULT)})


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


# Measured prefill throughput (tok/s), refined over the session; None until
# the first meaningful sample, when PREFILL_TPS_FALLBACK applies instead.
# Sizes the post-trim stall timeout — see _post_trim_timeout.
_prefill_tps = None


def _note_prefill_rate(stats):
    """Fold one call's prefill timing into the running rate estimate.

    Small prefills are skipped: with the prefix cache warm a call prefills
    only a few hundred tokens, and at that size fixed per-request overhead
    dominates the timing, which would drag the estimate far below the real
    throughput a long re-prefill achieves."""
    global _prefill_tps
    count    = stats.get("prompt_eval_count", 0)
    duration = stats.get("prompt_eval_duration", 0)   # nanoseconds
    if count < 256 or duration <= 0:
        return
    rate = count / (duration / 1e9)
    _prefill_tps = rate if _prefill_tps is None else (_prefill_tps + rate) / 2


def _post_trim_timeout(messages):
    """Stall timeout for the one call right after a trim pass.

    That call re-prefills the whole trimmed prompt in silence before its
    first byte, so the ordinary STREAM_TIMEOUT stall detector would kill a
    legitimate multi-minute CPU prefill (a fixed 900s constant once did,
    at 92% done). Sized from what the prefill will actually cost: the
    post-trim token estimate over the measured rate, with a safety factor
    for the estimate erring low and the rate degrading as the window fills.
    """
    rate = _prefill_tps or config.PREFILL_TPS_FALLBACK
    return max(config.STREAM_TIMEOUT,
               int(_total_tokens(messages) / rate * config.POST_TRIM_TIMEOUT_FACTOR))


def _stub_tool_outputs(messages, keep_last, start=0):
    """Collapse oversized tool outputs (from `start` on) to one-line stubs,
    keeping the `keep_last` most recent verbatim. Shared by trim_history's
    soft pass and run_turn's turn-end cleanup so the marker format and skip
    rules can't drift between them. Idempotent: a stub starts with
    TRIM_PREFIX and is skipped on every later pass. Returns the number of
    outputs collapsed, so callers can tell whether history (and therefore
    the KV prefix cache) was actually edited."""
    tool_idxs = [i for i, m in enumerate(messages)
                 if i >= start and m.get("role") == "tool"]
    old = tool_idxs[:-keep_last] if keep_last else tool_idxs
    targets = [i for i in old
               if not (messages[i].get("content") or "").startswith(config.TRIM_PREFIX)
               and len(messages[i].get("content") or "") > config.TRIM_MIN_CHARS]
    for i in targets:
        m       = messages[i]
        content = m.get("content", "")
        nlines  = content.count("\n") + 1
        m["content"] = (f"{config.TRIM_PREFIX}{m.get('name', 'tool')} — "
                        f"was {nlines} lines, {len(content)} chars]")
    return len(targets)


def trim_history(messages):
    """Shed context in place when the conversation approaches the window.

    Untouched history is free: it sits in the KV prefix cache and is never
    re-prefilled. Editing a message, by contrast, invalidates the cache from
    the edit point on — and on an SWA model (gemma) the next call then
    reprocesses the whole prompt from token 0, so the real cost of a trim
    pass is the *post-trim prompt size*. Two consequences shape this design:
    trimming is lazy (do nothing until the estimate crosses TRIM_AT_TOKENS,
    then shed in one pass — one cache bust, not one per step), and a pass
    sheds aggressively so the re-prefill it triggers is as small as possible.

    A pass sheds in priority order: first collapse every tool result the model
    has already replied to (its kept thinking is the distilled record of them,
    and raw reads are the biggest single contributor), then — only if that
    alone doesn't get back under the trigger — drop `thinking` from all but
    the last TRIM_KEEP_THINKING assistant messages that carry it. thinking
    only exists on the live turn (drop_thinking clears it at every turn
    boundary), so that fallback sheds the current turn's older reasoning.

    The token estimate uses the ~4 chars/token heuristic; it only needs to be
    right to within the headroom left below NUM_CTX.

    Idempotent: every collapsed/truncated message starts with TRIM_PREFIX and
    is skipped on later passes, and a dropped thinking field is simply gone.

    After the soft pass, a backstop hard-truncates any remaining oversized
    message outside a protected recent tail. Stub-collapsing alone can't help
    when the bloat is in long assistant/user messages, or when the keep-window
    itself is what's oversized — without this, those cases silently overflow
    NUM_CTX.

    Returns True when this pass edited history — i.e. the KV prefix cache was
    just busted and the next call will silently re-prefill the whole trimmed
    prompt, so run_turn gives it a prefill-sized stall timeout and one retry.
    Deliberately coarse on the hard-truncate path (reaching it almost always
    edits something, and a false True merely stretches one call's timeout);
    a False must be reliable, since that call keeps the short timeout.
    """
    if _total_tokens(messages) < config.TRIM_AT_TOKENS:
        return False
    # A tool result is disposable as soon as the model has replied to it: the
    # reply's thinking (kept for the whole turn) already carries the distilled
    # facts, while the raw output — often a capped-but-huge file read — is the
    # bulk of the window. So stub every result the model has processed, and
    # keep only the trailing ones after the last assistant message: those it
    # has not seen yet and is about to act on — stub one of them and the model
    # acts on a placeholder.
    last_asst = next((i for i in range(len(messages) - 1, -1, -1)
                      if messages[i].get("role") == "assistant"), 0)
    unseen  = sum(1 for m in messages[last_asst:] if m.get("role") == "tool")
    stubbed = _stub_tool_outputs(messages, unseen)
    if stubbed:
        config.console.print(f"[dim]compacted {stubbed} old tool output"
                      f"{'s' if stubbed != 1 else ''}[/dim]")

    # Thinking is the model's distilled record of everything stubbed above, so
    # it survives the pass whenever stubbing alone gets the estimate back under
    # the trigger. When it doesn't — a marathon turn whose thinking IS the bulk
    # — shed all but the last TRIM_KEEP_THINKING fields too. Without this
    # escape valve the estimate would sit above TRIM_AT_TOKENS forever and
    # every later step would stub its one newly-processed output: a cache bust
    # and a full re-prefill per step, the exact pattern trimming exists to
    # avoid (the only other relief is the ~90% backstop below, which drops old
    # thinking far more brutally).
    dropped = 0
    if _total_tokens(messages) >= config.TRIM_AT_TOKENS:
        thinking_idxs = [i for i, m in enumerate(messages)
                         if m.get("role") == "assistant" and m.get("thinking")]
        old_thinking = (thinking_idxs[:-config.TRIM_KEEP_THINKING]
                        if config.TRIM_KEEP_THINKING else thinking_idxs)
        for i in old_thinking:
            messages[i].pop("thinking", None)
        dropped = len(old_thinking)
        if dropped:
            config.console.print("[dim]shed older thinking to fit the window[/dim]")
    edited = bool(stubbed or dropped)

    if _total_tokens(messages) < config.HARD_TRUNCATE_AT_TOKENS:
        return edited
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
            return True

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
            return True

    # Step 3: still over budget — most likely the protected tail itself
    # (short contents, or the one thinking field we deliberately kept) is
    # simply too large. Warn rather than silently exceed NUM_CTX.
    config.console.print(
        "[yellow]warning: context still exceeds the safety margin after "
        "trimming; the model may lose the system prompt[/yellow]"
    )
    return True


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
    """A step-limit/empty-retry/loop-check nudge, or the empty assistant reply that
    prompted one — the pieces strip_nudges removes once a turn ends. Exposed
    separately so a restored session (main()'s apply_session call sites) can
    run the same filter: a turn interrupted before strip_nudges ran (a
    Ctrl-C or a dropped connection propagating out of run_turn) can persist
    one of these into an autosaved session, and it would otherwise sit there
    forever.
    """
    if m.get("role") == "user" and (m.get("content") in (STEP_LIMIT_NUDGE, EMPTY_RETRY_NUDGE)
                                     or (m.get("content") or "").startswith(LOOP_CHECK_PREFIX)):
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


def _says_stuck(content):
    # The probe asks for the verdict word first; tolerate the markdown or
    # punctuation a model may wrap it in, but don't search the whole reply —
    # "CONTINUE, I'm not stuck" must not read as STUCK.
    return (content or "").strip().lstrip("*#-> ").upper().startswith("STUCK")


def _add_stats(turn_stats, stats):
    for k, v in stats.items():
        turn_stats[k] = turn_stats.get(k, 0) + v
    _note_prefill_rate(stats)


def _print_answer(content):
    """A turn's final answer: rendered Markdown on the terminal, or the raw
    text on stdout when that is a pipe (everything else then goes to
    stderr), so `… | local_agent.py "review" > review.md` keeps just the
    answer."""
    if config.ANSWER_TO_STDOUT:
        sys.stdout.write(content.strip() + "\n")
        sys.stdout.flush()
    else:
        config.console.print(Markdown(content))


def _show_result(name, label, early, result, note=""):
    """Print a tool call's outcome: one line normally, the body too when it
    failed or /details is on. Display only — the model gets the full result
    either way. `note` is text agent.py appends to the result; it is shown
    with the body but not parsed for the outcome. Rich reads any "[word …]"
    as a markup tag and drops an unknown one silently, so everything here
    is escaped."""
    outcome = tool_outcome(name, result)
    failed  = tool_failed(name, result)
    result += note
    if early:
        # A failed command's body is printed below and ends with its exit.
        if outcome and name == "run_cmd" and not failed:
            config.console.print(f"[dim]  {escape(outcome)}[/dim]")
    else:
        tail = f" [dim]({escape(outcome)})[/dim]" if outcome else ""
        config.console.print(f"[cyan]→ {escape(label)}[/cyan]{tail}")
    if name in ("edit_file", "append_file") and not config.SHOW_DETAILS:
        # Their result is itself one short line, plus any syntax warning.
        config.console.print(f"[{'yellow' if failed else 'dim'}]{escape(result)}[/]")
    elif config.SHOW_DETAILS or failed:
        config.console.print(f"[dim]{escape(truncate(result))}[/dim]")
    if config.SHOW_DETAILS or failed or early:
        config.console.print()


def run_turn(messages, max_steps=20, check_every=20):
    # max_steps <= 0 means unlimited: no forced final step, no nudge.
    # check_every > 0 asks the model every that many tool steps whether it is
    # looping (see LOOP_CHECK_NUDGE); a STUCK verdict ends the turn on its
    # explanation, anything else lets it keep working with a fresh budget.
    # Under the default max_steps=20 a 20-step check can never fire (the
    # counter reaches 20 only as the loop exits); it is for max_steps=0 or a
    # higher cap, where nothing else would stop a model going in circles.
    # A turn fans out into several call_ollama requests (one per tool
    # round-trip); sum their stats and print one summary line when the turn
    # finishes, rather than a line per call.
    turn_stats = {}
    mark_turn_start()
    # (tool, canonical args) → (last result, times in a row it came back).
    repeats = {}
    calls = 0
    step = 0
    since_check = 0
    empty_retries = 0
    # Where this turn's messages begin, so drop_thinking/strip_nudges can find
    # them at the end and clean up what was fed back across the turn's tool
    # round-trips.
    turn_start = len(messages)
    try:
        while max_steps <= 0 or step < max_steps:
            deliver_interjections(messages)
            trimmed = trim_history(messages)
            # Last allowed step: force an answer. We keep the tool schemas in the
            # payload (dropping them would shift the cached prefix and bust nearly
            # the whole prefill) and instead append a nudge at the *tail* — only its
            # own tokens prefill, so the prefix cache stays intact.
            last = max_steps > 0 and step == max_steps - 1

            # Loop check, after the trim so the probe never goes out over the
            # threshold, and never on the forced last step — the nudge below
            # already ends the turn there.
            if check_every > 0 and since_check >= check_every and not last:
                since_check = 0
                config.console.print(f"[dim]loop check after {step} steps…[/dim]")
                probe = messages + [{"role": "user",
                                     "content": LOOP_CHECK_NUDGE.format(n=step)}]
                # If a trim just busted the cache, the probe is the call that
                # pays the silent re-prefill, so it takes the long timeout;
                # the main call below then reuses that prefix normally.
                timeout = _post_trim_timeout(messages) if trimmed else None
                content, thinking, tool_calls, cancelled, interrupted, stats = call_ollama(
                    probe, timeout=timeout, retry_stall=trimmed)
                trimmed = False
                if cancelled:
                    config.console.print("[yellow]cancelled[/yellow]")
                    return
                if stats:
                    calls += 1
                    _add_stats(turn_stats, stats)
                if interrupted:
                    # A partial answer to a discarded probe means nothing to
                    # the model; commit only the user's note.
                    _stop_and_steer(messages, "", "")
                    continue
                if _says_stuck(content) and not tool_calls:
                    # Commit the exchange so the turn ends on the model's own
                    # account of what it tried; the user gets control back
                    # with the whole turn still in context to steer from.
                    messages.append(probe[-1])
                    messages.append({"role": "assistant", "content": content})
                    config.console.print("[yellow]model reports it is stuck after "
                                         f"{step} steps — handing back to you[/yellow]")
                    _print_answer(content)
                    return
                # CONTINUE, a tool call, or anything else: the probe is
                # discarded and work goes on with a fresh budget.

            # A Tab-steer on the last step re-enters here without consuming
            # the step; the nudge already sent stays in force, so don't stack
            # a second copy after the user's note.
            if last and STEP_LIMIT_NUDGE not in (m.get("content") for m in messages[turn_start:]):
                messages.append({"role": "user", "content": STEP_LIMIT_NUDGE})

            # A trim pass just busted the prefix cache, so this call re-prefills
            # the whole trimmed prompt in silence — expected to take minutes on
            # CPU. Give exactly this call a stall timeout sized to that prefill
            # plus one retry (the server keeps its prefill progress, so a retry
            # after a genuine stall resumes nearly free); the next loop
            # iteration trims nothing (idempotent) and reverts to normal.
            timeout = _post_trim_timeout(messages) if trimmed else None
            content, thinking, tool_calls, cancelled, interrupted, stats = call_ollama(
                messages, timeout=timeout, retry_stall=trimmed)

            if cancelled:
                # User aborted mid-stream. Drop the partial reply (don't commit
                # it to history) and hand control back to the prompt.
                config.console.print("[yellow]cancelled[/yellow]")
                return

            if interrupted:
                # Steering shouldn't cost budget: no step consumed. Any
                # queued interjections follow the note at the next
                # iteration's deliver_interjections.
                _stop_and_steer(messages, content, thinking)
                continue

            if stats:
                calls += 1
                _add_stats(turn_stats, stats)

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
                    # A queued interjection is itself a continuation prompt, and
                    # the next iteration delivers it — a nudge in front of it
                    # would only be a second, vaguer user message in a row.
                    if not interjections_pending():
                        messages.append({"role": "user", "content": EMPTY_RETRY_NUDGE})
                    continue

                # Final answer (normal early finish, or the forced last step).
                if content.strip():
                    _print_answer(content)
                elif last:
                    config.console.print("[yellow]hit step limit; model returned no answer[/yellow]")
                else:
                    config.console.print("[yellow]model returned an empty answer[/yellow]")
                return

            # The model addressed the user before calling its tools. That text
            # streamed only into the transient Live region, which is wiped when
            # the stream ends, so without re-rendering it here the update would
            # vanish the moment the tool lines print — the same disposal
            # `thinking` gets. Thinking is scratch work and earns that; a
            # mid-turn update is written *for the user*, so it stays on screen,
            # above the tool calls it explains. Display only: the text is
            # already committed to assistant_msg either way, so history — and
            # the prefix cache — are untouched.
            if content.strip():
                config.console.print(Markdown(content))

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

                    # The model's own name for the tool may be an alias
                    # ("bash") that dispatch maps; show and record the real one.
                    real, fixed = normalize_call(name, args)
                    shown = real or name
                    # Rich reads any "[word …]" as a markup tag and drops an
                    # unknown one silently, so tool args and results — full of
                    # bracketed metadata like "[lines 1-100 of 543]" — must be
                    # escaped or the user sees them vanish.
                    # Tools that print (a diff, a confirmation prompt) or can
                    # run long get their call line up front; the quick
                    # read-only ones get one line afterwards, with the outcome.
                    label = tool_call_label(shown, fixed)
                    early = real is None or real in ("run_cmd", "edit_file", "append_file")
                    if early:
                        config.console.print(f"[cyan]→ {escape(label)}[/cyan]")
                    try:
                        result = dispatch(name, args)
                    except CommandInterrupted as e:
                        # The command ran, partly: it may have changed files,
                        # so "interrupted before this tool ran" would be a lie
                        # the model acts on. Answer this call with what it
                        # printed; the finally stubs any calls after it.
                        stopped = ((e.output + "\n") if e.output else "") + USER_STOPPED_CMD
                        messages.append({"role": "tool", "content": stopped, "name": shown})
                        answered += 1
                        raise
                    key = (shown, json.dumps(args, sort_keys=True, default=str))
                    last, n = repeats.get(key, (None, 0))
                    n = n + 1 if result == last else 1
                    repeats[key] = (result, n)
                    # Judged before the note is added: it would push the
                    # "[exit N]" and end-of-file markers off the end, and a
                    # failed command would show as "exit 0".
                    note = ("\n" + REPEAT_CALL_NOTE.format(name=shown, n=n)
                            if n >= config.REPEAT_CALL_LIMIT else "")
                    _show_result(shown, label, early, result, note)
                    result += note

                    # Tool results use role "tool", one message per call.
                    messages.append({"role": "tool", "content": result, "name": shown})
                    answered += 1
            finally:
                for tc in tool_calls[answered:]:
                    name = tc.get("function", {}).get("name", "tool")
                    messages.append({"role": "tool", "content": "[interrupted before this tool ran]", "name": name})

            # A productive tool round-trip clears the empty streak, so rare one-off
            # empties across a long turn don't accumulate toward the cap.
            empty_retries = 0
            step += 1
            since_check += 1
    finally:
        # Every exit from this function — normal return, a cancelled stream,
        # or an exception (a stalled connection, an HTTP error) propagating
        # out of call_ollama — is a turn boundary. Cleaning up here instead
        # of at each individual exit point means an abort can no longer skip
        # it and leave stale thinking/nudges sitting in committed history.
        drop_thinking(messages, turn_start)
        strip_nudges(messages, turn_start)
        # Also collapse the finished turn's oversized tool outputs, after
        # strip_nudges (it rebuilds the list, shifting indices). On a
        # thinking model drop_thinking above already busted the cache back
        # to turn_start, so these edits cost no extra re-prefill — they
        # shrink the one the next turn pays anyway. Completed turns stay
        # lean, so long sessions reach each new task with a nearly-empty
        # window instead of relying on a mid-turn trim later; the most
        # recent results stay verbatim so an immediate follow-up ("now edit
        # what you just read") still sees them. On a non-thinking model this
        # is the sole turn-boundary edit — a bust bounded to the turn's own
        # span, accepted for the same window-pressure reason.
        stubbed = _stub_tool_outputs(messages, config.KEEP_FULL_TOOL_RESULTS,
                                     start=turn_start)
        if stubbed:
            config.console.print(f"[dim]compacted {stubbed} tool output"
                          f"{'s' if stubbed != 1 else ''} from this turn[/dim]")
        # After the cleanup, so ctx% is what the next turn starts from.
        if turn_stats:
            pct = 100 * _total_tokens(messages) // config.NUM_CTX
            config.console.print(f"[dim]{fmt_stats(turn_stats, calls, turn_elapsed(), pct)}[/dim]")
        notify("turn finished")

# --------------------------------------------------------------------------- #
# First-message context
# --------------------------------------------------------------------------- #

# Leads an instructions file's text in the first user message. Imperative,
# like the other bracketed lead-ins: a small model treats unlabelled text as
# something to comment on, not rules to work by.
INSTRUCTIONS_HEAD = "[instructions from {path}; follow them while you work]"


def _git_root(start):
    """The nearest enclosing directory with a .git entry, or None. A plain
    walk instead of `git rev-parse`: this runs before every first message
    and must not depend on git being installed."""
    d = os.path.abspath(start)
    while True:
        if os.path.exists(os.path.join(d, ".git")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def find_instructions(start):
    """The project instructions file for `start`: the first of
    config.INSTRUCTION_FILES found walking up from `start` to the git root.
    Outside a repo only `start` itself is looked at — walking up to / would
    pick up whatever happens to sit in a parent directory."""
    d    = os.path.abspath(start)
    root = _git_root(d)
    while True:
        for name in config.INSTRUCTION_FILES:
            path = os.path.join(d, name)
            if os.path.isfile(path):
                return path
        if root is None or d == root:
            return None
        d = os.path.dirname(d)


def _instructions_block(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read().strip()
    except OSError:
        return None
    if not text:
        return None
    cap = config.INSTRUCTIONS_MAX_CHARS
    if len(text) > cap:
        text = (text[:cap] + f"\n[… cut at {cap} chars; read_file {path} "
                f"with start=1 for the rest]")
    return INSTRUCTIONS_HEAD.format(path=path) + "\n" + text


def context_header():
    """What the first user message of a conversation opens with: the cwd,
    whether it is a git repo, platform and date, then the global and the
    project instructions files, if any.

    Lives in the first user message, never in SYSTEM, so the cached static
    prefix stays byte-identical across projects and days. Read again at
    each /clear, so an edited AGENTS.md takes effect from the next
    conversation; a running one keeps what it was sent (rewriting the
    first message would bust the whole cache).
    """
    cwd  = os.getcwd()
    root = _git_root(cwd)
    env  = (f"Working directory: {cwd}\n"
            f"Git repo: {('yes, root ' + root) if root else 'no'} · "
            f"Platform: {sys.platform} · Date: {time.strftime('%Y-%m-%d')}")
    blocks = [env]
    project = find_instructions(cwd)
    for path in (config.GLOBAL_INSTRUCTIONS, project):
        block = path and _instructions_block(path)
        if block:
            blocks.append(block)
            config.console.print(f"[dim]instructions from {escape(path)}[/dim]")
    return "\n\n".join(blocks)

# --------------------------------------------------------------------------- #
# Prompt expansions
# --------------------------------------------------------------------------- #

# Leads the output of a `!cmd` the user ran at the prompt, when it rides along
# with their next message. Bracketed, like every other piece of metadata the
# model is shown, and it says who ran the command — otherwise a small model
# reads a pasted-looking command output as something it did itself.
SHELL_BLOCK_HEAD = "[the user ran this shell command: {cmd} — {status}]"


def run_shell_escape(cmd):
    """Run a `!cmd` / `!!cmd` typed at the prompt, show its output, and
    return it as a block for the model. No confirmation: the user typed it.

    Runs through the same run_shell as the model's run_cmd, so the output
    the model gets is capped the same way. A subshell, so `!cd` changes
    nothing that outlives it.
    """
    try:
        output, code, timed_out = run_shell(cmd)
    except CommandInterrupted as e:
        # Ctrl-C at the prompt used to escape main() with a traceback. The
        # user stopped their own command: keep what it printed, like a
        # timeout, rather than losing it.
        output, timed_out, code = e.output, False, None
    if timed_out:
        status = f"timed out after {config.CMD_TIMEOUT}s"
    elif code is None:
        status = "stopped with Ctrl-C"
    else:
        status = f"exit {code}"
    if output:
        config.console.print(f"[dim]{escape(output)}[/dim]")
    config.console.print(f"[dim]{escape(status)}[/dim]")
    return SHELL_BLOCK_HEAD.format(cmd=cmd, status=status) + "\n" + (output or "[no output]")

# Leads text piped in on stdin, which rides ahead of the prompt in the first
# message like `!cmd` output does. Says where it came from, so the model
# doesn't take a diff for something it produced.
PIPED_HEAD = "[input piped to the agent by the user]"

# An `@path` token: at the start of the prompt or after whitespace, so an
# email address or a decorator in pasted code is not one.
FILE_REF_RE     = re.compile(r"(?<!\S)@(\S+)")
# Punctuation that ends a sentence around a path ("look at @agent.py.") rather
# than belonging to it; stripped only when the token as typed isn't a file.
FILE_REF_TRAIL  = ".,;:!?)]}'\""
FILE_BLOCK_HEAD = "[contents of {path}, attached by the user]"
# `@path#10-40` (or `#10`) attaches just those lines.
FILE_RANGE_RE   = re.compile(r"#(\d+)(?:-(\d+))?$")


def _resolve_file_ref(token):
    """(path as typed, start, end) for an `@token` naming an existing file,
    or None. Tried as typed first, then without trailing sentence
    punctuation; each with an optional #A-B line range."""
    for cand in (token, token.rstrip(FILE_REF_TRAIL)):
        path, start, end = cand, 1, None
        rng = FILE_RANGE_RE.search(cand)
        if rng and not os.path.isfile(os.path.expanduser(cand)):
            path  = cand[:rng.start()]
            start = int(rng.group(1))
            end   = int(rng.group(2) or rng.group(1))
        if path and os.path.isfile(os.path.expanduser(path)):
            return path, start, end
    return None


def expand_file_refs(text):
    """Append the contents of every `@path` in `text` that names an existing
    file, each once, after the text itself.

    The contents come from read_file, so an attachment is exactly what the
    model's own first read of the file would return: numbered lines, the same
    100-line and size caps, and for a longer file the same imperative telling
    it how to read on. Attaching a big file therefore costs one read_file
    call's worth of prefill, not the whole file. Tokens that aren't files
    (an @mention, a path that doesn't exist) are left alone, and the @token
    stays in the text so the request still reads naturally.
    """
    blocks = []
    seen   = set()
    for m in FILE_REF_RE.finditer(text):
        ref = _resolve_file_ref(m.group(1))
        if ref is None:
            continue
        path, start, end = ref
        full = os.path.expanduser(path)
        if (full, start, end) in seen:
            continue
        seen.add((full, start, end))
        shown = path if end is None else f"{path} lines {start}-{end}"
        blocks.append(FILE_BLOCK_HEAD.format(path=shown) + "\n" + read_file(full, start, end))
        config.console.print(f"[dim]attached {escape(shown)}[/dim]")
    return "\n\n".join([text] + blocks)

# --------------------------------------------------------------------------- #
# Undo
# --------------------------------------------------------------------------- #

def undo_last_turn(messages, turns, announce=True):
    """Rewind the most recent turn: restore the files it changed from its
    git snapshot, cut its messages off the tail, and return its record (so
    main can put the prompt back in the input line and restore first_user_msg);
    None if there is nothing to undo.

    Cache impact: the cut is at the tail, so the retained history is exactly
    a prefix of what was last sent. A full-attention model keeps that prefix
    cached; on an SWA model (gemma) rolling back is the same kind of edit as
    a trim — at most one re-prefill of what remains (possibly just a
    checkpoint restore; unmeasured) — paid once, at a moment the user chose,
    and hidden by the warmup main starts right after.
    """
    if not turns:
        config.console.print("[dim]nothing to undo[/dim]\n")
        return None
    rec = turns.pop()

    if rec["snap"] is None:
        config.console.print("[yellow]not a git repo — conversation rewound, "
                             "files untouched[/yellow]")
    else:
        changes, head_moved = checkpoint.restore(rec["snap"])
        if changes is None:
            config.console.print("[yellow]git restore failed — conversation "
                                 "rewound, files untouched[/yellow]")
        for status, path in changes or []:
            verb = "removed" if status == "A" else "restored"
            config.console.print(f"[dim]{verb} {escape(path)}[/dim]")
        if head_moved:
            config.console.print("[yellow]HEAD moved during that turn; its "
                                 "commits were kept (files restored only)[/yellow]")

    del messages[rec["msg_index"]:]
    with contextlib.suppress(OSError):
        os.chdir(rec["cwd"])
    if not announce:
        return rec
    if "\n" in rec["prompt"]:
        # readline edits one line; a multi-line prompt inserted into it
        # garbles the display, so show it instead of pre-filling it.
        config.console.print(f"[dim]undid the last turn ({len(messages)} messages "
                             f"left); its prompt was:[/dim]\n{escape(rec['prompt'])}\n")
    else:
        config.console.print(f"[dim]undid the last turn ({len(messages)} messages "
                             f"left); its prompt is back in the input line[/dim]\n")
    return rec

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Minimal local coding agent over Ollama.")
    ap.add_argument("prompt",      nargs="*",      help="initial task (optional)")
    ap.add_argument("--model",     default=None,   help=f"Ollama model tag (default: {config.MODEL}, or the saved model when resuming)")
    ap.add_argument("--yes",       action="store_true", help="auto-approve writes and commands")
    ap.add_argument("--max-steps", type=int, default=20, help="max tool calls per task (0 = unlimited)")
    ap.add_argument("--check-every", type=int, default=20,
                    help="every N tool round-trips, ask the model whether it is looping and "
                         "stop if it says so (0 = never; inert under the default --max-steps)")
    # --resume takes a required name, and "the most recent" is a separate
    # flag with no value: an optional value next to the nargs="*" prompt
    # made `--resume "fix the test"` read the prompt as a session name.
    ap.add_argument("--resume",    metavar="NAME", default=None,
                    help="resume the saved session NAME")
    ap.add_argument("-c", "--continue", dest="cont", action="store_true",
                    help="resume the most recent saved session")
    args = ap.parse_args()
    if args.cont and args.resume is None:
        args.resume = ""

    if args.model:
        config.MODEL = args.model
    config.AUTO_YES = args.yes

    # stdin that isn't a terminal is input, not a keyboard: `git diff |
    # local_agent.py "review this"` runs one turn on it and exits. Nobody
    # can answer a confirmation, so writes and commands need --yes (see
    # tools.confirm). With stdout piped too, only the answer goes there.
    piped      = not sys.stdin.isatty()
    piped_text = ""
    if piped:
        piped_text         = sys.stdin.read()
        config.INTERACTIVE = False
        if not sys.stdout.isatty():
            config.ANSWER_TO_STDOUT = True
            config.console          = Console(stderr=True)
        if not args.prompt and not piped_text.strip():
            config.console.print("[red]nothing to do: no prompt and nothing on stdin[/red]")
            sys.exit(2)

    def custom_commands():
        return commands.load_custom_commands(_git_root(os.getcwd()) or os.getcwd())

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
        commands.install_completer(readline, custom_commands)

    messages       = [{"role": "system", "content": config.SYSTEM}]
    first_user_msg = True
    session_name   = None
    # The conversation's first prompt, shown by /sessions in place of a bare
    # timestamp. Not an LLM-written title: that would be an extra call, and
    # with Ollama's single cache slot it would evict the conversation.
    session_title  = ""
    # One record per turn, newest last, for /undo: where the turn's messages
    # start, its raw prompt, the cwd and first_user_msg it began with, and a
    # git snapshot of the working tree (None outside a repo). Indices into
    # `messages`, so anything that replaces the history (/clear, /resume)
    # empties it.
    turns          = []
    # Text pre-filled into the next prompt line — the prompt of an undone
    # turn, so it can be edited and resent.
    seed           = ""
    # Output of `!cmd`s run since the last turn, sent ahead of the next
    # prompt as part of the same user message — so it costs nothing until
    # there is a question about it, and never leaves two user messages in a
    # row.
    pending_shell  = []

    def autosave():
        # Skip a system-prompt-only conversation so exits without real work
        # don't litter the session store — and a piped one-shot run, which
        # a script may make hundreds of times.
        if len(messages) > 1 and not piped:
            with contextlib.suppress(OSError):
                save_session(session_name or default_ts_name(), messages, session_title)
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
                session_title  = data.get("title", "")
                first_user_msg = False
                config.console.print(f"[dim]resumed session '{escape(name)}' ({len(messages)} messages)[/dim]")
            except (OSError, ValueError, KeyError, TypeError):
                config.console.print("[yellow]could not read session; starting fresh[/yellow]")
        else:
            config.console.print("[yellow]no matching session; starting fresh[/yellow]")

    initial = " ".join(args.prompt).strip()
    # Piped text is sent as it is, never parsed as `!cmd` or `/command`.
    literal_first = False
    if piped and piped_text.strip():
        if initial:
            pending_shell.append(PIPED_HEAD + "\n" + _cap_output(piped_text.strip(),
                                                                 config.MAX_PIPED_CHARS))
        else:
            initial       = piped_text.strip()
            literal_first = True

    # Interactive start: prefill the static prefix — or, if a session was just
    # restored, the full restored history — while the user types their first
    # prompt. With an argv prompt the real request follows immediately, so a
    # warmup would just queue ahead of it for no gain.
    if not initial:
        # A snapshot, not the live list: the main loop may append the user's
        # first message to `messages` before this thread's request goes out.
        threading.Thread(target=warm_cache, args=(list(messages),), daemon=True).start()

    config.console.print(f"[bold]tiny-agent[/bold] · {escape(config.MODEL)} · {escape(os.getcwd())}")
    if not piped:
        config.console.print(
            "[dim]model warms up in the background; the first reply is slow if it "
            "hasn't finished. later turns reuse the KV cache. while a reply "
            "streams, just start typing to queue a message for the model; Tab "
            "stops the reply now so you can steer it; Esc cancels the reply. "
            "'!cmd' runs a shell command and sends its output with your next "
            "message, '@path' attaches a file, '/undo' takes back the last turn, "
            "'/help' lists every command, 'exit' quits.[/dim]\n"
        )

    while True:
        if initial:
            user    = initial
            initial = None
            config.console.print(f"[bold green]you[/bold green] {escape(user)}")
        else:
            try:
                user = read_multiline("[bold green]you[/bold green] ", seed).strip()
                seed = ""
            except (EOFError, KeyboardInterrupt):
                config.console.print("\nbye")
                return

        # The raw text of the prompt as typed, for /undo and the session
        # title; differs from `user` when a custom command expanded it.
        raw_prompt = user
        # Text written in /editor is a prompt, whatever it starts with: a
        # first line of "!rm …" or "/save x" must not run as a command.
        literal, literal_first = literal_first, False
        if not literal and user.lower() == "/editor":
            text = commands.edit_in_editor()
            if text is None or not text.strip():
                if text is not None:
                    config.console.print("[dim]empty prompt; nothing sent[/dim]\n")
                continue
            user = raw_prompt = text.strip()
            literal = True
            config.console.print(f"[bold green]you[/bold green] {escape(user)}")

        if not literal:
            if user.lower() in ("exit", "quit"):
                return
            if user.startswith("!"):
                local = user.startswith("!!")
                cmd   = user[2 if local else 1:].strip()
                if cmd:
                    block = run_shell_escape(cmd)
                    if not local:
                        pending_shell.append(block)
                        config.console.print("[dim]output goes to the model with your next message[/dim]")
                config.console.print()
                continue
            if user.lower() in ("/clear", "clear"):
                # Drop all conversation history but keep messages[0] (the static
                # system prompt). The system prompt + tool schemas are the cached
                # prefix, so this frees the context window without paying to
                # prefill them again. Resetting first_user_msg re-injects the cwd
                # context into the next first user message.
                del messages[1:]
                del turns[:]
                del pending_shell[:]
                first_user_msg = True
                session_title  = ""
                config.console.print("[dim]context cleared (system prompt preserved)[/dim]\n")
                continue
            if user.lower() in ("/details", "/thinking"):
                # Display only: what the model is sent (and the `think` flag,
                # which would change the prompt template) stays as it was.
                attr = "SHOW_DETAILS" if user.lower() == "/details" else "SHOW_THINKING"
                setattr(config, attr, not getattr(config, attr))
                what = ("full tool results" if attr == "SHOW_DETAILS"
                        else "the model's reasoning while it streams")
                config.console.print(f"[dim]{'showing' if getattr(config, attr) else 'hiding'} "
                                     f"{what}[/dim]\n")
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
                    title    = meta.get("title") or ""
                    title    = f" — {title}" if title else ""
                    config.console.print(f"[dim]{escape(name)}{marker}{escape(title)} · "
                                         f"{escape(str(saved_at))} · {n} messages[/dim]")
                config.console.print()
                continue
            if user.lower() == "/save" or user.lower().startswith("/save "):
                arg_name = user.split(maxsplit=1)[1].strip() if " " in user else ""
                name = arg_name or session_name or default_ts_name()
                save_session(name, messages, session_title)
                session_name = name
                config.console.print(f"[dim]saved session '{escape(name)}'[/dim]\n")
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
                        save_session(session_name, messages, session_title)
                try:
                    data = load_session(new_name)
                    apply_session(messages, data, args.model)
                    messages[:] = [m for m in messages if not _is_stray_nudge(m)]
                except (OSError, ValueError, KeyError, TypeError):
                    config.console.print("[yellow]could not read session; conversation unchanged[/yellow]\n")
                    continue
                session_name   = new_name
                session_title  = data.get("title", "")
                first_user_msg = False
                del turns[:]
                del pending_shell[:]
                config.console.print(f"[dim]resumed session '{escape(new_name)}' ({len(messages)} messages)[/dim]\n")
                threading.Thread(target=warm_cache, args=(list(messages),), daemon=True).start()
                continue
            if user.lower() == "/undo" or re.fullmatch(r"/undo\s+\d+", user.lower()):
                n   = max(1, int(user.split()[1])) if " " in user else 1
                rec = None
                # One turn at a time, newest first, so each restore lands on the
                # snapshot the next-older turn started from.
                for i in range(n):
                    if not turns:
                        break
                    rec = undo_last_turn(messages, turns, announce=(i == n - 1 or len(turns) == 1))
                if rec is None:
                    undo_last_turn(messages, turns)      # prints "nothing to undo"
                else:
                    first_user_msg = rec["first"]
                    seed           = "" if "\n" in rec["prompt"] else rec["prompt"]
                    if first_user_msg:
                        session_title = ""
                    threading.Thread(target=warm_cache, args=(list(messages),), daemon=True).start()
                continue
            if user.lower() == "/history":
                if not turns:
                    config.console.print("[dim]no turns to list yet (a resumed or cleared "
                                         "conversation starts a new list)[/dim]\n")
                    continue
                for i, rec in enumerate(turns, start=1):
                    first = rec["prompt"].split("\n", 1)[0]
                    more  = " …" if "\n" in rec["prompt"] or len(first) > 70 else ""
                    config.console.print(f"[dim]{i:3}  {escape(first[:70])}{more}[/dim]")
                config.console.print(f"[dim]/undo N takes back the last N of these "
                                     f"({len(turns)} in all)[/dim]\n")
                continue
            if user.lower() == "/export" or user.lower().startswith("/export "):
                target = user.split(maxsplit=1)[1].strip() if " " in user else ""
                target = os.path.expanduser(target or f"tiny-agent-{session_name or default_ts_name()}.md")
                try:
                    with open(target, "w", encoding="utf-8") as f:
                        f.write(commands.export_markdown(messages, session_title))
                    config.console.print(f"[dim]wrote {escape(target)}[/dim]\n")
                except OSError as e:
                    config.console.print(f"[red]could not write {escape(target)}: {escape(str(e))}[/red]\n")
                continue
            if user.lower() == "/help":
                commands.print_help(custom_commands())
                continue
            if user.startswith("/"):
                name, _, argstr = user.partition(" ")
                custom = custom_commands()
                if name.lower() in custom:
                    user = commands.expand_custom_command(custom[name.lower()][2],
                                                          argstr.strip(), run_shell)
                    config.console.print(f"[dim]{escape(name.lower())} → "
                                         f"{len(user)} chars[/dim]")
                elif re.fullmatch(r"/[A-Za-z][\w-]*", name) and not os.path.exists(name):
                    # Almost certainly a mistyped command, not a question about
                    # the path /word: don't spend a turn sending it to the model.
                    config.console.print(f"[yellow]unknown command {escape(name)}; "
                                         f"/help lists them[/yellow]\n")
                    seed = "" if "\n" in user else user
                    continue
        if not user:
            continue

        # Snapshot before the turn can touch anything. Stat-cached (see
        # checkpoint._write_tree), so on an unchanged tree it costs a few git
        # calls, not a re-hash of the repo.
        turns.append({"msg_index": len(messages), "prompt": raw_prompt, "first": first_user_msg,
                      "cwd": os.getcwd(), "snap": checkpoint.snapshot()})
        if first_user_msg:
            session_title = raw_prompt.split("\n", 1)[0][:80]

        # cwd and instructions injected into the FIRST user message only —
        # keeps the system prompt byte-identical across projects so the
        # prefix cache hits every session. Computed here, not once at
        # startup, so a `cd` tool call or a `/clear` (which resets
        # first_user_msg) picks up the current directory instead of whatever
        # it was when the process started.
        content        = "\n\n".join(pending_shell + [expand_file_refs(user)])
        content        = (context_header() + "\n\n" + content) if first_user_msg else content
        del pending_shell[:]
        first_user_msg = False
        messages.append({"role": "user", "content": content})

        failed = True
        try:
            run_turn(messages, max_steps=args.max_steps, check_every=args.check_every)
            failed = False
        except urllib.error.HTTPError as e:
            body = getattr(e, "body", "") or e.read().decode(errors="replace")
            config.console.print(f"[red]Ollama error {e.code}: {escape(body.strip() or str(e.reason))}[/red]")
        except urllib.error.URLError as e:
            config.console.print(f"[red]cannot reach Ollama at {escape(config.OLLAMA_URL)}: {escape(str(e))}[/red]")
        except (TimeoutError, OSError) as e:
            # A stalled read mid-stream (see STREAM_TIMEOUT in ollama.py) raises
            # a raw socket timeout here rather than a urllib error, since it
            # happens while iterating an already-opened response, not while
            # opening the connection. Without this the process would crash.
            config.console.print(
                f"[red]connection to Ollama stalled (no data for "
                f"{config.STREAM_TIMEOUT}s; the call after a trim pass gets "
                f"a longer window sized to its re-prefill, and one retry): "
                f"{escape(str(e))}[/red]"
            )
        except KeyboardInterrupt:
            config.console.print("\n[yellow]interrupted[/yellow]")

        # Anything typed during the turn's last step never reached a step
        # boundary — the turn ended first (final answer, cancel, step limit, or
        # an error out of run_turn). Rather than swallow it, carry it over as
        # the next prompt; `initial` prints it and runs it like a typed one.
        leftover = take_interjections()
        if leftover:
            initial = "\n".join(leftover)
        config.console.print()
        if piped:
            # One turn per run; the exit code says whether it got an answer.
            sys.exit(1 if failed else 0)
