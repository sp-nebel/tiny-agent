import sys
import select
import contextlib

from rich.text import Text

import config

try:                      # POSIX-only; used to read single keypresses mid-stream
    import termios
    import tty
except ImportError:
    termios = tty = None

try:                      # line editing + history for the interactive prompt
    import readline
except ImportError:
    readline = None

# --------------------------------------------------------------------------- #
# Terminal helpers
# --------------------------------------------------------------------------- #

# Terminal state from before cbreak_stdin switched modes, so cooked_stdin can
# hand it back for the length of one line-edited read.
_saved_tty = None


@contextlib.contextmanager
def cbreak_stdin():
    """Put stdin in cbreak mode so single keypresses read without Enter.

    No-op when stdin isn't a tty or termios is unavailable (e.g. piped input,
    non-POSIX). cbreak leaves ISIG enabled, so Ctrl-C still raises normally.
    """
    global _saved_tty
    if not (termios and tty and sys.stdin.isatty()):
        yield
        return
    fd  = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        _saved_tty = old
        yield
    finally:
        _saved_tty = None
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


@contextlib.contextmanager
def cooked_stdin():
    """Undo cbreak for the duration of the block, then put it back.

    Reading a whole line mid-stream needs canonical mode: in cbreak the
    terminal delivers characters one at a time and readline's line editing
    (backspace, arrows, paste) has nothing to work with.
    """
    if _saved_tty is None:
        yield
        return
    fd    = sys.stdin.fileno()
    inner = termios.tcgetattr(fd)
    termios.tcsetattr(fd, termios.TCSADRAIN, _saved_tty)
    try:
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, inner)


ESC = "\x1b"

# How long to wait for the rest of an escape sequence after an ESC byte. A
# terminal sends the whole of one (arrow keys, function keys, mouse reports)
# in a single burst, so anything still silent after this really was a bare
# Esc keypress. Paid only when Esc arrives, never on the streaming path.
ESC_SEQUENCE_WAIT = 0.02


def poll_keypress():
    """Classify one waiting keypress as (action, seed). Non-blocking.

    Actions: 'cancel' (Esc), 'interject' (Enter, or any printable character —
    which comes back as `seed`, the first character of the message, since
    cbreak turns echo off and it would otherwise be lost), or None for
    everything else.

    Esc only cancels when it arrives alone: arrow keys and friends are escape
    *sequences*, and cancelling a long reply because the user pressed Up would
    be a sharp edge on the one key that now cancels. Their trailing bytes are
    drained here so they can't be misread as a message.
    """
    if not sys.stdin.isatty():
        return None, ""
    if not select.select([sys.stdin], [], [], 0)[0]:
        return None, ""
    ch = sys.stdin.read(1)
    if ch == ESC:
        if not select.select([sys.stdin], [], [], ESC_SEQUENCE_WAIT)[0]:
            return "cancel", ""
        while select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.read(1)
        return None, ""
    if ch in ("\r", "\n"):
        return "interject", ""
    if ch.isprintable():
        return "interject", ch
    return None, ""     # backspace, tab, stray control characters


BRACKETED_PASTE_ON  = "\x1b[?2004h"
BRACKETED_PASTE_OFF = "\x1b[?2004l"


def read_prompt(prompt: str, seed: str = "") -> str:
    """console.input() with bracketed paste enabled, so a multi-line paste
    comes back as one message instead of one-per-line.

    `seed` pre-fills the line buffer — used for the character that opened the
    interject prompt, so typing straight into a stream keeps its first letter
    and stays editable. Without readline there is no buffer to pre-fill, so it
    is prepended to the result instead; the character is kept either way.

    Python's input()/readline parses the \\e[200~…\\e[201~ markers a terminal
    wraps pastes in, but — unlike bash — never emits the escape that turns
    that mode on, so we do it ourselves. No-op without a tty or readline
    (nothing would parse the markers, so the escapes would just leak into
    the text). Turned off again before returning so it's not left on during
    streaming, where poll_keypress() would read the paste's leading \\x1b
    as an escape sequence and swallow the start of it.
    """
    # Both the paste markers and the startup hook are readline's doing, and
    # input() only routes through readline when stdin is a tty.
    editing = bool(readline) and sys.stdin.isatty()
    if editing:
        sys.stdout.write(BRACKETED_PASTE_ON); sys.stdout.flush()
    if seed and editing:
        readline.set_startup_hook(lambda: readline.insert_text(seed))
    try:
        text = config.console.input(prompt)
        return text if editing else seed + text
    finally:
        if seed and editing:
            readline.set_startup_hook()
        if editing:
            sys.stdout.write(BRACKETED_PASTE_OFF); sys.stdout.flush()


# Messages typed while a reply was streaming, waiting to be handed to the
# model. Module-level rather than passed around because stdin already is:
# call_ollama re-enters itself on three retry paths (think fallback, refused
# connection, post-trim stall) and throws the partial stream away, so a queue
# carried in the return value would be thrown away with it. Drained by
# agent.run_turn at the next step boundary, and by main() after the turn for
# anything that never got there.
_interjections = []


def interjections_pending():
    return bool(_interjections)


def take_interjections():
    """Pop every queued interjection, oldest first."""
    queued = _interjections[:]
    del _interjections[:]
    return queued


def read_interjection(live, seed="", repaint=None):
    """Pause the live stream region, read one line from the user, queue it.

    `seed` is the character that opened the prompt, if it was opened by typing
    rather than by Enter; it is pre-filled into the line so nothing typed is
    lost (see read_prompt).

    Called from inside the streaming loop, where the terminal is in cbreak
    mode and Rich owns the bottom of the screen: stop the Live region (it is
    transient, so it erases itself), restore canonical mode for the read, then
    restart and repaint immediately — Live only paints on its own throttle, so
    without `repaint` the region would sit blank until the next chunk lands.

    No socket read is outstanding while the user types, and STREAM_TIMEOUT is
    a per-read timeout, so a slow typist can't time the connection out; the
    server keeps generating into the socket buffer meanwhile.
    """
    # Live.stop() forces vertical_overflow to "visible" so its last frame
    # renders whole, and start() doesn't undo it — left alone, the restarted
    # region would scroll the terminal instead of cropping, which is what
    # _render_stream's height budget exists to prevent.
    overflow = live.vertical_overflow
    live.stop()
    try:
        with cooked_stdin():
            text = read_prompt("[bold green]interject[/bold green] ", seed).strip()
    except (EOFError, KeyboardInterrupt):
        text = ""          # abandoning the line must not cancel the reply
    finally:
        live.start()
        live.vertical_overflow = overflow
        if repaint is not None:
            live.update(repaint)
    if text:
        _interjections.append(text)
        config.console.print("[dim]queued — the model sees it after this step[/dim]")
    return text


def _render_stream(thinking: str, content: str) -> Text:
    """Build the live view: reasoning above the answer-so-far, both dim.

    Lives only in the transient Live region, so when the stream finishes the
    whole thing is wiped and run_turn re-renders the content as Markdown —
    a final answer and a mid-turn update alike, so nothing the model wrote
    for the user is lost. Only the reasoning stays wiped: it is scratch work,
    not something addressed to the reader.

    The reasoning is shown through a sliding window over its tail: Rich's Live
    region crops anything taller than the terminal from the *bottom*, which
    would hide the newest streamed tokens. Instead we keep the view within the
    terminal height ourselves — answer in full, plus as many of the most recent
    thinking lines as fit above it — so the live tail is always what you see.
    """
    avail = max(4, config.console.size.height - 2)
    out   = Text()

    content_lines = content.splitlines() if content else []
    # Reserve rows for the answer; the rest is the reasoning window. -2 leaves
    # room for the "thinking" header and the blank separator line.
    think_budget = max(3, avail - len(content_lines) - 2)

    if thinking:
        tlines = thinking.splitlines() or [thinking]
        hidden = len(tlines) - think_budget
        if hidden > 0:
            tlines = tlines[-think_budget:]
            out.append(f"thinking (…{hidden} earlier line{'s' if hidden != 1 else ''})\n",
                       style="dim italic")
        else:
            out.append("thinking\n", style="dim italic")
        out.append("\n".join(tlines) + ("\n\n" if content else ""), style="dim italic")

    if content:
        out.append("\n".join(content_lines), style="dim")
    return out

# --------------------------------------------------------------------------- #
# Display helpers
# --------------------------------------------------------------------------- #

def fmt_args(args: dict) -> str:
    parts = []
    for k, v in args.items():
        s = repr(v)
        parts.append(f"{k}={s[:57] + '…' if len(s) > 60 else s}")
    return ", ".join(parts)


def approx_tokens(text: str) -> int:
    """Rough token count from the ~4-chars/token heuristic (the same estimate
    trim_history uses for its window-pressure check). Ollama exposes no
    tokenizer endpoint, so this is a display-only approximation — exact counts
    are only available for whole LLM calls, via the stats in fmt_stats."""
    return len(text) // 4


def truncate(text: str, n: int = 600) -> str:
    return text if len(text) <= n else text[:n] + f"\n… (~{approx_tokens(text)} tokens total)"


def fmt_stats(stats: dict, calls: int) -> str:
    """One-line summary of a whole turn, summed across its `calls` LLM calls.

    A turn fans out into several call_ollama requests (one per tool round-trip),
    so these are turn totals: prefill is the total tokens prefilled across the
    turn — the first call dominates it, since later calls reuse the KV prefix
    cache and only prefill the new tool results — and the rate is the
    token-weighted gen tok/s (summed eval_count over summed eval_duration).
    """
    p_tok = stats.get("prompt_eval_count", 0)
    p_dur = stats.get("prompt_eval_duration", 0) / 1e9
    g_tok = stats.get("eval_count", 0)
    g_dur = stats.get("eval_duration", 0) / 1e9
    rate  = f"{g_tok / g_dur:.1f} tok/s" if g_dur else "—"
    prefix = f"{calls} steps · " if calls > 1 else ""
    return f"{prefix}prefill {p_tok} tok in {p_dur:.1f}s · gen {g_tok} tok @ {rate}"
