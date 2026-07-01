import sys
import select
import contextlib

from rich.text import Text

import config

try:                      # POSIX-only; used to read a single cancel keypress
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

@contextlib.contextmanager
def cbreak_stdin():
    """Put stdin in cbreak mode so single keypresses read without Enter.

    No-op when stdin isn't a tty or termios is unavailable (e.g. piped input,
    non-POSIX). cbreak leaves ISIG enabled, so Ctrl-C still raises normally.
    """
    if not (termios and tty and sys.stdin.isatty()):
        yield
        return
    fd  = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def cancel_pressed() -> bool:
    """True if a cancel key (Esc or 'q') is waiting on stdin. Non-blocking."""
    if not sys.stdin.isatty():
        return False
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1) in ("\x1b", "q", "Q")
    return False


BRACKETED_PASTE_ON  = "\x1b[?2004h"
BRACKETED_PASTE_OFF = "\x1b[?2004l"


def read_prompt(prompt: str) -> str:
    """console.input() with bracketed paste enabled, so a multi-line paste
    comes back as one message instead of one-per-line.

    Python's input()/readline parses the \\e[200~…\\e[201~ markers a terminal
    wraps pastes in, but — unlike bash — never emits the escape that turns
    that mode on, so we do it ourselves. No-op without a tty or readline
    (nothing would parse the markers, so the escapes would just leak into
    the text). Turned off again before returning so it's not left on during
    streaming, where cancel_pressed() would otherwise misread the paste's
    leading \\x1b as the cancel key.
    """
    bracket = bool(readline) and sys.stdin.isatty()
    if bracket:
        sys.stdout.write(BRACKETED_PASTE_ON); sys.stdout.flush()
    try:
        return config.console.input(prompt)
    finally:
        if bracket:
            sys.stdout.write(BRACKETED_PASTE_OFF); sys.stdout.flush()


def _render_stream(thinking: str, content: str) -> Text:
    """Build the live view: reasoning above the answer-so-far, both dim.

    Lives only in the transient Live region, so when the stream finishes the
    whole thing — thinking included — is wiped and run_turn re-renders just
    the final content as Markdown.

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
