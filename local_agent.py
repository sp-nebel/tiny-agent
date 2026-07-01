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

Env vars:
    AGENT_MODEL           (default: gemma4:12b-it-qat)   — any Ollama model with tool support
    OLLAMA_URL            (default: http://localhost:11434)
    AGENT_THINK           (default: 1) — set to 0 to disable reasoning output
    AGENT_SUMMARIZE_TRIM  (default: 1) — set to 0 to elide trimmed tool outputs instead of summarizing them

Dependency: pip install rich
Note: ensure Ollama >= 0.20.2 for reliable Gemma 4 tool-call parsing.
"""

import os
import sys
import glob
import json
import time
import atexit
import select
import shutil
import difflib
import argparse
import threading
import subprocess
import contextlib
import urllib.request
import urllib.error

try:                      # POSIX-only; used to read a single cancel keypress
    import termios
    import tty
except ImportError:
    termios = tty = None

try:                      # line editing + history for the interactive prompt
    import readline
except ImportError:
    readline = None

from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.text import Text

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

OLLAMA_URL  = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL       = os.environ.get("AGENT_MODEL", "gemma4:12b-it-qat")

SESSION_DIR = os.path.expanduser("~/.tiny_agent_sessions")

MAX_READ_LINES = 100
MAX_GREP_HITS  = 20
MAX_GLOB_HITS  = 20
MAX_CMD_CHARS  = 8000
CMD_TIMEOUT    = 120
NUM_CTX        = 32768

# History trimming: when the conversation approaches the context window,
# collapse all but the N most recent tool outputs to a one-line stub. Editing
# history busts the KV prefix cache from the edit point on, so trimming only
# happens when the window is actually under pressure — one cache bust per
# long session, not one per step. Only outputs over the threshold collapse.
KEEP_FULL_TOOL_RESULTS = 3
TRIM_MIN_CHARS         = 400
TRIM_AT_TOKENS         = int(NUM_CTX * 0.7)

# Summarize-on-trim: when trim_history collapses an old tool output, instead of
# discarding it to a one-line stub, spawn a fresh empty Ollama session (same
# resident model) to digest it down to the task-relevant facts and keep THAT.
# Only ever runs at the trim moment — i.e. when the window is already under
# pressure and the cache is being busted anyway — so it costs nothing on short
# sessions. Disable with AGENT_SUMMARIZE_TRIM=0 to get the old [elided] stubs.
SUMMARIZE_ON_TRIM  = os.environ.get("AGENT_SUMMARIZE_TRIM", "1") not in ("0", "false", "")
SUMMARY_MAX_TOKENS = 256          # bounds the gen cost of each digest
# Sentinel prefixing every compacted message. A summary can exceed
# TRIM_MIN_CHARS, so "a stub is short → never re-collapsed" no longer holds;
# trim_history skips anything already starting with this prefix instead.
TRIM_PREFIX        = "[«compacted» "
SUMMARIZER_SYSTEM  = (
    "You compress one tool result for an autonomous coding agent. Given the "
    "task it was gathered for and the raw output, return a terse factual digest "
    "(at most ~8 lines) of ONLY what is relevant to the task: file paths, line "
    "numbers, symbol names, key values, error messages. No preamble, no advice."
)

# Directories never worth walking in a glob.
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv",
             ".mypy_cache", ".pytest_cache", ".ruff_cache"}

AUTO_YES = False
# Ask the model to emit reasoning in a separate `thinking` field. Streamed
# live as feedback on slow CPU runs, then discarded once the answer lands.
# Auto-disabled at runtime if the model doesn't support thinking.
THINK = os.environ.get("AGENT_THINK", "1") not in ("0", "false", "")

console = Console()

# --------------------------------------------------------------------------- #
# Static system prompt  (KEEP BYTE-IDENTICAL — this is the cached prefix)
#
# Tool *descriptions* are NOT here; they live in the tools parameter below,
# so the model reads them in its trained format. This prompt only sets
# working style.
# --------------------------------------------------------------------------- #

SYSTEM = """You are a coding assistant working in a local code repository.

Work lazily: use grep to locate relevant code, then read_file with a tight
line range around what you actually need. Do not read whole files when a
small range will do. Make one tool call at a time and wait for its result.

Bracketed lines like [lines 1-100 of 543] in tool results are metadata from
the tool, not file content. If a read was truncated and you need more of the
file, call read_file again with the start value the notice gives you.

When the task is done, reply in Markdown with no tool call."""

# --------------------------------------------------------------------------- #
# Tool schemas  (static — also part of the cached prefix)
# --------------------------------------------------------------------------- #

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search for a regex pattern in a file or directory tree. Returns matching lines with file path and line number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string",  "description": "Search pattern (regex)"},
                    "path":    {"type": "string",  "description": "File or directory to search (default '.')"},
                    "context": {"type": "integer", "description": "Show N lines of context before AND after each match (-C)"},
                    "before":  {"type": "integer", "description": "Show N lines before each match (-B); ignored if context is set"},
                    "after":   {"type": "integer", "description": "Show N lines after each match (-A); ignored if context is set"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a file. Output is line-numbered. "
                "Use start/end to read a specific range rather than the whole file. "
                "Returns at most 100 lines per call. If the requested range is "
                "longer, the result ends with a bracketed notice giving the start "
                "value for the next call - that notice is tool metadata, not file "
                "content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path":  {"type": "string",  "description": "File path"},
                    "start": {"type": "integer", "description": "First line to read (1-indexed, default 1)"},
                    "end":   {"type": "integer", "description": "Last line to read (inclusive)"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_files",
            "description": (
                "Find files and directories by name with a glob pattern. "
                "Use ** to recurse, e.g. '**/*.py' or 'src/**/test_*.py'. "
                "Searches file names/paths, not contents (use grep for contents). "
                "Common noise dirs (.git, node_modules, __pycache__) are skipped."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern, e.g. '**/*.py' or 'config*.yaml'"},
                    "path":    {"type": "string", "description": "Base directory to search from (default '.')"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List entries in a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path (default '.')"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cd",
            "description": (
                "Change the working directory. Relative paths in later tool "
                "calls are resolved from here. Returns the new working directory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory to change into"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Edit a file by replacing an exact string. old_string must match "
                "the file byte-for-byte (whitespace included) and be unique, "
                "unless replace_all is true. To CREATE a new file, pass an empty "
                "old_string and the full contents in new_string. Prefer a small, "
                "uniquely-identifying old_string over a large one. "
                "Will ask the user for confirmation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path":        {"type": "string",  "description": "File path to edit"},
                    "old_string":  {"type": "string",  "description": "Exact text to replace; empty string to create a new file"},
                    "new_string":  {"type": "string",  "description": "Replacement text (or full file contents when creating)"},
                    "replace_all": {"type": "boolean", "description": "Replace every occurrence instead of requiring a unique match (default false)"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_cmd",
            "description": "Run a shell command and return stdout + stderr. Will ask the user for confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "Shell command to run"},
                },
                "required": ["cmd"],
            },
        },
    },
]

# --------------------------------------------------------------------------- #
# Tools (implementations)
# --------------------------------------------------------------------------- #

def confirm(msg: str) -> bool:
    if AUTO_YES:
        return True
    try:
        ans = console.input(f"[yellow]{msg}[/yellow] [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return ans in ("y", "yes")


def show_diff(old: str, new: str, path: str, max_lines: int = 60):
    """Print a colored unified diff so edit confirmations aren't blind.

    Shown even under --yes: it costs nothing and is the only record of what
    the agent actually changed.
    """
    diff = list(difflib.unified_diff(
        old.splitlines(), new.splitlines(),
        fromfile=f"{path} (old)", tofile=f"{path} (new)", lineterm="",
    ))
    if len(diff) > max_lines:
        hidden = len(diff) - max_lines
        diff = diff[:max_lines] + [f"… ({hidden} more diff lines)"]
    out = Text()
    for line in diff:
        if line.startswith("+") and not line.startswith("+++"):
            style = "green"
        elif line.startswith("-") and not line.startswith("---"):
            style = "red"
        elif line.startswith("@@"):
            style = "cyan"
        else:
            style = "dim"
        out.append(line + "\n", style=style)
    console.print(out, end="")


def t_read_file(path, start=1, end=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return f"[no such file: {path}]"
    except UnicodeDecodeError:
        return f"[binary file, cannot display: {path}]"
    except OSError as e:
        return f"[error reading {path}: {e}]"

    total = len(lines)
    start = max(1, int(start))
    if start > total:
        return f"[start={start} is beyond end of file ({total} lines)]"
    # Cap the range even when end is explicit, so a huge end can't pull in
    # the whole file in one call.
    end = int(end) if end else total
    end = min(end, total, start + MAX_READ_LINES - 1)
    if end < start:
        return f"[invalid range: end={end} is before start={start}]"

    body = "".join(f"{n:5}  {ln}" for n, ln in enumerate(lines[start-1:end], start=start))
    if body and not body.endswith("\n"):
        body += "\n"
    if end < total:
        # Notice at both ends: small models attend poorly to the tail of a
        # long result, and an imperative is followed better than a hint.
        body = (
            f"[lines {start}-{end} of {total} - file continues]\n"
            + body
            + f"[TRUNCATED. To continue reading, call read_file with start={end+1}.]"
        )
    return body or "[empty file]"


def t_grep(pattern, path=".", context=0, before=0, after=0):
    # Base command differs (rg vs grep), but every flag below is accepted
    # identically by both, so the model sees one consistent interface.
    if shutil.which("rg"):
        cmd = ["rg", "-n", "--no-heading"]   # respects .gitignore by default
    else:
        cmd = ["grep", "-rn"] + [f"--exclude-dir={d}" for d in sorted(SKIP_DIRS)]
    if context:
        cmd += ["-C", str(int(context))]
    else:
        if before:
            cmd += ["-B", str(int(before))]
        if after:
            cmd += ["-A", str(int(after))]
    cmd += ["--", pattern, path]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=CMD_TIMEOUT)
    except subprocess.TimeoutExpired:
        return "[grep timed out]"
    # rg and grep agree: exit 0 = matches, 1 = no matches, ≥2 = real error
    # (bad regex, unreadable path). Don't let an error message pass as hits.
    if out.returncode > 1:
        return f"[grep error: {out.stderr.strip() or f'exit {out.returncode}'}]"
    res   = out.stdout.strip()
    hits  = res.splitlines()
    if len(hits) > MAX_GREP_HITS:
        res = "\n".join(hits[:MAX_GREP_HITS]) + f"\n[+{len(hits) - MAX_GREP_HITS} more matches]"
    return res or "[no matches]"


def t_find_files(pattern, path="."):
    base = path or "."
    try:
        matches = glob.glob(os.path.join(base, pattern), recursive=True)
    except OSError as e:
        return f"[error: {e}]"

    out = []
    for m in sorted(matches):
        # Drop anything living under a noise dir (any path component matches).
        if SKIP_DIRS.intersection(m.split(os.sep)):
            continue
        out.append(m + ("/" if os.path.isdir(m) else ""))

    if len(out) > MAX_GLOB_HITS:
        out = out[:MAX_GLOB_HITS] + [f"[+{len(out) - MAX_GLOB_HITS} more]"]
    return "\n".join(out) or "[no matches]"


def t_list_dir(path="."):
    try:
        entries = sorted(os.listdir(path))
    except OSError as e:
        return f"[error: {e}]"
    lines = [e + ("/" if os.path.isdir(os.path.join(path, e)) else "") for e in entries]
    return "\n".join(lines) or "[empty]"


def t_cd(path):
    try:
        os.chdir(path)
    except FileNotFoundError:
        return f"[no such directory: {path}]"
    except NotADirectoryError:
        return f"[not a directory: {path}]"
    except OSError as e:
        return f"[error changing directory: {e}]"
    return f"[cwd: {os.getcwd()}]"


def t_edit_file(path, old_string, new_string, replace_all=False):
    # Empty old_string ⇒ create a new file (the write_file behaviour, folded in).
    if old_string == "":
        if os.path.exists(path):
            return f"[{path} already exists; put the text to replace in old_string]"
        show_diff("", new_string, path)
        if not confirm(f"create {path} ({len(new_string)} chars)?"):
            return "[user declined write]"
        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_string)
        except OSError as e:
            return f"[error writing {path}: {e}]"
        return f"[created {path}, {len(new_string)} chars]"

    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        return f"[no such file: {path}; pass an empty old_string to create it]"
    except UnicodeDecodeError:
        return f"[binary file, cannot edit: {path}]"
    except OSError as e:
        return f"[error reading {path}: {e}]"

    count = content.count(old_string)
    if count == 0:
        return "[old_string not found; it must match the file exactly, whitespace included]"
    if count > 1 and not replace_all:
        return (f"[old_string matches {count} times; add surrounding context to "
                f"make it unique, or set replace_all=true]")

    n           = count if replace_all else 1
    new_content = content.replace(old_string, new_string, -1 if replace_all else 1)
    plural      = "s" if n != 1 else ""
    show_diff(content, new_content, path)
    if not confirm(f"edit {path} ({n} replacement{plural})?"):
        return "[user declined write]"
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
    except OSError as e:
        return f"[error writing {path}: {e}]"
    return f"[edited {path}: {n} replacement{plural}]"


def t_run_cmd(cmd):
    if not confirm(f"run: {cmd}"):
        return "[user declined command]"
    try:
        out = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=CMD_TIMEOUT
        )
    except subprocess.TimeoutExpired:
        return f"[timed out after {CMD_TIMEOUT}s]"
    combined = (out.stdout + out.stderr).strip()
    if len(combined) > MAX_CMD_CHARS:
        # Keep head AND tail: test runners and builds put the failure summary
        # at the end, and losing it makes the model re-run the command.
        head = MAX_CMD_CHARS // 4
        tail = MAX_CMD_CHARS - head
        combined = (combined[:head]
                    + f"\n[… {len(combined) - MAX_CMD_CHARS} chars elided …]\n"
                    + combined[-tail:])
    return combined or f"[exit {out.returncode}, no output]"


TOOLS = {
    "read_file":  t_read_file,
    "grep":       t_grep,
    "find_files": t_find_files,
    "list_dir":   t_list_dir,
    "cd":         t_cd,
    "edit_file":  t_edit_file,
    "run_cmd":    t_run_cmd,
}


def dispatch(name, args):
    fn = TOOLS.get(name)
    if fn is None:
        return f"[unknown tool: {name}]"
    try:
        return fn(**args)
    except TypeError as e:
        return f"[bad args for {name}: {e}]"
    except Exception as e:
        return f"[tool error: {e}]"

# --------------------------------------------------------------------------- #
# Ollama call  (streaming, native tool-call detection)
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
    avail = max(4, console.size.height - 2)
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


def call_ollama(messages):
    """Stream a chat turn.

    Returns (content: str, tool_calls: list, cancelled: bool, stats: dict).
    Exactly one of content/tool_calls is meaningful: a final answer has
    content and no tool_calls; a tool-calling turn has tool_calls. Reasoning
    arrives in a separate `thinking` field; it is shown live but never
    returned, so it is discarded once the answer is rendered. cancelled is
    True if the user pressed the cancel key (Esc/q) mid-stream; stats is a
    dict of raw token counters from the final chunk, empty ({}) in that case
    (the final chunk never arrived) — run_turn sums it across the turn.
    """
    global THINK

    payload = {
        "model":      MODEL,
        "messages":   messages,
        "tools":      TOOL_SCHEMAS,
        "stream":     True,
        "keep_alive": "30m",    # keeps model resident → prefix cache stays warm
        "options": {
            "num_ctx":     NUM_CTX,
        },
    }
    if THINK:
        payload["think"] = True

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )

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
        if THINK and e.code == 400 and "think" in body.lower():
            THINK = False
            return call_ollama(messages)
        e.body = body   # already consumed; stash for the handler in main()
        raise

    # cbreak lets us catch a single cancel keypress without blocking the
    # stream; transient=True clears the live region (thinking included) when
    # done, so run_turn re-renders only the final answer.
    with resp, cbreak_stdin():
        with Live(console=console, refresh_per_second=8, transient=True) as live:
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

    return content, tool_calls, cancelled, stats


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
        messages = [{"role": "system", "content": SYSTEM}]
    payload = {
        "model":      MODEL,
        "messages":   messages,
        "tools":      TOOL_SCHEMAS,
        "stream":     False,
        "keep_alive": "30m",
        "options": {
            "num_ctx":     NUM_CTX,
            "num_predict": 1,
        },
    }
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
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
    payload = {
        "model":      MODEL,
        "messages": [
            {"role": "system", "content": SUMMARIZER_SYSTEM},
            {"role": "user",   "content": f"Task:\n{task}\n\n{tool_name} output:\n{content}"},
        ],
        "stream":     False,
        "keep_alive": "30m",
        "options": {
            "num_ctx":     NUM_CTX,
            "num_predict": SUMMARY_MAX_TOKENS,
        },
    }
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=CMD_TIMEOUT) as resp:
            obj = json.loads(resp.read())
    except (OSError, ValueError):
        return None
    text = (obj.get("message") or {}).get("content", "").strip()
    return text or None

# --------------------------------------------------------------------------- #
# Session persistence (pause/resume a conversation across process restarts)
# --------------------------------------------------------------------------- #

def default_ts_name():
    return time.strftime("%Y-%m-%dT%H-%M")


def session_path(name):
    return os.path.join(SESSION_DIR, os.path.basename(name) + ".json")


def save_session(name, messages):
    os.makedirs(SESSION_DIR, exist_ok=True)
    data = {
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model":    MODEL,
        "cwd":      os.getcwd(),
        "messages": messages,
    }
    with open(session_path(name), "w", encoding="utf-8") as f:
        json.dump(data, f)


def load_session(name):
    with open(session_path(name), "r", encoding="utf-8") as f:
        return json.load(f)


def list_sessions():
    """Saved session file paths, newest-modified first."""
    files = glob.glob(os.path.join(SESSION_DIR, "*.json"))
    return sorted(files, key=os.path.getmtime, reverse=True)


def resolve_session(name):
    """name -> file path. Empty/None picks the most-recently-modified session.
    Returns None if nothing matches."""
    if name:
        p = session_path(name)
        return p if os.path.exists(p) else None
    files = list_sessions()
    return files[0] if files else None


def apply_session(messages, data, model_override=None):
    """Restore a loaded session dict into the live `messages` list.

    Mutates `messages` in place (`messages[:] = ...`) rather than rebinding it,
    so closures that captured the list — like the autosave handler — keep
    seeing the live conversation.
    """
    global MODEL
    messages[:] = data["messages"]
    with contextlib.suppress(OSError):
        os.chdir(data["cwd"])
    MODEL = model_override if model_override else data.get("model", MODEL)

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

# --------------------------------------------------------------------------- #
# Agent loop
# --------------------------------------------------------------------------- #

# Appended at the tail on the final allowed step to force a closing answer
# instead of yet another tool call. Tail-only, so the prefix cache is untouched.
STEP_LIMIT_NUDGE = (
    "[step limit reached — give your best final answer now using what you've "
    "gathered; do not call any more tools.]"
)


def trim_history(messages):
    """Collapse old tool outputs in place to conserve the context window.

    Untouched history is free: it sits in the KV prefix cache and is never
    re-prefilled. Editing a message, by contrast, invalidates the cache from
    the edit point on. So trimming is lazy — do nothing until the conversation
    approaches the context window, then collapse every tool result outside the
    keep-window in one pass. One cache bust per long session, not one per step.

    The token estimate uses the ~4 chars/token heuristic; it only needs to be
    right to within the 30% headroom left below NUM_CTX.

    When SUMMARIZE_ON_TRIM is set, each collapsed output is first run through
    summarize_output (a fresh session on the same model) so the kept stub is an
    informative digest rather than a bare "[elided]" line. That fires a burst of
    summarizer calls, but only here — at the already-expensive trim moment — so
    short sessions pay nothing. On failure it falls back to the plain stub.

    Idempotent by construction: every collapsed message is prefixed with
    TRIM_PREFIX and skipped on later passes (a length check no longer suffices,
    since a digest can be longer than TRIM_MIN_CHARS).
    """
    total_tokens = sum(len(m.get("content") or "") for m in messages) // 4
    if total_tokens < TRIM_AT_TOKENS:
        return
    task = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
    tool_idxs = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    old = tool_idxs[:-KEEP_FULL_TOOL_RESULTS] if KEEP_FULL_TOOL_RESULTS else tool_idxs
    targets = [i for i in old
               if not (messages[i].get("content") or "").startswith(TRIM_PREFIX)
               and len(messages[i].get("content") or "") > TRIM_MIN_CHARS]
    if not targets:
        return
    console.print(f"[dim]compacting {len(targets)} old tool output"
                  f"{'s' if len(targets) != 1 else ''}…[/dim]")
    for i in targets:
        m       = messages[i]
        content = m.get("content", "")
        name    = m.get("name", "tool")
        nlines  = content.count("\n") + 1
        marker  = f"{TRIM_PREFIX}{name} — was {nlines} lines, {len(content)} chars]"
        summary = summarize_output(name, content, task) if SUMMARIZE_ON_TRIM else None
        m["content"] = f"{marker}\n{summary}" if summary else marker


def run_turn(messages, max_steps=20):
    # A turn fans out into several call_ollama requests (one per tool
    # round-trip); sum their stats and print one summary line when the turn
    # finishes, rather than a line per call.
    turn_stats = {}
    calls = 0
    for step in range(max_steps):
        trim_history(messages)
        # Last allowed step: force an answer. We keep the tool schemas in the
        # payload (dropping them would shift the cached prefix and bust nearly
        # the whole prefill) and instead append a nudge at the *tail* — only its
        # own tokens prefill, so the prefix cache stays intact.
        last = step == max_steps - 1
        if last:
            messages.append({"role": "user", "content": STEP_LIMIT_NUDGE})

        content, tool_calls, cancelled, stats = call_ollama(messages)

        if cancelled:
            # User aborted mid-stream. Drop the partial reply (don't commit it
            # to history) and hand control back to the prompt.
            console.print("[yellow]cancelled[/yellow]")
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
        messages.append(assistant_msg)

        if not keep_tc:
            # Final answer (normal early finish, or the forced last step) —
            # render as Markdown, then the turn summary.
            if content.strip():
                console.print(Markdown(content))
            else:
                console.print("[yellow]hit step limit; model returned no answer[/yellow]")
            if turn_stats:
                console.print(f"[dim]{fmt_stats(turn_stats, calls)}[/dim]")
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

                console.print(f"[cyan]→ {name}({fmt_args(args)})[/cyan]")
                result = dispatch(name, args)
                console.print(f"[dim]{truncate(result)}[/dim]\n")

                # Tool results use role "tool", one message per call.
                messages.append({"role": "tool", "content": result, "name": name})
                answered += 1
        finally:
            for tc in tool_calls[answered:]:
                name = tc.get("function", {}).get("name", "tool")
                messages.append({"role": "tool", "content": "[interrupted before this tool ran]", "name": name})

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    global AUTO_YES, MODEL

    ap = argparse.ArgumentParser(description="Minimal local coding agent over Ollama.")
    ap.add_argument("prompt",      nargs="*",      help="initial task (optional)")
    ap.add_argument("--model",     default=None,   help=f"Ollama model tag (default: {MODEL}, or the saved model when resuming)")
    ap.add_argument("--yes",       action="store_true", help="auto-approve writes and commands")
    ap.add_argument("--max-steps", type=int, default=20, help="max tool calls per task")
    ap.add_argument("--resume",    nargs="?", const="", default=None,
                     help="resume a saved session by name; bare --resume resumes the most recent")
    args = ap.parse_args()

    if args.model:
        MODEL = args.model
    AUTO_YES = args.yes

    # Persistent prompt history: importing readline upgrades input() in place,
    # so console.input gets line editing and up-arrow recall for free.
    if readline:
        histfile = os.path.expanduser("~/.tiny_agent_history")
        with contextlib.suppress(OSError):
            readline.read_history_file(histfile)
        readline.set_history_length(500)

        def save_history():
            with contextlib.suppress(OSError):
                readline.write_history_file(histfile)
        atexit.register(save_history)

    messages       = [{"role": "system", "content": SYSTEM}]
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
                session_name   = name
                first_user_msg = False
                console.print(f"[dim]resumed session '{name}' ({len(messages)} messages)[/dim]")
            except (OSError, ValueError, KeyError, TypeError):
                console.print("[yellow]could not read session; starting fresh[/yellow]")
        else:
            console.print("[yellow]no matching session; starting fresh[/yellow]")

    # cwd injected into the FIRST user message only — keeps the system prompt
    # byte-identical across projects so the prefix cache hits every session.
    # (For a resumed session first_user_msg is already False, so this is only
    # ever used for a fresh one — computed after any resume-triggered chdir.)
    cwd_context = f"Working directory: {os.getcwd()}\n\n"

    initial = " ".join(args.prompt).strip()

    # Interactive start: prefill the static prefix — or, if a session was just
    # restored, the full restored history — while the user types their first
    # prompt. With an argv prompt the real request follows immediately, so a
    # warmup would just queue ahead of it for no gain.
    if not initial:
        # A snapshot, not the live list: the main loop may append the user's
        # first message to `messages` before this thread's request goes out.
        threading.Thread(target=warm_cache, args=(list(messages),), daemon=True).start()

    console.print(f"[bold]tiny-agent[/bold] · {MODEL} · {os.getcwd()}")
    console.print(
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
            console.print(f"[bold green]you[/bold green] {user}")
        else:
            try:
                user = console.input("[bold green]you[/bold green] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\nbye")
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
            console.print("[dim]context cleared (system prompt preserved)[/dim]\n")
            continue
        if user.lower() == "/sessions":
            files = list_sessions()
            if not files:
                console.print("[dim]no saved sessions[/dim]\n")
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
                console.print(f"[dim]{name}{marker} — {saved_at} · {n} messages[/dim]")
            console.print()
            continue
        if user.lower() == "/save" or user.lower().startswith("/save "):
            arg_name = user.split(maxsplit=1)[1].strip() if " " in user else ""
            name = arg_name or session_name or default_ts_name()
            save_session(name, messages)
            session_name = name
            console.print(f"[dim]saved session '{name}'[/dim]\n")
            continue
        if user.lower() == "/resume" or user.lower().startswith("/resume "):
            arg_name = user.split(maxsplit=1)[1].strip() if " " in user else ""
            path = resolve_session(arg_name)
            if not path:
                console.print("[yellow]no matching session; conversation unchanged[/yellow]\n")
                continue
            new_name = os.path.splitext(os.path.basename(path))[0]
            # Switching away from a named, non-empty session shouldn't lose it.
            if session_name and len(messages) > 1:
                with contextlib.suppress(OSError):
                    save_session(session_name, messages)
            try:
                data = load_session(new_name)
                apply_session(messages, data)
            except (OSError, ValueError, KeyError, TypeError):
                console.print("[yellow]could not read session; conversation unchanged[/yellow]\n")
                continue
            session_name   = new_name
            first_user_msg = False
            console.print(f"[dim]resumed session '{new_name}' ({len(messages)} messages)[/dim]\n")
            threading.Thread(target=warm_cache, args=(list(messages),), daemon=True).start()
            continue
        if not user:
            continue

        content        = (cwd_context + user) if first_user_msg else user
        first_user_msg = False
        messages.append({"role": "user", "content": content})

        try:
            run_turn(messages, max_steps=args.max_steps)
        except urllib.error.HTTPError as e:
            body = getattr(e, "body", "") or e.read().decode(errors="replace")
            console.print(f"[red]Ollama error {e.code}: {body.strip() or e.reason}[/red]")
        except urllib.error.URLError as e:
            console.print(f"[red]cannot reach Ollama at {OLLAMA_URL}: {e}[/red]")
        except KeyboardInterrupt:
            console.print("\n[yellow]interrupted[/yellow]")
        console.print()


if __name__ == "__main__":
    main()