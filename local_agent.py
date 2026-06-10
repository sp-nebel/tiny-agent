#!/usr/bin/env python3
"""
mini-agent — a minimal local coding agent for CPU-bound Ollama models.

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
    python mini_agent.py "review the null handling in AuthService"
    python mini_agent.py            # interactive; seed a task at the prompt

Env vars:
    AGENT_MODEL   (default: gemma3:4b)   — any Ollama model with tool support
    OLLAMA_URL    (default: http://localhost:11434)

Dependency: pip install rich
Note: ensure Ollama >= 0.20.2 for reliable Gemma 4 tool-call parsing.
"""

import os
import sys
import glob
import json
import select
import shutil
import argparse
import subprocess
import contextlib
import urllib.request
import urllib.error

try:                      # POSIX-only; used to read a single cancel keypress
    import termios
    import tty
except ImportError:
    termios = tty = None

from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.text import Text

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

OLLAMA_URL  = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL       = os.environ.get("AGENT_MODEL", "gemma4:latest")

MAX_READ_LINES = 100
MAX_GREP_HITS  = 20
MAX_GLOB_HITS  = 100
MAX_CMD_CHARS  = 8000
CMD_TIMEOUT    = 120
NUM_CTX        = 32768

# History trimming: keep the N most recent tool outputs verbatim and collapse
# older ones to a stub, so the context window (and per-turn prefill) doesn't
# grow without bound. Only outputs longer than the threshold are collapsed.
KEEP_FULL_TOOL_RESULTS = 3
TRIM_MIN_CHARS         = 400

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

When the task is done, reply in Markdown with no tool call."""

# --------------------------------------------------------------------------- #
# Tool schemas  (static — also part of the cached prefix)
# --------------------------------------------------------------------------- #

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a file. Output is line-numbered. "
                "Use start/end to read a specific range rather than the whole file."
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
            "name": "grep",
            "description": "Search for a pattern in a file or directory tree. Returns matching lines with file path and line number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern":     {"type": "string",  "description": "Search pattern (regex)"},
                    "path":        {"type": "string",  "description": "File or directory to search (default '.')"},
                    "ignore_case": {"type": "boolean", "description": "Case-insensitive match (-i)"},
                    "word":        {"type": "boolean", "description": "Match whole words only (-w)"},
                    "fixed":       {"type": "boolean", "description": "Treat pattern as a literal string, not a regex (-F)"},
                    "context":     {"type": "integer", "description": "Show N lines of context before AND after each match (-C)"},
                    "before":      {"type": "integer", "description": "Show N lines before each match (-B); ignored if context is set"},
                    "after":       {"type": "integer", "description": "Show N lines after each match (-A); ignored if context is set"},
                },
                "required": ["pattern"],
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
        body += f"[showing lines {start}–{end} of {total}; read more with start={end+1}]"
    return body or "[empty file]"


def t_grep(pattern, path=".", ignore_case=False, word=False, fixed=False,
           context=0, before=0, after=0):
    # Base command differs (rg vs grep), but every flag below is accepted
    # identically by both, so the model sees one consistent interface.
    if shutil.which("rg"):
        cmd = ["rg", "-n", "--no-heading"]   # respects .gitignore by default
    else:
        cmd = ["grep", "-rn"] + [f"--exclude-dir={d}" for d in sorted(SKIP_DIRS)]
    if ignore_case:
        cmd.append("-i")
    if word:
        cmd.append("-w")
    if fixed:
        cmd.append("-F")
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


def t_edit_file(path, old_string, new_string, replace_all=False):
    # Empty old_string ⇒ create a new file (the write_file behaviour, folded in).
    if old_string == "":
        if os.path.exists(path):
            return f"[{path} already exists; put the text to replace in old_string]"
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
        combined = combined[:MAX_CMD_CHARS] + "\n[output truncated]"
    return combined or f"[exit {out.returncode}, no output]"


TOOLS = {
    "read_file":  t_read_file,
    "grep":       t_grep,
    "find_files": t_find_files,
    "list_dir":   t_list_dir,
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


def call_ollama(messages):
    """Stream a chat turn.

    Returns (content: str, tool_calls: list, cancelled: bool).
    Exactly one of content/tool_calls is meaningful: a final answer has
    content and no tool_calls; a tool-calling turn has tool_calls. Reasoning
    arrives in a separate `thinking` field; it is shown live but never
    returned, so it is discarded once the answer is rendered. cancelled is
    True if the user pressed the cancel key (Esc/q) mid-stream.
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

                if tdelta or delta:
                    live.update(_render_stream(thinking, content))

                # Tool calls appear in a dedicated field, often in a chunk
                # where content is empty.
                tcs = msg.get("tool_calls", [])
                if tcs:
                    tool_calls.extend(tcs)

                if obj.get("done"):
                    break

    return content, tool_calls, cancelled

# --------------------------------------------------------------------------- #
# Display helpers
# --------------------------------------------------------------------------- #

def fmt_args(args: dict) -> str:
    parts = []
    for k, v in args.items():
        s = repr(v)
        parts.append(f"{k}={s[:57] + '…' if len(s) > 60 else s}")
    return ", ".join(parts)


def truncate(text: str, n: int = 600) -> str:
    return text if len(text) <= n else text[:n] + f"\n… ({len(text)} chars total)"

# --------------------------------------------------------------------------- #
# Agent loop
# --------------------------------------------------------------------------- #

def trim_history(messages):
    """Collapse old tool outputs in place to conserve the context window.

    The KEEP_FULL_TOOL_RESULTS most recent tool results stay verbatim; older
    ones are replaced by a one-line stub. The model almost never needs the raw
    bytes of a file it read many steps ago, but those bytes are re-prefilled on
    every turn — so dropping them keeps both the window and per-turn prefill
    bounded.

    Idempotent by construction: a stub is shorter than TRIM_MIN_CHARS, so it is
    never re-collapsed. Each tool result is therefore edited at most once, which
    keeps the prefix stable for the KV cache after that one-time change.
    """
    tool_idxs = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    old = tool_idxs[:-KEEP_FULL_TOOL_RESULTS] if KEEP_FULL_TOOL_RESULTS else tool_idxs
    for i in old:
        m       = messages[i]
        content = m.get("content", "")
        if len(content) > TRIM_MIN_CHARS:
            nlines = content.count("\n") + 1
            m["content"] = (
                f"[{m.get('name', 'tool')} output elided — "
                f"{nlines} lines, {len(content)} chars]"
            )


def run_turn(messages, max_steps=20):
    for _ in range(max_steps):
        trim_history(messages)
        content, tool_calls, cancelled = call_ollama(messages)

        if cancelled:
            # User aborted mid-stream. Drop the partial reply (don't commit it
            # to history) and hand control back to the prompt.
            console.print("[yellow]cancelled[/yellow]")
            return

        # Build the assistant history entry. The Ollama API expects tool_calls
        # to be included in the message when present.
        assistant_msg = {"role": "assistant", "content": content}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        if not tool_calls:
            # Final answer — render as Markdown.
            console.print(Markdown(content))
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

    console.print("[yellow]hit max steps without finishing[/yellow]")

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    global AUTO_YES, MODEL

    ap = argparse.ArgumentParser(description="Minimal local coding agent over Ollama.")
    ap.add_argument("prompt",      nargs="*",      help="initial task (optional)")
    ap.add_argument("--model",     default=MODEL,  help=f"Ollama model tag (default: {MODEL})")
    ap.add_argument("--yes",       action="store_true", help="auto-approve writes and commands")
    ap.add_argument("--max-steps", type=int, default=20, help="max tool calls per task")
    args = ap.parse_args()

    MODEL    = args.model
    AUTO_YES = args.yes

    messages = [{"role": "system", "content": SYSTEM}]
    # cwd injected into the FIRST user message only — keeps the system prompt
    # byte-identical across projects so the prefix cache hits every session.
    cwd_context    = f"Working directory: {os.getcwd()}\n\n"
    first_user_msg = True

    console.print(f"[bold]mini-agent[/bold] · {MODEL} · {os.getcwd()}")
    console.print(
        "[dim]first reply is slow (system prompt + tool schemas prefill); "
        "later turns reuse the KV cache. press Esc/q to cancel a reply, "
        "'/clear' to reset context, 'exit' to quit.[/dim]\n"
    )

    initial = " ".join(args.prompt).strip()

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