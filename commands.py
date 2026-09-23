import os
import re
import glob
import json
import shlex
import tempfile
import subprocess

from rich.markup import escape

import config

# --------------------------------------------------------------------------- #
# REPL helpers: custom commands, /help, /export, /editor, Tab completion
# --------------------------------------------------------------------------- #

# Built-in commands for /help and Tab completion, in the order /help lists
# them. main() implements them; this is only their one-line description.
BUILTIN_COMMANDS = [
    ("/help",         "list commands, built-in and custom"),
    ("/clear",        "reset the conversation (the cached system prompt stays)"),
    ("/undo [N]",     "take back the last turn, or the last N (files too, in a git repo)"),
    ("/history",      "list this conversation's turns, newest last, for /undo N"),
    ("/save [NAME]",  "save the conversation as a session"),
    ("/resume [NAME]", "resume a saved session (the most recent without NAME)"),
    ("/sessions",     "list saved sessions"),
    ("/export [PATH]", "write the conversation to a Markdown file"),
    ("/editor",       "write the next prompt in $VISUAL / $EDITOR"),
    ("/details",      "toggle full tool results under each call"),
    ("/thinking",     "toggle the live view of the model's reasoning"),
    ("!cmd",          "run a shell command; its output goes with your next message"),
    ("!!cmd",         "run a shell command; the output is only shown to you"),
    ("@path[#A-B]",   "attach a file (or lines A-B of it) to the message"),
    ("exit",          "quit"),
]


def builtin_names():
    return [c.split()[0] for c, _ in BUILTIN_COMMANDS if c.startswith("/")]

# ---- custom commands ------------------------------------------------------ #

def command_dirs(project_root):
    """Where custom commands live, highest priority first: the project's
    .tiny-agent/commands, then the user's."""
    return [os.path.join(project_root, ".tiny-agent", "commands"),
            os.path.join(config.CONFIG_DIR, "commands")]


def _split_frontmatter(text):
    """(meta dict, body) for a file that may open with a `---` block of
    `key: value` lines; only `description` is used."""
    meta = {}
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            for line in text[4:end].splitlines():
                key, sep, value = line.partition(":")
                if sep:
                    meta[key.strip().lower()] = value.strip()
            text = text[end + 4:].lstrip("\n")
    return meta, text


def load_custom_commands(project_root):
    """{"/name": (path, description, body)} from every command dir. A
    project command shadows a user one of the same name; neither can
    shadow a built-in."""
    found    = {}
    builtins = set(builtin_names())
    for d in reversed(command_dirs(project_root)):
        for path in sorted(glob.glob(os.path.join(d, "*.md"))):
            name = "/" + os.path.splitext(os.path.basename(path))[0].lower()
            if name in builtins:
                continue
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    meta, body = _split_frontmatter(f.read())
            except OSError:
                continue
            desc = meta.get("description") or next(
                (l.strip() for l in body.splitlines() if l.strip()), "")
            found[name] = (path, desc[:80], body)
    return found


SHELL_SUB_RE = re.compile(r"!`([^`]+)`")
POSITIONAL_RE = re.compile(r"\$([1-9])")


def expand_custom_command(body, argstr, run_shell):
    """Fill a command template: `$ARGUMENTS` is everything after the
    command name, `$1`…`$9` its shell-split words, and each !`cmd` is
    replaced by that command's output. `@path`s are left for the normal
    attachment pass. Arguments with no `$ARGUMENTS` to go into are
    appended, so `/review focus on errors` still says what to focus on.

    The shell substitutions run without confirmation: the user typed the
    command, and the template is a file they (or their project) chose to
    keep — the same trust as typing `!cmd`. Each one is printed as it runs.
    """
    try:
        words = shlex.split(argstr)
    except ValueError:
        words = argstr.split()
    used_args = "$ARGUMENTS" in body or POSITIONAL_RE.search(body)
    text = body.replace("$ARGUMENTS", argstr)
    text = POSITIONAL_RE.sub(lambda m: words[int(m.group(1)) - 1]
                             if int(m.group(1)) <= len(words) else "", text)

    def shell(m):
        cmd = m.group(1)
        config.console.print(f"[dim]running {escape(cmd)}[/dim]")
        try:
            out, code, timed_out = run_shell(cmd)
        except KeyboardInterrupt as e:          # CommandInterrupted
            out, code, timed_out = getattr(e, "output", ""), None, False
        if timed_out:
            out = (out + "\n" if out else "") + f"[timed out after {config.CMD_TIMEOUT}s]"
        elif code:
            out = (out + "\n" if out else "") + f"[exit {code}]"
        return out
    text = SHELL_SUB_RE.sub(shell, text).strip()
    if argstr and not used_args:
        text += "\n\n" + argstr
    return text

# ---- /help ---------------------------------------------------------------- #

def print_help(custom):
    w = max(len(c) for c, _ in BUILTIN_COMMANDS)
    for cmd, desc in BUILTIN_COMMANDS:
        config.console.print(f"[dim]{escape(cmd.ljust(w))}  {escape(desc)}[/dim]")
    if custom:
        config.console.print("[dim]custom commands:[/dim]")
        for name in sorted(custom):
            path, desc, _ = custom[name]
            config.console.print(f"[dim]{escape(name.ljust(w))}  {escape(desc)}  "
                                 f"({escape(path)})[/dim]")
    else:
        config.console.print("[dim]no custom commands; put NAME.md files in "
                             ".tiny-agent/commands/ or "
                             f"{escape(os.path.join(config.CONFIG_DIR, 'commands'))}/[/dim]")
    config.console.print("[dim]while a reply streams: type to queue a message, "
                         "Tab stops it to steer, Esc cancels. A trailing '\\' "
                         "continues the line.[/dim]\n")

# ---- /export -------------------------------------------------------------- #

def _fence(text):
    """A code fence longer than any backtick run inside `text`."""
    longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)


def export_markdown(messages, title=""):
    """The conversation as Markdown: prompts, answers, each tool call and
    its result. The system prompt is left out — it is the same every time."""
    out = [f"# {title or 'tiny-agent conversation'}", ""]
    for m in messages[1:]:
        role    = m.get("role")
        content = (m.get("content") or "").rstrip()
        if role == "user":
            out += ["## You", "", content, ""]
        elif role == "assistant":
            if content:
                out += ["## Assistant", "", content, ""]
            for tc in m.get("tool_calls") or []:
                fn   = tc.get("function", {})
                args = fn.get("arguments", {})
                if not isinstance(args, str):
                    args = json.dumps(args, ensure_ascii=False)
                out += [f"**→ `{fn.get('name', '?')}`** `{args}`", ""]
        elif role == "tool":
            fence = _fence(content)
            out += [f"{fence}text", content, fence, ""]
    return "\n".join(out).rstrip() + "\n"

# ---- /editor -------------------------------------------------------------- #

def edit_in_editor(initial=""):
    """Open $VISUAL / $EDITOR (vi if neither is set) on a temp file and
    return what was saved, or None if the editor couldn't be run."""
    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vi"
    fd, path = tempfile.mkstemp(prefix="tiny-agent-prompt-", suffix=".md")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(initial)
        try:
            code = subprocess.call(shlex.split(editor) + [path])
        except (OSError, ValueError) as e:
            config.console.print(f"[red]could not run editor {escape(editor)}: {escape(str(e))}[/red]")
            return None
        if code != 0:
            config.console.print(f"[yellow]{escape(editor)} exited with {code}; nothing sent[/yellow]")
            return None
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass

# ---- Tab completion ------------------------------------------------------- #

def complete(text, line_start, custom_names):
    """Candidates for readline: `@path` completes file names, a `/word` at
    the start of the line completes commands."""
    if text.startswith("@"):
        typed = text[1:]
        base  = os.path.expanduser(typed)
        out   = []
        for m in sorted(glob.glob(glob.escape(base) + "*")):
            shown = typed + m[len(base):] if m.startswith(base) else m
            out.append("@" + shown + ("/" if os.path.isdir(m) else ""))
        return out
    if text.startswith("/") and line_start:
        return [c for c in sorted(set(builtin_names()) | set(custom_names)) if c.startswith(text)]
    return []


def install_completer(readline, custom_names):
    """Tab-complete `@paths` and `/commands` at the main prompt.
    `custom_names` is called on each Tab press that completes a `/word` at
    the start of the line, so commands added while the agent runs are
    offered too."""
    # Only whitespace separates words: '/' and '@' are part of what is typed.
    readline.set_completer_delims(" \t\n")

    # readline calls this once per candidate (state 0, 1, …) for one Tab, so
    # the list is built at state 0 and indexed after — otherwise every
    # candidate would re-glob the directory and re-read every command file.
    matches = []

    def completer(text, state):
        if state == 0:
            line_start = readline.get_begidx() == 0
            names      = custom_names() if line_start and text.startswith("/") else ()
            matches[:] = complete(text, line_start, names)
        return matches[state] if state < len(matches) else None
    readline.set_completer(completer)
    if "libedit" in (readline.__doc__ or ""):
        readline.parse_and_bind("bind ^I rl_complete")
    else:
        readline.parse_and_bind("tab: complete")
