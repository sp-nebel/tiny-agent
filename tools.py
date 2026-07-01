import os
import re
import glob
import shutil
import fnmatch
import difflib
import subprocess

from rich.text import Text

import config

# --------------------------------------------------------------------------- #
# Tools (implementations)
# --------------------------------------------------------------------------- #

def confirm(msg: str) -> bool:
    if config.AUTO_YES:
        return True
    try:
        ans = config.console.input(f"[yellow]{msg}[/yellow] [y/N] ").strip().lower()
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
    config.console.print(out, end="")


# Matches read_file's "{n:5}  " line-number column (right-justified width 5,
# two spaces), so edit_file can detect it leaking into old_string.
_LINE_NUM_PREFIX_RE = re.compile(r"^\s*\d+  ", re.MULTILINE)


def _cap_output(text, max_chars=None):
    """Hard cap on a tool result's size, independent of any hit/line count
    the caller already applies. A single grep context block or a read_file
    line hitting minified/generated code can blow past those counts while
    staying well under them in item count, so this is a byte-level backstop.
    Keeps head and tail, like run_cmd's cap, since the useful part (a match,
    an error) can land at either end.
    """
    max_chars = max_chars or config.MAX_TOOL_OUTPUT_CHARS
    if len(text) <= max_chars:
        return text
    head = max_chars // 4
    tail = max_chars - head
    return (text[:head] + f"\n[… {len(text) - max_chars} chars elided …]\n" + text[-tail:])


def read_file(path, start=1, end=None):
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
    end = min(end, total, start + config.MAX_READ_LINES - 1)
    if end < start:
        return f"[invalid range: end={end} is before start={start}]"

    body = "".join(f"{n:5}  {ln}" for n, ln in enumerate(lines[start-1:end], start=start))
    if body and not body.endswith("\n"):
        body += "\n"
    body = _cap_output(body)
    if end < total:
        # Notice at both ends: small models attend poorly to the tail of a
        # long result, and an imperative is followed better than a hint.
        body = (
            f"[lines {start}-{end} of {total} - file continues]\n"
            + body
            + f"[TRUNCATED. To continue reading, call read_file with start={end+1}.]"
        )
    return body or "[empty file]"


def grep(pattern, path=".", context=0, before=0, after=0):
    # Base command differs (rg vs grep), but every flag below is accepted
    # identically by both, so the model sees one consistent interface.
    if shutil.which("rg"):
        cmd = ["rg", "-n", "--no-heading"]   # respects .gitignore by default
    else:
        cmd = ["grep", "-rn"] + [f"--exclude-dir={d}" for d in sorted(config.SKIP_DIRS)]
    if context:
        cmd += ["-C", str(int(context))]
    else:
        if before:
            cmd += ["-B", str(int(before))]
        if after:
            cmd += ["-A", str(int(after))]
    cmd += ["--", pattern, path]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=config.CMD_TIMEOUT)
    except subprocess.TimeoutExpired:
        return "[grep timed out]"
    # rg and grep agree: exit 0 = matches, 1 = no matches, ≥2 = real error
    # (bad regex, unreadable path). Don't let an error message pass as hits.
    if out.returncode > 1:
        return f"[grep error: {out.stderr.strip() or f'exit {out.returncode}'}]"
    res = out.stdout.strip()
    # With -A/-B/-C, rg and grep separate non-adjacent match groups with a
    # standalone "--" line, so counting raw lines against MAX_GREP_HITS caps
    # on far fewer real matches than the number implies (a context=3 hit is
    # 7 lines). Cap by match group instead when context is in play.
    if context or before or after:
        blocks = res.split("\n--\n") if res else []
        if len(blocks) > config.MAX_GREP_HITS:
            blocks = blocks[:config.MAX_GREP_HITS] + [f"[+{len(blocks) - config.MAX_GREP_HITS} more matches]"]
        res = "\n--\n".join(blocks)
    else:
        hits = res.splitlines()
        if len(hits) > config.MAX_GREP_HITS:
            res = "\n".join(hits[:config.MAX_GREP_HITS]) + f"\n[+{len(hits) - config.MAX_GREP_HITS} more matches]"
    return _cap_output(res) or "[no matches]"


def _glob_regex(pattern):
    """Compile a recursive glob pattern into a regex matched against a path
    relative to the search base, treating '**' as "zero or more path
    segments" the same way glob.glob(recursive=True) does — verified against
    it directly for the documented '**/*.py' / 'src/**/test_*.py' shapes.
    Each non-'**' segment goes through fnmatch.translate so '*'/'?'/'[...]'
    still behave per-segment (don't cross '/'), matching glob's own rules.
    """
    parts = pattern.split("/")
    out, i, n = [], 0, len(parts)
    while i < n:
        part = parts[i]
        if part == "**":
            while i + 1 < n and parts[i + 1] == "**":
                i += 1
            out.append(".*" if i + 1 == n else "(?:.*/)?")
        else:
            seg = fnmatch.translate(part)
            out.append(seg[len("(?s:"):-len(r")\Z")])
            if i + 1 < n and parts[i + 1] != "**":
                out.append("/")
        i += 1
    return re.compile("(?s:" + "".join(out) + r")\Z")


def find_files(pattern, path="."):
    base = path or "."
    out = []
    try:
        if "**" in pattern:
            # Walk manually and prune SKIP_DIRS during the walk rather than
            # glob-then-filter: glob.glob(recursive=True) descends into every
            # subtree first, so a node_modules or .git dir orders of magnitude
            # bigger than the rest of the repo gets fully walked regardless.
            rx = _glob_regex(pattern)
            for root, dirs, files in os.walk(base):
                dirs[:] = [d for d in dirs if d not in config.SKIP_DIRS]
                rel_root = os.path.relpath(root, base)
                for name in dirs + files:
                    rel  = name if rel_root in (".", "") else os.path.join(rel_root, name)
                    full = os.path.normpath(os.path.join(base, rel))
                    if rx.match(rel):
                        out.append(full)
        else:
            for m in glob.glob(os.path.join(base, pattern)):
                if not config.SKIP_DIRS.intersection(m.split(os.sep)):
                    out.append(m)
    except OSError as e:
        return f"[error: {e}]"

    out = [m + ("/" if os.path.isdir(m) else "") for m in sorted(set(out))]
    if len(out) > config.MAX_GLOB_HITS:
        out = out[:config.MAX_GLOB_HITS] + [f"[+{len(out) - config.MAX_GLOB_HITS} more]"]
    return _cap_output("\n".join(out)) or "[no matches]"


def list_dir(path="."):
    try:
        entries = sorted(os.listdir(path))
    except OSError as e:
        return f"[error: {e}]"
    if len(entries) > config.MAX_LIST_HITS:
        hidden  = len(entries) - config.MAX_LIST_HITS
        entries = entries[:config.MAX_LIST_HITS]
    else:
        hidden = 0
    lines = [e + ("/" if os.path.isdir(os.path.join(path, e)) else "") for e in entries]
    if hidden:
        lines.append(f"[+{hidden} more]")
    return _cap_output("\n".join(lines)) or "[empty]"


def cd(path):
    try:
        os.chdir(path)
    except FileNotFoundError:
        return f"[no such directory: {path}]"
    except NotADirectoryError:
        return f"[not a directory: {path}]"
    except OSError as e:
        return f"[error changing directory: {e}]"
    return f"[cwd: {os.getcwd()}]"


def edit_file(path, old_string, new_string, replace_all=False):
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
        # Common small-model failure: copying read_file's "   12  " line-number
        # column into old_string. Detect it and say so directly instead of the
        # generic mismatch message, since "match exactly" alone tends to make
        # the model retry the same mistake with more surrounding lines.
        stripped = _LINE_NUM_PREFIX_RE.sub("", old_string)
        if stripped != old_string and content.count(stripped) > 0:
            return ("[old_string not found - it still has read_file's line-number "
                    "prefix (e.g. '   12  '); that's display metadata, not file "
                    "content. Strip it from the start of each line and try again]")
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


def run_cmd(cmd):
    if not confirm(f"run: {cmd}"):
        return "[user declined command]"
    try:
        out = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=config.CMD_TIMEOUT
        )
    except subprocess.TimeoutExpired:
        return f"[timed out after {config.CMD_TIMEOUT}s]"
    combined = (out.stdout + out.stderr).strip()
    if len(combined) > config.MAX_CMD_CHARS:
        # Keep head AND tail: test runners and builds put the failure summary
        # at the end, and losing it makes the model re-run the command.
        head = config.MAX_CMD_CHARS // 4
        tail = config.MAX_CMD_CHARS - head
        combined = (combined[:head]
                    + f"\n[… {len(combined) - config.MAX_CMD_CHARS} chars elided …]\n"
                    + combined[-tail:])
    return combined or f"[exit {out.returncode}, no output]"


TOOLS = {
    "read_file":  read_file,
    "grep":       grep,
    "find_files": find_files,
    "list_dir":   list_dir,
    "cd":         cd,
    "edit_file":  edit_file,
    "run_cmd":    run_cmd,
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
