import os
import re
import glob
import shutil
import difflib
import subprocess

from rich.text import Text
from rich.markup import escape

import config

# --------------------------------------------------------------------------- #
# Tools (implementations)
# --------------------------------------------------------------------------- #

def confirm(msg: str):
    """Ask before a destructive action. Returns (approved, reason).

    Anything that isn't a yes or a bare no is taken as a denial *with a
    reason*, which the caller hands back to the model: "use the test runner,
    not python directly" turns a dead end into a redirect, where a plain
    refusal just invites the same call again.
    """
    if config.AUTO_YES:
        return True, ""
    try:
        # Both the message (a path or a shell command can contain brackets)
        # and the literal hint must be escaped: Rich reads a bracketed run
        # starting with a lowercase letter as a markup tag and drops an
        # unknown one silently — "[y/N/reason]" was never displayed.
        ans = config.console.input(
            f"[yellow]{escape(msg)}[/yellow] " + escape("[y/N/reason] ")).strip()
    except (EOFError, KeyboardInterrupt):
        return False, ""
    if ans.lower() in ("y", "yes"):
        return True, ""
    if ans.lower() in ("", "n", "no"):
        return False, ""
    return False, ans


def declined(what: str, reason: str) -> str:
    """Tool result for a refused action, with the user's reason when given.

    The imperative tail is deliberate: a small model reading only
    "[user declined write]" tends to re-issue the identical call.
    """
    if not reason:
        return f"[user declined {what}]"
    return (f"[user declined {what}. Their reason: {reason}. Follow it and "
            f"change your approach; do not repeat the same call.]")


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


# Matches a candidate line-number prefix loosely enough to find the number;
# _strip_line_number_prefix then verifies it against read_file's exact
# "{n:5}  " format before treating it as display metadata.
_LINE_NUM_CANDIDATE_RE = re.compile(r"^( *)(\d+)  ")


def _strip_line_number_prefix(text):
    """If every line of `text` starts with read_file's exact numbered-line
    column ("{n:5}  " - right-justified to width 5, then two spaces) and the
    numbers are consecutive, return `text` with that column removed from
    every line. Otherwise return None.

    Checking the exact padded format (not just "some digits then two
    spaces") is what keeps this from misfiring on genuine numeric-prefixed
    file content like "42  widgets sold" - real column data is very rarely
    padded to precisely width 5, and single differing lines break both the
    exact-format check and (for multi-line old_string) the consecutive-number
    check.
    """
    if not text:
        return None
    lines = text.splitlines(keepends=True)
    numbers = []
    stripped_lines = []
    for ln in lines:
        m = _LINE_NUM_CANDIDATE_RE.match(ln)
        if not m:
            return None
        n = int(m.group(2))
        expected = f"{n:5}  "
        if not ln.startswith(expected):
            return None
        numbers.append(n)
        stripped_lines.append(ln[len(expected):])
    if numbers != list(range(numbers[0], numbers[0] + len(numbers))):
        return None
    return "".join(stripped_lines)


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

    # Cap in whole-line units, not _cap_output's mid-string elision: the
    # notice below asserts lines start-end are fully present, so eliding the
    # middle of that range (as a byte-level cap would on long/minified lines)
    # would make the notice false and the continuation hint unable to
    # recover what was cut.
    rendered  = []
    size      = 0
    line_note = ""
    last_line = start - 1
    for n, ln in enumerate(lines[start-1:end], start=start):
        piece = f"{n:5}  {ln}"
        if size + len(piece) > config.MAX_TOOL_OUTPUT_CHARS:
            if not rendered:
                # Even a single line blows the budget (e.g. minified code).
                rendered.append(piece[:config.MAX_TOOL_OUTPUT_CHARS])
                line_note = f"[line {n} truncated at {config.MAX_TOOL_OUTPUT_CHARS} chars]\n"
                last_line = n
            break
        rendered.append(piece)
        size += len(piece)
        last_line = n

    body = "".join(rendered)
    if body and not body.endswith("\n"):
        body += "\n"
    body += line_note
    end = last_line

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


def find_files(pattern, path="."):
    # glob.glob(recursive=True) correctly handles '**' as zero-or-more path
    # segments (including absolute patterns and '**' embedded in `path`), and
    # never matches a leading dot — reproducing that by hand (a prior version
    # of this function did, for a pruned-walk perf win) turned out to diverge
    # from glob's semantics in several confirmed ways, so this stays on glob.
    base = path or "."
    try:
        matches = glob.glob(os.path.join(base, pattern), recursive=True)
    except OSError as e:
        return f"[error: {e}]"

    out = []
    for m in sorted(matches):
        # Drop anything living under a noise dir (any path component matches).
        if config.SKIP_DIRS.intersection(m.split(os.sep)):
            continue
        out.append(m + ("/" if os.path.isdir(m) else ""))

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
        # An existing but *empty* file is treated like a missing one: small
        # models routinely make a file first (run_cmd touch, or a create call
        # with empty new_string) and then try to fill it, and the old
        # "already exists; put the text to replace in old_string" answer left
        # them no legal move — there is nothing in an empty file to put in
        # old_string — so they escaped to shell redirection. A non-empty file
        # still refuses: silently overwriting it would be a different tool.
        try:
            nonempty = os.path.exists(path) and os.path.getsize(path) > 0
        except OSError as e:
            return f"[error reading {path}: {e}]"
        if nonempty:
            return (f"[{path} already exists and has content. Pass the exact text "
                    f"to replace in old_string; an empty old_string only creates a "
                    f"new file or fills an empty one]")
        show_diff("", new_string, path)
        ok, reason = confirm(f"create {path} ({len(new_string)} chars)?")
        if not ok:
            return declined("write", reason)
        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_string)
        except OSError as e:
            return f"[error writing {path}: {e}]"
        return f"[created {path}, {len(new_string)} chars]"

    # Read and write the raw bytes (newline=""): a default-mode round trip
    # translates every CRLF to LF, so a one-line edit silently rewrote the
    # line endings of a whole Windows-style file.
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            content = f.read()
    except FileNotFoundError:
        return f"[no such file: {path}; pass an empty old_string to create it]"
    except UnicodeDecodeError:
        return f"[binary file, cannot edit: {path}]"
    except OSError as e:
        return f"[error reading {path}: {e}]"

    # The model saw the file through read_file, which shows LF, so its
    # strings are LF. Convert them to the file's ending and match the raw
    # text, rather than normalising the file, so every byte outside the
    # replacement is left exactly as it was — a file with mixed endings
    # included. Such a file may hold the target with LF endings, so the LF
    # form is the fallback when the CRLF one doesn't match.
    old_lf = old_string.replace("\r\n", "\n")
    new_lf = new_string.replace("\r\n", "\n")
    old_string, new_string = old_lf, new_lf
    if "\r\n" in content:
        old_crlf = old_lf.replace("\n", "\r\n")
        if content.count(old_crlf) or not content.count(old_lf):
            old_string, new_string = old_crlf, new_lf.replace("\n", "\r\n")

    count = content.count(old_string)
    if count == 0:
        # Common small-model failure: copying read_file's "   12  " line-number
        # column into old_string. Detect it and say so directly instead of the
        # generic mismatch message, since "match exactly" alone tends to make
        # the model retry the same mistake with more surrounding lines.
        stripped = _strip_line_number_prefix(old_lf)
        if stripped is not None and content.replace("\r\n", "\n").count(stripped) > 0:
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
    # Diffed as LF: difflib would otherwise show a stray \r on every line.
    show_diff(content.replace("\r\n", "\n"), new_content.replace("\r\n", "\n"), path)
    ok, reason = confirm(f"edit {path} ({n} replacement{plural})?")
    if not ok:
        return declined("write", reason)
    try:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(new_content)
    except OSError as e:
        return f"[error writing {path}: {e}]"
    return f"[edited {path}: {n} replacement{plural}]"


def append_file(path, text):
    """Append to a file, creating it if missing.

    Exists because the toolset otherwise has no legal way to add text to a
    file: appending via edit_file needs the file's exact last line as an
    anchor, which small models rarely manage, so they detour through run_cmd
    (echo, heredocs, python -c) and hit shell quoting and command-line
    limits. Appending also never re-emits existing content: generated tokens
    are the slow ones (prefill is batched, generation is one at a time), so
    rewriting a file to add a paragraph costs the whole file in generation.

    Rewrites the whole file rather than opening in append mode so the diff
    shows context and the file's existing line ending is preserved. A
    separating newline is inserted when the file doesn't end with one, so
    the appended text always starts on its own line.
    """
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            raw = f.read()
    except FileNotFoundError:
        raw = None
    except UnicodeDecodeError:
        return f"[binary file, cannot append: {path}]"
    except OSError as e:
        return f"[error reading {path}: {e}]"

    eol         = "\r\n" if raw and "\r\n" in raw else "\n"
    existing    = (raw or "").replace("\r\n", "\n")
    addition    = text if (not existing or existing.endswith("\n")) else "\n" + text
    new_content = existing + addition

    show_diff(existing, new_content, path)
    if raw is None:
        ok, reason = confirm(f"create {path} ({len(text)} chars)?")
    else:
        ok, reason = confirm(f"append {len(text)} chars to {path}?")
    if not ok:
        return declined("write", reason)
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline=eol) as f:
            f.write(new_content)
    except OSError as e:
        return f"[error writing {path}: {e}]"
    if raw is None:
        return f"[created {path}, {len(text)} chars]"
    # The line count lets the model aim a follow-up read_file at the new
    # tail without first reading the whole file to find it.
    return (f"[appended {len(text)} chars to {path}; it now has "
            f"{len(new_content.splitlines())} lines]")


def run_shell(cmd):
    """Run `cmd` in a shell: (capped combined output, exit code), or (None,
    None) on timeout. Shared by run_cmd and the user's `!cmd` prompt escape,
    so both see the same cap and timeout."""
    try:
        out = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=config.CMD_TIMEOUT
        )
    except subprocess.TimeoutExpired:
        return None, None
    combined = (out.stdout + out.stderr).strip()
    # Keep head AND tail: test runners and builds put the failure summary at
    # the end, and losing it makes the model re-run the command.
    return _cap_output(combined, config.MAX_CMD_CHARS), out.returncode


def run_cmd(cmd):
    ok, reason = confirm(f"run: {cmd}")
    if not ok:
        return declined("command", reason)
    combined, code = run_shell(cmd)
    if code is None:
        return f"[timed out after {config.CMD_TIMEOUT}s]"
    if not combined:
        return f"[exit {code}, no output]"
    # A failure with output used to look exactly like a success: the code was
    # only reported when there was nothing else to show. Report it whenever
    # it is non-zero, at the end, where the shell puts it too and where test
    # runners print their summary. A clean exit adds nothing — output with
    # no exit line reads as success.
    return combined if code == 0 else f"{combined}\n[exit {code}]"


TOOLS = {
    "read_file":   read_file,
    "grep":        grep,
    "find_files":  find_files,
    "list_dir":    list_dir,
    "cd":          cd,
    "edit_file":   edit_file,
    "append_file": append_file,
    "run_cmd":     run_cmd,
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
