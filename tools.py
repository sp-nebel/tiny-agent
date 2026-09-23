import io
import os
import re
import ast
import json
import glob
import shlex
import inspect
import tempfile
import warnings
import shutil
import signal
import difflib
import contextlib
import subprocess

from rich.text import Text
from rich.markup import escape

import config
from ui import notify

# --------------------------------------------------------------------------- #
# Tools (implementations)
# --------------------------------------------------------------------------- #

# Session-long approvals from answering `a`: "edit" for every file edit, or
# "cmd:<prefix>" for commands starting with that prefix. Process-lifetime
# only, never saved — an approval is given to this session, not the repo.
_always_allowed = set()

# How many words name a command, for the `a` answer: `git checkout main`
# approves `git checkout *`, not all of git. Longest listed prefix wins;
# anything unlisted is one word. (After OpenCode's permission/arity.ts.)
COMMAND_ARITY = {
    "git": 2, "npm": 2, "npm run": 3, "pnpm": 2, "pnpm run": 3, "yarn": 2,
    "yarn run": 3, "npx": 2, "bun": 2, "bun run": 3, "cargo": 2, "go": 2,
    "docker": 2, "docker compose": 3, "kubectl": 2, "pip": 2, "pip3": 2,
    "python -m": 3, "python3 -m": 3, "uv": 2, "uv run": 3, "poetry": 2,
    "poetry run": 3, "make": 2, "bundle": 2, "bundle exec": 3, "dotnet": 2,
    "gradle": 2, "mvn": 2, "systemctl": 2, "brew": 2, "apt": 2, "dnf": 2,
}

# Anything that can chain, substitute or redirect: with one of these in the
# command, a prefix says nothing about what runs ("git status; rm -rf x"
# starts with "git status"), so `a` is not offered and no approval matches.
_SHELL_META = re.compile(r"[;&|<>`$\n(){}]")


def command_prefix(cmd):
    """The approval prefix for `cmd` ("git checkout"), or None when the
    command can't be approved by prefix."""
    if _SHELL_META.search(cmd):
        return None
    try:
        words = shlex.split(cmd)
    except ValueError:
        return None
    if not words or "=" in words[0]:
        return None                 # FOO=1 cmd: the variable could be anything
    arity = 1
    for k in (3, 2, 1):
        if " ".join(words[:k]) in COMMAND_ARITY:
            arity = COMMAND_ARITY[" ".join(words[:k])]
            break
    return " ".join(words[:arity])


def command_allow(cmd):
    """confirm()'s `allow` for a shell command, or None."""
    prefix = command_prefix(cmd)
    if prefix is None:
        return None
    return ("cmd:" + prefix, f"commands starting with '{prefix}'")


EDIT_ALLOW = ("edit", "all file edits")


def confirm(msg: str, allow=None):
    """Ask before a destructive action. Returns (approved, reason).

    Anything that isn't a yes or a bare no is taken as a denial *with a
    reason*, which the caller hands back to the model: "use the test runner,
    not python directly" turns a dead end into a redirect, where a plain
    refusal just invites the same call again.

    `allow` is (key, description) when the action can be approved for the
    rest of the session: `a` answers yes and remembers the key, and later
    actions with that key pass without asking.
    """
    if config.AUTO_YES:
        return True, ""
    if not config.INTERACTIVE:
        # stdin was piped: there is no one to ask. Refuse, and say why in
        # terms the model can act on instead of a bare "declined".
        config.console.print(f"[yellow]refused without a terminal to confirm: "
                             f"{escape(msg)} (rerun with --yes to allow)[/yellow]")
        return False, ("there is no terminal to confirm this in this run, so "
                       "edits and commands are refused. Answer with what you "
                       "found and what you would change")
    if allow and allow[0] in _always_allowed:
        config.console.print(f"[dim]auto-approved ({escape(allow[1])} are allowed "
                             f"this session)[/dim]")
        return True, ""
    notify("waiting for your confirmation")
    hint = "[y/N/a/reason] " if allow else "[y/N/reason] "
    extra = f"[dim](a = always allow {escape(allow[1])})[/dim] " if allow else ""
    try:
        # Both the message (a path or a shell command can contain brackets)
        # and the literal hint must be escaped: Rich reads a bracketed run
        # starting with a lowercase letter as a markup tag and drops an
        # unknown one silently — "[y/N/reason]" was never displayed.
        ans = config.console.input(
            f"[yellow]{escape(msg)}[/yellow] " + extra + escape(hint)).strip()
    except (EOFError, KeyboardInterrupt):
        return False, ""
    if ans.lower() in ("y", "yes"):
        return True, ""
    if ans.lower() in ("", "n", "no"):
        return False, ""
    if allow and ans.lower() in ("a", "always"):
        _always_allowed.add(allow[0])
        config.console.print(f"[dim]{escape(allow[1])} are allowed for the rest "
                             f"of this session[/dim]")
        return True, ""
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
    head  = max_chars // 4
    tail  = max_chars - head
    saved = _save_full_output(text)
    # The path turns "the middle is gone" into something the model can act
    # on: grep the file for the failing test instead of re-running a suite
    # that took minutes.
    where = (f"; full output saved in {saved} - grep or read_file it instead "
             f"of running the command again" if saved else "")
    return (text[:head] + f"\n[… {len(text) - max_chars} chars elided{where} …]\n"
            + text[-tail:])


def _save_full_output(text):
    """Write an oversized tool result to a temp file; its path, or None.
    Never raises: a full disk only costs the pointer, not the result."""
    try:
        folder = os.path.join(tempfile.gettempdir(), "tiny-agent")
        os.makedirs(folder, exist_ok=True)
        fd, path = tempfile.mkstemp(prefix="output-", suffix=".txt", dir=folder)
        with os.fdopen(fd, "w", encoding="utf-8", errors="replace") as f:
            f.write(text)
        return path
    except OSError:
        return None


def _similar_paths(path, limit=3):
    """Existing files whose name is close to `path`'s, for a "did you mean".

    A small model that guesses a path wrong (src/util.py for lib/utils.py)
    otherwise spends a step or two on find_files before trying again. The
    walk is bounded so a huge tree can't stall the error message.
    """
    want = os.path.basename(path.rstrip("/"))
    if not want:
        return []
    by_name = {}
    seen    = 0
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in config.SKIP_DIRS and not d.startswith(".")]
        # Checked per file, not per directory: one flat directory of 100k
        # files would otherwise blow straight through the cap.
        for f in files[:config.SIMILAR_PATHS_SCAN - seen]:
            by_name.setdefault(f, []).append(os.path.normpath(os.path.join(root, f)))
        seen += len(files)
        if seen >= config.SIMILAR_PATHS_SCAN:
            break
    out = []
    for name in difflib.get_close_matches(want, list(by_name), n=limit, cutoff=0.6):
        out.extend(sorted(by_name[name]))
    return out[:limit]


def _missing(path, advice=""):
    """"[no such file]" plus the nearest real paths, then `advice`."""
    extra = []
    close = _similar_paths(path)
    if close:
        extra.append(f"Did you mean: {', '.join(close)}?")
    if advice:
        extra.append(advice)
    if not extra:
        return f"[no such file: {path}]"
    return f"[no such file: {path}. {' '.join(extra)}]"


def _decode(data):
    """Bytes → (text, is_utf8). Anything that isn't UTF-8 decodes as
    latin-1, which never fails: a NUL byte is the binary test, not a decode
    error, so latin-1 source files are shown instead of refused."""
    try:
        return data.decode("utf-8"), True
    except UnicodeDecodeError:
        return data.decode("latin-1"), False


def read_file(path, start=1, end=None):
    if os.path.isdir(path):
        # A directory is a common wrong guess; answering with its listing
        # saves the step the model would otherwise spend on list_dir.
        return f"[{path} is a directory, not a file. Its entries:]\n" + list_dir(path)
    try:
        with open(path, "rb") as f:
            data = f.read()
    except FileNotFoundError:
        return _missing(path)
    except OSError as e:
        return f"[error reading {path}: {e}]"
    if b"\0" in data[:4096]:
        return f"[binary file, cannot display: {path}]"
    text, utf8 = _decode(data)
    enc_note = "" if utf8 else (f"[{path} is not UTF-8; shown decoded as latin-1. "
                                f"edit_file and append_file cannot change it]\n")
    # Universal newlines, as text-mode open() gave before: \r\n and a lone
    # \r both end a line, and nothing else does (str.splitlines would also
    # split on form feeds and \u2028, shifting every line number after one).
    lines = io.StringIO(text.replace("\r\n", "\n").replace("\r", "\n")).readlines()

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
    elif body:
        # Said outright, so the model stops re-reading past the end to check.
        body += f"[end of file, {total} line{'s' if total != 1 else ''}]"
    return (enc_note + body) if body else "[empty file]"


_GREP_LINENO_RE = re.compile(r"(\d+)([:-])")


def _grep_records(raw):
    """Parse `--null` grep/rg output into match groups.

    Each line is "path\\0N:text" (a match) or "path\\0N-text" (context); a
    bare "--" separates non-adjacent groups. The NUL is what makes this
    parseable at all: with plain "path:N:text" a path or a context line
    containing ':' or '-' is ambiguous. Returns a list of groups, each a list
    of (path, lineno, is_match, text).
    """
    groups, cur = [], []
    for line in raw.split("\n"):
        if line == "--":
            if cur:
                groups.append(cur)
            cur = []
            continue
        path, sep, rest = line.partition("\0")
        if not sep:
            continue
        m = _GREP_LINENO_RE.match(rest)
        if not m:
            continue
        cur.append((path, int(m.group(1)), m.group(2) == ":", rest[m.end():]))
    if cur:
        groups.append(cur)
    return groups


def grep(pattern, path=".", context=0, before=0, after=0, include=None):
    # Base command differs (rg vs grep), but every flag below is accepted
    # identically by both, so the model sees one consistent interface. -H
    # and --null make the output shape the same too, whatever `path` is.
    if shutil.which("rg"):
        cmd = ["rg", "-n", "-H", "--null", "--no-heading"]   # respects .gitignore
        if include:
            cmd += ["-g", include]
    else:
        # -E: rg's regex dialect is extended; basic mode would read "a|b"
        # or "(x)" literally under grep and as a regex under rg.
        cmd = (["grep", "-rnHIZE"]
               + [f"--exclude-dir={d}" for d in sorted(config.SKIP_DIRS)])
        if include:
            cmd += [f"--include={include}"]
    if context:
        cmd += ["-C", str(int(context))]
    else:
        if before:
            cmd += ["-B", str(int(before))]
        if after:
            cmd += ["-A", str(int(after))]
    cmd += ["--", pattern, path]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                             timeout=config.CMD_TIMEOUT)
    except subprocess.TimeoutExpired:
        return "[grep timed out; narrow the path or the pattern]"
    # rg and grep agree: exit 0 = matches, 1 = no matches, ≥2 = real error
    # (bad regex, unreadable path). Don't let an error message pass as hits.
    # GNU grep also exits 2 when *some* files were unreadable; keep its hits.
    if out.returncode > 1 and not out.stdout.strip():
        return f"[grep error: {out.stderr.strip() or f'exit {out.returncode}'}]"

    # Cap on real matches, keeping whole groups: with context a hit is
    # several lines, so a raw line count caps on far fewer matches than
    # MAX_GREP_HITS implies. Always keep the first group, however big.
    ctx    = bool(context or before or after)
    groups = _grep_records(out.stdout)
    if not ctx:
        # Adjacent hits come back as one run with no "--" between them;
        # without context every hit stands alone.
        groups = [[r] for g in groups for r in g]
    kept, shown, total = [], 0, 0
    for g in groups:
        n = sum(1 for r in g if r[2])
        total += n
        if not kept or shown + n <= config.MAX_GREP_HITS:
            kept.append(g)
            shown += n

    # Grouped by file, each path printed once: repeating a long path on
    # every hit spent most of the result on it. Lines are cut at
    # GREP_MAX_LINE_CHARS so one minified line can't use up the whole cap.
    lines, cur_path = [], None
    for gi, g in enumerate(kept):
        if ctx and gi and g[0][0] == cur_path:
            lines.append("--")
        for p, n, is_match, text in g:
            if p != cur_path:
                if lines:
                    lines.append("")
                lines.append(p)
                cur_path = p
            if len(text) > config.GREP_MAX_LINE_CHARS:
                text = text[:config.GREP_MAX_LINE_CHARS] + " [line cut]"
            lines.append(f"{n}{':' if is_match else '-'}{text}")
    if total > shown:
        lines.append(f"[+{total - shown} more matches; narrow the path, the "
                     f"pattern, or use include]")
    return _cap_output("\n".join(lines)) or "[no matches]"


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


# Typographic characters a model tends to "correct" when it copies code:
# mapped to ASCII on both sides for the last fuzzy pass only.
_ASCII_MAP = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-",
    "\u2014": "-", "\u2212": "-", "\u00a0": " ",
})

# Tried in order, only after an exact match failed; each compares whole
# lines. Deliberately no edit-distance matching (OpenCode's Levenshtein and
# block-anchor matchers): under --yes a loose match edits the wrong code
# with nobody looking.
_FUZZY_PASSES = [
    ("ignoring trailing whitespace",               lambda l: l.rstrip()),
    ("ignoring indentation",                       lambda l: l.strip()),
    ("ignoring indentation and quote/dash style",  lambda l: l.translate(_ASCII_MAP).strip()),
]


def _leading_ws(line):
    return line[:len(line) - len(line.lstrip(" \t"))]


def _fuzzy_match(content, old, new):
    """Find `old` in `content` by whole lines, more loosely than exactly.

    Returns (offset, matched_text, new_text, label) for a unique match,
    ("ambiguous", label, count) when a pass matches more than once, or None.
    `content` is the raw file (lines may end in \r); `old`/`new` are LF.

    The matched region is the file's own text, and the replacement is
    spliced in at its offset rather than via str.replace, which could hit an
    earlier substring ("  foo()" inside "    foo()"). When the indentation
    differed, new_string is shifted by the same amount, so a model that
    dropped a level of indent from both strings still writes correctly
    indented code.
    """
    old_lines = old.split("\n")
    trailing  = old.endswith("\n")
    if trailing:
        old_lines = old_lines[:-1]
    if not any(l.strip() for l in old_lines):
        return None
    file_lines = content.split("\n")
    n = len(old_lines)
    for label, norm in _FUZZY_PASSES:
        target = [norm(l) for l in old_lines]
        # Once per line, not once per window it falls in.
        normed = [norm(l) for l in file_lines]
        hits = [i for i in range(len(file_lines) - n + 1)
                if normed[i] == target[0] and normed[i:i + n] == target]
        if not hits:
            continue
        if len(hits) > 1:
            return ("ambiguous", label, len(hits))
        i       = hits[0]
        offset  = sum(len(l) + 1 for l in file_lines[:i])
        matched = "\n".join(file_lines[i:i + n])
        if trailing and i + n < len(file_lines):
            matched += "\n"
        elif matched.endswith("\r"):
            matched = matched[:-1]     # the line ending isn't part of old_string
        # Same line count by construction, so this only trips on whitespace
        # runs far longer than the model wrote — not the region it meant.
        if len(matched) > 2 * len(old) + 200:
            return None
        k        = next(j for j, l in enumerate(old_lines) if l.strip())
        old_ind  = _leading_ws(old_lines[k])
        file_ind = _leading_ws(file_lines[i + k])
        if old_ind != file_ind:
            new = "\n".join(file_ind + l[len(old_ind):] if l.strip() and l.startswith(old_ind) else l
                            for l in new.split("\n"))
        if "\r\n" in matched or file_lines[i].endswith("\r"):
            new = new.replace("\n", "\r\n")
        return offset, matched, new, label
    return None


def _syntax_error(path, text):
    """First syntax error in `text` as one line, or None. Built in for .py
    and .json; other extensions via config.SYNTAX_CHECK_CMDS, run on the
    file as written (so `text` is unused there)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".py":
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")    # invalid-escape noise on stderr
                ast.parse(text, filename=path)
        except SyntaxError as e:
            return f"line {e.lineno}: {e.msg}"
        except ValueError as e:                    # NUL bytes
            return str(e)
        return None
    if ext == ".json":
        try:
            json.loads(text)
        except ValueError as e:
            return f"line {getattr(e, 'lineno', '?')}: {getattr(e, 'msg', e)}"
        return None
    cmd = config.SYNTAX_CHECK_CMDS.get(ext)
    if not cmd:
        return None
    try:
        out = subprocess.run(cmd.replace("{path}", shlex.quote(path)), shell=True,
                             capture_output=True, text=True, errors="replace",
                             timeout=config.SYNTAX_CHECK_TIMEOUT)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode == 0:
        return None
    first = next((l.strip() for l in (out.stdout + out.stderr).splitlines() if l.strip()), "")
    return first[:300] or f"exit {out.returncode}"


def _syntax_note(path, before, after):
    """Tool-result suffix when this write broke the file's syntax.

    Only for errors the write *introduced*: a file that already failed to
    parse (a JSON file with comments, say, or one the model is midway
    through fixing) would otherwise get the warning on every edit. The
    custom-command check can't see the before state and always reports.
    """
    err = _syntax_error(path, after)
    if not err:
        return ""
    ext = os.path.splitext(path)[1].lower()
    if before is not None and ext in (".py", ".json") and _syntax_error(path, before):
        return ""
    return (f"\n[warning: {path} now has a syntax error, {err}. "
            f"Fix it before moving on.]")


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
        ok, reason = confirm(f"create {path} ({len(new_string)} chars)?", EDIT_ALLOW)
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
        return (f"[created {path}, {len(new_string)} chars]"
                + _syntax_note(path, None, new_string))

    # Read and write the raw bytes (newline=""): a default-mode round trip
    # translates every CRLF to LF, so a one-line edit silently rewrote the
    # line endings of a whole Windows-style file.
    if old_string == new_string:
        return ("[old_string and new_string are identical, so this edit would "
                "change nothing. Put the changed text in new_string]")
    # Strict UTF-8, unlike read_file: a lossy decode would rewrite bytes the
    # model never saw when the file is written back.
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            content = f.read()
    except FileNotFoundError:
        return _missing(path, "To create it, pass an empty old_string.")
    except IsADirectoryError:
        return f"[{path} is a directory, not a file]"
    except UnicodeDecodeError:
        return f"[{path} is binary or not UTF-8 text, so edit_file cannot change it]"
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
    note  = ""
    if count == 0:
        # Common small-model failure: copying read_file's "   12  " line-number
        # column into old_string. Detect it and say so directly instead of the
        # generic mismatch message, since "match exactly" alone tends to make
        # the model retry the same mistake with more surrounding lines. Checked
        # before the fuzzy pass on purpose: this mistake is taught, not
        # silently forgiven, or the model keeps making it.
        stripped = _strip_line_number_prefix(old_lf)
        if stripped is not None and content.replace("\r\n", "\n").count(stripped) > 0:
            return ("[old_string not found - it still has read_file's line-number "
                    "prefix (e.g. '   12  '); that's display metadata, not file "
                    "content. Strip it from the start of each line and try again]")
        fuzzy = _fuzzy_match(content, old_lf, new_lf)
        if fuzzy is None:
            return "[old_string not found; it must match the file exactly, whitespace included]"
        if fuzzy[0] == "ambiguous":
            return (f"[old_string not found exactly; {fuzzy[1]} it matches "
                    f"{fuzzy[2]} places. Copy the exact text from read_file and "
                    f"add surrounding lines to make it unique]")
        offset, matched, new_text, label = fuzzy
        new_content = content[:offset] + new_text + content[offset + len(matched):]
        n, note     = 1, f" (matched {label})"
        if replace_all:
            # A loose match is only trusted when unique, so "all" is one.
            note = f" (matched {label}; replace_all needs an exact match, so only this one was replaced)"
    elif count > 1 and not replace_all:
        return (f"[old_string matches {count} times; add surrounding context to "
                f"make it unique, or set replace_all=true]")
    else:
        n           = count if replace_all else 1
        new_content = content.replace(old_string, new_string, -1 if replace_all else 1)
    plural = "s" if n != 1 else ""
    # Diffed as LF: difflib would otherwise show a stray \r on every line.
    show_diff(content.replace("\r\n", "\n"), new_content.replace("\r\n", "\n"), path)
    ok, reason = confirm(f"edit {path} ({n} replacement{plural})?", EDIT_ALLOW)
    if not ok:
        return declined("write", reason)
    try:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(new_content)
    except OSError as e:
        return f"[error writing {path}: {e}]"
    return (f"[edited {path}: {n} replacement{plural}{note}]"
            + _syntax_note(path, content, new_content))


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
    except IsADirectoryError:
        return f"[{path} is a directory, not a file]"
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
        ok, reason = confirm(f"create {path} ({len(text)} chars)?", EDIT_ALLOW)
    else:
        ok, reason = confirm(f"append {len(text)} chars to {path}?", EDIT_ALLOW)
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
    check = _syntax_note(path, existing if raw is not None else None, new_content)
    if raw is None:
        return f"[created {path}, {len(text)} chars]" + check
    # The line count lets the model aim a follow-up read_file at the new
    # tail without first reading the whole file to find it.
    return (f"[appended {len(text)} chars to {path}; it now has "
            f"{len(new_content.splitlines())} lines]" + check)


class CommandInterrupted(KeyboardInterrupt):
    """Ctrl-C while run_shell's command ran, carrying what it printed so far.

    Still a KeyboardInterrupt, so the turn aborts exactly as before; the
    output rides along so run_turn can tell the model the command *did* run
    (and may have changed files) instead of claiming it never started.
    """

    def __init__(self, output):
        super().__init__()
        self.output = output


def _kill_group(proc):
    # start_new_session made the shell a process-group leader, so its pid is
    # the group id; the group holds every child it started, `cmd &` included.
    with contextlib.suppress(OSError):
        os.killpg(proc.pid, signal.SIGKILL)


def _collect_after_kill(proc):
    """Drain what the killed group printed. Bounded: a grandchild that left
    the group (setsid, a daemonizing server) can still hold the pipe open,
    and waiting on it is the hang this whole dance exists to avoid."""
    try:
        out, _ = proc.communicate(timeout=5)
    except subprocess.TimeoutExpired as e:
        out = e.output or ""
        if isinstance(out, bytes):
            out = out.decode("utf-8", errors="replace")
        for f in (proc.stdout, proc.stdin):
            with contextlib.suppress(OSError, AttributeError):
                f.close()
    return out or ""


def run_shell(cmd, timeout=None):
    """Run `cmd` in a shell: (capped combined output, exit code, timed_out).
    Shared by run_cmd and the user's `!cmd` prompt escape, so both see the
    same cap and timeout.

    Popen in its own session rather than subprocess.run: run's timeout kills
    only the shell, and a grandchild still holding the pipe (a dev server,
    `cmd &`) kept communicate() blocked long past the timeout. Killing the
    whole process group closes every copy of the pipe, and the output
    gathered before the kill is returned instead of dropped — it usually
    says why the command hung. stdin is /dev/null so a command waiting for
    input gets EOF at once instead of sitting there until the timeout.
    """
    timeout = timeout or config.CMD_TIMEOUT
    try:
        proc = subprocess.Popen(
            cmd, shell=True, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, errors="replace",
            start_new_session=True,
        )
    except OSError as e:
        return f"[could not start the shell: {e}]", 127, False
    timed_out = False
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        out       = _collect_after_kill(proc)
        timed_out = True
    except KeyboardInterrupt:
        # In its own session the command never saw the terminal's SIGINT,
        # so stop it here, or it would outlive the turn it belonged to.
        _kill_group(proc)
        raise CommandInterrupted(_cap_output(_collect_after_kill(proc).strip(),
                                             config.MAX_CMD_CHARS))
    # Keep head AND tail: test runners and builds put the failure summary at
    # the end, and losing it makes the model re-run the command.
    return _cap_output(out.strip(), config.MAX_CMD_CHARS), proc.returncode, timed_out


def run_cmd(cmd, timeout=None):
    ok, reason = confirm(f"run: {cmd}", command_allow(cmd))
    if not ok:
        return declined("command", reason)
    # Clamped: under --yes the model's number is the only limit there is.
    limit = config.CMD_TIMEOUT
    if timeout:
        limit = max(1, min(int(timeout), config.MAX_CMD_TIMEOUT))
    combined, code, timed_out = run_shell(cmd, limit)
    if timed_out:
        # What happened, then the two fixes a small model can act on; a bare
        # "[timed out]" made it rerun the identical command.
        return ((combined + "\n" if combined else "")
                + f"[timed out after {limit}s and was killed. If it just needs "
                f"longer, run it again with timeout=SECONDS (at most "
                f"{config.MAX_CMD_TIMEOUT}). If it waits for input or never exits "
                f"(a server, a watcher), do not run it with run_cmd.]")
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


# Names small models reach for from other agents' toolsets. Mapping them
# costs nothing and saves a round-trip on "[unknown tool]".
TOOL_ALIASES = {
    "bash": "run_cmd", "shell": "run_cmd", "sh": "run_cmd", "run": "run_cmd",
    "exec": "run_cmd", "execute": "run_cmd", "run_command": "run_cmd",
    "terminal": "run_cmd",
    "read": "read_file", "cat": "read_file", "view": "read_file",
    "open": "read_file", "open_file": "read_file", "view_file": "read_file",
    "write": "edit_file", "write_file": "edit_file", "create_file": "edit_file",
    "edit": "edit_file", "str_replace": "edit_file", "replace": "edit_file",
    "append": "append_file",
    "search": "grep", "rg": "grep", "search_files": "grep",
    "ls": "list_dir", "list": "list_dir", "list_files": "list_dir",
    "list_directory": "list_dir",
    "glob": "find_files", "find": "find_files", "find_file": "find_files",
    "chdir": "cd",
}

# Tools that create a file whole: mapped onto edit_file with an empty
# old_string, which is edit_file's create mode.
_CREATE_ALIASES = {"write", "write_file", "create_file"}

# Argument names from the same toolsets, tried in order against the real
# tool's parameters; only applied when the alias isn't itself a parameter
# and the real one wasn't passed.
ARG_ALIASES = {
    "file_path": ("path",), "filepath": ("path",), "file": ("path",),
    "filename": ("path",), "directory": ("path",), "dir": ("path",),
    "command": ("cmd",), "old_str": ("old_string",), "new_str": ("new_string",),
    "old": ("old_string",), "new": ("new_string",),
    "content": ("new_string", "text"), "contents": ("new_string", "text"),
    "start_line": ("start",), "end_line": ("end",), "query": ("pattern",),
    "regex": ("pattern",), "glob": ("pattern", "include"),
}


def _bare_name(name):
    """The model's tool name without case or a "functions."-style prefix."""
    n = (name or "").strip().lower()
    for prefix in ("functions.", "tools.", "default_api."):
        if n.startswith(prefix):
            n = n[len(prefix):]
    return n


def tool_name(name):
    """The real tool a model-supplied name refers to, or None. Tolerates
    case, a "functions." prefix and the aliases above."""
    n = _bare_name(name)
    n = TOOL_ALIASES.get(n, n)
    return n if n in TOOLS else None


def _repair_args(raw_name, real, fn, args):
    params = inspect.signature(fn).parameters
    fixed  = dict(args)
    for key in list(fixed):
        if key in params:
            continue
        for target in ARG_ALIASES.get(key.lower(), ()):
            if target in params and target not in fixed:
                fixed[target] = fixed.pop(key)
                break
    if real == "edit_file" and raw_name in _CREATE_ALIASES:
        fixed.setdefault("old_string", "")
    return fixed


def _bad_args(name, fn, args):
    """Name the missing or unknown parameters, and list the real ones. A
    raw TypeError ("missing 1 required positional argument") is Python's
    wording about Python; the model needs the tool's parameter names."""
    params   = inspect.signature(fn).parameters
    required = [p for p, v in params.items() if v.default is inspect.Parameter.empty]
    missing  = [p for p in required if p not in args]
    unknown  = [k for k in args if k not in params]
    if not (missing or unknown):
        return None
    problems = []
    if missing:
        problems.append("missing required " + ", ".join(repr(p) for p in missing))
    if unknown:
        problems.append("unknown " + ", ".join(repr(k) for k in unknown))
    return (f"[bad args for {name}: {'; '.join(problems)}. Its parameters are: "
            f"{', '.join(params)}]")


def normalize_call(name, args):
    """(real tool name or None, repaired args) — what dispatch will run, so
    the display can show it too."""
    args = args if isinstance(args, dict) else {}
    real = tool_name(name)
    if real is None:
        return None, args
    return real, _repair_args(_bare_name(name), real, TOOLS[real], args)


def dispatch(name, args):
    real, args = normalize_call(name, args)
    if real is None:
        return f"[unknown tool: {name}. The tools are: {', '.join(TOOLS)}]"
    fn   = TOOLS[real]
    bad  = _bad_args(real, fn, args)
    if bad:
        return bad
    try:
        return fn(**args)
    except TypeError as e:
        return f"[bad args for {real}: {e}]"
    except Exception as e:
        return f"[tool error: {e}]"
