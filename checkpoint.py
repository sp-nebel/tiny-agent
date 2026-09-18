import os
import shutil
import tempfile
import subprocess

import config

# --------------------------------------------------------------------------- #
# Git working-tree snapshots (backing store for /undo)
# --------------------------------------------------------------------------- #
#
# Each snapshot is a git tree object written through a *temporary* index
# (GIT_INDEX_FILE), so taking or restoring one never touches the user's real
# index, HEAD, branches or stash — a turn's snapshot is invisible to anything
# the user does with git. Leaning on git rather than an undo journal kept by
# edit_file/append_file means files changed by run_cmd (a formatter, a
# codegen step, an `rm`) are covered too.
#
# What it cannot undo, by construction: gitignored files and anything outside
# the repo (never in the tree), commits made during the turn (HEAD is left
# alone — restore only reports that it moved), and side effects that aren't
# file contents at all (installed packages, network calls). Outside a git
# repo there is no snapshot and /undo rewinds only the conversation.
#
# The tree objects are unreferenced, so a `git gc` could in principle prune
# them — but only after gc.pruneExpire (two weeks by default), far longer
# than any session keeps its undo stack.


def _git(args, root, index=None):
    """Run git in `root`; (returncode, stdout). Never raises — a missing git
    binary or a hung call reads as a failure, which callers treat as "no
    snapshot" rather than crashing the REPL."""
    env = dict(os.environ)
    if index:
        env["GIT_INDEX_FILE"] = index
    try:
        out = subprocess.run(["git"] + args, cwd=root, env=env, capture_output=True,
                             text=True, timeout=config.CMD_TIMEOUT)
    except (OSError, subprocess.SubprocessError):
        return 1, ""
    return out.returncode, out.stdout


def _repo_root(path):
    code, out = _git(["rev-parse", "--show-toplevel"], path)
    return out.strip() if code == 0 and out.strip() else None


def _head(root):
    # --verify -q: a repo with no commits yet has no HEAD; that is None, not
    # an error.
    code, out = _git(["rev-parse", "--verify", "-q", "HEAD"], root)
    return out.strip() if code == 0 else None


def _write_tree(root):
    """The working tree as it is right now (tracked + untracked, minus
    gitignored), as a tree hash; None on failure.

    The temp index starts as a copy of the real one so `add -A` can trust its
    stat cache and only hash files that actually changed — starting from an
    empty index would re-hash the whole repo on every turn.
    """
    tmp = tempfile.mkdtemp(prefix="tiny-agent-")
    try:
        index = os.path.join(tmp, "index")
        code, out = _git(["rev-parse", "--git-path", "index"], root)
        real = os.path.join(root, out.strip()) if code == 0 else ""
        # A fresh repo may have no index file yet; git creates one at the
        # temp path. An *empty* file there would be rejected as corrupt, so
        # nothing is created unless there is something to copy.
        if real and os.path.isfile(real):
            shutil.copyfile(real, index)
        if _git(["add", "-A"], root, index)[0] != 0:
            return None
        code, out = _git(["write-tree"], root, index)
        return out.strip() if code == 0 and out.strip() else None
    except OSError:
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def snapshot(path=None):
    """Snapshot the working tree of the repo containing `path` (default: the
    current directory). Returns {"root", "tree", "head"}, or None outside a
    git repo or if git fails."""
    root = _repo_root(path or os.getcwd())
    if not root:
        return None
    tree = _write_tree(root)
    if not tree:
        return None
    return {"root": root, "tree": tree, "head": _head(root)}


def _changes(root, old_tree, new_tree):
    """[(status, path)] between two trees. -z so paths with spaces, quotes or
    non-ASCII come through unquoted; --no-renames so a rename reads as a
    delete plus an add, each of which restore knows how to undo."""
    code, out = _git(["diff", "--name-status", "--no-renames", "-z",
                      old_tree, new_tree], root)
    if code != 0:
        return None
    parts = out.split("\0")
    return [(parts[i], parts[i + 1]) for i in range(0, len(parts) - 1, 2)]


def _remove_empty_parents(root, rel):
    """After deleting a file the turn created, drop directories it created
    for it too, stopping at the first non-empty one (or the repo root)."""
    d = os.path.dirname(os.path.join(root, rel))
    while os.path.abspath(d) != os.path.abspath(root):
        try:
            os.rmdir(d)          # only succeeds on an empty directory
        except OSError:
            return
        d = os.path.dirname(d)


def restore(snap):
    """Put the working tree back to `snap`. Returns (changes, head_moved):
    changes is [(status, path)] as seen from the snapshot — "M" rewritten,
    "D" recreated, "A" removed — or None if git failed and nothing was
    touched.

    Only the paths that differ are written, so the rest of the tree keeps its
    mtimes (no spurious rebuilds, no editor reload prompts). Files are
    written through a temp index loaded from the snapshot tree; the real
    index is never touched, so the user's staged changes survive — anything
    staged during the turn now just shows as a diff against the working tree.
    """
    root    = snap["root"]
    current = _write_tree(root)
    if not current:
        return None, False
    changes = _changes(root, snap["tree"], current)
    if changes is None:
        return None, False
    head_moved = _head(root) != snap["head"]
    if not changes:
        return [], head_moved

    rewrite = [p for s, p in changes if s != "A"]
    if rewrite:
        tmp = tempfile.mkdtemp(prefix="tiny-agent-")
        try:
            index = os.path.join(tmp, "index")
            if _git(["read-tree", snap["tree"]], root, index)[0] != 0:
                return None, head_moved
            if _git(["checkout-index", "-f", "--"] + rewrite, root, index)[0] != 0:
                return None, head_moved
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    for s, p in changes:
        if s == "A":
            try:
                os.remove(os.path.join(root, p))
            except OSError:
                continue
            _remove_empty_parents(root, p)
    return changes, head_moved
