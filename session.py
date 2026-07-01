import os
import glob
import json
import time
import contextlib

import config

# --------------------------------------------------------------------------- #
# Session persistence (pause/resume a conversation across process restarts)
# --------------------------------------------------------------------------- #

def default_ts_name():
    return time.strftime("%Y-%m-%dT%H-%M")


def session_path(name):
    return os.path.join(config.SESSION_DIR, os.path.basename(name) + ".json")


def save_session(name, messages):
    os.makedirs(config.SESSION_DIR, exist_ok=True)
    data = {
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model":    config.MODEL,
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
    files = glob.glob(os.path.join(config.SESSION_DIR, "*.json"))
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
    messages[:] = data["messages"]
    with contextlib.suppress(OSError):
        os.chdir(data["cwd"])
    config.MODEL = model_override if model_override else data.get("model", config.MODEL)
