import os

from rich.console import Console

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

OLLAMA_URL  = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL       = os.environ.get("AGENT_MODEL", "gemma4:12b-it-qat")

SESSION_DIR = os.path.expanduser("~/.tiny_agent_sessions")

MAX_READ_LINES = 100
MAX_GREP_HITS  = 20
MAX_GLOB_HITS  = 20
MAX_LIST_HITS  = 200
MAX_CMD_CHARS  = 8000
CMD_TIMEOUT    = 120
NUM_CTX        = 32768

# Socket timeout for the streaming /api/chat request — applies per network
# read, not to the whole generation, so a normal (if slow) CPU decode keeps
# resetting it as chunks trickle in. Only fires if the connection genuinely
# hangs with nothing arriving at all. A very slow CPU-only prefill can
# legitimately take longer than the default; raise via AGENT_STREAM_TIMEOUT.
STREAM_TIMEOUT = int(os.environ.get("AGENT_STREAM_TIMEOUT", "300"))

# Hard cap on any single tool result (grep/read_file/list_dir/find_files), so
# one call — a long grep context block, a read_file line hitting minified or
# generated code — can't dump an outsized chunk into the context window in a
# single shot. Same head+tail-keeping shape as run_cmd's existing cap.
MAX_TOOL_OUTPUT_CHARS = 8000

# History trimming: when the conversation approaches the context window,
# collapse all but the N most recent tool outputs to a one-line stub. Editing
# history busts the KV prefix cache from the edit point on, so trimming only
# happens when the window is actually under pressure — one cache bust per
# long session, not one per step. Only outputs over the threshold collapse.
KEEP_FULL_TOOL_RESULTS = 3
TRIM_MIN_CHARS         = 400
TRIM_AT_TOKENS         = int(NUM_CTX * 0.7)

# Backstop for when collapsing eligible tool outputs still isn't enough (e.g.
# bloat from long assistant/user messages, or the keep-window itself is
# oversized): hard-truncate remaining oversized messages outside a protected
# recent-message tail so the prompt can never silently exceed NUM_CTX and
# push the system prompt out the front of Ollama's context.
HARD_TRUNCATE_AT_TOKENS = int(NUM_CTX * 0.9)
KEEP_RECENT_MESSAGES    = 6

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

read_file output is line-numbered for your reference only (e.g. "   12  foo").
That number and the two spaces after it are not part of the file — never
include them in edit_file's old_string, or the match will fail.

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
                "unless replace_all is true. Do NOT include read_file's line-number "
                "column (e.g. '   12  ') in old_string - that is display metadata, "
                "not file content, and including it will make the match fail. "
                "To CREATE a new file, pass an empty old_string and the full "
                "contents in new_string. Prefer a small, uniquely-identifying "
                "old_string over a large one. Will ask the user for confirmation."
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
