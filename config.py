import os
import json

from rich.console import Console

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

OLLAMA_URL  = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL       = os.environ.get("AGENT_MODEL", "gemma4:12b-it-qat")

SESSION_DIR = os.path.expanduser("~/.tiny_agent_sessions")

MAX_READ_LINES = 100
MAX_GREP_HITS  = 20
GREP_MAX_LINE_CHARS = 300
MAX_GLOB_HITS  = 20
MAX_LIST_HITS  = 200
# Files looked at for a "did you mean" when a path doesn't exist.
SIMILAR_PATHS_SCAN = 5000
MAX_CMD_CHARS  = 8000
CMD_TIMEOUT    = 120
# Ceiling on run_cmd's per-call timeout parameter — under --yes the model's
# own number would otherwise be the only limit.
MAX_CMD_TIMEOUT = 600

# Ollama's host-memory prompt-cache saves (and, for SWA models, per-checkpoint
# state) scale with resident context tokens. On memory-constrained boxes a
# long session can push the runner OOM at the moment it saves or swaps slot
# prompts. Lower via AGENT_NUM_CTX if that happens.
NUM_CTX = int(os.environ.get("AGENT_NUM_CTX", "24576"))

# Socket timeout for the streaming /api/chat request — applies per network
# read, not to the whole generation, so a normal (if slow) CPU decode keeps
# resetting it as chunks trickle in. Only fires if the connection genuinely
# hangs with nothing arriving at all. A very slow CPU-only prefill can
# legitimately take longer than the default; raise via AGENT_STREAM_TIMEOUT.
STREAM_TIMEOUT = int(os.environ.get("AGENT_STREAM_TIMEOUT", "300"))

# The one *expected* multi-minute silence: the first call after trim_history
# edits history. On an SWA model the edit forces Ollama to re-prefill the
# whole trimmed prompt from token 0 before the first byte arrives — on CPU
# that routinely exceeds STREAM_TIMEOUT with nothing wrong (a fixed 900s
# constant once killed such a prefill at 92% done). run_turn sizes that one
# call's timeout as tokens/rate × factor, using the prefill rate measured
# from live stats when available and this deliberately conservative fallback
# before one exists (the reference box measures ~14–16 tok/s). The factor
# absorbs the ~4 chars/token estimate erring low and the rate degrading as
# the window fills.
PREFILL_TPS_FALLBACK     = 8.0
POST_TRIM_TIMEOUT_FACTOR = 2.0

# Retries for a request that failed before anything streamed: 429/5xx, or a
# refused/reset connection (usually Ollama mid-restart, e.g. systemd bouncing
# it back up after an OOM kill, which takes a few seconds). Exponential
# backoff from RETRY_BASE_DELAY, capped at RETRY_MAX_DELAY, with jitter; a
# context overflow is never retried. The payload resent is byte-identical,
# so this has no cache impact beyond what the restart itself already cost.
RETRY_MAX        = 5
RETRY_BASE_DELAY = 2
RETRY_MAX_DELAY  = 30

# Hard cap on any single tool result (grep/read_file/list_dir/find_files), so
# one call — a long grep context block, a read_file line hitting minified or
# generated code — can't dump an outsized chunk into the context window in a
# single shot. Same head+tail-keeping shape as run_cmd's existing cap.
MAX_TOOL_OUTPUT_CHARS = 8000

# History trimming: collapse old tool outputs to one-line stubs. This happens
# at two moments only, both already-paid cache busts (editing history
# invalidates the KV prefix cache from the edit point on, so it must never
# happen per step): at every turn boundary for the finished turn's outputs
# (piggybacked on drop_thinking's bust), keeping the N most recent verbatim
# for follow-up tasks — after the turn its thinking is stripped, so those
# outputs are the only remaining record; and mid-turn in one lazy pass when
# the estimate crosses TRIM_AT_TOKENS, where every already-processed output
# collapses (the live turn's thinking holds their distilled facts) and only
# the trailing results the model hasn't seen yet stay verbatim. Only outputs
# over the threshold collapse.
KEEP_FULL_TOOL_RESULTS = 3
TRIM_MIN_CHARS         = 400
TRIM_AT_TOKENS         = int(NUM_CTX * 0.7)

# Fallback floor for the mid-turn pass: thinking is preferentially kept (it is
# the model's distilled record of the outputs stubbed away), but when stubbing
# alone can't get the estimate back under TRIM_AT_TOKENS — a marathon turn
# whose thinking is itself the bulk — all but this many thinking fields are
# shed too. Without the fallback the estimate would sit above the trigger
# forever and every later step would stub its one new output: a cache bust and
# a full re-prefill per step. The cost of any bust is the post-trim prompt
# size, since SWA models reprocess from token 0 after an edit; the
# hard-truncate backstop escalates further, to only the single most recent.
TRIM_KEEP_THINKING = 4

# Backstop for when collapsing eligible tool outputs still isn't enough (e.g.
# bloat from long assistant/user messages, or the keep-window itself is
# oversized): hard-truncate remaining oversized messages outside a protected
# recent-message tail so the prompt can never silently exceed NUM_CTX and
# push the system prompt out the front of Ollama's context.
HARD_TRUNCATE_AT_TOKENS = int(NUM_CTX * 0.9)
KEEP_RECENT_MESSAGES    = 6

# Sentinel prefixing every compacted/truncated message, checked instead of a
# length heuristic: a hard-truncated message keeps TRIM_MIN_CHARS of content,
# so "a stub is short → never re-collapsed" wouldn't recognise it and every
# later pass would re-edit it, destroying the original "was N chars" figure.
TRIM_PREFIX        = "[«compacted» "

# Syntax check after edit_file/append_file writes a file, reported as one
# line in the tool result — a light stand-in for an LSP. .py (ast) and
# .json are built in; more extensions via a JSON object of shell commands
# with {path} as the placeholder, e.g.
#   AGENT_SYNTAX_CHECKS='{".js": "node --check {path}", ".sh": "bash -n {path}"}'
try:
    SYNTAX_CHECK_CMDS = json.loads(os.environ.get("AGENT_SYNTAX_CHECKS", "{}"))
except ValueError:
    SYNTAX_CHECK_CMDS = {}
SYNTAX_CHECK_TIMEOUT = 20

# Instructions files put into the first user message: the global one, then
# the first of INSTRUCTION_FILES found walking up from the cwd to the git
# root. Each is cut at INSTRUCTIONS_MAX_CHARS — it rides in every request of
# the conversation.
CONFIG_DIR = os.path.join(os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config"),
                          "tiny-agent")
GLOBAL_INSTRUCTIONS    = os.path.join(CONFIG_DIR, "AGENTS.md")
INSTRUCTION_FILES      = ("AGENTS.md", "CLAUDE.md")
INSTRUCTIONS_MAX_CHARS = 3000

# Directories never worth walking in a glob.
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv",
             ".mypy_cache", ".pytest_cache", ".ruff_cache"}

# The same tool call returning the same result this many times in a row in
# one turn gets a "stop repeating this" note appended to its result.
REPEAT_CALL_LIMIT = 3

AUTO_YES = False
# False when stdin is piped: confirm() can't ask, so it refuses (unless
# AUTO_YES). ANSWER_TO_STDOUT is set when stdout is piped too: only the
# final answer goes there, and the console writes to stderr.
INTERACTIVE      = True
ANSWER_TO_STDOUT = False
# Cap on text piped in on stdin; the rest is saved to a file like any
# oversized tool output, and the notice gives its path.
MAX_PIPED_CHARS  = 16000
# Ask the model to emit reasoning in a separate `thinking` field. Streamed
# live as feedback on slow CPU runs, then discarded once the answer lands.
# Auto-disabled at runtime if the model doesn't support thinking.
THINK = os.environ.get("AGENT_THINK", "1") not in ("0", "false", "")

# Bell + desktop notification when a turn ends or a confirmation waits,
# once the turn has run at least NOTIFY_AFTER seconds. AGENT_NOTIFY=0 turns
# it off.
NOTIFY       = os.environ.get("AGENT_NOTIFY", "1") not in ("0", "false", "")
NOTIFY_AFTER = 20

# Display toggles, flipped by /details and /thinking. Neither changes what
# the model is sent.
SHOW_DETAILS  = False   # full tool results under each call, not just failures
SHOW_THINKING = True    # the live reasoning view while a reply streams

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
            "description": "Search for a regex pattern in a file or directory tree. Returns matching lines grouped by file, each as line number and text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string",  "description": "Search pattern (regex)"},
                    "path":    {"type": "string",  "description": "File or directory to search (default '.')"},
                    "include": {"type": "string",  "description": "Only search files whose name matches this glob, e.g. '*.py'"},
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
                "contents in new_string in one call. To add text at the END of a "
                "file use append_file instead. Prefer a small, uniquely-identifying "
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
            "name": "append_file",
            "description": (
                "Append text to the end of a file, creating the file if it does "
                "not exist. "
                "A newline is inserted first if the file does not already end "
                "with one. Will ask the user for confirmation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "text": {"type": "string", "description": "Text to append, exactly as it should appear in the file"},
                },
                "required": ["path", "text"],
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
                    "cmd":     {"type": "string",  "description": "Shell command to run"},
                    "timeout": {"type": "integer", "description": "Seconds before the command is killed (default 120, max 600)"},
                },
                "required": ["cmd"],
            },
        },
    },
]
