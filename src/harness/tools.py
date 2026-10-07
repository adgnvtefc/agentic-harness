"""Tools the agent can call: Python implementations plus the JSON schemas the model sees."""

import json
import subprocess
from pathlib import Path

# All file and shell access is confined to this directory for Phase 1.
WORKSPACE = Path(__file__).resolve().parents[2] / "workspace"

# ~10k chars is roughly 2.5k tokens: enough to be useful, small enough that one
# tool result can't eat a 32k context window.
MAX_OUTPUT_CHARS = 10_000


def _truncate(text: str) -> str:
    """Keep the head and tail of long output; errors and summaries usually live at the end."""
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    half = MAX_OUTPUT_CHARS // 2
    omitted = len(text) - MAX_OUTPUT_CHARS
    return f"{text[:half]}\n\n... [{omitted} characters omitted] ...\n\n{text[-half:]}"


def _resolve(path: str) -> Path:
    """Turn a model-supplied path into an absolute path inside WORKSPACE, or raise.

    resolve() collapses ".." and follows symlinks *before* the check, so
    "../../.ssh/id_rsa" and "link_to_home/.ssh" are both caught.
    """
    full = (WORKSPACE / path).resolve()
    if not full.is_relative_to(WORKSPACE):
        raise PermissionError(f"Path {path!r} is outside the workspace. Use paths relative to the workspace.")
    return full


def read_file(path: str) -> str:
    """Return the contents of `path` (relative to WORKSPACE).

    Refuse paths that resolve outside WORKSPACE (think: "../../.ssh/id_rsa").
    """
    full = _resolve(path)
    try:
        text = full.read_text()
    except UnicodeDecodeError:
        return f"{path} is a binary file and can't be shown as text."
    if not text:
        return f"{path} is empty."
    return _truncate(text)


def write_file(path: str, content: str) -> str:
    """Write `content` to `path` (relative to WORKSPACE), creating parent dirs.

    Same path rule as read_file. Return a short confirmation the model can read.
    """
    full = _resolve(path)
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content)
    return f"Wrote {len(content)} characters to {path}."


def bash(command: str, timeout: int = 30) -> str:
    """Run `command` in a shell with cwd=WORKSPACE and return combined stdout/stderr + exit code.

    Before running, print the command and ask the human y/N. Anything but "y" means
    return a message telling the model the user declined. (Real sandboxing is Phase 2.)
    Truncate huge output so one command can't flood the context window.
    """
    print(f"\n  agent wants to run: {command}")
    try:
        answer = input("  allow? [y/N] ").strip().lower()
    except EOFError:  # no terminal attached (e.g. piped input): treat as "no"
        answer = ""
    if answer != "y":
        return "The user declined to run this command. Try a different approach or ask the user."

    WORKSPACE.mkdir(exist_ok=True)
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=WORKSPACE,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,  # a command waiting for input would otherwise hang until timeout
        )
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout}s and was killed."

    output = result.stdout + result.stderr
    return _truncate(f"{output}\n(exit code {result.returncode})".strip())


# name -> Python function. execute_tool dispatches through this.
TOOLS = {"read_file": read_file, "write_file": write_file, "bash": bash}

# What the model sees: one entry per tool, in the OpenAI "function" tool format.
# Descriptions matter a lot more for a 27B model than for a frontier model.
TOOL_SCHEMAS = [
    {
        "type": "function",                 # always "function"; the only type that exists
        "function": {
            "name": "read_file",            # must match the key in TOOLS exactly
            "description": "Read a text file from the workspace and return its contents.",
            "parameters": {                 # a JSON Schema for the arguments object
                "type": "object",           # arguments are always one object, i.e. keyword args
                "properties": {
                    "path": {               # must match the Python parameter name exactly
                        "type": "string",
                        "description": "Path relative to the workspace, e.g. 'src/main.py'.",
                    },
                },
                "required": ["path"],       # which properties the model must provide
            },
        },
    },
    {
        "type": "function",                 # always "function"; the only type that exists
        "function": {
            "name": "write_file",            # must match the key in TOOLS exactly
            "description": "Write content to a file in the workspace, creating parent directories. Overwrites the file if it already exists.",
            "parameters": {                 # a JSON Schema for the arguments object
                "type": "object",           # arguments are always one object, i.e. keyword args
                "properties": {
                    "path": {               # must match the Python parameter name exactly
                        "type": "string",
                        "description": "Path relative to the workspace, e.g. 'src/main.py'.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file"
                    }
                },
                "required": ["path", "content"],       # which properties the model must provide
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Run a shell command in the workspace directory and return its output and exit code. "
                "The user must approve each command, and may decline. "
                "Use read_file/write_file for files instead of cat, echo, or heredocs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to run, e.g. 'python primes.py'.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Seconds before the command is killed. Defaults to 30; raise it for slow commands like test suites.",
                    },
                },
                "required": ["command"],  # timeout is optional: the Python default (30) applies when omitted
            },
        },
    },
]


def execute_tool(name: str, arguments_json: str) -> str:
    """Parse the model's JSON arguments, call the matching tool, return its result as a string.

    This must never raise. Unknown tool, invalid JSON, wrong argument names, and
    exceptions from the tool itself all become an error string that goes back to
    the model, so it can see what went wrong and try again.
    """
    # Every failure is *returned*, not raised: the error text becomes the tool result,
    # and the model reads it on its next turn and can fix its own mistake.
    fn = TOOLS.get(name)
    if fn is None:
        return f"Error: unknown tool {name!r}. Available tools: {', '.join(TOOLS)}."

    try:
        # Some servers send "" instead of "{}" for a call with no arguments.
        arguments = json.loads(arguments_json or "{}")
    except json.JSONDecodeError as e:
        return f"Error: tool arguments were not valid JSON ({e}). Send a JSON object like {{\"path\": \"notes.txt\"}}."
    if not isinstance(arguments, dict):
        return f"Error: tool arguments must be a JSON object, got {type(arguments).__name__}."

    try:
        # ** unpacks the dict as keyword args: {"path": "a.txt"} -> fn(path="a.txt")
        return str(fn(**arguments))
    except TypeError as e:  # missing / unexpected argument names
        return f"Error: bad arguments for {name}: {e}"
    except Exception as e:  # anything the tool itself raised (PermissionError, FileNotFoundError, ...)
        return f"Error: {type(e).__name__}: {e}"
