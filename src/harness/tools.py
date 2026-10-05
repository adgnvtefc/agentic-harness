"""Tools the agent can call: Python implementations plus the JSON schemas the model sees."""

from pathlib import Path

# All file and shell access is confined to this directory for Phase 1.
WORKSPACE = Path(__file__).resolve().parents[2] / "workspace"


def read_file(path: str) -> str:
    """Return the contents of `path` (relative to WORKSPACE).

    Refuse paths that resolve outside WORKSPACE (think: "../../.ssh/id_rsa").
    """
    raise NotImplementedError


def write_file(path: str, content: str) -> str:
    """Write `content` to `path` (relative to WORKSPACE), creating parent dirs.

    Same path rule as read_file. Return a short confirmation the model can read.
    """
    raise NotImplementedError


def bash(command: str, timeout: int = 30) -> str:
    """Run `command` in a shell with cwd=WORKSPACE and return combined stdout/stderr + exit code.

    Before running, print the command and ask the human y/N. Anything but "y" means
    return a message telling the model the user declined. (Real sandboxing is Phase 2.)
    Truncate huge output so one command can't flood the context window.
    """
    raise NotImplementedError


# name -> Python function. execute_tool dispatches through this.
TOOLS = {}

# What the model sees: one entry per tool, in the OpenAI "function" tool format.
# Descriptions matter a lot more for a 27B model than for a frontier model.
TOOL_SCHEMAS = []


def execute_tool(name: str, arguments_json: str) -> str:
    """Parse the model's JSON arguments, call the matching tool, return its result as a string.

    This must never raise. Unknown tool, invalid JSON, wrong argument names, and
    exceptions from the tool itself all become an error string that goes back to
    the model, so it can see what went wrong and try again.
    """
    raise NotImplementedError
