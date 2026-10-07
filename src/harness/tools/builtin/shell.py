"""Shell tool: runs commands in WORKSPACE after the human approves them."""

import subprocess

from harness.tools.base import WORKSPACE, Tool, truncate


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
    return truncate(f"{output}\n(exit code {result.returncode})".strip())


BASH = Tool(
    fn=bash,
    schema={
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
)
