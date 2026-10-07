"""The bash tool: execution in the workspace, output handling, timeouts. (Approval is tested in test_policy.py.)"""

import subprocess
import time
import uuid

import pytest

from harness.tools.base import MAX_OUTPUT_CHARS
from harness.tools.builtin.shell import bash


class TestExecution:
    def test_runs_in_workspace(self, workspace):
        assert bash("pwd").splitlines()[0] == str(workspace)

    def test_can_see_workspace_files(self, workspace):
        (workspace / "present.txt").write_text("")
        assert "present.txt" in bash("ls")

    def test_reports_exit_code_zero(self):
        assert bash("true").endswith("(exit code 0)")

    def test_reports_nonzero_exit_code(self):
        assert bash("exit 3").endswith("(exit code 3)")

    def test_captures_stdout_and_stderr(self):
        out = bash("echo to-out; echo to-err >&2")
        assert "to-out" in out and "to-err" in out

    def test_shell_features_work(self):
        assert bash("echo a | tr a b && echo $((2 + 3))").startswith("b\n5")

    def test_stdin_is_closed(self):
        # A command that reads stdin must get EOF at once, not hang until the timeout.
        start = time.monotonic()
        out = bash("cat", timeout=10)
        assert time.monotonic() - start < 5
        assert out.endswith("(exit code 0)")

    def test_long_output_is_truncated(self):
        out = bash(f"python3 -c \"print('x' * {MAX_OUTPUT_CHARS * 2})\"")
        assert "characters omitted" in out
        assert out.endswith("(exit code 0)")  # the tail, with the exit code, survives truncation

    def test_recreates_missing_workspace(self, workspace):
        workspace.rmdir()
        assert bash("true").endswith("(exit code 0)")
        assert workspace.is_dir()


class TestTimeout:
    def test_slow_command_is_killed(self):
        start = time.monotonic()
        assert bash("sleep 10", timeout=1) == "Command timed out after 1s and was killed."
        assert time.monotonic() - start < 5

    def test_timeout_kills_child_processes(self):
        marker = f"sleep 3{uuid.uuid4().int % 1000:03d}"  # unique, so we only ever find our own process
        try:
            bash(f"{marker}; echo done", timeout=1)
            time.sleep(0.3)
            survivors = subprocess.run(["pgrep", "-f", marker], capture_output=True, text=True).stdout
            assert survivors == ""
        finally:
            subprocess.run(["pkill", "-f", marker])


class TestNoLongerAsks:
    def test_runs_without_prompting(self, answer_prompt):
        # Approval moved to the Policy. Called directly, bash() just runs.
        prompts = answer_prompt("n")
        assert bash("echo hi").startswith("hi")
        assert prompts == []
