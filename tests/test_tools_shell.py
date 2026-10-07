"""The bash tool: human approval, execution in the workspace, output handling, timeouts."""

import subprocess
import time
import uuid

import pytest

from harness.tools.base import MAX_OUTPUT_CHARS
from harness.tools.builtin.shell import bash

DECLINED = "The user declined to run this command. Try a different approach or ask the user."


class TestApproval:
    @pytest.mark.parametrize("answer", ["y", "Y", " y ", "y\n"])
    def test_yes_runs(self, answer_prompt, answer):
        answer_prompt(answer)
        assert bash("echo hi").startswith("hi")

    @pytest.mark.parametrize("answer", ["", "n", "N", "no", "yes", "yy", "sure"])
    def test_anything_else_declines(self, answer_prompt, answer):
        # Only an explicit "y" approves. Even "yes" declines: strict beats surprising.
        answer_prompt(answer)
        assert bash("echo hi") == DECLINED

    def test_no_terminal_declines(self, answer_prompt):
        answer_prompt(EOFError)
        assert bash("echo hi") == DECLINED

    def test_declined_command_does_not_run(self, answer_prompt, workspace):
        answer_prompt("n")
        bash("touch should_not_exist")
        assert not (workspace / "should_not_exist").exists()

    def test_command_is_shown_before_asking(self, approve, capsys):
        bash("echo visible-command")
        assert "agent wants to run: echo visible-command" in capsys.readouterr().out
        assert approve == ["  allow? [y/N] "]


class TestExecution:
    def test_runs_in_workspace(self, approve, workspace):
        assert bash("pwd").splitlines()[0] == str(workspace)

    def test_can_see_workspace_files(self, approve, workspace):
        (workspace / "present.txt").write_text("")
        assert "present.txt" in bash("ls")

    def test_reports_exit_code_zero(self, approve):
        assert bash("true").endswith("(exit code 0)")

    def test_reports_nonzero_exit_code(self, approve):
        assert bash("exit 3").endswith("(exit code 3)")

    def test_captures_stdout_and_stderr(self, approve):
        out = bash("echo to-out; echo to-err >&2")
        assert "to-out" in out and "to-err" in out

    def test_shell_features_work(self, approve):
        assert bash("echo a | tr a b && echo $((2 + 3))").startswith("b\n5")

    def test_stdin_is_closed(self, approve):
        # A command that reads stdin must get EOF at once, not hang until the timeout.
        start = time.monotonic()
        out = bash("cat", timeout=10)
        assert time.monotonic() - start < 5
        assert out.endswith("(exit code 0)")

    def test_long_output_is_truncated(self, approve):
        out = bash(f"python3 -c \"print('x' * {MAX_OUTPUT_CHARS * 2})\"")
        assert "characters omitted" in out
        assert out.endswith("(exit code 0)")  # the tail, with the exit code, survives truncation

    def test_recreates_missing_workspace(self, approve, workspace):
        workspace.rmdir()
        assert bash("true").endswith("(exit code 0)")
        assert workspace.is_dir()


class TestTimeout:
    def test_slow_command_is_killed(self, approve):
        start = time.monotonic()
        assert bash("sleep 10", timeout=1) == "Command timed out after 1s and was killed."
        assert time.monotonic() - start < 5

    def test_timeout_kills_child_processes(self, approve):
        marker = f"sleep 3{uuid.uuid4().int % 1000:03d}"  # unique, so we only ever find our own process
        try:
            bash(f"{marker}; echo done", timeout=1)
            time.sleep(0.3)
            survivors = subprocess.run(["pgrep", "-f", marker], capture_output=True, text=True).stdout
            assert survivors == ""
        finally:
            subprocess.run(["pkill", "-f", marker])
