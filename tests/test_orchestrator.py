"""The orchestrator: building the client, wiring config -> toolset -> agent, and the CLI."""

import json
import re

import pytest

from harness import orchestrator
from harness.policy import deny_all, terminal_approver
from harness.tools import Toolset
from tests.fakes import FakeClient, reply


@pytest.fixture
def env(monkeypatch):
    """A controlled environment: the real .env is never read, and OMLX_* vars start unset."""
    monkeypatch.setattr(orchestrator, "load_dotenv", lambda: None)
    monkeypatch.delenv("OMLX_API_KEY", raising=False)
    monkeypatch.delenv("OMLX_BASE_URL", raising=False)
    return monkeypatch


class TestMakeClient:
    def test_missing_key_exits_with_help(self, env):
        with pytest.raises(SystemExit, match="OMLX_API_KEY is not set"):
            orchestrator.make_client()

    def test_empty_key_counts_as_missing(self, env):
        env.setenv("OMLX_API_KEY", "")
        with pytest.raises(SystemExit):
            orchestrator.make_client()

    def test_default_base_url(self, env):
        env.setenv("OMLX_API_KEY", "test-key")
        client = orchestrator.make_client()
        assert str(client.base_url).rstrip("/") == "http://127.0.0.1:8000/v1"
        assert client.api_key == "test-key"

    def test_base_url_from_env(self, env):
        env.setenv("OMLX_API_KEY", "test-key")
        env.setenv("OMLX_BASE_URL", "http://example.test:9999/v1")
        assert str(orchestrator.make_client().base_url).rstrip("/") == "http://example.test:9999/v1"

    def test_long_timeout_for_local_models(self, env):
        env.setenv("OMLX_API_KEY", "test-key")
        assert orchestrator.make_client().timeout == 600


@pytest.fixture
def agents_dir(tmp_path, monkeypatch):
    """A temp agents/ folder the orchestrator reads from, plus a FakeClient it builds."""
    directory = tmp_path / "agents"
    directory.mkdir()
    monkeypatch.setattr(orchestrator, "AGENTS_DIR", directory)
    return directory


def write_agent(directory, name, tools=None, prompt="Test prompt."):
    tools_yaml = "" if tools is None else "tools:\n" + "".join(f"  - {t}\n" for t in tools)
    (directory / f"{name}.md").write_text(f"---\nmodel: m\ntemperature: 0.1\nmax_steps: 3\n{tools_yaml}---\n{prompt}\n")


@pytest.fixture
def captured_run(monkeypatch):
    """Replace make_client and run_agent; record what the orchestrator hands the agent."""
    seen = {}

    def fake_run_agent(task, config, client, toolset, tracer):
        seen.update(task=task, config=config, client=client, toolset=toolset, tracer=tracer)
        return "agent result"

    monkeypatch.setattr(orchestrator, "make_client", lambda: FakeClient([reply("unused")]))
    monkeypatch.setattr(orchestrator, "run_agent", fake_run_agent)
    return seen


class TestRun:
    def test_wires_config_client_and_toolset(self, agents_dir, captured_run):
        write_agent(agents_dir, "base", tools=["read_file", "bash"])
        assert orchestrator.run("a task") == "agent result"
        assert captured_run["task"] == "a task"
        assert captured_run["config"]["model"] == "m"
        assert captured_run["config"]["system_prompt"] == "Test prompt."
        assert isinstance(captured_run["client"], FakeClient)
        assert isinstance(captured_run["toolset"], Toolset)
        assert list(captured_run["toolset"].tools) == ["read_file", "bash"]
        assert captured_run["toolset"].policy.approver is terminal_approver  # a human answers "ask"

    def test_unattended_uses_deny_all(self, agents_dir, captured_run):
        write_agent(agents_dir, "base", tools=["bash"])
        orchestrator.run("t", unattended=True)
        assert captured_run["toolset"].policy.approver is deny_all

    def test_default_agent_is_base(self, agents_dir, captured_run):
        write_agent(agents_dir, "base", prompt="I am base.")
        orchestrator.run("t")
        assert captured_run["config"]["system_prompt"] == "I am base."

    def test_named_agent(self, agents_dir, captured_run):
        write_agent(agents_dir, "base")
        write_agent(agents_dir, "reader", tools=["read_file"], prompt="I am reader.")
        orchestrator.run("t", agent_name="reader")
        assert captured_run["config"]["system_prompt"] == "I am reader."
        assert list(captured_run["toolset"].tools) == ["read_file"]

    def test_no_tools_key_means_no_tools(self, agents_dir, captured_run):
        # Access is opt-in: forgetting `tools:` must not hand out bash.
        write_agent(agents_dir, "base", tools=None)
        orchestrator.run("t")
        assert captured_run["toolset"].tools == {}

    def test_unknown_agent(self, agents_dir, captured_run):
        with pytest.raises(FileNotFoundError):
            orchestrator.run("t", agent_name="ghost")

    def test_tool_typo_fails_before_any_model_call(self, agents_dir, captured_run):
        write_agent(agents_dir, "base", tools=["read_fle"])
        with pytest.raises(ValueError, match="read_fle"):
            orchestrator.run("t")
        assert captured_run == {}  # the agent never started

    def test_config_includes_agent_name(self, agents_dir, captured_run):
        write_agent(agents_dir, "reader")
        orchestrator.run("t", agent_name="reader")
        assert captured_run["config"]["name"] == "reader"

    def test_end_to_end_with_fake_model(self, agents_dir, monkeypatch):
        # Not mocking run_agent: the real loop runs, only the model is scripted.
        write_agent(agents_dir, "base", tools=["read_file"])
        client = FakeClient([reply("hello from the fake model")])
        monkeypatch.setattr(orchestrator, "make_client", lambda: client)
        assert orchestrator.run("say hi") == "hello from the fake model"
        assert client.requests[0]["messages"][0]["content"] == "Test prompt."


class TestMain:
    @pytest.fixture
    def run_calls(self, agents_dir, monkeypatch):
        """Replace orchestrator.run; record the keyword arguments main() passes it."""
        write_agent(agents_dir, "base")
        write_agent(agents_dir, "reader")
        calls = []

        def fake_run(task, **kwargs):
            calls.append({"task": task, **kwargs})
            return "printed answer"

        monkeypatch.setattr(orchestrator, "run", fake_run)
        return calls

    def test_joins_words_into_one_task(self, run_calls, capsys):
        orchestrator.main(["list", "the", "files"])
        assert run_calls == [{"task": "list the files", "agent_name": "base", "unattended": False}]
        assert capsys.readouterr().out.strip() == "printed answer"

    def test_quoted_task(self, run_calls):
        orchestrator.main(["list the files"])
        assert run_calls[0]["task"] == "list the files"

    def test_agent_flag(self, run_calls):
        orchestrator.main(["--agent", "reader", "hi"])
        assert run_calls[0]["agent_name"] == "reader"

    def test_unattended_flag(self, run_calls):
        orchestrator.main(["--unattended", "hi"])
        assert run_calls[0]["unattended"] is True

    def test_flags_after_task(self, run_calls):
        orchestrator.main(["do", "it", "--unattended", "--agent", "reader"])
        assert run_calls == [{"task": "do it", "agent_name": "reader", "unattended": True}]

    def test_unknown_agent_lists_real_ones(self, run_calls, capsys):
        with pytest.raises(SystemExit) as exit_info:
            orchestrator.main(["--agent", "ghost", "hi"])
        assert exit_info.value.code == 2  # argparse's "bad usage" exit code
        assert "invalid choice: 'ghost' (choose from base, reader)" in capsys.readouterr().err
        assert run_calls == []

    @pytest.mark.parametrize("argv", [[], [""], ["   "]])
    def test_no_task_is_a_usage_error(self, run_calls, capsys, argv):
        with pytest.raises(SystemExit) as exit_info:
            orchestrator.main(argv)
        assert exit_info.value.code == 2
        assert "usage: harness" in capsys.readouterr().err
        assert run_calls == []

    def test_help(self, run_calls, capsys):
        with pytest.raises(SystemExit) as exit_info:
            orchestrator.main(["--help"])
        assert exit_info.value.code == 0
        out = capsys.readouterr().out
        assert "--agent" in out and "--unattended" in out

    def test_reads_real_command_line_by_default(self, run_calls, monkeypatch):
        monkeypatch.setattr("sys.argv", ["harness", "from", "argv"])
        orchestrator.main()
        assert run_calls[0]["task"] == "from argv"


class TestTraceFiles:
    @pytest.fixture
    def fake_model(self, agents_dir, monkeypatch):
        write_agent(agents_dir, "base", tools=["read_file"])
        monkeypatch.setattr(orchestrator, "make_client", lambda: FakeClient([reply("the answer")]))

    def trace_files(self, traces_dir):
        return sorted(traces_dir.glob("*.jsonl"))

    def test_each_run_writes_one_trace(self, fake_model, traces_dir):
        orchestrator.run("t")
        orchestrator.run("t")
        assert len(self.trace_files(traces_dir)) == 2

    def test_trace_name(self, fake_model, traces_dir):
        orchestrator.run("t")
        name = self.trace_files(traces_dir)[0].name
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}_base_[0-9a-f]{4}\.jsonl", name)

    def test_trace_contents(self, fake_model, traces_dir):
        orchestrator.run("say hi")
        events = [json.loads(line) for line in self.trace_files(traces_dir)[0].read_text().splitlines()]
        assert [e["event"] for e in events] == ["run_start", "model_response", "run_end"]
        assert events[0]["task"] == "say hi"
        assert events[0]["config"]["name"] == "base"
        assert events[0]["approver"] == "terminal_approver"
        assert events[-1]["answer"] == "the answer"

    def test_unattended_is_visible_in_trace(self, fake_model, traces_dir):
        orchestrator.run("t", unattended=True)
        start = json.loads(self.trace_files(traces_dir)[0].read_text().splitlines()[0])
        assert start["approver"] == "deny_all"

    def test_path_printed_to_stderr(self, fake_model, traces_dir, capsys):
        orchestrator.run("t")
        captured = capsys.readouterr()
        assert f"trace: {self.trace_files(traces_dir)[0]}" in captured.err
        assert "trace:" not in captured.out  # stdout stays clean for the answer

    def test_crashed_run_still_leaves_a_trace(self, agents_dir, traces_dir, monkeypatch, capsys):
        write_agent(agents_dir, "base")
        monkeypatch.setattr(orchestrator, "make_client", lambda: FakeClient([]))  # first model call fails
        with pytest.raises(AssertionError):
            orchestrator.run("t")
        events = [json.loads(line) for line in self.trace_files(traces_dir)[0].read_text().splitlines()]
        assert events[-1]["outcome"] == "crashed"
        assert "trace:" in capsys.readouterr().err  # the path is printed even after a crash
