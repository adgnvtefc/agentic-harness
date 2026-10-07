"""The orchestrator: building the client, wiring config -> toolset -> agent, and the CLI."""

import pytest

from harness import orchestrator
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

    def fake_run_agent(task, config, client, toolset):
        seen.update(task=task, config=config, client=client, toolset=toolset)
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

    def test_end_to_end_with_fake_model(self, agents_dir, monkeypatch):
        # Not mocking run_agent: the real loop runs, only the model is scripted.
        write_agent(agents_dir, "base", tools=["read_file"])
        client = FakeClient([reply("hello from the fake model")])
        monkeypatch.setattr(orchestrator, "make_client", lambda: client)
        assert orchestrator.run("say hi") == "hello from the fake model"
        assert client.requests[0]["messages"][0]["content"] == "Test prompt."


class TestMain:
    def test_joins_arguments_into_one_task(self, monkeypatch, capsys):
        seen = {}
        monkeypatch.setattr(orchestrator, "run", lambda task: seen.setdefault("task", task) and "printed answer")
        monkeypatch.setattr("sys.argv", ["harness", "list", "the", "files"])
        orchestrator.main()
        assert seen["task"] == "list the files"
        assert capsys.readouterr().out.strip() == "printed answer"

    @pytest.mark.parametrize("argv", [["harness"], ["harness", ""], ["harness", "   "]])
    def test_no_task_prints_usage(self, monkeypatch, argv):
        monkeypatch.setattr("sys.argv", argv)
        with pytest.raises(SystemExit, match="usage"):
            orchestrator.main()
