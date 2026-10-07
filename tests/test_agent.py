"""The agent loop, driven by a scripted FakeClient instead of a real model."""

import pytest

from harness.agent import call_model, run_agent
from harness.tools import Toolset
from tests.fakes import FakeClient, always, call, reply

CONFIG = {"model": "test-model", "temperature": 0.3, "max_steps": 5, "system_prompt": "You are a test agent."}


def run(replies, task="do the thing", tools=("echo",), **config_overrides):
    """Run the loop against scripted replies. Returns (final answer, client) for inspection."""
    client = FakeClient(replies)
    answer = run_agent(task, {**CONFIG, **config_overrides}, client, Toolset(list(tools)))
    return answer, client


@pytest.fixture(autouse=True)
def _fake_tools(fake_tools):
    """Every test here may use the echo/boom/count test tools."""


class TestFinishing:
    def test_answer_without_tools_returns_immediately(self):
        answer, client = run([reply("done")])
        assert answer == "done"
        assert len(client.requests) == 1

    def test_none_content_becomes_empty_string(self):
        answer, _ = run([reply(None)])
        assert answer == ""

    def test_tool_then_answer(self):
        answer, client = run([reply(tool_calls=[call("echo", {"text": "hi"})]), reply("final")])
        assert answer == "final"
        assert len(client.requests) == 2

    def test_stops_at_max_steps(self):
        answer, client = run(always(lambda: reply(tool_calls=[call("echo", {"text": "again"})])), max_steps=3)
        assert answer == "Stopped after 3 steps without a final answer."
        assert len(client.requests) == 3  # exactly max_steps model calls, no more

    def test_max_steps_one(self):
        answer, client = run(always(lambda: reply(tool_calls=[call("echo", {"text": "x"})])), max_steps=1)
        assert answer.startswith("Stopped after 1 steps")
        assert len(client.requests) == 1


class TestRequests:
    def test_first_request_is_system_then_task(self):
        _, client = run([reply("ok")], task="list files")
        assert client.requests[0]["messages"] == [
            {"role": "system", "content": "You are a test agent."},
            {"role": "user", "content": "list files"},
        ]

    def test_model_and_temperature_come_from_config(self):
        _, client = run([reply("ok")], model="some-model", temperature=0.9)
        assert client.requests[0]["model"] == "some-model"
        assert client.requests[0]["temperature"] == 0.9

    def test_sends_only_the_toolsets_schemas(self):
        _, client = run([reply("ok")], tools=("echo", "count"))
        assert [t["function"]["name"] for t in client.requests[0]["tools"]] == ["echo", "count"]

    def test_omits_tools_key_when_toolset_empty(self):
        # Some servers reject "tools": [], so the key must be absent, not empty.
        _, client = run([reply("ok")], tools=())
        assert "tools" not in client.requests[0]

    def test_tools_identical_every_turn(self):
        _, client = run([reply(tool_calls=[call("echo", {"text": "a"})]), reply("ok")])
        assert client.requests[0]["tools"] == client.requests[1]["tools"]

    def test_history_is_append_only(self):
        # Each request extends the previous one without editing it: that's what lets the
        # server reuse its prompt cache instead of re-reading the whole conversation.
        _, client = run(
            [
                reply(tool_calls=[call("echo", {"text": "1"})]),
                reply(tool_calls=[call("echo", {"text": "2"})]),
                reply("done"),
            ]
        )
        sent = [r["messages"] for r in client.requests]
        for earlier, later in zip(sent, sent[1:]):
            assert later[: len(earlier)] == earlier
            assert len(later) > len(earlier)


class TestToolRoundTrip:
    def test_assistant_turn_then_tool_result(self):
        tc = call("echo", {"text": "hi"}, id="call_abc")
        _, client = run([reply(tool_calls=[tc]), reply("done")])
        history = client.requests[1]["messages"]

        assistant, tool_result = history[2], history[3]
        assert assistant["role"] == "assistant"
        assert assistant["tool_calls"][0]["id"] == "call_abc"
        assert assistant["tool_calls"][0]["function"] == {"name": "echo", "arguments": '{"text": "hi"}'}
        assert tool_result == {"role": "tool", "tool_call_id": "call_abc", "content": "hi"}

    def test_every_tool_call_gets_exactly_one_reply_in_order(self):
        calls = [call("echo", {"text": t}, id=f"id_{t}") for t in ("a", "b", "c")]
        _, client = run([reply(tool_calls=calls), reply("done")])
        results = [m for m in client.requests[1]["messages"] if m["role"] == "tool"]
        assert [(m["tool_call_id"], m["content"]) for m in results] == [("id_a", "a"), ("id_b", "b"), ("id_c", "c")]

    def test_assistant_message_has_no_null_fields(self):
        # A tool-only turn has content=None; some servers reject "content": null next to tool_calls.
        _, client = run([reply(None, tool_calls=[call("echo", {"text": "x"})]), reply("done")])
        assistant = client.requests[1]["messages"][2]
        assert "content" not in assistant
        assert None not in assistant.values()

    def test_reasoning_is_dropped_from_history(self):
        _, client = run([reply(tool_calls=[call("echo", {"text": "x"})], reasoning="secret thoughts"), reply("done")])
        assert "reasoning_content" not in client.requests[1]["messages"][2]
        assert "secret thoughts" not in str(client.requests[1]["messages"])

    def test_final_text_alongside_reasoning(self):
        answer, _ = run([reply("the answer", reasoning="thinking...")])
        assert answer == "the answer"


class TestModelMistakesDoNotCrashTheLoop:
    """Small models get tool calls wrong. Each mistake must come back as text the model can fix."""

    def last_tool_result(self, client):
        return [m for m in client.requests[-1]["messages"] if m["role"] == "tool"][-1]["content"]

    def test_invented_tool(self):
        _, client = run([reply(tool_calls=[call("hack_the_planet", {})]), reply("sorry")])
        assert self.last_tool_result(client).startswith("Error: unknown tool 'hack_the_planet'")

    def test_tool_not_in_this_agents_toolset(self, approve):
        _, client = run([reply(tool_calls=[call("bash", {"command": "rm -rf /"})]), reply("ok")], tools=("echo",))
        assert self.last_tool_result(client).startswith("Error: unknown tool 'bash'")
        assert approve == []  # never even asked the human

    def test_malformed_json_arguments(self):
        _, client = run([reply(tool_calls=[call("echo", "{text: oops")]), reply("ok")])
        assert "not valid JSON" in self.last_tool_result(client)

    def test_tool_raises(self):
        answer, client = run([reply(tool_calls=[call("boom")]), reply("recovered")], tools=("boom",))
        assert self.last_tool_result(client) == "Error: RuntimeError: kaboom"
        assert answer == "recovered"


class TestOutput:
    def test_prints_each_call_and_result(self, capsys):
        run([reply(tool_calls=[call("echo", {"text": "hello"})]), reply("done")])
        out = capsys.readouterr().out
        assert '[step 1] echo({"text": "hello"})' in out
        assert "  -> hello" in out

    def test_long_results_are_previewed_but_sent_in_full(self, capsys):
        long_text = "z" * 500
        _, client = run([reply(tool_calls=[call("echo", {"text": long_text})]), reply("done")])
        result_line = next(line for line in capsys.readouterr().out.splitlines() if line.startswith("  -> "))
        assert result_line == "  -> " + "z" * 200 + "..."  # terminal gets a 200-char preview
        assert client.requests[1]["messages"][-1]["content"] == long_text  # the model gets all of it


class TestCallModel:
    def test_returns_the_message(self):
        client = FakeClient([reply("hi")])
        msg = call_model(client, [{"role": "user", "content": "x"}], "m", 0.5, [])
        assert msg.content == "hi"
        assert client.requests == [{"model": "m", "messages": [{"role": "user", "content": "x"}], "temperature": 0.5}]

    def test_includes_tools_when_given(self):
        client = FakeClient([reply("hi")])
        call_model(client, [], "m", 0.5, [{"type": "function"}])
        assert client.requests[0]["tools"] == [{"type": "function"}]
