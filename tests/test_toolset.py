"""The registry and Toolset: what an agent can see, what it can run, and how failures come back."""

import inspect
import re

import pytest

from harness.tools import REGISTRY, Toolset


class TestRegistry:
    def test_builtins_registered(self):
        assert {"read_file", "write_file", "bash"} <= set(REGISTRY)

    def test_keys_match_tool_names(self):
        assert all(key == tool.name for key, tool in REGISTRY.items())


@pytest.mark.parametrize("name", sorted(REGISTRY))
class TestSchemaMatchesFunction:
    """Every built-in tool's schema must agree with its Python function.

    The model only sees the schema, and execute() calls fn(**arguments). If they drift,
    the tool fails only when the model calls it, so catch it here instead.
    """

    def test_schema_shape(self, name):
        schema = REGISTRY[name].schema
        assert schema["type"] == "function"
        fn = schema["function"]
        assert re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", fn["name"])  # what model APIs accept
        assert fn["description"].strip()
        assert fn["parameters"]["type"] == "object"

    def test_properties_match_parameters(self, name):
        tool = REGISTRY[name]
        params = inspect.signature(tool.fn).parameters
        assert set(tool.schema["function"]["parameters"]["properties"]) == set(params)

    def test_required_matches_parameters_without_defaults(self, name):
        tool = REGISTRY[name]
        params = inspect.signature(tool.fn).parameters
        no_default = {p for p, v in params.items() if v.default is inspect.Parameter.empty}
        assert set(tool.schema["function"]["parameters"].get("required", [])) == no_default

    def test_every_property_is_documented(self, name):
        for prop, spec in REGISTRY[name].schema["function"]["parameters"]["properties"].items():
            assert spec.get("type"), f"{name}.{prop} has no type"
            assert spec.get("description", "").strip(), f"{name}.{prop} has no description"


class TestConstruction:
    def test_picks_named_tools(self):
        assert set(Toolset(["read_file", "bash"]).tools) == {"read_file", "bash"}

    def test_empty(self):
        toolset = Toolset([])
        assert toolset.tools == {}
        assert toolset.schemas() == []

    def test_unknown_name_fails_at_startup(self):
        with pytest.raises(ValueError, match=r"Unknown tool\(s\) \['bsh'\]") as err:
            Toolset(["read_file", "bsh"])
        assert "read_file, write_file, bash" in str(err.value)  # tells you the valid names

    def test_reports_every_unknown_name(self):
        with pytest.raises(ValueError, match=r"\['nope', 'nada'\]"):
            Toolset(["nope", "read_file", "nada"])

    def test_duplicates_collapse(self):
        # Model APIs reject duplicate tool names, so listing a tool twice must not send it twice.
        assert len(Toolset(["bash", "bash"]).schemas()) == 1

    def test_independent_of_each_other(self):
        a, b = Toolset(["read_file"]), Toolset(["bash"])
        assert set(a.tools) == {"read_file"} and set(b.tools) == {"bash"}


class TestSchemas:
    def test_only_own_tools(self):
        names = [s["function"]["name"] for s in Toolset(["read_file"]).schemas()]
        assert names == ["read_file"]

    def test_order_follows_config(self):
        # A stable order keeps the request byte-identical across runs (prompt caching).
        names = [s["function"]["name"] for s in Toolset(["bash", "read_file", "write_file"]).schemas()]
        assert names == ["bash", "read_file", "write_file"]

    def test_are_the_registry_schemas(self):
        assert Toolset(["bash"]).schemas() == [REGISTRY["bash"].schema]


class TestExecute:
    def test_calls_tool_with_keyword_args(self, fake_tools):
        assert Toolset(["echo"]).execute("echo", '{"text": "hi", "suffix": "!"}') == "hi!"

    def test_optional_argument_uses_default(self, fake_tools):
        assert Toolset(["echo"]).execute("echo", '{"text": "hi"}') == "hi"

    def test_result_is_stringified(self, fake_tools):
        assert Toolset(["count"]).execute("count", "{}") == "42"

    @pytest.mark.parametrize("arguments", ["", None])
    def test_empty_arguments_mean_no_arguments(self, fake_tools, arguments):
        assert Toolset(["count"]).execute("count", arguments) == "42"

    def test_real_tool_end_to_end(self, workspace):
        toolset = Toolset(["read_file", "write_file"])
        assert toolset.execute("write_file", '{"path": "a.txt", "content": "hi"}') == "Wrote 2 characters to a.txt."
        assert toolset.execute("read_file", '{"path": "a.txt"}') == "hi"


class TestExecuteRefusesToolsNotGiven:
    """The security property: the registry has a tool, but this agent wasn't given it."""

    def test_registered_but_not_given(self, workspace):
        result = Toolset(["read_file"]).execute("write_file", '{"path": "x.txt", "content": "pwned"}')
        assert result == "Error: unknown tool 'write_file'. Available tools: read_file."
        assert not (workspace / "x.txt").exists()

    def test_bash_not_given_never_prompts(self, answer_prompt, workspace):
        prompts = answer_prompt("y")
        result = Toolset(["read_file"]).execute("bash", '{"command": "touch pwned"}')
        assert result.startswith("Error: unknown tool 'bash'")
        assert prompts == []  # refused before it ever got to ask
        assert not (workspace / "pwned").exists()

    def test_empty_toolset_runs_nothing(self):
        assert Toolset([]).execute("read_file", '{"path": "a"}') == (
            "Error: unknown tool 'read_file'. Available tools: (none)."
        )

    def test_invented_tool(self):
        assert Toolset(["read_file"]).execute("delete_everything", "{}").startswith("Error: unknown tool")


class TestExecuteReturnsErrorsInsteadOfRaising:
    """execute() must never raise: every failure becomes text the model can read and react to."""

    def test_invalid_json(self, fake_tools):
        result = Toolset(["echo"]).execute("echo", "{text: hi}")
        assert result.startswith("Error: tool arguments were not valid JSON")
        assert '{"path": "notes.txt"}' in result  # shows the model what valid looks like

    @pytest.mark.parametrize(("arguments", "kind"), [("[1, 2]", "list"), ('"hi"', "str"), ("3", "int"), ("null", "NoneType")])
    def test_json_that_is_not_an_object(self, fake_tools, arguments, kind):
        assert Toolset(["echo"]).execute("echo", arguments) == f"Error: tool arguments must be a JSON object, got {kind}."

    def test_missing_required_argument(self, fake_tools):
        result = Toolset(["echo"]).execute("echo", "{}")
        assert result.startswith("Error: bad arguments for echo:")
        assert "text" in result

    def test_unexpected_argument(self, fake_tools):
        result = Toolset(["echo"]).execute("echo", '{"text": "a", "colour": "red"}')
        assert result.startswith("Error: bad arguments for echo:")
        assert "colour" in result

    def test_tool_exception(self, fake_tools):
        assert Toolset(["boom"]).execute("boom", "{}") == "Error: RuntimeError: kaboom"

    def test_real_tool_exceptions_are_named(self):
        toolset = Toolset(["read_file"])
        assert toolset.execute("read_file", '{"path": "nope.txt"}').startswith("Error: FileNotFoundError:")
        assert toolset.execute("read_file", '{"path": "../x"}').startswith("Error: PermissionError:")

    @pytest.mark.parametrize(
        ("name", "arguments"),
        [
            ("", ""),
            ("echo", "}{"),
            ("echo", '{"text": 123}'),  # wrong type: echo does str + str, which raises TypeError
            ("echo", '{"text": "a"' * 100),
            ("echo", "\x00\xff"),
            ("boom", '{"unexpected": true}'),
            ("../../bash", "{}"),
            ("read_file", '{"path": null}'),
            ("read_file", '{"path": ["a", "b"]}'),
        ],
    )
    def test_never_raises_on_garbage(self, fake_tools, name, arguments):
        result = Toolset(["echo", "boom", "read_file"]).execute(name, arguments)
        assert isinstance(result, str) and result.startswith("Error:")
