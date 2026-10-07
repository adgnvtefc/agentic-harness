"""load_agent_config: parsing agent .md files (YAML settings + Markdown system prompt)."""

import pytest
import yaml

from harness.config_loader import load_agent_config


@pytest.fixture
def agent_file(tmp_path):
    """Write `text` to a temp .md file and return its path."""

    def write(text: str):
        path = tmp_path / "agent.md"
        path.write_text(text)
        return path

    return write


class TestValidFiles:
    def test_settings_and_prompt(self, agent_file):
        path = agent_file("---\nmodel: m\ntemperature: 0.6\nmax_steps: 20\n---\nYou are helpful.\n")
        assert load_agent_config(path) == {
            "model": "m",
            "temperature": 0.6,
            "max_steps": 20,
            "system_prompt": "You are helpful.",
        }

    def test_yaml_types_are_converted(self, agent_file):
        config = load_agent_config(agent_file("---\nt: 0.6\nn: 20\nflag: true\ntools:\n  - a\n  - b\n---\n"))
        assert config["t"] == 0.6 and isinstance(config["t"], float)
        assert config["n"] == 20 and isinstance(config["n"], int)
        assert config["flag"] is True
        assert config["tools"] == ["a", "b"]

    def test_accepts_str_path(self, agent_file):
        assert load_agent_config(str(agent_file("---\nmodel: m\n---\nhi")))["model"] == "m"

    def test_empty_settings_block(self, agent_file):
        assert load_agent_config(agent_file("---\n---\nprompt")) == {"system_prompt": "prompt"}

    def test_empty_body_gives_empty_prompt(self, agent_file):
        assert load_agent_config(agent_file("---\nmodel: m\n---\n"))["system_prompt"] == ""

    def test_prompt_is_stripped_but_keeps_inner_formatting(self, agent_file):
        body = "\n\n# Role\n\nLine one.\n  indented\n\n"
        assert load_agent_config(agent_file(f"---\nmodel: m\n---{body}"))["system_prompt"] == (
            "# Role\n\nLine one.\n  indented"
        )

    def test_dashes_inside_prompt_are_kept(self, agent_file):
        # Only the first closing '---' ends the settings; later ones belong to the prompt.
        config = load_agent_config(agent_file("---\nmodel: m\n---\nabove\n---\nbelow"))
        assert config == {"model": "m", "system_prompt": "above\n---\nbelow"}

    def test_windows_line_endings(self, agent_file):
        config = load_agent_config(agent_file("---\r\nmodel: m\r\n---\r\nhi\r\n"))
        assert config == {"model": "m", "system_prompt": "hi"}

    def test_opening_line_tolerates_trailing_whitespace(self, agent_file):
        assert load_agent_config(agent_file("---  \nmodel: m\n---\nhi"))["model"] == "m"

    def test_closing_line_tolerates_trailing_whitespace(self, agent_file):
        assert load_agent_config(agent_file("---\nmodel: m\n---  \nhi"))["model"] == "m"


class TestInvalidFiles:
    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_agent_config(tmp_path / "nope.md")

    @pytest.mark.parametrize("text", ["", "\n", "model: m\n---\nhi", "hello\n---\nmodel: m\n---\n"])
    def test_must_start_with_dashes(self, agent_file, text):
        with pytest.raises(ValueError, match="must start with a '---' line"):
            load_agent_config(agent_file(text))

    def test_missing_closing_dashes(self, agent_file):
        with pytest.raises(ValueError, match="no closing '---' line"):
            load_agent_config(agent_file("---\nmodel: m\nYou are helpful."))

    @pytest.mark.parametrize("settings", ["- a\n- b", "just a string", "42"])
    def test_settings_must_be_a_mapping(self, agent_file, settings):
        with pytest.raises(ValueError, match="settings must be 'key: value' lines"):
            load_agent_config(agent_file(f"---\n{settings}\n---\nhi"))

    def test_error_names_the_file(self, agent_file):
        path = agent_file("no frontmatter")
        with pytest.raises(ValueError, match=str(path)):
            load_agent_config(path)

    def test_malformed_yaml(self, agent_file):
        with pytest.raises(yaml.YAMLError):
            load_agent_config(agent_file("---\nmodel: [unclosed\n---\nhi"))

    def test_yaml_cannot_construct_python_objects(self, agent_file):
        # safe_load refuses python/* tags; plain yaml.load would execute this.
        with pytest.raises(yaml.constructor.ConstructorError):
            load_agent_config(agent_file("---\nx: !!python/object/apply:os.system ['echo pwned']\n---\nhi"))
