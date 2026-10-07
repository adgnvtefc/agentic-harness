"""Load agent definitions from files. Reads files and returns data; builds nothing."""

from pathlib import Path

import yaml


def load_agent_config(path: str | Path) -> dict:
    """Read an agent file: YAML settings between '---' lines, then the system prompt.

        ---
        model: Qwen3.8-27B-MLX-8bit     <- settings (YAML)
        temperature: 0.6
        ---
        You are a coding agent...       <- everything below is the system prompt

    Returns the settings as a dict, plus a "system_prompt" key holding the body.
    """

    # reads a path and each line
    lines = Path(path).read_text().splitlines()

    # The settings block must open on the very first line and close on a later "---" line.
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"{path}: must start with a '---' line")
    try:
        # strip() so trailing whitespace on the closing line is tolerated, same as the opening one
        end = [line.strip() for line in lines].index("---", 1)
    except ValueError:
        raise ValueError(f"{path}: no closing '---' line after the settings") from None

    # safe_load only builds plain data (dicts, lists, strings, numbers); plain yaml.load
    # can construct arbitrary Python objects, which you never want from a file.
    
    # loads everything between "---" as settings dict
    settings = yaml.safe_load("\n".join(lines[1:end])) or {}
    if not isinstance(settings, dict):
        raise ValueError(f"{path}: settings must be 'key: value' lines")

    # loads all lines after the last "---" as systems prompt
    settings["system_prompt"] = "\n".join(lines[end + 1 :]).strip()
    return settings
