"""Control plane: load config, build what the agent needs, hand it over, run.

The orchestrator wires things together and does no work itself. Everything else
is imported here; nothing imports the orchestrator.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from harness.agent import run_agent
from harness.config_loader import load_agent_config

# Located relative to this file, so `uv run harness` works from any directory.
AGENTS_DIR = Path(__file__).resolve().parents[2] / "agents"


def make_client() -> OpenAI:
    """Load .env and return an OpenAI client pointed at oMLX (OMLX_BASE_URL, OMLX_API_KEY)."""
    # reads .env, copies each KEY=value into process env, os.environ can see all
    load_dotenv()

    # get the api key
    api_key = os.environ.get("OMLX_API_KEY")
    if not api_key:
        raise SystemExit("OMLX_API_KEY is not set. Copy .env.example to .env and fill it in.")
    return OpenAI(
        base_url=os.environ.get("OMLX_BASE_URL", "http://127.0.0.1:8000/v1"),
        api_key=api_key,
        timeout=600,
    )


def run(task: str, agent_name: str = "base") -> str:
    """Load the named agent's config, build its client, run it on `task`."""
    config = load_agent_config(AGENTS_DIR / f"{agent_name}.md")
    client = make_client()
    return run_agent(task, config, client)


def main() -> None:
    """CLI entry point: `uv run harness "your task here"`."""
    # this allows us to pass in command line arguments without quotes
    task = " ".join(sys.argv[1:]).strip()
    if not task:
        raise SystemExit('usage: uv run harness "your task here"')
    print(run(task))
