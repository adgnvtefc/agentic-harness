"""The agent loop: call the model, run any tool calls, feed results back, repeat."""

import os

from dotenv import load_dotenv
from openai import OpenAI

SYSTEM_PROMPT = """Hello! You are a helpful AI agent."""  # Write this yourself: who the agent is, where it works, how to use tools.


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


def call_model(client: OpenAI, messages: list[dict]):
    """Send `messages` + TOOL_SCHEMAS to MODEL. Return the assistant message object."""
    raise NotImplementedError


def run_agent(task: str, max_steps: int = 20) -> str:
    """Run the loop until the model answers without calling a tool, or max_steps runs out.

    Return the model's final text answer. Print each tool call and a short
    preview of its result as you go, so you can watch the agent work.
    """
    raise NotImplementedError


def main() -> None:
    """CLI entry point: `uv run harness "your task here"`."""
    raise NotImplementedError
