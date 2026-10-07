"""The agent loop: call the model, run any tool calls, feed results back, repeat."""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

from harness.tools import TOOL_SCHEMAS, execute_tool

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
    # This becomes: POST {base_url}/chat/completions with a JSON body like
    #   {
    #     "model": "Qwen3.8-27B-MLX-8bit",       which model on the server (one server can host many)
    #     "messages": [                           the ENTIRE conversation so far; the model is stateless
    #       {"role": "system", "content": "..."},
    #       {"role": "user", "content": "..."},
    #       {"role": "assistant", "content": None, "tool_calls": [...]},
    #       {"role": "tool", "tool_call_id": "call_1", "content": "..."},
    #     ],
    #     "tools": [                              menu of functions the model MAY ask us to run
    #       {"type": "function", "function": {"name": ..., "description": ..., "parameters": <JSON Schema>}},
    #     ],
    #     "temperature": 0.6                      randomness; lower = more consistent tool calls
    #   }
    kwargs = {
        "model": os.environ.get("MODEL", "Qwen3.8-27B-MLX-8bit"),
        "messages": messages,
        "temperature": 0.6,
    }
    # Some servers reject an empty "tools" list, so only send it once tools exist.
    if TOOL_SCHEMAS:
        kwargs["tools"] = TOOL_SCHEMAS
    response = client.chat.completions.create(**kwargs)

    # The response JSON looks like
    #   {
    #     "id": "...", "model": "...",
    #     "choices": [                            list because you can request n>1 samples; we use 1
    #       {
    #         "index": 0,
    #         "finish_reason": "stop" | "tool_calls" | "length",   why the model stopped generating
    #         "message": {
    #           "role": "assistant",
    #           "content": "text answer" or None,
    #           "tool_calls": [                   present only when the model wants tools run
    #             {"id": "call_1", "type": "function",
    #              "function": {"name": "bash", "arguments": "{\"command\": \"ls\"}"}},   arguments is a JSON STRING
    #           ],
    #         },
    #       }
    #     ],
    #     "usage": {"prompt_tokens": ..., "completion_tokens": ...},
    #   }
    # The SDK parses this into Python objects, so fields are attributes: response.choices[0].message.content
    # NOTE: for qwen, also have reasoning_content
    return response.choices[0].message


def run_agent(task: str, max_steps: int = 20) -> str:
    """Run the loop until the model answers without calling a tool, or max_steps runs out.

    Return the model's final text answer. Print each tool call and a short
    preview of its result as you go, so you can watch the agent work.
    """

    # creates a client to send request to
    client = make_client()

    # The conversation IS the agent's memory. Every turn we send all of it.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]

    for step in range(1, max_steps + 1):
        # calls the client with model and these messages
        msg = call_model(client, messages)

        # Append the assistant turn exactly as returned, tool_calls and their ids included,
        # so the tool results we add next have something to point back to.
        # reasoning_content is dropped: smaller context, and earlier messages stay byte-identical,
        # which keeps the server's prompt cache hitting. (Keeping it is a Phase 3 experiment.)
        messages.append(msg.model_dump(exclude_none=True, exclude={"reasoning_content"}))

        # the model needs to make tool calls for there to be indicated a final response
        if not msg.tool_calls:
            return msg.content or ""

        # The model may request several tools in one turn; each needs exactly one reply.
        for call in msg.tool_calls:
            # prints function call name and arguments
            print(f"[step {step}] {call.function.name}({call.function.arguments})")
            # executes teh tool
            result = execute_tool(call.function.name, call.function.arguments)
            # determines preview to print
            preview = result if len(result) <= 200 else result[:200] + "..."
            print(f"  -> {preview}")
            # give tool full context id
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    return f"Stopped after {max_steps} steps without a final answer."


def main() -> None:
    """CLI entry point: `uv run harness "your task here"`."""
    # this allows us to pass in command line arguments without quotes
    task = " ".join(sys.argv[1:]).strip()
    if not task:
        raise SystemExit('usage: uv run harness "your task here"')
    print(run_agent(task))
