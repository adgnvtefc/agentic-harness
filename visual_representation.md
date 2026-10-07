# How the harness works

A picture of the agent as it exists today, kept up to date as the code changes.
Diagrams are [Mermaid](https://mermaid.js.org/), which GitHub renders automatically.

**Last updated:** 2026-10-07, at commit `73b68b5` (policy layer, CLI flags, approver messages)

---

## 1. The big picture

One command builds everything the agent needs, then hands it over. The orchestrator
wires things together; the agent loop does the work; every tool call passes through
one checkpoint.

```mermaid
flowchart TD
    user(["You<br/>uv run harness 'task' --agent base --unattended"])

    subgraph control["Control plane: orchestrator.py"]
        direction LR
        cli["main()<br/>parses flags"] --> run["run()"]
    end

    subgraph inputs["Config files"]
        direction LR
        agentmd["agents/base.md<br/>model, temperature, max_steps,<br/>tools, system prompt"]
        env[".env<br/>OMLX_API_KEY"]
    end

    subgraph built["Built fresh for each run"]
        direction LR
        cfg["config dict"]
        client["OpenAI client"]
        policy["Policy<br/>terminal_approver, or<br/>deny_all if --unattended"]
        toolset["Toolset<br/>only the tools this agent lists"]
        policy --> toolset
    end

    agentloop["Agent loop<br/>agent.py: run_agent()"]
    omlx[("oMLX server<br/>Qwen3.8-27B")]
    ws[("workspace/<br/>the only folder tools can touch")]

    user --> cli
    run -->|reads| inputs
    inputs -->|"load_agent_config, make_client"| built
    built -->|handed to| agentloop
    agentloop <-->|"POST /v1/chat/completions"| omlx
    agentloop <-->|"every tool call"| toolset
    toolset <--> ws
```

---

## 2. One run, step by step

The model is stateless: every turn sends the **whole** conversation. The loop ends
when the model replies without asking for a tool, or when `max_steps` runs out.

```mermaid
sequenceDiagram
    autonumber
    actor You
    participant O as Orchestrator
    participant A as Agent loop
    participant M as Model (oMLX)
    participant T as Toolset
    participant P as Policy
    participant F as Tool function

    You->>O: harness "write primes.py and run it"
    O->>O: load config, make client,<br/>build Policy + Toolset
    O->>A: run_agent(task, config, client, toolset)
    A->>A: messages = [system, user task]<br/>tools = toolset.schemas()

    loop each step, up to max_steps
        A->>M: messages + tool schemas
        M-->>A: assistant message
        A->>A: append it to messages<br/>(reasoning_content dropped)
        alt no tool_calls
            A-->>O: final text answer
            O-->>You: print answer
        else has tool_calls
            loop each tool call, in order
                A->>T: execute(name, arguments JSON)
                T->>P: check(tool, parsed args)
                alt permission is allow
                    P-->>T: None (go)
                else permission is ask
                    P->>You: approver: y/N prompt<br/>(or deny_all: no prompt)
                    P-->>T: None, or a refusal message
                end
                opt approved
                    T->>F: fn(**arguments)
                    F-->>T: result text
                end
                T-->>A: result or error, always a string
                A->>A: append {role: tool, tool_call_id, content}
            end
        end
    end
    Note over A: max_steps reached: return<br/>"Stopped after N steps"
```

---

## 3. Inside `Toolset.execute`: the one checkpoint

Every tool call goes through these checks in order. **Nothing here raises:** each
failure becomes text the model reads next turn, so it can fix its own mistake.
Malformed calls are rejected *before* a human is ever asked.

```mermaid
flowchart TD
    start(["execute(name, arguments_json)"]) --> known{"name in this<br/>agent's toolset?"}
    known -->|no| e1["Error: unknown tool<br/>+ list of available tools"]
    known -->|yes| parse{"arguments are<br/>valid JSON?"}
    parse -->|no| e2["Error: not valid JSON<br/>+ example of valid JSON"]
    parse -->|yes| obj{"a JSON object?"}
    obj -->|no| e3["Error: must be a JSON object"]
    obj -->|yes| perm{"tool.permission"}
    perm -->|allow| runfn
    perm -->|ask| approver{"approver"}
    approver -->|"terminal: you typed y"| runfn
    approver -->|"terminal: anything else"| r1["The user declined this call.<br/>Try a different approach."]
    approver -->|"deny_all (--unattended)<br/>or no terminal"| r2["Unavailable in this run.<br/>Don't retry; say what you couldn't do."]
    runfn["tool.fn(**arguments)"] --> ok{"raised?"}
    ok -->|no| result["result as a string"]
    ok -->|TypeError| e4["Error: bad arguments"]
    ok -->|other exception| e5["Error: ExceptionType: message"]

    classDef err fill:#fde2e2,stroke:#c0392b,color:#000
    classDef refuse fill:#fff3cd,stroke:#b7950b,color:#000
    classDef good fill:#d5f5e3,stroke:#1e8449,color:#000
    class e1,e2,e3,e4,e5 err
    class r1,r2 refuse
    class result good
```

---

## 4. The tools

| Tool | Permission | What it does | Guardrails |
|---|---|---|---|
| `read_file` | allow | Return a file's text | Paths can't leave `workspace/`; binary/empty files say so; long output truncated (head + tail) |
| `write_file` | allow | Create or overwrite a file | Same path rule; creates parent folders |
| `bash` | **ask** | Run a shell command in `workspace/` | Approval first; stdin closed; timeout kills the whole process group; output truncated; exit code always reported |

An agent only gets the tools its `.md` file lists. A tool it wasn't given can't run,
even if the model invents a call to it.

---

## 5. What the model sees: the conversation grows

From a real run of `"Write primes.py that prints the first 10 primes, run it"`:

```text
request 1: [system] [user]
request 2: [system] [user] [assistant: write_file] [tool: Wrote 381 characters]
request 3: [system] [user] [assistant: write_file] [tool: ...] [assistant: bash] [tool: [2, 3, 5, ... 29] (exit code 0)]
answer:    "Done! I created primes.py and ran it. The output was ..."
```

History is append-only: each request starts with the previous one, unchanged. That's
what lets oMLX reuse its prompt cache instead of re-reading the whole conversation.

---

## 6. How the code is organized

Arrows mean "imports". They only point one way: nothing imports the orchestrator,
tools never import the machinery around them, and `policy.py` imports nothing at runtime.

```mermaid
flowchart TD
    orch["orchestrator.py<br/>control plane, CLI"] --> agent["agent.py<br/>the loop"]
    orch --> loader["config_loader.py<br/>reads agents/*.md"]
    orch --> policy["policy.py<br/>Policy, approvers"]
    orch --> tools_api["tools/__init__.py<br/>public API"]
    agent --> tools_api
    tools_api --> toolset["tools/toolset.py<br/>REGISTRY + Toolset"]
    toolset --> policy
    toolset --> builtin["tools/builtin/<br/>fs.py, shell.py"]
    toolset --> base["tools/base.py<br/>Tool, WORKSPACE, truncate"]
    builtin --> base
    base -.->|"Permission type only"| policy
```

---

## Not built yet

What's planned, in rough order. Each becomes a diagram change when it lands.

- **Trace logging:** every request, response, and tool call written to a file per run.
- **More tools:** `list_dir`, `edit`, eventually `web_search` and MCP servers.
- **Sandbox:** `bash` runs in a container; then a `--yes` mode can auto-approve safely.
- **Context management:** compaction when the conversation nears the 32k window.
- **Provider adapters:** oMLX and native Anthropic behind one interface.
