# infra_setup.md — Infrastructure for the Controller Study

> **Status:** v0.2 (2026-10-11). Companion to `experiment_plan.md` (what to run) and `AGENTS.md` (engineering contract).
> **Stack:** Python 3.12, `uv`, Harbor (as a locked project dependency), Daytona sandboxes, Terminal-Bench 2.1 (or 2.0), OpenRouter for every model including Jev, JSONL + DuckDB for results.
> Items marked **[verify]** depend on third-party interfaces that change; check them against the pinned version before writing code that relies on them.

---

## 1. Topology

```
Local machine (uv project: agentbench)
│
├── agentbench CLI ── plan / run / analyze / cleanup
│      │
│      └── per trial: `uv run harbor run ... --agent-import-path agentbench.harbor_adapter:ControlledAgent --env daytona`
│
├── Harbor (in-process, locked in uv.lock)
│      ├── creates a fresh Daytona sandbox from the task image
│      ├── calls ControlledAgent.run()  ← runs on the LOCAL machine
│      │       ├── reasoning model  ──► OpenRouter (Sonnet / Qwen, provider pinned)
│      │       ├── bash tool        ──► environment.exec(...) inside the sandbox
│      │       └── controller       ──► none | rules | small LLM (OpenRouter) | Jev (OpenRouter) | placebo
│      ├── runs the task verifier in the sandbox after the agent stops → reward
│      └── deletes the sandbox
│
└── results/<experiment>/<task>/<arm>/<attempt>/  (manifest, events, decisions, usage, outcome)
        └── DuckDB for analysis
```

Key property: the agent loop and all API keys stay on the local machine. The sandbox only ever receives shell commands. No OpenRouter key enters a task container.

---

## 2. Locked decisions

| Decision | Choice | Notes |
|---|---|---|
| Language / env | Python 3.12 via `uv` | Confirm Harbor's `requires-python` is satisfied **[verify]** |
| Harbor install | **Project dependency** (`uv add "harbor[daytona]"`), not `uv tool install` | Keeps the exact Harbor version in `uv.lock` so trials are replayable |
| Sandbox provider | Daytona via Harbor's `--env daytona` | Harbor owns sandbox lifecycle; we only audit |
| Benchmark | Terminal-Bench 2.1 recommended; 2.0 allowed | One version per study, chosen before Phase 1 |
| Model provider | OpenRouter for reasoning models, small-LLM controller, and Jev | One key, one billing ledger, `usage.cost` on every response |
| Agent integration | Custom **external** agent (`BaseAgent`), not an installed agent | Loop runs host-side; commands go through `environment.exec` |
| Concurrency | 1 (raise to 2 only after Phase 0 proves stable) | |

---

## 3. Accounts and prerequisites

- **Daytona:** account + API key (`DAYTONA_API_KEY`). Check your plan's per-sandbox CPU / memory / disk limits and concurrent-sandbox limit against the tasks' resource requests (from each task's `task.toml`). Note any task whose requirements exceed the plan → exclusion with reason.
- **OpenRouter:** account + API key(s) (`OPENROUTER_API_KEY`). Create **one key per phase** (smoke, calibration, dev, eval) and set a **credit limit on each key**. This is the only spend cap that is enforced by the provider rather than by our code.
- **Git** for the project and for cloning the benchmark repo (task metadata).
- **Docker locally:** optional; useful only for debugging a task image without Daytona.
- **No Anthropic, TypeSafe, or Alibaba keys needed** — everything routes through OpenRouter.

---

## 4. Project bootstrap

```bash
# 1. Create the project
uv init --package agentbench
cd agentbench
uv python pin 3.12

# 2. Runtime dependencies (pin Harbor to an exact version once chosen)
uv add "harbor[daytona]"            # then: uv add "harbor[daytona]==<X.Y.Z>"
uv add openai httpx pydantic pyyaml typer python-dotenv
uv add duckdb pandas numpy scipy matplotlib

# 3. Dev dependencies
uv add --dev pytest pytest-asyncio respx ruff

# 4. Sanity checks
uv run harbor --version
uv run harbor run --help            # confirm flags used below exist in the pinned version [verify]
```

`.env.example` (copy to `.env`, never commit `.env`):

```dotenv
OPENROUTER_API_KEY=
DAYTONA_API_KEY=
AGENTBENCH_RESULTS_DIR=results
AGENTBENCH_PHASE=smoke           # smoke | calibration | dev | eval
```

`.gitignore` must include: `.env`, `results/` (except `results/_fixtures/`), `jobs/`, `.venv/`, `__pycache__/`, `*.duckdb`.

---

## 5. Repository layout

```text
.
├── AGENTS.md
├── experiment_plan.md
├── infra_setup.md
├── README.md
├── pyproject.toml
├── uv.lock
├── .env.example
├── docs/
│   └── decision_log.md
├── agents/                         # one YAML per arm (experiment_plan.md §4)
│   ├── sonnet-native.yaml
│   ├── sonnet-rules.yaml
│   ├── sonnet-smallllm.yaml
│   ├── sonnet-jev.yaml
│   ├── sonnet-placebo.yaml
│   ├── qwen-native.yaml
│   └── qwen-jev.yaml
├── tasks/
│   ├── catalog.yaml                # all tasks + metadata + exclusions
│   ├── calibration.yaml
│   ├── development.yaml
│   └── evaluation.yaml             # frozen + checksum before Phase 4
├── experiments/
│   ├── smoke.yaml
│   ├── calibration.yaml
│   ├── dev.yaml
│   ├── eval.yaml
│   └── qwen_eval.yaml
├── config/
│   ├── benchmark.yaml              # dataset id + version/commit
│   ├── models.yaml                 # aliases → OpenRouter slugs + provider pinning
│   ├── limits.yaml                 # frozen after calibration
│   ├── triggers.yaml               # trigger + guard parameters
│   └── prompts/
│       ├── system.md
│       ├── guidance.md             # templates I / R / V
│       ├── jev_criteria.yaml       # option → description for Q-progress / Q-finish
│       └── smallllm_controller.md
├── src/agentbench/
│   ├── cli.py                      # validate | plan | run | replay | analyze | cleanup
│   ├── config.py                   # strict Pydantic schemas (unknown fields fail)
│   ├── runner.py                   # trial grid, randomization, harbor invocation, manifests
│   ├── harbor_adapter.py           # ControlledAgent(BaseAgent)
│   ├── loop.py                     # the single ReAct loop
│   ├── tools.py                    # bash + finish, output truncation
│   ├── triggers.py                 # T1–T4 detection + throttling
│   ├── state.py                    # compact state builder (size-capped, leak-checked)
│   ├── guards.py                   # G1–G4
│   ├── controllers/
│   │   ├── base.py                 # Controller protocol: decide(question, state) -> Decision
│   │   ├── none.py
│   │   ├── rules.py
│   │   ├── small_llm.py
│   │   ├── jev.py
│   │   └── placebo.py
│   ├── openrouter.py               # chat client: provider pinning, usage + cost capture
│   ├── ledger.py                   # spend admission control
│   ├── telemetry.py                # event schemas, JSONL writers
│   ├── replay.py                   # offline controller study (Phase 2)
│   ├── analysis.py                 # bootstrap, Pareto, strata
│   └── cleanup.py                  # Daytona orphan audit
├── scripts/
│   └── build_catalog.py            # reads task.toml files → tasks/catalog.yaml
└── tests/
    ├── unit/
    └── integration/
```

All arms share `loop.py`; arms differ only by YAML and which controller class is loaded.

---

## 6. Configuration

### 6.1 `config/benchmark.yaml`

```yaml
schema_version: 1
benchmark: terminal-bench
version: "2.1"                                  # or "2.0" — decide once
harbor_dataset: terminal-bench/terminal-bench-2-1   # 2.0: terminal-bench@2.0
source_repo: https://github.com/harbor-framework/terminal-bench-2-1
source_commit: CONFIGURE_ME                     # pin the commit you catalogued
```

Pin the dataset to a specific version/commit if the installed Harbor supports a version suffix for this dataset **[verify]**. Record the resolved dataset identity in every manifest either way.

### 6.2 `config/models.yaml`

```yaml
schema_version: 1
gateway:
  base_url: https://openrouter.ai/api/v1
  api_key_env: OPENROUTER_API_KEY
models:
  sonnet_main:
    slug: anthropic/claude-sonnet-4.6          # or a newer Sonnet; freeze before Phase 1
    provider_order: [anthropic]                # exact provider slug from the model page [verify]
    allow_fallbacks: false
    temperature: 0
    max_output_tokens: 8192
    reasoning: null                            # set explicitly if used; identical across paired arms
    prompt_caching: true
  qwen_main:
    slug: qwen/qwen3-coder
    provider_order: [CONFIGURE_ME]             # one host only
    quantization: CONFIGURE_ME                 # e.g. fp8 / bf16 — must match the pinned host
    allow_fallbacks: false
    temperature: 0
    max_output_tokens: 8192
  smallllm_controller:
    slug: anthropic/claude-haiku-4.5           # verify slug; any newer Haiku is fine if frozen
    provider_order: [anthropic]
    allow_fallbacks: false
    temperature: 0
    max_output_tokens: 32
  jev:
    slug: typesafe/jev-1.13                    # NEVER ~typesafe/jev-latest
    endpoint: https://openrouter.ai/api/v1/systemone
    context_tokens: 32000
```

`validate` must fail if any value is `CONFIGURE_ME`, if an arm references an unknown alias, or if paired arms (e.g. `sonnet-native` / `sonnet-jev`) differ in anything except `controller`.

### 6.3 Arm YAML (example: `agents/sonnet-jev.yaml`)

```yaml
schema_version: 2
id: sonnet-jev
scaffold: custom-react
reasoning:
  model_ref: sonnet_main
controller:
  type: jev                     # none | rules | small_llm | jev | placebo
  model_ref: jev                # only for small_llm / jev
  criteria_ref: config/prompts/jev_criteria.yaml
  fallback: safe_default        # continue (Q-progress) / finish (Q-finish)
  timeout_s: 2
triggers_ref: config/triggers.yaml
guidance_ref: config/prompts/guidance.md
limits_ref: config/limits.yaml#sonnet
tools: [bash, finish]
tool_output_truncation: {head_chars: 4000, tail_chars: 8000}
context_compaction: false
telemetry: {messages: true, actions: true, decisions: true, usage: true, trigger_states: true}
```

`sonnet-native.yaml` is identical except `controller: {type: none}`. Triggers are still detected and their compact states logged (`trigger_states: true`) — this feeds the offline study.

### 6.4 Experiment YAML (example: `experiments/eval.yaml`)

```yaml
schema_version: 2
id: tb21-eval-v1
phase: eval
tasks: tasks/evaluation.yaml
arms:
  - agents/sonnet-native.yaml
  - agents/sonnet-rules.yaml
  - agents/sonnet-smallllm.yaml
  - agents/sonnet-jev.yaml
  - agents/sonnet-placebo.yaml
attempts: 3
randomize_arm_order: true
seed: 20261011
concurrency: 1
budget:
  hard_cap_usd: 600
  soft_warning_usd: 450
require_frozen_tasks: true
```

---

## 7. OpenRouter integration (reasoning models + small-LLM controller)

Use the OpenAI-compatible SDK pointed at OpenRouter; it gives direct control over provider routing and usage capture.

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1",
                     api_key=os.environ["OPENROUTER_API_KEY"])

resp = await client.chat.completions.create(
    model=m.slug,
    messages=messages,
    tools=TOOLS,                      # bash, finish
    temperature=m.temperature,
    max_tokens=m.max_output_tokens,
    extra_body={
        "provider": {
            "order": m.provider_order,
            "allow_fallbacks": False,
            "require_parameters": True,   # only hosts that support tools [verify]
            # "quantizations": [m.quantization],   # Qwen only [verify]
        },
        "usage": {"include": True},       # harmless if already default [verify]
    },
)
```

Record for **every** call: request id, `model` and serving `provider` returned in the response, prompt / completion / cached-read / cache-write tokens, `usage.cost` (USD), latency, retry count. If a retry happens, the failed attempt's cost (if any) is also charged.

**Prompt caching (Anthropic models):** mark the system prompt and the stable conversation prefix with `cache_control` breakpoints so long trajectories don't re-bill the full history **[verify OpenRouter's current syntax]**. Controller guidance is appended at the end of the conversation, so it does not break the cached prefix. Caching is configured identically in all arms of a reasoning model.

**Provider pinning is mandatory.** Sonnet 4.6 is served on OpenRouter by several hosts (Anthropic, Vertex, Bedrock, Azure, Claude Platform on AWS); Qwen3 Coder by several hosts with different quantizations. With fallbacks on, two arms could silently run on different hosts. If the pinned host is down, the trial fails as `infra_error` and is re-run later — never re-routed.

---

## 8. Jev integration (via OpenRouter)

- **Model:** `typesafe/jev-1.13`. No TypeSafe account or key; the OpenRouter key authenticates.
- **Endpoints:** System One API `POST https://openrouter.ai/api/v1/systemone` (TypeSafe-SDK compatible, preferred) or Decisions API `POST https://openrouter.ai/api/alpha/decisions` (alpha — may change).
- **Context:** 32k tokens for `state` + `questions`; our compact state is capped at ~6k.
- **Billing:** input tokens only; every response includes `usage.cost`.

Request shape (field names **[verify]** against https://docs.typesafe.ai/primitives/choice before coding):

```python
payload = {
    "model": "typesafe/jev-1.13",
    "state": compact_state,                       # dict from state.py
    "questions": {
        "next_step": {                            # Q-progress
            "type": "choice",
            "criteria": {
                "continue":    "The agent is making progress toward the task goal.",
                "investigate": "The agent is acting without enough information about the environment.",
                "recover":     "The agent is stuck repeating a failing approach.",
                "abandon":     "The task is very unlikely to be completed within the remaining budget.",
            },
        }
    },
}
r = await http.post("https://openrouter.ai/api/v1/systemone",
                    headers={"Authorization": f"Bearer {key}"},
                    json=payload, timeout=2.0)
```

Log per call: selected option, full probability distribution, confidence, returned model string, `usage.cost`, latency, HTTP status. Any timeout/error/malformed response → guard G4 fallback + `controller_error` event.

The small-LLM controller receives the same `state` and the same option descriptions in a prompt, and must return `{"choice": "<option>"}` (validated; invalid → G4).

---

## 9. Harbor integration

### 9.1 The custom agent

External agents implement `BaseAgent`; the method names below follow Harbor's docs but signatures change between versions **[verify against the pinned `harbor/agents/base.py`]**.

```python
from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

class ControlledAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        return "agentbench-controlled"

    def version(self) -> str:
        return agentbench.__version__          # + git commit in the manifest

    async def setup(self, environment: BaseEnvironment) -> None:
        pass                                   # nothing installed in the sandbox

    async def run(self, instruction: str, environment: BaseEnvironment,
                  context: AgentContext) -> None:
        arm = load_arm(os.environ["AGENTBENCH_ARM_YAML"])   # see 9.2
        await run_loop(arm, instruction, environment, context)
```

The `bash` tool calls `environment.exec(command, timeout_sec=...)` and returns exit code + truncated stdout/stderr. Populate `AgentContext` with token counts / cost where the pinned Harbor version supports it, but treat **our own telemetry** as the source of truth for usage and cost.

### 9.2 Passing the arm to the agent

The runner invokes Harbor **once per trial** (task × arm × attempt), which keeps manifests simple and randomization exact:

```bash
AGENTBENCH_ARM_YAML=agents/sonnet-jev.yaml \
AGENTBENCH_TRIAL_ID=<uuid> \
uv run harbor run \
  -d terminal-bench/terminal-bench-2-1 \
  --include-task-name <task-name> \
  --agent-import-path agentbench.harbor_adapter:ControlledAgent \
  --env daytona \
  -n 1 -k 1
```

If the pinned Harbor version supports agent kwargs in a generated job config (`harbor run --config job.yaml`), prefer that over environment variables and store the generated config in the manifest **[verify]**. Either way, the full command/config is saved before execution.

### 9.3 Reading results

`passed` comes only from Harbor's verifier reward. After each trial, the runner parses Harbor's job/trial output directory (structure **[verify]** in the pinned version), copies the result and any trajectory into `results/.../harbor_result.json`, and writes `outcome.json`.

### 9.4 Smoke commands (Phase 0)

```bash
# 1. Oracle solutions, 1 task, Daytona — proves dataset download, sandbox, verifier
uv run harbor run -d terminal-bench/terminal-bench-2-1 -a oracle -l 1 --env daytona -n 1

# 2. Custom agent with a mocked model on the same task (no API spend)
AGENTBENCH_ARM_YAML=agents/_mock.yaml uv run harbor run -d terminal-bench/terminal-bench-2-1 \
  --include-task-name <task> --agent-import-path agentbench.harbor_adapter:ControlledAgent \
  --env daytona -n 1 -k 1

# 3. Same, real Sonnet, native arm
AGENTBENCH_ARM_YAML=agents/sonnet-native.yaml uv run harbor run ...
```

### 9.5 Building the task catalog

```bash
git clone https://github.com/harbor-framework/terminal-bench-2-1.git vendor/tb21
git -C vendor/tb21 checkout <commit>                # record in config/benchmark.yaml
uv run python scripts/build_catalog.py vendor/tb21 > tasks/catalog.yaml
```

`build_catalog.py` reads each task's `task.toml` (category, timeouts, CPU/memory, GPU flags) and writes one entry per task with `excluded_reason` filled in for anything Daytona or the budget can't support. Never copy tests or solutions into this repo.

---

## 10. Daytona

- Harbor creates, uses, and deletes sandboxes; our code does not create sandboxes.
- One fresh sandbox per trial; nothing shared between arms or attempts.
- Before Phase 1, compare each candidate task's resource request with your plan limits; mismatches → exclusion.
- Some tasks need outbound internet at run time; don't apply egress restrictions that break them. Never pass API keys into the sandbox (the external-agent design makes this automatic).
- **Orphan audit** (`agentbench cleanup --orphaned --dry-run`): list sandboxes on the account via the Daytona SDK, match them against trial IDs/labels recorded by the runner, and delete only matched orphans after confirmation. Stopped ≠ deleted — stopped sandboxes keep their disk and may keep billing **[verify Daytona SDK package name and labeling support]**.
- Run the audit after every phase and after any interrupted run.
- Reconcile Daytona usage from its dashboard per phase; report it separately from model spend.

---

## 11. Spend control (three layers)

| Layer | Mechanism | Enforced by |
|---|---|---|
| 1. Provider | Credit limit on the phase's OpenRouter key | OpenRouter (hard) |
| 2. Project | `ledger.py`: before each trial, `spent + reserved_in_flight + trial_cap ≤ hard_cap_usd`, else stop admitting trials | Our code |
| 3. Trial | Loop checks steps / USD / tokens / wall after every model call; G1 terminates | Our code |

The ledger reads actual `usage.cost` values, not price-table estimates. `config/pricing.yaml` is kept only for `plan` (pre-run estimates) and must carry a date.

---

## 12. Telemetry and storage

Per trial directory `results/<experiment>/<task>/<arm>/<attempt>/`:

| File | Contents |
|---|---|
| `manifest.json` | Trial id, git commit, `uv.lock` hash, Harbor version, dataset id/commit, task id, arm YAML (inlined), resolved model slugs + providers, limits, trigger params, seed, Harbor command/config, start time |
| `events.jsonl` | Every model call, tool call, trigger, guard activation, termination |
| `trigger_states.jsonl` | Compact state at every trigger (all arms, including native) |
| `decision_events.jsonl` | Controller question, response, distribution, latency, cost, fallback flag, applied action |
| `usage.json` | Per-model token and cost totals (main vs. controller), cache categories |
| `harbor_result.json` | Copied Harbor output |
| `outcome.json` | `passed`, reward, termination reason, wall-time breakdown |

`agentbench analyze` loads all trials into DuckDB, then writes `summary.csv`, `per_task.csv`, `comparisons.csv`, `report.md`, and `figures/`.

`agentbench replay` (Phase 2) reads `trigger_states.jsonl` from native runs, sends each state to Jev / small LLM / rules offline, and writes decision-quality tables. It is the only place grader outcomes and controller outputs are joined.

---

## 13. Milestones and acceptance tests

| Milestone | Deliverable | Acceptance |
|---|---|---|
| M0 Contracts | Project skeleton, schemas, all YAML, `validate`, `plan` (offline, zero spend) | `validate` fails on `CONFIGURE_ME`, unknown aliases, mismatched paired arms, unfrozen eval tasks |
| M1 Harbor + Daytona | Oracle smoke; orphan audit | Reward captured; 0 orphans after success, exception, Ctrl-C, timeout |
| M2 Loop | `loop.py`, tools, triggers, native arm; mocked-model unit tests | Mocked runs hit every termination path; trigger states logged; costs match `usage.cost` on 1 real trial |
| M3 Controllers | rules, small LLM, Jev, placebo, guards G1–G4 | Fixture tests: malformed Jev response, timeout, `abandon` blocked by G2, 3rd finish accepted by G3, controller output can't alter tools/prompts |
| M4 Ledger + runner | Trial grid, randomization, resumability, ledger | Ledger refuses a trial that would cross the cap; re-running skips completed trials only |
| M5 Replay + analysis | `replay`, `analyze`, figures | Synthetic-data test reproduces known effect sizes; per-task bootstrap correct |

---

## 14. Verify-before-coding checklist

- [ ] Harbor version chosen and pinned; `harbor run --help` flags (`--include-task-name`, `--agent-import-path`, `--env`, `-k`, `-n`, `-l`, `--config`) confirmed
- [ ] `BaseAgent` method signatures and `environment.exec` arguments confirmed in pinned source
- [ ] Harbor result directory structure confirmed
- [ ] Dataset id resolves (`terminal-bench/terminal-bench-2-1` or `terminal-bench@2.0`) and version/commit pinning approach decided
- [ ] OpenRouter provider slugs for Sonnet (Anthropic host), Haiku, and the chosen Qwen host confirmed on each model page
- [ ] OpenRouter `provider` options (`order`, `allow_fallbacks`, `require_parameters`, `quantizations`) and caching syntax confirmed
- [ ] Jev request/response field names confirmed; `typesafe/jev-1.13` returns a stable model string
- [ ] Daytona plan limits vs. task resource requests; Daytona SDK package for orphan audit
- [ ] Credit limits set on every OpenRouter key

---

## 15. Known pitfalls

- **Silent provider re-routing** on OpenRouter when fallbacks are enabled → different hosts across arms.
- **Qwen quantization** differs by host; pin both host and quantization.
- **Version aliases** (`~typesafe/jev-latest`, unversioned model names) break reproducibility.
- **Terminal-Bench 2.0 drift:** several 2.0 tasks depend on external resources that changed after release; 2.1 fixes these.
- **Stopped Daytona sandboxes** still exist; only deletion frees them.
- **Harbor's LiteLLM model strings** (`openrouter/...`) matter only for Harbor's built-in agents (e.g. the optional `terminus-2` reference); our custom agent calls OpenRouter directly.
- **Cost from token counts × price table** will drift from reality (caching, provider pricing); always use `usage.cost`.