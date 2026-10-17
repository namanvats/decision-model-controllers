# Decision log

Every deviation from `AGENTS.md` / `experiment_plan.md` / `infra_plan.md` and every frozen choice goes here, dated.

| Date | Decision | Reason |
|---|---|---|
| 2026-10-11 | Benchmark: **Terminal-Bench 2.1**, hub dataset `terminal-bench/terminal-bench-2-1@sha256:7d7bdc1c…a0699a` (89 tasks); catalog built from `harbor-framework/terminal-bench-2-1@7131e437` | Plan §3/D7 recommendation; pinned by content digest so the dataset cannot drift under `latest` |
| 2026-10-11 | Harbor pinned to **0.24.0** (`harbor[daytona]`, Daytona SDK 0.223.1) | Locked in `uv.lock`; interfaces verified against installed source |
| 2026-10-11 | Reasoning model **`anthropic/claude-sonnet-5.5`** ($2/$10), `reasoning: {effort: high}`, automatic prompt caching | Chosen by the PI. Reasoning is mandatory on Sonnet 5.5 (default `high`), so the effort is pinned explicitly |
| 2026-10-11 | **No `temperature`** sent to Claude 5.5 models | No Claude 5.5 endpoint lists `temperature` in `supported_parameters`; with `require_parameters: true` OpenRouter rejected every endpoint (404 "No endpoints found"). Sampling is therefore not greedy; attempts are replicates (§3) |
| 2026-10-11 | Small-LLM controller **`anthropic/claude-haiku-5.5`** ($0.10/$0.50, 20× cheaper than Sonnet 5.5), `reasoning: {enabled: false}` | Clears the ≥3× cost-gap rule. Reasoning is optional on Haiku 5.5 but defaults to `medium`, which would add latency and cost for a one-word choice |
| 2026-10-11 | Qwen arm model **`qwen/qwen3.8-max-0902`** | Chosen by the PI. Reasoning is mandatory (default `xhigh`); the effort must be pinned before Phase 1. Priced $2/$6, so it is not a cheap model; Phase 5 budget estimates change accordingly |
| 2026-10-11 | Jev `typesafe/jev-1.13` via `POST /api/v1/systemone`; state capped at ~6k tokens | Sources disagree on context (32k vs 64k); our cap is far below either. Version aliases (`~…`, `*-latest`) are rejected by config validation |
| 2026-10-11 | Agent wired with `-a module:Class` and `--ak` kwargs (`ControlledAgentOptions`) instead of `--agent-import-path` + env vars | `--agent-import-path` is deprecated in Harbor 0.24; kwargs are validated by Harbor and recorded in the job config |
| 2026-10-11 | Compact state adds one field, `verified_after_last_edit` (shown to every controller) | Needed by the rules arm's T3 policy (§5.6); identical for all controllers so it is not an asymmetry |
| 2026-10-11 | T2 output-hash repetition ignores empty outputs | Otherwise silent commands (`mkdir`, `cd …&&`) trivially fire T2 |
| 2026-10-11 | Tool calls run with `environment.exec` in a fresh shell each time; the system prompt says so | Harbor's Daytona exec opens a new session per call (cwd/env do not persist) |
| 2026-10-11 | Catalog exclusions: agent timeout > 3600 s (`build-pov-ray`, `sam-cell-seg`); no GPU tasks exist | Task timeouts far beyond budget (§7.1). Re-run `build_catalog.py` with your Daytona plan's CPU/memory limits before Phase 1 |
| 2026-10-11 | Infra classification: `infra_error` = no Harbor result, no reward, non-timeout Harbor exception, or loop `agent_error`; Harbor `AgentTimeoutError` counts as a normal failure (`harbor_timeout`) | §10.2: infra faults re-run once and reported separately |
| 2026-10-11 | Concurrency: experiments default to 1; `AGENTBENCH_CONCURRENCY` overrides it (`.env.example`: 16). The runner pre-downloads the pinned dataset into Harbor's cache before starting parallel trials; the effective value is recorded in every manifest | infra_plan §2 suggested 1→2; 10–20 is needed to finish the 540-trial eval in about a day. Arm order is randomized, so any latency effect from parallel load hits all arms equally. Keep one value per phase |
| 2026-10-11 | **No provider or quantization pinning** for any model (Sonnet, Haiku, Qwen, Jev); only `require_parameters: true` is sent | PI decision: models matter, hosts do not. Deviates from infra_plan §7 / experiment_plan §13 (provider drift). Mitigation: the serving host is logged on every call and reported per arm in `report.md`; `require_parameters` keeps hosts that would drop tools, structured output, or reasoning out of the pool |
