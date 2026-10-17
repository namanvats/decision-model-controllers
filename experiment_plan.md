# experiment_plan.md — Decision-Delegated Agent Controllers on Terminal-Bench

> **Status:** Experiment plan v0.2 (2026-10-11). Supersedes the arm list, run plan, and comparison sections of `AGENTS.md` v0.1. `AGENTS.md` remains the engineering contract (controls, safety rules, anti-patterns). Every deviation from `AGENTS.md` is listed in §14 and must also be recorded in `docs/decision_log.md`.
> **Companion:** `infra_setup.md` (how to build and run it).
> **Target output:** arXiv preprint + workshop submission (e.g. ICLR 2027 workshops; contributions due ~1 Feb 2027, exact CFPs after 29 Nov 2026).

---

## 1. Research question

**Can a cheap, fast decision model (Jev), consulted only at well-defined decision points, make a terminal agent cheaper without making it worse — and does it do better than a deterministic rule set, a small LLM asked the same question, or random advice at the same moments?**

### What we will claim (if the data supports it)
- An estimate, with uncertainty, of how all-in cost and pass rate change when a fixed agent loop is given each controller.
- Which decision points (stuck, repeated failure, finish request, checkpoint) account for the change.
- Whether Jev's confidence tracks real outcomes at those decision points.

### What we will NOT claim
- Statistical equivalence of pass rates (89 tasks cannot support a tight margin; see §10).
- Anything about Claude Code / OpenHands (dropped from the core study; scaffold comparisons are not causal).
- That results transfer to other benchmarks, models, or Jev versions.

---

## 2. Hypotheses

| ID | Hypothesis | Comparison |
|---|---|---|
| H1 | Jev lowers mean all-in USD per attempt vs. native, and the pass-rate drop is no larger than the pre-declared margin. | C1 |
| H2 | Jev matches or beats a small LLM controller on end-to-end outcome at lower controller cost and latency. | C2 |
| H3 | Jev beats deterministic rules. | C3 |
| H4 | Jev beats a placebo controller that intervenes at the same moments with random choices. | C4 |

**Pre-declared null/negative outcomes we will report as findings:** no cost change; rules ≈ Jev; small LLM ≈ Jev (consistent with REFLEX's limited advantage over a cheap generative cascade); placebo ≈ Jev (nudges matter, decisions don't); savings only from abandoning doomed runs; premature finishes reduce pass rate.

---

## 3. Design overview

One custom agent loop, five interchangeable controllers, same tasks.

| Held fixed across arms (within a reasoning model) | Varied |
|---|---|
| Reasoning model slug + pinned OpenRouter provider, temperature, reasoning/effort setting, max output tokens | Controller (`none`, `rules`, `small_llm`, `jev`, `placebo`) |
| System prompt and tool set (`bash`, `finish`) | |
| Tool-output truncation policy, no context compaction | |
| Step / USD / token / wall limits | |
| Trigger detector and guidance-message templates | |
| Sandbox spec (Daytona via Harbor), task list, attempt count | |

- **Unit of analysis:** the task. Attempts are replicates within a task, not paired with attempts in other arms (LLM sampling diverges immediately).
- **Arm order:** randomized within each task × attempt block, seeded, recorded in the manifest.
- **Benchmark:** Terminal-Bench **2.1** recommended (decide before Phase 1; never mix versions). 2.1 fixes 28 of 89 tasks, including 9 whose external dependencies drifted after 2.0 was built. If 2.0 is chosen, record the reason in the decision log and expect flaky tasks.

---

## 4. Arms

### 4.1 Core arms (primary reasoning model: Sonnet)

| Arm ID | Controller | Controller model | Purpose |
|---|---|---|---|
| `sonnet-native` | none (triggers detected and logged, never acted on) | — | Baseline |
| `sonnet-rules` | deterministic policy (§5.6) | — | Is a model needed at all? |
| `sonnet-smallllm` | small LLM, same state + options, JSON-constrained | Haiku-class via OpenRouter | The "why not a cheap LLM?" baseline |
| `sonnet-jev` | Jev choice question | `typesafe/jev-1.13` via OpenRouter | Treatment |
| `sonnet-placebo` | random choice from Jev's dev-phase choice distribution, same triggers, fixed seed | — | Separates "any nudge" from "informed nudge" |

### 4.2 Secondary arms (conditional on Phase 1 showing Qwen can pass ≥ ~20% of candidate tasks)

| Arm ID | Controller |
|---|---|
| `qwen-native` | none |
| `qwen-jev` | Jev |

### 4.3 Optional appendix reference (only if budget remains after Phase 4)
- Harbor's `terminus-2` with the same Sonnet slug, on the held-out tasks, 1 attempt. Reported only as "a neutral reference scaffold scores X on these tasks" — never compared causally.

### 4.4 Model choices (decide and freeze before Phase 1)
- **Sonnet slug:** default `anthropic/claude-sonnet-4.6` (well characterized, $3 / $15 per M on OpenRouter). Newer Sonnet releases exist on OpenRouter; choosing one is fine, but it must be frozen before Phase 1 and the reason logged.
- **Small LLM controller:** Haiku-class (default `anthropic/claude-haiku-4.5`; verify slug). Must be cheaper than the reasoning model by at least ~3×, or the comparison is meaningless.
- **Qwen:** default `qwen/qwen3-coder`, pinned to one provider and one quantization (multiple hosts serve different quantizations).
- **Jev:** `typesafe/jev-1.13`. Never use `~typesafe/jev-latest`.

---

## 5. Controller contract

### 5.1 Agent loop (identical for all arms)
- Tools: `bash(command, timeout_s)` executed in the sandbox via Harbor; `finish(summary)` to request completion.
- A text-only reply with no tool call is treated as a finish request (all arms).
- One main-model call = one step.
- Tool output truncated to a fixed budget (e.g. first 4k + last 8k chars) — same in all arms.
- No context compaction in v1 (removes a confound; cost of long histories is controlled by prompt caching and limits).

### 5.2 Triggers (detected in every arm, including `native`, and logged)

| ID | Trigger | Deterministic definition (parameters frozen after dev) |
|---|---|---|
| T1 | Repeated failure | ≥ 2 consecutive `bash` calls with non-zero exit or timeout |
| T2 | No progress | Same normalized command issued ≥ 2× in the last 5 steps, or identical output hash on ≥ 2 of the last 3 calls |
| T3 | Finish request | `finish` called or text-only reply |
| T4 | Checkpoint | Every N steps (default N = 10) |

T4 replaces `AGENTS.md`'s "implementation→verification transition", which cannot be detected deterministically (deviation D3).

Throttling: at most one controller call per step; a given trigger type cannot re-fire within 3 steps; hard cap `max_controller_calls` per trial.

### 5.3 Questions and choices
Two small choice sets (smaller action sets are more reliable for typed decision models):

**Q-progress** (asked at T1, T2, T4):

| Choice | Meaning | Effect |
|---|---|---|
| `continue` | Agent is making progress | No message |
| `investigate` | Agent is acting without enough information | Append template **I** |
| `recover` | Agent is stuck on a failing approach | Append template **R** |
| `abandon` | Trajectory is very unlikely to succeed within budget | Terminate, `termination_reason = controller_abandon` (counts as fail) — only if guard G2 passes, else treated as `recover` |

**Q-finish** (asked at T3):

| Choice | Meaning | Effect |
|---|---|---|
| `finish` | Evidence in the transcript shows the explicit requirements are met | Terminate normally |
| `verify` | Requirements are unchecked or evidence is missing | Append template **V**, continue loop |

Templates I, R, V are fixed text, versioned in `config/prompts/guidance.md`, identical across all controller arms, and prefixed `[controller guidance — advisory]`. They never contain commands, code, file paths, or anything derived from tests or solutions.

### 5.4 Compact state (what every controller sees)
```
task_instruction        (truncated to 3,000 chars)
trigger                 (T1..T4)
step / max_steps, usd_used / max_trial_usd
last_5_actions          [{command (≤300 chars), exit_code, output_tail (≤500 chars)}]
consecutive_failures, repeated_command_flag
verification_like_cmds_run   (regex: test runners, running the built artifact, diff/cmp checks)
finish_summary          (T3 only; ≤1,000 chars)
controller_history      (previous choices this trial)
```
Hard cap ~6k tokens (well under Jev's 32k context). Never includes test files, verifier output, reference solutions, or rewards.

### 5.5 Guards (override any controller)
- **G1:** hard limits (steps, USD, tokens, wall) always terminate, regardless of advice.
- **G2 (abandon):** allowed only if `step ≥ 0.4 × max_steps` AND (`consecutive_failures ≥ 3` OR T2 fired ≥ 2 times this trial).
- **G3 (verify loop):** at most 2 `verify` responses per trial; the 3rd finish request is accepted.
- **G4 (fallback):** controller timeout (2 s for Jev, 10 s for small LLM), HTTP error, or malformed output → log `controller_error`, apply `continue` (Q-progress) or `finish` (Q-finish). Never silently retried without logging and charging the cost.

### 5.6 Rules arm policy
- T1, T2 → `recover`; T4 → `continue`.
- T3 → `verify` if no verification-like command ran after the last file-modifying command, else `finish`.
- `abandon` if G2 passes and `consecutive_failures ≥ 4`.

### 5.7 Small-LLM arm parity
Same state JSON, same option names with the same one-line descriptions Jev receives as `criteria`, temperature 0, response constrained to `{"choice": "<option>"}`. Prompt tuned on dev tasks with the same effort budget as Jev's criteria wording (record iterations for both).

### 5.8 Placebo arm
Same triggers and guards. Choice sampled from Jev's empirical choice frequencies per question type, measured on dev runs, with a fixed seed. Requires dev-phase Jev runs to exist first.

---

## 6. Limits

Limits are set from data, not guessed. The `AGENTS.md` defaults (25 steps, $1) are likely too tight for Terminal-Bench and leave the controller nothing to save.

1. **Phase 1 (calibration)** runs `sonnet-native` with generous calibration caps: `max_agent_steps: 100`, `max_trial_usd: 3.00`, wall = task timeout + buffer.
2. Experiment caps are then set to roughly the **90th percentile of steps and USD among passing native runs**, rounded up, and frozen.
3. The same caps apply to every arm of a reasoning model. Qwen gets its own caps by the same rule.

---

## 7. Task selection

1. **Candidate pool:** all tasks in the chosen Terminal-Bench version, minus documented exclusions (GPU required, very large images/downloads, environment features Daytona cannot provide, or task timeouts far beyond budget). Exclusions are listed with reasons in `tasks/catalog.yaml` *before* any model runs.
2. **Candidates for calibration:** ~30 tasks sampled across categories from the pool (seeded).
3. **Calibration (Phase 1):** 2 attempts of `sonnet-native` per candidate. Each task gets a stratum: `hard` (0/2), `mid` (1/2), `easy` (2/2).
4. **Split:** from the calibrated tasks, draw **8 dev tasks** and **30–40 held-out tasks**, stratified. Cap the `hard` stratum at ~20% of held-out tasks (they inform abandon-savings but cannot reveal pass-rate harm).
5. **Discard calibration attempts** from all evaluation analyses (selecting on an outcome and reusing it inflates the baseline via regression to the mean). They are used only for stratification and limits.
6. **Freeze:** `tasks/evaluation.yaml` gets `frozen: true` and a SHA-256 checksum before Phase 4. No tuning, prompt edits, or threshold changes on held-out tasks.

---

## 8. Phases and go/no-go gates

| Phase | Work | Trials (Sonnet) | Gate to proceed |
|---|---|---|---|
| 0. Infra smoke | Oracle agent on 1–2 tasks via Daytona; custom loop on 1 task with mocked and then real model; Jev + small-LLM contract tests; cleanup audit | ~3–5 | Rewards captured, costs reconcile with OpenRouter `usage.cost`, zero orphaned sandboxes |
| 1. Calibration | §6 + §7 on ~30 candidates; Qwen native on ~15 of them (1 attempt) | ~60 | ≥ 25 tasks in `mid`+`easy`; caps frozen; Qwen go/no-go decided |
| 2. Offline controller study | Replay logged native trigger states to Jev, small LLM, rules; score against grader outcomes (§9.3) | 0 (controller calls only) | Jev beats base-rate and rules on at least one question type; otherwise reconsider before spending on Phase 4 |
| 3. Dev tuning | 8 dev tasks × {native, rules, small LLM, Jev} × 2 attempts; tune criteria wording, small-LLM prompt, trigger thresholds; then placebo × 2; freeze everything | ~80 | All configs, prompts, thresholds, margin, and task list frozen + checksummed |
| 4. Held-out evaluation | 30–40 tasks × 5 Sonnet arms × 3 attempts | ~450–600 | — |
| 5. Qwen replication (conditional) | Held-out tasks × {qwen-native, qwen-jev} × 3 | (Qwen) ~180–240 | — |
| 6. Analysis + write-up | §9–§11; arXiv; workshop | 0 | — |

If Phase 2 shows Jev cannot distinguish stuck/done states better than trivial baselines, the offline study itself is a publishable negative result; Phase 4 can be shrunk to native vs. Jev vs. small LLM.

---

## 9. Comparisons

### 9.1 Confirmatory (declared now, tested once on held-out data)

| ID | Comparison | Question | Primary metric | Guardrail |
|---|---|---|---|---|
| **C1 (primary)** | `sonnet-jev` vs `sonnet-native` | Does Jev cut cost without hurting success? | All-in USD per attempt (per-task geometric-mean ratio) | Pass-rate difference vs. margin −10 pp |

### 9.2 Secondary (reported with CIs; Holm-adjusted for the family C2–C5)

| ID | Comparison | Question | Metrics |
|---|---|---|---|
| C2 | `sonnet-jev` vs `sonnet-smallllm` | Is Jev better, cheaper, or faster than a small LLM making the same decision? | Pass rate, USD/attempt, controller USD, controller latency |
| C3 | `sonnet-jev` vs `sonnet-rules` | Is a model needed? | Pass rate, USD/attempt |
| C4 | `sonnet-jev` vs `sonnet-placebo` | Are Jev's decisions informative? | Pass rate, USD/attempt, abandon precision |
| C5 | `qwen-jev` vs `qwen-native` | Does the effect replicate on a weaker model? | Same as C1 |

### 9.3 Mechanism and decision quality (exploratory, labeled as such)

- **M1 — Offline decision quality (Phase 2 + held-out native logs):** for Q-finish states in native runs the label is the trial's grader outcome (native finish = end of trial); for Q-progress states the label is the eventual outcome. Report AUROC of P(`finish`) and P(`abandon`), calibration curves/ECE, accuracy at the argmax, for Jev, small LLM, rules, base rate. Grader output is used only for scoring, never shown to controllers.
- **M2 — Trigger breakdown:** intervention counts, choice distribution, and downstream outcome by trigger type and arm.
- **M3 — Where the savings come from:** USD saved on failed vs. passed attempts; abandon count and abandon precision (fraction of abandoned (task, arm) cells whose native pass rate is 0); verify-induced extra cost vs. pass-rate gain.
- **M4 — Guard activity:** how often G2/G3/G4 overrode the controller.
- **M5 — Overhead:** controller calls per trial, controller USD and latency as a share of totals.
- **M6 — Strata:** all C-comparisons split by calibration stratum (`easy`/`mid`/`hard`) and task category.

Individual pass→fail "flips" are not identifiable (trajectories diverge); all mechanism claims are at the distribution level.

---

## 10. Metrics and statistics

### 10.1 Per-trial metrics
`passed` (Harbor grader only), termination reason, main/controller calls, input/output/cache-read/cache-write tokens per model, `usd_main` and `usd_controller` from OpenRouter `usage.cost`, `usd_daytona` (reconciled separately; reported, not in primary metric unless attributable per trial), wall time split (provisioning, agent, verifier), model latency, controller latency, tool calls, tool failures, repeated commands, interventions by type.

### 10.2 Aggregation
- Per task and arm: pass rate = mean over attempts; USD/attempt = mean over attempts.
- **USD per solved attempt** = total arm USD / total arm passes (undefined if zero passes; reported as such).
- Failed, timed-out, budget-exhausted, and abandoned attempts stay in all denominators. Infra/API faults are re-run once and reported separately.

### 10.3 Inference
- Paired by task; 10,000 bootstrap resamples over tasks; percentile 95% CIs.
- **C1 cost:** geometric mean across tasks of (USD_jev / USD_native); CI from the bootstrap.
- **C1 pass rate:** mean per-task difference; one-sided 95% lower bound compared with −10 pp. Declared honestly: with ~36 tasks × 3 attempts, the expected half-width of the CI is roughly 8–10 pp (assuming a per-task paired SD of ~0.3), so the guardrail may be inconclusive. If it is, we report "no evidence of a large drop" rather than "noninferior".
- Pareto plots: pass rate vs. USD/attempt, vs. tokens, vs. wall time, one point per arm with CI whiskers.
- Exploratory results are visually and textually separated from confirmatory ones.

---

## 11. Budget

Planning assumption until Phase 1 replaces it: mean Sonnet trial ≈ $0.60 (all-in model cost). Ceiling uses the frozen per-trial cap C.

| Item | Trials | Expected | Ceiling |
|---|---|---|---|
| Phase 0 smoke | ~5 | ~$3 | ~$15 |
| Phase 1 calibration (Sonnet, calibration caps) | 60 | ~$36 | $180 |
| Phase 1 Qwen check | 15 | ~$2 | — |
| Phase 3 dev | 80 | ~$48 | 80 × C |
| Phase 4 held-out (36 tasks × 5 arms × 3) | 540 | ~$324 | 540 × C |
| Phase 5 Qwen (36 × 2 × 3) | 216 | ~$20 | — |
| Small-LLM controller calls (dev + eval) | — | ~$5–10 | — |
| Jev controller calls | — | < $1 | — |
| **Total model spend** | | **~$440** | depends on C |

Daytona compute is billed separately and reconciled against the Daytona dashboard.

**Minimum viable version** (if funds are short): drop rules and placebo from Phase 4 (keep them in dev). Held-out = 36 × {native, small LLM, Jev} × 3 = 324 trials (~$195 expected). C3/C4 become dev-only, exploratory observations.

Spend controls: OpenRouter key credit limit per phase (hard, provider-enforced), project ledger with admission control before every trial, per-trial caps inside the loop (see `infra_setup.md` §11).

---

## 12. Freeze checklist (before Phase 4)

- [ ] Benchmark version and dataset commit/ID
- [ ] `tasks/evaluation.yaml` frozen + checksum; exclusions documented
- [ ] Model slugs, pinned providers, quantization (Qwen), temperature, effort settings
- [ ] Limits (steps, USD, tokens, wall) per reasoning model
- [ ] Trigger parameters, throttling, guard thresholds (G2, G3, G4 timeouts)
- [ ] Guidance templates I/R/V, Jev `criteria` text, small-LLM prompt, placebo distribution + seed
- [ ] Noninferiority margin (−10 pp) and analysis script version
- [ ] Git commit and `uv.lock` hash recorded in every trial manifest

---

## 13. Threats to validity

- **Benchmark size:** 89 tasks total; small held-out set → wide CIs. Mitigated by framing cost as primary and reporting strata.
- **Selection:** task filtering by calibration outcome; mitigated by discarding calibration attempts and reporting strata.
- **Provider drift:** OpenRouter may route to different hosts; mitigated by provider pinning with fallbacks disabled and logging the serving provider per call.
- **Jev version drift:** hosted model; we pin `jev-1.13` and log the model string returned in every response. Results apply to that version only.
- **Vendor claims:** Jev's calibration claims are the vendor's; we measure calibration ourselves (M1).
- **Prompt-tuning asymmetry:** Jev criteria vs. small-LLM prompt; mitigated by equal dev effort, logged iterations.
- **Abandon inflates savings on hopeless tasks:** reported separately (M3, M6) and the `hard` stratum is capped.
- **Single scaffold:** results are for one minimal loop; production scaffolds may already handle these decisions.
- **Benchmark contamination of models:** applies equally to all arms; does not bias controller comparisons.

---

## 14. Deviations from AGENTS.md v0.1

| ID | Change | Reason |
|---|---|---|
| D1 | Claude Code / OpenHands-SDK / Opus arms removed from the core study (optional appendix reference only) | Not causal; not about Jev; saves M5 engineering |
| D2 | Added `small_llm` and `placebo` arms as core | Required baselines for C2 and C4 |
| D3 | Trigger "implementation→verification transition" replaced by periodic checkpoint T4 | Not deterministically detectable |
| D4 | Added `abandon` choice with guard G2 | Gives the controller a real cost-saving channel |
| D5 | Limits set from calibration data instead of fixed 25 steps / $1 | Defaults likely cause floor effects on Terminal-Bench |
| D6 | OpenRouter is the provider for all models including Jev; pricing taken from `usage.cost` | Single billing source; provider pinning available |
| D7 | Terminal-Bench 2.1 recommended over 2.0 | Fixes drifted dependencies and misspecified tasks |
| D8 | Offline controller study (Phase 2) added | Cheap de-risking; standalone result |

---

## 15. Paper outline (workshop length)

1. Introduction: controllers for agent loops; typed decision models.
2. Related work: early termination and agent control (EET, AgentStop, Bayesian control for coding agents), model cascades and selective prediction, REFLEX and other Jev evaluations.
3. Setup: loop, triggers, controllers, guards, benchmark, accounting.
4. Offline decision quality (M1).
5. End-to-end results (C1–C5), Pareto plots.
6. Mechanism (M2–M6).
7. Limitations (§13) and conclusion.
Appendix: full manifests, prompts/criteria, per-task tables, optional terminus-2 reference.