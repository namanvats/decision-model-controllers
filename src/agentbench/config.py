import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLACEHOLDER = "CONFIGURE_ME"
QUESTIONS = ("progress", "finish")


class ConfigError(Exception):
    pass


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelCfg(Strict):
    kind: Literal["openrouter", "mock"] = "openrouter"
    slug: str
    temperature: float | None = None  # None: not sent (Claude 5.x endpoints reject it)
    max_output_tokens: int = 8192
    reasoning: dict[str, Any] | None = None
    prompt_caching: bool = False


class JevCfg(Strict):
    slug: str
    endpoint: str
    context_tokens: int


class Gateway(Strict):
    base_url: str
    api_key_env: str


class ModelsFile(Strict):
    schema_version: int
    gateway: Gateway
    models: dict[str, dict[str, Any]]  # validated per alias on use, so unused models may be unset
    jev: JevCfg


class Limits(Strict):
    max_agent_steps: int
    max_trial_usd: float
    max_total_tokens: int
    max_wall_s: float
    max_controller_calls: int


class TriggerCfg(Strict):
    schema_version: int
    t1_consecutive_failures: int
    t2_command_window: int
    t2_command_repeats: int
    t2_output_window: int
    t2_output_repeats: int
    t4_every_n_steps: int
    refire_cooldown_steps: int
    g2_min_step_fraction: float
    g2_min_consecutive_failures: int
    g2_min_t2_fires: int
    g3_max_verify: int
    rules_abandon_min_failures: int


class ControllerCfg(Strict):
    type: Literal["none", "rules", "small_llm", "jev", "placebo"]
    model_ref: str | None = None
    criteria_ref: str | None = None
    prompt_ref: str | None = None
    distribution_ref: str | None = None
    timeout_s: float | None = None
    finish_threshold: float | None = None  # jev: verify only when P(finish) is below this
    questions: list[Literal["progress", "finish"]] | None = None  # None: consulted on both


class ReasoningRef(Strict):
    model_ref: str


class Truncation(Strict):
    head_chars: int
    tail_chars: int


class ArmCfg(Strict):
    schema_version: int
    id: str
    scaffold: Literal["custom-react"]
    reasoning: ReasoningRef
    controller: ControllerCfg
    system_prompt_ref: str
    triggers_ref: str
    guidance_ref: str
    limits_ref: str
    tools: list[Literal["bash", "finish"]]
    tool_output_truncation: Truncation
    context_compaction: Literal[False]


class Budget(Strict):
    hard_cap_usd: float
    soft_warning_usd: float


class ExperimentCfg(Strict):
    schema_version: int
    id: str
    phase: Literal["smoke", "calibration", "dev", "eval"]
    tasks: str
    arms: list[str]
    attempts: int
    randomize_arm_order: bool
    seed: int
    concurrency: int = 1
    limits_ref: str | None = None
    budget: Budget
    require_frozen_tasks: bool


class TaskEntry(Strict):
    name: str
    category: str | None = None
    stratum: Literal["easy", "mid", "hard"] | None = None


class TaskList(Strict):
    schema_version: int = 1
    frozen: bool = False
    checksum: str | None = None
    tasks: list[TaskEntry]


class QuestionSpec(Strict):
    instructions: str
    criteria: dict[str, str]


class Criteria(Strict):
    schema_version: int
    progress: QuestionSpec
    finish: QuestionSpec


class PlaceboDist(Strict):
    schema_version: int
    seed: int
    source: str
    progress: dict[str, float]
    finish: dict[str, float]


def resolve(ref: str | Path) -> Path:
    path = Path(ref)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_yaml(path: str | Path) -> Any:
    path = resolve(path)
    if not path.exists():
        raise ConfigError(f"missing file: {path.relative_to(PROJECT_ROOT, walk_up=True)}")
    return yaml.safe_load(path.read_text())


def read_ref(ref: str) -> Any:
    """Read `file.yaml` or `file.yaml#key`."""
    file, _, key = ref.partition("#")
    data = read_yaml(file)
    if key:
        if key not in data:
            raise ConfigError(f"{ref}: unknown key '{key}'")
        return data[key]
    return data


def placeholders(obj: Any, where: str) -> list[str]:
    if obj == PLACEHOLDER:
        return [where]
    if isinstance(obj, dict):
        return [p for k, v in obj.items() for p in placeholders(v, f"{where}.{k}")]
    if isinstance(obj, list):
        return [p for i, v in enumerate(obj) for p in placeholders(v, f"{where}[{i}]")]
    return []


def parse[T: BaseModel](model: type[T], data: Any, where: str) -> T:
    if found := placeholders(data, where):
        raise ConfigError(f"{PLACEHOLDER} not filled in: {', '.join(found)}")
    try:
        return model.model_validate(data)
    except ValidationError as e:
        raise ConfigError(f"{where}: {e}") from e


def load_models() -> ModelsFile:
    return ModelsFile.model_validate(read_yaml("config/models.yaml"))


def load_model(alias: str) -> ModelCfg:
    models = load_models().models
    if alias not in models:
        raise ConfigError(f"unknown model alias '{alias}' (known: {sorted(models)})")
    return _pinned(parse(ModelCfg, models[alias], f"models.{alias}"))


def _pinned[M: (ModelCfg, JevCfg)](cfg: M) -> M:
    if cfg.slug.startswith("~") or cfg.slug.endswith("-latest"):
        raise ConfigError(f"{cfg.slug}: version aliases are not reproducible; pin a version")
    return cfg


def load_guidance(ref: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current = None
    for line in resolve(ref).read_text().splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            sections[current] = []
        elif current:
            sections[current].append(line)
    guidance = {k: "\n".join(v).strip() for k, v in sections.items()}
    if set(guidance) != {"I", "R", "V"}:
        raise ConfigError(f"{ref}: expected sections I, R, V; got {sorted(guidance)}")
    return guidance


@dataclass(frozen=True)
class ResolvedArm:
    cfg: ArmCfg
    path: Path
    model: ModelCfg
    limits: Limits
    limits_ref: str
    triggers: TriggerCfg
    system_prompt: str
    guidance: dict[str, str]
    controller_model: ModelCfg | None = None
    jev: JevCfg | None = None
    criteria: Criteria | None = None
    controller_prompt: str | None = None
    placebo: PlaceboDist | None = None

    @property
    def id(self) -> str:
        return self.cfg.id

    @property
    def controller_type(self) -> str:
        return self.cfg.controller.type

    def snapshot(self) -> dict[str, Any]:
        return {
            "arm": self.cfg.model_dump(),
            "model": self.model.model_dump(),
            "limits_ref": self.limits_ref,
            "limits": self.limits.model_dump(),
            "triggers": self.triggers.model_dump(),
            "controller_model": self.controller_model and self.controller_model.model_dump(),
            "jev": self.jev and self.jev.model_dump(),
            "criteria": self.criteria and self.criteria.model_dump(),
            "controller_prompt": self.controller_prompt,
            "placebo": self.placebo and self.placebo.model_dump(),
            "system_prompt": self.system_prompt,
            "guidance": self.guidance,
        }


def load_arm(path: str | Path, limits_ref: str | None = None) -> ResolvedArm:
    path = resolve(path)
    cfg = parse(ArmCfg, read_yaml(path), path.name)
    limits_ref = limits_ref or cfg.limits_ref
    ctrl = cfg.controller
    extra: dict[str, Any] = {}

    if ctrl.type in ("small_llm", "jev"):
        if not (ctrl.model_ref and ctrl.criteria_ref and ctrl.timeout_s):
            raise ConfigError(f"{path.name}: {ctrl.type} needs model_ref, criteria_ref, timeout_s")
        extra["criteria"] = parse(Criteria, read_yaml(ctrl.criteria_ref), ctrl.criteria_ref)
    if ctrl.type == "small_llm":
        if not ctrl.prompt_ref:
            raise ConfigError(f"{path.name}: small_llm needs prompt_ref")
        extra["controller_model"] = load_model(ctrl.model_ref)
        extra["controller_prompt"] = resolve(ctrl.prompt_ref).read_text()
    if ctrl.type == "jev":
        if ctrl.model_ref != "jev":
            raise ConfigError(f"{path.name}: jev controller must use model_ref 'jev'")
        extra["jev"] = _pinned(load_models().jev)
    if ctrl.type == "placebo":
        if not ctrl.distribution_ref:
            raise ConfigError(f"{path.name}: placebo needs distribution_ref")
        extra["placebo"] = parse(
            PlaceboDist, read_yaml(ctrl.distribution_ref), ctrl.distribution_ref
        )

    return ResolvedArm(
        cfg=cfg,
        path=path,
        model=load_model(cfg.reasoning.model_ref),
        limits=parse(Limits, read_ref(limits_ref), limits_ref),
        limits_ref=limits_ref,
        triggers=parse(TriggerCfg, read_yaml(cfg.triggers_ref), cfg.triggers_ref),
        system_prompt=resolve(cfg.system_prompt_ref).read_text().strip(),
        guidance=load_guidance(cfg.guidance_ref),
        **extra,
    )


def load_experiment(path: str | Path) -> ExperimentCfg:
    return parse(ExperimentCfg, read_yaml(path), Path(path).name)


def load_tasks(path: str | Path) -> TaskList:
    return parse(TaskList, read_yaml(path), Path(path).name)


def task_checksum(tasks: TaskList) -> str:
    rows = sorted((t.model_dump() for t in tasks.tasks), key=lambda t: t["name"])
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def load_benchmark() -> dict[str, Any]:
    return read_yaml("config/benchmark.yaml")


def _pairing_key(arm: ResolvedArm) -> dict[str, Any]:
    return {
        **arm.cfg.model_dump(exclude={"id", "controller"}),
        "model": arm.model.model_dump(),
        "limits": arm.limits.model_dump(),
    }


def validate_experiment(
    path: str | Path, check_phase: bool = True
) -> tuple[ExperimentCfg | None, list[ResolvedArm], TaskList | None, list[str]]:
    """Load everything an experiment needs; return all problems instead of failing on the first."""
    errors: list[str] = []
    try:
        exp = load_experiment(path)
    except ConfigError as e:
        return None, [], None, [str(e)]

    arms: list[ResolvedArm] = []
    for arm_path in exp.arms:
        try:
            arms.append(load_arm(arm_path, exp.limits_ref))
        except ConfigError as e:
            errors.append(f"{arm_path}: {e}")

    ids = [a.id for a in arms]
    if len(set(ids)) != len(ids):
        errors.append(f"duplicate arm ids: {ids}")

    by_model: dict[str, list[ResolvedArm]] = {}
    for arm in arms:
        by_model.setdefault(arm.cfg.reasoning.model_ref, []).append(arm)
    for group in by_model.values():
        ref = _pairing_key(group[0])
        for other in group[1:]:
            diff = sorted(k for k, v in _pairing_key(other).items() if ref.get(k) != v)
            if diff:
                errors.append(
                    f"paired arms {group[0].id} / {other.id} differ beyond controller: {diff}"
                )

    tasks = None
    try:
        tasks = load_tasks(exp.tasks)
        if not tasks.tasks:
            errors.append(f"{exp.tasks}: no tasks")
        if exp.require_frozen_tasks:
            if not tasks.frozen:
                errors.append(f"{exp.tasks}: not frozen (run `agentbench tasks freeze`)")
            elif tasks.checksum != task_checksum(tasks):
                errors.append(f"{exp.tasks}: checksum mismatch — task list changed after freezing")
    except ConfigError as e:
        errors.append(str(e))

    if placeholders(load_benchmark(), "benchmark"):
        errors.append(f"config/benchmark.yaml: {PLACEHOLDER} not filled in")

    phase = os.environ.get("AGENTBENCH_PHASE")
    if check_phase and phase != exp.phase:
        errors.append(f"AGENTBENCH_PHASE={phase!r} but experiment phase is {exp.phase!r}")

    return exp, arms, tasks, errors
