from collections import Counter

from agentbench.config import TriggerCfg
from agentbench.tools import VERIFY_RE, WRITE_RE, Action, last_match, normalize_command

PRIORITY = ("T1", "T2", "T4")


class TriggerDetector:
    """Deterministic T1/T2/T4 detection with per-type cooldown (experiment_plan.md §5.2)."""

    def __init__(self, cfg: TriggerCfg):
        self.cfg = cfg
        self.actions: list[Action] = []
        self.consecutive_failures = 0
        self.verification_cmds = 0
        self.last_write: tuple[int, int] | None = None  # (action index, offset in command)
        self.last_verify: tuple[int, int] | None = None
        self.last_fired: dict[str, int] = {}
        self.fire_counts: Counter[str] = Counter()

    def observe(self, action: Action) -> None:
        self.actions.append(action)
        self.consecutive_failures = self.consecutive_failures + 1 if action.failed else 0
        if (pos := last_match(WRITE_RE, action.command)) is not None:
            self.last_write = (action.index, pos)
        if (pos := last_match(VERIFY_RE, action.command)) is not None:
            self.verification_cmds += 1
            self.last_verify = (action.index, pos)

    @property
    def verified_after_last_edit(self) -> bool:
        if self.last_verify is None:
            return False
        return self.last_write is None or self.last_verify > self.last_write

    def repeated_command(self) -> bool:
        window = self.actions[-self.cfg.t2_command_window :]
        counts = Counter(normalize_command(a.command) for a in window)
        return any(n >= self.cfg.t2_command_repeats for n in counts.values())

    def repeated_output(self) -> bool:
        window = [a for a in self.actions[-self.cfg.t2_output_window :] if a.output.strip()]
        counts = Counter(a.output_hash for a in window)
        return any(n >= self.cfg.t2_output_repeats for n in counts.values())

    def candidates(self, step: int) -> list[str]:
        found = []
        if self.consecutive_failures >= self.cfg.t1_consecutive_failures:
            found.append("T1")
        if self.repeated_command() or self.repeated_output():
            found.append("T2")
        if step % self.cfg.t4_every_n_steps == 0:
            found.append("T4")
        return found

    def fire(self, step: int, candidates: list[str]) -> str | None:
        """Pick at most one trigger per step, skipping types still in cooldown."""
        for trigger in PRIORITY:
            last = self.last_fired.get(trigger)
            if trigger in candidates and (
                last is None or step - last > self.cfg.refire_cooldown_steps
            ):
                self.last_fired[trigger] = step
                self.fire_counts[trigger] += 1
                return trigger
        return None

    def finish_requested(self) -> str:
        self.fire_counts["T3"] += 1
        return "T3"
