import hashlib
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

DEFAULT_TIMEOUT_S = 120
MAX_TIMEOUT_S = 1800
TIMEOUT_EXIT_CODE = 124

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command in the task environment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command to run."},
                    "timeout_s": {
                        "type": "integer",
                        "description": f"Timeout in seconds (default {DEFAULT_TIMEOUT_S}, "
                        f"max {MAX_TIMEOUT_S}).",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Declare the task complete.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "What you did and how you verified it.",
                    }
                },
                "required": ["summary"],
            },
        },
    },
]

# Test runners, diff/cmp checks, running a built artifact, or running a script file.
VERIFY_RE = re.compile(
    r"\b(pytest|unittest|tox|nox|ctest|diff|cmp|cargo\s+(test|run)|go\s+(test|run)"
    r"|npm\s+(run\s+)?test|yarn\s+test|sha256sum\s+-c|md5sum\s+-c)\b"
    r"|\bmake\b[^;&|\n]*\b(test|tests|check|testsuite)\b"
    r"|(^|[;&|(]\s*)\./[\w.-]+"
    r"|\b(python[\d.]*|perl|node|ruby|bash|sh|Rscript|julia|php)\s+(-\S+\s+)*"
    r"[\w./~-]+\.(py|pl|js|mjs|rb|sh|R|jl|php)\b"
)
WRITE_RE = re.compile(
    r"(?<![\d<>&])>>?(?!&)(?!\s*/dev/null)"
    r"|\btee\b|\bsed\s+(-\w*\s+)*-i|\bperl\s+(-\S+\s+)*-\w*i|\b(cp|mv|rm|mkdir|touch|chmod|ln|patch|install)\b"
    r"|\bgit\s+(apply|checkout|reset|commit|merge|rebase|am|restore)\b"
    r"|<<-?\s*['\"]?\w+|\bpip3?\s+install\b|\bapt(-get)?\s+install\b"
)


@dataclass
class ExecOutput:
    exit_code: int
    output: str
    timed_out: bool = False


ExecFn = Callable[[str, int], Awaitable[ExecOutput]]


@dataclass
class Action:
    index: int
    step: int
    command: str
    exit_code: int
    timed_out: bool
    output: str
    output_hash: str

    @property
    def failed(self) -> bool:
        return self.timed_out or self.exit_code != 0


def truncate(text: str, head: int, tail: int) -> str:
    if len(text) <= head + tail:
        return text
    omitted = len(text) - head - tail
    return f"{text[:head]}\n[... {omitted} chars omitted ...]\n{text[-tail:]}"


def normalize_command(command: str) -> str:
    return " ".join(command.split())


def output_hash(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def last_match(pattern: re.Pattern, command: str) -> int | None:
    """Start offset of the last match, so edits and checks in one command can be ordered."""
    starts = [m.start() for m in pattern.finditer(command)]
    return starts[-1] if starts else None


def is_verification(command: str) -> bool:
    return bool(VERIFY_RE.search(command))


def is_write(command: str) -> bool:
    return bool(WRITE_RE.search(command))


def parse_tool_call(tool_call: dict) -> tuple[str, dict, str | None]:
    function = tool_call.get("function") or {}
    name = function.get("name", "")
    try:
        args = json.loads(function.get("arguments") or "{}")
    except json.JSONDecodeError as e:
        return name, {}, f"invalid JSON arguments: {e}"
    if not isinstance(args, dict):
        return name, {}, "arguments must be a JSON object"
    return name, args, None


def clamp_timeout(value: object) -> int:
    try:
        timeout = int(value) if value is not None else DEFAULT_TIMEOUT_S
    except (TypeError, ValueError):
        timeout = DEFAULT_TIMEOUT_S
    return max(1, min(timeout, MAX_TIMEOUT_S))


def format_action(action: Action, timeout_s: int) -> str:
    status = (
        f"timed out after {timeout_s}s" if action.timed_out else f"exit_code: {action.exit_code}"
    )
    return f"{status}\n{action.output}" if action.output else status
