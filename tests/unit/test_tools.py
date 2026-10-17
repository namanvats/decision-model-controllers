import pytest

from agentbench.tools import (
    clamp_timeout,
    is_verification,
    is_write,
    normalize_command,
    parse_tool_call,
    truncate,
)


def test_truncate_keeps_head_and_tail():
    text = "a" * 10 + "b" * 100 + "c" * 10
    out = truncate(text, 10, 10)
    assert out.startswith("a" * 10) and out.endswith("c" * 10) and "100 chars omitted" in out
    assert truncate("short", 10, 10) == "short"


@pytest.mark.parametrize(
    "cmd",
    [
        "pytest -q",
        "cd /app && make test",
        "./build/app --check",
        "diff out.txt expected.txt",
        "cargo test",
    ],
)
def test_verification_commands(cmd):
    assert is_verification(cmd)


@pytest.mark.parametrize(
    "cmd",
    [
        "echo hi > f.txt",
        "sed -i 's/a/b/' x",
        "cat <<EOF > a\nx\nEOF",
        "pip install numpy",
        "git apply p.diff",
        "cp a b",
    ],
)
def test_write_commands(cmd):
    assert is_write(cmd)


@pytest.mark.parametrize(
    "cmd", ["ls -la", "cat file 2>/dev/null", "grep x y 2>&1", "python3 -c 'print(1)'"]
)
def test_read_only_commands(cmd):
    assert not is_write(cmd)


def test_parse_tool_call_errors():
    assert parse_tool_call({"function": {"name": "bash", "arguments": "{bad"}})[2]
    assert parse_tool_call({"function": {"name": "bash", "arguments": "[1]"}})[2]
    name, args, err = parse_tool_call(
        {"function": {"name": "bash", "arguments": '{"command": "ls"}'}}
    )
    assert (name, args, err) == ("bash", {"command": "ls"}, None)


def test_clamp_timeout_and_normalize():
    assert clamp_timeout(None) == 120 and clamp_timeout(99999) == 1800 and clamp_timeout("x") == 120
    assert normalize_command("  ls   -la ") == "ls -la"
