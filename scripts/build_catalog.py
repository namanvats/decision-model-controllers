"""Build tasks/catalog.yaml from a Terminal-Bench checkout's task.toml files.

    uv run python scripts/build_catalog.py vendor/tb21 > tasks/catalog.yaml

Reads metadata and resource requests only; never copies tests or solutions.
"""

import argparse
import subprocess
import sys
import tomllib
from pathlib import Path

import yaml


def exclusion(task: dict, args: argparse.Namespace) -> str | None:
    env = task.get("environment", {})
    agent_timeout = task.get("agent", {}).get("timeout_sec", 0)
    if env.get("gpus", 0) > 0:
        return "gpu_required"
    if env.get("cpus", 1) > args.max_cpus:
        return f"cpus {env['cpus']} > plan limit {args.max_cpus}"
    if env.get("memory_mb", 0) > args.max_memory_mb:
        return f"memory {env['memory_mb']}MB > plan limit {args.max_memory_mb}MB"
    if env.get("storage_mb", 0) > args.max_storage_mb:
        return f"storage {env['storage_mb']}MB > plan limit {args.max_storage_mb}MB"
    if agent_timeout > args.max_agent_timeout_s:
        return f"agent timeout {agent_timeout:.0f}s > budget {args.max_agent_timeout_s}s"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", type=Path)
    parser.add_argument("--max-cpus", type=int, default=4)
    parser.add_argument("--max-memory-mb", type=int, default=8192)
    parser.add_argument("--max-storage-mb", type=int, default=10240)
    parser.add_argument("--max-agent-timeout-s", type=int, default=3600)
    args = parser.parse_args()

    commit = subprocess.run(
        ["git", "-C", str(args.repo), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    entries = []
    for path in sorted(args.repo.glob("tasks/*/task.toml")):
        task = tomllib.loads(path.read_text())
        meta, env = task.get("metadata", {}), task.get("environment", {})
        entries.append(
            {
                "name": task["task"]["name"],
                "category": meta.get("category"),
                "difficulty": meta.get("difficulty"),
                "agent_timeout_sec": task.get("agent", {}).get("timeout_sec"),
                "verifier_timeout_sec": task.get("verifier", {}).get("timeout_sec"),
                "cpus": env.get("cpus"),
                "memory_mb": env.get("memory_mb"),
                "storage_mb": env.get("storage_mb"),
                "gpus": env.get("gpus"),
                "allow_internet": env.get("allow_internet"),
                "docker_image": env.get("docker_image"),
                "excluded_reason": exclusion(task, args),
            }
        )
    excluded = sum(e["excluded_reason"] is not None for e in entries)
    print(f"# {len(entries)} tasks, {excluded} excluded; source commit {commit}", file=sys.stderr)
    yaml.safe_dump(
        {
            "schema_version": 1,
            "source_commit": commit,
            "exclusion_rules": {k: v for k, v in vars(args).items() if k != "repo"},
            "tasks": entries,
        },
        sys.stdout,
        sort_keys=False,
    )


if __name__ == "__main__":
    main()
