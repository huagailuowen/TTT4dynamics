#!/usr/bin/env python3
"""Repair canonical custom metadata after in-place LeRobot resume collection."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
FORMAL_SCRIPT = HERE / "collect_libero_workspace_rich_eef_400eps_lerobot_2026-07-16_hai-machine.py"
DEFAULT_DATASET = (
    HERE.parent
    / "data/various_actions/"
    "libero_mu0100_workspace_rich_eef_400eps_lerobot_2026-07-16_hai-machine"
)
DEFAULT_WORK_METADATA = (
    HERE.parent
    / "data/various_actions/resume_work/"
    "libero_workspace_rich_eef_400eps_2026-07-16_hai-machine/"
    "meta/push_box_episode_metadata.jsonl"
)
DEFAULT_MISSING_LOG = (
    HERE.parent
    / "data/various_actions/"
    "resume_workspace_rich_eef_400eps_part064-087_2026-07-16_hai-machine.log"
)
SUMMARY_RE = re.compile(
    r"^\[\d+/\d+\] ep=(\d+) group=(\S+) family=(\S+) frames=(\d+) "
    r"span=\(([^,]+),([^,]+),([^\)]+)\)m box=([\d.]+)cm contact=(\d+) grasp=(\d+)"
)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


formal = load_module(FORMAL_SCRIPT, "workspace_rich_metadata_repair_base")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_summaries(path: Path) -> dict[int, dict[str, Any]]:
    summaries: dict[int, dict[str, Any]] = {}
    for line in path.read_text(errors="replace").replace("\r", "\n").splitlines():
        match = SUMMARY_RE.match(line)
        if not match:
            continue
        episode, group, family, frames, sx, sy, sz, box_cm, contact, grasp = match.groups()
        summaries[int(episode)] = {
            "behavior_group": group,
            "family": family,
            "frames_in_lerobot_episode": int(frames),
            "eef_xyz_span_m": [float(sx), float(sy), float(sz)],
            "final_box_displacement_m": float(box_cm) / 100.0,
            "robot_box_contact_steps": int(contact),
            "robosuite_grasping_steps": int(grasp),
        }
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--work-metadata", type=Path, default=DEFAULT_WORK_METADATA)
    parser.add_argument("--missing-log", type=Path, default=DEFAULT_MISSING_LOG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_path = args.dataset / "meta/push_box_episode_metadata.jsonl"
    prefix = read_jsonl(metadata_path)
    resumed = read_jsonl(args.work_metadata)
    summaries = parse_summaries(args.missing_log)
    plan = formal.make_plan()
    if len(prefix) != 64 or len(resumed) != 312 or len(plan) != 400:
        raise RuntimeError(
            f"Unexpected inputs: prefix={len(prefix)} resumed={len(resumed)} plan={len(plan)}"
        )

    by_episode = {int(row["episode_index"]): row for row in prefix + resumed}
    for episode in range(64, 88):
        item = plan[episode]
        summary = summaries.get(episode)
        if summary is None:
            raise RuntimeError(f"Missing archived summary for episode {episode}")
        if summary["family"] != item["family"] or summary["behavior_group"] != item["behavior_group"]:
            raise RuntimeError(f"Plan/log mismatch at episode {episode}")
        by_episode[episode] = {
            "episode_index": episode,
            "collection_index": episode,
            "behavior_group": item["behavior_group"],
            "family": item["family"],
            "family_repetition": item["family_repetition"],
            "parameter_seed": item["parameter_seed"],
            "sim_seed": None,
            "friction_mu": 0.1,
            "target_visible": False,
            "language_instructions": formal.TASK_PROMPTS[item["family"]],
            "parameters": {
                "metadata_recovered_from_plan_and_archived_log": True,
                "detailed_rollout_parameters_unavailable": True,
            },
            "metrics": summary,
        }

    canonical = [by_episode[index] for index in range(400)]
    for index, row in enumerate(canonical):
        row["episode_index"] = index
        row["collection_index"] = index

    backup = metadata_path.with_name(
        "push_box_episode_metadata_prefix000-063_before_resume_repair.jsonl"
    )
    if not backup.exists():
        shutil.copy2(metadata_path, backup)
    write_jsonl(metadata_path, canonical)

    family_counts = Counter(row["family"] for row in canonical)
    group_counts = Counter(row["behavior_group"] for row in canonical)
    micro_counts = Counter(
        primitive
        for sequence in formal.MICRO_SCHEDULE
        for primitive in sequence
    )
    expected_family_counts = Counter(formal.FAMILY_COUNTS)
    if family_counts != expected_family_counts:
        raise RuntimeError(
            f"Canonical family counts differ: actual={family_counts}, expected={expected_family_counts}"
        )
    plan_json = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    manifest = {
        "schema": "workspace_rich_eef_collection_plan_v1",
        "episode_count": 400,
        "plan_sha256": hashlib.sha256(plan_json.encode()).hexdigest(),
        "group_counts": dict(sorted(group_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "micro_primitive_counts": dict(sorted(micro_counts.items())),
        "metadata_sources": {
            "episodes_000_063": "original detailed metadata",
            "episodes_064_087": "full plan plus archived rollout summary",
            "episodes_088_399": "resume detailed metadata",
        },
    }
    manifest_path = args.dataset / "meta/collection_plan_manifest_2026-07-16_hai-machine.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
