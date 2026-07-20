#!/usr/bin/env python3
"""Append the unfinished suffix of the formal 400-episode plan in place."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from fastwam.datasets.lerobot.lerobot.lerobot_dataset import LeRobotDataset


HERE = Path(__file__).resolve().parent
FORMAL_SCRIPT = HERE / "collect_libero_workspace_rich_eef_400eps_lerobot_2026-07-16_hai-machine.py"
DEFAULT_DATASET = (
    HERE.parent
    / "data/various_actions/"
    "libero_mu0100_workspace_rich_eef_400eps_lerobot_2026-07-16_hai-machine"
)
DEFAULT_GRAPH = (
    HERE.parent
    / "data/various_actions/calibration/eef_pose_transition_graph_2026-07-16_hai-machine.json"
)
DEFAULT_WORK_OUTPUT = (
    HERE.parent
    / "data/various_actions/resume_work/"
    "libero_workspace_rich_eef_400eps_2026-07-16_hai-machine"
)
REPO_ID = "libero_mu0100_workspace_rich_eef_300eps_hai_machine"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


formal = load_module(FORMAL_SCRIPT, "workspace_rich_eef_formal_resume_base")
rich = formal.rich


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--work-output", type=Path, default=DEFAULT_WORK_OUTPUT)
    parser.add_argument("--start-index", type=int)
    return parser.parse_args()


def install_bounded_epilogue() -> None:
    original = rich.run_workspace_context_segment

    def bounded_epilogue(
        recorder: Any,
        repetition: int,
        rng: Any,
        *,
        prefix: str,
        count: int = 2,
    ) -> dict[str, Any]:
        primary_frames = len(recorder.rows)
        if primary_frames >= 500:
            return {
                "adaptive_epilogue": True,
                "requested_pose_count": count,
                "executed_pose_count": 0,
                "skip_reason": "primary_frame_budget",
                "primary_frames": primary_frames,
            }
        try:
            result = original(
                recorder,
                repetition,
                rng,
                prefix=prefix,
                count=min(count, 1),
            )
            result["adaptive_epilogue"] = True
            result["requested_pose_count"] = count
            result["executed_pose_count"] = min(count, 1)
            result["primary_frames"] = primary_frames
            return result
        except ValueError as error:
            if "terminated episode" not in str(error):
                raise
            return {
                "adaptive_epilogue": True,
                "requested_pose_count": count,
                "executed_pose_count": 1,
                "ended_at_environment_horizon": True,
                "primary_frames": primary_frames,
            }

    rich.run_workspace_context_segment = bounded_epilogue


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset.resolve()
    dataset = LeRobotDataset(
        repo_id=REPO_ID,
        root=dataset_root,
        download_videos=False,
        video_codec="h264",
    )
    existing = dataset.num_episodes
    if args.start_index is not None and args.start_index != existing:
        raise RuntimeError(
            f"Requested start index {args.start_index}, but dataset contains {existing} episodes"
        )
    if existing >= formal.EXPECTED_EPISODES:
        raise RuntimeError(f"Dataset already contains {existing} episodes")

    formal.load_validated_graph(args.graph)
    full_plan = formal.make_plan()
    remaining_plan = full_plan[existing:]
    remaining_counts_raw = Counter(item["family"] for item in remaining_plan)
    remaining_counts = {
        family: remaining_counts_raw[family]
        for family in formal.FAMILY_COUNTS
        if remaining_counts_raw[family]
    }
    remaining_groups = {
        family: formal.GROUP_BY_FAMILY[family] for family in remaining_counts
    }
    remaining_prompts = {
        family: formal.TASK_PROMPTS[family] for family in remaining_counts
    }

    if args.work_output.exists():
        shutil.rmtree(args.work_output)
    original_create_dataset = rich.base.create_dataset
    rich.base.create_dataset = lambda output, repo_id: dataset
    rich.FAMILY_COUNTS = remaining_counts
    rich.EXPECTED_EPISODES = len(remaining_plan)
    rich.GROUP_BY_FAMILY = remaining_groups
    rich.TASK_PROMPTS = remaining_prompts
    rich.make_plan = lambda: remaining_plan
    rich.run_family = formal.run_family
    rich.episode_task_prompts = formal.episode_task_prompts
    rich.observable_interaction = lambda recorder: True
    install_bounded_epilogue()

    forwarded = [str(rich.__file__), "--output", str(args.work_output)]
    original_argv = sys.argv
    print(
        f"[resume] existing={existing} remaining={len(remaining_plan)} "
        f"dataset={dataset_root}",
        flush=True,
    )
    try:
        sys.argv = forwarded
        rich.main()
    finally:
        sys.argv = original_argv
        rich.base.create_dataset = original_create_dataset


if __name__ == "__main__":
    main()
