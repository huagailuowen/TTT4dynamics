#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "collect_libero_push_box_event_tap_segmented80_10action_lerobot_2026-07-05_hai-machine.py"
)
CONFIG_PATH = (
    REPO_ROOT
    / "configs"
    / "libero_push_box_fric80_noboard_A026_3friction_stability_probe_2026-07-19_hai-machine.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "outputs"
    / "pushbox"
    / "old_method_box_stability_analysis"
    / "libero_push_box_fric80_noboard_A026_3friction_stability_probe_lerobot_2026-07-19_hai-machine"
)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


source = load_module(SOURCE_SCRIPT, "fric80_noboard_stability_source_hai_machine")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a three-friction probe with the unchanged former fric80 no-board controller."
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    config = source.configure_current_dataset(source.base.load_config(source.CONFIG_PATH))

    requested_A = float(experiment["action"]["A"])
    requested_steps = int(experiment["action"]["push_steps"])
    action = next(
        dict(candidate)
        for candidate in source.MID_DENSE_ACTIONS
        if abs(float(candidate["A"]) - requested_A) < 1e-12
        and int(candidate["push_steps"]) == requested_steps
    )
    config["frictions"] = [float(mu) for mu in experiment["frictions"]]
    config["friction_count"] = len(config["frictions"])
    config["friction_schedule"] = "diagnostic subset of exact former segmented fric80 schedule"
    config["actions"] = [action]
    config["action_count"] = 1
    config["dataset_name"] = (
        "libero_push_box_fric80_noboard_A026_3friction_stability_probe_lerobot_hai-machine"
    )

    actual = len(config["frictions"]) * len(config["actions"])
    expected = int(experiment["expected_episode_count"])
    if actual != expected:
        raise RuntimeError(f"Configured {actual} episodes, expected {expected}")

    recording = experiment["recording"]
    summary = source.collect(
        config,
        output_root=args.output_root.resolve(),
        overwrite=bool(args.overwrite),
        seed=int(experiment["seed"]),
        video_codec=str(recording["video_codec"]),
        video_crf=int(recording["video_crf"]),
        jpeg_quality=int(recording["jpeg_quality"]),
    )
    print(json.dumps(source.base.to_jsonable(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()
