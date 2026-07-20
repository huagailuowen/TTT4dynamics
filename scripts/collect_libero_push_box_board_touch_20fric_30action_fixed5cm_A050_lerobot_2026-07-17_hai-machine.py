#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RAMP_SOURCE_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "collect_libero_push_box_board_touch_fixed5cm_ramp_latched_brake_probe_lerobot_2026-07-17_hai-machine.py"
)
CONFIG_PATH = (
    REPO_ROOT
    / "configs"
    / "libero_push_box_board_touch_20fric_30action_fixed5cm_A050_2026-07-17_hai-machine.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "data"
    / "pushbox"
    / "libero_push_box_board_touch_20fric_30action_full8_A450_hidden_lerobot_2026-07-17_hai-machine"
)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ramp = load_module(RAMP_SOURCE_SCRIPT, "formal_fixed5cm_A050_source_hai_machine")
collector = ramp.collector
base = ramp.base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect the formal 20-friction x 30-action fixed-5-cm board-touch LeRobot dataset."
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(base.to_jsonable(value), indent=2), encoding="utf-8")


def configure_dataset(experiment: dict[str, Any]) -> dict[str, Any]:
    config = collector.configure_dataset(experiment)
    template = experiment["action_template"]
    for action in config["actions"]:
        action["profile"] = [float(template["first_frame_fraction"]), 1.0]
        action["description"] = str(template["description"])
    config["action_template"] = dict(template)
    return config


def create_dataset(root: Path, *, config: dict[str, Any], experiment: dict[str, Any]) -> Any:
    recording = experiment["recording"]
    return base.LeRobotDataset.create(
        repo_id="libero_push_box_board_touch_20fric_30action_fixed5cm_A050_hidden_hai_machine",
        root=root,
        fps=int(config["fps"]),
        features=base.build_features(int(config["camera_resolution"])),
        use_videos=True,
        video_codec=str(recording["video_codec"]),
        is_compute_episode_stats_image=False,
    )


def collect(
    experiment: dict[str, Any],
    config: dict[str, Any],
    *,
    output_root: Path,
    overwrite: bool,
    seed: int,
) -> dict[str, Any]:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} exists; pass --overwrite")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    recording = experiment["recording"]
    template = experiment["action_template"]
    controller_cfg = template["controller"]
    base.patch_lerobot_video_crf(int(recording["video_crf"]))
    dataset_root = output_root / "hidden_straight_lerobot"
    dataset = create_dataset(dataset_root, config=config, experiment=experiment)

    rows: list[dict[str, Any]] = []
    created_at = dt.datetime.now().isoformat()
    dataset_type = "libero_push_box_board_touch_20fric_30action_fixed5cm_A050_hidden_lerobot_hai-machine"
    metadata = {
        "created_at": created_at,
        "dataset_type": dataset_type,
        "target_visible": False,
        "split": "straight",
        "camera_resolution": int(config["camera_resolution"]),
        "fps": int(config["fps"]),
        "video_codec": str(recording["video_codec"]),
        "video_crf": int(recording["video_crf"]),
        "jpeg_quality": int(recording["jpeg_quality"]),
        "state_source": "true LIBERO obs robot0_eef_pos, robot0_eef_quat converted to axis-angle, robot0_gripper_qpos",
        "action_semantics": "FastWAM EEF delta-position action; 0.7A first frame, then A, one-way brake, then latched zero",
        "action_template": dict(template),
        "friction_schedule": dict(experiment["friction_schedule"]),
        "touch_preparation": dict(experiment["touch_preparation"]),
        "board": dict(experiment["board"]),
        "episodes": [],
    }
    manifest = {
        "created_at": created_at,
        "dataset_type": f"{dataset_type}_collection",
        "output_root": str(output_root),
        "hidden_straight_lerobot": str(dataset_root),
        "config_path": str(CONFIG_PATH),
        "config": config,
        "episodes": [],
    }

    def autosave() -> None:
        write_json(output_root / "manifest.json", manifest)
        base.write_dataset_metadata(dataset_root, metadata, rows)

    total = len(config["frictions"]) * len(config["actions"])
    count = 0
    for mu_index, mu in enumerate(config["frictions"]):
        for action_cfg in config["actions"]:
            action_id = int(action_cfg["action_id"])
            amplitude = float(action_cfg["A"])
            case_id = (
                f"board_touch_m{mu_index:02d}_{base.mu_tag(float(mu))}_"
                f"a{action_id:02d}_A{int(round(amplitude * 1000)):03d}_fixed5cm"
            )
            bddl = base.write_hidden_bddl(
                config,
                bddl_dir=output_root / "bddl",
                geometry_id=case_id,
            )
            base_case = collector.make_case(
                config,
                mu=float(mu),
                action_cfg=action_cfg,
                case_id=case_id,
                bddl_file=bddl,
            )
            case = ramp.preserve_case_attributes(
                base_case,
                replace(base_case, pusher_max_pos_action=float(template["pusher_max_pos_action"])),
            )
            episode_index, metrics = ramp.rollout(
                case,
                dataset=dataset,
                amplitude=amplitude,
                first_fraction=float(template["first_frame_fraction"]),
                travel_m=float(template["target_eef_travel_m"]),
                controller_cfg=controller_cfg,
                recorded_steps=int(template["post_launch_recording_frames"]),
                seed=seed,
                fps=int(config["fps"]),
                jpeg_quality=int(recording["jpeg_quality"]),
            )
            row = {
                "episode_index": int(episode_index),
                "case_id": case_id,
                "mu_index": int(mu_index),
                "mu": float(mu),
                "mu_tag": base.mu_tag(float(mu)),
                "action_id": action_id,
                "A": amplitude,
                "first_frame_fraction": float(template["first_frame_fraction"]),
                "first_frame_action": float(template["first_frame_fraction"]) * amplitude,
                "target_eef_travel_m": float(template["target_eef_travel_m"]),
                "controller": dict(controller_cfg),
                "init_xy": [float(value) for value in config["init_xy"]],
                "target_xy": list(base.fixed_scene_target_xy(config)),
                "bddl_file": bddl,
                "board": dict(config["board"]),
                "metrics": metrics,
            }
            rows.append(row)
            metadata["episodes"].append(row)
            manifest["episodes"].append(row)
            count += 1
            print(
                f"collect {count:03d}/{total:03d} {case_id} "
                f"disp={metrics['final_box_displacement_from_launch_m'] * 100.0:.2f}cm "
                f"peak_vx={metrics['peak_box_vx_mps']:.3f}m/s "
                f"eef_max={metrics['max_eef_travel_m'] * 100.0:.2f}cm "
                f"contacts={metrics['sampled_contact_episode_count']}",
                flush=True,
            )
            autosave()

    summary_counts = Counter(row["mu_tag"] for row in rows)
    summary = {
        "episode_count": len(rows),
        "expected_episode_count": total,
        "hidden_straight_lerobot": str(dataset_root),
        "count_by_mu": dict(sorted(summary_counts.items())),
        "action_range_A": [float(config["action_peak_min"]), float(config["action_peak_max"])],
        "first_frame_fraction": float(template["first_frame_fraction"]),
        "target_eef_travel_m": float(template["target_eef_travel_m"]),
        "frictions": [float(mu) for mu in config["frictions"]],
    }
    write_json(output_root / "summary.json", summary)
    autosave()
    print(f"manifest={output_root / 'manifest.json'}", flush=True)
    print(f"hidden_root={dataset_root}", flush=True)
    return summary


def main() -> None:
    args = parse_args()
    experiment = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    config = configure_dataset(experiment)
    expected = int(experiment["expected_episode_count"])
    actual = len(config["frictions"]) * len(config["actions"])
    if actual != expected:
        raise RuntimeError(f"Configured {actual} episodes, expected {expected}")
    summary = collect(
        experiment,
        config,
        output_root=args.output_root.resolve(),
        overwrite=bool(args.overwrite),
        seed=int(args.seed),
    )
    print(json.dumps(base.to_jsonable(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()
