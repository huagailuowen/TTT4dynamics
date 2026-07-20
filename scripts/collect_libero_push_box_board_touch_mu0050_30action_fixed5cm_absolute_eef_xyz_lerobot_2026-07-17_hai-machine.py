#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import datetime as dt
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
FORMAL_SOURCE_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "collect_libero_push_box_board_touch_20fric_30action_fixed5cm_A050_lerobot_2026-07-17_hai-machine.py"
)
CONFIG_PATH = (
    REPO_ROOT
    / "configs"
    / "libero_push_box_board_touch_mu0050_30action_fixed5cm_absolute_eef_xyz_2026-07-17_hai-machine.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "data"
    / "pushbox"
    / "libero_push_box_board_touch_mu0050_30action_fixed5cm_absolute_eef_xyz_lerobot_2026-07-17_hai-machine"
)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


formal = load_module(FORMAL_SOURCE_SCRIPT, "mu0050_absolute_eef_formal_source_hai_machine")
ramp = formal.ramp
collector = formal.collector
base = formal.base
touch = ramp.touch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect fixed-mu PushBox rollouts with absolute EEF XYZ actions."
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(base.to_jsonable(value), indent=2), encoding="utf-8")


def absolute_action(
    observation_state: np.ndarray,
    environment_action: np.ndarray,
    *,
    translation_scale_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    relative = np.asarray(
        base._env_action_to_fastwam_action(environment_action.astype(np.float32)),
        dtype=np.float32,
    )
    absolute = relative.copy()
    absolute[:3] = (
        np.asarray(observation_state[:3], dtype=np.float32)
        + np.float32(translation_scale_m) * relative[:3]
    )
    return relative, absolute


def rollout_absolute(
    case: Any,
    *,
    dataset: Any,
    amplitude: float,
    first_fraction: float,
    travel_m: float,
    controller_cfg: dict[str, Any],
    recorded_steps: int,
    seed: int,
    fps: int,
    jpeg_quality: int,
    translation_scale_m: float,
) -> tuple[int, dict[str, Any]]:
    env = touch.make_env(case, seed=seed)
    rows: list[dict[str, Any]] = []
    stored_actions: list[np.ndarray] = []
    relative_actions: list[np.ndarray] = []
    phase = "drive"
    brake_frames = 0
    brake_trigger_frame = None
    contact_frames: list[int] = []
    contact_episode_count = 0
    contact_active = False
    try:
        base.preposition_fixed_start(env)
        env.step_count = 0
        env._last_scripted_action = np.zeros(7, dtype=np.float64)
        env._last_scripted_phase = None
        initial_box_xyz, _ = env.box_pose()
        push_start = int(case.pusher_approach_steps) + int(case.pusher_descend_steps)
        for _ in range(push_start):
            env.step()
        touch_state = collector.prepare_touch_unrecorded(env, case=case)
        launch_box_xyz, _ = env.box_pose()
        launch_eef = np.asarray(env._last_obs["robot0_eef_pos"], dtype=np.float64).copy()
        target_x = float(launch_eef[0] + travel_m)
        previous_x = float(launch_eef[0])
        hold_yz = launch_eef[1:3].copy()

        base.remove_current_episode_images(dataset)
        episode_index = int(dataset.meta.total_episodes)
        task = base.prompt_for_case("observation", "straight")
        for frame_index in range(int(recorded_steps)):
            obs_for_frame = ramp.copy_obs(env._last_obs)
            eef_before = np.asarray(obs_for_frame["robot0_eef_pos"], dtype=np.float64)
            eef_vx = float((eef_before[0] - previous_x) * float(fps)) if frame_index else 0.0
            previous_x = float(eef_before[0])
            remaining = float(target_x - eef_before[0])

            if phase == "drive" and frame_index > 0:
                lookahead_m = (
                    float(controller_cfg["brake_trigger_lookahead_frames"])
                    * max(0.0, eef_vx)
                    / float(fps)
                )
                if remaining <= lookahead_m:
                    phase = "brake"
                    brake_trigger_frame = int(frame_index)

            if phase == "drive":
                command_x = float(amplitude * first_fraction) if frame_index == 0 else float(amplitude)
            elif phase == "brake":
                if brake_frames >= int(controller_cfg["maximum_brake_frames"]) or (
                    brake_frames > 0 and eef_vx <= float(controller_cfg["stop_speed_mps"])
                ):
                    phase = "locked_zero"
                    command_x = 0.0
                else:
                    command_x = float(
                        np.clip(
                            -float(controller_cfg["brake_gain_action_per_mps"]) * max(0.0, eef_vx),
                            -float(amplitude),
                            0.0,
                        )
                    )
                    brake_frames += 1
            else:
                command_x = 0.0

            action = ramp.make_action(case, env, command_x=command_x, hold_yz=hold_yz)
            observation_state = np.asarray(base._obs_to_state(obs_for_frame), dtype=np.float32)
            relative, stored = absolute_action(
                observation_state,
                action,
                translation_scale_m=translation_scale_m,
            )
            env.step_count = push_start + min(frame_index, 15)
            _, _, _, info = env.step(action)
            row = dict(info["push_box"])
            row.update(
                {
                    "frame_index": int(frame_index),
                    "phase": phase,
                    "eef_x_before_m": float(eef_before[0]),
                    "eef_vx_before_mps": eef_vx,
                    "remaining_before_m": remaining,
                    "command_x": command_x,
                }
            )
            rows.append(row)
            relative_actions.append(relative)
            stored_actions.append(stored)

            has_contact = bool(touch.board_box_contacts(env))
            if has_contact:
                contact_frames.append(int(frame_index))
                if not contact_active:
                    contact_episode_count += 1
            contact_active = has_contact

            agent, wrist = base._obs_to_images(obs_for_frame)
            frame = {
                "observation.images.image": agent,
                "observation.images.wrist_image": wrist,
                "observation.state": observation_state,
                "action": stored,
            }
            dataset.add_frame(frame, task=task, timestamp=float(frame_index) / float(fps))
            base.write_image_for_last_frame(
                dataset,
                "observation.images.image",
                frame_index,
                agent,
                jpeg_quality=jpeg_quality,
            )
            base.write_image_for_last_frame(
                dataset,
                "observation.images.wrist_image",
                frame_index,
                wrist,
                jpeg_quality=jpeg_quality,
            )

        final_eef = np.asarray(env._last_obs["robot0_eef_pos"], dtype=np.float64).copy()
        final_box_xyz, _ = env.box_pose()
        dataset.save_episode()
    finally:
        env.close()

    eef_x = np.asarray([float(row["eef_x_before_m"]) for row in rows] + [float(final_eef[0])])
    eef_travel = eef_x - float(launch_eef[0])
    eef_vx = np.diff(eef_x) * float(fps)
    commands = np.asarray([float(row["command_x"]) for row in rows])
    significant = commands[np.abs(commands) > 1e-3]
    sign_changes = int(np.sum(np.sign(significant[1:]) != np.sign(significant[:-1]))) if len(significant) > 1 else 0
    box_velocity = np.asarray([row["box_vxy"] for row in rows], dtype=np.float64)
    absolute_array = np.asarray(stored_actions, dtype=np.float32)
    relative_array = np.asarray(relative_actions, dtype=np.float32)
    return episode_index, {
        "A": float(amplitude),
        "first_frame_fraction": float(first_fraction),
        "first_frame_action": float(amplitude * first_fraction),
        "target_travel_m": float(travel_m),
        "max_eef_travel_m": float(np.max(eef_travel)),
        "final_eef_travel_m": float(final_eef[0] - launch_eef[0]),
        "eef_overshoot_m": float(max(0.0, np.max(eef_travel) - travel_m)),
        "peak_eef_vx_mps": float(np.max(eef_vx)),
        "minimum_eef_vx_mps": float(np.min(eef_vx)),
        "brake_trigger_frame": brake_trigger_frame,
        "brake_frames": int(brake_frames),
        "significant_x_action_sign_changes": sign_changes,
        "nonzero_x_action_frames": int(np.sum(np.abs(commands) > 1e-3)),
        "sampled_contact_frames": contact_frames,
        "sampled_contact_episode_count": int(contact_episode_count),
        "final_phase": phase,
        "final_box_displacement_from_launch_m": float(
            np.linalg.norm(np.asarray(final_box_xyz[:2]) - np.asarray(launch_box_xyz[:2]))
        ),
        "final_box_forward_from_launch_m": float(final_box_xyz[0] - launch_box_xyz[0]),
        "final_box_lateral_from_launch_m": float(final_box_xyz[1] - launch_box_xyz[1]),
        "peak_box_vx_mps": float(np.max(box_velocity[:, 0])),
        "initial_box_xyz_m": np.asarray(initial_box_xyz, dtype=float).tolist(),
        "launch_box_xyz_m": np.asarray(launch_box_xyz, dtype=float).tolist(),
        "final_box_xyz_m": np.asarray(final_box_xyz, dtype=float).tolist(),
        "absolute_action_min": absolute_array.min(axis=0).astype(float).tolist(),
        "absolute_action_max": absolute_array.max(axis=0).astype(float).tolist(),
        "relative_action_min": relative_array.min(axis=0).astype(float).tolist(),
        "relative_action_max": relative_array.max(axis=0).astype(float).tolist(),
        "touch_preparation": touch_state,
    }


def main() -> None:
    args = parse_args()
    experiment = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    source_experiment = json.loads(formal.CONFIG_PATH.read_text(encoding="utf-8"))
    config = formal.configure_dataset(source_experiment)
    config["dataset_name"] = str(experiment["dataset_name"])
    config["camera_resolution"] = int(experiment["camera_resolution"])
    config["fps"] = int(experiment["fps"])
    config["frictions"] = [float(experiment["friction_mu"])]
    expected = int(experiment["expected_episode_count"])
    if len(config["actions"]) != expected:
        raise RuntimeError(f"Source action grid has {len(config['actions'])} actions, expected {expected}")

    output_root = args.output_root.resolve()
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_root} exists; pass --overwrite")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    recording = experiment["recording"]
    action_schema = experiment["absolute_action"]
    source_template = source_experiment["action_template"]
    base.patch_lerobot_video_crf(int(recording["video_crf"]))
    features = copy.deepcopy(base.build_features(int(config["camera_resolution"])))
    features["action"]["names"] = list(action_schema["names"])
    dataset_root = output_root / "hidden_straight_lerobot"
    dataset = base.LeRobotDataset.create(
        repo_id="libero_push_box_board_touch_mu0050_30action_fixed5cm_absolute_eef_xyz_hai_machine",
        root=dataset_root,
        fps=int(config["fps"]),
        features=features,
        use_videos=True,
        video_codec=str(recording["video_codec"]),
        is_compute_episode_stats_image=False,
    )

    created_at = dt.datetime.now().isoformat()
    rows: list[dict[str, Any]] = []
    metadata = {
        "created_at": created_at,
        "dataset_type": "libero_push_box_board_touch_mu0050_30action_fixed5cm_absolute_eef_xyz_lerobot_hai-machine",
        "friction_mu": float(experiment["friction_mu"]),
        "target_visible": False,
        "split": "straight",
        "camera_resolution": int(config["camera_resolution"]),
        "fps": int(config["fps"]),
        "action_schema": dict(action_schema),
        "source_action_grid": dict(source_experiment["action_schedule"]),
        "source_action_template": dict(source_template),
        "episodes": [],
    }
    manifest = {
        "created_at": created_at,
        "config_path": str(args.config.resolve()),
        "output_root": str(output_root),
        "hidden_straight_lerobot": str(dataset_root),
        "episodes": [],
    }

    def autosave() -> None:
        write_json(output_root / "manifest.json", manifest)
        base.write_dataset_metadata(dataset_root, metadata, rows)

    mu = float(experiment["friction_mu"])
    for action_index, action_cfg in enumerate(config["actions"]):
        action_id = int(action_cfg["action_id"])
        amplitude = float(action_cfg["A"])
        case_id = f"board_touch_mu0050_a{action_id:02d}_A{int(round(amplitude * 1000)):03d}_fixed5cm_absxyz"
        bddl = base.write_hidden_bddl(config, bddl_dir=output_root / "bddl", geometry_id=case_id)
        base_case = collector.make_case(
            config,
            mu=mu,
            action_cfg=action_cfg,
            case_id=case_id,
            bddl_file=bddl,
        )
        case = ramp.preserve_case_attributes(
            base_case,
            replace(base_case, pusher_max_pos_action=float(source_template["pusher_max_pos_action"])),
        )
        episode_index, metrics = rollout_absolute(
            case,
            dataset=dataset,
            amplitude=amplitude,
            first_fraction=float(source_template["first_frame_fraction"]),
            travel_m=float(source_template["target_eef_travel_m"]),
            controller_cfg=source_template["controller"],
            recorded_steps=int(source_template["post_launch_recording_frames"]),
            seed=int(args.seed),
            fps=int(config["fps"]),
            jpeg_quality=int(recording["jpeg_quality"]),
            translation_scale_m=float(action_schema["translation_scale_m_per_normalized_unit"]),
        )
        row = {
            "episode_index": int(episode_index),
            "case_id": case_id,
            "action_index": int(action_index),
            "action_id": action_id,
            "mu": mu,
            "A": amplitude,
            "action_schema": str(action_schema["schema"]),
            "metrics": metrics,
        }
        rows.append(row)
        metadata["episodes"].append(row)
        manifest["episodes"].append(row)
        print(
            f"collect {action_index + 1:02d}/{expected:02d} mu={mu:.3f} A={amplitude:.6f} "
            f"disp={metrics['final_box_displacement_from_launch_m'] * 100.0:.2f}cm "
            f"peak_vx={metrics['peak_box_vx_mps']:.3f}m/s "
            f"eef_max={metrics['max_eef_travel_m'] * 100.0:.2f}cm",
            flush=True,
        )
        autosave()

    absolute_min = np.min(
        np.asarray([row["metrics"]["absolute_action_min"] for row in rows], dtype=np.float64),
        axis=0,
    )
    absolute_max = np.max(
        np.asarray([row["metrics"]["absolute_action_max"] for row in rows], dtype=np.float64),
        axis=0,
    )
    conversion = {
        **action_schema,
        "paired_episode_count": len(rows),
        "paired_frame_count": len(rows) * int(source_template["post_launch_recording_frames"]),
        "absolute_action_min": absolute_min.astype(float).tolist(),
        "absolute_action_max": absolute_max.astype(float).tolist(),
    }
    summary = {
        "episode_count": len(rows),
        "expected_episode_count": expected,
        "friction_mu": mu,
        "action_range_A": [float(config["action_peak_min"]), float(config["action_peak_max"])],
        "action_schema": str(action_schema["schema"]),
        "hidden_straight_lerobot": str(dataset_root),
    }
    write_json(dataset_root / "meta" / "absolute_action_conversion_2026-07-17_hai-machine.json", conversion)
    write_json(output_root / "summary.json", summary)
    autosave()
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
