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

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BOARD_TOUCH_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "render_libero_push_box_gripped_board_gap_vs_touch_probe_2026-07-17_hai-machine.py"
)
CONFIG_PATH = (
    REPO_ROOT
    / "configs"
    / "libero_push_box_board_touch_20fric_30action_full8_A450_2026-07-17_hai-machine.json"
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


touch = load_module(BOARD_TOUCH_SCRIPT, "formal_board_touch_collector_hai_machine")
board = touch.board_probe
source_dataset = board.source_dataset
source = board.source
base = source.base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect the formal 20-friction x 30-action strict-touch rigid-board LeRobot dataset."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--video-codec",
        default=source.VIDEO_CODEC,
        choices=["h264", "hevc", "libsvtav1", "h264_nvenc"],
    )
    parser.add_argument("--video-crf", type=int, default=source.VIDEO_CRF)
    parser.add_argument("--jpeg-quality", type=int, default=source.JPEG_QUALITY)
    return parser.parse_args()


def segmented_values(schedule: dict[str, Any]) -> list[float]:
    values: list[float] = []
    for segment in schedule["segments"]:
        values.extend(
            np.linspace(
                float(segment["start"]),
                float(segment["stop"]),
                int(segment["count"]),
                endpoint=bool(segment["endpoint"]),
                dtype=np.float64,
            ).astype(float).tolist()
        )
    return values


def configure_dataset(experiment: dict[str, Any]) -> dict[str, Any]:
    config = source_dataset.configure_dataset()
    prepare = dict(config["prepare_config"])
    prepare["descend_steps"] = 45
    prepare["prepare_position_gain"] = 8.0
    config["prepare_config"] = prepare

    frictions = segmented_values(experiment["friction_schedule"])
    action_schedule = experiment["action_schedule"]
    amplitudes = np.linspace(
        float(action_schedule["minimum_A"]),
        float(action_schedule["maximum_A"]),
        int(action_schedule["count"]),
        endpoint=bool(action_schedule["endpoint"]),
        dtype=np.float64,
    ).astype(float).tolist()
    profile = [float(value) for value in experiment["action_template"]["normalized_profile"]]
    actions = [
        {
            "action_id": int(action_id),
            "A": float(amplitude),
            "push_steps": int(experiment["action_template"]["controller_push_phase_steps"]),
            "profile": profile,
            "description": "strict-touch fixed full8 profile; peak A is the only action variable",
        }
        for action_id, amplitude in enumerate(amplitudes)
    ]

    config["dataset_name"] = str(experiment["dataset_name"])
    config["camera_resolution"] = int(experiment["camera_resolution"])
    config["fps"] = int(experiment["fps"])
    config["frictions"] = frictions
    config["friction_count"] = len(frictions)
    config["friction_min"] = min(frictions)
    config["friction_max"] = max(frictions)
    config["friction_schedule"] = experiment["friction_schedule"]
    config["actions"] = actions
    config["action_count"] = len(actions)
    config["action_peak_min"] = min(amplitudes)
    config["action_peak_max"] = max(amplitudes)
    config["action_template"] = dict(experiment["action_template"])
    config["touch_preparation"] = dict(experiment["touch_preparation"])
    config["board"] = dict(experiment["board"])
    return config


def make_case(
    config: dict[str, Any],
    *,
    mu: float,
    action_cfg: dict[str, Any],
    case_id: str,
    bddl_file: str,
) -> Any:
    case = base.build_fixed_case(
        config,
        mu=float(mu),
        action_cfg=action_cfg,
        case_id=case_id,
        bddl_file=bddl_file,
        camera_resolution=int(config["camera_resolution"]),
    )
    updated = replace(
        case,
        pusher_push_yz_hold_gain=8.0,
        pusher_push_yz_max_action=0.25,
        pusher_gripper=1.0,
    )
    object.__setattr__(updated, "hai_action_id", getattr(case, "hai_action_id"))
    object.__setattr__(updated, "hai_action_profile", getattr(case, "hai_action_profile"))
    return updated


def serial_geometry(state: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.astype(float).tolist() if isinstance(value, np.ndarray) else value
        for key, value in state.items()
    }


def prepare_touch_unrecorded(env: Any, *, case: Any) -> dict[str, Any]:
    push_start = int(case.pusher_approach_steps) + int(case.pusher_descend_steps)
    sampled_contact = False
    steps = 0
    for steps in range(1, int(touch.TOUCH_MAX_STEPS) + 1):
        geometry = touch.geometry_state(env)
        contacts = touch.board_box_contacts(env)
        if contacts:
            sampled_contact = True
            break

        gap = float(geometry["gap_m"])
        if gap > 0.015:
            action_x = 0.10
        elif gap > 0.004:
            action_x = 0.045
        elif gap > float(touch.TOUCH_TARGET_GAP_M):
            action_x = 0.012
        else:
            action_x = 0.008
        action = np.zeros(7, dtype=np.float64)
        action[0] = action_x
        action[1] = float(np.clip(-5.0 * float(geometry["center_y_error_m"]), -0.06, 0.06))
        action[-1] = float(case.pusher_gripper)
        env.step_count = push_start - 1
        env.step(action)
        if touch.board_box_contacts(env):
            sampled_contact = True
            break

    final_geometry = touch.geometry_state(env)
    if not sampled_contact:
        raise RuntimeError(
            "Touch preparation did not reach a sampled board-box contact; "
            f"final gap={float(final_geometry['gap_m']) * 1000.0:.3f} mm"
        )
    return {
        "steps": int(steps),
        "sampled_contact": True,
        "state": serial_geometry(final_geometry),
        "box_speed_mps": float(np.linalg.norm(env.box_velocity()[:2])),
        "recorded": False,
    }


def push_action(case: Any, env: Any, *, amplitude: float, profile_value: float, hold_yz: np.ndarray) -> np.ndarray:
    action = np.zeros(7, dtype=np.float64)
    action[0] = float(amplitude) * float(profile_value)
    eef = np.asarray(env._last_obs["robot0_eef_pos"], dtype=np.float64)
    yz = float(case.pusher_push_yz_hold_gain) * (hold_yz - eef[1:3])
    action[1:3] = np.clip(
        yz,
        -float(case.pusher_push_yz_max_action),
        float(case.pusher_push_yz_max_action),
    )
    action[:3] = np.clip(
        action[:3],
        -float(case.pusher_max_pos_action),
        float(case.pusher_max_pos_action),
    )
    action[-1] = float(case.pusher_gripper)
    return action


def rollout_to_lerobot(
    case: Any,
    *,
    dataset: Any,
    amplitude: float,
    profile: list[float],
    post_launch_steps: int,
    seed: int,
    fps: int,
    jpeg_quality: int,
) -> tuple[int, dict[str, Any]]:
    env = touch.make_env(case, seed=seed)
    rows: list[dict[str, Any]] = []
    sampled_contact_frames: list[int] = []
    contact_episode_count = 0
    contact_active = False
    release_box_vxy = None
    try:
        base.preposition_fixed_start(env)
        env.step_count = 0
        env._last_scripted_action = np.zeros(7, dtype=np.float64)
        env._last_scripted_phase = None
        initial_box_xyz, _ = env.box_pose()
        push_start = int(case.pusher_approach_steps) + int(case.pusher_descend_steps)
        for _ in range(push_start):
            env.step()

        touch_state = prepare_touch_unrecorded(env, case=case)
        launch_box_xyz, _ = env.box_pose()
        launch_geometry = serial_geometry(touch.geometry_state(env))
        launch_box_speed = float(np.linalg.norm(env.box_velocity()[:2]))
        hold_yz = np.asarray(env._last_obs["robot0_eef_pos"], dtype=np.float64)[1:3].copy()

        base.remove_current_episode_images(dataset)
        episode_index = int(dataset.meta.total_episodes)
        task = base.prompt_for_case("observation", "straight")
        for local in range(int(post_launch_steps)):
            profile_value = float(profile[local]) if local < len(profile) else 0.0
            action = push_action(
                case,
                env,
                amplitude=float(amplitude),
                profile_value=profile_value,
                hold_yz=hold_yz,
            )
            obs_for_frame = env._last_obs
            env.step_count = push_start + min(local, len(profile) - 1)
            _, _, _, info = env.step(action)
            row = dict(info["push_box"])
            row["local_frame"] = int(local)
            row["profile_value"] = float(profile_value)
            rows.append(row)

            contacts = touch.board_box_contacts(env)
            has_contact = bool(contacts)
            if has_contact:
                sampled_contact_frames.append(int(local))
                if not contact_active:
                    contact_episode_count += 1
            contact_active = has_contact
            if local == len(profile) - 1:
                release_box_vxy = np.asarray(row["box_vxy"], dtype=np.float64).copy()

            agent, wrist = base._obs_to_images(obs_for_frame)
            frame = {
                "observation.images.image": agent,
                "observation.images.wrist_image": wrist,
                "observation.state": base._obs_to_state(obs_for_frame),
                "action": base._env_action_to_fastwam_action(action.astype(np.float32)),
            }
            dataset.add_frame(frame, task=task, timestamp=float(local) / float(fps))
            base.write_image_for_last_frame(
                dataset,
                "observation.images.image",
                local,
                agent,
                jpeg_quality=jpeg_quality,
            )
            base.write_image_for_last_frame(
                dataset,
                "observation.images.wrist_image",
                local,
                wrist,
                jpeg_quality=jpeg_quality,
            )
        dataset.save_episode()
    finally:
        env.close()

    if not rows or release_box_vxy is None:
        raise RuntimeError("Board-touch rollout ended before the fixed action profile was released")
    final_box_xyz = np.asarray(rows[-1]["box_xyz"], dtype=np.float64)
    velocity = np.asarray([row["box_vxy"] for row in rows], dtype=np.float64)
    return episode_index, {
        "steps": len(rows),
        "initial_box_xyz_m": initial_box_xyz.astype(float).tolist(),
        "launch_box_xyz_m": np.asarray(launch_box_xyz, dtype=np.float64).astype(float).tolist(),
        "final_box_xyz_m": final_box_xyz.astype(float).tolist(),
        "prelaunch_box_drift_m": float(np.linalg.norm(launch_box_xyz[:2] - initial_box_xyz[:2])),
        "launch_box_speed_mps": launch_box_speed,
        "final_displacement_from_launch_m": float(np.linalg.norm(final_box_xyz[:2] - launch_box_xyz[:2])),
        "final_forward_from_launch_m": float(final_box_xyz[0] - launch_box_xyz[0]),
        "final_lateral_from_launch_m": float(final_box_xyz[1] - launch_box_xyz[1]),
        "peak_box_vx_mps": float(np.max(velocity[:, 0])),
        "release_box_vxy_mps": release_box_vxy.astype(float).tolist(),
        "sampled_contact_frames": sampled_contact_frames,
        "sampled_contact_episode_count": int(contact_episode_count),
        "launch_geometry": launch_geometry,
        "touch_preparation": touch_state,
    }


def create_dataset(root: Path, *, config: dict[str, Any], video_codec: str) -> Any:
    return base.LeRobotDataset.create(
        repo_id="libero_push_box_board_touch_20fric_30action_full8_A450_hidden_hai_machine",
        root=root,
        fps=int(config["fps"]),
        features=base.build_features(int(config["camera_resolution"])),
        use_videos=True,
        video_codec=video_codec,
        is_compute_episode_stats_image=False,
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(base.to_jsonable(value), indent=2), encoding="utf-8")


def collect(
    experiment: dict[str, Any],
    config: dict[str, Any],
    *,
    output_root: Path,
    overwrite: bool,
    seed: int,
    video_codec: str,
    video_crf: int,
    jpeg_quality: int,
) -> dict[str, Any]:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} exists; pass --overwrite")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    base.patch_lerobot_video_crf(video_crf)
    dataset_root = output_root / "hidden_straight_lerobot"
    dataset = create_dataset(dataset_root, config=config, video_codec=video_codec)
    profile = [float(value) for value in config["action_template"]["normalized_profile"]]
    post_launch_steps = int(config["action_template"]["post_launch_recording_frames"])

    rows: list[dict[str, Any]] = []
    created_at = dt.datetime.now().isoformat()
    dataset_type = "libero_push_box_board_touch_20fric_30action_full8_A450_hidden_lerobot_hai-machine"
    metadata = {
        "created_at": created_at,
        "dataset_type": dataset_type,
        "target_visible": False,
        "split": "straight",
        "camera_resolution": int(config["camera_resolution"]),
        "fps": int(config["fps"]),
        "video_codec": str(video_codec),
        "video_crf": int(video_crf),
        "jpeg_quality": int(jpeg_quality),
        "state_source": "true LIBERO obs robot0_eef_pos, robot0_eef_quat converted to axis-angle, robot0_gripper_qpos",
        "action_semantics": "FastWAM EEF delta-position action; first 8 recorded frames use full A",
        "action_template": dict(config["action_template"]),
        "touch_preparation": dict(config["touch_preparation"]),
        "board": dict(config["board"]),
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
                f"board_touch_m{mu_index:02d}_{base.mu_tag(mu)}_"
                f"a{action_id:02d}_A{int(round(amplitude * 1000)):03d}_full8"
            )
            bddl = base.write_hidden_bddl(
                config,
                bddl_dir=output_root / "bddl",
                geometry_id=case_id,
            )
            case = make_case(
                config,
                mu=float(mu),
                action_cfg=action_cfg,
                case_id=case_id,
                bddl_file=bddl,
            )
            episode_index, metrics = rollout_to_lerobot(
                case,
                dataset=dataset,
                amplitude=amplitude,
                profile=profile,
                post_launch_steps=post_launch_steps,
                seed=seed,
                fps=int(config["fps"]),
                jpeg_quality=jpeg_quality,
            )
            row = {
                "episode_index": int(episode_index),
                "case_id": case_id,
                "mu_index": int(mu_index),
                "mu": float(mu),
                "mu_tag": base.mu_tag(float(mu)),
                "action_id": action_id,
                "A": amplitude,
                "full_speed_frames": 8,
                "profile": profile,
                "profile_area": float(sum(profile) * amplitude),
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
                f"disp={metrics['final_displacement_from_launch_m'] * 100.0:.2f}cm "
                f"peak_vx={metrics['peak_box_vx_mps']:.3f}m/s "
                f"lateral={metrics['final_lateral_from_launch_m'] * 100.0:+.2f}cm",
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
        "action_profile": profile,
        "frictions": [float(mu) for mu in config["frictions"]],
    }
    write_json(output_root / "summary.json", summary)
    autosave()
    print(f"manifest={output_root / 'manifest.json'}", flush=True)
    print(f"hidden_root={dataset_root}", flush=True)
    return summary


def main() -> None:
    args = parse_args()
    experiment = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
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
        video_codec=str(args.video_codec),
        video_crf=int(args.video_crf),
        jpeg_quality=int(args.jpeg_quality),
    )
    print(json.dumps(base.to_jsonable(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()
