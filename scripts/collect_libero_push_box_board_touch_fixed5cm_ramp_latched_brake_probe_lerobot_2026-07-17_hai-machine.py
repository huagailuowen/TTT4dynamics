#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "collect_libero_push_box_board_touch_fixed_travel_probe_lerobot_2026-07-17_hai-machine.py"
)
CONFIG_PATH = (
    REPO_ROOT
    / "configs"
    / "libero_push_box_board_touch_fixed5cm_ramp_latched_brake_probe_2026-07-17_hai-machine.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "outputs"
    / "pushbox"
    / "board_touch"
    / "fixed_travel"
    / "libero_push_box_board_touch_fixed5cm_ramp_latched_brake_12eps_lerobot_2026-07-17_hai-machine"
)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


source = load_module(SOURCE_SCRIPT, "fixed5cm_ramp_latched_source_hai_machine")
collector = source.collector
base = source.base
touch = source.touch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe ramped high actions with a one-way latched brake.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def copy_obs(obs: dict[str, Any]) -> dict[str, Any]:
    return {
        key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value
        for key, value in obs.items()
    }


def preserve_case_attributes(original: Any, updated: Any) -> Any:
    object.__setattr__(updated, "hai_action_id", getattr(original, "hai_action_id"))
    object.__setattr__(updated, "hai_action_profile", getattr(original, "hai_action_profile"))
    return updated


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(base.to_jsonable(value), indent=2), encoding="utf-8")


def make_action(case: Any, env: Any, *, command_x: float, hold_yz: np.ndarray) -> np.ndarray:
    action = np.zeros(7, dtype=np.float64)
    action[0] = float(command_x)
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


def rollout(
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
) -> tuple[int, dict[str, Any]]:
    env = touch.make_env(case, seed=seed)
    rows: list[dict[str, Any]] = []
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
            obs_for_frame = copy_obs(env._last_obs)
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

            action = make_action(case, env, command_x=command_x, hold_yz=hold_yz)
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
                "observation.state": base._obs_to_state(obs_for_frame),
                "action": base._env_action_to_fastwam_action(action.astype(np.float32)),
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
        "touch_preparation": touch_state,
    }


def plot_results(path: Path, rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), constrained_layout=True)
    colors = {0.3: "#176b87", 0.5: "#2b8a66", 0.7: "#d17a22", 1.0: "#a13f35"}
    for fraction in sorted({float(row["first_frame_fraction"]) for row in rows}):
        selected = sorted(
            (row for row in rows if abs(float(row["first_frame_fraction"]) - fraction) < 1e-12),
            key=lambda row: float(row["A"]),
        )
        a = [float(row["A"]) for row in selected]
        color = colors[fraction]
        label = f"first={fraction:.1f}A"
        axes[0].plot(a, [row["final_box_displacement_from_launch_m"] * 100 for row in selected], "o-", color=color, label=label)
        axes[1].plot(a, [row["peak_box_vx_mps"] for row in selected], "o-", color=color)
        axes[2].plot(a, [row["max_eef_travel_m"] * 100 for row in selected], "o-", color=color)
    axes[0].set_ylabel("Box displacement (cm)")
    axes[1].set_ylabel("Peak box vx (m/s)")
    axes[2].set_ylabel("Maximum EEF travel (cm)")
    axes[0].legend(frameon=False)
    for axis in axes:
        axis.set_xlabel("Peak action A")
        axis.grid(alpha=0.25)
    travel_cm = float(rows[0]["target_travel_m"]) * 100.0
    fig.suptitle(f"Fixed {travel_cm:g} cm: ramped first frame and latched one-way brake")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_root} exists; pass --overwrite")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    experiment = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    formal_experiment = json.loads(collector.CONFIG_PATH.read_text(encoding="utf-8"))
    config = collector.configure_dataset(formal_experiment)
    config["camera_resolution"] = int(experiment["camera_resolution"])
    config["fps"] = int(experiment["fps"])
    total = len(experiment["peak_amplitudes"]) * len(experiment["first_frame_fractions"])
    if total != int(experiment["expected_episode_count"]):
        raise RuntimeError(f"Configured {total} episodes, expected {experiment['expected_episode_count']}")

    base.patch_lerobot_video_crf(int(experiment["recording"]["video_crf"]))
    dataset_root = output_root / "lerobot"
    dataset = base.LeRobotDataset.create(
        repo_id="libero_push_box_board_touch_fixed_travel_ramp_latched_brake_probe_hai_machine",
        root=dataset_root,
        fps=int(experiment["fps"]),
        features=base.build_features(int(experiment["camera_resolution"])),
        use_videos=True,
        video_codec=str(experiment["recording"]["video_codec"]),
        is_compute_episode_stats_image=False,
    )

    rows: list[dict[str, Any]] = []
    metadata = {
        "created_at": dt.datetime.now().isoformat(),
        "dataset_type": "libero_push_box_board_touch_fixed_travel_ramp_latched_brake_probe_lerobot_hai-machine",
        "experiment_config": experiment,
        "episodes": [],
    }
    count = 0
    mu = float(experiment["friction"])
    travel_tag = int(round(float(experiment["travel_m"]) * 1000.0))
    for fraction_index, first_fraction in enumerate(experiment["first_frame_fractions"]):
        for action_id, amplitude in enumerate(experiment["peak_amplitudes"]):
            case_id = (
                f"fixed_travel_L{travel_tag:03d}_ramp_f{int(round(float(first_fraction) * 100)):03d}_"
                f"A{int(round(float(amplitude) * 1000)):04d}_mu{int(round(mu * 10000)):04d}"
            )
            action_cfg = {"action_id": int(action_id), "A": float(amplitude), "push_steps": 16}
            bddl = base.write_hidden_bddl(config, bddl_dir=output_root / "bddl", geometry_id=case_id)
            base_case = collector.make_case(
                config,
                mu=mu,
                action_cfg=action_cfg,
                case_id=case_id,
                bddl_file=bddl,
            )
            case = preserve_case_attributes(
                base_case,
                replace(base_case, pusher_max_pos_action=float(experiment["pusher_max_pos_action"])),
            )
            episode_index, metrics = rollout(
                case,
                dataset=dataset,
                amplitude=float(amplitude),
                first_fraction=float(first_fraction),
                travel_m=float(experiment["travel_m"]),
                controller_cfg=experiment["controller"],
                recorded_steps=int(experiment["recorded_steps"]),
                seed=int(args.seed),
                fps=int(experiment["fps"]),
                jpeg_quality=int(experiment["recording"]["jpeg_quality"]),
            )
            row = {
                "episode_index": int(episode_index),
                "case_id": case_id,
                "fraction_index": int(fraction_index),
                "action_id": int(action_id),
                "mu": mu,
                "A": float(amplitude),
                "first_frame_fraction": float(first_fraction),
                "metrics": metrics,
            }
            rows.append(row)
            metadata["episodes"].append(row)
            count += 1
            print(
                f"ramp {count:02d}/{total:02d} first={float(first_fraction):.1f}A A={float(amplitude):.2f} "
                f"box={metrics['final_box_displacement_from_launch_m'] * 100:.2f}cm "
                f"peak_vx={metrics['peak_box_vx_mps']:.3f}m/s "
                f"eef_max={metrics['max_eef_travel_m'] * 100:.2f}cm "
                f"sign_changes={metrics['significant_x_action_sign_changes']}",
                flush=True,
            )
            base.write_dataset_metadata(dataset_root, metadata, rows)

    summary = {
        "experiment": experiment["experiment"],
        "episode_count": len(rows),
        "experiment_config": experiment,
        "lerobot_root": str(dataset_root),
        "results": rows,
    }
    write_json(output_root / "summary.json", summary)
    base.write_dataset_metadata(dataset_root, metadata, rows)
    plot_results(output_root / "fixed_travel_ramp_latched_brake_comparison.png", [row["metrics"] for row in rows])
    print(f"summary={output_root / 'summary.json'}", flush=True)
    print(f"plot={output_root / 'fixed_travel_ramp_latched_brake_comparison.png'}", flush=True)
    print(f"lerobot={dataset_root}", flush=True)


if __name__ == "__main__":
    main()
