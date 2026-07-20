#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
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
    / "collect_libero_push_box_board_touch_20fric_30action_full8_A450_lerobot_2026-07-17_hai-machine.py"
)
PROBE_CONFIG_PATH = (
    REPO_ROOT
    / "configs"
    / "libero_push_box_board_touch_fixed_travel_probe_2026-07-17_hai-machine.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "outputs"
    / "pushbox"
    / "board_touch"
    / "fixed_travel"
    / "libero_push_box_board_touch_fixed_travel_3_5_8cm_3mu_6A_lerobot_2026-07-17_hai-machine"
)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


collector = load_module(SOURCE_SCRIPT, "fixed_travel_probe_source_hai_machine")
base = collector.base
touch = collector.touch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect a fixed-EEF-travel rigid-board LeRobot probe.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def copy_obs(obs: dict[str, Any]) -> dict[str, Any]:
    return {
        key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value
        for key, value in obs.items()
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(base.to_jsonable(value), indent=2), encoding="utf-8")


def make_action(
    case: Any,
    env: Any,
    *,
    command_x: float,
    hold_yz: np.ndarray,
) -> np.ndarray:
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
    mu: float,
    amplitude: float,
    travel_m: float,
    controller_cfg: dict[str, Any],
    recorded_steps: int,
    seed: int,
    fps: int,
    jpeg_quality: int,
) -> tuple[int, dict[str, Any]]:
    env = touch.make_env(case, seed=seed)
    rows: list[dict[str, Any]] = []
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
        reached_frame = None
        base.remove_current_episode_images(dataset)
        episode_index = int(dataset.meta.total_episodes)
        task = base.prompt_for_case("observation", "straight")

        for frame_index in range(int(recorded_steps)):
            obs_for_frame = copy_obs(env._last_obs)
            eef_before = np.asarray(obs_for_frame["robot0_eef_pos"], dtype=np.float64)
            eef_vx = float((eef_before[0] - previous_x) * float(fps)) if frame_index else 0.0
            previous_x = float(eef_before[0])
            error_x = float(target_x - eef_before[0])
            raw_command_x = (
                float(controller_cfg["position_gain_action_per_m"]) * error_x
                - float(controller_cfg["velocity_damping_action_per_mps"]) * eef_vx
            )
            command_x = float(np.clip(raw_command_x, -float(amplitude), float(amplitude)))
            action = make_action(case, env, command_x=command_x, hold_yz=hold_yz)
            env.step_count = push_start + min(frame_index, 15)
            _, _, _, info = env.step(action)
            row = dict(info["push_box"])
            row["frame_index"] = int(frame_index)
            row["eef_x_before_m"] = float(eef_before[0])
            row["eef_vx_before_mps"] = eef_vx
            row["endpoint_error_before_m"] = error_x
            row["command_x"] = command_x
            rows.append(row)

            eef_after = np.asarray(env._last_obs["robot0_eef_pos"], dtype=np.float64)
            if reached_frame is None and float(eef_after[0]) >= target_x - float(controller_cfg["target_tolerance_m"]):
                reached_frame = int(frame_index)
            contacts = touch.board_box_contacts(env)
            has_contact = bool(contacts)
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
    box_velocity = np.asarray([row["box_vxy"] for row in rows], dtype=np.float64)
    commands = np.asarray([float(row["command_x"]) for row in rows], dtype=np.float64)
    return episode_index, {
        "mu": float(mu),
        "A": float(amplitude),
        "target_travel_m": float(travel_m),
        "launch_eef_xyz_m": launch_eef.astype(float).tolist(),
        "target_eef_x_m": target_x,
        "final_eef_xyz_m": final_eef.astype(float).tolist(),
        "final_eef_travel_m": float(final_eef[0] - launch_eef[0]),
        "max_eef_travel_m": float(np.max(eef_travel)),
        "eef_overshoot_m": float(max(0.0, np.max(eef_travel) - travel_m)),
        "final_eef_error_m": float(target_x - final_eef[0]),
        "peak_eef_vx_mps": float(np.max(eef_vx)) if len(eef_vx) else 0.0,
        "minimum_eef_vx_mps": float(np.min(eef_vx)) if len(eef_vx) else 0.0,
        "reached_frame": reached_frame,
        "nonzero_x_action_frames": int(np.sum(np.abs(commands) > 1e-4)),
        "negative_brake_frames": int(np.sum(commands < -1e-4)),
        "initial_box_xyz_m": np.asarray(initial_box_xyz, dtype=float).tolist(),
        "launch_box_xyz_m": np.asarray(launch_box_xyz, dtype=float).tolist(),
        "final_box_xyz_m": np.asarray(final_box_xyz, dtype=float).tolist(),
        "final_box_displacement_from_launch_m": float(
            np.linalg.norm(np.asarray(final_box_xyz[:2]) - np.asarray(launch_box_xyz[:2]))
        ),
        "final_box_forward_from_launch_m": float(final_box_xyz[0] - launch_box_xyz[0]),
        "final_box_lateral_from_launch_m": float(final_box_xyz[1] - launch_box_xyz[1]),
        "peak_box_vx_mps": float(np.max(box_velocity[:, 0])),
        "sampled_contact_frames": contact_frames,
        "sampled_contact_episode_count": int(contact_episode_count),
        "touch_preparation": touch_state,
    }


def plot_results(path: Path, rows: list[dict[str, Any]], frictions: list[float]) -> None:
    fig, axes = plt.subplots(len(frictions), 2, figsize=(12.0, 10.0), constrained_layout=True)
    colors = {0.03: "#176b87", 0.05: "#d17a22", 0.08: "#a13f35"}
    for row_index, mu in enumerate(frictions):
        selected_mu = [row for row in rows if abs(float(row["mu"]) - mu) < 1e-12]
        for travel in sorted({float(row["target_travel_m"]) for row in selected_mu}):
            selected = sorted(
                (row for row in selected_mu if abs(float(row["target_travel_m"]) - travel) < 1e-12),
                key=lambda row: float(row["A"]),
            )
            a = [float(row["A"]) for row in selected]
            color = colors[travel]
            label = f"target {travel * 100:.0f} cm"
            axes[row_index, 0].plot(
                a,
                [float(row["max_eef_travel_m"]) * 100.0 for row in selected],
                "o-",
                color=color,
                label=label,
            )
            axes[row_index, 1].plot(
                a,
                [float(row["final_box_displacement_from_launch_m"]) * 100.0 for row in selected],
                "o-",
                color=color,
                label=label,
            )
        axes[row_index, 0].set_ylabel(f"mu={mu:.3f}\nMax EEF travel (cm)")
        axes[row_index, 1].set_ylabel(f"mu={mu:.3f}\nBox displacement (cm)")
        for axis in axes[row_index]:
            axis.set_xlabel("Action limit A")
            axis.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)
    axes[0, 1].legend(frameon=False)
    fig.suptitle("Strict-touch board: speed-limited fixed EEF travel")
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

    probe = json.loads(PROBE_CONFIG_PATH.read_text(encoding="utf-8"))
    formal_experiment = json.loads(collector.CONFIG_PATH.read_text(encoding="utf-8"))
    config = collector.configure_dataset(formal_experiment)
    config["camera_resolution"] = int(probe["camera_resolution"])
    config["fps"] = int(probe["fps"])
    expected = len(probe["frictions"]) * len(probe["amplitudes"]) * len(probe["travel_lengths_m"])
    if expected != int(probe["expected_episode_count"]):
        raise RuntimeError(f"Configured {expected} episodes, expected {probe['expected_episode_count']}")

    base.patch_lerobot_video_crf(int(probe["recording"]["video_crf"]))
    dataset_root = output_root / "lerobot"
    dataset = base.LeRobotDataset.create(
        repo_id="libero_push_box_board_touch_fixed_travel_probe_hai_machine",
        root=dataset_root,
        fps=int(probe["fps"]),
        features=base.build_features(int(probe["camera_resolution"])),
        use_videos=True,
        video_codec=str(probe["recording"]["video_codec"]),
        is_compute_episode_stats_image=False,
    )

    rows: list[dict[str, Any]] = []
    metadata = {
        "created_at": dt.datetime.now().isoformat(),
        "dataset_type": "libero_push_box_board_touch_fixed_travel_probe_lerobot_hai-machine",
        "probe_config": probe,
        "episodes": [],
    }
    count = 0
    for travel_index, travel_m in enumerate(probe["travel_lengths_m"]):
        for mu_index, mu in enumerate(probe["frictions"]):
            for action_id, amplitude in enumerate(probe["amplitudes"]):
                case_id = (
                    f"fixed_travel_L{int(round(float(travel_m) * 1000)):03d}_"
                    f"mu{int(round(float(mu) * 10000)):04d}_A{int(round(float(amplitude) * 1000)):03d}"
                )
                action_cfg = {"action_id": int(action_id), "A": float(amplitude), "push_steps": 16}
                bddl = base.write_hidden_bddl(
                    config,
                    bddl_dir=output_root / "bddl",
                    geometry_id=case_id,
                )
                case = collector.make_case(
                    config,
                    mu=float(mu),
                    action_cfg=action_cfg,
                    case_id=case_id,
                    bddl_file=bddl,
                )
                episode_index, metrics = rollout(
                    case,
                    dataset=dataset,
                    mu=float(mu),
                    amplitude=float(amplitude),
                    travel_m=float(travel_m),
                    controller_cfg=probe["controller"],
                    recorded_steps=int(probe["recorded_steps"]),
                    seed=int(args.seed),
                    fps=int(probe["fps"]),
                    jpeg_quality=int(probe["recording"]["jpeg_quality"]),
                )
                row = {
                    "episode_index": int(episode_index),
                    "case_id": case_id,
                    "travel_index": int(travel_index),
                    "mu_index": int(mu_index),
                    "action_id": int(action_id),
                    "mu": float(mu),
                    "A": float(amplitude),
                    "target_travel_m": float(travel_m),
                    "controller": dict(probe["controller"]),
                    "metrics": metrics,
                }
                rows.append(row)
                metadata["episodes"].append(row)
                count += 1
                print(
                    f"probe {count:02d}/{expected:02d} L={float(travel_m) * 100:.0f}cm "
                    f"mu={float(mu):.3f} A={float(amplitude):.3f} "
                    f"eef_max={metrics['max_eef_travel_m'] * 100:.2f}cm "
                    f"overshoot={metrics['eef_overshoot_m'] * 100:.2f}cm "
                    f"box={metrics['final_box_displacement_from_launch_m'] * 100:.2f}cm",
                    flush=True,
                )
                base.write_dataset_metadata(dataset_root, metadata, rows)

    summary = {
        "experiment": probe["experiment"],
        "episode_count": len(rows),
        "probe_config": probe,
        "lerobot_root": str(dataset_root),
        "results": rows,
    }
    write_json(output_root / "summary.json", summary)
    base.write_dataset_metadata(dataset_root, metadata, rows)
    plot_results(output_root / "fixed_travel_comparison.png", [row["metrics"] for row in rows], [float(mu) for mu in probe["frictions"]])
    print(f"summary={output_root / 'summary.json'}", flush=True)
    print(f"plot={output_root / 'fixed_travel_comparison.png'}", flush=True)
    print(f"lerobot={dataset_root}", flush=True)


if __name__ == "__main__":
    main()
