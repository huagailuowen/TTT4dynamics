#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPARISON_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "render_libero_push_box_gripped_board_gap_vs_touch_probe_2026-07-17_hai-machine.py"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "outputs"
    / "pushbox"
    / "board_touch"
    / "libero_push_box_gripped_board_touch_wide_A_mu015_2026-07-17_hai-machine"
)

AMPLITUDES = tuple(
    [round(value, 3) for value in np.arange(0.02, 0.201, 0.02)]
    + [round(value, 3) for value in np.arange(0.225, 0.501, 0.025)]
    + [0.525, 0.550, 0.575, 0.600, 0.650, 0.700, 0.750, 0.800]
)
FIXED_ACTION_PROFILE = np.asarray([0.5] + [1.0] * 7 + [0.5, 0.0], dtype=np.float64)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


comparison = load_module(COMPARISON_SCRIPT, "touch_wide_A_comparison_hai_machine")


def rollout_fixed_release(
    config: dict[str, Any],
    *,
    amplitude: float,
    action_id: int,
    bddl_file: str,
    seed: int,
) -> dict[str, Any]:
    case = comparison.board_probe.make_case(
        config,
        amplitude=amplitude,
        bddl_file=bddl_file,
        action_id=action_id,
    )
    env = comparison.make_env(case, seed=seed)
    frames: list[np.ndarray] = []
    rows = []
    sampled_contact_frames = []
    contact_episode_count = 0
    contact_active = False
    motion_frame = None
    release_box_vxy = None
    try:
        comparison.board_probe.source.base.preposition_fixed_start(env)
        env.step_count = 0
        env._last_scripted_action = np.zeros(7, dtype=np.float64)
        env._last_scripted_phase = None
        initial_box_xyz, _ = env.box_pose()
        push_start = int(case.pusher_approach_steps) + int(case.pusher_descend_steps)

        for _ in range(push_start):
            obs, _, _, info = env.step()
            comparison.append_frame(
                frames,
                env,
                obs,
                mode="touch",
                amplitude=amplitude,
                phase=str(info["push_box"]["phase"]),
                initial_box_xyz=initial_box_xyz,
            )

        touch = comparison.prepare_touch(
            env,
            case=case,
            frames=frames,
            amplitude=amplitude,
            initial_box_xyz=initial_box_xyz,
        )
        launch_box_xyz, _ = env.box_pose()
        launch_geometry = comparison.geometry_state(env)
        launch_box_speed = float(np.linalg.norm(env.box_velocity()[:2]))
        hold_yz = np.asarray(env._last_obs["robot0_eef_pos"], dtype=np.float64)[1:3].copy()

        for local in range(comparison.POST_LAUNCH_STEPS):
            profile_value = float(FIXED_ACTION_PROFILE[local]) if local < len(FIXED_ACTION_PROFILE) else 0.0
            action = np.zeros(7, dtype=np.float64)
            action[0] = float(amplitude) * profile_value
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

            # Keep the controller in its push gain regime. After the fixed
            # chunk, zero x action holds the board while the box slides away.
            env.step_count = push_start + min(local, len(FIXED_ACTION_PROFILE) - 1)
            obs, _, _, info = env.step(action)
            row = dict(info["push_box"])
            row["local_frame"] = int(local)
            rows.append(row)

            contacts = comparison.board_box_contacts(env)
            has_contact = bool(contacts)
            if has_contact:
                sampled_contact_frames.append(int(local))
                if not contact_active:
                    contact_episode_count += 1
            contact_active = has_contact

            box_xyz = np.asarray(row["box_xyz"], dtype=np.float64)
            vx = float(row["box_vxy"][0])
            if motion_frame is None and (
                float(box_xyz[0] - launch_box_xyz[0]) > comparison.board_probe.source.CONTACT_MOVE_M
                or abs(vx) > comparison.board_probe.source.CONTACT_SPEED_MPS
            ):
                motion_frame = int(local)
            if local == len(FIXED_ACTION_PROFILE) - 1:
                release_box_vxy = np.asarray(row["box_vxy"], dtype=np.float64).copy()

            phase = "fixed_push" if local < len(FIXED_ACTION_PROFILE) - 1 else "release_hold"
            comparison.append_frame(
                frames,
                env,
                obs,
                mode="touch",
                amplitude=amplitude,
                phase=phase,
                initial_box_xyz=initial_box_xyz,
            )
    finally:
        env.close()

    final_box_xyz = np.asarray(rows[-1]["box_xyz"], dtype=np.float64)
    velocity = np.asarray([row["box_vxy"] for row in rows], dtype=np.float64)
    if release_box_vxy is None:
        raise RuntimeError("Fixed action profile ended without a release velocity sample")
    return {
        "mode": "touch_fixed_release",
        "action_id": int(action_id),
        "A": float(amplitude),
        "friction_mu": comparison.board_probe.FRICTION_MU,
        "launch_geometry": launch_geometry,
        "launch_box_speed_mps": launch_box_speed,
        "prelaunch_box_drift_m": float(np.linalg.norm(launch_box_xyz[:2] - initial_box_xyz[:2])),
        "final_displacement_from_launch_m": float(np.linalg.norm(final_box_xyz[:2] - launch_box_xyz[:2])),
        "final_forward_from_launch_m": float(final_box_xyz[0] - launch_box_xyz[0]),
        "final_lateral_from_launch_m": float(final_box_xyz[1] - launch_box_xyz[1]),
        "peak_box_vx_mps": float(np.max(velocity[:, 0])),
        "release_box_vxy_mps": release_box_vxy.astype(float).tolist(),
        "contact_frame": motion_frame,
        "sampled_contact_frames": sampled_contact_frames,
        "sampled_contact_episode_count": int(contact_episode_count),
        "stop_reason": "fixed_10_step_release",
        "touch_preparation": touch,
        "frames": frames,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: float(row["A"]))
    amplitudes = np.asarray([float(row["A"]) for row in ordered], dtype=np.float64)
    distance_cm = np.asarray(
        [float(row["final_displacement_from_launch_m"]) * 100.0 for row in ordered], dtype=np.float64
    )
    peak_vx = np.asarray([float(row["peak_box_vx_mps"]) for row in ordered], dtype=np.float64)
    lateral_cm = np.asarray(
        [abs(float(row["final_lateral_from_launch_m"])) * 100.0 for row in ordered], dtype=np.float64
    )
    distance_delta = np.diff(distance_cm)
    peak_delta = np.diff(peak_vx)
    moving = np.flatnonzero(distance_cm >= 1.0)
    under_60 = np.flatnonzero(distance_cm <= 60.0)
    clean = np.flatnonzero((distance_cm <= 60.0) & (lateral_cm <= 1.0))
    return {
        "movement_threshold_A": None if not len(moving) else float(amplitudes[moving[0]]),
        "distance_range_cm": [float(np.min(distance_cm)), float(np.max(distance_cm))],
        "peak_vx_range_mps": [float(np.min(peak_vx)), float(np.max(peak_vx))],
        "max_abs_lateral_cm": float(np.max(lateral_cm)),
        "distance_adjacent_deltas_cm": distance_delta.astype(float).tolist(),
        "distance_decrease_count_over_1cm": int(np.sum(distance_delta < -1.0)),
        "largest_adjacent_distance_drop_cm": float(min(0.0, np.min(distance_delta))),
        "peak_vx_adjacent_deltas_mps": peak_delta.astype(float).tolist(),
        "peak_vx_decrease_count_over_0.02mps": int(np.sum(peak_delta < -0.02)),
        "A_values_under_60cm": amplitudes[under_60].astype(float).tolist(),
        "A_values_under_60cm_and_1cm_lateral": amplitudes[clean].astype(float).tolist(),
    }


def plot_results(path: Path, rows: list[dict[str, Any]]) -> None:
    ordered = sorted(rows, key=lambda row: float(row["A"]))
    amplitudes = np.asarray([float(row["A"]) for row in ordered], dtype=np.float64)
    distance_cm = np.asarray([float(row["final_displacement_from_launch_m"]) * 100.0 for row in ordered])
    peak_vx = np.asarray([float(row["peak_box_vx_mps"]) for row in ordered])
    lateral_cm = np.asarray([abs(float(row["final_lateral_from_launch_m"])) * 100.0 for row in ordered])
    contact_frame = np.asarray(
        [-1 if row["contact_frame"] is None else int(row["contact_frame"]) for row in ordered]
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), constrained_layout=True)
    color = "#137b70"
    axes[0, 0].plot(amplitudes, distance_cm, "o-", color=color, markersize=4)
    axes[0, 0].axhline(60.0, color="#bc4b31", linestyle="--", linewidth=1.2, label="60 cm limit")
    axes[0, 0].set_ylabel("Final displacement (cm)")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].plot(amplitudes, peak_vx, "o-", color="#c76828", markersize=4)
    axes[0, 1].set_ylabel("Peak box vx (m/s)")
    axes[1, 0].plot(amplitudes, lateral_cm, "o-", color="#37699b", markersize=4)
    axes[1, 0].axhline(1.0, color="#bc4b31", linestyle="--", linewidth=1.2, label="1 cm lateral")
    axes[1, 0].set_ylabel("Absolute lateral drift (cm)")
    axes[1, 0].legend(frameon=False)
    axes[1, 1].plot(amplitudes, contact_frame, "o-", color="#75634f", markersize=4)
    axes[1, 1].set_ylabel("Motion-trigger frame after launch")
    for axis in axes.flat:
        axis.set_xlabel("Action amplitude A")
        axis.grid(alpha=0.25)
    fig.suptitle("Strict touch-start rigid-board sweep at mu=0.15")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep a wide action-amplitude range from strict board contact.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config = comparison.board_probe.source_dataset.configure_dataset()
    prepare = dict(config["prepare_config"])
    prepare["descend_steps"] = 45
    prepare["prepare_position_gain"] = 8.0
    config["prepare_config"] = prepare
    manifest = json.loads(comparison.board_probe.SOURCE_MANIFEST.read_text(encoding="utf-8"))
    bddl_file = next(
        row["bddl_file"]
        for row in manifest["episodes"]
        if abs(float(row["mu"]) - comparison.board_probe.FRICTION_MU) < 1e-12
    )

    rows = []
    for action_id, amplitude in enumerate(AMPLITUDES):
        result = rollout_fixed_release(
            config,
            amplitude=float(amplitude),
            action_id=action_id,
            bddl_file=bddl_file,
            seed=int(args.seed),
        )
        frames = result.pop("frames")
        video = output_root / f"touch_wide_a{action_id:02d}_A{int(round(amplitude * 1000)):03d}.mp4"
        comparison.board_probe.write_video(video, frames)
        result["video"] = str(video)
        rows.append(result)
        print(
            f"touch-wide {action_id + 1:02d}/{len(AMPLITUDES):02d} A={amplitude:.3f} "
            f"distance={result['final_displacement_from_launch_m'] * 100.0:.2f}cm "
            f"peak_vx={result['peak_box_vx_mps']:.3f}m/s "
            f"lateral={result['final_lateral_from_launch_m'] * 100.0:+.2f}cm "
            f"contact={result['touch_preparation']['sampled_contact']}",
            flush=True,
        )

    analysis = summarize(rows)
    payload = {
        "experiment": "strict touch-start rigid-board wide action-amplitude sweep",
        "friction_mu": comparison.board_probe.FRICTION_MU,
        "amplitudes": list(AMPLITUDES),
        "fixed_action_profile": FIXED_ACTION_PROFILE.astype(float).tolist(),
        "release_logic": "fixed 10-step chunk followed by zero x action; no event trigger",
        "analysis": analysis,
        "results": rows,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(comparison.board_probe.source.base.to_jsonable(payload), indent=2), encoding="utf-8"
    )
    plot_path = output_root / "touch_wide_A_sweep.png"
    plot_results(plot_path, rows)
    print(f"summary={summary_path}", flush=True)
    print(f"plot={plot_path}", flush=True)


if __name__ == "__main__":
    main()
