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
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
BOARD_PROBE_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "render_libero_push_box_gripped_board_alignment_probe_2026-07-17_hai-machine.py"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "outputs"
    / "pushbox"
    / "board_touch"
    / "libero_push_box_gripped_board_gap_vs_touch_mu015_2026-07-17_hai-machine"
)

MODES = ("gap", "touch")
TOUCH_TARGET_GAP_M = 0.0003
TOUCH_MAX_STEPS = 200
POST_LAUNCH_STEPS = 80


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


board_probe = load_module(BOARD_PROBE_SCRIPT, "gap_vs_touch_board_probe_hai_machine")


def box_collision_center(env: Any) -> np.ndarray:
    sim = env.inner_env.sim
    return np.asarray(sim.data.geom_xpos[sim.model.geom_name2id("cream_cheese_1_g1")], dtype=np.float64)


def geometry_state(env: Any) -> dict[str, Any]:
    sim = env.inner_env.sim
    board_id = sim.model.geom_name2id("gripper0_board_tool_collision")
    center = np.asarray(sim.data.geom_xpos[board_id], dtype=np.float64)
    rotation = np.asarray(sim.data.geom_xmat[board_id], dtype=np.float64).reshape(3, 3)
    extent = np.abs(rotation) @ np.asarray(board_probe.BOARD_HALF_SIZE, dtype=np.float64)
    box_xyz, _ = env.box_pose()
    box_id = sim.model.geom_name2id("cream_cheese_1_g1")
    box_center = np.asarray(sim.data.geom_xpos[box_id], dtype=np.float64)
    box_rotation = np.asarray(sim.data.geom_xmat[box_id], dtype=np.float64).reshape(3, 3)
    box_extent = np.abs(box_rotation) @ np.asarray(sim.model.geom_size[box_id], dtype=np.float64)
    board_front_x = float(center[0] + extent[0])
    box_back_x = float(box_center[0] - box_extent[0])
    normal = rotation[:, 1].copy()
    if normal[0] < 0.0:
        normal *= -1.0
    normal /= np.linalg.norm(normal)
    return {
        "board_center": center,
        "box_body_xyz": np.asarray(box_xyz, dtype=np.float64),
        "box_collision_center": box_center,
        "gap_m": float(box_back_x - board_front_x),
        "normal_error_deg": float(np.degrees(np.arccos(np.clip(normal[0], -1.0, 1.0)))),
        "center_y_error_m": float(center[1] - box_center[1]),
        "center_z_error_m": float(center[2] - box_center[2]),
        "board_bottom_m": float(center[2] - extent[2]),
    }


def board_box_contacts(env: Any) -> list[dict[str, Any]]:
    sim = env.inner_env.sim
    contacts = []
    for index in range(int(sim.data.ncon)):
        contact = sim.data.contact[index]
        name1 = sim.model.geom_id2name(int(contact.geom1)) or ""
        name2 = sim.model.geom_id2name(int(contact.geom2)) or ""
        names = {name1, name2}
        if "gripper0_board_tool_collision" not in names:
            continue
        if not any("cream_cheese" in name for name in names):
            continue
        contacts.append(
            {
                "geoms": [name1, name2],
                "point": np.asarray(contact.pos, dtype=np.float64).astype(float).tolist(),
                "distance_m": float(contact.dist),
            }
        )
    return contacts


def label_frame(
    frame: np.ndarray,
    *,
    mode: str,
    amplitude: float,
    phase: str,
    gap_m: float,
    displacement_m: float,
) -> np.ndarray:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    draw = ImageDraw.Draw(image)
    draw.rectangle((2, 2, 222, 50), fill=(0, 0, 0))
    draw.text((6, 5), f"{mode}  mu=0.150  A={amplitude:.3f}", fill=(255, 255, 255))
    draw.text((6, 20), f"{phase}  gap={gap_m * 100.0:+.2f}cm", fill=(255, 205, 145))
    draw.text((6, 35), f"box displacement={displacement_m * 100.0:.1f}cm", fill=(185, 225, 255))
    return np.asarray(image, dtype=np.uint8)


def append_frame(
    frames: list[np.ndarray],
    env: Any,
    obs: dict[str, Any],
    *,
    mode: str,
    amplitude: float,
    phase: str,
    initial_box_xyz: np.ndarray,
) -> None:
    geometry = geometry_state(env)
    agent, _ = board_probe.source.base._obs_to_images(obs)
    displacement = float(np.linalg.norm(geometry["box_body_xyz"][:2] - initial_box_xyz[:2]))
    frames.append(
        label_frame(
            agent,
            mode=mode,
            amplitude=amplitude,
            phase=phase,
            gap_m=float(geometry["gap_m"]),
            displacement_m=displacement,
        )
    )


def prepare_touch(
    env: Any,
    *,
    case: Any,
    frames: list[np.ndarray],
    amplitude: float,
    initial_box_xyz: np.ndarray,
) -> dict[str, Any]:
    push_start = int(case.pusher_approach_steps) + int(case.pusher_descend_steps)
    sampled_contact = False
    steps = 0
    for steps in range(1, TOUCH_MAX_STEPS + 1):
        geometry = geometry_state(env)
        gap = float(geometry["gap_m"])
        contacts = board_box_contacts(env)
        if contacts:
            sampled_contact = True
            break

        if gap > 0.015:
            action_x = 0.10
        elif gap > 0.004:
            action_x = 0.045
        elif gap > TOUCH_TARGET_GAP_M:
            action_x = 0.012
        else:
            action_x = 0.008
        action = np.zeros(7, dtype=np.float64)
        action[0] = action_x
        action[1] = float(np.clip(-5.0 * float(geometry["center_y_error_m"]), -0.06, 0.06))
        # Keep the calibrated paddle height. Centering on the thin box would
        # lower the paddle into the tabletop.
        action[2] = 0.0
        action[-1] = float(case.pusher_gripper)

        # Keep the low-gain prepare controller active during the gentle closing
        # motion, then restore the real push clock at launch.
        env.step_count = push_start - 1
        obs, _, _, _ = env.step(action)
        append_frame(
            frames,
            env,
            obs,
            mode="touch",
            amplitude=amplitude,
            phase="gentle_touch",
            initial_box_xyz=initial_box_xyz,
        )
        if board_box_contacts(env):
            sampled_contact = True
            break
    if not sampled_contact:
        final_gap = float(geometry_state(env)["gap_m"])
        raise RuntimeError(
            f"Touch preparation did not reach a real board-box contact in {TOUCH_MAX_STEPS} steps; "
            f"final gap={final_gap * 1000.0:.3f} mm"
        )
    return {
        "steps": int(steps),
        "sampled_contact": bool(sampled_contact),
        "state": geometry_state(env),
        "box_speed_mps": float(np.linalg.norm(env.box_velocity()[:2])),
    }


def make_env(case: Any, *, seed: int) -> Any:
    original_gripper = board_probe.install_board_gripper()
    try:
        env = board_probe.source.base.LiberoPushBoxEnv(case, repo_root=REPO_ROOT, seed=seed)
        env.reset()
        board_probe.configure_board_collision(env)
    finally:
        board_probe.restore_board_gripper(original_gripper)
    return env


def rollout(
    config: dict[str, Any],
    *,
    mode: str,
    amplitude: float,
    action_id: int,
    bddl_file: str,
    seed: int,
) -> dict[str, Any]:
    case = board_probe.make_case(config, amplitude=amplitude, bddl_file=bddl_file, action_id=action_id)
    env = make_env(case, seed=seed)
    frames: list[np.ndarray] = []
    launch_rows = []
    sampled_contact_frames = []
    contact_episode_count = 0
    contact_active = False
    forced_stop = False
    stop_reason = None
    contact_frame = None
    try:
        board_probe.source.base.preposition_fixed_start(env)
        env.step_count = 0
        env._last_scripted_action = np.zeros(7, dtype=np.float64)
        env._last_scripted_phase = None
        initial_box_xyz, _ = env.box_pose()

        push_start = int(case.pusher_approach_steps) + int(case.pusher_descend_steps)
        for _ in range(push_start):
            obs, _, _, info = env.step()
            append_frame(
                frames,
                env,
                obs,
                mode=mode,
                amplitude=amplitude,
                phase=str(info["push_box"]["phase"]),
                initial_box_xyz=initial_box_xyz,
            )

        touch = None
        if mode == "touch":
            touch = prepare_touch(
                env,
                case=case,
                frames=frames,
                amplitude=amplitude,
                initial_box_xyz=initial_box_xyz,
            )

        launch_box_xyz, _ = env.box_pose()
        launch_geometry = geometry_state(env)
        launch_box_speed = float(np.linalg.norm(env.box_velocity()[:2]))
        env.step_count = push_start
        env._last_scripted_action = np.zeros(7, dtype=np.float64)
        env._last_scripted_phase = "descend"

        for local in range(POST_LAUNCH_STEPS):
            if forced_stop:
                obs, _, _, info = env.step(board_probe.source.brake_action(case))
                phase = "event_stop"
            else:
                obs, _, _, info = env.step()
                phase = str(info["push_box"]["phase"])
            row = dict(info["push_box"])
            row["local_frame"] = int(local)
            launch_rows.append(row)
            contacts = board_box_contacts(env)
            has_contact = bool(contacts)
            if has_contact:
                sampled_contact_frames.append(int(local))
                if not contact_active:
                    contact_episode_count += 1
            contact_active = has_contact

            box_xyz = np.asarray(row["box_xyz"], dtype=np.float64)
            vx = float(row["box_vxy"][0])
            if contact_frame is None and (
                float(box_xyz[0] - launch_box_xyz[0]) > board_probe.source.CONTACT_MOVE_M
                or abs(vx) > board_probe.source.CONTACT_SPEED_MPS
            ):
                contact_frame = int(local)
            if contact_frame is not None and not forced_stop:
                frames_after_contact = int(local) - contact_frame
                target_vx = max(
                    board_probe.source.CONTACT_SPEED_MPS,
                    board_probe.source.TRIGGER_VX_RATIO * float(case.pusher_push_action_end),
                )
                vx_ready = abs(vx) >= target_vx
                timeout_ready = frames_after_contact >= board_probe.source.MAX_CONTACT_HOLD
                if frames_after_contact >= board_probe.source.HOLD_AFTER_CONTACT and (vx_ready or timeout_ready):
                    forced_stop = True
                    stop_reason = "vx_ready" if vx_ready else "timeout"

            append_frame(
                frames,
                env,
                obs,
                mode=mode,
                amplitude=amplitude,
                phase=phase,
                initial_box_xyz=initial_box_xyz,
            )
    finally:
        env.close()

    final_box_xyz = np.asarray(launch_rows[-1]["box_xyz"], dtype=np.float64)
    velocity = np.asarray([row["box_vxy"] for row in launch_rows], dtype=np.float64)
    serial_launch_geometry = {
        key: value.astype(float).tolist() if isinstance(value, np.ndarray) else value
        for key, value in launch_geometry.items()
    }
    serial_touch = None
    if touch is not None:
        serial_touch = dict(touch)
        serial_touch["state"] = {
            key: value.astype(float).tolist() if isinstance(value, np.ndarray) else value
            for key, value in touch["state"].items()
        }
    return {
        "mode": mode,
        "action_id": int(action_id),
        "A": float(amplitude),
        "friction_mu": board_probe.FRICTION_MU,
        "launch_geometry": serial_launch_geometry,
        "launch_box_speed_mps": launch_box_speed,
        "prelaunch_box_drift_m": float(np.linalg.norm(launch_box_xyz[:2] - initial_box_xyz[:2])),
        "final_displacement_from_launch_m": float(np.linalg.norm(final_box_xyz[:2] - launch_box_xyz[:2])),
        "final_forward_from_launch_m": float(final_box_xyz[0] - launch_box_xyz[0]),
        "final_lateral_from_launch_m": float(final_box_xyz[1] - launch_box_xyz[1]),
        "peak_box_vx_mps": float(np.max(velocity[:, 0])),
        "contact_frame": contact_frame,
        "sampled_contact_frames": sampled_contact_frames,
        "sampled_contact_episode_count": int(contact_episode_count),
        "stop_reason": stop_reason,
        "touch_preparation": serial_touch,
        "frames": frames,
    }


def monotonic_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: float(row["A"]))
    distance_cm = np.asarray([float(row["final_displacement_from_launch_m"]) * 100.0 for row in ordered])
    peak_vx = np.asarray([float(row["peak_box_vx_mps"]) for row in ordered])
    return {
        "distance_adjacent_deltas_cm": np.diff(distance_cm).astype(float).tolist(),
        "distance_decrease_count": int(np.sum(np.diff(distance_cm) < -0.5)),
        "peak_vx_adjacent_deltas_mps": np.diff(peak_vx).astype(float).tolist(),
        "peak_vx_decrease_count": int(np.sum(np.diff(peak_vx) < -0.02)),
        "max_abs_lateral_cm": float(
            max(abs(float(row["final_lateral_from_launch_m"])) for row in ordered) * 100.0
        ),
        "max_prelaunch_drift_mm": float(max(float(row["prelaunch_box_drift_m"]) for row in ordered) * 1000.0),
    }


def plot_summary(path: Path, results: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    styles = {"gap": ("#d04a2b", "o"), "touch": ("#147d78", "s")}
    for mode in MODES:
        rows = sorted((row for row in results if row["mode"] == mode), key=lambda row: float(row["A"]))
        a = [float(row["A"]) for row in rows]
        color, marker = styles[mode]
        axes[0].plot(a, [float(row["final_displacement_from_launch_m"]) * 100.0 for row in rows], marker=marker, color=color, label=mode)
        axes[1].plot(a, [float(row["peak_box_vx_mps"]) for row in rows], marker=marker, color=color, label=mode)
        axes[2].plot(a, [abs(float(row["final_lateral_from_launch_m"])) * 100.0 for row in rows], marker=marker, color=color, label=mode)
    axes[0].set_ylabel("Final displacement (cm)")
    axes[1].set_ylabel("Peak box vx (m/s)")
    axes[2].set_ylabel("Absolute lateral drift (cm)")
    for axis in axes:
        axis.set_xlabel("Peak action A")
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Rigid board launcher: gap impact vs gentle pre-contact (mu=0.15)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare board impact against gentle pre-contact launch.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config = board_probe.source_dataset.configure_dataset()
    prepare = dict(config["prepare_config"])
    prepare["descend_steps"] = 45
    prepare["prepare_position_gain"] = 8.0
    config["prepare_config"] = prepare
    manifest = json.loads(board_probe.SOURCE_MANIFEST.read_text(encoding="utf-8"))
    bddl_file = next(
        row["bddl_file"]
        for row in manifest["episodes"]
        if abs(float(row["mu"]) - board_probe.FRICTION_MU) < 1e-12
    )

    results = []
    for mode in MODES:
        for action_id, amplitude in enumerate(board_probe.AMPLITUDES):
            result = rollout(
                config,
                mode=mode,
                amplitude=float(amplitude),
                action_id=action_id,
                bddl_file=bddl_file,
                seed=int(args.seed),
            )
            frames = result.pop("frames")
            video = output_root / f"board_{mode}_mu1500_a{action_id:02d}_A{int(round(amplitude * 1000)):03d}.mp4"
            board_probe.write_video(video, frames)
            result["video"] = str(video)
            results.append(result)
            launch = result["launch_geometry"]
            print(
                f"{mode:5s} {action_id + 1:02d}/{len(board_probe.AMPLITUDES):02d} "
                f"A={amplitude:.3f} distance={result['final_displacement_from_launch_m'] * 100.0:.1f}cm "
                f"peak_vx={result['peak_box_vx_mps']:.3f} "
                f"lateral={result['final_lateral_from_launch_m'] * 100.0:+.2f}cm "
                f"launch_gap={float(launch['gap_m']) * 100.0:+.2f}cm "
                f"pre_drift={result['prelaunch_box_drift_m'] * 1000.0:.2f}mm",
                flush=True,
            )

    modes = {mode: monotonic_summary([row for row in results if row["mode"] == mode]) for mode in MODES}
    payload = {
        "experiment": "rigid board gap impact versus gentle pre-contact launch",
        "friction_mu": board_probe.FRICTION_MU,
        "amplitudes": list(board_probe.AMPLITUDES),
        "fixed_action_profile": [0.5] + [1.0] * 13 + [0.5, 0.0],
        "touch_target_gap_m": TOUCH_TARGET_GAP_M,
        "mode_summary": modes,
        "results": results,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(board_probe.source.base.to_jsonable(payload), indent=2), encoding="utf-8")
    plot_path = output_root / "gap_vs_touch_metrics.png"
    plot_summary(plot_path, results)
    print(f"summary={summary_path}", flush=True)
    print(f"plot={plot_path}", flush=True)


if __name__ == "__main__":
    main()
