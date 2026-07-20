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

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
MASS_DEMO_SCRIPT = REPO_ROOT / "scripts" / "render_libero_push_box_mass_sweep_mu0080_mid40cm_2026-07-16_hai-machine.py"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "pushbox" / "demos" / "libero_push_box_mass_oracle_force_and_lowkp_mu0040_2026-07-16_hai-machine"
FRICTION_MU = 0.04
FPS = 20
CAMERA_RESOLUTION = 224
VIDEO_CRF = 18
MASS_TARGETS_KG = (0.25, 0.35, 0.45, 0.60, 0.75)

ORACLE_FORCE_N = 0.60
ORACLE_PRE_STEPS = 20
ORACLE_FORCE_STEPS = 5
ORACLE_SETTLE_STEPS = 150

LOW_TRANSLATION_KP = 10.0
REFERENCE_MASS_KG = 0.45
TARGET_DISPLACEMENT_M = 0.40
LOW_KP_CALIBRATION_A = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.65, 0.80)


def load_mass_demo() -> Any:
    spec = importlib.util.spec_from_file_location("mass_demo_base_for_oracle_lowkp_hai_machine", MASS_DEMO_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import mass demo script: {MASS_DEMO_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


demo = load_mass_demo()
demo.FRICTION_MU = FRICTION_MU


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare direct-force and low-OSC-stiffness mass sensitivity in LIBERO.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(value), indent=2), encoding="utf-8")


def make_case(*, amplitude: float, bddl_file: str) -> Any:
    target_xy = (
        float(demo.base.INIT_XY[0] + demo.base.DUMMY_TARGET_DISTANCE),
        float(demo.base.INIT_XY[1]),
    )
    return demo.base.build_push_case(
        {"mu": FRICTION_MU, "mode": "event_hold", "A": float(amplitude), "hold": 3},
        case_id=f"mass_oracle_lowkp_mu0040_A{int(round(amplitude * 1000)):03d}",
        bddl_file=bddl_file,
        target_xy=target_xy,
        camera_resolution=CAMERA_RESOLUTION,
    )


def frame_with_lines(obs: dict[str, Any], lines: list[str]) -> np.ndarray:
    agent, _ = demo.base._obs_to_images(obs)
    image = Image.fromarray(np.asarray(agent, dtype=np.uint8))
    draw = ImageDraw.Draw(image)
    height = 5 + 14 * len(lines)
    draw.rectangle((3, 3, 221, height), fill=(0, 0, 0))
    for index, line in enumerate(lines):
        draw.text((7, 6 + 14 * index), line, fill=(255, 255, 255) if index == 0 else (215, 232, 250))
    return np.asarray(image, dtype=np.uint8)


def write_video(path: Path, frames: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        path,
        fps=FPS,
        codec="libx264",
        macro_block_size=None,
        ffmpeg_params=["-crf", str(VIDEO_CRF), "-pix_fmt", "yuv420p"],
    ) as writer:
        for frame in frames:
            writer.append_data(frame)


def write_comparison(path: Path, results: list[dict[str, Any]]) -> None:
    blank = np.zeros((CAMERA_RESOLUTION, CAMERA_RESOLUTION, 3), dtype=np.uint8)
    max_frames = max(len(result["frames"]) for result in results)
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        path,
        fps=FPS,
        codec="libx264",
        macro_block_size=None,
        ffmpeg_params=["-crf", str(VIDEO_CRF), "-pix_fmt", "yuv420p"],
    ) as writer:
        for frame_index in range(max_frames):
            tiles = [
                result["frames"][min(frame_index, len(result["frames"]) - 1)]
                for result in results
            ]
            tiles.append(blank)
            writer.append_data(
                np.concatenate(
                    [np.concatenate(tiles[:3], axis=1), np.concatenate(tiles[3:6], axis=1)],
                    axis=0,
                )
            )


def without_frames(result: dict[str, Any]) -> dict[str, Any]:
    payload = dict(result)
    payload.pop("frames", None)
    return payload


def direct_force_rollout(*, mass_kg: float, bddl_file: str, seed: int) -> dict[str, Any]:
    case = make_case(amplitude=0.0, bddl_file=bddl_file)
    env = demo.base.LiberoPushBoxEnv(case, repo_root=REPO_ROOT, seed=int(seed))
    frames: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    hold_action = np.zeros(7, dtype=np.float64)
    hold_action[-1] = 1.0
    try:
        obs = env.reset()
        mass_info = demo.set_box_mass_kg(env, float(mass_kg))
        body_ids = list(mass_info["body_ids"])
        initial_xyz, _ = env.box_pose()
        total_steps = ORACLE_PRE_STEPS + ORACLE_FORCE_STEPS + ORACLE_SETTLE_STEPS
        for frame_index in range(total_steps):
            pulse = ORACLE_PRE_STEPS <= frame_index < ORACLE_PRE_STEPS + ORACLE_FORCE_STEPS
            for body_id in body_ids:
                env.inner_env.sim.data.xfrc_applied[body_id, :] = 0.0
                if pulse:
                    env.inner_env.sim.data.xfrc_applied[body_id, 0] = ORACLE_FORCE_N
            obs, _, _, _ = env.step(hold_action)
            box_xyz, _ = env.box_pose()
            box_qvel = env.box_velocity()
            displacement = float(np.linalg.norm(box_xyz[:2] - initial_xyz[:2]))
            phase = "direct_force" if pulse else ("pre_hold" if frame_index < ORACLE_PRE_STEPS else "free_slide")
            frames.append(
                frame_with_lines(
                    obs,
                    [
                        f"oracle | mu={FRICTION_MU:.2f} mass={mass_kg * 1000.0:.0f}g",
                        f"Fx={ORACLE_FORCE_N:.2f}N for {ORACLE_FORCE_STEPS / FPS:.2f}s | {phase}",
                        f"box displacement={displacement * 100.0:.1f}cm",
                    ],
                )
            )
            rows.append(
                {
                    "frame_index": frame_index,
                    "phase": phase,
                    "force_x_N": ORACLE_FORCE_N if pulse else 0.0,
                    "box_xyz_m": np.asarray(box_xyz, dtype=np.float64),
                    "box_qvel": np.asarray(box_qvel, dtype=np.float64),
                }
            )
    finally:
        try:
            for body_id in body_ids:
                env.inner_env.sim.data.xfrc_applied[body_id, :] = 0.0
        except Exception:
            pass
        env.close()

    final_xyz = np.asarray(rows[-1]["box_xyz_m"], dtype=np.float64)
    velocities = np.asarray([row["box_qvel"][:2] for row in rows], dtype=np.float64)
    return {
        "mode": "direct_body_force_oracle",
        "friction_mu": FRICTION_MU,
        "mass_kg": float(mass_kg),
        "force_x_N": ORACLE_FORCE_N,
        "force_duration_s": ORACLE_FORCE_STEPS / FPS,
        "commanded_impulse_Ns": ORACLE_FORCE_N * ORACLE_FORCE_STEPS / FPS,
        "initial_box_xyz_m": np.asarray(initial_xyz, dtype=np.float64),
        "final_box_xyz_m": final_xyz,
        "final_displacement_m": float(np.linalg.norm(final_xyz[:2] - initial_xyz[:2])),
        "max_box_planar_speed_mps": float(np.max(np.linalg.norm(velocities, axis=1))),
        "frames": frames,
    }


def set_translation_stiffness(controller: Any, kp_xyz: np.ndarray, kd_xyz: np.ndarray) -> None:
    controller.kp[:3] = np.asarray(kp_xyz, dtype=np.float64)
    controller.kd[:3] = np.asarray(kd_xyz, dtype=np.float64)


def low_kp_rollout(
    *,
    mass_kg: float,
    amplitude: float,
    bddl_file: str,
    seed: int,
    capture_frames: bool,
) -> dict[str, Any]:
    case = make_case(amplitude=amplitude, bddl_file=bddl_file)
    env = demo.base.LiberoPushBoxEnv(case, repo_root=REPO_ROOT, seed=int(seed))
    frames: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    try:
        _, initial_xyz, mass_info = demo.prepare_rollout(env, target_mass_kg=float(mass_kg))
        controller = env._controller()
        if controller is None or not hasattr(controller, "kp") or not hasattr(controller, "kd"):
            raise RuntimeError("OSC controller does not expose kp/kd.")
        default_kp_xyz = np.asarray(controller.kp[:3], dtype=np.float64).copy()
        default_kd_xyz = np.asarray(controller.kd[:3], dtype=np.float64).copy()
        low_kp_xyz = np.full(3, LOW_TRANSLATION_KP, dtype=np.float64)
        low_kd_xyz = 2.0 * np.sqrt(low_kp_xyz)
        push_start = int(case.pusher_approach_steps) + int(case.pusher_descend_steps)
        push_end = push_start + int(case.pusher_push_steps)
        obs = env._last_obs
        for frame_index in range(int(case.max_steps)):
            if push_start <= frame_index < push_end:
                set_translation_stiffness(controller, low_kp_xyz, low_kd_xyz)
            else:
                set_translation_stiffness(controller, default_kp_xyz, default_kd_xyz)
            obs, _, _, info = env.step()
            row = dict(info["push_box"])
            rows.append(row)
            if capture_frames:
                box_xyz = np.asarray(row["box_xyz"], dtype=np.float64)
                displacement = float(np.linalg.norm(box_xyz[:2] - initial_xyz[:2]))
                active_kp = LOW_TRANSLATION_KP if push_start <= frame_index < push_end else float(default_kp_xyz[0])
                frames.append(
                    frame_with_lines(
                        obs,
                        [
                            f"LIBERO OSC | mu={FRICTION_MU:.2f} mass={mass_kg * 1000.0:.0f}g",
                            f"push kp={active_kp:.0f} fixed A={amplitude:.2f} | {row['phase']}",
                            f"box displacement={displacement * 100.0:.1f}cm",
                        ],
                    )
                )
    finally:
        env.close()

    final_xyz = np.asarray(rows[-1]["box_xyz"], dtype=np.float64)
    velocities = np.asarray([row["box_vxy"] for row in rows], dtype=np.float64)
    push_rows = [row for row in rows if row["phase"] == "push"]
    eef_x = np.asarray([row["eef_xyz"][0] for row in push_rows], dtype=np.float64)
    return {
        "mode": "libero_osc_low_translation_stiffness",
        "friction_mu": FRICTION_MU,
        "mass": mass_info,
        "amplitude": float(amplitude),
        "translation_kp_push": LOW_TRANSLATION_KP,
        "translation_kd_push": float(2.0 * np.sqrt(LOW_TRANSLATION_KP)),
        "default_translation_kp": default_kp_xyz,
        "initial_box_xyz_m": initial_xyz,
        "final_box_xyz_m": final_xyz,
        "final_displacement_m": float(np.linalg.norm(final_xyz[:2] - initial_xyz[:2])),
        "max_box_planar_speed_mps": float(np.max(np.linalg.norm(velocities, axis=1))),
        "push_eef_forward_m": float(eef_x[-1] - eef_x[0]) if len(eef_x) > 1 else 0.0,
        "push_eef_backward_steps": int(np.sum(np.diff(eef_x) < -1e-4)) if len(eef_x) > 1 else 0,
        "frames": frames,
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists; pass --overwrite: {output}")
        shutil.rmtree(output)
    (output / "videos" / "oracle").mkdir(parents=True, exist_ok=True)
    (output / "videos" / "low_kp").mkdir(parents=True, exist_ok=True)
    bddl_file = demo.base.write_geometry_bddl(
        repo_root=REPO_ROOT,
        bddl_dir=output / "bddl",
        geometry_id="mass_oracle_lowkp_mu0040_hidden",
        init_xy=demo.base.INIT_XY,
        target_xy=(
            float(demo.base.INIT_XY[0] + demo.base.DUMMY_TARGET_DISTANCE),
            float(demo.base.INIT_XY[1]),
        ),
        init_half_size=0.002,
        target_radius=demo.base.TARGET_RADIUS,
        target_rgba=(0.0, 0.8, 0.2, 0.0),
    )

    oracle_results = []
    for index, mass in enumerate(MASS_TARGETS_KG):
        result = direct_force_rollout(mass_kg=mass, bddl_file=bddl_file, seed=int(args.seed))
        path = output / "videos" / "oracle" / f"{index + 1:02d}_mass_{int(round(mass * 1000)):04d}g.mp4"
        write_video(path, result["frames"])
        result["video"] = str(path)
        oracle_results.append(result)
        print(
            f"[oracle {index + 1}/5] mass={mass:.2f}kg -> "
            f"{result['final_displacement_m'] * 100.0:.1f}cm, "
            f"peak_v={result['max_box_planar_speed_mps']:.3f}m/s",
            flush=True,
        )
    oracle_comparison = output / "videos" / "00_oracle_direct_force_mass_comparison.mp4"
    write_comparison(oracle_comparison, oracle_results)

    calibration = []
    for index, amplitude in enumerate(LOW_KP_CALIBRATION_A):
        result = low_kp_rollout(
            mass_kg=REFERENCE_MASS_KG,
            amplitude=float(amplitude),
            bddl_file=bddl_file,
            seed=int(args.seed),
            capture_frames=False,
        )
        calibration.append(without_frames(result))
        print(
            f"[low-kp calibration {index + 1:02d}/{len(LOW_KP_CALIBRATION_A):02d}] "
            f"A={amplitude:.2f} -> {result['final_displacement_m'] * 100.0:.1f}cm",
            flush=True,
        )
    chosen = min(calibration, key=lambda item: abs(float(item["final_displacement_m"]) - TARGET_DISPLACEMENT_M))
    chosen_amplitude = float(chosen["amplitude"])

    low_kp_results = []
    for index, mass in enumerate(MASS_TARGETS_KG):
        result = low_kp_rollout(
            mass_kg=mass,
            amplitude=chosen_amplitude,
            bddl_file=bddl_file,
            seed=int(args.seed),
            capture_frames=True,
        )
        path = output / "videos" / "low_kp" / f"{index + 1:02d}_mass_{int(round(mass * 1000)):04d}g.mp4"
        write_video(path, result["frames"])
        result["video"] = str(path)
        low_kp_results.append(result)
        print(
            f"[low-kp {index + 1}/5] mass={mass:.2f}kg -> "
            f"{result['final_displacement_m'] * 100.0:.1f}cm, "
            f"peak_v={result['max_box_planar_speed_mps']:.3f}m/s, "
            f"eef_dx={result['push_eef_forward_m'] * 100.0:.1f}cm",
            flush=True,
        )
    low_kp_comparison = output / "videos" / "00_libero_low_kp_mass_comparison.mp4"
    write_comparison(low_kp_comparison, low_kp_results)

    summary = {
        "created_at": dt.datetime.now().isoformat(),
        "artifact_type": "video_demo_only_not_a_training_dataset",
        "mujoco_version": "2.3.7",
        "robosuite_version": "1.4.0",
        "friction_mu": FRICTION_MU,
        "mass_targets_kg": MASS_TARGETS_KG,
        "oracle": {
            "description": "Direct world-frame +x force applied at the box body COM through mjData.xfrc_applied.",
            "force_x_N": ORACLE_FORCE_N,
            "force_steps": ORACLE_FORCE_STEPS,
            "force_duration_s": ORACLE_FORCE_STEPS / FPS,
            "comparison_video": str(oracle_comparison),
            "results": [without_frames(result) for result in oracle_results],
        },
        "libero_low_kp": {
            "description": "Original formal event-hold EEF delta action, but translational OSC kp is reduced only during push.",
            "translation_kp_push": LOW_TRANSLATION_KP,
            "reference_mass_kg": REFERENCE_MASS_KG,
            "target_displacement_m": TARGET_DISPLACEMENT_M,
            "chosen_amplitude": chosen_amplitude,
            "calibration": calibration,
            "comparison_video": str(low_kp_comparison),
            "results": [without_frames(result) for result in low_kp_results],
        },
    }
    write_json(output / "summary.json", summary)
    print(f"oracle_comparison={oracle_comparison}", flush=True)
    print(f"low_kp_chosen_A={chosen_amplitude:.2f}", flush=True)
    print(f"low_kp_comparison={low_kp_comparison}", flush=True)


if __name__ == "__main__":
    main()
