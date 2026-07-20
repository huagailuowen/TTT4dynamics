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
BASE_SCRIPT = REPO_ROOT / "scripts" / "collect_libero_push_box_formal_6fric_50pair_35_35_direct_lerobot_hai-machine.py"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "pushbox" / "demos" / "libero_push_box_mass_sweep_mu0080_mid40cm_2026-07-16_hai-machine"
FRICTION_MU = 0.08
TARGET_DISPLACEMENT_M = 0.40
HOLD_AFTER_CONTACT = 3
FPS = 20
CAMERA_RESOLUTION = 224
VIDEO_CRF = 18
CALIBRATION_AMPLITUDES = np.arange(0.20, 0.401, 0.02, dtype=np.float64)
MASS_TARGETS_KG = (0.001, 0.003, None, 0.030, 0.150)


def load_base_module() -> Any:
    spec = importlib.util.spec_from_file_location("formal6_mass_demo_base_hai_machine", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import base script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a fixed-action LIBERO box-mass comparison at mu=0.08.")
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
    target_xy = (float(base.INIT_XY[0] + base.DUMMY_TARGET_DISTANCE), float(base.INIT_XY[1]))
    plan = {
        "mu": FRICTION_MU,
        "mode": "event_hold",
        "A": float(amplitude),
        "hold": HOLD_AFTER_CONTACT,
    }
    return base.build_push_case(
        plan,
        case_id=f"mass_demo_mu0080_A{int(round(amplitude * 1000)):03d}",
        bddl_file=bddl_file,
        target_xy=target_xy,
        camera_resolution=CAMERA_RESOLUTION,
    )


def box_body_ids(env: Any) -> list[int]:
    model = env.inner_env.sim.model
    obj = env.inner_env.get_object(env.case.box_name)
    ids = []
    for body_id in range(int(model.nbody)):
        name = model.body_id2name(body_id) or ""
        if name == obj.name or name.startswith(f"{obj.name}_"):
            ids.append(int(body_id))
    if not ids:
        raise RuntimeError(f"No MuJoCo body found for object {obj.name!r}")
    return ids


def native_box_mass_kg(env: Any) -> float:
    model = env.inner_env.sim.model
    return float(sum(float(model.body_mass[body_id]) for body_id in box_body_ids(env)))


def set_box_mass_kg(env: Any, target_mass_kg: float) -> dict[str, Any]:
    model = env.inner_env.sim.model
    ids = box_body_ids(env)
    original_total = float(sum(float(model.body_mass[body_id]) for body_id in ids))
    if original_total <= 0.0 or target_mass_kg <= 0.0:
        raise ValueError("Both native and target box masses must be positive.")
    scale = float(target_mass_kg) / original_total
    for body_id in ids:
        model.body_mass[body_id] *= scale
        model.body_inertia[body_id] *= scale
    env.inner_env.sim.forward()
    return {
        "body_ids": ids,
        "body_names": [model.body_id2name(body_id) for body_id in ids],
        "native_mass_kg": original_total,
        "target_mass_kg": float(target_mass_kg),
        "mass_scale": scale,
    }


def prepare_rollout(env: Any, *, target_mass_kg: float | None) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    env.reset()
    native_mass = native_box_mass_kg(env)
    mass_info = set_box_mass_kg(env, native_mass if target_mass_kg is None else float(target_mass_kg))
    base.move_to_start(env, (0.0, 0.0))
    env.step_count = 0
    env._last_scripted_action = np.zeros(7, dtype=np.float64)
    env._last_scripted_phase = None
    obs = env._last_obs
    initial_xyz, _ = env.box_pose()
    return obs, np.asarray(initial_xyz, dtype=np.float64), mass_info


def label_frame(frame: np.ndarray, *, mass_kg: float, amplitude: float, phase: str, displacement_m: float) -> np.ndarray:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    draw = ImageDraw.Draw(image)
    lines = [
        f"mu={FRICTION_MU:.2f}  mass={mass_kg * 1000.0:.2f}g  fixed A={amplitude:.3f}",
        f"phase={phase}  box displacement={displacement_m * 100.0:.1f}cm",
    ]
    draw.rectangle((3, 3, 221, 36), fill=(0, 0, 0))
    draw.text((7, 6), lines[0], fill=(255, 255, 255))
    draw.text((7, 20), lines[1], fill=(220, 235, 255))
    return np.asarray(image, dtype=np.uint8)


def rollout(
    *,
    amplitude: float,
    mass_kg: float | None,
    bddl_file: str,
    seed: int,
    capture_frames: bool,
) -> dict[str, Any]:
    case = make_case(amplitude=amplitude, bddl_file=bddl_file)
    env = base.LiberoPushBoxEnv(case, repo_root=REPO_ROOT, seed=int(seed))
    rows: list[dict[str, Any]] = []
    frames: list[np.ndarray] = []
    try:
        obs, initial_xyz, mass_info = prepare_rollout(env, target_mass_kg=mass_kg)
        actual_mass = float(mass_info["target_mass_kg"])
        if capture_frames:
            agent, _ = base._obs_to_images(obs)
            frames.append(label_frame(agent, mass_kg=actual_mass, amplitude=amplitude, phase="prepared", displacement_m=0.0))
        for _ in range(int(case.max_steps)):
            obs, _, _, info = env.step()
            row = dict(info["push_box"])
            rows.append(row)
            if capture_frames:
                box_xyz = np.asarray(row["box_xyz"], dtype=np.float64)
                displacement = float(np.linalg.norm(box_xyz[:2] - initial_xyz[:2]))
                agent, _ = base._obs_to_images(obs)
                frames.append(
                    label_frame(
                        agent,
                        mass_kg=actual_mass,
                        amplitude=amplitude,
                        phase=str(row["phase"]),
                        displacement_m=displacement,
                    )
                )
    finally:
        env.close()

    final_xyz = np.asarray(rows[-1]["box_xyz"], dtype=np.float64)
    velocities = np.asarray([row["box_vxy"] for row in rows], dtype=np.float64)
    result = {
        "friction_mu": FRICTION_MU,
        "amplitude": float(amplitude),
        "hold_after_contact": HOLD_AFTER_CONTACT,
        "mass": mass_info,
        "initial_box_xyz_m": initial_xyz,
        "final_box_xyz_m": final_xyz,
        "final_displacement_m": float(np.linalg.norm(final_xyz[:2] - initial_xyz[:2])),
        "final_forward_m": float(final_xyz[0] - initial_xyz[0]),
        "max_box_planar_speed_mps": float(np.max(np.linalg.norm(velocities, axis=1))),
        "step_count": len(rows),
        "frames": frames,
    }
    return result


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
    with imageio.get_writer(
        path,
        fps=FPS,
        codec="libx264",
        macro_block_size=None,
        ffmpeg_params=["-crf", str(VIDEO_CRF), "-pix_fmt", "yuv420p"],
    ) as writer:
        for frame_index in range(max_frames):
            tiles = []
            for result in results:
                frames = result["frames"]
                tiles.append(frames[min(frame_index, len(frames) - 1)])
            tiles.append(blank)
            top = np.concatenate(tiles[:3], axis=1)
            bottom = np.concatenate(tiles[3:6], axis=1)
            writer.append_data(np.concatenate([top, bottom], axis=0))


def without_frames(result: dict[str, Any]) -> dict[str, Any]:
    payload = dict(result)
    payload.pop("frames", None)
    return payload


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists; pass --overwrite: {output}")
        shutil.rmtree(output)
    (output / "videos").mkdir(parents=True, exist_ok=True)
    bddl_file = base.write_geometry_bddl(
        repo_root=REPO_ROOT,
        bddl_dir=output / "bddl",
        geometry_id="mass_sweep_mu0080_hidden",
        init_xy=base.INIT_XY,
        target_xy=(float(base.INIT_XY[0] + base.DUMMY_TARGET_DISTANCE), float(base.INIT_XY[1])),
        init_half_size=0.002,
        target_radius=base.TARGET_RADIUS,
        target_rgba=(0.0, 0.8, 0.2, 0.0),
    )

    calibration = []
    for index, amplitude in enumerate(CALIBRATION_AMPLITUDES):
        result = rollout(
            amplitude=float(amplitude),
            mass_kg=None,
            bddl_file=bddl_file,
            seed=int(args.seed),
            capture_frames=False,
        )
        calibration.append(without_frames(result))
        print(
            f"[calibration {index + 1:02d}/{len(CALIBRATION_AMPLITUDES):02d}] "
            f"A={amplitude:.3f} displacement={result['final_displacement_m'] * 100.0:.1f}cm",
            flush=True,
        )
    chosen = min(calibration, key=lambda item: abs(float(item["final_displacement_m"]) - TARGET_DISPLACEMENT_M))
    chosen_amplitude = float(chosen["amplitude"])
    native_mass = float(chosen["mass"]["native_mass_kg"])

    results = []
    resolved_masses = [native_mass if mass is None else float(mass) for mass in MASS_TARGETS_KG]
    for index, mass in enumerate(resolved_masses):
        result = rollout(
            amplitude=chosen_amplitude,
            mass_kg=mass,
            bddl_file=bddl_file,
            seed=int(args.seed),
            capture_frames=True,
        )
        tag = f"mass_{mass * 1000.0:07.2f}g".replace(".", "p")
        video = output / "videos" / f"{index + 1:02d}_{tag}.mp4"
        write_video(video, result["frames"])
        result["video"] = str(video)
        results.append(result)
        print(
            f"[mass {index + 1}/5] {mass * 1000.0:.2f}g -> "
            f"{result['final_displacement_m'] * 100.0:.1f}cm, "
            f"peak_v={result['max_box_planar_speed_mps']:.3f}m/s",
            flush=True,
        )

    comparison = output / "videos" / "00_mass_comparison_mu0080_fixed_action.mp4"
    write_comparison(comparison, results)
    summary = {
        "created_at": dt.datetime.now().isoformat(),
        "artifact_type": "video_demo_only_not_a_training_dataset",
        "purpose": "Measure box-mass dependence under one fixed push action and fixed friction.",
        "friction_mu": FRICTION_MU,
        "target_calibration_displacement_m": TARGET_DISPLACEMENT_M,
        "chosen_amplitude": chosen_amplitude,
        "native_box_mass_kg": native_mass,
        "mass_targets_kg": resolved_masses,
        "action_policy": "original formal-6friction event_hold profile: first command 0.5A, then A until contact, hold A for 3 steps, then zero",
        "inertia_policy": "scale all object body inertias by the same ratio as body mass",
        "comparison_video": str(comparison),
        "calibration": calibration,
        "mass_results": [without_frames(result) for result in results],
    }
    write_json(output / "summary.json", summary)
    print(f"chosen A={chosen_amplitude:.3f}; native mass={native_mass * 1000.0:.3f}g", flush=True)
    print(f"comparison={comparison}", flush=True)


if __name__ == "__main__":
    main()
