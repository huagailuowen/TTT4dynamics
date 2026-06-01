#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from dataclasses import replace
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
FASTWAM_ROOT = REPO_ROOT.parent / "FastWAM"
LIBERO_CONFIG_PATH = REPO_ROOT.parent / "LIBERO" / ".libero_config"

os.environ.setdefault("LIBERO_CONFIG_PATH", str(LIBERO_CONFIG_PATH))

for path in (REPO_ROOT, FASTWAM_ROOT / "src", FASTWAM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fastwam.datasets.lerobot.lerobot.lerobot_dataset import LeRobotDataset

from ttt4dynamics.cases import DynamicCarrierCase, load_cases
from ttt4dynamics.dynamic_env import DynamicCarrierEnv, create_libero_env_for_case
from ttt4dynamics.planner import PlannerConfig, ScriptedDynamicCarrierPlanner
from ttt4dynamics.trajectories import TrajectorySpec


FLAT_PROMPT = (
    "track the moving cream cheese box on the platform, pick it up, "
    "and place it on the static target region"
)
BOX_PROMPT = (
    "track the moving cream cheese box inside the open tray, pick it from above, "
    "and place it on the static target region"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect dynamic-carrier demonstrations in FastWAM/LeRobot format."
    )
    parser.add_argument("--cases", type=Path, default=REPO_ROOT / "configs/dynamic_carrier_cases.json")
    parser.add_argument("--output", type=Path, required=True, help="Output LeRobot dataset root.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--camera-resolution", type=int, default=224)
    parser.add_argument("--speed-multiplier", type=float, default=1.6)
    parser.add_argument("--max-attempts", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--repo-id", default="ttt_dynamic_carrier_cream")
    parser.add_argument("--video-codec", default="h264", choices=["h264", "hevc", "libsvtav1", "h264_nvenc"])
    parser.add_argument("--intercept-lead-s", type=float, default=0.42)
    parser.add_argument("--position-gain", type=float, default=10.0)
    parser.add_argument("--max-pos-action", type=float, default=1.0)
    parser.add_argument("--xy-tolerance", type=float, default=0.035)
    parser.add_argument("--target-xy-tolerance", type=float, default=0.055)
    parser.add_argument("--z-tolerance", type=float, default=0.035)
    parser.add_argument("--line-yaw-range", type=float, default=0.28)
    parser.add_argument("--loop-yaw-range", type=float, default=0.35)
    parser.add_argument("--period-jitter", type=float, default=0.12)
    parser.add_argument("--amplitude-jitter", type=float, default=0.14)
    parser.add_argument("--save-failed-metadata", action="store_true")
    return parser.parse_args()


def _build_features(camera_resolution: int) -> dict[str, dict[str, Any]]:
    names_action = ["dx", "dy", "dz", "dax", "day", "daz", "gripper_open"]
    names_state = [
        "eef_x",
        "eef_y",
        "eef_z",
        "eef_axis_x",
        "eef_axis_y",
        "eef_axis_z",
        "gripper_qpos_0",
        "gripper_qpos_1",
    ]
    image_shape = (3, int(camera_resolution), int(camera_resolution))
    return {
        "observation.images.image": {
            "dtype": "video",
            "shape": image_shape,
            "names": ["channel", "height", "width"],
        },
        "observation.images.wrist_image": {
            "dtype": "video",
            "shape": image_shape,
            "names": ["channel", "height", "width"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": names_state,
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": names_action,
        },
    }


def _quat_to_axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    if quat.shape[0] != 4:
        raise ValueError(f"Expected quaternion with shape (4,), got {quat.shape}")
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        return np.zeros(3, dtype=np.float32)
    quat /= norm
    if quat[0] < 0.0:
        quat *= -1.0
    w = float(np.clip(quat[0], -1.0, 1.0))
    xyz = quat[1:4]
    sin_half = float(np.linalg.norm(xyz))
    if sin_half < 1e-8:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * math.atan2(sin_half, w)
    axis = xyz / sin_half
    return (axis * angle).astype(np.float32)


def _obs_to_images(obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    agent = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]).astype(np.uint8)
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]).astype(np.uint8)
    return agent, wrist


def _obs_to_state(obs: dict[str, Any]) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            _quat_to_axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float64)),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)


def _env_action_to_fastwam_action(action: np.ndarray) -> np.ndarray:
    out = np.asarray(action, dtype=np.float32).copy()
    # LIBERO env convention here is -1=open, +1=close. FastWAM's LIBERO
    # training/eval path expects dataset convention 1=open, 0=close.
    out[-1] = (1.0 - out[-1]) / 2.0
    return out


def _prompt_for_case(case: DynamicCarrierCase) -> list[str]:
    if "box" in case.access_mode.lower() or "tray" in case.access_mode.lower():
        task = BOX_PROMPT
        coarse = "dynamic carrier open-tray pick and place"
    else:
        task = FLAT_PROMPT
        coarse = "dynamic carrier flat-platform pick and place"
    return [coarse, task, "successful scripted demonstration", "success"]


def _jitter(value: float, rng: np.random.Generator, width: float) -> float:
    return float(value * rng.uniform(1.0 - width, 1.0 + width))


def _sample_motion(
    base_motion: TrajectorySpec,
    rng: np.random.Generator,
    *,
    speed_multiplier: float,
    period_jitter: float,
    amplitude_jitter: float,
    line_yaw_range: float,
    loop_yaw_range: float,
) -> TrajectorySpec:
    family = base_motion.family.lower()
    ax, ay = base_motion.amplitude
    period = _jitter(base_motion.period / speed_multiplier, rng, period_jitter)
    direction = int(rng.choice([-1, 1]))
    phase = float(rng.uniform(0.0, 2.0 * math.pi))

    if family == "line":
        motion = replace(
            base_motion,
            amplitude=(_jitter(ax, rng, amplitude_jitter), 0.0),
            period=period,
            phase=phase,
            direction=direction,
            yaw=float(rng.uniform(-line_yaw_range, line_yaw_range)),
            harmonics=[],
        )
    elif family == "irregular_loop":
        harmonic_count = int(rng.integers(1, 3))
        harmonics = []
        for _ in range(harmonic_count):
            harmonics.append(
                {
                    "order": int(rng.choice([2, 3, 4])),
                    "x": float(rng.uniform(0.04, 0.16)),
                    "y": float(rng.uniform(0.04, 0.16)),
                    "phase": float(rng.uniform(-math.pi, math.pi)),
                }
            )
        motion = replace(
            base_motion,
            amplitude=(_jitter(ax, rng, amplitude_jitter), _jitter(ay, rng, amplitude_jitter)),
            period=period,
            phase=phase,
            direction=direction,
            yaw=float(rng.uniform(-loop_yaw_range, loop_yaw_range)),
            harmonics=harmonics,
        )
    else:
        motion = replace(
            base_motion,
            period=period,
            phase=phase,
            direction=direction,
        )

    return motion


def _sample_case_variant(
    base_cases: list[DynamicCarrierCase],
    rng: np.random.Generator,
    attempt_index: int,
    *,
    speed_multiplier: float,
    period_jitter: float,
    amplitude_jitter: float,
    line_yaw_range: float,
    loop_yaw_range: float,
) -> DynamicCarrierCase:
    base = base_cases[attempt_index % len(base_cases)]
    for resample_idx in range(100):
        motion = _sample_motion(
            base.motion,
            rng,
            speed_multiplier=speed_multiplier,
            period_jitter=period_jitter,
            amplitude_jitter=amplitude_jitter,
            line_yaw_range=line_yaw_range,
            loop_yaw_range=loop_yaw_range,
        )
        case = replace(
            base,
            case_id=f"{base.case_id}_speed{speed_multiplier:.2f}_v{attempt_index:04d}_{resample_idx:02d}",
            motion=motion,
        )
        try:
            case.validate_target_separation()
            return case
        except ValueError:
            continue
    raise RuntimeError(f"Could not sample a separated trajectory variant for base case {base.case_id}")


def _write_image_for_last_frame(dataset: LeRobotDataset, key: str, frame_index: int, image: np.ndarray) -> None:
    path = dataset._get_image_file_path(
        episode_index=dataset.episode_buffer["episode_index"],
        image_key=key,
        frame_index=frame_index,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path, quality=95)


def _remove_current_episode_images(dataset: LeRobotDataset) -> None:
    if dataset.episode_buffer is None:
        return
    episode_index = dataset.episode_buffer["episode_index"]
    for key in dataset.meta.video_keys:
        image_dir = dataset._get_image_file_path(
            episode_index=episode_index,
            image_key=key,
            frame_index=0,
        ).parent
        if image_dir.is_dir():
            shutil.rmtree(image_dir)


def _save_rollout_to_dataset(
    dataset: LeRobotDataset,
    *,
    case: DynamicCarrierCase,
    repo_root: Path,
    camera_resolution: int,
    seed: int,
    planner_config: PlannerConfig,
) -> dict[str, Any]:
    base_env, init_state, task_description = create_libero_env_for_case(
        case,
        repo_root=repo_root,
        camera_resolution=camera_resolution,
        seed=seed,
    )
    env = DynamicCarrierEnv(base_env, case)
    planner = ScriptedDynamicCarrierPlanner(env, planner_config)
    task = _prompt_for_case(case)
    phase_counts: dict[str, int] = {}
    success = False
    done = False
    frame_count = 0

    try:
        _remove_current_episode_images(dataset)
        obs = env.reset(init_state=init_state)
        planner.reset()
        for _ in range(int(case.max_steps)):
            phase_counts[str(planner.phase.value)] = phase_counts.get(str(planner.phase.value), 0) + 1
            action = planner.act(obs)
            agent, wrist = _obs_to_images(obs)
            frame = {
                "observation.images.image": agent,
                "observation.images.wrist_image": wrist,
                "observation.state": _obs_to_state(obs),
                "action": _env_action_to_fastwam_action(action),
            }
            dataset.add_frame(frame, task=task, timestamp=frame_count / float(case.control_freq))
            _write_image_for_last_frame(dataset, "observation.images.image", frame_count, agent)
            _write_image_for_last_frame(dataset, "observation.images.wrist_image", frame_count, wrist)
            frame_count += 1

            obs, _, done, _ = env.step(action)
            success = bool(env.check_success())
            if success or planner.is_done():
                success = bool(env.check_success())
                break

        if success:
            dataset.save_episode()
        else:
            _remove_current_episode_images(dataset)
            dataset.clear_episode_buffer()

        return {
            "success": bool(success),
            "steps": int(frame_count),
            "seed": int(seed),
            "case": case.as_dict(),
            "task_description": task_description,
            "phase_counts": phase_counts,
            "final_phase": str(planner.phase.value),
        }
    finally:
        env.close()


def collect_lerobot_dataset(args: argparse.Namespace) -> None:
    base_cases = load_cases(args.cases)
    output = args.output.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists; pass --overwrite to replace it: {output}")
        shutil.rmtree(output)

    rng = np.random.default_rng(int(args.seed))
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=output,
        fps=20,
        features=_build_features(args.camera_resolution),
        use_videos=True,
        video_codec=args.video_codec,
        is_compute_episode_stats_image=False,
    )
    planner_config = PlannerConfig(
        intercept_lead_s=float(args.intercept_lead_s),
        position_gain=float(args.position_gain),
        max_pos_action=float(args.max_pos_action),
        xy_tolerance=float(args.xy_tolerance),
        target_xy_tolerance=float(args.target_xy_tolerance),
        z_tolerance=float(args.z_tolerance),
    )

    metadata = {
        "created_at": dt.datetime.now().isoformat(),
        "dataset_type": "ttt_dynamic_carrier_lerobot",
        "episodes_requested": int(args.episodes),
        "speed_multiplier": float(args.speed_multiplier),
        "camera_resolution": int(args.camera_resolution),
        "seed": int(args.seed),
        "base_cases": [case.as_dict() for case in base_cases],
        "successes": [],
        "failures": [],
        "prompts": {
            "flat": FLAT_PROMPT,
            "open_box": BOX_PROMPT,
        },
    }

    successes = 0
    attempts = 0
    while successes < int(args.episodes) and attempts < int(args.max_attempts):
        case = _sample_case_variant(
            base_cases,
            rng,
            attempts,
            speed_multiplier=float(args.speed_multiplier),
            period_jitter=float(args.period_jitter),
            amplitude_jitter=float(args.amplitude_jitter),
            line_yaw_range=float(args.line_yaw_range),
            loop_yaw_range=float(args.loop_yaw_range),
        )
        episode_seed = int(args.seed + attempts)
        attempts += 1
        result = _save_rollout_to_dataset(
            dataset,
            case=case,
            repo_root=args.repo_root,
            camera_resolution=int(args.camera_resolution),
            seed=episode_seed,
            planner_config=planner_config,
        )

        if result["success"]:
            result["episode_index"] = int(successes)
            metadata["successes"].append(result)
            successes += 1
            print(
                f"[success {successes:04d}/{args.episodes}] "
                f"attempt={attempts:04d} case={case.case_id} steps={result['steps']}"
            )
        else:
            if args.save_failed_metadata:
                metadata["failures"].append(result)
            print(
                f"[failed] attempt={attempts:04d} case={case.case_id} "
                f"steps={result['steps']} final_phase={result['final_phase']}"
            )

        metadata["attempts"] = int(attempts)
        metadata["episodes_collected"] = int(successes)
        (output / "dynamic_carrier_generation_metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

    if successes < int(args.episodes):
        raise RuntimeError(
            f"Only collected {successes}/{args.episodes} successful demos after {attempts} attempts."
        )

    print(f"Wrote LeRobot dataset with {successes} successful episodes: {output}")


def main() -> None:
    args = parse_args()
    collect_lerobot_dataset(args)


if __name__ == "__main__":
    main()
