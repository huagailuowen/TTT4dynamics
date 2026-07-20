#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
LIBERO_REPO = REPO_ROOT.parent / "LIBERO"
FASTWAM_ROOT = REPO_ROOT.parent / "FastWAM"
for path in (REPO_ROOT, SCRIPTS_DIR, LIBERO_REPO, FASTWAM_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from collect_libero_push_box_rollout_target_lerobot_dataset import (  # noqa: E402
    _env_action_to_fastwam_action,
    _obs_to_images,
    _obs_to_state,
    build_features,
    patch_lerobot_video_crf,
    prompt_for_case,
    to_jsonable,
    write_dataset_metadata,
)
from fastwam.datasets.lerobot.lerobot.lerobot_dataset import LeRobotDataset  # noqa: E402
from ttt4dynamics.push_box_libero import LiberoPushBoxEnv  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs" / "libero_two_box_collision_9speed_20mass_2026-07-16_hai-machine.json"
DEFAULT_OUTPUT = REPO_ROOT / "data/mass/original_mass_grid_9speed_20mass_2026-07-16_hai-machine"
COLLISION_DEMO = REPO_ROOT / "scripts" / "render_libero_two_box_collision_mass_demo_2026-07-16_hai-machine.py"
VIDEO_CODEC = "h264"
VIDEO_CRF = 18
JPEG_QUALITY = 98
TASK = prompt_for_case("observation", "straight")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect a 9-speed x 20-mass two-block collision LeRobot dataset.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video-codec", choices=["h264", "hevc", "libsvtav1", "h264_nvenc"], default=VIDEO_CODEC)
    parser.add_argument("--video-crf", type=int, default=VIDEO_CRF)
    parser.add_argument("--jpeg-quality", type=int, default=JPEG_QUALITY)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(value), indent=2), encoding="utf-8")


def load_collision_demo() -> Any:
    spec = importlib.util.spec_from_file_location("two_box_collision_dataset_base", COLLISION_DEMO)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load collision demo: {COLLISION_DEMO}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_demo(demo: Any, config: dict[str, Any]) -> None:
    demo.FRICTION_MU = float(config["friction_mu"])
    demo.PROJECTILE_MASS_KG = float(config["projectile_mass_kg"])
    demo.PROJECTILE_INIT_XY = tuple(float(v) for v in config["projectile_init_xy"])
    demo.TARGET_INIT_XY = tuple(float(v) for v in config["target_init_xy"])
    demo.CONTROLLER_TRANSLATION_SCALE = float(config["controller_translation_scale"])
    demo.BRAKE_PROFILE = np.asarray(config["brake_profile"], dtype=np.float64)
    demo.HAI_OBJECT_CENTER_GAP_M = float(config["object_center_gap_m"])
    demo.HAI_IMPACT_LATERAL_OFFSET_MAX_M = float(config["impact_lateral_offset_max_m"])


def profile_for_action(action_cfg: dict[str, Any]) -> np.ndarray:
    steps = int(action_cfg["push_steps"])
    if steps < 2:
        raise ValueError(f"push_steps must be >=2, got {steps}")
    return np.asarray([0.7] + [1.0] * (steps - 2) + [0.7], dtype=np.float64)


def hash_actions(actions: np.ndarray) -> str:
    values = np.ascontiguousarray(actions, dtype="<f8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def copy_obs(obs: dict[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, value in obs.items():
        copied[key] = np.array(value, copy=True) if isinstance(value, np.ndarray) else value
    return copied


def set_zero_object_velocity(demo: Any, env: Any, name: str) -> None:
    obj = demo.object_instance(env, name)
    for joint_name in obj.joints:
        env.inner_env.sim.data.set_joint_qvel(joint_name, np.zeros(6, dtype=np.float64))
    env.inner_env.sim.forward()


def place_objects_at_exact_initial_poses(demo: Any, env: Any) -> dict[str, float]:
    projectile = demo.object_instance(env, demo.PROJECTILE_NAME)
    target = demo.object_instance(env, demo.TARGET_NAME)
    projectile_qpos = np.asarray(
        env.inner_env.sim.data.get_joint_qpos(projectile.joints[-1]), dtype=np.float64
    ).copy()
    target_qpos = np.asarray(
        env.inner_env.sim.data.get_joint_qpos(target.joints[-1]), dtype=np.float64
    ).copy()
    common_z = float(target_qpos[2])
    common_quat = target_qpos[3:7].copy()
    projectile_qpos[:2] = np.asarray(demo.PROJECTILE_INIT_XY, dtype=np.float64)
    target_qpos[:2] = np.asarray(demo.TARGET_INIT_XY, dtype=np.float64)
    projectile_qpos[2] = common_z
    target_qpos[2] = common_z
    projectile_qpos[3:7] = common_quat
    target_qpos[3:7] = common_quat
    env.inner_env.sim.data.set_joint_qpos(projectile.joints[-1], projectile_qpos)
    env.inner_env.sim.data.set_joint_qpos(target.joints[-1], target_qpos)
    env.inner_env.sim.data.set_joint_qvel(projectile.joints[-1], np.zeros(6, dtype=np.float64))
    env.inner_env.sim.data.set_joint_qvel(target.joints[-1], np.zeros(6, dtype=np.float64))
    env.inner_env.sim.forward()
    projectile_xyz, _, _ = demo.object_state(env, demo.PROJECTILE_NAME)
    target_xyz, _, _ = demo.object_state(env, demo.TARGET_NAME)
    return {
        "center_gap_m": float(target_xyz[0] - projectile_xyz[0]),
        "lateral_offset_m": float(target_xyz[1] - projectile_xyz[1]),
    }


def build_fixed_actions(demo: Any, action_cfg: dict[str, Any], recorded_steps: int) -> np.ndarray:
    demo.LAUNCH_PROFILE = profile_for_action(action_cfg)
    pulse = np.asarray(demo.fixed_launch_actions(float(action_cfg["A"])), dtype=np.float64)
    if pulse.ndim != 2 or pulse.shape[1] != 7:
        raise RuntimeError(f"Expected pulse shape [T,7], got {pulse.shape}")
    if len(pulse) > int(recorded_steps):
        raise RuntimeError(f"Pulse has {len(pulse)} frames but recorded_steps={recorded_steps}")
    hold = np.zeros((int(recorded_steps) - len(pulse), 7), dtype=np.float64)
    return np.concatenate([pulse, hold], axis=0)


def rollout_episode(
    demo: Any,
    *,
    bddl_file: str,
    target_mass_kg: float,
    action_cfg: dict[str, Any],
    recorded_steps: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], np.ndarray]:
    case = demo.make_case(bddl_file=bddl_file)
    env = LiberoPushBoxEnv(case, repo_root=REPO_ROOT, seed=int(seed))
    records: list[dict[str, Any]] = []
    actions = build_fixed_actions(demo, action_cfg, recorded_steps)
    try:
        env.reset()
        projectile_mass_info = demo.set_object_mass(env, demo.PROJECTILE_NAME, float(demo.PROJECTILE_MASS_KG))
        target_mass_info = demo.set_object_mass(env, demo.TARGET_NAME, float(target_mass_kg))
        demo.set_object_contact_properties(env, demo.PROJECTILE_NAME, rgba=(0.10, 0.35, 0.95, 1.0))
        demo.set_object_contact_properties(env, demo.TARGET_NAME, rgba=(0.95, 0.25, 0.05, 1.0))
        set_zero_object_velocity(demo, env, demo.PROJECTILE_NAME)
        set_zero_object_velocity(demo, env, demo.TARGET_NAME)
        place_objects_at_exact_initial_poses(demo, env)
        env._last_obs = env._refresh_obs()

        projectile_geoms = demo.object_geom_ids(env, demo.PROJECTILE_NAME)
        target_geoms = demo.object_geom_ids(env, demo.TARGET_NAME)
        setup_steps = int(demo.establish_projectile_contact(env, projectile_geoms))
        if not demo.robot_projectile_contact(env, projectile_geoms):
            raise RuntimeError("Robot failed to establish contact with the projectile block")
        alignment = place_objects_at_exact_initial_poses(demo, env)
        if abs(alignment["lateral_offset_m"]) > 1e-6:
            raise RuntimeError(f"Initial blocks are not laterally aligned: {alignment}")
        env._last_obs = env._refresh_obs()

        base_limits = demo.controller_translation_limits(env)
        demo.set_controller_translation_scale(
            env,
            base_limits,
            float(demo.CONTROLLER_TRANSLATION_SCALE),
        )
        initial_projectile_xyz, _, initial_projectile_vel = demo.object_state(env, demo.PROJECTILE_NAME)
        initial_target_xyz, _, initial_target_vel = demo.object_state(env, demo.TARGET_NAME)

        for frame_index, commanded_action in enumerate(actions):
            obs_before = env._last_obs if env._last_obs is not None else env._refresh_obs()
            obs_before = copy_obs(obs_before)
            _, _, _, info = env.step(commanded_action)
            push_info = dict(info.get("push_box", {}))
            actual_action = np.asarray(push_info.get("action", commanded_action), dtype=np.float64)
            if not np.array_equal(actual_action, commanded_action):
                raise RuntimeError(f"Environment changed commanded action at frame {frame_index}")

            projectile_xyz, projectile_quat, projectile_vel = demo.object_state(env, demo.PROJECTILE_NAME)
            target_xyz, target_quat, target_vel = demo.object_state(env, demo.TARGET_NAME)
            records.append(
                {
                    "frame_index": int(frame_index),
                    "obs": obs_before,
                    "action": actual_action.copy(),
                    "phase": "launch" if frame_index < len(demo.LAUNCH_PROFILE) else "brake_or_coast",
                    "projectile_xyz": projectile_xyz.copy(),
                    "projectile_quat": projectile_quat.copy(),
                    "projectile_vel": projectile_vel.copy(),
                    "target_xyz": target_xyz.copy(),
                    "target_quat": target_quat.copy(),
                    "target_vel": target_vel.copy(),
                    "robot_projectile_contact": bool(demo.robot_projectile_contact(env, projectile_geoms)),
                    "robot_target_contact": bool(demo.robot_projectile_contact(env, target_geoms)),
                    "block_block_contact": bool(demo.contact_between(env, projectile_geoms, target_geoms)),
                }
            )
    finally:
        env.close()

    sampled_block_contact_frames = [r["frame_index"] for r in records if r["block_block_contact"]]
    transfer_event_frames = [
        r["frame_index"]
        for r in records
        if (
            np.linalg.norm(np.asarray(r["target_xyz"][:2]) - np.asarray(initial_target_xyz[:2])) > 1e-4
            or np.linalg.norm(np.asarray(r["target_vel"][:2])) > 5e-3
        )
    ]
    robot_projectile_frames = [r["frame_index"] for r in records if r["robot_projectile_contact"]]
    robot_target_frames = [r["frame_index"] for r in records if r["robot_target_contact"]]
    if not transfer_event_frames:
        raise RuntimeError("No projectile-to-target momentum-transfer event was observed")
    first_collision = int(transfer_event_frames[0])
    last_robot_projectile = int(robot_projectile_frames[-1]) if robot_projectile_frames else -1
    if first_collision <= last_robot_projectile:
        raise RuntimeError(
            f"Robot had not left projectile before collision: collision={first_collision}, robot_end={last_robot_projectile}"
        )
    if robot_target_frames:
        raise RuntimeError(f"Robot directly contacted target at frames {robot_target_frames}")

    preimpact_vel = initial_projectile_vel if first_collision == 0 else records[first_collision - 1]["projectile_vel"]
    preimpact_frame = max(0, first_collision - 1)
    signed_impact_lateral_offset_m = float(
        records[preimpact_frame]["projectile_xyz"][1] - records[preimpact_frame]["target_xyz"][1]
    )
    signed_postevent_lateral_offset_m = float(
        records[first_collision]["projectile_xyz"][1] - records[first_collision]["target_xyz"][1]
    )
    impact_lateral_offset_m = abs(signed_impact_lateral_offset_m)
    postevent_lateral_offset_m = abs(signed_postevent_lateral_offset_m)
    if impact_lateral_offset_m > float(demo.HAI_IMPACT_LATERAL_OFFSET_MAX_M):
        raise RuntimeError(
            f"Off-center collision: lateral_offset={impact_lateral_offset_m:.6f}m "
            f"> {float(demo.HAI_IMPACT_LATERAL_OFFSET_MAX_M):.6f}m"
        )
    target_peak_vx = max(float(r["target_vel"][0]) for r in records)
    projectile_peak_preimpact_vx = max(
        float(r["projectile_vel"][0]) for r in records[: max(1, first_collision)]
    )
    final_projectile_xyz = np.asarray(records[-1]["projectile_xyz"], dtype=np.float64)
    final_target_xyz = np.asarray(records[-1]["target_xyz"], dtype=np.float64)
    diagnostics = {
        "setup_steps_not_recorded": setup_steps,
        "recorded_steps": len(records),
        "first_block_collision_frame": first_collision,
        "collision_event_definition": "first target planar displacement >0.1 mm or planar speed >5 mm/s",
        "sampled_block_contact_frames": sampled_block_contact_frames,
        "last_robot_projectile_contact_frame": last_robot_projectile,
        "separation_frames": int(first_collision - last_robot_projectile),
        "robot_target_contact_frames": robot_target_frames,
        "clean_collision": True,
        "initial_center_gap_m": float(alignment["center_gap_m"]),
        "initial_lateral_offset_m": float(alignment["lateral_offset_m"]),
        "impact_lateral_offset_m": impact_lateral_offset_m,
        "signed_impact_lateral_offset_m": signed_impact_lateral_offset_m,
        "postevent_lateral_offset_m": postevent_lateral_offset_m,
        "signed_postevent_lateral_offset_m": signed_postevent_lateral_offset_m,
        "preimpact_projectile_vx_mps": float(preimpact_vel[0]),
        "projectile_peak_preimpact_vx_mps": projectile_peak_preimpact_vx,
        "target_peak_vx_mps": target_peak_vx,
        "initial_projectile_xyz": np.asarray(initial_projectile_xyz, dtype=float).tolist(),
        "initial_target_xyz": np.asarray(initial_target_xyz, dtype=float).tolist(),
        "initial_target_velocity": np.asarray(initial_target_vel, dtype=float).tolist(),
        "final_projectile_xyz": final_projectile_xyz.astype(float).tolist(),
        "final_target_xyz": final_target_xyz.astype(float).tolist(),
        "projectile_displacement_m": float(np.linalg.norm(final_projectile_xyz[:2] - initial_projectile_xyz[:2])),
        "target_displacement_m": float(np.linalg.norm(final_target_xyz[:2] - initial_target_xyz[:2])),
        "projectile_mass": projectile_mass_info,
        "target_mass": target_mass_info,
    }
    return records, diagnostics, actions


def write_image_for_frame(
    dataset: LeRobotDataset,
    key: str,
    frame_index: int,
    image: np.ndarray,
    *,
    jpeg_quality: int,
) -> None:
    path = dataset._get_image_file_path(
        episode_index=dataset.episode_buffer["episode_index"],
        image_key=key,
        frame_index=int(frame_index),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path, quality=int(jpeg_quality))


def save_lerobot_episode(
    dataset: LeRobotDataset,
    records: list[dict[str, Any]],
    *,
    fps: int,
    jpeg_quality: int,
) -> int:
    episode_index = int(dataset.meta.total_episodes)
    for frame_index, record in enumerate(records):
        agent, wrist = _obs_to_images(record["obs"])
        frame = {
            "observation.images.image": agent,
            "observation.images.wrist_image": wrist,
            "observation.state": _obs_to_state(record["obs"]),
            "action": _env_action_to_fastwam_action(np.asarray(record["action"], dtype=np.float32)),
        }
        dataset.add_frame(frame, task=TASK, timestamp=float(frame_index) / float(fps))
        write_image_for_frame(dataset, "observation.images.image", frame_index, agent, jpeg_quality=jpeg_quality)
        write_image_for_frame(dataset, "observation.images.wrist_image", frame_index, wrist, jpeg_quality=jpeg_quality)
    dataset.save_episode()
    return episode_index


def diagnostics_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "frame_index": int(r["frame_index"]),
            "phase": str(r["phase"]),
            "action": np.asarray(r["action"], dtype=float).tolist(),
            "projectile_xyz": np.asarray(r["projectile_xyz"], dtype=float).tolist(),
            "projectile_quat": np.asarray(r["projectile_quat"], dtype=float).tolist(),
            "projectile_velocity": np.asarray(r["projectile_vel"], dtype=float).tolist(),
            "target_xyz": np.asarray(r["target_xyz"], dtype=float).tolist(),
            "target_quat": np.asarray(r["target_quat"], dtype=float).tolist(),
            "target_velocity": np.asarray(r["target_vel"], dtype=float).tolist(),
            "robot_projectile_contact": bool(r["robot_projectile_contact"]),
            "robot_target_contact": bool(r["robot_target_contact"]),
            "block_block_contact": bool(r["block_block_contact"]),
        }
        for r in records
    ]


def create_dataset(root: Path, config: dict[str, Any], video_codec: str) -> LeRobotDataset:
    return LeRobotDataset.create(
        repo_id="libero_two_box_collision_9speed_20mass_hai_machine",
        root=root,
        fps=int(config["fps"]),
        features=build_features(int(config["camera_resolution"])),
        use_videos=True,
        video_codec=video_codec,
        is_compute_episode_stats_image=False,
    )


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    output_root = args.output_root.resolve()
    config = load_json(config_path)
    actions_cfg = list(config["actions"])
    masses = [float(v) for v in config["target_masses_kg"]]
    expected_total = len(actions_cfg) * len(masses)
    if len(actions_cfg) != 9 or len(masses) != 20:
        raise ValueError(f"Expected 9 actions and 20 masses, got {len(actions_cfg)} and {len(masses)}")

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_root} exists; pass --overwrite")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "config_used.json", config)

    demo = load_collision_demo()
    configure_demo(demo, config)
    bddl_file = demo.write_two_box_bddl(output_root)
    patch_lerobot_video_crf(int(args.video_crf))
    dataset_root = output_root / "libero_two_box_collision_9speed_20mass_180eps_lerobot_2026-07-16_hai-machine"
    dataset = create_dataset(dataset_root, config, str(args.video_codec))

    metadata: dict[str, Any] = {
        "created_at": dt.datetime.now().isoformat(),
        "dataset_type": config["dataset_type"],
        "episode_count_expected": expected_total,
        "grid": {"action_count": len(actions_cfg), "target_mass_count": len(masses)},
        "projectile_mass_kg": float(config["projectile_mass_kg"]),
        "target_masses_kg": masses,
        "friction_mu": float(config["friction_mu"]),
        "camera_resolution": int(config["camera_resolution"]),
        "fps": int(config["fps"]),
        "video_codec": str(args.video_codec),
        "video_crf": int(args.video_crf),
        "jpeg_quality": int(args.jpeg_quality),
        "state_source": "true LIBERO obs robot0_eef_pos, robot0_eef_quat converted to axis-angle, robot0_gripper_qpos",
        "action_source": "the exact explicit 7D LIBERO OSC action executed in the simulator and converted with the established FastWAM mapping",
        "episodes": [],
    }
    manifest: dict[str, Any] = {
        "created_at": dt.datetime.now().isoformat(),
        "dataset_type": config["dataset_type"],
        "output_root": str(output_root),
        "lerobot_root": str(dataset_root),
        "config_path": str(config_path),
        "bddl_file": str(bddl_file),
        "episodes": [],
    }
    rows: list[dict[str, Any]] = []
    expected_action_hashes: dict[int, str] = {}

    def autosave() -> None:
        write_json(output_root / "manifest.json", manifest)
        with (output_root / "episodes.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(to_jsonable(row)) + "\n")
        write_dataset_metadata(dataset_root, metadata, rows)

    count = 0
    for action_cfg in actions_cfg:
        action_id = int(action_cfg["action_id"])
        for mass_index, target_mass_kg in enumerate(masses):
            records, diagnostics, fixed_actions = rollout_episode(
                demo,
                bddl_file=bddl_file,
                target_mass_kg=target_mass_kg,
                action_cfg=action_cfg,
                recorded_steps=int(config["recorded_steps"]),
                seed=int(args.seed),
            )
            current_hash = hash_actions(fixed_actions)
            expected_hash = expected_action_hashes.setdefault(action_id, current_hash)
            if current_hash != expected_hash:
                raise RuntimeError(f"Action changed across masses for action_id={action_id}")

            episode_index = save_lerobot_episode(
                dataset,
                records,
                fps=int(config["fps"]),
                jpeg_quality=int(args.jpeg_quality),
            )
            case_id = f"a{action_id:02d}_m{mass_index:02d}_{int(round(target_mass_kg * 1000)):05d}g"
            diagnostics_file = output_root / "diagnostics" / f"episode_{episode_index:06d}_{case_id}.json"
            write_json(diagnostics_file, diagnostics_rows(records))
            row = {
                "episode_index": int(episode_index),
                "case_id": case_id,
                "action_id": action_id,
                "A": float(action_cfg["A"]),
                "push_steps": int(action_cfg["push_steps"]),
                "launch_profile": profile_for_action(action_cfg).astype(float).tolist(),
                "calibrated_preimpact_vx_mps": float(action_cfg["calibrated_preimpact_vx_mps"]),
                "target_mass_index": int(mass_index),
                "target_mass_kg": float(target_mass_kg),
                "target_mass_g": float(target_mass_kg * 1000.0),
                "projectile_mass_kg": float(config["projectile_mass_kg"]),
                "friction_mu": float(config["friction_mu"]),
                "projectile_init_xy": list(config["projectile_init_xy"]),
                "target_init_xy": list(config["target_init_xy"]),
                "action_sha256": current_hash,
                "diagnostics_file": str(diagnostics_file),
                "metrics": diagnostics,
            }
            rows.append(row)
            metadata["episodes"].append(row)
            manifest["episodes"].append(row)
            count += 1
            print(
                f"collect {count:03d}/{expected_total:03d} {case_id} "
                f"v={diagnostics['preimpact_projectile_vx_mps']:.3f}m/s "
                f"target_dx={diagnostics['target_displacement_m'] * 100.0:.1f}cm "
                f"gap={diagnostics['separation_frames']}f",
                flush=True,
            )
            autosave()

    speeds_by_action: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        speeds_by_action[int(row["action_id"])].append(float(row["metrics"]["preimpact_projectile_vx_mps"]))
    summary = {
        "episode_count": len(rows),
        "expected_episode_count": expected_total,
        "lerobot_root": str(dataset_root),
        "count_by_action": dict(sorted(Counter(int(r["action_id"]) for r in rows).items())),
        "count_by_mass_g": dict(sorted(Counter(float(r["target_mass_g"]) for r in rows).items())),
        "preimpact_speed_by_action": {
            str(action_id): {
                "min_mps": min(values),
                "max_mps": max(values),
                "span_mps": max(values) - min(values),
                "mean_mps": float(np.mean(values)),
            }
            for action_id, values in sorted(speeds_by_action.items())
        },
        "all_collisions_clean": all(bool(r["metrics"]["clean_collision"]) for r in rows),
        "all_robot_target_contacts_absent": all(not r["metrics"]["robot_target_contact_frames"] for r in rows),
        "action_sha256_by_action": {str(k): v for k, v in sorted(expected_action_hashes.items())},
    }
    write_json(output_root / "summary.json", summary)
    autosave()
    print(json.dumps(to_jsonable(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()
