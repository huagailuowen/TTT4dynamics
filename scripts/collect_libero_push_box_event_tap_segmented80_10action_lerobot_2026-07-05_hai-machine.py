#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = REPO_ROOT / "scripts" / "collect_libero_push_box_70fric_9action_fixed_scene_hidden_lerobot_hai-machine.py"
CONFIG_PATH = REPO_ROOT / "configs" / "libero_push_box_70fric_9action_fixed_scene_highforce_2026-07-02_hai-machine.json"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "pushbox" / "libero_push_box_event_tap_segmented80_10action_hidden_lerobot_A500_offset160_stop_2026-07-05_hai-machine"

VIDEO_CODEC = "h264"
VIDEO_CRF = 18
JPEG_QUALITY = 98
CONTACT_MOVE_M = 0.001
CONTACT_SPEED_MPS = 0.03
CONTACT_OFFSET_X = -0.160
TRIGGER_VX_RATIO = 2.2
MAX_CONTACT_HOLD = 3
HOLD_AFTER_CONTACT = 0

MID_DENSE_ACTIONS = [
    {"action_id": 0, "A": 0.080, "push_steps": 16, "description": "lowest anchor that still reaches from 0.16m offset"},
    {"action_id": 1, "A": 0.110, "push_steps": 12, "description": "mid-low"},
    {"action_id": 2, "A": 0.140, "push_steps": 10, "description": "mid-low dense"},
    {"action_id": 3, "A": 0.170, "push_steps": 10, "description": "mid dense"},
    {"action_id": 4, "A": 0.210, "push_steps": 10, "description": "mid dense"},
    {"action_id": 5, "A": 0.260, "push_steps": 10, "description": "mid-high dense"},
    {"action_id": 6, "A": 0.320, "push_steps": 10, "description": "mid-high dense"},
    {"action_id": 7, "A": 0.340, "push_steps": 10, "description": "clean high bridge; targeted scan showed no local upward bump"},
    {"action_id": 8, "A": 0.360, "push_steps": 10, "description": "clean high; replaces unstable A=0.43"},
    {"action_id": 9, "A": 0.500, "push_steps": 10, "description": "maximum stable high-force action"},
]


def load_base_module() -> Any:
    spec = importlib.util.spec_from_file_location("collect70_event_tap_base_hai_machine", BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


base = load_base_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect current event-triggered segmented-80 push-box LeRobot dataset.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video-codec", default=VIDEO_CODEC, choices=["h264", "hevc", "libsvtav1", "h264_nvenc"])
    parser.add_argument("--video-crf", type=int, default=VIDEO_CRF)
    parser.add_argument("--jpeg-quality", type=int, default=JPEG_QUALITY)
    return parser.parse_args()


def configure_current_dataset(config: dict[str, Any]) -> dict[str, Any]:
    config = dict(config)
    prepare = dict(config["prepare_config"])
    prepare["contact_offset_x"] = CONTACT_OFFSET_X
    config["prepare_config"] = prepare
    seg1 = np.linspace(0.002, 0.05, 30, endpoint=False)
    seg2 = np.linspace(0.05, 0.15, 40, endpoint=False)
    seg3 = np.linspace(0.15, 0.20, 10, endpoint=True)
    config["frictions"] = np.concatenate([seg1, seg2, seg3]).astype(float).tolist()
    config["friction_count"] = len(config["frictions"])
    config["friction_schedule"] = "segmented_80: [0.002,0.05)x30 + [0.05,0.15)x40 + [0.15,0.20]x10"
    config["actions"] = [dict(a) for a in MID_DENSE_ACTIONS]
    config["action_count"] = len(config["actions"])
    config["dataset_name"] = "libero_push_box_event_tap_segmented80_10action_hidden_lerobot_A500_offset160_stop_hai-machine"
    config["event_tap"] = {
        "contact_offset_x": CONTACT_OFFSET_X,
        "contact_move_m": CONTACT_MOVE_M,
        "contact_speed_mps": CONTACT_SPEED_MPS,
        "hold_after_contact": HOLD_AFTER_CONTACT,
        "trigger_vx_ratio": TRIGGER_VX_RATIO,
        "max_contact_hold": MAX_CONTACT_HOLD,
        "after_mode": "stop",
    }
    return config


def brake_action(case: Any) -> np.ndarray:
    action = np.zeros(7, dtype=np.float64)
    action[-1] = float(case.pusher_gripper)
    return action


def rollout_to_lerobot_event_tap(case: Any, *, dataset: Any, seed: int, fps: int, jpeg_quality: int) -> tuple[int, dict[str, Any]]:
    env = base.LiberoPushBoxEnv(case, repo_root=REPO_ROOT, seed=seed)
    rows: list[dict[str, Any]] = []
    phase_counts: Counter[str] = Counter()
    contact_frame = None
    forced_stop = False
    forced_stop_frame = None
    try:
        obs = env.reset()
        reset_eef, start_eef = base.preposition_fixed_start(env)
        env.step_count = 0
        env._last_scripted_action = np.zeros(7, dtype=np.float64)
        env._last_scripted_phase = None
        obs = env._last_obs
        initial_xyz, _ = env.box_pose()
        init_x = float(initial_xyz[0])
        base.remove_current_episode_images(dataset)
        episode_index = int(dataset.meta.total_episodes)
        task = base.prompt_for_case("observation", "straight")
        for frame_idx in range(int(case.max_steps)):
            obs_for_frame = obs
            if forced_stop:
                obs, _, _, info = env.step(brake_action(case))
                info["push_box"]["phase"] = "event_stop"
            else:
                obs, _, _, info = env.step()
            row = dict(info["push_box"])
            row["frame_index"] = len(rows)
            rows.append(row)
            phase_counts[str(row.get("phase", "unknown"))] += 1

            agent, wrist = base._obs_to_images(obs_for_frame)
            action = np.asarray(row["action"], dtype=np.float32)
            frame = {
                "observation.images.image": agent,
                "observation.images.wrist_image": wrist,
                "observation.state": base._obs_to_state(obs_for_frame),
                "action": base._env_action_to_fastwam_action(action),
            }
            dataset.add_frame(frame, task=task, timestamp=float(frame_idx) / float(fps))
            base.write_image_for_last_frame(dataset, "observation.images.image", frame_idx, agent, jpeg_quality=jpeg_quality)
            base.write_image_for_last_frame(dataset, "observation.images.wrist_image", frame_idx, wrist, jpeg_quality=jpeg_quality)

            box_x = float(row["box_xyz"][0])
            vx = float(row["box_vxy"][0])
            if contact_frame is None and ((box_x - init_x) > CONTACT_MOVE_M or abs(vx) > CONTACT_SPEED_MPS):
                contact_frame = len(rows) - 1
            if contact_frame is not None and not forced_stop:
                frames_after_contact = len(rows) - 1 - int(contact_frame)
                target_vx = max(CONTACT_SPEED_MPS, TRIGGER_VX_RATIO * float(case.pusher_push_action_end))
                vx_ready = abs(vx) >= target_vx
                timeout_ready = frames_after_contact >= MAX_CONTACT_HOLD
                min_hold_ready = frames_after_contact >= HOLD_AFTER_CONTACT
                if min_hold_ready and (vx_ready or timeout_ready):
                    forced_stop = True
                    forced_stop_frame = len(rows)
        dataset.save_episode()
    finally:
        env.close()

    metrics = base.compute_metrics(rows, initial_xyz=initial_xyz, reset_eef=reset_eef, start_eef=start_eef, case=case)
    phases = [str(r.get("phase", "")) for r in rows]
    vxs = np.asarray([float(r["box_vxy"][0]) for r in rows], dtype=np.float64)
    push_start = int(case.pusher_approach_steps) + int(case.pusher_descend_steps)
    peak_i = int(np.argmax(vxs)) if len(vxs) else -1
    metrics.update(
        {
            "phase_counts": dict(phase_counts),
            "steps": len(rows),
            "contact_frame": contact_frame,
            "contact_local": None if contact_frame is None else int(contact_frame - push_start),
            "peak_vx": float(vxs[peak_i]) if peak_i >= 0 else None,
            "peak_local": int(peak_i - push_start) if peak_i >= 0 else None,
            "forced_stop_frame": forced_stop_frame,
            "event_stop_frames": int(sum(1 for phase in phases if phase == "event_stop")),
        }
    )
    return episode_index, metrics


def create_dataset(root: Path, *, config: dict[str, Any], video_codec: str) -> Any:
    return base.LeRobotDataset.create(
        repo_id="libero_push_box_event_tap_segmented80_10action_hidden_hai_machine",
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


def collect(config: dict[str, Any], *, output_root: Path, overwrite: bool, seed: int, video_codec: str, video_crf: int, jpeg_quality: int) -> dict[str, Any]:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} exists; pass --overwrite")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    base.patch_lerobot_video_crf(video_crf)
    dataset_root = output_root / "hidden_straight_lerobot"
    dataset = create_dataset(dataset_root, config=config, video_codec=video_codec)
    rows: list[dict[str, Any]] = []
    metadata = {
        "created_at": dt.datetime.now().isoformat(),
        "dataset_type": "libero_push_box_event_tap_segmented80_10action_hidden_lerobot_hai-machine",
        "target_visible": False,
        "split": "straight",
        "camera_resolution": int(config["camera_resolution"]),
        "fps": int(config["fps"]),
        "video_codec": str(video_codec),
        "video_crf": int(video_crf),
        "jpeg_quality": int(jpeg_quality),
        "state_source": "true LIBERO obs robot0_eef_pos, robot0_eef_quat converted to axis-angle, robot0_gripper_qpos",
        "event_tap": dict(config["event_tap"]),
        "episodes": [],
    }
    manifest = {
        "created_at": dt.datetime.now().isoformat(),
        "dataset_type": "libero_push_box_event_tap_segmented80_10action_hidden_lerobot_collection_hai-machine",
        "output_root": str(output_root),
        "hidden_straight_lerobot": str(dataset_root),
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
            case_id = f"event_m{mu_index:02d}_{base.mu_tag(mu)}_a{action_id:02d}_A{int(round(float(action_cfg['A']) * 1000)):03d}_n{int(action_cfg['push_steps']):02d}"
            bddl = base.write_hidden_bddl(config, bddl_dir=output_root / "bddl", geometry_id=case_id)
            case = base.build_fixed_case(
                config,
                mu=float(mu),
                action_cfg=action_cfg,
                case_id=case_id,
                bddl_file=bddl,
                camera_resolution=int(config["camera_resolution"]),
            )
            episode_index, metrics = rollout_to_lerobot_event_tap(case, dataset=dataset, seed=seed, fps=int(config["fps"]), jpeg_quality=jpeg_quality)
            profile = base.profile_for_steps(int(action_cfg["push_steps"])).astype(float).tolist()
            row = {
                "episode_index": int(episode_index),
                "case_id": case_id,
                "mu_index": int(mu_index),
                "mu": float(mu),
                "mu_tag": base.mu_tag(float(mu)),
                "action_id": action_id,
                "A": float(action_cfg["A"]),
                "push_steps": int(action_cfg["push_steps"]),
                "profile": profile,
                "profile_area": float(sum(profile) * float(action_cfg["A"])),
                "init_xy": [float(v) for v in config["init_xy"]],
                "target_xy": list(base.fixed_scene_target_xy(config)),
                "bddl_file": bddl,
                "event_tap": dict(config["event_tap"]),
                "metrics": metrics,
            }
            rows.append(row)
            metadata["episodes"].append(row)
            manifest["episodes"].append(row)
            count += 1
            print(
                f"collect {count:03d}/{total:03d} {case_id} "
                f"disp={metrics['final_displacement_m'] * 100:.1f}cm "
                f"peak_vx={metrics.get('peak_vx')} contact={metrics.get('contact_local')} stop={metrics.get('forced_stop_frame')}",
                flush=True,
            )
            autosave()

    summary_counts = Counter(row["mu_tag"] for row in rows)
    summary = {
        "episode_count": len(rows),
        "expected_episode_count": total,
        "hidden_straight_lerobot": str(dataset_root),
        "count_by_mu": dict(sorted(summary_counts.items())),
        "event_tap": dict(config["event_tap"]),
    }
    write_json(output_root / "summary.json", summary)
    autosave()
    print(f"manifest={output_root / 'manifest.json'}", flush=True)
    print(f"hidden_root={dataset_root}", flush=True)
    return summary


def main() -> None:
    args = parse_args()
    config = base.load_config(CONFIG_PATH)
    config = configure_current_dataset(config)
    summary = collect(
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
