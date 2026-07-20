#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import datetime as dt
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
from generate_libero_push_box_adaptation_dataset import build_case, write_geometry_bddl  # noqa: E402
from ttt4dynamics.push_box_libero import LiberoPushBoxEnv  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs" / "libero_push_box_70fric_9action_fixed_scene_hai-machine.json"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "pushbox" / "libero_push_box_70fric_9action_fixed_scene_hidden_lerobot_hai-machine_2026-07-01"
DEFAULT_CALIBRATION = REPO_ROOT / "tmp" / "libero_push_box_70fric_9action_fixed_scene_calibration_2026-07-01"
VIDEO_CODEC = "h264"
VIDEO_CRF = 18
JPEG_QUALITY = 98
CONTACT_MOVE_M = 0.001
CONTACT_SPEED_MPS = 0.03
MONOTONIC_TOL_CM = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect/calibrate 70-friction x 9-action fixed-scene hidden LeRobot push-box data.")
    parser.add_argument("--mode", choices=["calibrate", "collect", "both"], default="calibrate")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--calibration-root", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--video-codec", default=VIDEO_CODEC, choices=["h264", "hevc", "libsvtav1", "h264_nvenc"])
    parser.add_argument("--video-crf", type=int, default=VIDEO_CRF)
    parser.add_argument("--jpeg-quality", type=int, default=JPEG_QUALITY)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["frictions"] = np.linspace(float(data["friction_min"]), float(data["friction_max"]), int(data["friction_count"])).astype(float).tolist()
    return data


def mu_tag(mu: float) -> str:
    return f"mu{int(round(float(mu) * 10000)):04d}"


def profile_for_steps(push_steps: int) -> np.ndarray:
    n = int(push_steps)
    if n < 4:
        raise ValueError(f"push_steps must be >= 4, got {n}")
    return np.asarray([0.5] + [1.0] * (n - 3) + [0.5, 0.0], dtype=np.float64)


def patched_impulse_action(self: LiberoPushBoxEnv, eef: np.ndarray, target: np.ndarray) -> np.ndarray:
    action = np.zeros(7, dtype=np.float64)
    push_start = int(self.case.pusher_approach_steps) + int(self.case.pusher_descend_steps)
    local = max(0, int(self.step_count) - push_start)
    amp = float(self.case.pusher_push_action_end)
    profile = np.asarray(getattr(self.case, "hai_action_profile"), dtype=np.float64)
    action[0] = amp * (float(profile[local]) if local < len(profile) else 0.0)
    yz_delta = target[1:3] - eef[1:3]
    yz = float(self.case.pusher_push_yz_hold_gain) * yz_delta
    yz_limit = float(self.case.pusher_push_yz_max_action)
    action[1:3] = np.clip(yz, -yz_limit, yz_limit)
    action[:3] = np.clip(action[:3], -float(self.case.pusher_max_pos_action), float(self.case.pusher_max_pos_action))
    action[-1] = float(self.case.pusher_gripper)
    return action


LiberoPushBoxEnv._impulse_push_action = patched_impulse_action  # type: ignore[method-assign]


def fixed_scene_target_xy(config: dict[str, Any]) -> tuple[float, float]:
    init_xy = tuple(float(v) for v in config["init_xy"])
    return (init_xy[0] + float(config["dummy_target_distance"]), init_xy[1])


def build_fixed_case(
    config: dict[str, Any],
    *,
    mu: float,
    action_cfg: dict[str, Any],
    case_id: str,
    bddl_file: str,
    camera_resolution: int,
) -> Any:
    target_xy = fixed_scene_target_xy(config)
    init_xy = tuple(float(v) for v in config["init_xy"])
    prepare = config["prepare_config"]
    push_steps = int(action_cfg["push_steps"])
    max_steps = int(prepare["approach_steps"]) + int(prepare["descend_steps"]) + push_steps + int(prepare["retreat_steps"]) + int(prepare["settle_steps"])
    base = build_case(
        case_id=case_id,
        domain="70fric_9action_fixed_scene_hidden",
        friction_group=mu_tag(float(mu)),
        friction_mu=float(mu),
        geometry_id=case_id,
        init_xy=init_xy,
        target_distance=float(np.linalg.norm(np.asarray(target_xy) - np.asarray(init_xy))),
        bddl_file=bddl_file,
        target_radius=float(config["target_radius"]),
        push_distance_x=0.14,
        max_steps=max_steps,
        camera_resolution=int(camera_resolution),
    )
    case = replace(
        base,
        pusher_approach_steps=int(prepare["approach_steps"]),
        pusher_descend_steps=int(prepare["descend_steps"]),
        pusher_push_steps=push_steps,
        pusher_retreat_steps=int(prepare["retreat_steps"]),
        pusher_settle_steps=int(prepare["settle_steps"]),
        pusher_contact_offset_xy=(float(prepare["contact_offset_x"]), 0.0),
        pusher_push_mode="impulse",
        pusher_push_action_end=float(action_cfg["A"]),
        pusher_push_controller_scale=10.0,
        pusher_max_push_controller_scale=20.0,
        pusher_push_controller_scale_ramp_steps=2,
        pusher_push_action_delta=1.0,
        pusher_max_pos_action=1.0,
        pusher_prepare_position_gain=float(prepare["prepare_position_gain"]),
        pusher_prepare_max_pos_action=float(prepare["prepare_max_pos_action"]),
        pusher_prepare_action_delta=float(prepare["prepare_action_delta"]),
        controller_output_scale=1.0,
        enable_controller_output_scaling=False,
        target_xy=target_xy,
    )
    object.__setattr__(case, "hai_action_id", int(action_cfg["action_id"]))
    object.__setattr__(case, "hai_action_profile", profile_for_steps(push_steps).astype(float).tolist())
    return case


def write_hidden_bddl(config: dict[str, Any], *, bddl_dir: Path, geometry_id: str) -> str:
    init_xy = tuple(float(v) for v in config["init_xy"])
    target_xy = fixed_scene_target_xy(config)
    return write_geometry_bddl(
        repo_root=REPO_ROOT,
        bddl_dir=bddl_dir,
        geometry_id=geometry_id,
        init_xy=init_xy,
        target_xy=target_xy,
        init_half_size=0.002,
        target_radius=float(config["target_radius"]),
        target_rgba=(0.0, 0.0, 0.0, 0.0),
    )


def preposition_fixed_start(env: LiberoPushBoxEnv) -> tuple[np.ndarray, np.ndarray]:
    obs = env._last_obs if env._last_obs is not None else env._refresh_obs()
    reset_eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    # Keep the same fixed robot start used during previous calibration runs; this just lets the reset settle.
    for _ in range(30):
        obs = env._last_obs if env._last_obs is not None else env._refresh_obs()
        eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        action = env._cartesian_action(eef, reset_eef, float(env.case.pusher_gripper), max_action=0.35, position_gain=4.0)
        obs, _, _, _ = env.step(action)
    return reset_eef, np.asarray(env._last_obs["robot0_eef_pos"], dtype=np.float64)


def compute_metrics(rows: list[dict[str, Any]], *, initial_xyz: np.ndarray, reset_eef: np.ndarray, start_eef: np.ndarray, case: Any) -> dict[str, Any]:
    final_xyz = np.asarray(rows[-1]["box_xyz"], dtype=np.float64)
    initial_xy = np.asarray(initial_xyz[:2], dtype=np.float64)
    final_xy = final_xyz[:2]
    push_start = int(case.pusher_approach_steps) + int(case.pusher_descend_steps)
    profile = np.asarray(getattr(case, "hai_action_profile"), dtype=np.float64)
    first_contact_local = None
    first_contact_action = None
    first_contact_profile = None
    min_gap_cm = None
    push_eef_x = []
    for row in rows:
        if row["phase"] != "push":
            continue
        local = int(row["frame_index"]) - push_start
        box_xyz = np.asarray(row["box_xyz"], dtype=np.float64)
        eef_xyz = np.asarray(row["eef_xyz"], dtype=np.float64)
        gap_cm = float((box_xyz[0] - eef_xyz[0]) * 100.0)
        min_gap_cm = gap_cm if min_gap_cm is None else min(min_gap_cm, gap_cm)
        push_eef_x.append(float(eef_xyz[0]))
        moved = float(box_xyz[0] - initial_xyz[0]) > CONTACT_MOVE_M
        fast = float(np.linalg.norm(row["box_vxy"])) > CONTACT_SPEED_MPS
        if first_contact_local is None and (moved or fast):
            first_contact_local = int(local)
            first_contact_action = float(row["action"][0])
            first_contact_profile = float(profile[local]) if 0 <= local < len(profile) else None
    eef_dx = np.diff(np.asarray(push_eef_x, dtype=np.float64)) if len(push_eef_x) > 1 else np.zeros(0, dtype=np.float64)
    return {
        "reset_eef_xyz": reset_eef.astype(float).tolist(),
        "recorded_start_eef_xyz": start_eef.astype(float).tolist(),
        "initial_xy": initial_xy.astype(float).tolist(),
        "final_xy": final_xy.astype(float).tolist(),
        "final_displacement_m": float(np.linalg.norm(final_xy - initial_xy)),
        "final_forward_m": float(final_xy[0] - initial_xy[0]),
        "lateral_m": float(final_xy[1] - initial_xy[1]),
        "first_contact_local": first_contact_local,
        "first_contact_action_x": first_contact_action,
        "first_contact_profile": first_contact_profile,
        "contact_at_peak": bool(first_contact_profile == 1.0),
        "min_push_gap_cm": min_gap_cm,
        "push_eef_backward_steps": int(np.sum(eef_dx < -1e-4)),
        "push_backward_action_count": int(sum(1 for row in rows if row["phase"] == "push" and row["action"][0] < -1e-6)),
    }


def probe_case(case: Any, *, seed: int) -> dict[str, Any]:
    env = LiberoPushBoxEnv(case, repo_root=REPO_ROOT, seed=seed)
    rows: list[dict[str, Any]] = []
    try:
        env.reset()
        reset_eef, start_eef = preposition_fixed_start(env)
        env.step_count = 0
        env._last_scripted_action = np.zeros(7, dtype=np.float64)
        env._last_scripted_phase = None
        initial_xyz, _ = env.box_pose()
        for _ in range(int(case.max_steps)):
            _, _, _, info = env.step()
            row = dict(info["push_box"])
            row["frame_index"] = len(rows)
            rows.append(row)
    finally:
        env.close()
    return compute_metrics(rows, initial_xyz=initial_xyz, reset_eef=reset_eef, start_eef=start_eef, case=case)


def calibrate(config: dict[str, Any], *, calibration_root: Path, seed: int) -> dict[str, Any]:
    if calibration_root.exists():
        shutil.rmtree(calibration_root)
    calibration_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    actions = list(config["actions"])
    for mu_index, mu in enumerate(config["frictions"]):
        dists_cm = []
        action_rows = []
        for action_cfg in actions:
            action_id = int(action_cfg["action_id"])
            case_id = f"calib_m{mu_index:02d}_{mu_tag(mu)}_a{action_id:02d}"
            bddl = write_hidden_bddl(config, bddl_dir=calibration_root / "bddl", geometry_id=case_id)
            case = build_fixed_case(config, mu=mu, action_cfg=action_cfg, case_id=case_id, bddl_file=bddl, camera_resolution=int(config["probe_resolution"]))
            metrics = probe_case(case, seed=seed)
            dist_cm = float(metrics["final_displacement_m"]) * 100.0
            row = {
                "mu_index": int(mu_index),
                "mu": float(mu),
                "mu_tag": mu_tag(float(mu)),
                "action_id": action_id,
                "A": float(action_cfg["A"]),
                "push_steps": int(action_cfg["push_steps"]),
                "profile": profile_for_steps(int(action_cfg["push_steps"])).astype(float).tolist(),
                "profile_area": float(profile_for_steps(int(action_cfg["push_steps"])).sum() * float(action_cfg["A"])),
                "metrics": metrics,
                "dist_cm": dist_cm,
            }
            rows.append(row)
            action_rows.append(row)
            dists_cm.append(dist_cm)
        violations = []
        for i in range(len(dists_cm) - 1):
            if dists_cm[i + 1] + MONOTONIC_TOL_CM < dists_cm[i]:
                violations.append({
                    "from_action": i,
                    "to_action": i + 1,
                    "from_cm": dists_cm[i],
                    "to_cm": dists_cm[i + 1],
                    "drop_cm": dists_cm[i] - dists_cm[i + 1],
                })
        report = {
            "mu_index": int(mu_index),
            "mu": float(mu),
            "mu_tag": mu_tag(float(mu)),
            "dist_cm_by_action": dists_cm,
            "monotonic_tol_cm": MONOTONIC_TOL_CM,
            "monotonic_with_tol": not violations,
            "violations": violations,
            "contact_at_peak_by_action": [bool(r["metrics"]["contact_at_peak"]) for r in action_rows],
        }
        reports.append(report)
        status = "OK" if not violations else "VIOLATION"
        print(f"calib {mu_index + 1:02d}/{len(config['frictions']):02d} {mu_tag(mu)} {status} dists={[round(x, 1) for x in dists_cm]}", flush=True)
        write_json(calibration_root / "calibration_summary.json", {
            "config": config,
            "rows": rows,
            "reports": reports,
            "violation_count": int(sum(len(r["violations"]) for r in reports)),
        })
    result = {
        "config": config,
        "rows": rows,
        "reports": reports,
        "violation_count": int(sum(len(r["violations"]) for r in reports)),
        "violating_mu_count": int(sum(1 for r in reports if r["violations"])),
    }
    write_json(calibration_root / "calibration_summary.json", result)
    return result


def write_image_for_last_frame(dataset: LeRobotDataset, key: str, frame_index: int, image: np.ndarray, *, jpeg_quality: int) -> None:
    path = dataset._get_image_file_path(
        episode_index=dataset.episode_buffer["episode_index"],
        image_key=key,
        frame_index=frame_index,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path, quality=int(jpeg_quality))


def remove_current_episode_images(dataset: LeRobotDataset) -> None:
    if dataset.episode_buffer is None:
        return
    episode_index = dataset.episode_buffer["episode_index"]
    for key in dataset.meta.video_keys:
        image_dir = dataset._get_image_file_path(episode_index=episode_index, image_key=key, frame_index=0).parent
        if image_dir.is_dir():
            shutil.rmtree(image_dir)


def rollout_to_lerobot(case: Any, *, dataset: LeRobotDataset, seed: int, fps: int, jpeg_quality: int) -> tuple[int, dict[str, Any]]:
    env = LiberoPushBoxEnv(case, repo_root=REPO_ROOT, seed=seed)
    rows: list[dict[str, Any]] = []
    phase_counts: Counter[str] = Counter()
    try:
        obs = env.reset()
        reset_eef, start_eef = preposition_fixed_start(env)
        env.step_count = 0
        env._last_scripted_action = np.zeros(7, dtype=np.float64)
        env._last_scripted_phase = None
        obs = env._last_obs
        initial_xyz, _ = env.box_pose()
        remove_current_episode_images(dataset)
        episode_index = int(dataset.meta.total_episodes)
        task = prompt_for_case("observation", "straight")
        for frame_idx in range(int(case.max_steps)):
            obs_for_frame = obs
            obs, _, _, info = env.step()
            row = dict(info["push_box"])
            row["frame_index"] = len(rows)
            rows.append(row)
            phase_counts[str(row.get("phase", "unknown"))] += 1
            agent, wrist = _obs_to_images(obs_for_frame)
            action = np.asarray(row["action"], dtype=np.float32)
            frame = {
                "observation.images.image": agent,
                "observation.images.wrist_image": wrist,
                "observation.state": _obs_to_state(obs_for_frame),
                "action": _env_action_to_fastwam_action(action),
            }
            dataset.add_frame(frame, task=task, timestamp=float(frame_idx) / float(fps))
            write_image_for_last_frame(dataset, "observation.images.image", frame_idx, agent, jpeg_quality=jpeg_quality)
            write_image_for_last_frame(dataset, "observation.images.wrist_image", frame_idx, wrist, jpeg_quality=jpeg_quality)
        dataset.save_episode()
    finally:
        env.close()
    metrics = compute_metrics(rows, initial_xyz=initial_xyz, reset_eef=reset_eef, start_eef=start_eef, case=case)
    metrics["phase_counts"] = dict(phase_counts)
    metrics["steps"] = len(rows)
    return episode_index, metrics


def create_dataset(root: Path, *, config: dict[str, Any], video_codec: str) -> LeRobotDataset:
    return LeRobotDataset.create(
        repo_id="libero_push_box_70fric_9action_fixed_scene_hidden_hai_machine",
        root=root,
        fps=int(config["fps"]),
        features=build_features(int(config["camera_resolution"])),
        use_videos=True,
        video_codec=video_codec,
        is_compute_episode_stats_image=False,
    )


def collect(config: dict[str, Any], *, output_root: Path, overwrite: bool, seed: int, video_codec: str, video_crf: int, jpeg_quality: int) -> dict[str, Any]:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} exists; pass --overwrite")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    patch_lerobot_video_crf(video_crf)
    dataset_root = output_root / "hidden_straight_lerobot"
    dataset = create_dataset(dataset_root, config=config, video_codec=video_codec)
    rows: list[dict[str, Any]] = []
    metadata = {
        "created_at": dt.datetime.now().isoformat(),
        "dataset_type": "libero_push_box_70fric_9action_fixed_scene_hidden_lerobot_hai-machine",
        "target_visible": False,
        "split": "straight",
        "camera_resolution": int(config["camera_resolution"]),
        "fps": int(config["fps"]),
        "video_codec": str(video_codec),
        "video_crf": int(video_crf),
        "jpeg_quality": int(jpeg_quality),
        "state_source": "true LIBERO obs robot0_eef_pos, robot0_eef_quat converted to axis-angle, robot0_gripper_qpos",
        "episodes": [],
    }
    manifest = {
        "created_at": dt.datetime.now().isoformat(),
        "dataset_type": "libero_push_box_70fric_9action_fixed_scene_hidden_lerobot_collection_hai-machine",
        "output_root": str(output_root),
        "hidden_straight_lerobot": str(dataset_root),
        "config": config,
        "episodes": [],
    }

    def autosave() -> None:
        write_json(output_root / "manifest.json", manifest)
        write_dataset_metadata(dataset_root, metadata, rows)

    total = len(config["frictions"]) * len(config["actions"])
    count = 0
    for mu_index, mu in enumerate(config["frictions"]):
        for action_cfg in config["actions"]:
            action_id = int(action_cfg["action_id"])
            case_id = f"m{mu_index:02d}_{mu_tag(mu)}_a{action_id:02d}_A{int(round(float(action_cfg['A']) * 1000)):03d}_n{int(action_cfg['push_steps']):02d}"
            bddl = write_hidden_bddl(config, bddl_dir=output_root / "bddl", geometry_id=case_id)
            case = build_fixed_case(config, mu=mu, action_cfg=action_cfg, case_id=case_id, bddl_file=bddl, camera_resolution=int(config["camera_resolution"]))
            episode_index, metrics = rollout_to_lerobot(case, dataset=dataset, seed=seed, fps=int(config["fps"]), jpeg_quality=jpeg_quality)
            row = {
                "episode_index": int(episode_index),
                "case_id": case_id,
                "mu_index": int(mu_index),
                "mu": float(mu),
                "mu_tag": mu_tag(float(mu)),
                "action_id": action_id,
                "A": float(action_cfg["A"]),
                "push_steps": int(action_cfg["push_steps"]),
                "profile": profile_for_steps(int(action_cfg["push_steps"])).astype(float).tolist(),
                "profile_area": float(profile_for_steps(int(action_cfg["push_steps"])).sum() * float(action_cfg["A"])),
                "init_xy": [float(v) for v in config["init_xy"]],
                "target_xy": list(fixed_scene_target_xy(config)),
                "bddl_file": bddl,
                "metrics": metrics,
            }
            rows.append(row)
            metadata["episodes"].append(row)
            manifest["episodes"].append(row)
            count += 1
            print(f"collect {count:03d}/{total:03d} {case_id} disp={metrics['final_displacement_m'] * 100:.1f}cm", flush=True)
            autosave()
    summary_counts = Counter(row["mu_tag"] for row in rows)
    summary = {
        "episode_count": len(rows),
        "expected_episode_count": total,
        "hidden_straight_lerobot": str(dataset_root),
        "count_by_mu": dict(sorted(summary_counts.items())),
    }
    write_json(output_root / "summary.json", summary)
    autosave()
    print(f"manifest={output_root / 'manifest.json'}", flush=True)
    print(f"hidden_root={dataset_root}", flush=True)
    return summary


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(value), indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    if args.mode in {"calibrate", "both"}:
        result = calibrate(config, calibration_root=args.calibration_root.resolve(), seed=int(args.seed))
        print(json.dumps({"calibration_root": str(args.calibration_root.resolve()), "violation_count": result["violation_count"], "violating_mu_count": result["violating_mu_count"]}, indent=2), flush=True)
    if args.mode in {"collect", "both"}:
        summary = collect(
            config,
            output_root=args.output_root.resolve(),
            overwrite=bool(args.overwrite),
            seed=int(args.seed),
            video_codec=str(args.video_codec),
            video_crf=int(args.video_crf),
            jpeg_quality=int(args.jpeg_quality),
        )
        print(json.dumps(to_jsonable(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()
