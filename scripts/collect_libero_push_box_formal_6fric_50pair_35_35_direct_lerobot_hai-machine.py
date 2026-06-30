#!/usr/bin/env python3
from __future__ import annotations

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

OUTPUT_ROOT = REPO_ROOT / "data" / "libero_push_box_6fric_50pairs_straight_jitter_35_35_direct_lerobot_hai-machine_2026-06-29"
FPS = 20
CAMERA_RESOLUTION = 224
VIDEO_CODEC = "h264"
VIDEO_CRF = 18
JPEG_QUALITY = 98
CONTACT_MOVE_M = 0.001
CONTACT_SPEED_MPS = 0.03
FAST_PROFILE = np.asarray([0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0], dtype=np.float64)
INIT_XY = (-0.245, -0.035)
DUMMY_TARGET_DISTANCE = 0.80
TARGET_RADIUS = 0.025
BASE_SEED = 20260629
JITTER_X_RANGE = (-0.025, 0.025)
JITTER_Y_RANGE = (-0.030, 0.030)
JITTER_PREPOSITION_STEPS = 30
MAX_ATTEMPTS_PER_MU = 1800

FRICTIONS = [0.005, 0.010, 0.020, 0.050, 0.100, 0.150]
BUCKET_ORDER = ["short", "mid", "long"]
BUCKET_QUOTA = {"short": 17, "mid": 17, "long": 16}
BUCKET_RANGES_M = {
    "short": (0.150, 0.280),
    "mid": (0.280, 0.400),
    "long": (0.400, 0.560),
}
QUALITY = {
    "max_final_displacement_m": 0.600,
    "max_abs_lateral_m": 0.040,
    "max_push_eef_backward_steps": 0,
}
PREPARE_CONFIG = {
    "approach_steps": 35,
    "descend_steps": 35,
    "prepare_position_gain": 7.0,
    "prepare_max_pos_action": 0.70,
    "prepare_action_delta": 0.14,
    "contact_offset_x": -0.115,
}
SAMPLER = {
    0.005: {
        "mode": "event_hold",
        "short": {"A": (0.042, 0.062), "hold": [3]},
        "mid": {"A": (0.064, 0.078), "hold": [3, 4]},
        "long": {"A": (0.078, 0.100), "hold": [4]},
    },
    0.010: {
        "mode": "event_hold",
        "short": {"A": (0.060, 0.085), "hold": [3]},
        "mid": {"A": (0.090, 0.115), "hold": [3]},
        "long": {"A": (0.120, 0.148), "hold": [3]},
    },
    0.020: {
        "mode": "event_hold",
        "short": {"A": (0.082, 0.112), "hold": [4]},
        "mid": {"A": (0.125, 0.155), "hold": [4]},
        "long": {"A": (0.165, 0.220), "hold": [4]},
    },
    0.050: {
        "mode": "event_hold",
        "short": {"A": (0.135, 0.190), "hold": [3]},
        "mid": {"A": (0.200, 0.250), "hold": [3]},
        "long": {"A": (0.255, 0.315), "hold": [3]},
    },
    0.100: {
        "mode": "fixed_fast",
        "short": {"A": (0.185, 0.250), "hold": [0]},
        "mid": {"A": (0.280, 0.355), "hold": [0]},
        "long": {"A": (0.370, 0.470), "hold": [0]},
    },
    0.150: {
        "mode": "fixed_fast",
        "short": {"A": (0.200, 0.285), "hold": [0]},
        "mid": {"A": (0.320, 0.390), "hold": [0]},
        "long": {"A": (0.410, 0.510), "hold": [0]},
    },
}


def mu_tag(mu: float) -> str:
    return f"mu{int(round(float(mu) * 10000)):04d}"


def patched_impulse_action(self: LiberoPushBoxEnv, eef: np.ndarray, target: np.ndarray) -> np.ndarray:
    action = np.zeros(7, dtype=np.float64)
    kind = str(getattr(self.case, "hai_profile_kind", "fixed_fast"))
    push_start = int(self.case.pusher_approach_steps) + int(self.case.pusher_descend_steps)
    local = max(0, int(self.step_count) - push_start)
    amp = float(self.case.pusher_push_action_end)

    if kind == "event_hold":
        if not hasattr(self, "_event_contact_seen"):
            self._event_contact_seen = False
            self._event_hold_remaining = int(getattr(self.case, "hai_hold_after_contact", 3))
            self._event_first_contact_local = None
            self._event_stop_local = None
        box_xyz, _ = self.box_pose()
        qvel = self.box_velocity()
        moved = self._initial_box_xyz is not None and float(box_xyz[0] - self._initial_box_xyz[0]) > CONTACT_MOVE_M
        fast = float(np.linalg.norm(qvel[:2])) > CONTACT_SPEED_MPS
        if (not self._event_contact_seen) and (moved or fast):
            self._event_contact_seen = True
            self._event_first_contact_local = int(local)
            self._event_hold_remaining = int(getattr(self.case, "hai_hold_after_contact", 3))
        if local == 0:
            x = 0.5 * amp
        elif not self._event_contact_seen:
            x = amp
        elif self._event_hold_remaining > 0:
            x = amp
            self._event_hold_remaining -= 1
        else:
            x = 0.0
            if self._event_stop_local is None:
                self._event_stop_local = int(local)
    else:
        x = amp * (float(FAST_PROFILE[local]) if local < len(FAST_PROFILE) else 0.0)

    action[0] = x
    yz_delta = target[1:3] - eef[1:3]
    yz = float(self.case.pusher_push_yz_hold_gain) * yz_delta
    yz_limit = float(self.case.pusher_push_yz_max_action)
    action[1:3] = np.clip(yz, -yz_limit, yz_limit)
    action[:3] = np.clip(action[:3], -float(self.case.pusher_max_pos_action), float(self.case.pusher_max_pos_action))
    action[-1] = float(self.case.pusher_gripper)
    return action


LiberoPushBoxEnv._impulse_push_action = patched_impulse_action  # type: ignore[method-assign]


def classify_displacement(distance_m: float) -> str | None:
    for bucket in BUCKET_ORDER:
        lo, hi = BUCKET_RANGES_M[bucket]
        if lo <= distance_m < hi:
            return bucket
    return None


def choose_sampler_bucket(rng: np.random.Generator, counts: dict[str, int]) -> str:
    remaining = np.asarray([max(0, BUCKET_QUOTA[b] - counts.get(b, 0)) for b in BUCKET_ORDER], dtype=np.float64)
    remaining = remaining / remaining.sum()
    return str(rng.choice(BUCKET_ORDER, p=remaining))


def sample_plan(mu: float, rng: np.random.Generator, counts: dict[str, int]) -> dict[str, Any]:
    prior_bucket = choose_sampler_bucket(rng, counts)
    spec = SAMPLER[mu]
    bucket_spec = spec[prior_bucket]
    return {
        "mu": float(mu),
        "mode": str(spec["mode"]),
        "sampler_bucket": prior_bucket,
        "A": float(rng.uniform(*bucket_spec["A"])),
        "hold": int(rng.choice(bucket_spec["hold"])),
    }


def sample_jitter(rng: np.random.Generator) -> tuple[float, float]:
    return (float(rng.uniform(*JITTER_X_RANGE)), float(rng.uniform(*JITTER_Y_RANGE)))


def build_push_case(plan: dict[str, Any], *, case_id: str, bddl_file: str, target_xy: tuple[float, float], camera_resolution: int) -> Any:
    push_steps = 18 if plan["mode"] == "event_hold" else 10
    max_steps = int(PREPARE_CONFIG["approach_steps"]) + int(PREPARE_CONFIG["descend_steps"]) + push_steps + 60 + 100
    base = build_case(
        case_id=case_id,
        domain="formal_direct_lerobot_jitter_35_35",
        friction_group=mu_tag(float(plan["mu"])),
        friction_mu=float(plan["mu"]),
        geometry_id=case_id,
        init_xy=INIT_XY,
        target_distance=float(np.linalg.norm(np.asarray(target_xy) - np.asarray(INIT_XY))),
        bddl_file=bddl_file,
        target_radius=TARGET_RADIUS,
        push_distance_x=0.14,
        max_steps=max_steps,
        camera_resolution=camera_resolution,
    )
    case = replace(
        base,
        pusher_approach_steps=int(PREPARE_CONFIG["approach_steps"]),
        pusher_descend_steps=int(PREPARE_CONFIG["descend_steps"]),
        pusher_push_steps=push_steps,
        pusher_retreat_steps=60,
        pusher_settle_steps=100,
        pusher_contact_offset_xy=(float(PREPARE_CONFIG["contact_offset_x"]), 0.0),
        pusher_push_mode="impulse",
        pusher_push_action_end=float(plan["A"]),
        pusher_push_controller_scale=10.0,
        pusher_max_push_controller_scale=20.0,
        pusher_push_controller_scale_ramp_steps=2,
        pusher_push_action_delta=1.0,
        pusher_max_pos_action=1.0,
        pusher_prepare_position_gain=float(PREPARE_CONFIG["prepare_position_gain"]),
        pusher_prepare_max_pos_action=float(PREPARE_CONFIG["prepare_max_pos_action"]),
        pusher_prepare_action_delta=float(PREPARE_CONFIG["prepare_action_delta"]),
        controller_output_scale=1.0,
        enable_controller_output_scaling=False,
        target_xy=target_xy,
    )
    object.__setattr__(case, "hai_profile_kind", str(plan["mode"]))
    object.__setattr__(case, "hai_hold_after_contact", int(plan.get("hold", 3)))
    return case


def move_to_start(env: LiberoPushBoxEnv, jitter_xy: tuple[float, float], *, steps: int = JITTER_PREPOSITION_STEPS) -> tuple[np.ndarray, np.ndarray]:
    obs = env._last_obs if env._last_obs is not None else env._refresh_obs()
    reset_eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    start_xyz = reset_eef.copy()
    start_xyz[0] += float(jitter_xy[0])
    start_xyz[1] += float(jitter_xy[1])
    for _ in range(int(steps)):
        obs = env._last_obs if env._last_obs is not None else env._refresh_obs()
        eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        action = env._cartesian_action(eef, start_xyz, float(env.case.pusher_gripper), max_action=0.35, position_gain=4.0)
        obs, _, _, _ = env.step(action)
    return reset_eef, np.asarray(env._last_obs["robot0_eef_pos"], dtype=np.float64)


def metrics_from_rows(rows: list[dict[str, Any]], *, initial_xyz: np.ndarray, reset_eef: np.ndarray, start_eef: np.ndarray, jitter_xy: tuple[float, float]) -> dict[str, Any]:
    final_xyz = np.asarray(rows[-1]["box_xyz"], dtype=np.float64)
    initial_xy = np.asarray(initial_xyz[:2], dtype=np.float64)
    final_xy = final_xyz[:2]
    phases = [row["phase"] for row in rows]
    push_idxs = [i for i, p in enumerate(phases) if p == "push"]
    eef_x = np.asarray([rows[i]["eef_xyz"][0] for i in push_idxs], dtype=np.float64)
    eef_dx = np.diff(eef_x) if eef_x.size > 1 else np.zeros(0, dtype=np.float64)
    box = np.asarray([row["box_xyz"] for row in rows], dtype=np.float64)
    first_contact_any = None
    for local, idx in enumerate(push_idxs):
        moved = float(rows[idx]["box_xyz"][0] - initial_xyz[0]) > CONTACT_MOVE_M
        qvel = np.asarray(rows[idx].get("box_qvel", [0, 0, 0, 0, 0, 0]), dtype=np.float64)
        fast = bool(qvel.size >= 2 and np.linalg.norm(qvel[:2]) > CONTACT_SPEED_MPS)
        if moved or fast:
            first_contact_any = int(local)
            break
    return {
        "jitter_xy": [float(jitter_xy[0]), float(jitter_xy[1])],
        "reset_eef_xyz": reset_eef.astype(float).tolist(),
        "recorded_start_eef_xyz": start_eef.astype(float).tolist(),
        "initial_xy": initial_xy.astype(float).tolist(),
        "final_xy": final_xy.astype(float).tolist(),
        "final_displacement_m": float(np.linalg.norm(final_xy - initial_xy)),
        "final_forward_m": float(final_xy[0] - initial_xy[0]),
        "lateral_m": float(final_xy[1] - initial_xy[1]),
        "max_lateral_m": float(np.max(np.abs(box[:, 1] - initial_xy[1]))),
        "push_backward_action_count": int(sum(1 for i in push_idxs if rows[i]["action"][0] < -1e-6)),
        "push_eef_backward_steps": int(np.sum(eef_dx < -1e-4)),
        "first_contact_any_local": first_contact_any,
    }


def probe_rollout(case: Any, *, jitter_xy: tuple[float, float]) -> dict[str, Any]:
    env = LiberoPushBoxEnv(case, repo_root=REPO_ROOT, seed=0)
    rows: list[dict[str, Any]] = []
    try:
        env.reset()
        reset_eef, start_eef = move_to_start(env, jitter_xy)
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
    return metrics_from_rows(rows, initial_xyz=initial_xyz, reset_eef=reset_eef, start_eef=start_eef, jitter_xy=jitter_xy)


def write_image_for_last_frame(dataset: LeRobotDataset, key: str, frame_index: int, image: np.ndarray) -> None:
    path = dataset._get_image_file_path(
        episode_index=dataset.episode_buffer["episode_index"],
        image_key=key,
        frame_index=frame_index,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path, quality=JPEG_QUALITY)


def remove_current_episode_images(dataset: LeRobotDataset) -> None:
    if dataset.episode_buffer is None:
        return
    episode_index = dataset.episode_buffer["episode_index"]
    for key in dataset.meta.video_keys:
        image_dir = dataset._get_image_file_path(episode_index=episode_index, image_key=key, frame_index=0).parent
        if image_dir.is_dir():
            shutil.rmtree(image_dir)


def dataset_rollout(case: Any, *, dataset: LeRobotDataset, domain: str, jitter_xy: tuple[float, float]) -> tuple[int, dict[str, Any]]:
    env = LiberoPushBoxEnv(case, repo_root=REPO_ROOT, seed=0)
    rows: list[dict[str, Any]] = []
    phase_counts: Counter[str] = Counter()
    try:
        obs = env.reset()
        reset_eef, start_eef = move_to_start(env, jitter_xy)
        env.step_count = 0
        env._last_scripted_action = np.zeros(7, dtype=np.float64)
        env._last_scripted_phase = None
        obs = env._last_obs
        initial_xyz, _ = env.box_pose()
        remove_current_episode_images(dataset)
        episode_index = int(dataset.meta.total_episodes)
        for frame_idx in range(int(case.max_steps)):
            obs_for_frame = obs
            obs, _, _, info = env.step()
            row = dict(info["push_box"])
            row["frame_index"] = len(rows)
            rows.append(row)
            phase_counts[str(row.get("phase", "unknown"))] += 1
            action = np.asarray(row["action"], dtype=np.float32)
            agent, wrist = _obs_to_images(obs_for_frame)
            frame = {
                "observation.images.image": agent,
                "observation.images.wrist_image": wrist,
                "observation.state": _obs_to_state(obs_for_frame),
                "action": _env_action_to_fastwam_action(action),
            }
            dataset.add_frame(frame, task=prompt_for_case(domain, "straight"), timestamp=float(frame_idx) / float(FPS))
            write_image_for_last_frame(dataset, "observation.images.image", frame_idx, agent)
            write_image_for_last_frame(dataset, "observation.images.wrist_image", frame_idx, wrist)
        dataset.save_episode()
    finally:
        env.close()
    metrics = metrics_from_rows(rows, initial_xyz=initial_xyz, reset_eef=reset_eef, start_eef=start_eef, jitter_xy=jitter_xy)
    metrics["phase_counts"] = dict(phase_counts)
    metrics["steps"] = len(rows)
    return episode_index, metrics


def quality_ok(metrics: dict[str, Any]) -> bool:
    return (
        float(metrics["final_displacement_m"]) <= float(QUALITY["max_final_displacement_m"])
        and abs(float(metrics["max_lateral_m"])) <= float(QUALITY["max_abs_lateral_m"])
        and int(metrics["push_eef_backward_steps"]) <= int(QUALITY["max_push_eef_backward_steps"])
        and int(metrics["push_backward_action_count"]) == 0
    )


def create_dataset(root: Path, *, repo_id: str) -> LeRobotDataset:
    if root.exists():
        shutil.rmtree(root)
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        fps=FPS,
        features=build_features(CAMERA_RESOLUTION),
        use_videos=True,
        video_codec=VIDEO_CODEC,
        is_compute_episode_stats_image=False,
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(value), indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(to_jsonable(row)) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    patch_lerobot_video_crf(VIDEO_CRF)
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    bddl_root = OUTPUT_ROOT / "bddl"
    datasets = {
        "observation": create_dataset(
            OUTPUT_ROOT / "hidden_straight_lerobot",
            repo_id="libero_push_box_6fric_50pairs_hidden_straight_jitter_35_35_direct_hai_machine",
        ),
        "task": create_dataset(
            OUTPUT_ROOT / "visible_straight_lerobot",
            repo_id="libero_push_box_6fric_50pairs_visible_straight_jitter_35_35_direct_hai_machine",
        ),
    }
    subset_rows: dict[str, list[dict[str, Any]]] = {domain: [] for domain in datasets}
    subset_metadata = {
        domain: {
            "created_at": dt.datetime.now().isoformat(),
            "dataset_type": "libero_push_box_6fric_50pairs_direct_lerobot_hai-machine",
            "domain": domain,
            "target_visible": bool(domain == "task"),
            "split": "straight",
            "camera_resolution": CAMERA_RESOLUTION,
            "fps": FPS,
            "video_codec": VIDEO_CODEC,
            "video_crf": VIDEO_CRF,
            "jpeg_quality": JPEG_QUALITY,
            "state_source": "true LIBERO obs robot0_eef_pos, robot0_eef_quat converted to axis-angle, robot0_gripper_qpos",
            "episodes": [],
        }
        for domain in datasets
    }
    manifest = {
        "created_at": dt.datetime.now().isoformat(),
        "dataset_type": "libero_push_box_6fric_50pairs_direct_lerobot_collection_hai-machine",
        "output_root": str(OUTPUT_ROOT),
        "roots": {
            "observation_straight": str(OUTPUT_ROOT / "hidden_straight_lerobot"),
            "task_straight": str(OUTPUT_ROOT / "visible_straight_lerobot"),
        },
        "frictions": FRICTIONS,
        "pairs_per_friction": sum(BUCKET_QUOTA.values()),
        "bucket_quota": BUCKET_QUOTA,
        "bucket_ranges_m": BUCKET_RANGES_M,
        "quality": QUALITY,
        "prepare_config": PREPARE_CONFIG,
        "sampler": SAMPLER,
        "jitter": {
            "seed": BASE_SEED,
            "x_range_m": list(JITTER_X_RANGE),
            "y_range_m": list(JITTER_Y_RANGE),
            "preposition_steps": JITTER_PREPOSITION_STEPS,
        },
        "pairs": [],
        "reject_counts": {},
    }

    def autosave() -> None:
        write_json(OUTPUT_ROOT / "manifest.json", manifest)
        for domain in datasets:
            root = OUTPUT_ROOT / ("hidden_straight_lerobot" if domain == "observation" else "visible_straight_lerobot")
            write_dataset_metadata(root, subset_metadata[domain], subset_rows[domain])

    dummy_target_xy = (INIT_XY[0] + DUMMY_TARGET_DISTANCE, INIT_XY[1])
    for mu_i, mu in enumerate(FRICTIONS):
        rng = np.random.default_rng(BASE_SEED + mu_i * 1009)
        counts = {bucket: 0 for bucket in BUCKET_ORDER}
        attempts = 0
        pair_index = 0
        print(f"start {mu_tag(mu)} counts={counts}", flush=True)
        while not all(counts[bucket] >= BUCKET_QUOTA[bucket] for bucket in BUCKET_ORDER):
            attempts += 1
            if attempts > MAX_ATTEMPTS_PER_MU:
                raise RuntimeError(f"too many attempts for {mu_tag(mu)} counts={counts}")
            plan = sample_plan(mu, rng, counts)
            jitter_xy = sample_jitter(rng)
            attempt_id = f"{mu_tag(mu)}_try{attempts:04d}"
            probe_bddl = write_geometry_bddl(
                repo_root=REPO_ROOT,
                bddl_dir=bddl_root / "probe",
                geometry_id=attempt_id,
                init_xy=INIT_XY,
                target_xy=dummy_target_xy,
                init_half_size=0.002,
                target_radius=TARGET_RADIUS,
                target_rgba=(0.0, 0.0, 0.0, 0.0),
            )
            probe_case = build_push_case(plan, case_id=attempt_id, bddl_file=probe_bddl, target_xy=dummy_target_xy, camera_resolution=32)
            probe_metrics = probe_rollout(probe_case, jitter_xy=jitter_xy)
            actual_bucket = classify_displacement(float(probe_metrics["final_displacement_m"]))
            if actual_bucket is None:
                manifest["reject_counts"]["bucket_out_of_range"] = int(manifest["reject_counts"].get("bucket_out_of_range", 0)) + 1
                continue
            if counts.get(actual_bucket, 0) >= BUCKET_QUOTA[actual_bucket]:
                reason = f"bucket_full_{actual_bucket}"
                manifest["reject_counts"][reason] = int(manifest["reject_counts"].get(reason, 0)) + 1
                continue
            if not quality_ok(probe_metrics):
                manifest["reject_counts"]["quality_probe"] = int(manifest["reject_counts"].get("quality_probe", 0)) + 1
                continue

            pair_id = (
                f"{mu_tag(mu)}_{pair_index:03d}_{actual_bucket}_{plan['mode']}"
                f"_A{int(round(plan['A'] * 1000)):03d}_h{int(plan.get('hold', 0))}_jitter"
            )
            target_xy = tuple(float(x) for x in probe_metrics["final_xy"])
            pair_row = {
                "pair_id": pair_id,
                "mu": float(mu),
                "mu_tag": mu_tag(mu),
                "bucket": actual_bucket,
                "plan": dict(plan),
                "jitter_xy": [float(jitter_xy[0]), float(jitter_xy[1])],
                "target_xy": list(target_xy),
                "probe_metrics": probe_metrics,
                "episodes": {},
            }
            accepted = True
            for domain, rgba in {
                "observation": (0.0, 0.0, 0.0, 0.0),
                "task": (0.0, 0.8, 0.2, 0.45),
            }.items():
                visibility = "invisible" if domain == "observation" else "visible"
                bddl_file = write_geometry_bddl(
                    repo_root=REPO_ROOT,
                    bddl_dir=bddl_root / visibility,
                    geometry_id=f"{pair_id}_{visibility}",
                    init_xy=INIT_XY,
                    target_xy=target_xy,
                    init_half_size=0.002,
                    target_radius=TARGET_RADIUS,
                    target_rgba=rgba,
                )
                case = build_push_case(pair_row["plan"], case_id=f"{pair_id}_{visibility}", bddl_file=bddl_file, target_xy=target_xy, camera_resolution=CAMERA_RESOLUTION)
                episode_index, metrics = dataset_rollout(case, dataset=datasets[domain], domain=domain, jitter_xy=jitter_xy)
                row = {
                    "episode_index": int(episode_index),
                    "domain": domain,
                    "visibility": visibility,
                    "pair_id": pair_id,
                    "case_id": case.case_id,
                    "mu": float(mu),
                    "mu_tag": mu_tag(mu),
                    "bucket": actual_bucket,
                    "plan": dict(plan),
                    "jitter_xy": [float(jitter_xy[0]), float(jitter_xy[1])],
                    "target_xy": list(target_xy),
                    "bddl_file": bddl_file,
                    "metrics": metrics,
                }
                pair_row["episodes"][domain] = row
                subset_rows[domain].append(row)
                subset_metadata[domain]["episodes"].append(row)
                if classify_displacement(float(metrics["final_displacement_m"])) != actual_bucket or not quality_ok(metrics):
                    accepted = False
            if not accepted:
                raise RuntimeError(f"accepted probe failed replay quality for {pair_id}; inspect output before continuing")

            manifest["pairs"].append(pair_row)
            counts[actual_bucket] += 1
            pair_index += 1
            autosave()
            total_mu = sum(counts.values())
            print(
                f"accept {pair_id} {total_mu:02d}/50 counts={counts} "
                f"dist={probe_metrics['final_displacement_m'] * 100:.1f}cm A={plan['A']:.3f} hold={plan.get('hold', 0)}",
                flush=True,
            )
        print(f"done {mu_tag(mu)} counts={counts} attempts={attempts}", flush=True)

    summary_counts = defaultdict(Counter)
    for pair in manifest["pairs"]:
        summary_counts[pair["mu_tag"]][pair["bucket"]] += 1
    summary = {
        "pair_count": len(manifest["pairs"]),
        "episode_count": sum(len(v) for v in subset_rows.values()),
        "roots": manifest["roots"],
        "by_mu_bucket": {mu: dict(counter) for mu, counter in sorted(summary_counts.items())},
        "reject_counts": manifest["reject_counts"],
    }
    write_json(OUTPUT_ROOT / "summary.json", summary)
    autosave()
    print(f"manifest={OUTPUT_ROOT / 'manifest.json'}", flush=True)
    print(f"hidden_root={OUTPUT_ROOT / 'hidden_straight_lerobot'}", flush=True)
    print(f"visible_root={OUTPUT_ROOT / 'visible_straight_lerobot'}", flush=True)
    print(json.dumps(to_jsonable(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()
