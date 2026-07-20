#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY_CONFIG = (
    REPO_ROOT
    / "configs"
    / "libero_plus_push_box_matched_action_velocity_3object_3friction_2026-07-19_hai-machine.json"
)
ASSET_ROLLOUT_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "collect_libero_plus_push_box_official_assets_full_trajectory_preview_lerobot_2026-07-18_hai-machine.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-config", type=Path, default=DEFAULT_STUDY_CONFIG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class NullDataset:
    def __init__(self) -> None:
        self.meta = SimpleNamespace(total_episodes=0)

    def add_frame(self, *_: Any, **__: Any) -> None:
        return None

    def save_episode(self) -> None:
        self.meta.total_episodes += 1


def first_stopped_frame(
    velocity: np.ndarray,
    *,
    threshold: float,
    consecutive: int,
    start: int,
) -> int | None:
    for index in range(start, max(start, len(velocity) - consecutive + 1)):
        if np.all(np.abs(velocity[index : index + consecutive]) <= threshold):
            return int(index)
    return None


def sample_after_step(values: np.ndarray, seconds: float, fps: float) -> float:
    index = min(len(values) - 1, max(0, int(round(seconds * fps)) - 1))
    return float(values[index])


def summarize_trace(
    trace: list[dict[str, Any]],
    metrics: dict[str, Any],
    *,
    fps: float,
    stop_threshold: float,
    stop_consecutive: int,
) -> dict[str, Any]:
    push_start = int(metrics["full_trajectory"]["push_start_frame"])
    push_steps = int(
        sum(
            count
            for phase, count in metrics["full_trajectory"]["phase_frame_counts"].items()
            if phase.startswith("push_")
        )
    )
    push_trace = trace[push_start : push_start + push_steps]
    if len(push_trace) != push_steps:
        raise RuntimeError(
            f"Velocity trace length mismatch: expected {push_steps}, got {len(push_trace)}"
        )
    direction = np.asarray(metrics["direction_xy"], dtype=np.float64)
    box_qvel = np.asarray([row["target_qvel"] for row in push_trace], dtype=np.float64)
    box_velocity = box_qvel[:, :2] @ direction
    angular_speed = np.linalg.norm(box_qvel[:, 3:6], axis=1)
    target_xyz = np.asarray([row["target_xyz"] for row in push_trace], dtype=np.float64)
    eef_xyz = np.asarray([row["eef_xyz"] for row in push_trace], dtype=np.float64)
    previous_eef = np.asarray(trace[push_start - 1]["eef_xyz"], dtype=np.float64)
    eef_with_previous = np.vstack([previous_eef, eef_xyz])
    eef_velocity = np.diff(eef_with_previous[:, :2], axis=0) * fps @ direction
    peak_index = int(np.argmax(box_velocity))
    stopped_index = first_stopped_frame(
        box_velocity,
        threshold=stop_threshold,
        consecutive=stop_consecutive,
        start=peak_index,
    )
    fit_end = stopped_index if stopped_index is not None else len(box_velocity)
    fit_indices = np.arange(peak_index, fit_end, dtype=np.int64)
    fit_indices = fit_indices[box_velocity[fit_indices] > stop_threshold]
    if len(fit_indices) >= 3:
        slope, _ = np.polyfit(
            fit_indices.astype(np.float64) / fps,
            box_velocity[fit_indices],
            1,
        )
        fitted_deceleration = max(0.0, -float(slope))
    else:
        fitted_deceleration = 0.0
    launch_xyz = np.asarray(metrics["launch_box_xyz_m"], dtype=np.float64)
    displacement = (target_xyz[:, :2] - launch_xyz[:2]) @ direction
    peak_eef_velocity = float(np.max(eef_velocity))
    return {
        "push_trace": push_trace,
        "peak_target_velocity_mps": float(box_velocity[peak_index]),
        "peak_target_velocity_frame": peak_index,
        "time_to_peak_s": float((peak_index + 1) / fps),
        "peak_eef_velocity_mps": peak_eef_velocity,
        "target_to_eef_peak_velocity_ratio": (
            float(box_velocity[peak_index] / peak_eef_velocity)
            if peak_eef_velocity > 1e-9
            else None
        ),
        "target_velocity_at_0p10s_mps": sample_after_step(box_velocity, 0.10, fps),
        "target_velocity_at_0p25s_mps": sample_after_step(box_velocity, 0.25, fps),
        "target_velocity_at_0p50s_mps": sample_after_step(box_velocity, 0.50, fps),
        "target_velocity_at_1p00s_mps": sample_after_step(box_velocity, 1.00, fps),
        "fitted_post_peak_deceleration_mps2": fitted_deceleration,
        "stop_time_s": (
            float((stopped_index + 1) / fps) if stopped_index is not None else None
        ),
        "projected_displacement_at_0p50s_m": sample_after_step(displacement, 0.50, fps),
        "final_projected_displacement_m": float(metrics["box_projected_displacement_m"]),
        "peak_target_angular_speed_radps": float(np.max(angular_speed)),
        "contact_frame_count": len(metrics["contact_frames"]),
        "contact_episode_count": int(metrics["contact_episode_count"]),
        "drive_frame_count": int(
            metrics["full_trajectory"]["phase_frame_counts"].get("push_drive", 0)
        ),
        "brake_frame_count": int(metrics["brake_frames"]),
        "launch_target_xyz_m": launch_xyz.astype(float).tolist(),
    }


def create_state_only_dataset(base: Any, root: Path, fps: int) -> Any:
    state_names = [
        "target_x_m",
        "target_y_m",
        "target_z_m",
        "target_vx_mps",
        "target_vy_mps",
        "target_vz_mps",
        "target_wx_radps",
        "target_wy_radps",
        "target_wz_radps",
        "eef_x_m",
        "eef_y_m",
        "eef_z_m",
        "gripper_target_contact",
        "friction_mu",
        "object_id",
    ]
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(state_names),),
            "names": state_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["dx", "dy", "dz", "dax", "day", "daz", "gripper"],
        },
    }
    return base.LeRobotDataset.create(
        repo_id="libero_plus_push_box_matched_action_velocity_state_only_hai_machine",
        root=root,
        fps=fps,
        features=features,
        use_videos=False,
    )


def add_trace_episode(
    dataset: Any,
    push_trace: list[dict[str, Any]],
    *,
    object_id: int,
    object_name: str,
    friction_mu: float,
    fps: int,
) -> None:
    for frame_index, row in enumerate(push_trace):
        qvel = np.asarray(row["target_qvel"], dtype=np.float32)
        state = np.asarray(
            list(row["target_xyz"])
            + qvel[:3].astype(float).tolist()
            + qvel[3:6].astype(float).tolist()
            + list(row["eef_xyz"])
            + [float(row["contact"]), float(friction_mu), float(object_id)],
            dtype=np.float32,
        )
        dataset.add_frame(
            {
                "observation.state": state,
                "action": np.asarray(row["action"], dtype=np.float32),
            },
            task=[
                f"matched-action velocity comparison for {object_name}",
                f"push target at friction mu={friction_mu:.3f}",
                "state-only scripted observation rollout",
                "success",
            ],
            timestamp=float(frame_index) / float(fps),
        )
    dataset.save_episode()


def main() -> None:
    args = parse_args()
    study_path = args.study_config.resolve()
    study = json.loads(study_path.read_text(encoding="utf-8"))
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else (REPO_ROOT / study["output_root"]).resolve()
    )
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_root} exists; pass --overwrite")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    assets = load_module(ASSET_ROLLOUT_SCRIPT, "matched_velocity_asset_rollout_hai_machine")
    source_path = (REPO_ROOT / study["source_experiment_config"]).resolve()
    experiment = json.loads(source_path.read_text(encoding="utf-8"))
    experiment["initial_position"] = {
        "front_back_x_jitter_m": [0.0, 0.0],
        "horizontal_y_jitter_m": [0.0, 0.0],
    }
    experiment["distractor_randomization"] = {"enabled": False}
    experiment["background_randomization"]["presets"] = [
        experiment["background_randomization"]["presets"][0]
    ]
    assets.experiment = experiment
    assets.legacy._NATIVE_PRESETS = experiment["background_randomization"]["presets"]

    base = assets.full.base
    base.remove_current_episode_images = lambda dataset: None
    base.write_image_for_last_frame = lambda *args, **kwargs: None
    base._obs_to_images = lambda obs: (
        np.zeros((1, 1, 3), dtype=np.uint8),
        np.zeros((1, 1, 3), dtype=np.uint8),
    )

    trace_store: dict[str, list[dict[str, Any]]] = {"rows": []}
    original_make_env = assets.full.touch.make_env

    def recording_make_env(case: Any, *, seed: int) -> Any:
        env = original_make_env(case, seed=seed)
        original_step = env.step

        def recording_step(action: np.ndarray) -> Any:
            result = original_step(action)
            obj = env.inner_env.get_object(env.case.box_name)
            joint_name = obj.joints[-1]
            trace_store["rows"].append(
                {
                    "action": np.asarray(action, dtype=float).tolist(),
                    "target_xyz": np.asarray(env.box_pose()[0], dtype=float).tolist(),
                    "target_qvel": np.asarray(
                        env.inner_env.sim.data.get_joint_qvel(joint_name), dtype=float
                    ).tolist(),
                    "eef_xyz": np.asarray(
                        env._last_obs["robot0_eef_pos"], dtype=float
                    ).tolist(),
                    "contact": bool(assets.native_all_gripper_box_contact(env)),
                }
            )
            return result

        env.step = recording_step
        return env

    assets.full.touch.make_env = recording_make_env

    formal_experiment = json.loads(
        assets.legacy.formal.CONFIG_PATH.read_text(encoding="utf-8")
    )
    formal_config = assets.legacy.formal.configure_dataset(formal_experiment)
    variants = {int(row["object_id"]): row for row in experiment["object_variants"]}
    action_cfg = dict(study["matched_action"])
    fps = int(experiment["fps"])
    state_dataset_root = output_root / "lerobot_state_only"
    state_dataset = create_state_only_dataset(base, state_dataset_root, fps)
    null_dataset = NullDataset()
    rows = []
    fixed_seed = int(study["fixed_seed"])

    for friction_mu in [float(value) for value in study["friction_mu"]]:
        for object_id in [int(value) for value in study["object_ids"]]:
            variant = dict(variants[object_id])
            case_id = (
                f"matched_A025_{variant['name']}_"
                f"mu{int(round(friction_mu * 10000)):04d}"
            )
            bddl = base.write_hidden_bddl(
                formal_config,
                bddl_dir=output_root / "bddl",
                geometry_id=case_id,
            )
            source_action_cfg = {
                "action_id": int(action_cfg["action_id"]),
                "A": float(action_cfg["A"]),
                "push_steps": 16,
            }
            base_case = assets.legacy.collector.make_case(
                formal_config,
                mu=friction_mu,
                action_cfg=source_action_cfg,
                case_id=case_id,
                bddl_file=bddl,
            )
            case = assets.full.ramp.preserve_case_attributes(
                base_case,
                replace(
                    base_case,
                    pusher_max_pos_action=float(
                        experiment["controller"]["pusher_max_pos_action"]
                    ),
                ),
            )
            trace_store["rows"] = []
            _, metrics = assets.official_asset_rollout(
                case,
                dataset=null_dataset,
                action_cfg=action_cfg,
                variant=variant,
                experiment=experiment,
                seed=fixed_seed,
            )
            summary = summarize_trace(
                trace_store["rows"],
                metrics,
                fps=float(fps),
                stop_threshold=float(study["stop_speed_threshold_mps"]),
                stop_consecutive=int(study["stop_consecutive_frames"]),
            )
            push_trace = summary.pop("push_trace")
            add_trace_episode(
                state_dataset,
                push_trace,
                object_id=object_id,
                object_name=str(variant["display_name"]),
                friction_mu=friction_mu,
                fps=fps,
            )
            row = {
                "object_id": object_id,
                "object_name": str(variant["display_name"]),
                "friction_mu": friction_mu,
                "A": float(action_cfg["A"]),
                **summary,
            }
            rows.append(row)
            print(
                f"RESULT object={row['object_name']} mu={friction_mu:.3f} "
                f"peak={row['peak_target_velocity_mps']:.4f}m/s "
                f"decel={row['fitted_post_peak_deceleration_mps2']:.4f}m/s2 "
                f"stop={row['stop_time_s']}s "
                f"distance={row['final_projected_displacement_m']:.4f}m",
                flush=True,
            )

    payload = {
        "study_config": study,
        "source_experiment_config": str(source_path),
        "controlled_variables": {
            "A": float(action_cfg["A"]),
            "target_projected_eef_travel_m": float(
                action_cfg["target_projected_travel_m"]
            ),
            "target_mass_kg": float(study["target_mass_kg"]),
            "seed": fixed_seed,
            "initial_position_jitter_m": [0.0, 0.0],
            "distractors": False,
            "background_preset_count": 1,
            "video_generated": False,
        },
        "lerobot_state_only_root": str(state_dataset_root),
        "rows": rows,
    }
    (output_root / "velocity_report.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    with (output_root / "velocity_report.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
