from __future__ import annotations

from dataclasses import dataclass
import datetime as _dt
import json
from pathlib import Path
from typing import Any

import numpy as np

from .cases import DynamicCarrierCase
from .dynamic_env import DynamicCarrierEnv, create_libero_env_for_case
from .planner import PlannerConfig, ScriptedDynamicCarrierPlanner

try:
    import h5py
except ModuleNotFoundError:  # FastWAM's current venv does not include h5py.
    h5py = None


@dataclass(frozen=True)
class CollectionConfig:
    output_path: Path
    repo_root: Path
    camera_resolution: int = 128
    episodes_per_case: int = 10
    seed: int = 0
    static_success_threshold: float = 0.95
    require_static_gate: bool = False


def collect_dataset(
    cases: list[DynamicCarrierCase],
    config: CollectionConfig,
    planner_config: PlannerConfig | None = None,
) -> None:
    if config.require_static_gate:
        validate_static_gates(cases, config, planner_config=planner_config)

    if config.output_path.suffix.lower() in {".h5", ".hdf5"}:
        _collect_dataset_hdf5(cases, config, planner_config=planner_config)
    else:
        _collect_dataset_npz_dir(cases, config, planner_config=planner_config)


def _collect_dataset_hdf5(
    cases: list[DynamicCarrierCase],
    config: CollectionConfig,
    planner_config: PlannerConfig | None = None,
) -> None:
    if h5py is None:
        raise ModuleNotFoundError(
            "h5py is required for .h5/.hdf5 output. "
            "Use an output directory path for NPZ+JSON collection, or install h5py in this environment."
        )

    config.output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(config.output_path, "w") as h5:
        data_group = h5.create_group("data")
        data_group.attrs["date"] = _dt.datetime.now().isoformat()
        data_group.attrs["dataset_type"] = "ttt4dynamics_dynamic_carrier"
        data_group.attrs["episodes_per_case"] = int(config.episodes_per_case)
        data_group.attrs["case_count"] = int(len(cases))
        data_group.attrs["cases_json"] = json.dumps([case.as_dict() for case in cases])

        demo_index = 0
        for case in cases:
            case.validate_target_separation()
            for episode_idx in range(config.episodes_per_case):
                episode_seed = int(config.seed + demo_index)
                result = collect_episode(
                    case=case,
                    repo_root=config.repo_root,
                    camera_resolution=config.camera_resolution,
                    seed=episode_seed,
                    planner_config=planner_config,
                )
                demo_group = data_group.create_group(f"demo_{demo_index:06d}")
                _write_episode_group(demo_group, case, episode_idx, episode_seed, result)
                demo_index += 1


def _collect_dataset_npz_dir(
    cases: list[DynamicCarrierCase],
    config: CollectionConfig,
    planner_config: PlannerConfig | None = None,
) -> None:
    config.output_path.mkdir(parents=True, exist_ok=True)
    metadata = {
        "date": _dt.datetime.now().isoformat(),
        "dataset_type": "ttt4dynamics_dynamic_carrier_npz",
        "episodes_per_case": int(config.episodes_per_case),
        "case_count": int(len(cases)),
        "cases": [case.as_dict() for case in cases],
        "demos": [],
    }

    demo_index = 0
    for case in cases:
        case.validate_target_separation()
        for episode_idx in range(config.episodes_per_case):
            episode_seed = int(config.seed + demo_index)
            result = collect_episode(
                case=case,
                repo_root=config.repo_root,
                camera_resolution=config.camera_resolution,
                seed=episode_seed,
                planner_config=planner_config,
            )
            stem = f"demo_{demo_index:06d}"
            npz_path = config.output_path / f"{stem}.npz"
            json_path = config.output_path / f"{stem}.json"
            _write_npz_episode(npz_path, json_path, case, episode_idx, episode_seed, result)
            metadata["demos"].append(
                {
                    "demo_id": stem,
                    "case_id": case.case_id,
                    "episode_idx": episode_idx,
                    "seed": episode_seed,
                    "success": bool(result["success"]),
                    "steps": int(result["steps"]),
                    "npz": npz_path.name,
                    "json": json_path.name,
                }
            )
            demo_index += 1

    (config.output_path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def validate_static_gates(
    cases: list[DynamicCarrierCase],
    config: CollectionConfig,
    planner_config: PlannerConfig | None = None,
) -> dict[str, float]:
    rates: dict[str, float] = {}
    for case in cases:
        frozen = case.frozen_variant()
        successes = 0
        trials = max(1, min(5, config.episodes_per_case))
        for idx in range(trials):
            result = collect_episode(
                case=frozen,
                repo_root=config.repo_root,
                camera_resolution=config.camera_resolution,
                seed=int(config.seed + idx),
                planner_config=planner_config,
            )
            successes += int(result["success"])
        rate = successes / trials
        rates[case.case_id] = rate
        if rate < config.static_success_threshold:
            raise RuntimeError(
                f"Static feasibility gate failed for {case.case_id}: "
                f"{successes}/{trials} = {rate:.3f}, "
                f"required >= {config.static_success_threshold:.3f}"
            )
    return rates


def collect_episode(
    *,
    case: DynamicCarrierCase,
    repo_root: Path,
    camera_resolution: int,
    seed: int,
    planner_config: PlannerConfig | None = None,
) -> dict[str, Any]:
    base_env, init_state, task_description = create_libero_env_for_case(
        case,
        repo_root=repo_root,
        camera_resolution=camera_resolution,
        seed=seed,
    )
    env = DynamicCarrierEnv(base_env, case)
    planner = ScriptedDynamicCarrierPlanner(env, planner_config)
    states = []
    actions = []
    eef_xyz = []
    payload_xyz = []
    carrier_xyz = []
    carrier_velocity_xy = []
    phase_names = []
    dynamic_infos = []

    try:
        obs = env.reset(init_state=init_state)
        planner.reset()
        success = False
        done = False
        for _ in range(int(case.max_steps)):
            states.append(env.get_sim_state().copy())
            eef_xyz.append(env.eef_position(obs).copy())
            payload_xyz.append(env.payload_position().copy())
            carrier_xyz.append(env.carrier_position().copy())
            carrier_velocity_xy.append(env.carrier_velocity_xy().copy())
            phase_names.append(str(planner.phase.value))

            action = planner.act(obs)
            actions.append(action.copy())
            obs, _, done, info = env.step(action)
            dynamic_infos.append(info.get("dynamic_carrier", {}))
            success = bool(env.check_success())
            if success or done or planner.is_done():
                success = bool(env.check_success())
                break

        return {
            "success": bool(success),
            "task_description": task_description,
            "states": np.asarray(states, dtype=np.float64),
            "actions": np.asarray(actions, dtype=np.float64),
            "eef_xyz": np.asarray(eef_xyz, dtype=np.float64),
            "payload_xyz": np.asarray(payload_xyz, dtype=np.float64),
            "carrier_xyz": np.asarray(carrier_xyz, dtype=np.float64),
            "carrier_velocity_xy": np.asarray(carrier_velocity_xy, dtype=np.float64),
            "phase_names": np.asarray(phase_names, dtype="U32"),
            "dynamic_infos": dynamic_infos,
            "steps": int(len(actions)),
            "final_phase": str(planner.phase.value),
        }
    finally:
        env.close()


def _write_episode_group(
    group: Any,
    case: DynamicCarrierCase,
    episode_idx: int,
    seed: int,
    result: dict[str, Any],
) -> None:
    group.attrs["case_id"] = case.case_id
    group.attrs["case_json"] = json.dumps(case.as_dict())
    group.attrs["episode_idx"] = int(episode_idx)
    group.attrs["seed"] = int(seed)
    group.attrs["success"] = bool(result["success"])
    group.attrs["steps"] = int(result["steps"])
    group.attrs["final_phase"] = result["final_phase"]
    group.attrs["task_description"] = result["task_description"]
    group.attrs["dynamic_infos_json"] = json.dumps(result["dynamic_infos"])

    group.create_dataset("states", data=result["states"], compression="gzip")
    group.create_dataset("actions", data=result["actions"], compression="gzip")
    group.create_dataset("eef_xyz", data=result["eef_xyz"], compression="gzip")
    group.create_dataset("payload_xyz", data=result["payload_xyz"], compression="gzip")
    group.create_dataset("carrier_xyz", data=result["carrier_xyz"], compression="gzip")
    group.create_dataset("carrier_velocity_xy", data=result["carrier_velocity_xy"], compression="gzip")
    phase_dtype = h5py.string_dtype("utf-8") if h5py is not None else None
    group.create_dataset("planner_phase", data=result["phase_names"].astype(object), dtype=phase_dtype)


def _write_npz_episode(
    npz_path: Path,
    json_path: Path,
    case: DynamicCarrierCase,
    episode_idx: int,
    seed: int,
    result: dict[str, Any],
) -> None:
    np.savez_compressed(
        npz_path,
        states=result["states"],
        actions=result["actions"],
        eef_xyz=result["eef_xyz"],
        payload_xyz=result["payload_xyz"],
        carrier_xyz=result["carrier_xyz"],
        carrier_velocity_xy=result["carrier_velocity_xy"],
        planner_phase=result["phase_names"],
    )
    meta = {
        "case_id": case.case_id,
        "case": case.as_dict(),
        "episode_idx": int(episode_idx),
        "seed": int(seed),
        "success": bool(result["success"]),
        "steps": int(result["steps"]),
        "final_phase": result["final_phase"],
        "task_description": result["task_description"],
        "dynamic_infos": result["dynamic_infos"],
    }
    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
