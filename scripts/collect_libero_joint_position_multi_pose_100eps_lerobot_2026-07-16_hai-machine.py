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
SCRIPTS_DIR = REPO_ROOT / "scripts"
LIBERO_REPO = REPO_ROOT.parent / "LIBERO"
FASTWAM_ROOT = REPO_ROOT.parent / "FastWAM-TTT"
for path in (REPO_ROOT, SCRIPTS_DIR, LIBERO_REPO, FASTWAM_ROOT, FASTWAM_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

BASE_COLLECTOR_SCRIPT = (
    SCRIPTS_DIR / "collect_libero_push_box_formal_6fric_50pair_35_35_direct_lerobot_hai-machine.py"
)
DEMO_SCRIPT = SCRIPTS_DIR / "render_libero_action_world_model_diverse_demos_2026-07-15_hai-machine.py"
DEFAULT_OUTPUT = (
    REPO_ROOT / "data" / "various_actions"
    / "libero_mu0100_joint_position_multi_pose_100eps_lerobot_2026-07-16_hai-machine"
)

EPISODE_COUNT = 100
BASE_SEED = 20260716
SIM_SEED = 20260716
JOINT_COUNT = 7
JOINT_CONTROLLER_DELTA_RAD = 0.05
MAX_NORMALIZED_JOINT_ACTION = 0.80
MAX_COMMAND_DELTA_RAD = JOINT_CONTROLLER_DELTA_RAD * MAX_NORMALIZED_JOINT_ACTION
JOINT_TOLERANCE_RAD = 0.020
JOINT_VELOCITY_TOLERANCE_RAD_S = 0.15
MIN_STEPS_PER_JOINT = 4
MAX_STEPS_PER_JOINT = 26
SETTLE_STEPS_PER_JOINT = 2
POSE_HOLD_STEPS = 6
INITIAL_HOLD_STEPS = 6
JOINT_MAX_OFFSETS_RAD = np.asarray([0.30, 0.25, 0.30, 0.25, 0.30, 0.25, 0.35], dtype=np.float64)
JOINT_MIN_POSE_CHANGE_RAD = np.asarray([0.08, 0.07, 0.08, 0.07, 0.08, 0.07, 0.09], dtype=np.float64)
TASK_PROMPT = [
    "explore robot joint-position dynamics on a tabletop",
    "visit multiple robot poses by moving one arm joint at a time",
    "observe how sequential joint-angle commands change the robot and scene",
    "joint-action-conditioned world-model trajectory",
]


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_module(BASE_COLLECTOR_SCRIPT, "joint_pose_base_collector_hai_machine")
demo = load_module(DEMO_SCRIPT, "joint_pose_demo_base_hai_machine")

from ttt4dynamics.push_box_libero import LiberoPushBoxEnv, ensure_libero_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect 100 LIBERO JOINT_POSITION episodes with multiple poses and randomized per-joint order."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(base.to_jsonable(value), indent=2), encoding="utf-8")


class LiberoJointPositionEnv(LiberoPushBoxEnv):
    def __init__(self, case: Any, *, repo_root: Path, seed: int):
        ensure_libero_config(repo_root)
        from libero.libero.envs import OffScreenRenderEnv

        self.case = case
        self.repo_root = Path(repo_root).resolve()
        self.seed = int(seed)
        bddl_file = Path(case.bddl_file)
        if not bddl_file.is_absolute():
            bddl_file = self.repo_root / bddl_file
        if not bddl_file.exists():
            raise FileNotFoundError(f"Joint-position BDDL not found: {bddl_file}")
        self.env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_file),
            controller="JOINT_POSITION",
            camera_heights=int(case.camera_resolution),
            camera_widths=int(case.camera_resolution),
            control_freq=float(case.control_freq),
            horizon=2000,
            ignore_done=True,
        )
        self.env.seed(self.seed)
        self.step_count = 0
        self._last_obs: dict[str, Any] | None = None
        self._initial_box_xyz: np.ndarray | None = None
        self._last_scripted_action = np.zeros(8, dtype=np.float64)
        self._last_scripted_phase: str | None = None
        self._base_controller_output_min = None
        self._base_controller_output_max = None
        self._active_controller_output_scale = None

    @property
    def robot(self) -> Any:
        return self.inner_env.robots[0]

    def reset(self) -> dict[str, Any]:
        self.step_count = 0
        self.env.reset()
        self._set_box_contact_dynamics()
        self._zero_box_velocity()
        self.inner_env.sim.forward()
        self._initial_box_xyz, _ = self.box_pose()
        self._last_obs = self._refresh_obs()
        if str(self.robot.controller.name) != "JOINT_POSITION":
            raise RuntimeError(f"Expected JOINT_POSITION controller, got {self.robot.controller.name}")
        if int(self.robot.action_dim) != 8:
            raise RuntimeError(f"Expected 8D [7 joints + gripper] action, got {self.robot.action_dim}")
        return self._last_obs

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        command = np.asarray(action, dtype=np.float64)
        if command.shape != (8,):
            raise ValueError(f"Expected joint action shape (8,), got {command.shape}")
        obs, reward, done, info = self.env.step(command)
        self.step_count += 1
        self._last_obs = self._refresh_obs()
        return self._last_obs, float(reward), bool(done), dict(info)

    def joint_positions(self) -> np.ndarray:
        return np.asarray(self.robot._joint_positions, dtype=np.float64).copy()

    def joint_velocities(self) -> np.ndarray:
        return np.asarray(self.robot._joint_velocities, dtype=np.float64).copy()

    def joint_limits(self) -> np.ndarray:
        model = self.inner_env.sim.model
        return np.asarray(model.jnt_range[self.robot._ref_joint_indexes], dtype=np.float64).copy()


def build_features() -> dict[str, dict[str, Any]]:
    image_shape = (3, demo.CAMERA_RESOLUTION, demo.CAMERA_RESOLUTION)
    state_names = (
        [f"joint_q{i}_rad" for i in range(JOINT_COUNT)]
        + [f"joint_dq{i}_rad_s" for i in range(JOINT_COUNT)]
        + ["eef_x_m", "eef_y_m", "eef_z_m"]
        + ["eef_axis_x_rad", "eef_axis_y_rad", "eef_axis_z_rad"]
        + ["gripper_qpos_0", "gripper_qpos_1"]
    )
    action_names = [f"delta_joint_q{i}_rad" for i in range(JOINT_COUNT)] + ["gripper_open"]
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
            "shape": (22,),
            "names": state_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (8,),
            "names": action_names,
        },
    }


def create_dataset(root: Path) -> Any:
    return base.LeRobotDataset.create(
        repo_id="libero_mu0100_joint_position_multi_pose_100eps_hai_machine",
        root=root,
        fps=demo.FPS,
        features=build_features(),
        use_videos=True,
        video_codec="h264",
        is_compute_episode_stats_image=False,
    )


def joint_state(env: LiberoJointPositionEnv, obs: dict[str, Any]) -> np.ndarray:
    return np.concatenate(
        [
            env.joint_positions().astype(np.float32),
            env.joint_velocities().astype(np.float32),
            base._obs_to_state(obs),
        ],
        axis=0,
    ).astype(np.float32)


def env_action_to_dataset_action(env_action: np.ndarray) -> np.ndarray:
    action = np.asarray(env_action, dtype=np.float32).copy()
    action[:JOINT_COUNT] *= float(JOINT_CONTROLLER_DELTA_RAD)
    action[-1] = (1.0 - action[-1]) / 2.0
    return action


class JointEpisodeRecorder:
    def __init__(self, *, env: LiberoJointPositionEnv, dataset: Any):
        self.env = env
        self.dataset = dataset
        self.obs = env.reset()
        self.initial_box_xyz, _ = env.box_pose()
        self.initial_joint_qpos = env.joint_positions()
        self.frame_index = 0
        self.last_gripper_env = -1.0
        self.phase_counts: Counter[str] = Counter()
        self.env_actions: list[np.ndarray] = []
        self.dataset_actions: list[np.ndarray] = []
        self.max_box_speed_mps = 0.0
        self.max_eef_speed_mps = 0.0
        self.robot_box_contact_steps = 0
        self.grasping_steps = 0
        base.remove_current_episode_images(dataset)

    def _add_frame(self, action_env: np.ndarray, phase: str) -> None:
        agent, wrist = base._obs_to_images(self.obs)
        action_dataset = env_action_to_dataset_action(action_env)
        frame = {
            "observation.images.image": agent,
            "observation.images.wrist_image": wrist,
            "observation.state": joint_state(self.env, self.obs),
            "action": action_dataset,
        }
        self.dataset.add_frame(
            frame,
            task=TASK_PROMPT,
            timestamp=float(self.frame_index) / float(demo.FPS),
        )
        base.write_image_for_last_frame(
            self.dataset,
            "observation.images.image",
            self.frame_index,
            agent,
        )
        base.write_image_for_last_frame(
            self.dataset,
            "observation.images.wrist_image",
            self.frame_index,
            wrist,
        )
        self.phase_counts[phase] += 1
        self.frame_index += 1

    def step(self, action_env: np.ndarray, phase: str) -> None:
        command = np.asarray(action_env, dtype=np.float64).copy()
        command[:JOINT_COUNT] = np.clip(command[:JOINT_COUNT], -1.0, 1.0)
        command[-1] = float(np.clip(command[-1], -1.0, 1.0))
        nonzero_joint_dims = int(np.sum(np.abs(command[:JOINT_COUNT]) > 1e-8))
        if nonzero_joint_dims > 1:
            raise RuntimeError(f"More than one arm joint commanded in one step: {command}")

        eef_t = np.asarray(self.obs["robot0_eef_pos"], dtype=np.float64)
        self._add_frame(command, phase)
        obs_tp1, _, _, _ = self.env.step(command)
        eef_tp1 = np.asarray(obs_tp1["robot0_eef_pos"], dtype=np.float64)
        box_qvel = self.env.box_velocity()
        self.max_box_speed_mps = max(self.max_box_speed_mps, float(np.linalg.norm(box_qvel[:2])))
        self.max_eef_speed_mps = max(
            self.max_eef_speed_mps,
            float(np.linalg.norm(eef_tp1 - eef_t) * demo.FPS),
        )
        self.robot_box_contact_steps += int(demo.robot_box_contact(self.env))
        self.grasping_steps += int(demo.grasping(self.env))
        self.env_actions.append(command)
        self.dataset_actions.append(env_action_to_dataset_action(command))
        self.last_gripper_env = float(command[-1])
        self.obs = obs_tp1

    def hold(self, steps: int, *, gripper_env: float, phase: str) -> None:
        for _ in range(int(steps)):
            action = np.zeros(8, dtype=np.float64)
            action[-1] = float(gripper_env)
            self.step(action, phase)

    def add_terminal_observation(self) -> None:
        terminal_action = np.zeros(8, dtype=np.float64)
        terminal_action[-1] = float(self.last_gripper_env)
        self._add_frame(terminal_action, "terminal_observation")

    def summary(self, *, final_pose_errors: list[float], segments: list[dict[str, Any]]) -> dict[str, Any]:
        final_box_xyz, _ = self.env.box_pose()
        actions = np.asarray(self.env_actions, dtype=np.float64)
        arm_nonzero_counts = (
            np.sum(np.abs(actions[:, :JOINT_COUNT]) > 1e-8, axis=1) if actions.size else np.zeros(0)
        )
        segment_errors = [float(segment["final_abs_error_rad"]) for segment in segments]
        return {
            "frames_in_lerobot_episode": int(self.frame_index),
            "steps_with_effective_actions": len(self.env_actions),
            "terminal_observation_added": True,
            "phase_counts": dict(self.phase_counts),
            "initial_joint_qpos_rad": self.initial_joint_qpos,
            "final_joint_qpos_rad": self.env.joint_positions(),
            "final_pose_error_norms_rad": final_pose_errors,
            "mean_final_pose_error_norm_rad": float(np.mean(final_pose_errors)) if final_pose_errors else 0.0,
            "max_final_pose_error_norm_rad": float(np.max(final_pose_errors)) if final_pose_errors else 0.0,
            "mean_segment_final_abs_error_rad": float(np.mean(segment_errors)) if segment_errors else 0.0,
            "max_segment_final_abs_error_rad": float(np.max(segment_errors)) if segment_errors else 0.0,
            "max_nonzero_arm_joint_dims_per_action": int(np.max(arm_nonzero_counts)) if arm_nonzero_counts.size else 0,
            "initial_box_xyz_m": self.initial_box_xyz,
            "final_box_xyz_m": final_box_xyz,
            "final_box_displacement_m": float(np.linalg.norm(final_box_xyz[:2] - self.initial_box_xyz[:2])),
            "max_box_planar_speed_mps": float(self.max_box_speed_mps),
            "max_eef_speed_mps": float(self.max_eef_speed_mps),
            "robot_box_contact_steps": int(self.robot_box_contact_steps),
            "robosuite_grasping_steps": int(self.grasping_steps),
            "max_abs_dataset_action_by_dim": (
                np.max(np.abs(np.asarray(self.dataset_actions)), axis=0)
                if self.dataset_actions
                else np.zeros(8, dtype=np.float64)
            ),
        }


def sample_distinct_pose(
    *,
    nominal: np.ndarray,
    previous: np.ndarray,
    limits: np.ndarray,
    rng: np.random.Generator,
    amplitude_scale: float,
) -> np.ndarray:
    target = np.empty(JOINT_COUNT, dtype=np.float64)
    lower = limits[:, 0] + 0.15
    upper = limits[:, 1] - 0.15
    max_offsets = JOINT_MAX_OFFSETS_RAD * float(amplitude_scale)
    for joint_index in range(JOINT_COUNT):
        candidate = float(previous[joint_index])
        for _ in range(64):
            proposal = float(nominal[joint_index] + rng.uniform(-max_offsets[joint_index], max_offsets[joint_index]))
            proposal = float(np.clip(proposal, lower[joint_index], upper[joint_index]))
            if abs(proposal - previous[joint_index]) >= JOINT_MIN_POSE_CHANGE_RAD[joint_index]:
                candidate = proposal
                break
        if abs(candidate - previous[joint_index]) < JOINT_MIN_POSE_CHANGE_RAD[joint_index]:
            sign = -1.0 if previous[joint_index] >= nominal[joint_index] else 1.0
            candidate = float(
                np.clip(
                    previous[joint_index] + sign * JOINT_MIN_POSE_CHANGE_RAD[joint_index],
                    lower[joint_index],
                    upper[joint_index],
                )
            )
        target[joint_index] = candidate
    return target


def move_one_joint(
    recorder: JointEpisodeRecorder,
    *,
    pose_index: int,
    joint_index: int,
    target_rad: float,
    gripper_env: float,
) -> dict[str, Any]:
    start_frame = int(recorder.frame_index)
    start_qpos = recorder.env.joint_positions()
    reached = False
    action_steps = 0
    for local_step in range(MAX_STEPS_PER_JOINT):
        qpos = recorder.env.joint_positions()
        qvel = recorder.env.joint_velocities()
        error = float(target_rad - qpos[joint_index])
        if (
            local_step >= MIN_STEPS_PER_JOINT
            and abs(error) <= JOINT_TOLERANCE_RAD
            and abs(float(qvel[joint_index])) <= JOINT_VELOCITY_TOLERANCE_RAD_S
        ):
            reached = True
            break
        action = np.zeros(8, dtype=np.float64)
        action[joint_index] = float(
            np.clip(
                error / JOINT_CONTROLLER_DELTA_RAD,
                -MAX_NORMALIZED_JOINT_ACTION,
                MAX_NORMALIZED_JOINT_ACTION,
            )
        )
        action[-1] = float(gripper_env)
        recorder.step(action, f"pose_{pose_index:02d}_joint_{joint_index}")
        action_steps += 1

    recorder.hold(
        SETTLE_STEPS_PER_JOINT,
        gripper_env=gripper_env,
        phase=f"pose_{pose_index:02d}_joint_{joint_index}_settle",
    )
    final_qpos = recorder.env.joint_positions()
    final_error = float(target_rad - final_qpos[joint_index])
    reached = bool(reached or abs(final_error) <= JOINT_TOLERANCE_RAD)
    return {
        "pose_index": pose_index,
        "joint_index": joint_index,
        "start_frame": start_frame,
        "end_frame_exclusive": int(recorder.frame_index),
        "start_angle_rad": float(start_qpos[joint_index]),
        "target_angle_rad": float(target_rad),
        "final_angle_rad": float(final_qpos[joint_index]),
        "final_abs_error_rad": abs(final_error),
        "reached_tolerance": reached,
        "action_steps": action_steps,
        "settle_steps": SETTLE_STEPS_PER_JOINT,
    }


def run_multi_pose_episode(
    recorder: JointEpisodeRecorder,
    *,
    episode_index: int,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[float]]:
    nominal = recorder.initial_joint_qpos.copy()
    limits = recorder.env.joint_limits()
    random_pose_count = 5 + int(episode_index % 2)
    amplitude_scale = float(rng.uniform(0.75, 1.0))
    target_poses: list[np.ndarray] = []
    previous = nominal.copy()
    for _ in range(random_pose_count):
        target = sample_distinct_pose(
            nominal=nominal,
            previous=previous,
            limits=limits,
            rng=rng,
            amplitude_scale=amplitude_scale,
        )
        target_poses.append(target)
        previous = target
    target_poses.append(nominal.copy())

    recorder.hold(INITIAL_HOLD_STEPS, gripper_env=-1.0, phase="initial_observation")
    pose_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    final_pose_errors: list[float] = []
    for pose_index, target in enumerate(target_poses):
        order = rng.permutation(JOINT_COUNT).astype(int).tolist()
        gripper_env = -1.0 if int(rng.integers(0, 2)) == 0 else 1.0
        pose_start_frame = int(recorder.frame_index)
        pose_segments = []
        for joint_index in order:
            segment = move_one_joint(
                recorder,
                pose_index=pose_index,
                joint_index=int(joint_index),
                target_rad=float(target[joint_index]),
                gripper_env=gripper_env,
            )
            segment_rows.append(segment)
            pose_segments.append(segment)
        recorder.hold(
            POSE_HOLD_STEPS,
            gripper_env=gripper_env,
            phase=f"pose_{pose_index:02d}_hold",
        )
        actual = recorder.env.joint_positions()
        pose_error = float(np.linalg.norm(actual - target))
        final_pose_errors.append(pose_error)
        pose_rows.append(
            {
                "pose_index": pose_index,
                "is_return_home_pose": bool(pose_index == len(target_poses) - 1),
                "start_frame": pose_start_frame,
                "end_frame_exclusive": int(recorder.frame_index),
                "target_joint_qpos_rad": target,
                "actual_joint_qpos_after_hold_rad": actual,
                "joint_order": order,
                "gripper_env_command": gripper_env,
                "gripper_open_dataset_value": (1.0 - gripper_env) / 2.0,
                "final_pose_error_norm_rad": pose_error,
                "all_joint_segments_reached": bool(all(row["reached_tolerance"] for row in pose_segments)),
            }
        )
    return pose_rows, segment_rows, final_pose_errors


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {output}")
    if output.exists():
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    base.patch_lerobot_video_crf(demo.VIDEO_CRF)
    dataset = create_dataset(output)
    bddl_file = demo.write_geometry_bddl(
        repo_root=REPO_ROOT,
        bddl_dir=output / "bddl",
        geometry_id="joint_position_multi_pose_mu0100_hidden",
        init_xy=demo.INIT_XY,
        target_xy=demo.TARGET_XY,
        init_half_size=0.002,
        target_radius=0.025,
        target_rgba=(0.0, 0.8, 0.2, 0.0),
    )
    case = demo.build_demo_case(bddl_file)

    episode_rows: list[dict[str, Any]] = []
    metadata = {
        "created_at": dt.datetime.now().isoformat(),
        "dataset_type": "libero_mu0100_joint_position_multi_pose_100eps_lerobot_2026-07-16_hai-machine",
        "purpose": "joint-action-conditioned world-model training with sequential one-joint-at-a-time multi-pose motion",
        "episode_count_expected": EPISODE_COUNT,
        "separate_from_eef_dataset": True,
        "eef_dataset_mixed_in": False,
        "friction_mu": demo.FRICTION_MU,
        "target_visible": False,
        "camera_resolution": demo.CAMERA_RESOLUTION,
        "fps": demo.FPS,
        "video_crf": demo.VIDEO_CRF,
        "controller": "JOINT_POSITION",
        "controller_action_dim": 8,
        "controller_output_delta_rad_at_abs_action_1": JOINT_CONTROLLER_DELTA_RAD,
        "dataset_action_dim": 8,
        "dataset_action_names": [f"delta_joint_q{i}_rad" for i in range(JOINT_COUNT)] + ["gripper_open"],
        "dataset_action_units": ["rad"] * JOINT_COUNT + ["unit_interval"],
        "observation_state_dim": 22,
        "observation_state_layout": "joint_qpos[7] rad + joint_qvel[7] rad/s + eef_pos[3] m + eef_axis_angle[3] rad + gripper_qpos[2]",
        "single_joint_policy": "At most one of the seven arm-joint action dimensions is non-zero per control step.",
        "pose_policy": "Each episode visits 5 or 6 random poses followed by home; every pose uses a fresh random permutation of joints 0..6.",
        "alignment": "Each delta_joint_action_t is stored with observation_t before env.step(action_t); a terminal observation is appended.",
        "episodes": episode_rows,
    }
    manifest = {
        "created_at": metadata["created_at"],
        "output": str(output),
        "expected_episodes": EPISODE_COUNT,
        "controller": "JOINT_POSITION",
        "friction_mu": demo.FRICTION_MU,
        "episodes": episode_rows,
    }

    def autosave() -> None:
        write_json(output / "collection_manifest.json", manifest)
        base.write_dataset_metadata(
            output,
            base.to_jsonable(metadata),
            base.to_jsonable(episode_rows),
        )

    autosave()
    for collection_index in range(EPISODE_COUNT):
        parameter_seed = BASE_SEED + collection_index
        rng = np.random.default_rng(parameter_seed)
        env = LiberoJointPositionEnv(case, repo_root=REPO_ROOT, seed=SIM_SEED)
        try:
            recorder = JointEpisodeRecorder(env=env, dataset=dataset)
            pose_rows, segment_rows, final_pose_errors = run_multi_pose_episode(
                recorder,
                episode_index=collection_index,
                rng=rng,
            )
            recorder.add_terminal_observation()
            episode_index = int(dataset.meta.total_episodes)
            summary = recorder.summary(final_pose_errors=final_pose_errors, segments=segment_rows)
            if int(summary["max_nonzero_arm_joint_dims_per_action"]) > 1:
                raise RuntimeError(f"Episode {collection_index} contains multi-joint arm actions")
            dataset.save_episode()
        finally:
            env.close()

        row = {
            "episode_index": episode_index,
            "collection_index": collection_index,
            "parameter_seed": parameter_seed,
            "sim_seed": SIM_SEED,
            "friction_mu": demo.FRICTION_MU,
            "controller": "JOINT_POSITION",
            "random_pose_count": 5 + int(collection_index % 2),
            "total_pose_count_including_return_home": len(pose_rows),
            "poses": pose_rows,
            "joint_segments": segment_rows,
            "metrics": summary,
        }
        episode_rows.append(row)
        reached_segments = sum(int(segment["reached_tolerance"]) for segment in segment_rows)
        print(
            f"[{collection_index + 1:03d}/{EPISODE_COUNT:03d}] ep={episode_index:03d} "
            f"poses={len(pose_rows)} frames={summary['frames_in_lerobot_episode']} "
            f"segments={reached_segments}/{len(segment_rows)} "
            f"pose_err_mean={summary['mean_final_pose_error_norm_rad']:.3f}rad "
            f"box={summary['final_box_displacement_m'] * 100.0:.1f}cm",
            flush=True,
        )
        autosave()

    summary = {
        "completed_at": dt.datetime.now().isoformat(),
        "output": str(output),
        "episode_count": len(episode_rows),
        "expected_episode_count": EPISODE_COUNT,
        "controller": "JOINT_POSITION",
        "action_dim": 8,
        "state_dim": 22,
        "friction_mu": demo.FRICTION_MU,
        "total_lerobot_frames": int(sum(row["metrics"]["frames_in_lerobot_episode"] for row in episode_rows)),
        "total_poses": int(sum(row["total_pose_count_including_return_home"] for row in episode_rows)),
        "total_joint_segments": int(sum(len(row["joint_segments"]) for row in episode_rows)),
        "reached_joint_segments": int(
            sum(
                int(segment["reached_tolerance"])
                for row in episode_rows
                for segment in row["joint_segments"]
            )
        ),
        "mean_episode_pose_error_norm_rad": float(
            np.mean([row["metrics"]["mean_final_pose_error_norm_rad"] for row in episode_rows])
        ),
        "max_nonzero_arm_joint_dims_per_action": int(
            max(row["metrics"]["max_nonzero_arm_joint_dims_per_action"] for row in episode_rows)
        ),
    }
    write_json(output / "collection_summary.json", summary)
    autosave()
    print(json.dumps(base.to_jsonable(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()
