from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .cases import DynamicCarrierCase
from .trajectories import ProceduralTrajectory


@dataclass
class DynamicStepInfo:
    case_id: str
    t: float
    carrier_xy: list[float]
    carrier_velocity_xy: list[float]
    payload_xyz: list[float]
    payload_detached: bool
    payload_attached_to_gripper: bool
    custom_success: bool


class DynamicCarrierEnv:
    """LIBERO environment wrapper with a scripted moving carrier.

    This wrapper keeps the base BDDL task intact and kinematically moves a
    carrier object according to a procedural trajectory. Until the payload is
    grasped, it is kept at a fixed offset from the carrier. The
    collector uses this as a clean first benchmark before moving to fully
    physical platform/contact dynamics.
    """

    def __init__(self, base_env: Any, case: DynamicCarrierCase):
        self.base_env = base_env
        self.case = case
        self.trajectory = ProceduralTrajectory(case.motion)
        self.dt = 1.0 / float(case.control_freq)
        self.step_count = 0
        self.payload_detached = False
        self.payload_attached_to_gripper = False
        self._carrier_z: float | None = None
        self._carrier_quat: np.ndarray | None = None
        self._payload_quat: np.ndarray | None = None
        self._payload_offset: np.ndarray | None = None
        self._payload_grasp_offset: np.ndarray | None = None
        self._release_payload_pos: np.ndarray | None = None
        self._last_obs: dict[str, Any] | None = None

    @property
    def t(self) -> float:
        return self.step_count * self.dt

    @property
    def inner_env(self) -> Any:
        return self.base_env.env

    def reset(self, init_state: np.ndarray | None = None) -> dict[str, Any]:
        self.step_count = 0
        self.payload_detached = False
        self.payload_attached_to_gripper = False
        self._payload_grasp_offset = None
        self._release_payload_pos = None
        self.base_env.reset()
        if init_state is not None:
            obs = self.base_env.set_init_state(init_state)
        else:
            obs = self._refresh_obs()

        carrier_pos, carrier_quat = self.get_object_pose(self.case.carrier_name)
        payload_pos, payload_quat = self.get_object_pose(self.case.payload_name)
        self._carrier_z = float(carrier_pos[2])
        self._carrier_quat = carrier_quat.copy()
        self._payload_quat = payload_quat.copy()

        if self.case.payload_offset_xyz is None:
            offset = payload_pos - carrier_pos
            if np.linalg.norm(offset[:2]) > 0.18:
                offset[:2] = 0.0
            self._payload_offset = offset
        else:
            self._payload_offset = np.asarray(self.case.payload_offset_xyz, dtype=np.float64)

        self._apply_kinematic_motion(0.0)
        obs = self._refresh_obs()
        self._last_obs = obs
        return obs

    def step(self, action: np.ndarray | list[float]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        action_arr = np.asarray(action, dtype=np.float64)
        obs, reward, base_done, info = self.base_env.step(action_arr)

        self._update_payload_attachment(action_arr, obs)
        self.step_count += 1
        self._apply_kinematic_motion(self.t, obs)
        obs = self._refresh_obs()
        custom_success = self.check_success()

        dyn_info = self._build_step_info(custom_success)
        info = dict(info)
        info["dynamic_carrier"] = dyn_info.__dict__
        self._last_obs = obs
        return obs, float(reward), bool(base_done or custom_success), info

    def close(self) -> None:
        return self.base_env.close()

    def get_sim_state(self) -> np.ndarray:
        return self.base_env.get_sim_state()

    def get_object_pose(self, object_name: str) -> tuple[np.ndarray, np.ndarray]:
        obj = self.inner_env.get_object(object_name)
        if not getattr(obj, "joints", None):
            body_id = self.inner_env.sim.model.body_name2id(obj.root_body)
            pos = np.asarray(self.inner_env.sim.data.body_xpos[body_id], dtype=np.float64)
            quat = np.asarray(self.inner_env.sim.data.body_xquat[body_id], dtype=np.float64)
            return pos.copy(), quat.copy()
        qpos = np.asarray(self.inner_env.sim.data.get_joint_qpos(obj.joints[-1]), dtype=np.float64)
        if qpos.shape[0] < 7:
            raise ValueError(f"Object {object_name} does not expose a free-joint qpos: {qpos}")
        return qpos[:3].copy(), qpos[3:7].copy()

    def set_object_pose(self, object_name: str, pos: np.ndarray, quat: np.ndarray | None = None) -> None:
        obj = self.inner_env.get_object(object_name)
        if not getattr(obj, "joints", None):
            body_id = self.inner_env.sim.model.body_name2id(obj.root_body)
            self.inner_env.sim.model.body_pos[body_id] = np.asarray(pos, dtype=np.float64)
            if quat is not None:
                self.inner_env.sim.model.body_quat[body_id] = np.asarray(quat, dtype=np.float64)
            return
        current = np.asarray(self.inner_env.sim.data.get_joint_qpos(obj.joints[-1]), dtype=np.float64)
        next_qpos = current.copy()
        next_qpos[:3] = np.asarray(pos, dtype=np.float64)
        if quat is not None:
            next_qpos[3:7] = np.asarray(quat, dtype=np.float64)
        self.inner_env.sim.data.set_joint_qpos(obj.joints[-1], next_qpos)

    def predict_carrier_xy(self, future_t: float) -> np.ndarray:
        return self.trajectory.sample(float(future_t)).xy.copy()

    def predict_payload_xy(self, future_t: float) -> np.ndarray:
        offset = self._payload_offset if self._payload_offset is not None else np.zeros(3)
        return self.predict_carrier_xy(future_t) + offset[:2]

    def predict_payload_grasp_xy(self, future_t: float) -> np.ndarray:
        grasp_offset = np.asarray(self.case.grasp_offset_xyz, dtype=np.float64)
        return self.predict_payload_xy(future_t) + grasp_offset[:2]

    def carrier_velocity_xy(self, t: float | None = None) -> np.ndarray:
        return self.trajectory.sample(self.t if t is None else float(t)).velocity_xy.copy()

    def payload_position(self) -> np.ndarray:
        pos, _ = self.get_object_pose(self.case.payload_name)
        return pos

    def payload_grasp_position(self) -> np.ndarray:
        return self.payload_position() + np.asarray(self.case.grasp_offset_xyz, dtype=np.float64)

    def carrier_position(self) -> np.ndarray:
        pos, _ = self.get_object_pose(self.case.carrier_name)
        return pos

    def eef_position(self, obs: dict[str, Any] | None = None) -> np.ndarray:
        source = self._last_obs if obs is None else obs
        if source is None:
            source = self._refresh_obs()
        return np.asarray(source["robot0_eef_pos"], dtype=np.float64)

    def check_success(self) -> bool:
        released = self.payload_detached and not self.payload_attached_to_gripper
        if not released:
            return False
        payload_pos = self.payload_position()
        if self._success_pose_ok(payload_pos):
            return True
        if self._release_payload_pos is not None and self._success_pose_ok(self._release_payload_pos):
            return True
        return False

    def _success_pose_ok(self, payload_pos: np.ndarray) -> bool:
        carrier_pos = self.carrier_position()
        target_xy = np.asarray(self.case.target_xy, dtype=np.float64)
        xy_ok = np.linalg.norm(payload_pos[:2] - target_xy) <= float(self.case.target_radius)
        clear_carrier = (
            np.linalg.norm(payload_pos[:2] - carrier_pos[:2])
            >= float(self.case.platform_radius + self.case.object_radius)
        )
        if self.case.target_z is None:
            if self._carrier_z is None:
                z_ok = True
            else:
                min_z = float(self._carrier_z) - 0.20
                max_z = float(self._carrier_z) + 0.08
                z_ok = min_z <= float(payload_pos[2]) <= max_z
        else:
            z_ok = abs(float(payload_pos[2]) - float(self.case.target_z)) <= float(
                self.case.target_z_tolerance
            )
        return bool(xy_ok and clear_carrier and z_ok)

    def _apply_kinematic_motion(self, t: float, obs: dict[str, Any] | None = None) -> None:
        if self._carrier_z is None:
            return
        state = self.trajectory.sample(t)
        carrier_pos = np.asarray([state.xy[0], state.xy[1], self._carrier_z], dtype=np.float64)
        self.set_object_pose(self.case.carrier_name, carrier_pos, self._carrier_quat)

        if self.payload_attached_to_gripper:
            eef_pos = self.eef_position(obs)
            grasp_offset = np.asarray(self.case.grasp_offset_xyz, dtype=np.float64)
            payload_pos = eef_pos - grasp_offset
            self.set_object_pose(self.case.payload_name, payload_pos, self._payload_quat)
        elif not self.payload_detached:
            offset = self._payload_offset if self._payload_offset is not None else np.zeros(3)
            payload_pos = carrier_pos + offset
            self.set_object_pose(self.case.payload_name, payload_pos, self._payload_quat)

        self.inner_env.sim.forward()

    def _update_payload_attachment(self, action: np.ndarray, obs: dict[str, Any]) -> None:
        if action.shape[0] == 0:
            return

        gripper_cmd = float(action[-1])
        if self.payload_attached_to_gripper:
            if gripper_cmd < -0.2:
                self._release_payload_pos = self.payload_position().copy()
                self.payload_attached_to_gripper = False
                self.payload_detached = True
            return

        if self.payload_detached:
            return

        grasp_pos = self.payload_grasp_position()
        eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        gripper_closing = bool(gripper_cmd > 0.2)
        attach_distance = min(float(self.case.grasp_release_distance), 0.035)
        attach_height = min(float(self.case.grasp_release_height), 0.05)
        xy_close = np.linalg.norm(grasp_pos[:2] - eef_pos[:2]) <= attach_distance
        z_close = abs(float(grasp_pos[2] - eef_pos[2])) <= attach_height
        if gripper_closing and xy_close and z_close:
            self.payload_detached = True
            self.payload_attached_to_gripper = True
            self._release_payload_pos = None
            self._payload_grasp_offset = np.asarray(self.case.grasp_offset_xyz, dtype=np.float64)

    def _refresh_obs(self) -> dict[str, Any]:
        self.inner_env.sim.forward()
        self.inner_env._post_process()
        self.inner_env._update_observables(force=True)
        return self.inner_env._get_observations()

    def _build_step_info(self, custom_success: bool) -> DynamicStepInfo:
        state = self.trajectory.sample(self.t)
        payload = self.payload_position()
        return DynamicStepInfo(
            case_id=self.case.case_id,
            t=float(self.t),
            carrier_xy=state.xy.astype(float).tolist(),
            carrier_velocity_xy=state.velocity_xy.astype(float).tolist(),
            payload_xyz=payload.astype(float).tolist(),
            payload_detached=bool(self.payload_detached),
            payload_attached_to_gripper=bool(self.payload_attached_to_gripper),
            custom_success=bool(custom_success),
        )


def create_libero_env_for_case(
    case: DynamicCarrierCase,
    *,
    repo_root: str | Path,
    camera_resolution: int = 128,
    seed: int | None = None,
) -> tuple[Any, np.ndarray | None, str]:
    """Create a LIBERO OffScreenRenderEnv plus optional benchmark init state."""
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    repo_root = Path(repo_root)
    init_state = None
    task_description = f"dynamic carrier case {case.case_id}"

    if case.bddl_file is not None:
        bddl_file = case.resolved_bddl_file(repo_root)
        if bddl_file is None or not bddl_file.exists():
            raise FileNotFoundError(f"BDDL file for case {case.case_id} not found: {bddl_file}")
    elif case.suite_name is not None and case.task_id is not None:
        bench = benchmark.get_benchmark_dict()[case.suite_name]()
        task = bench.get_task(int(case.task_id))
        task_description = task.language
        bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        init_states = bench.get_task_init_states(int(case.task_id))
        init_state = init_states[0]
    else:
        raise ValueError(
            f"Case {case.case_id} must provide either bddl_file or suite_name/task_id."
        )

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=int(camera_resolution),
        camera_widths=int(camera_resolution),
    )
    if seed is not None:
        env.seed(int(seed))
    return env, init_state, task_description
