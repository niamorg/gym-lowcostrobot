import os
import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium.spaces import Box, Dict

from gym_lowcostrobot import ASSETS_PATH
from gym_lowcostrobot.inverse_kinematics import InverseKinematicsZYYYX, damped_least_squares_ik
from gym_lowcostrobot.utils import wrap_neg_pi_pi, rotmat_to_eulers


class BaseEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        model_path: str,
        observation_cameras: list[str] = [],
        observation_images_shape: tuple[int, int] = (96, 96),
        action_mode: str = "ee",
        block_gripper: bool = False,
        mujoco_steps: int = 100,
        mujoco_timestep: float = 0.002,
        render_mode: str | None = None,
        end_effector_site: str = "end_effector_site",
    ):
        self.model = mujoco.MjModel.from_xml_path(os.path.join(ASSETS_PATH, model_path))
        self.data = mujoco.MjData(self.model)
        self.ik = InverseKinematicsZYYYX(self.model)

        # Simulation parameters.
        self.model.opt.timestep = mujoco_timestep
        self.mujoco_steps = mujoco_steps  # number of MuJoCo simulation steps per environment step.
        self.metadata["render_fps"] = round(1 / (mujoco_timestep * mujoco_steps))

        # (Base) action space.
        assert action_mode in ["ee", "ee_pose", "joint"], f"Invalid action mode: {action_mode}."
        self.action_mode = action_mode
        self.block_gripper = block_gripper
        action_shape = (3 if action_mode == "ee" else 5) + int(not block_gripper)
        self.action_space = Box(low=-1.0, high=1.0, shape=(action_shape,), dtype=np.float32)

        # (Base) observation spaces.
        self.num_dof = (5 if block_gripper else 6)
        observation_subspaces = {
            'joint_pos': Box(low=-np.pi, high=np.pi, shape=(self.num_dof,)),
            'joint_vel': Box(low=-np.inf, high=np.inf, shape=(self.num_dof,)),
            'gripper': Box(low=-np.inf, high=np.inf),    # TODO: (niamorg) range
            'ee_pos': Box(low=-np.inf, high=np.inf, shape=(3,)),
            'ee_vel': Box(low=-np.inf, high=np.inf, shape=(3,)),
            'ee_quat': Box(low=-1.0, high=1.0, shape=(4,)),
            'ee_roll': Box(low=-np.pi, high=np.pi),
            'ee_pitch': Box(low=-np.pi/2, high=np.pi/2),
            'ee_yaw': Box(low=-np.pi, high=np.pi),
            # 'ee_axis_angle': Box(low=-np.inf, high=np.inf, shape=(3,)),
        }
        self.cameras = [cam.split('_')[1] for cam in observation_cameras]
        for cam in self.cameras:
            observation_subspaces[f'image_{cam}'] = Box(0, 255, shape=(3, *observation_images_shape), dtype=np.uint8)

        self.observation_images_shape = observation_images_shape
        if self.cameras:
            self.renderer = mujoco.Renderer(self.model, height=observation_images_shape[0], width=observation_images_shape[1])

        self.observation_space = Dict(observation_subspaces)

        # Some aliases (views of arrays).
        self.ee_id = self.model.site(end_effector_site).id
        self.ee_pos = self.data.site(self.ee_id).xpos
        self.ee_rot = self.data.site(self.ee_id).xmat.reshape(3,3)
        self.joint_pos = self.data.qpos[:self.num_dof]  # qpos = [q1, q2, q3, q4, q5, gripper, ...]
        self.joint_vel = self.data.qvel[:self.num_dof]  # qvel = [dq1, dq2, dq3, dq4, dq5, dgripper, ...]

        # Render utilities.
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        if render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False)
        elif self.render_mode == "rgb_array":
            self.rgb_array_renderer = mujoco.Renderer(self.model, height=640, width=640)


    def apply_action(self, action: np.ndarray):
        if not self.action_space.contains(action):
            raise ValueError(f"Action out of action space ({self.action_space}): {action = }")
        
        if self.action_mode == "ee":
            ee_action = action[:3]
            target_ee_pos = self.ee_pos + ee_action * 0.05
            # target_ee_pos[2] = np.maximum(0, target_ee_pos[2]) # TODO: ROMAIN: questionable...
            self.data.ctrl[:-1] = damped_least_squares_ik(self.model, self.data, target_ee_pos, self.ee_id)
            self.data.ctrl[-1] = np.clip(np.pi * action[-1], *self.model.jnt_range[-1].T) if not self.block_gripper else 0.0

        elif self.action_mode == "ee_pose":
            dxee, droll, dpitch = action[:3], action[3], action[4]
            
            target_ee_pos = self.ee_pos + dxee * 0.04
            target_ee_pos[2] = np.maximum(0, target_ee_pos[2])

            roll, pitch, _ = rotmat_to_eulers(self.ee_rot)
            target_roll, target_pitch = wrap_neg_pi_pi(roll + 0.28 * droll), pitch + 0.28 * dpitch    # 16° max.
            if np.abs(target_pitch) > np.pi/2:
                target_pitch = np.sign(target_pitch) * np.pi - target_pitch

            joints, _ = self.ik.solve(target_ee_pos, target_roll, target_pitch)
            self.data.ctrl[:-1] = joints
            self.data.ctrl[-1] = np.clip(np.pi * action[-1], *self.model.jnt_range[-1].T) if not self.block_gripper else 0.0

        elif self.action_mode == "joint":
            raise NotImplementedError("Joint action mode not implemented.")

        if self.render_mode == "human":
            for _ in range(self.mujoco_steps):
                mujoco.mj_step(self.model, self.data)
                self.viewer.sync()
        else:
            mujoco.mj_step(self.model, self.data, nstep=self.mujoco_steps)

        mujoco.mj_forward(self.model, self.data)


    def get_robot_observation(self) -> dict[str, np.ndarray]:
        jacp, jacr = np.zeros((3, self.model.nv)), None
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_id)
        ee_vel = jacp[:,:5] @ self.data.qvel[:5]

        mujoco.mju_mat2Quat(ee_quat := np.empty(4), self.ee_rot.reshape(9))
        roll, pitch, yaw = rotmat_to_eulers(self.ee_rot)
        observation = {
            "joint_pos": self.joint_pos.astype(np.float32),
            "joint_vel": self.joint_vel.astype(np.float32),
            "gripper": np.array([self.joint_pos[-1]], dtype=np.float32),
            "ee_pos": self.ee_pos.astype(np.float32),
            "ee_vel": ee_vel.astype(np.float32),
            "ee_quat": ee_quat.astype(np.float32),
            "ee_roll": np.array([roll], dtype=np.float32),
            "ee_pitch": np.array([pitch], dtype=np.float32),
            "ee_yaw": np.array([yaw], dtype=np.float32),
        }
        for cam in self.cameras:
            self.renderer.update_scene(self.data, camera=f"camera_{cam}")
            observation[f"image_{cam}"] = self.renderer.render().transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)

        return observation


    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        info = self._reset()

        if self.render_mode == "human":
            self.viewer.sync()
        
        return self.get_observation(), info
    

    def render(self):
        if self.render_mode == "human":
            self.viewer.sync()
        elif self.render_mode == "rgb_array":
            self.rgb_array_renderer.update_scene(self.data, camera="camera_vizu")
            return self.rgb_array_renderer.render()


    def close(self):
        if self.render_mode == "human":
            self.viewer.close()
        if self.render_mode == "rgb_array":
            self.rgb_array_renderer.close()
        if self.cameras:
            self.renderer.close()

# --- Methods to override --- #
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        raise NotImplementedError

    def get_observation(self) -> dict[str, np.ndarray] | np.ndarray:
        raise NotImplementedError

    def _reset(self) -> dict:
        raise NotImplementedError
# --------------------------- #