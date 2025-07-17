from gym_lowcostrobot.envs.base_env import BaseEnv
from gymnasium.spaces import Box
import numpy as np
import mujoco

class MyReachCubeEnv(BaseEnv):

    CUBE_SPAWN_ZONE_MIN = np.array([-0.15, 0.015, 0])
    CUBE_SPAWN_ZONE_MAX = np.array([0.15, 0.20, 0])

    def __init__(self, distance_threshold=0.05, episodic=True, block_gripper=True, **kwargs):
        super().__init__(model_path="reach_cube.xml", block_gripper=block_gripper, **kwargs)
        self.initial_joint_pos = np.zeros_like(self.joint_pos)
        self.distance_threshold = distance_threshold
        self.episodic = episodic

        # Complete the observation space.
        self.observation_space.spaces['cube_pos'] = Box(low=-np.inf, high=np.inf, shape=(3,))
        self.observation_space.spaces['cube_vel'] = Box(low=-np.inf, high=np.inf, shape=(3,))
        self.observation_space.spaces['ee_to_cube'] = Box(low=-np.inf, high=np.inf, shape=(3,))

        # Some aliases (views of arrays).
        self.cube_pos = self.data.qpos[6:6+3]
        self.cube_vel = self.data.qvel[6:6+3]
        self.cube_quat = self.data.qpos[6+3:6+7]


    def get_observation(self) -> dict[str, np.ndarray]:
        obs = self.get_robot_observation()
        obs['cube_pos'] = self.cube_pos.astype(np.float32)
        obs['cube_vel'] = self.cube_vel.astype(np.float32)
        obs['ee_to_cube'] = (self.cube_pos - self.ee_pos).astype(np.float32)
        return obs


    def _reset(self) -> dict: 
        # Sample cube positions.
        self.cube_pos[:] = self.np_random.uniform(self.CUBE_SPAWN_ZONE_MIN, self.CUBE_SPAWN_ZONE_MAX)
        self.cube_quat[:] = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Set derived quantities.
        mujoco.mj_forward(self.model, self.data)

        # Pop the cube off the floor by advancing the simulation 1s. (TODO: not ideal).
        mujoco.mj_step(self.model, self.data, int(1 / self.model.opt.timestep))

        self.count = 0

        return {}


    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.apply_action(action)

        reward, info = self.compute_reward()

        self.count = int(info["reached"]) * (self.count + 1)
        info["is_success"] = info["reached"] if self.episodic else (self.count >= 5)  # TODO: hardcoded value of 5...

        return self.get_observation(), reward, (info["is_success"] if self.episodic else False), False, info


    def compute_reward(self):
        ee_to_cube = np.linalg.norm(self.ee_pos - self.cube_pos)
        reward = 1 - np.maximum(0, (ee_to_cube - self.distance_threshold) / 0.5)

        if reached := (reward == 1):
            reward = 100.0 if self.episodic else 10.0

        return reward, {"reached": reached}
