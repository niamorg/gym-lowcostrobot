from gym_lowcostrobot.envs.base_env import BaseEnv
from gymnasium.spaces import Box
import numpy as np
import mujoco

class MyPushCubeEnv(BaseEnv):

    CUBE_SPAWN_ZONE_MIN = np.array([-0.15, 0.015, 0])
    CUBE_SPAWN_ZONE_MAX = np.array([0.15, 0.20, 0])
    TARGET_SPAWN_ZONE_MIN = np.array([-0.15, 0.015, 0])
    TARGET_SPAWN_ZONE_MAX = np.array([0.15, 0.20, 0])
    WORK_ZONE_MIN = np.array([-0.30, -0.20])
    WORK_ZONE_MAX = np.array([0.30, 0.23])

    def __init__(self, distance_threshold=0.05, **kwargs):
        super().__init__(model_path="push_cube.xml", **kwargs)
        self.initial_joint_pos = np.zeros_like(self.joint_pos)
        self.distance_threshold = distance_threshold
        
        # Complete the observation space.
        self.observation_space.spaces['cube_pos'] = Box(low=-np.inf, high=np.inf, shape=(3,))
        self.observation_space.spaces['cube_vel'] = Box(low=-np.inf, high=np.inf, shape=(3,))
        self.observation_space.spaces['ee_to_cube'] = Box(low=-np.inf, high=np.inf, shape=(3,))
        self.observation_space.spaces['cube_to_target'] = Box(low=-np.inf, high=np.inf, shape=(3,))
        self.observation_space.spaces['target_pos'] = Box(low=-np.inf, high=np.inf, shape=(2,))

        # Some aliases (views of arrays).
        self.cube_pos = self.data.qpos[6:6+3]
        self.cube_vel = self.data.qvel[6:6+3]
        self.cube_quat = self.data.qpos[6+3:6+7]


    def get_observation(self) -> dict[str, np.ndarray]:
        obs = self.get_robot_observation()
        obs['target_pos'] = self.target_pos[:2].astype(np.float32)
        obs['cube_pos'] = self.cube_pos.astype(np.float32)
        obs['cube_vel'] = self.cube_vel.astype(np.float32)
        obs['ee_to_cube'] = (self.cube_pos - self.ee_pos).astype(np.float32)
        obs['cube_to_target'] = (self.target_pos - self.cube_pos).astype(np.float32)
        return obs


    def _reset(self) -> tuple[np.ndarray, dict]:
        self.joint_pos[:] = self.initial_joint_pos
 
        # Sample cube and target positions.
        while True:
            self.cube_pos[:] = self.np_random.uniform(self.CUBE_SPAWN_ZONE_MIN, self.CUBE_SPAWN_ZONE_MAX)
            self.cube_quat[:] = np.array([1.0, 0.0, 0.0, 0.0])
            self.target_pos = self.np_random.uniform(self.TARGET_SPAWN_ZONE_MIN, self.TARGET_SPAWN_ZONE_MAX)
            if np.linalg.norm(self.cube_pos - self.target_pos) > 2 * self.distance_threshold:
                break
        
        # Update visualization.
        self.model.geom("target_region").pos = self.target_pos[:]
        
        # Set derived quantities.
        mujoco.mj_forward(self.model, self.data)
        for _ in range(int(0.2 / self.model.opt.timestep)):  # To pop the cube off the floor (TODO: not ideal).
            mujoco.mj_step(self.model, self.data)

        return self.get_observation(), {}


    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.apply_action(action)

        reward, info = self.compute_reward()
        info["is_success"] = info["on_target"]

        in_work_zone = np.all(self.WORK_ZONE_MIN < self.cube_pos[:2]) and np.all(self.cube_pos[:2] < self.WORK_ZONE_MAX)

        return self.get_observation(), reward, (info["is_success"] or not in_work_zone), False, info


    def compute_reward(self):
        ee_to_cube = np.linalg.norm(self.ee_pos - self.cube_pos)
        reaching_reward = 1 - np.maximum(0, (ee_to_cube - 0.02) / 0.4)

        reached = (reaching_reward == 1)

        cube_to_target = np.linalg.norm(self.target_pos - self.cube_pos)
        pushing_reward = 1 - np.maximum(0, (cube_to_target - self.distance_threshold) / 0.4)

        reward = reaching_reward + reached * 20 * pushing_reward
        
        if on_target := (cube_to_target < self.distance_threshold):
            reward = 400

        return reward, {"on_target": on_target, "reached": reached, "reaching_reward": reaching_reward, "pushing_reward": pushing_reward}
