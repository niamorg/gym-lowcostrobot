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

    def __init__(self, distance_threshold=0.05, episodic=True, block_gripper=True, **kwargs):
        super().__init__(model_path="push_cube.xml", block_gripper=block_gripper, **kwargs)
        self.initial_joint_pos = np.zeros_like(self.joint_pos)
        self.distance_threshold = distance_threshold
        self.episodic = episodic
        
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


    def _reset(self) -> dict: 
        # Sample cube and target positions.
        while True:
            self.cube_pos[:] = self.np_random.uniform(self.CUBE_SPAWN_ZONE_MIN, self.CUBE_SPAWN_ZONE_MAX)
            self.cube_quat[:] = np.array([1.0, 0.0, 0.0, 0.0])
            self.target_pos = self.np_random.uniform(self.TARGET_SPAWN_ZONE_MIN, self.TARGET_SPAWN_ZONE_MAX)
            if np.linalg.norm(self.cube_pos - self.target_pos) > 2 * self.distance_threshold:
                break
        
        # Update visualization.
        self.model.geom("target_region").pos = self.target_pos[:]
        mujoco.mj_forward(self.model, self.data)

        # Pop the cube off the floor by advancing the simulation 1s. (TODO: not ideal).
        mujoco.mj_step(self.model, self.data, int(1 / self.model.opt.timestep))

        self.init_dist_cube_to_target = np.linalg.norm(self.cube_pos - self.target_pos)

        return {}


    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.apply_action(action)

        reward, info = self.compute_reward()
        info["is_success"] = info["on_target"]

        in_work_zone = np.all(self.WORK_ZONE_MIN < self.cube_pos[:2]) and np.all(self.cube_pos[:2] < self.WORK_ZONE_MAX)

        terminated = (info["is_success"] or not in_work_zone) if self.episodic else False

        return self.get_observation(), reward, terminated, False, info


    def compute_reward(self):
        ee_to_cube = np.linalg.norm(self.ee_pos - self.cube_pos)
        reaching_reward = 1. - np.maximum(0., (ee_to_cube - 0.03) / 0.4)

        reached = np.isclose(reaching_reward, 1.)

        cube_to_target = np.linalg.norm(self.target_pos - self.cube_pos)
        # pushing_reward = 1. - np.maximum(0., (cube_to_target - self.distance_threshold) / 0.4) # Old reward
        pushing_reward = (self.init_dist_cube_to_target - cube_to_target) / (self.init_dist_cube_to_target - self.distance_threshold)

        reward = reaching_reward + reached *  np.clip(20 * pushing_reward, -1, 20)
        
        if on_target := (cube_to_target < self.distance_threshold):
            reward = 400.0 if self.episodic else 40.0

        return reward, {"on_target": on_target, "reached": reached, "reaching_reward": reaching_reward, "pushing_reward": pushing_reward}





# ------------------------------- TEST SECTION ------------------------------- #
def test_push_cube_env(render_mode=None):
    import gymnasium as gym
    import time
    import numpy as np

    env = gym.make("MyPushCube-v0", render_mode=render_mode)
    env.action_space.seed(0)
    for seed in [0,1]:
        print(f"Seed: {seed}")
        env.reset(seed=seed)
        for _ in range(10 * (5 * seed + 1)):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample() if seed == 0 else (env.unwrapped.cube_pos - env.unwrapped.ee_pos).astype(np.float32))
            print(reward, 2, terminated, info["reaching_reward"], info["pushing_reward"], {k:round(float(v),2) for k,v in info.items()})
            if render_mode == "human":
                time.sleep(0.01)
    
    print("--- Test pushing reward ---")
    env.reset(seed=0)
    cube_init_pos = env.unwrapped.data.qpos[6:6+3].copy()
    target_pos = env.unwrapped.target_pos.copy()
    for t in np.linspace(0, 1, 10):
        env.unwrapped.data.qpos[6:6+3] = cube_init_pos + t * (target_pos - cube_init_pos)
        mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        print(np.round(reward, 2), terminated, {k:round(float(v),2) for k,v in info.items()})
        if render_mode == "human":
            time.sleep(0.1) 
    env.close()


if __name__ == "__main__":
    test_push_cube_env(render_mode="human")
# ---------------------------------------------------------------------------- #