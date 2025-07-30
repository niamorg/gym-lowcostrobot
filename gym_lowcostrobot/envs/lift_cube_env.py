from gym_lowcostrobot.envs.base_env import BaseEnv
from gymnasium.spaces import Box
import numpy as np
import mujoco

class LiftCubeEnv(BaseEnv):

    CUBE_SPAWN_ZONE_MIN = np.array([-0.15, 0.015, 0])
    CUBE_SPAWN_ZONE_MAX = np.array([0.15, 0.20, 0])
    WORK_ZONE_MIN = np.array([-0.30, -0.20])
    WORK_ZONE_MAX = np.array([0.30, 0.23])

    def __init__(self, height_threshold=0.1, episodic=False, **kwargs):
        super().__init__(model_path="lift_cube.xml", **kwargs)
        self.initial_joint_pos = np.zeros_like(self.joint_pos)
        self.height_threshold = height_threshold
        self.episodic = episodic
        
        # Complete the observation space.
        self.observation_space.spaces['cube_pos'] = Box(low=-np.inf, high=np.inf, shape=(3,))
        self.observation_space.spaces['cube_vel'] = Box(low=-np.inf, high=np.inf, shape=(3,))
        self.observation_space.spaces['ee_to_cube'] = Box(low=-np.inf, high=np.inf, shape=(3,))
        self.observation_space.spaces['cube_height'] = Box(low=-np.inf, high=np.inf, shape=(1,))

        # Some aliases (views of arrays).
        self.cube_pos = self.data.qpos[6:6+3]
        self.cube_vel = self.data.qvel[6:6+3]
        self.cube_quat = self.data.qpos[6+3:6+7]


    def get_observation(self) -> dict[str, np.ndarray]:
        obs = self.get_robot_observation()
        obs['cube_pos'] = self.cube_pos.astype(np.float32)
        obs['cube_vel'] = self.cube_vel.astype(np.float32)
        obs['ee_to_cube'] = (self.cube_pos - self.ee_pos).astype(np.float32)
        obs['cube_height'] = np.array([self.cube_pos[2]], dtype=np.float32)
        return obs


    def _reset(self) -> dict: 
        # Sample cube and target positions.
        self.cube_pos[:] = self.np_random.uniform(self.CUBE_SPAWN_ZONE_MIN, self.CUBE_SPAWN_ZONE_MAX)
        self.cube_quat[:] = np.array([1.0, 0.0, 0.0, 0.0])

        # Pop the cube off the floor by advancing the simulation 1s. (TODO: not ideal).
        mujoco.mj_step(self.model, self.data, int(1 / self.model.opt.timestep))
        mujoco.mj_forward(self.model, self.data)

        self.cube_init_z = self.cube_pos[2]

        return {}


    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.apply_action(action)

        reward, info = self.compute_reward()
        info["is_success"] = info["high_enough"]

        in_work_zone = np.all(self.WORK_ZONE_MIN < self.cube_pos[:2]) and np.all(self.cube_pos[:2] < self.WORK_ZONE_MAX)

        terminated = bool(info["is_success"] or not in_work_zone) if self.episodic else False

        return self.get_observation(), reward, terminated, False, info


    def compute_reward(self):
        ee_to_cube = np.linalg.norm(self.ee_pos - self.cube_pos)
        reaching_reward = 1. - np.maximum(0., (ee_to_cube - 0.03) / 0.4)

        reached = np.isclose(reaching_reward, 1.)

        lifting_reward = (self.cube_pos[2] - self.cube_init_z) / (self.height_threshold - self.cube_init_z)
        
        reward = reaching_reward + reached * 20 * np.clip(lifting_reward, 0, 1)

        if high_enough := (self.cube_pos[2] > self.height_threshold):
            reward = 400.0 if self.episodic else 40.0

        return reward, {"high_enough": high_enough, "reached": reached, "reaching_reward": reaching_reward, "lifting_reward": lifting_reward}





# ------------------------------- TEST SECTION ------------------------------- #
def test_lift_cube_env(render_mode=None):
    import gymnasium as gym
    import time
    import numpy as np

    env = gym.make("LiftCube-v0", render_mode=render_mode, max_episode_steps=50)
    env.action_space.seed(0)
    for seed in [0,1]:
        print(f"\n ------ Seed: {seed} {"random" if seed == 0 else "ee towards cube"} ------")
        env.reset(seed=seed)
        for _ in range(10 * (4 * seed + 1)):
            action = env.action_space.sample() if seed == 0 else np.append(env.unwrapped.cube_pos - env.unwrapped.ee_pos, 0).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            # print(f"{terminated=} {reward=:.5f}", {k:(round(float(v),5) if isinstance(v, np.floating) else int(v)) for k,v in info.items()})
            if render_mode == "human":
                time.sleep(0.01)
        if seed == 1:
            assert info["reached"], "The end-effector should have reached the cube."
            print("="*100)
            print("✓ TEST: Reaching reward: passed")
            print("="*100)
    
    for episodic in [False, True]:
        print(f"--- Test lifting reward episodic={episodic} ---")
        if episodic:
            env = gym.make("LiftCube-v0", render_mode=render_mode, episodic=True)
        env.reset(seed=0)
        env.unwrapped.model.opt.gravity[2] = 0.0
        cube_init_pos = env.unwrapped.data.qpos[6:6+3].copy()
        target_pos = cube_init_pos.copy() + np.array([0, 0, 0.12])
        for t in np.linspace(0, 1, 10):
            env.unwrapped.data.qpos[6:6+3] = cube_init_pos + t * (target_pos - cube_init_pos)
            mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            print(f"{terminated=} {reward=:.5f}", {k:(round(float(v),5) if isinstance(v, np.floating) else int(v)) for k,v in info.items()})
            if render_mode == "human":
                time.sleep(0.1)
            if terminated or truncated:
                assert info["high_enough"], "The cube should have been lifted."
                break
        assert info["high_enough"], "The cube should have been lifted."
        env.close()
    print("="*100)
    print("✓ TEST: Lifting reward: passed")
    print("="*100)

if __name__ == "__main__":
    test_lift_cube_env(render_mode=None)
# ---------------------------------------------------------------------------- #