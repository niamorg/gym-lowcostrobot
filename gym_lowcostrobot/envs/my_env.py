from gymnasium.envs.mujoco import MujocoEnv
from gym_lowcostrobot import ASSETS_PATH
import numpy as np
import os


class MyEnv(MujocoEnv):

    def __init__(self, **kwargs):
        MujocoEnv.__init__(
            self,
            os.path.join(ASSETS_PATH, "push_cube.xml"),
            frame_skip=5,
            observation_space=None,
            **kwargs
        )

    def step(self, action):
        return None, 0, False, False, {}

    def reset_model(self) -> np.ndarray:
        # reset the joint positions to the initial positions
        # choose new random cube position
        # choose new random target position
        # mujoco.mj_forward(self.model, self.data)
        return None


