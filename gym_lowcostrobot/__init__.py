import os

from gymnasium.envs.registration import register

__version__ = "0.0.1"

ASSETS_PATH = os.path.join(os.path.dirname(__file__), "assets", "low_cost_robot_6dof")

register(
    id="MyPushCube-v0",
    entry_point="gym_lowcostrobot.envs:MyPushCubeEnv",
    max_episode_steps=50,
)

register(
    id="MyReachCube-v0",
    entry_point="gym_lowcostrobot.envs:MyReachCubeEnv",
    max_episode_steps=20,
)

register(
    id="LiftCube-v0",
    entry_point="gym_lowcostrobot.envs:LiftCubeEnv",
    max_episode_steps=50,
)

register(
    id="PickPlaceCube-v0",
    entry_point="gym_lowcostrobot.envs:PickPlaceCubeEnv",
    max_episode_steps=50,
)

register(
    id="PushCube-v0",
    entry_point="gym_lowcostrobot.envs:PushCubeEnv",
    max_episode_steps=50,
)

register(
    id="ReachCube-v0",
    entry_point="gym_lowcostrobot.envs:ReachCubeEnv",
    max_episode_steps=50,
)

register(
    id="StackTwoCubes-v0",
    entry_point="gym_lowcostrobot.envs:StackTwoCubesEnv",
    max_episode_steps=50,
)

register(
    id="PushCubeLoop-v0",
    entry_point="gym_lowcostrobot.envs:PushCubeLoopEnv",
    max_episode_steps=50,
)

# Register with gym_lowcostrobot prefix
register(
    id="gym_lowcostrobot/LiftCube-v0",
    entry_point="gym_lowcostrobot.envs:LiftCubeEnv",
    max_episode_steps=50,
)

register(
    id="gym_lowcostrobot/PickPlaceCube-v0",
    entry_point="gym_lowcostrobot.envs:PickPlaceCubeEnv",
    max_episode_steps=50,
)

register(
    id="gym_lowcostrobot/PushCube-v0",
    entry_point="gym_lowcostrobot.envs:PushCubeEnv",
    max_episode_steps=50,
)

register(
    id="gym_lowcostrobot/ReachCube-v0",
    entry_point="gym_lowcostrobot.envs:ReachCubeEnv",
    max_episode_steps=50,
)

register(
    id="gym_lowcostrobot/StackTwoCubes-v0",
    entry_point="gym_lowcostrobot.envs:StackTwoCubesEnv",
    max_episode_steps=50,
)

register(
    id="gym_lowcostrobot/PushCubeLoop-v0",
    entry_point="gym_lowcostrobot.envs:PushCubeLoopEnv",
    max_episode_steps=50,
)
