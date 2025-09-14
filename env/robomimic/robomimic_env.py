import gym
import h5py
import numpy as np
from robomimic.config import config_factory
from robomimic.utils import env_utils, file_utils, obs_utils
from robosuite.wrappers import GymWrapper

def get_dataset_path(prefix, env_name):
    """Get dataset path for given environment name and data prefix"""
    dataset_paths = {
        "can-mh": f"{prefix}/can/mh/low_dim.hdf5",
        "can-ph": f"{prefix}/can/ph/low_dim.hdf5",
        "lift-mh": f"{prefix}/lift/mh/low_dim.hdf5",
        "lift-ph": f"{prefix}/lift/ph/low_dim.hdf5",
        "square-mh": f"{prefix}/square/mh/low_dim.hdf5",
        "square-ph": f"{prefix}/square/ph/low_dim.hdf5",
    }
    return dataset_paths[env_name]


class RobomimicEnv(gym.Env):
    def __init__(self, env_name, data_prefix="/home/ubuntu/minghao/data/robomimic"):
        super().__init__()
        path = get_dataset_path(data_prefix, env_name)
        env_meta = file_utils.get_env_metadata_from_dataset(path)
        env = env_utils.create_env_from_metadata(
            env_meta=env_meta,
            env_name=env_meta["env_name"],
            render=False,
            render_offscreen=False,
            use_image_obs=False
        ).env
        env.ignore_done = False
        env._max_episode_steps = env.horizon
        keys = [
            "object-state",
            "robot0_joint_pos",
            "robot0_joint_pos_cos",
            "robot0_joint_pos_sin",
            "robot0_joint_vel",
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "robot0_gripper_qvel",
        ]
        self.env = GymWrapper(env, keys=keys)
        self._max_episode_steps = self.env.horizon
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def step(self, action: np.ndarray):
        
        observation, reward, done, info = self.env.step(action)

        if self.env._check_success():
            done = True

        return observation, reward, done, info

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def render(self, mode="human"):
        return self.env.render(mode)