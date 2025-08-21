import os
import numpy as np
import gym
from env.pusht.pusht_env import PushTEnv
from utils import aggregate_dct

class PushTWrapper(PushTEnv):
    def __init__(
            self, 
            with_velocity=True,
            with_target=True,
        ):
        super().__init__(
            with_velocity=with_velocity,
            with_target=with_target, 
        )
        self.action_dim = self.action_space.shape[0]
    
    def reset(self):
        """Override reset to return full state as proprio for robomimic compatibility"""
        self._setup()
        if self.block_cog is not None:
            self.block.center_of_gravity = self.block_cog
        if self.damping is not None:
            self.space.damping = self.damping

        # use legacy RandomState for compatibility
        state = self.reset_to_state
        if state is None:
            rs = self.random_state
            if self.with_velocity:
                state = np.array(
                    [
                        rs.randint(50, 450),
                        rs.randint(50, 450),
                        rs.randint(100, 400),
                        rs.randint(100, 400),
                        rs.randn() * 2 * np.pi - np.pi,
                        0,  # set random velocity to 0
                        0,  # set random velocity to 0
                    ]
                )
            else:
                state = np.array(
                    [
                        rs.randint(50, 450),
                        rs.randint(50, 450),
                        rs.randint(100, 400),
                        rs.randint(100, 400),
                        rs.randn() * 2 * np.pi - np.pi,
                    ]
                )
        self._set_state(state)

        self.coverage_arr = []
        state = self._get_obs()
        visual = self._render_frame("rgb_array")
        
        # For robomimic compatibility, return full state as proprio
        proprio = state  # Full 7D state: [agent_x, agent_y, block_x, block_y, angle, vel_x, vel_y]
        
        observation = {
            "visual": visual,
            "proprio": proprio
        }
        return observation, state

    def step(self, action):
        """Override step to return full state as proprio for robomimic compatibility"""
        # Call parent step method
        obs, reward, done, info = super().step(action)
        
        # Get full state and use it as proprio
        state = info["state"]  # Full 7D state from parent's info
        visual = obs["visual"]
        
        # For robomimic compatibility, return full state as proprio
        proprio = state  # Full 7D state: [agent_x, agent_y, block_x, block_y, angle, vel_x, vel_y]
        
        observation = {
            "visual": visual,
            "proprio": proprio
        }
        
        return observation, reward, done, info
    
    def sample_random_init_goal_states(self, seed):
        """
        Return two random states: one as the initial state and one as the goal state.
        """
        rs = np.random.RandomState(seed)
        
        def generate_state():
            if self.with_velocity:
                return np.array(
                    [
                        rs.randint(50, 450),
                        rs.randint(50, 450),
                        rs.randint(100, 400),
                        rs.randint(100, 400),
                        rs.randn() * 2 * np.pi - np.pi,
                        0,
                        0,  # agent velocities default 0
                    ]
                )
            else:
                return np.array(
                    [
                        rs.randint(50, 450),
                        rs.randint(50, 450),
                        rs.randint(100, 400),
                        rs.randint(100, 400),
                        rs.randn() * 2 * np.pi - np.pi,
                    ]
                )
        
        init_state = generate_state()
        goal_state = generate_state()
        
        return init_state, goal_state
    
    def update_env(self, env_info):
        self.shape = env_info['shape']
    
    def eval_state(self, goal_state, cur_state):
        """
        Return True if the goal is reached
        [agent_x, agent_y, T_x, T_y, angle, agent_vx, agent_vy]
        """
        # if position difference is < 20, and angle difference < np.pi/9, then success
        pos_diff = np.linalg.norm(goal_state[:4] - cur_state[:4])
        angle_diff = np.abs(goal_state[4] - cur_state[4])
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
        success = pos_diff < 20 and angle_diff < np.pi / 9
        state_dist = np.linalg.norm(goal_state - cur_state)
        return {
            'success': success,
            'state_dist': state_dist,
        }

    def prepare(self, seed, init_state):
        """
        Reset with controlled init_state
        obs: (H W C)
        state: (state_dim)
        """
        self.seed(seed)
        self.reset_to_state = init_state
        obs, state = self.reset()
        return obs, state

    def step_multiple(self, actions):
        """
        infos: dict, each key has shape (T, ...)
        """
        obses = []
        rewards = []
        dones = []
        infos = []
        for action in actions:
            o, r, d, info = self.step(action)
            obses.append(o)
            rewards.append(r)
            dones.append(d)
            infos.append(info)
        obses = aggregate_dct(obses)
        rewards = np.stack(rewards)
        dones = np.stack(dones)
        infos = aggregate_dct(infos)
        return obses, rewards, dones, infos

    def rollout(self, seed, init_state, actions):
        """
        only returns np arrays of observations and states
        seed: int
        init_state: (state_dim, )
        actions: (T, action_dim)
        obses: dict (T, H, W, C)
        states: (T, D)
        """
        obs, state = self.prepare(seed, init_state)
        obses, rewards, dones, infos = self.step_multiple(actions)
        for k in obses.keys():
            obses[k] = np.vstack([np.expand_dims(obs[k], 0), obses[k]])
        states = np.vstack([np.expand_dims(state, 0), infos["state"]])
        states = np.stack(states)
        return obses, states
