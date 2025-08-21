"""
Robomimic Can Environment for DINO World Model Planning
Based on the robomimic can manipulation task
"""

import gym
import numpy as np
from gym import spaces


class RobomimicCanEnv(gym.Env):
    """
    Environment for robomimic can manipulation task.
    State: [agent_x, agent_y, block_x, block_y, angle, vel_x, vel_y] (7D)
    Action: [dx, dy] (2D)
    """
    
    def __init__(self, with_velocity=True, with_target=True):
        super().__init__()
        
        self.with_velocity = with_velocity
        self.with_target = with_target
        
        # Action space: 2D continuous control [dx, dy]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
        # State space: 7D if with_velocity, 5D otherwise
        state_dim = 7 if with_velocity else 5
        self.observation_space = spaces.Dict({
            'visual': spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
            'proprio': spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        })
        
        # State bounds (from dataset statistics)
        self.state_bounds = {
            'agent_x': (-0.5, 0.5),
            'agent_y': (0.0, 1.0), 
            'block_x': (-0.1, 0.1),
            'block_y': (-0.1, 0.1),
            'angle': (-np.pi, np.pi),
            'vel_x': (-0.5, 0.5),
            'vel_y': (-1.0, 1.0)
        }
        
        self.reset()
    
    def reset(self, seed=None):
        """Reset environment to random initial state"""
        if seed is not None:
            np.random.seed(seed)
            
        # Sample random initial state within bounds
        self.state = np.array([
            np.random.uniform(*self.state_bounds['agent_x']),     # agent_x
            np.random.uniform(*self.state_bounds['agent_y']),     # agent_y  
            np.random.uniform(*self.state_bounds['block_x']),     # block_x
            np.random.uniform(*self.state_bounds['block_y']),     # block_y
            np.random.uniform(*self.state_bounds['angle']),       # angle
        ])
        
        if self.with_velocity:
            vel = np.array([
                np.random.uniform(*self.state_bounds['vel_x']),   # vel_x
                np.random.uniform(*self.state_bounds['vel_y']),   # vel_y
            ])
            self.state = np.concatenate([self.state, vel])
        
        # Generate dummy visual observation (224x224x3 image)
        visual = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        
        observation = {
            'visual': visual,
            'proprio': self.state.astype(np.float32)
        }
        
        info = {'state': self.state.copy()}
        return observation, info
    
    def step(self, action):
        """Take action and return next observation"""
        action = np.clip(action, -1.0, 1.0)
        
        # Simple dynamics: update agent position with action
        self.state[0] += action[0] * 0.1  # agent_x
        self.state[1] += action[1] * 0.1  # agent_y
        
        # Update velocity if enabled
        if self.with_velocity and len(self.state) == 7:
            self.state[5] = action[0]  # vel_x
            self.state[6] = action[1]  # vel_y
        
        # Clip state to bounds
        self.state[0] = np.clip(self.state[0], *self.state_bounds['agent_x'])
        self.state[1] = np.clip(self.state[1], *self.state_bounds['agent_y'])
        if len(self.state) == 7:
            self.state[5] = np.clip(self.state[5], *self.state_bounds['vel_x'])
            self.state[6] = np.clip(self.state[6], *self.state_bounds['vel_y'])
        
        # Generate dummy visual observation
        visual = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        
        observation = {
            'visual': visual,
            'proprio': self.state.astype(np.float32)
        }
        
        # Simple reward: negative distance to goal (placeholder)
        goal_pos = np.array([0.0, 0.5])  # Target position
        agent_pos = self.state[:2]
        distance = np.linalg.norm(agent_pos - goal_pos)
        reward = -distance
        
        # Success if within threshold
        success = distance < 0.1
        done = success
        
        info = {
            'state': self.state.copy(),
            'success': success,
            'distance': distance
        }
        
        return observation, reward, done, info
    
    def sample_random_init_goal_states(self, seeds):
        """Sample random initial and goal states for planning"""
        init_states = []
        goal_states = []
        
        for seed in seeds:
            np.random.seed(seed)
            
            # Sample initial state
            init_state = np.array([
                np.random.uniform(*self.state_bounds['agent_x']),
                np.random.uniform(*self.state_bounds['agent_y']),
                np.random.uniform(*self.state_bounds['block_x']),
                np.random.uniform(*self.state_bounds['block_y']),
                np.random.uniform(*self.state_bounds['angle']),
            ])
            
            # Sample goal state  
            goal_state = np.array([
                np.random.uniform(*self.state_bounds['agent_x']),
                np.random.uniform(*self.state_bounds['agent_y']),
                np.random.uniform(*self.state_bounds['block_x']),
                np.random.uniform(*self.state_bounds['block_y']),
                np.random.uniform(*self.state_bounds['angle']),
            ])
            
            if self.with_velocity:
                init_vel = np.array([
                    np.random.uniform(*self.state_bounds['vel_x']),
                    np.random.uniform(*self.state_bounds['vel_y']),
                ])
                goal_vel = np.array([
                    np.random.uniform(*self.state_bounds['vel_x']),
                    np.random.uniform(*self.state_bounds['vel_y']),
                ])
                init_state = np.concatenate([init_state, init_vel])
                goal_state = np.concatenate([goal_state, goal_vel])
            
            init_states.append(init_state)
            goal_states.append(goal_state)
        
        return np.array(init_states), np.array(goal_states)
    
    def rollout(self, seed, init_state, actions):
        """Rollout trajectory from initial state with given actions
        
        Args:
            seed: int - random seed
            init_state: (state_dim,) - initial state
            actions: (T, action_dim) - action sequence
        
        Returns:
            observations: dict with 'visual' (T, H, W, C) and 'proprio' (T, state_dim)
            states: (T, state_dim) - state trajectory
        """
        np.random.seed(seed)
        
        # Set initial state
        self.state = init_state.copy()
        
        obs_traj = {'visual': [], 'proprio': []}
        state_traj = []
        
        # Rollout trajectory
        for action in actions:
            obs, _, _, info = self.step(action)
            obs_traj['visual'].append(obs['visual'])
            obs_traj['proprio'].append(obs['proprio'])
            state_traj.append(info['state'])
        
        observations = {
            'visual': np.array(obs_traj['visual']),
            'proprio': np.array(obs_traj['proprio'])
        }
        states = np.array(state_traj)
        
        return observations, states
    
    def prepare(self, seed, init_state):
        """Prepare observation for given state
        
        Args:
            seed: int - random seed
            init_state: (state_dim,) - state to prepare observation for
            
        Returns:
            observation: dict with 'visual' and 'proprio'
            state: (state_dim,) - the prepared state
        """
        np.random.seed(seed)
        
        # Set state
        self.state = init_state.copy()
        
        # Generate dummy visual observation
        visual = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        
        observation = {
            'visual': visual,
            'proprio': self.state.astype(np.float32)
        }
        
        return observation, self.state.copy()
    
    def eval_state(self, goal_state, cur_state):
        """Evaluate if current state has reached the goal state
        
        Args:
            goal_state: (state_dim,) - target state
            cur_state: (state_dim,) - current state
            
        Returns:
            dict with success (bool) and state_dist (float)
        """
        # For robomimic can task, success is based on agent position and block position
        # Agent position: state[:2], Block position: state[2:4], Angle: state[4]
        
        # Position difference (agent + block)
        pos_diff = np.linalg.norm(goal_state[:4] - cur_state[:4])
        
        # Angle difference
        angle_diff = np.abs(goal_state[4] - cur_state[4])
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
        
        # Success criteria (adjust thresholds as needed)
        success = pos_diff < 0.1 and angle_diff < np.pi / 6  # More lenient than pusht
        
        # Overall state distance
        state_dist = np.linalg.norm(goal_state - cur_state)
        
        return {
            'success': success,
            'state_dist': state_dist,
        }
    
    def update_env(self, env_info_list):
        """Update environment configuration (placeholder)"""
        pass