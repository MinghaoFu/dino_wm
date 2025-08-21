"""
Model Predictive Control (MPC) wrapper for proprioceptive-only planning
Uses a sub-planner (like CEM) to optimize actions at each timestep
"""

import torch
import numpy as np
from typing import Optional, Dict, Any


class ProprioOnlyMPCPlanner:
    """
    MPC planner that uses proprioceptive states only
    
    At each timestep:
    1. Use sub-planner to optimize action sequence
    2. Execute first action  
    3. Get new state from environment/world model
    4. Repeat until goal reached or max iterations
    """
    
    def __init__(
        self,
        sub_planner,
        max_iter: Optional[int] = None,
        n_taken_actions: int = 1,
        device: str = "cuda:0",
        **kwargs
    ):
        """
        Args:
            sub_planner: Planner to use for optimization (e.g., CEM)
            max_iter: Maximum MPC iterations (None = unlimited)
            n_taken_actions: Number of actions to execute before replanning
            device: Device for computations
        """
        self.sub_planner = sub_planner
        self.max_iter = max_iter
        self.n_taken_actions = n_taken_actions
        self.device = device
        
    def plan(
        self, 
        proprio_0: np.ndarray,
        proprio_g: np.ndarray, 
        actions: Optional[torch.Tensor] = None,
        env_interface: Optional[Any] = None
    ):
        """
        Execute MPC planning loop
        
        Args:
            proprio_0: (n_evals, proprio_dim) - initial proprioceptive states
            proprio_g: (n_evals, proprio_dim) - goal proprioceptive states
            actions: Optional warm-start actions
            env_interface: Environment interface for state transitions
            
        Returns:
            all_actions: (n_evals, total_steps, action_dim) - executed actions
            action_lengths: (n_evals,) - number of actions per episode
        """
        n_evals = proprio_0.shape[0]
        device = torch.device(self.device)
        
        # Convert to tensors
        current_proprio = torch.tensor(proprio_0, dtype=torch.float32, device=device)
        goal_proprio = torch.tensor(proprio_g, dtype=torch.float32, device=device)
        
        # Storage for all executed actions
        all_actions = []
        action_lengths = torch.zeros(n_evals, dtype=torch.long, device=device)
        
        # Track which episodes are still active
        active_mask = torch.ones(n_evals, dtype=torch.bool, device=device)
        
        print(f"Starting MPC planning for {n_evals} episodes...")
        
        iteration = 0
        while active_mask.any() and (self.max_iter is None or iteration < self.max_iter):
            if iteration % 5 == 0:
                print(f"MPC iteration {iteration}, active episodes: {active_mask.sum().item()}")
            
            # Only plan for active episodes
            active_indices = torch.where(active_mask)[0]
            if len(active_indices) == 0:
                break
                
            active_current = current_proprio[active_indices]
            active_goal = goal_proprio[active_indices]
            
            # Use sub-planner to optimize action sequence
            planned_actions, _ = self.sub_planner.plan(
                active_current.cpu().numpy(),
                active_goal.cpu().numpy(),
                actions=actions[active_indices.cpu()].cpu() if actions is not None else None
            )
            
            # Execute first n_taken_actions for each active episode
            executed_actions = torch.tensor(planned_actions[:, :self.n_taken_actions], device=device)  # (active_evals, n_taken, action_dim)
            
            # Store executed actions
            if iteration == 0:
                # Initialize storage with first actions
                batch_executed = torch.zeros(n_evals, self.n_taken_actions, executed_actions.shape[-1], device=device)
                batch_executed[active_indices] = executed_actions
                all_actions.append(batch_executed)
            else:
                # Add new actions for active episodes
                batch_executed = torch.zeros(n_evals, self.n_taken_actions, executed_actions.shape[-1], device=device)
                batch_executed[active_indices] = executed_actions
                all_actions.append(batch_executed)
            
            # Update action lengths for active episodes
            action_lengths[active_indices] += self.n_taken_actions
            
            # Get new states after executing actions
            if env_interface is not None:
                # Use environment interface to get true next states
                new_states = env_interface.step_multiple(
                    current_proprio[active_indices].cpu().numpy(),
                    executed_actions.cpu().numpy()
                )
                current_proprio[active_indices] = torch.tensor(new_states, device=device)
            else:
                # Use world model to predict next states
                new_states = self._predict_next_states(
                    active_current,
                    executed_actions
                )
                current_proprio[active_indices] = new_states
            
            # Check which episodes have reached their goals
            distances = torch.norm(current_proprio - goal_proprio, dim=1)
            goal_reached = distances < 0.1  # Success threshold
            
            # Deactivate episodes that reached goal or exceeded max length
            max_length_reached = action_lengths >= 50  # Maximum episode length
            active_mask = active_mask & ~goal_reached & ~max_length_reached
            
            iteration += 1
        
        # Concatenate all executed actions
        if all_actions:
            all_actions_tensor = torch.cat(all_actions, dim=1)  # (n_evals, total_steps, action_dim)
        else:
            all_actions_tensor = torch.zeros(n_evals, 0, self.sub_planner.action_dim, device=device)
        
        print(f"MPC completed after {iteration} iterations")
        print(f"Final action lengths: {action_lengths.cpu().numpy()}")
        
        return all_actions_tensor, action_lengths

    def _predict_next_states(self, current_proprio, actions):
        """
        Predict next proprioceptive states using world model
        
        Args:
            current_proprio: (batch, proprio_dim) - current states
            actions: (batch, n_actions, action_dim) - action sequence
            
        Returns:
            next_proprio: (batch, proprio_dim) - predicted next states
        """
        with torch.no_grad():
            # Simulate forward through the action sequence
            state = current_proprio.clone()
            
            for t in range(actions.shape[1]):
                action = actions[:, t:t+1]  # (batch, 1, action_dim)
                
                # Encode current state and action
                proprio_emb = self.sub_planner.proprio_encoder(state.unsqueeze(1))  # (batch, 1, emb_dim)
                action_emb = self.sub_planner.action_encoder(action)  # (batch, 1, emb_dim)
                
                # Predict next state embedding
                combined_emb = torch.cat([proprio_emb, action_emb], dim=-1)
                next_emb = self.sub_planner.predictor(combined_emb)  # (batch, 1, emb_dim)
                
                # Decode back to proprioceptive space (simplified)
                if not hasattr(self, 'state_projection'):
                    self.state_projection = torch.nn.Linear(
                        next_emb.shape[-1], 
                        state.shape[-1]
                    ).to(state.device)
                
                delta_proprio = self.state_projection(next_emb.squeeze(1))
                state = state + 0.05 * delta_proprio  # Small update step
        
        return state

    def evaluate_actions(self, proprio_0, proprio_g, actions):
        """
        Evaluate action sequences by computing final distances to goal
        """
        return self.sub_planner.evaluate_actions(proprio_0, proprio_g, actions)


class ProprioEnvironmentInterface:
    """
    Interface for executing actions and getting new proprioceptive states
    Can use either real environment or world model simulation
    """
    
    def __init__(self, use_world_model=True, world_model_components=None, real_env=None):
        """
        Args:
            use_world_model: If True, use world model for state transitions
            world_model_components: Dict with 'proprio_encoder', 'action_encoder', 'predictor'
            real_env: Real environment for ground-truth state transitions
        """
        self.use_world_model = use_world_model
        self.world_model_components = world_model_components
        self.real_env = real_env
        
    def step_multiple(self, current_states, actions):
        """
        Execute actions and return new states
        
        Args:
            current_states: (batch, proprio_dim) - current proprioceptive states
            actions: (batch, n_steps, action_dim) - actions to execute
            
        Returns:
            new_states: (batch, proprio_dim) - new proprioceptive states
        """
        if self.use_world_model:
            return self._world_model_step(current_states, actions)
        else:
            return self._real_env_step(current_states, actions)
    
    def _world_model_step(self, current_states, actions):
        """Use world model for state prediction"""
        # Find device from the first parameter of any component
        device = 'cpu'
        if self.world_model_components:
            for component in self.world_model_components.values():
                try:
                    device = next(component.parameters()).device
                    break
                except:
                    continue
        
        current_states = torch.tensor(current_states, dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            state = current_states.clone()
            
            # Execute each action in sequence
            for t in range(actions.shape[1]):
                action = actions[:, t:t+1]  # (batch, 1, action_dim)
                
                # Simple physics-based state update for proprioceptive states
                # This is a simplification - using kinematic model instead of world model
                # For robomimic can task: [agent_x, agent_y, block_x, block_y, angle, vel_x, vel_y]
                if state.shape[-1] == 7:  # robomimic can task
                    # Extract action components (2D action for x, y movement)
                    action_xy = action.squeeze(1)  # (batch, 2)
                    
                    # Update agent position
                    state[:, 0] += action_xy[:, 0] * 0.1  # agent_x
                    state[:, 1] += action_xy[:, 1] * 0.1  # agent_y
                    
                    # Update velocity (simple momentum)
                    state[:, 5] = action_xy[:, 0] * 0.5  # vel_x
                    state[:, 6] = action_xy[:, 1] * 0.5  # vel_y
                    
                    # Block position and angle remain mostly unchanged (simplified physics)
                    # In reality, these would be affected by agent-block interactions
                else:
                    # Generic update for other state dimensions
                    delta = 0.01 * action.squeeze(1)[:, :state.shape[-1]]
                    state += delta
        
        return state.cpu().numpy()
    
    def _real_env_step(self, current_states, actions):
        """Use real environment for ground-truth state transitions"""
        new_states = []
        
        for i, (state, action_seq) in enumerate(zip(current_states, actions)):
            # Set environment to current state
            self.real_env.set_state(state)
            
            # Execute action sequence
            for action in action_seq:
                obs, _, _, info = self.real_env.step(action)
                state = info['state']  # Get updated state
            
            new_states.append(state)
        
        return np.array(new_states)