"""
CEM Planner for proprioceptive-only planning
Simplified version that works directly with proprioceptive states
"""

import torch
import numpy as np
from einops import rearrange, repeat


class ProprioOnlyCEMPlanner:
    def __init__(
        self,
        horizon,
        topk,
        num_samples,
        var_scale,
        opt_steps,
        eval_every,
        preprocessor,
        encoder,
        proprio_encoder,
        action_encoder,
        predictor,
        device,
        **kwargs,
    ):
        self.horizon = horizon
        self.topk = topk
        self.num_samples = num_samples
        self.var_scale = var_scale
        self.opt_steps = opt_steps
        self.eval_every = eval_every
        self.preprocessor = preprocessor
        self.encoder = encoder
        self.proprio_encoder = proprio_encoder
        self.action_encoder = action_encoder
        self.predictor = predictor
        self.device = device
        
        # Action dimension from preprocessor
        self.action_dim = preprocessor.action_dim

    def init_mu_sigma(self, proprio_0, actions=None):
        """Initialize mean and variance for CEM optimization"""
        n_evals = proprio_0.shape[0]
        sigma = self.var_scale * torch.ones([n_evals, self.horizon, self.action_dim], device=self.device)
        
        if actions is not None:
            # Initialize with ground truth actions if available
            mu = actions[:, :self.horizon].clone().to(self.device)
            if mu.shape[1] < self.horizon:
                # Pad with zeros if actions are shorter than horizon
                padding = torch.zeros(n_evals, self.horizon - mu.shape[1], self.action_dim, device=self.device)
                mu = torch.cat([mu, padding], dim=1)
        else:
            # Initialize with random actions
            mu = torch.zeros([n_evals, self.horizon, self.action_dim], device=self.device)
        
        return mu.to(self.device), sigma.to(self.device)

    def plan(self, proprio_0, proprio_g, actions=None):
        """
        Plan actions using CEM optimization
        
        Args:
            proprio_0: (n_evals, proprio_dim) - initial proprioceptive states
            proprio_g: (n_evals, proprio_dim) - goal proprioceptive states  
            actions: Optional initial actions for warm start
        
        Returns:
            best_actions: (n_evals, horizon, action_dim) - optimized actions
            action_lengths: (n_evals,) - length of each action sequence
        """
        # Convert to tensors and move to device
        proprio_0 = torch.tensor(proprio_0, dtype=torch.float32, device=self.device)
        proprio_g = torch.tensor(proprio_g, dtype=torch.float32, device=self.device)
        
        n_evals = proprio_0.shape[0]
        
        # Initialize CEM parameters
        mu, sigma = self.init_mu_sigma(proprio_0, actions)
        
        best_actions = mu.clone()
        
        print(f"Starting CEM optimization with {self.opt_steps} steps...")
        
        for step in range(self.opt_steps):
            # Sample action sequences
            eps = torch.randn(n_evals, self.num_samples, self.horizon, self.action_dim, device=self.device)
            action_samples = mu.unsqueeze(1) + sigma.unsqueeze(1) * eps
            action_samples = action_samples.view(n_evals * self.num_samples, self.horizon, self.action_dim)
            
            # Evaluate action sequences
            costs = self.evaluate_actions(
                proprio_0.repeat(self.num_samples, 1),
                proprio_g.repeat(self.num_samples, 1),
                action_samples
            )
            
            # Reshape costs back to (n_evals, num_samples)
            costs = costs.view(n_evals, self.num_samples)
            
            # Select top-k actions
            _, topk_indices = torch.topk(costs, self.topk, dim=1, largest=False)
            
            # Update mu and sigma based on top-k samples
            for i in range(n_evals):
                topk_actions = action_samples[i * self.num_samples:(i + 1) * self.num_samples][topk_indices[i]]
                mu[i] = topk_actions.mean(dim=0)
                sigma[i] = topk_actions.std(dim=0) + 1e-6  # Add small epsilon for numerical stability
            
            # Log progress
            if step % self.eval_every == 0:
                best_cost = costs.min(dim=1)[0].mean().item()
                print(f"CEM Step {step}: Best cost = {best_cost:.4f}")
        
        # Return best actions found
        best_actions = mu
        action_lengths = torch.full((n_evals,), self.horizon, dtype=torch.long)
        
        return best_actions, action_lengths

    def evaluate_actions(self, proprio_0, proprio_g, actions):
        """
        Evaluate action sequences by forward simulation using full world model
        
        Args:
            proprio_0: (batch, proprio_dim) - initial states
            proprio_g: (batch, proprio_dim) - goal states
            actions: (batch, horizon, action_dim) - action sequences
        
        Returns:
            costs: (batch,) - cost for each action sequence
        """
        batch_size = actions.shape[0]
        
        with torch.no_grad():
            current_proprio = proprio_0.clone()
            
            # Create dummy visual observations (since we only have proprio)
            # The visual encoder expects (batch * time, channels, height, width) after reshape
            dummy_visual = torch.zeros(batch_size, 1, 3, 224, 224, device=self.device)
            
            # Forward simulate each action sequence
            for t in range(self.horizon):
                action = actions[:, t:t+1]  # (batch, 1, action_dim)
                
                # Create observation dict
                obs = {
                    'visual': dummy_visual,
                    'proprio': current_proprio.unsqueeze(1)  # (batch, 1, proprio_dim)
                }
                
                # Use the world model's encode method
                # This properly handles visual + proprio + action encoding
                z = self._encode_obs_action(obs, action)
                
                # Reshape for predictor: (batch, time, num_patches, emb_dim) -> (batch, time*num_patches, emb_dim)
                T = z.shape[1]
                num_patches = z.shape[2]
                z_reshaped = rearrange(z, "b t p d -> b (t p) d")
                
                # Use predictor to get next state embedding
                next_z_reshaped = self.predictor(z_reshaped)  # (batch, time*num_patches, emb_dim)
                next_z = rearrange(next_z_reshaped, "b (t p) d -> b t p d", t=T)  # (batch, 1, num_patches, emb_dim)
                
                # Extract proprio information from the predicted embedding
                # This is a simplification - ideally use the decoder
                if not hasattr(self, 'proprio_projection'):
                    # Use the proprio encoder's output dimension for projection
                    proprio_emb_dim = 32  # From config
                    self.proprio_projection = torch.nn.Linear(
                        next_z.shape[-1], 
                        current_proprio.shape[-1]
                    ).to(self.device)
                
                # Use mean of patches as global representation
                global_repr = next_z.mean(dim=2)  # (batch, 1, emb_dim)
                delta_proprio = self.proprio_projection(global_repr.squeeze(1))
                
                # Update state with small step
                current_proprio = current_proprio + 0.05 * delta_proprio
            
            # Compute cost as distance to goal
            final_distance = torch.norm(current_proprio - proprio_g, dim=1)
            costs = final_distance
        
        return costs
    
    def _encode_obs_action(self, obs, action):
        """
        Encode observations and actions using the world model's encoding scheme
        """
        # Reshape visual to (batch*time, channels, height, width) for encoder
        visual = obs['visual']
        batch_size = visual.shape[0]
        visual_reshaped = rearrange(visual, "b t c h w -> (b t) c h w")
        
        # Encode visual observations
        visual_emb = self.encoder(visual_reshaped)  # (batch*time, num_patches, visual_emb_dim)
        visual_emb = rearrange(visual_emb, "(b t) p d -> b t p d", b=batch_size)  # (batch, time, num_patches, visual_emb_dim)
        
        # Encode proprio and action
        proprio_emb = self.proprio_encoder(obs['proprio'])  # (batch, time, proprio_emb_dim)
        action_emb = self.action_encoder(action)  # (batch, time, action_emb_dim)
        
        # Concatenate according to concat_dim=1 (from config)
        # Tile proprio and action to match visual patches
        num_patches = visual_emb.shape[2]
        
        # Repeat proprio and action embeddings for each patch
        proprio_tiled = proprio_emb.unsqueeze(2).repeat(1, 1, num_patches, 1)
        action_tiled = action_emb.unsqueeze(2).repeat(1, 1, num_patches, 1)
        
        # Concatenate along feature dimension
        z = torch.cat([visual_emb, proprio_tiled, action_tiled], dim=-1)
        
        return z