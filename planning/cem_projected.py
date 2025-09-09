"""
CEM Planner for Projected Latent Representation
Specialized version that ensures goal encoding uses the same 64D projected features as rollout.
"""
import torch
from einops import repeat
from planning.cem import CEMPlanner
from utils import move_to_device


class CEMProjectedPlanner(CEMPlanner):
    """
    CEM Planner modified for projected latent representation.
    Key difference: Goal encoding uses projected 64D features instead of raw 384D DINO features.
    """
    
    def __init__(self, horizon, topk, num_samples, var_scale, opt_steps, eval_every, 
                 wm, action_dim, objective_fn, preprocessor, evaluator, wandb_run, **kwargs):
        # Call parent constructor (ignore extra kwargs like 'target')
        super().__init__(horizon, topk, num_samples, var_scale, opt_steps, eval_every, 
                        wm, action_dim, objective_fn, preprocessor, evaluator, wandb_run)
    
    def encode_goal_projected(self, trans_obs_g):
        """
        Encode goal observation using projected 64D representation.
        This matches the representation used by rollout() for consistent comparison.
        """
        # Create dummy action for goal encoding (same as rollout does)
        batch_size = trans_obs_g['visual'].shape[0]
        seq_len = trans_obs_g['visual'].shape[1] 
        
        # Create dummy action - need to match action encoding dimensions
        # Action encoder expects 10D (7D action * num_action_repeat + padding)
        dummy_action_raw = torch.zeros(batch_size, seq_len, 10,
                                     device=trans_obs_g['visual'].device)
        
        # Use the same encode() method as rollout to get 64D projected features
        z, _ = self.wm.encode(trans_obs_g, dummy_action_raw)
        
        # Return in dict format for compatibility with rollout output
        z_obs_g = {"visual": z}  # 64D mixed features
        return z_obs_g

    def plan(self, obs_0, obs_g, actions=None):
        """
        Modified plan() method that uses projected goal encoding.
        Everything else remains the same as original CEM planner.
        """
        trans_obs_0 = move_to_device(
            self.preprocessor.transform_obs(obs_0), self.device
        )
        trans_obs_g = move_to_device(
            self.preprocessor.transform_obs(obs_g), self.device
        )
        
        # KEY CHANGE: Use projected goal encoding instead of raw DINO features
        z_obs_g = self.encode_goal_projected(trans_obs_g)

        mu, sigma = self.init_mu_sigma(obs_0, actions)
        mu, sigma = mu.to(self.device), sigma.to(self.device)
        n_evals = mu.shape[0]

        for i in range(self.opt_steps):
            # optimize individual instances
            for traj in range(n_evals):
                cur_trans_obs_0 = {
                    key: repeat(
                        arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples
                    )
                    for key, arr in trans_obs_0.items()
                }
                cur_z_obs_g = {
                    key: repeat(
                        arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples
                    )
                    for key, arr in z_obs_g.items()
                }
                action = (
                    torch.randn(self.num_samples, self.horizon, self.action_dim).to(
                        self.device
                    )
                    * sigma[traj]
                    + mu[traj]
                )
                action[0] = mu[traj]  # optional: make the first one mu itself
                with torch.no_grad():
                    i_z_obses, i_zs = self.wm.rollout(
                        obs_0=cur_trans_obs_0,
                        act=action,
                    )

                loss = self.objective_fn(i_z_obses, cur_z_obs_g)
                topk_idx = torch.argsort(loss)[: self.topk]
                mu[traj] = action[topk_idx].mean(dim=0)
                sigma[traj] = action[topk_idx].std(dim=0)

        # Final optimization with best trajectory
        best_traj = torch.argmin(
            torch.stack(
                [
                    self.objective_fn(
                        *self.wm.rollout(
                            obs_0={
                                key: arr[traj].unsqueeze(0)
                                for key, arr in trans_obs_0.items()
                            },
                            act=mu[traj].unsqueeze(0),
                        ),
                        {
                            key: arr[traj].unsqueeze(0)
                            for key, arr in z_obs_g.items()
                        },
                    ).item()
                    for traj in range(n_evals)
                ]
            )
        )

        return mu[best_traj], sigma[best_traj]