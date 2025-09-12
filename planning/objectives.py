import numpy as np
import torch
import torch.nn as nn


def create_objective_fn(alpha, base, mode="last", projected_dim=64):
    """
    Loss calculated on the last pred frame.
    Args:
        alpha: int
        base: int. only used for objective_fn_all
    Returns:
        loss: tensor (B, )
    """
    metric = nn.MSELoss(reduction="none")

    def objective_fn_last(z_obs_pred, z_obs_tgt):
        """
        Args:
            z_obs_pred: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
            z_obs_tgt: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
        Returns:
            loss: tensor (B, )
        """
        loss_visual = metric(z_obs_pred["visual"][:, -1:], z_obs_tgt["visual"]).mean(
            dim=tuple(range(1, z_obs_pred["visual"].ndim))
        )
        loss_proprio = metric(z_obs_pred["proprio"][:, -1:], z_obs_tgt["proprio"]).mean(
            dim=tuple(range(1, z_obs_pred["proprio"].ndim))
        )
        loss = loss_visual + alpha * loss_proprio
        return loss

    def objective_fn_all(z_obs_pred, z_obs_tgt):
        """
        Loss calculated on all pred frames.
        Args:
            z_obs_pred: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
            z_obs_tgt: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
        Returns:
            loss: tensor (B, )
        """
        coeffs = np.array(
            [base**i for i in range(z_obs_pred["visual"].shape[1])], dtype=np.float32
        )
        coeffs = torch.tensor(coeffs / np.sum(coeffs)).to(z_obs_pred["visual"].device)
        loss_visual = metric(z_obs_pred["visual"], z_obs_tgt["visual"]).mean(
            dim=tuple(range(2, z_obs_pred["visual"].ndim))
        )
        loss_proprio = metric(z_obs_pred["proprio"], z_obs_tgt["proprio"]).mean(
            dim=tuple(range(2, z_obs_pred["proprio"].ndim))
        )
        loss_visual = (loss_visual * coeffs).mean(dim=1)
        loss_proprio = (loss_proprio * coeffs).mean(dim=1)
        loss = loss_visual + alpha * loss_proprio
        return loss

    def objective_fn_projected_last(z_obs_pred, z_obs_tgt):
        """
        Objective function for projected latent representation (last frame only).
        Args:
            z_obs_pred: dict, {'projected': (B, T, patches, projected_dim), 'action': (B, T, patches, action_dim)} - rollout predictions 
            z_obs_tgt: dict, {'projected': (B, T, patches, projected_dim)} - goal state
        Returns:
            loss: tensor (B, )
        """
        # Use projected features directly from the dict
        pred_projected = z_obs_pred["projected"][:, -1:, :, :]  # Final timestep, projected_dim
        tgt_projected = z_obs_tgt["projected"]  # Goal, projected_dim features only
        
        # Compute MSE loss in projected space
        loss_projected = metric(pred_projected, tgt_projected).mean(
            dim=tuple(range(1, pred_projected.ndim))
        )
        return loss_projected
        
    def objective_fn_projected_all(z_obs_pred, z_obs_tgt):
        """
        Objective function for projected latent representation (all frames).
        Args:
            z_obs_pred: dict, {'projected': (B, T, patches, projected_dim), 'action': (B, T, patches, action_dim)} - rollout predictions
            z_obs_tgt: dict, {'projected': (B, T, patches, projected_dim)} - goal state
        Returns:
            loss: tensor (B, )
        """
        coeffs = np.array(
            [base**i for i in range(z_obs_pred["projected"].shape[1])], dtype=np.float32
        )
        coeffs = torch.tensor(coeffs / np.sum(coeffs)).to(z_obs_pred["projected"].device)
        
        # Use projected features directly from the dict
        pred_projected = z_obs_pred["projected"]  # All timesteps, projected_dim
        tgt_projected = z_obs_tgt["projected"]  # Goal, projected_dim features only
        
        # Compute MSE loss in projected space for all frames
        loss_projected = metric(pred_projected, tgt_projected).mean(
            dim=tuple(range(2, pred_projected.ndim))
        )
        loss_projected = (loss_projected * coeffs).mean(dim=1)
        return loss_projected

    if mode == "last":
        return objective_fn_last
    elif mode == "all":
        return objective_fn_all
    elif mode == "projected_last":
        return objective_fn_projected_last
    elif mode == "projected_all":
        return objective_fn_projected_all
    else:
        raise NotImplementedError
