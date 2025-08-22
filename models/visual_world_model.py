import torch
import torch.nn as nn
from torchvision import transforms
from einops import rearrange, repeat
from .flow_kl_loss import ConditionalFlowKLLoss

class VWorldModel(nn.Module):
    def __init__(
        self,
        image_size,  # 224
        num_hist,
        num_pred,
        encoder,
        proprio_encoder,
        action_encoder,
        decoder,
        predictor,
        proprio_dim=0,
        action_dim=0,
        concat_dim=0,
        num_action_repeat=7,
        num_proprio_repeat=7,
        train_encoder=True,
        train_predictor=False,
        train_decoder=True,
    ):
        super().__init__()
        self.num_hist = num_hist
        self.num_pred = num_pred
        self.encoder = encoder
        self.proprio_encoder = proprio_encoder
        self.action_encoder = action_encoder
        self.decoder = decoder  # decoder could be None
        self.predictor = predictor  # predictor could be None
        self.train_encoder = train_encoder
        self.train_predictor = train_predictor
        self.train_decoder = train_decoder
        self.num_action_repeat = num_action_repeat
        self.num_proprio_repeat = num_proprio_repeat
        self.proprio_dim = proprio_dim * num_proprio_repeat 
        self.action_dim = action_dim * num_action_repeat 
        self.emb_dim = self.encoder.emb_dim + (self.action_dim + self.proprio_dim) * (concat_dim) # Not used

        print(f"num_action_repeat: {self.num_action_repeat}")
        print(f"num_proprio_repeat: {self.num_proprio_repeat}")
        print(f"proprio encoder: {proprio_encoder}")
        print(f"action encoder: {action_encoder}")
        print(f"proprio_dim: {proprio_dim}, after repeat: {self.proprio_dim}")
        print(f"action_dim: {action_dim}, after repeat: {self.action_dim}")
        print(f"emb_dim: {self.emb_dim}")

        self.concat_dim = concat_dim # 0 or 1
        assert concat_dim == 0 or concat_dim == 1, f"concat_dim {concat_dim} not supported."
        print("Model emb_dim: ", self.emb_dim)

        if "dino" in self.encoder.name:
            decoder_scale = 16  # from vqvae
            num_side_patches = image_size // decoder_scale
            self.encoder_image_size = num_side_patches * encoder.patch_size
            self.encoder_transform = transforms.Compose(
                [transforms.Resize(self.encoder_image_size)]
            )
        else:
            # set self.encoder_transform to identity transform
            self.encoder_transform = lambda x: x

        self.decoder_criterion = nn.MSELoss()
        self.decoder_latent_loss_weight = 0.25
        self.emb_criterion = nn.MSELoss()
        
        # Linear alignment loss for state supervision
        self.state_consistency_loss_weight = 1.0
        self.alignment_W = None  # Linear transformation matrix W: R^{64} -> R^{state_dim}
        self.alignment_regularization = 1e-4  # L2 regularization on W
        
        # Conditional Flow KL divergence loss
        self.flow_kl_loss_weight = 1.0
        self.flow_kl_loss_enabled = False  # Will be set via config
        self.flow_kl_loss = ConditionalFlowKLLoss(
            z_dim=128,  # DINO feature dimension
            state_dim=7,  # Robomimic state dimension
            hidden_dim=256
        )

    def train(self, mode=True):
        super().train(mode)
        if self.train_encoder:
            self.encoder.train(mode)
        if self.predictor is not None and self.train_predictor:
            self.predictor.train(mode)
        if self.proprio_encoder is not None:
            self.proprio_encoder.train(mode)
        self.action_encoder.train(mode)
        if self.decoder is not None and self.train_decoder:
            self.decoder.train(mode)

    def eval(self):
        super().eval()
        self.encoder.eval()
        if self.predictor is not None:
            self.predictor.eval()
        if self.proprio_encoder is not None:
            self.proprio_encoder.eval()
        self.action_encoder.eval()
        if self.decoder is not None:
            self.decoder.eval()

    def encode(self, obs, act): 
        """
        input :  obs (dict): "visual", "proprio", (b, num_frames, 3, img_size, img_size) 
        output:    z (tensor): (b, num_frames, num_patches, emb_dim)
        """
        z_dct = self.encode_obs(obs)
        act_emb = self.encode_act(act)
        if self.concat_dim == 0:
            z = torch.cat(
                    [z_dct['visual'], z_dct['proprio'].unsqueeze(2), act_emb.unsqueeze(2)], dim=2 # add as an extra token
                )  # (b, num_frames, num_patches + 2, dim)
        if self.concat_dim == 1:
            proprio_tiled = repeat(z_dct['proprio'].unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['visual'].shape[2])
            proprio_repeated = proprio_tiled.repeat(1, 1, 1, self.num_proprio_repeat)
            act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['visual'].shape[2])
            act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
            z = torch.cat(
                [z_dct['visual'], proprio_repeated, act_repeated], dim=3
            )  # (b, num_frames, num_patches, dim + action_dim)
        return z
    
    def encode_act(self, act):
        act = self.action_encoder(act) # (b, num_frames, action_emb_dim)
        return act
    
    def encode_proprio(self, proprio):
        proprio = self.proprio_encoder(proprio)
        return proprio

    def encode_obs(self, obs):
        """
        input : obs (dict): "visual", "proprio" (b, t, 3, img_size, img_size)
        output:   z (dict): "visual", "proprio" (b, t, num_patches, encoder_emb_dim)
        """
        visual = obs['visual']
        b = visual.shape[0]
        visual = rearrange(visual, "b t ... -> (b t) ...")
        visual = self.encoder_transform(visual)
        visual_embs = self.encoder.forward(visual)
        visual_embs = rearrange(visual_embs, "(b t) p d -> b t p d", b=b)

        proprio = obs['proprio']
        proprio_emb = self.encode_proprio(proprio)
        
        return {"visual": visual_embs, "proprio": proprio_emb}

    def predict(self, z):  # in embedding space
        """
        input : z: (b, num_hist, num_patches, emb_dim)
        output: z: (b, num_hist, num_patches, emb_dim)
        """
        T = z.shape[1]
        # reshape to a batch of windows of inputs
        z = rearrange(z, "b t p d -> b (t p) d")
        # (b, num_hist * num_patches per img, emb_dim)
        z = self.predictor(z)
        z = rearrange(z, "b (t p) d -> b t p d", t=T)
        return z

    def decode(self, z):
        """
        input :   z: (b, num_frames, num_patches, emb_dim)
        output: obs: (b, num_frames, 3, img_size, img_size)
        """
        z_obs, z_act = self.separate_emb(z)
        obs, diff = self.decode_obs(z_obs)
        return obs, diff

    def decode_obs(self, z_obs):
        """
        input :   z: (b, num_frames, num_patches, emb_dim)
        output: obs: (b, num_frames, 3, img_size, img_size)
        """
        b, num_frames, num_patches, emb_dim = z_obs["visual"].shape
        visual, diff = self.decoder(z_obs["visual"])  # (b*num_frames, 3, 224, 224)
        visual = rearrange(visual, "(b t) c h w -> b t c h w", t=num_frames)
        obs = {
            "visual": visual,
            "proprio": z_obs["proprio"], # Note: no decoder for proprio for now!
        }
        return obs, diff
    
    def separate_emb(self, z):
        """
        input: z (tensor)
        output: z_obs (dict), z_act (tensor)
        """
        if self.concat_dim == 0:
            z_visual, z_proprio, z_act = z[:, :, :-2, :], z[:, :, -2, :], z[:, :, -1, :]
        elif self.concat_dim == 1:
            z_visual, z_proprio, z_act = z[..., :-(self.proprio_dim + self.action_dim)], \
                                         z[..., -(self.proprio_dim + self.action_dim) :-self.action_dim],  \
                                         z[..., -self.action_dim:]
            # remove tiled dimensions
            z_proprio = z_proprio[:, :, 0, : self.proprio_dim // self.num_proprio_repeat]
            z_act = z_act[:, :, 0, : self.action_dim // self.num_action_repeat]
        z_obs = {"visual": z_visual, "proprio": z_proprio}
        return z_obs, z_act
    
    def compute_detailed_state_alignment(self, obs, act, state):
        """
        Compute detailed per-dimension alignment metrics for analysis
        Returns dict with per-dimension metrics
        """
        with torch.no_grad():
            self.eval()
            z = self.encode(obs, act)
            z_src = z[:, : self.num_hist, :, :]
            
            if self.predictor is not None:
                z_pred = self.predict(z_src)
                
                if state is not None and self.state_projection is not None:
                    # Extract visual embeddings for state consistency
                    if self.concat_dim == 0:
                        z_visual_for_state = z_pred[:, :, :-2, :]
                    elif self.concat_dim == 1:
                        z_visual_for_state = z_pred[:, :, :, :-(self.proprio_dim + self.action_dim)]
                    
                    # Take half of the features for state consistency
                    half_dim = z_visual_for_state.shape[-1] // 2
                    z_state_features = z_visual_for_state[:, :, :, :half_dim]
                    z_state_avg = z_state_features.mean(dim=2)
                    predicted_state = self.state_projection(z_state_avg)
                    
                    state_tgt = state[:, self.num_pred:, :]
                    
                    # Per-dimension metrics
                    state_dim = state_tgt.shape[-1]
                    per_dim_metrics = {}
                    
                    for dim in range(state_dim):
                        pred_dim = predicted_state[:, :, dim].flatten()
                        tgt_dim = state_tgt[:, :, dim].flatten()
                        
                        # MAE per dimension
                        mae_dim = torch.mean(torch.abs(pred_dim - tgt_dim))
                        per_dim_metrics[f"state_dim_{dim}_mae"] = mae_dim.item()
                        
                        # Correlation per dimension
                        if len(pred_dim) > 1:
                            pred_centered = pred_dim - torch.mean(pred_dim)
                            tgt_centered = tgt_dim - torch.mean(tgt_dim)
                            correlation = torch.sum(pred_centered * tgt_centered) / (
                                torch.sqrt(torch.sum(pred_centered**2)) * torch.sqrt(torch.sum(tgt_centered**2)) + 1e-8
                            )
                            per_dim_metrics[f"state_dim_{dim}_correlation"] = correlation.item()
                        
                        # RMSE per dimension
                        rmse_dim = torch.sqrt(torch.mean((pred_dim - tgt_dim)**2))
                        per_dim_metrics[f"state_dim_{dim}_rmse"] = rmse_dim.item()
                    
                    return per_dim_metrics
            
            return {}

    def forward(self, obs, act, state=None):
        """
        input:  obs (dict):  "visual", "proprio" (b, num_frames, 3, img_size, img_size)
                act: (b, num_frames, action_dim)
                state: (b, num_frames, state_dim) - true state variables for consistency loss
        output: z_pred: (b, num_hist, num_patches, emb_dim)
                visual_pred: (b, num_hist, 3, img_size, img_size)
                visual_reconstructed: (b, num_frames, 3, img_size, img_size)
        """
        loss = 0
        loss_components = {}
        z = self.encode(obs, act)
        z_src = z[:, : self.num_hist, :, :]  # (b, num_hist, num_patches, dim)
        z_tgt = z[:, self.num_pred :, :, :]  # (b, num_hist, num_patches, dim)
        visual_src = obs['visual'][:, : self.num_hist, ...]  # (b, num_hist, 3, img_size, img_size)
        visual_tgt = obs['visual'][:, self.num_pred :, ...]  # (b, num_hist, 3, img_size, img_size)

        if self.predictor is not None:
            z_pred = self.predict(z_src)
            if self.decoder is not None:
                obs_pred, diff_pred = self.decode(
                    z_pred.detach()
                )  # recon loss should only affect decoder
                visual_pred = obs_pred['visual']
                recon_loss_pred = self.decoder_criterion(visual_pred, visual_tgt)
                decoder_loss_pred = (
                    recon_loss_pred + self.decoder_latent_loss_weight * diff_pred
                )
                loss_components["decoder_recon_loss_pred"] = recon_loss_pred
                loss_components["decoder_vq_loss_pred"] = diff_pred
                loss_components["decoder_loss_pred"] = decoder_loss_pred
            else:
                visual_pred = None

            # Compute loss for visual, proprio dims (i.e. exclude action dims)
            if self.concat_dim == 0:
                z_visual_loss = self.emb_criterion(z_pred[:, :, :-2, :], z_tgt[:, :, :-2, :].detach())
                z_proprio_loss = self.emb_criterion(z_pred[:, :, -2, :], z_tgt[:, :, -2, :].detach())
                z_loss = self.emb_criterion(z_pred[:, :, :-1, :], z_tgt[:, :, :-1, :].detach())
            elif self.concat_dim == 1:
                z_visual_loss = self.emb_criterion(
                    z_pred[:, :, :, :-(self.proprio_dim + self.action_dim)], \
                    z_tgt[:, :, :, :-(self.proprio_dim + self.action_dim)].detach()
                )
                z_proprio_loss = self.emb_criterion(
                    z_pred[:, :, :, -(self.proprio_dim + self.action_dim): -self.action_dim], 
                    z_tgt[:, :, :, -(self.proprio_dim + self.action_dim): -self.action_dim].detach()
                )
                z_loss = self.emb_criterion(
                    z_pred[:, :, :, :-self.action_dim], 
                    z_tgt[:, :, :, :-self.action_dim].detach()
                )

            loss = loss + z_loss
            loss_components["z_loss"] = z_loss
            loss_components["z_visual_loss"] = z_visual_loss
            loss_components["z_proprio_loss"] = z_proprio_loss
            
            # Linear alignment loss with true state variables
            if state is not None:
                # Initialize alignment matrix W if not done yet
                if self.alignment_W is None:
                    state_dim = state.shape[-1]
                    half_dim = self.encoder.emb_dim // 2  # 64 dimensions from DINO
                    # W: R^{64} -> R^{state_dim}
                    self.alignment_W = nn.Parameter(torch.randn(half_dim, state_dim, device=state.device) * 0.01)
                
                # Extract visual embeddings and use half for state alignment
                if self.concat_dim == 0:
                    z_visual_for_state = z_pred[:, :, :-2, :]  # (b, num_hist, num_patches, emb_dim)
                elif self.concat_dim == 1:
                    z_visual_for_state = z_pred[:, :, :, :-(self.proprio_dim + self.action_dim)]
                
                # Take first half of the features (64 dims) for state alignment
                half_dim = z_visual_for_state.shape[-1] // 2
                z_hat = z_visual_for_state[:, :, :, :half_dim]  # (b, num_hist, num_patches, 64)
                
                # Average over patches to get single representation per timestep
                z_hat_avg = z_hat.mean(dim=2)  # (b, num_hist, 64)
                
                # Get target states for the predicted frames
                z_target = state[:, self.num_pred:, :]  # (b, num_hist, state_dim)
                
                # Optional: Center the representations
                z_hat_centered = z_hat_avg - torch.mean(z_hat_avg, dim=(0,1), keepdim=True)
                z_target_centered = z_target - torch.mean(z_target, dim=(0,1), keepdim=True)
                
                # Linear alignment: W^T @ z_hat ≈ z_target
                # z_hat: (b, num_hist, 64), W: (64, state_dim) -> projected: (b, num_hist, state_dim)
                z_projected = torch.matmul(z_hat_centered, self.alignment_W)
                
                # L2 alignment loss: ||W^T @ z_hat - z_target||_2^2
                alignment_loss = torch.mean((z_projected - z_target_centered)**2)
                
                # L2 regularization on W: λ ||W||_F^2
                w_regularization = self.alignment_regularization * torch.sum(self.alignment_W**2)
                
                # Total state consistency loss
                state_consistency_loss = alignment_loss + w_regularization
                loss = loss + self.state_consistency_loss_weight * state_consistency_loss
                loss_components["state_consistency_loss"] = state_consistency_loss
                loss_components["alignment_loss"] = alignment_loss
                loss_components["w_regularization"] = w_regularization
                
                # 7D Aligned Temporal Dynamics Loss (configurable, same weight as original z_loss)
                if hasattr(self, 'dynamics_7d_loss_weight') and self.dynamics_7d_loss_weight > 0:
                    # Get source 7D aligned features (from input frames)
                    if self.concat_dim == 0:
                        z_visual_src = z_src[:, :, :-2, :]  # (b, num_hist, num_patches, emb_dim)
                    elif self.concat_dim == 1:
                        z_visual_src = z_src[:, :, :, :-(self.proprio_dim + self.action_dim)]
                    
                    # Extract 7D aligned features from source
                    half_dim = z_visual_src.shape[-1] // 2
                    z_hat_src = z_visual_src[:, :, :, :half_dim]  # (b, num_hist, num_patches, 64)
                    z_hat_src_avg = z_hat_src.mean(dim=2)  # (b, num_hist, 64)
                    z_hat_src_centered = z_hat_src_avg - torch.mean(z_hat_src_avg, dim=(0,1), keepdim=True)
                    z_7d_src = torch.matmul(z_hat_src_centered, self.alignment_W)  # (b, num_hist, 7)
                    
                    # Target 7D aligned features (ground truth from next timestep)
                    z_7d_target = z_target_centered  # Ground truth state for predicted frames
                    
                    # MSE loss for 7D aligned temporal dynamics: predict o_t from o_{t-1} (same as z_loss)
                    dynamics_7d_loss = torch.mean((z_projected - z_7d_target)**2)
                    loss = loss + self.dynamics_7d_loss_weight * dynamics_7d_loss
                    loss_components["dynamics_7d_loss"] = dynamics_7d_loss
            
            # DINO feature reconstruction loss (384D -> 128D -> 384D)
            if hasattr(self.encoder, 'recon_dino_loss') and self.encoder.recon_dino_loss:
                # Get DINO reconstruction loss from encoder
                dino_recon_loss = getattr(self.encoder, 'last_recon_loss', None)
                if dino_recon_loss is not None and hasattr(self, 'dino_recon_loss_weight'):
                    loss = loss + self.dino_recon_loss_weight * dino_recon_loss
                    loss_components["dino_recon_loss"] = dino_recon_loss
            
            # Conditional Flow KL divergence loss
            if self.flow_kl_loss_enabled and state is not None:
                # Extract DINO visual features from z_pred for KL loss
                # z_pred shape: (batch_size, num_hist, num_patches, emb_dim)
                if self.concat_dim == 0:
                    # Visual features are in all patches except last two (proprio, action)
                    z_visual = z_pred[:, :, :-2, :]  # (batch_size, num_hist, num_patches-2, emb_dim)
                elif self.concat_dim == 1:
                    # Visual features are in the first part of embedding dimension
                    visual_dim = self.encoder.emb_dim  # 128 for DINO
                    z_visual = z_pred[:, :, :, :visual_dim]  # (batch_size, num_hist, num_patches, 128)
                
                # Average over patches to get per-frame visual features
                z_visual_avg = z_visual.mean(dim=2)  # (batch_size, num_hist, visual_dim)
                
                # Take state for the predicted frames (matching z_pred temporal dimension)
                state_pred_frames = state[:, self.num_pred:, :]  # (batch_size, num_hist, state_dim)
                
                # Compute conditional flow KL loss
                flow_kl_loss = self.flow_kl_loss(z_visual_avg, state_pred_frames)
                loss = loss + self.flow_kl_loss_weight * flow_kl_loss
                loss_components["flow_kl_loss"] = flow_kl_loss
        else:
            visual_pred = None
            z_pred = None

        if self.decoder is not None:
            obs_reconstructed, diff_reconstructed = self.decode(
                z.detach()
            )  # recon loss should only affect decoder
            visual_reconstructed = obs_reconstructed["visual"]
            recon_loss_reconstructed = self.decoder_criterion(visual_reconstructed, obs['visual'])
            decoder_loss_reconstructed = (
                recon_loss_reconstructed
                + self.decoder_latent_loss_weight * diff_reconstructed
            )

            loss_components["decoder_recon_loss_reconstructed"] = (
                recon_loss_reconstructed
            )
            loss_components["decoder_vq_loss_reconstructed"] = diff_reconstructed
            loss_components["decoder_loss_reconstructed"] = (
                decoder_loss_reconstructed
            )
            loss = loss + decoder_loss_reconstructed
        else:
            visual_reconstructed = None
        loss_components["loss"] = loss
        return z_pred, visual_pred, visual_reconstructed, loss, loss_components

    def replace_actions_from_z(self, z, act):
        act_emb = self.encode_act(act)
        if self.concat_dim == 0:
            z[:, :, -1, :] = act_emb
        elif self.concat_dim == 1:
            act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z.shape[2])
            act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
            z[..., -self.action_dim:] = act_repeated
        return z


    def rollout(self, obs_0, act):
        """
        input:  obs_0 (dict): (b, n, 3, img_size, img_size)
                  act: (b, t+n, action_dim)
        output: embeddings of rollout obs
                visuals: (b, t+n+1, 3, img_size, img_size)
                z: (b, t+n+1, num_patches, emb_dim)
        """
        num_obs_init = obs_0['visual'].shape[1]
        act_0 = act[:, :num_obs_init]
        action = act[:, num_obs_init:] 
        z = self.encode(obs_0, act_0)
        t = 0
        inc = 1
        while t < action.shape[1]:
            z_pred = self.predict(z[:, -self.num_hist :])
            z_new = z_pred[:, -inc:, ...]
            z_new = self.replace_actions_from_z(z_new, action[:, t : t + inc, :])
            z = torch.cat([z, z_new], dim=1)
            t += inc

        z_pred = self.predict(z[:, -self.num_hist :])
        z_new = z_pred[:, -1 :, ...] # take only the next pred
        z = torch.cat([z, z_new], dim=1)
        z_obses, z_acts = self.separate_emb(z)
        return z_obses, z