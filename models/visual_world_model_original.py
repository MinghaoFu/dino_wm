import torch
import torch.nn as nn
from torchvision import transforms
from einops import rearrange, repeat

class VWorldModelOriginal(nn.Module):
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
        concat_dim=1,  # Original DINO-WM uses concat_dim=1 (tile across patches)
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
        self.concat_dim = concat_dim
        
        # Original DINO-WM dimensions
        # visual_dim = encoder.emb_dim (384D DINO features)
        # proprio_dim = 32D (proprioception)
        # action_dim = 16D (action embedding)
        self.emb_dim = self.encoder.emb_dim + (self.action_dim + self.proprio_dim) * (concat_dim)
        
        print(f"Original DINO-WM Model Configuration:")
        print(f"Visual encoder dim: {self.encoder.emb_dim}")
        print(f"Proprio dim: {self.proprio_dim}")
        print(f"Action dim: {self.action_dim}")
        print(f"Concat dim: {self.concat_dim}")
        print(f"Total embedding dim: {self.emb_dim}")

    def encode(self, obs, action):
        """
        Encode observations and actions following original DINO-WM pattern
        Args:
            obs: dict with 'visual' and 'proprio' keys
            action: action tensor
        Returns:
            z: encoded features (B, T, num_patches, feature_dim)
        """
        B, T = obs['visual'].shape[:2]
        
        # Encode visual observations
        visual_obs = obs['visual']  # (B, T, 3, H, W)
        visual_obs = rearrange(visual_obs, "b t c h w -> (b t) c h w")
        z_visual = self.encoder(visual_obs)  # (BT, num_patches, 384)
        z_visual = rearrange(z_visual, "(b t) n d -> b t n d", b=B, t=T)
        
        # Encode proprioception and actions
        z_proprio = self.proprio_encoder(obs['proprio'])  # (B, T, 32)
        z_action = self.action_encoder(action)  # (B, T, 16)
        
        # Concatenate following original DINO-WM pattern
        if self.concat_dim == 0:
            # Concatenate as additional tokens
            z_proprio = rearrange(z_proprio, "b t d -> b t 1 d")
            z_action = rearrange(z_action, "b t d -> b t 1 d")
            z = torch.cat([z_visual, z_proprio, z_action], dim=2)  # (B, T, num_patches+2, feature_dim)
        else:
            # Tile and repeat across patches (original DINO-WM default)
            num_patches = z_visual.shape[2]
            z_proprio = repeat(z_proprio, "b t d -> b t n d", n=num_patches)
            z_action = repeat(z_action, "b t d -> b t n d", n=num_patches)
            z = torch.cat([z_visual, z_proprio, z_action], dim=-1)  # (B, T, num_patches, 384+32+16=432)
            
        return z

    def decode_obs(self, z):
        """
        Decode observations from encoded features
        Args:
            z: encoded features 
        Returns:
            visual: decoded visual observations
            diff: reconstruction difference (if available)
        """
        if self.decoder is None:
            return None, None
            
        # Original DINO-WM decoder only uses visual features (384D)
        if self.concat_dim == 1:
            # Extract only visual features
            z_visual = z[:, :, :, :self.encoder.emb_dim]  # (B, T, patches, 384)
        else:
            # For concat_dim=0, take only visual patches (first set of tokens)
            num_visual_patches = z.shape[2] - 2  # Subtract proprio and action tokens
            z_visual = z[:, :, :num_visual_patches, :]  # Only visual patches
            
        visual, diff = self.decoder(z_visual)
        return visual, diff

    def rollout(self, obs_0, actions):
        """
        Rollout world model predictions
        Args:
            obs_0: initial observation dict
            actions: action sequence (B, T, action_dim)
        Returns:
            dict with rollout results
        """
        B, T = actions.shape[:2]
        device = actions.device
        
        # Initialize rollout with first observation
        rollout_obs = {'visual': [obs_0['visual']], 'proprio': [obs_0['proprio']]}
        z_hist = []
        
        # Encode initial history
        for t in range(self.num_hist):
            if t < len(rollout_obs['visual']):
                obs_t = {
                    'visual': rollout_obs['visual'][t].unsqueeze(1),  # (B, 1, 3, H, W)
                    'proprio': rollout_obs['proprio'][t].unsqueeze(1)  # (B, 1, proprio_dim)
                }
                action_t = actions[:, t:t+1]  # (B, 1, action_dim)
                z_t = self.encode(obs_t, action_t)  # (B, 1, patches, feature_dim)
                z_hist.append(z_t)
        
        # Pad history if needed
        while len(z_hist) < self.num_hist:
            z_hist.append(torch.zeros_like(z_hist[0]))
            
        # Rollout predictions
        for t in range(self.num_hist, T):
            # Stack history
            z_input = torch.cat(z_hist[-self.num_hist:], dim=1)  # (B, num_hist, patches, feature_dim)
            
            # Predict next step
            action_t = actions[:, t:t+1]  # (B, 1, action_dim)
            z_pred = self.predictor(z_input, action_t)  # (B, 1, patches, feature_dim)
            
            # Decode to get visual observation
            visual_pred, _ = self.decode_obs(z_pred)  # (B, 1, 3, H, W)
            
            # Extract proprio from prediction (if concat_dim=1, it's embedded in z_pred)
            if self.concat_dim == 1:
                # Extract proprio from predicted features
                proprio_pred = z_pred[:, :, 0, self.encoder.emb_dim:self.encoder.emb_dim+self.proprio_dim]  # (B, 1, proprio_dim)
            else:
                # For concat_dim=0, proprio is a separate token
                proprio_pred = z_pred[:, :, -2, :]  # Second to last token
                
            # Store predictions
            rollout_obs['visual'].append(visual_pred.squeeze(1))
            rollout_obs['proprio'].append(proprio_pred.squeeze(1))
            z_hist.append(z_pred)
        
        # Convert to tensors
        rollout_visual = torch.stack(rollout_obs['visual'][1:], dim=1)  # (B, T, 3, H, W)
        rollout_proprio = torch.stack(rollout_obs['proprio'][1:], dim=1)  # (B, T, proprio_dim)
        
        return {
            'visual': rollout_visual,
            'proprio': rollout_proprio
        }

    def forward(self, obs, action):
        """
        Forward pass following original DINO-WM pattern
        Args:
            obs: observation dict with 'visual' and 'proprio'
            action: action tensor
        Returns:
            loss_components: dict with individual loss components
        """
        B, T = obs['visual'].shape[:2]
        
        # Encode all observations and actions
        z = self.encode(obs, action)  # (B, T, patches, feature_dim)
        
        loss_components = {}
        
        # Predictor loss (if predictor is available)
        if self.predictor is not None and self.train_predictor:
            # Use history to predict next timesteps
            z_hist = z[:, :self.num_hist]  # (B, num_hist, patches, feature_dim)
            z_target = z[:, self.num_hist:self.num_hist+self.num_pred]  # (B, num_pred, patches, feature_dim)
            
            # Flatten for predictor (original DINO-WM pattern)
            B, T, patches, feature_dim = z_hist.shape
            z_hist_flat = rearrange(z_hist, "b t p d -> b (t p) d")
            
            z_pred_flat = self.predictor(z_hist_flat)  # (B, num_hist*patches, feature_dim)
            z_pred = rearrange(z_pred_flat, "b (t p) d -> b t p d", t=self.num_pred, p=patches)
            
            # MSE loss on predicted features
            predictor_loss = nn.MSELoss()(z_pred, z_target)
            loss_components['predictor_loss'] = predictor_loss
        
        # Decoder reconstruction loss (if decoder is available)
        if self.decoder is not None and self.train_decoder:
            visual_pred, visual_diff = self.decode_obs(z)  # (B, T, 3, H, W)
            visual_target = obs['visual']  # (B, T, 3, H, W)
            
            # MSE loss on reconstructed images
            recon_loss = nn.MSELoss()(visual_pred, visual_target)
            loss_components['recon_loss'] = recon_loss
            
            # Store for visualization
            loss_components["visual_pred"] = visual_pred.detach()
            loss_components["visual_tgt"] = visual_target.detach()
        
        # Total loss
        total_loss = 0
        for key, loss in loss_components.items():
            if 'loss' in key and isinstance(loss, torch.Tensor):
                total_loss += loss
        
        loss_components['total_loss'] = total_loss
        return loss_components