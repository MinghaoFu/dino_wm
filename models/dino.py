import torch
import torch.nn as nn
import torch.nn.functional as F

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

class DinoV2Encoder(nn.Module):
    def __init__(self, name, feature_key, projected_dim=128, recon_dino_loss=False):
        super().__init__()
        self.name = name
        self.base_model = torch.hub.load("facebookresearch/dinov2", name)
        self.feature_key = feature_key
        self.base_emb_dim = self.base_model.num_features
        self.projected_dim = projected_dim
        self.recon_dino_loss = recon_dino_loss
        
        if feature_key == "x_norm_patchtokens":
            self.latent_ndim = 2
        elif feature_key == "x_norm_clstoken":
            self.latent_ndim = 1
        else:
            raise ValueError(f"Invalid feature key: {feature_key}")

        self.patch_size = self.base_model.patch_size
        
        # Projection layer to reduce dimensionality
        self.projection = nn.Linear(self.base_emb_dim, self.projected_dim)
        
        # Reconstruction decoder if enabled
        if self.recon_dino_loss:
            self.reconstruction_decoder = nn.Linear(self.projected_dim, self.base_emb_dim)
            self.last_recon_loss = None  # Store latest reconstruction loss
        
        self.emb_dim = self.projected_dim

    def forward(self, x):
        emb_original = self.base_model.forward_features(x)[self.feature_key]
        # Project to lower dimension
        emb_projected = self.projection(emb_original)
        
        if self.latent_ndim == 1:
            emb_projected = emb_projected.unsqueeze(1) # dummy patch dim
        
        # For DINO reconstruction loss
        if self.recon_dino_loss:
            emb_reconstructed = self.reconstruction_decoder(emb_projected.squeeze(1) if self.latent_ndim == 1 else emb_projected)
            # Compute and store reconstruction loss (MSE between original and reconstructed)
            self.last_recon_loss = F.mse_loss(emb_reconstructed, emb_original)
            return emb_projected
        
        return emb_projected