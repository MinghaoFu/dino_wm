import torch
import torch.nn as nn
import torch.nn.functional as F

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

class DinoV2Encoder(nn.Module):
    def __init__(self, name, feature_key, projected_dim=128, z_dim=12, recon_dino_loss=False):
        super().__init__()
        self.name = name
        self.base_model = torch.hub.load("facebookresearch/dinov2", name)
        self.feature_key = feature_key
        self.base_emb_dim = self.base_model.num_features
        self.projected_dim = projected_dim
        self.z_dim = z_dim  # New configurable z dimension
        self.recon_dino_loss = recon_dino_loss
        
        if feature_key == "x_norm_patchtokens":
            self.latent_ndim = 2
        elif feature_key == "x_norm_clstoken":
            self.latent_ndim = 1
        else:
            raise ValueError(f"Invalid feature key: {feature_key}")

        self.patch_size = self.base_model.patch_size
        
        # MLP encoder: 384D → z_dim (e.g., 12D)
        self.mlp_encoder = nn.Sequential(
            nn.Linear(self.base_emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.z_dim)
        )
        
        # Keep old projection for backward compatibility with reconstruction
        self.projection = nn.Linear(self.base_emb_dim, self.projected_dim)
        
        # Reconstruction decoder if enabled (12D z → 384D DINO features)
        if self.recon_dino_loss:
            self.reconstruction_decoder = nn.Sequential(
                nn.Linear(self.z_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(), 
                nn.Linear(128, self.base_emb_dim)
            )
            self.last_recon_loss = None  # Store latest reconstruction loss
        
        self.emb_dim = self.z_dim  # Use z_dim as the embedding dimension

    def forward(self, x):
        emb_original = self.base_model.forward_features(x)[self.feature_key]
        
        # Use MLP encoder: 384D → z_dim (e.g., 12D)
        emb_z = self.mlp_encoder(emb_original)
        
        if self.latent_ndim == 1:
            emb_z = emb_z.unsqueeze(1)  # dummy patch dim
        
        # For DINO reconstruction loss (12D z → 384D DINO features)
        if self.recon_dino_loss:
            emb_reconstructed = self.reconstruction_decoder(emb_z.squeeze(1) if self.latent_ndim == 1 else emb_z)
            # Compute and store reconstruction loss (MSE between original and reconstructed)
            self.last_recon_loss = F.mse_loss(emb_reconstructed, emb_original)
        
        return emb_z