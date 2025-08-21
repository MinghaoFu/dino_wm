import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

class ConditionalFlowKLLoss(nn.Module):
    """
    KL divergence loss based on conditional flow for noise prediction.
    
    The loss computes KL(ε, N(0,1)) where ε = f(z,s) is predicted noise
    from the estimated distribution z and true state variables s.
    """
    
    def __init__(self, z_dim=128, state_dim=7, hidden_dim=256):
        """
        Args:
            z_dim: Dimension of estimated distribution z (e.g., 128 for DINO features)
            state_dim: Dimension of true state variables s (e.g., 7 for robomimic)
            hidden_dim: Hidden dimension for the flow network f(z,s)
        """
        super().__init__()
        self.z_dim = z_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        # Flow network f(z,s) that predicts noise ε
        self.flow_network = nn.Sequential(
            nn.Linear(z_dim + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim),  # Output noise ε with same dim as z
        )
        
        # Standard normal distribution N(0,1) for KL divergence target
        self.standard_normal = Normal(0.0, 1.0)
        
    def forward(self, z, state):
        """
        Compute KL divergence loss KL(ε, N(0,1))
        
        Args:
            z: Estimated distribution features (batch_size, seq_len, z_dim)
            state: True state variables (batch_size, seq_len, state_dim)
            
        Returns:
            kl_loss: KL divergence loss scalar
        """
        batch_size, seq_len, _ = z.shape
        
        # Move flow network to same device as input tensors
        device = z.device
        self.flow_network = self.flow_network.to(device)
        
        # Flatten temporal dimension for processing
        z_flat = z.reshape(-1, self.z_dim)  # (batch_size * seq_len, z_dim)
        state_flat = state.reshape(-1, self.state_dim)  # (batch_size * seq_len, state_dim)
        
        # Concatenate z and s as input to flow network
        zs_input = torch.cat([z_flat, state_flat], dim=1)  # (batch_size * seq_len, z_dim + state_dim)
        
        # Predict noise ε = f(z,s)
        epsilon = self.flow_network(zs_input)  # (batch_size * seq_len, z_dim)
        
        # Create distribution from predicted noise
        # Assume epsilon represents mean of a unit variance Gaussian
        epsilon_dist = Normal(epsilon, torch.ones_like(epsilon))
        
        # Create standard normal distribution with same shape
        standard_normal_expanded = Normal(
            torch.zeros_like(epsilon), 
            torch.ones_like(epsilon)
        )
        
        # Compute KL divergence: KL(ε || N(0,1))
        kl_loss = kl_divergence(epsilon_dist, standard_normal_expanded)
        
        # Average over all dimensions and samples
        kl_loss = kl_loss.mean()
        
        return kl_loss
    
    def get_predicted_noise(self, z, state):
        """
        Get the predicted noise ε = f(z,s) without computing loss
        
        Args:
            z: Estimated distribution features (batch_size, seq_len, z_dim)  
            state: True state variables (batch_size, seq_len, state_dim)
            
        Returns:
            epsilon: Predicted noise (batch_size, seq_len, z_dim)
        """
        batch_size, seq_len, _ = z.shape
        
        # Flatten temporal dimension
        z_flat = z.reshape(-1, self.z_dim)
        state_flat = state.reshape(-1, self.state_dim)
        
        # Concatenate and predict
        zs_input = torch.cat([z_flat, state_flat], dim=1)
        epsilon_flat = self.flow_network(zs_input)
        
        # Reshape back to original temporal structure
        epsilon = epsilon_flat.reshape(batch_size, seq_len, self.z_dim)
        
        return epsilon