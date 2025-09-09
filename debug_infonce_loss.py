#!/usr/bin/env python3
"""
Debug script to analyze InfoNCE consistency loss magnitude
"""
import torch
import numpy as np

def infonce_loss_debug(z_aligned, z_target, temperature=0.1):
    """
    Analyze InfoNCE loss computation step by step
    """
    print(f"Input shapes: z_aligned={z_aligned.shape}, z_target={z_target.shape}")
    
    # Flatten batch and time dimensions
    z_aligned_flat = z_aligned.reshape(-1, z_aligned.shape[-1])  # (b*num_hist, 7)
    z_target_flat = z_target.reshape(-1, z_target.shape[-1])    # (b*num_hist, 7)
    print(f"Flattened shapes: z_aligned_flat={z_aligned_flat.shape}, z_target_flat={z_target_flat.shape}")
    
    # Check value ranges
    print(f"z_aligned_flat range: [{z_aligned_flat.min():.3f}, {z_aligned_flat.max():.3f}], mean={z_aligned_flat.mean():.3f}, std={z_aligned_flat.std():.3f}")
    print(f"z_target_flat range: [{z_target_flat.min():.3f}, {z_target_flat.max():.3f}], mean={z_target_flat.mean():.3f}, std={z_target_flat.std():.3f}")
    
    # Normalize features
    z_aligned_norm = torch.nn.functional.normalize(z_aligned_flat, dim=1)
    z_target_norm = torch.nn.functional.normalize(z_target_flat, dim=1)
    print(f"After normalization:")
    print(f"z_aligned_norm range: [{z_aligned_norm.min():.3f}, {z_aligned_norm.max():.3f}], norm mean={z_aligned_norm.norm(dim=1).mean():.3f}")
    print(f"z_target_norm range: [{z_target_norm.min():.3f}, {z_target_norm.max():.3f}], norm mean={z_target_norm.norm(dim=1).mean():.3f}")
    
    # Compute similarity matrix
    logits = torch.matmul(z_aligned_norm, z_target_norm.T) / temperature  # (N, N)
    print(f"Logits shape: {logits.shape}, temperature: {temperature}")
    print(f"Logits range: [{logits.min():.3f}, {logits.max():.3f}], mean={logits.mean():.3f}")
    
    # Diagonal (positive pairs) vs off-diagonal (negative pairs)
    diagonal = torch.diag(logits)
    off_diagonal = logits[~torch.eye(logits.size(0), dtype=torch.bool)]
    print(f"Positive pairs (diagonal) range: [{diagonal.min():.3f}, {diagonal.max():.3f}], mean={diagonal.mean():.3f}")
    print(f"Negative pairs (off-diagonal) range: [{off_diagonal.min():.3f}, {off_diagonal.max():.3f}], mean={off_diagonal.mean():.3f}")
    
    # Positive pairs are on the diagonal
    labels = torch.arange(logits.shape[0], device=logits.device)
    
    # Cross-entropy loss
    loss_ce = torch.nn.functional.cross_entropy(logits, labels)
    
    # Additional analysis
    softmax_logits = torch.nn.functional.softmax(logits, dim=1)
    diagonal_probs = torch.diag(softmax_logits)
    print(f"Diagonal probabilities (positive pair similarities): [{diagonal_probs.min():.3f}, {diagonal_probs.max():.3f}], mean={diagonal_probs.mean():.3f}")
    
    print(f"Final InfoNCE loss: {loss_ce.item():.4f}")
    
    return loss_ce

def simulate_current_architecture():
    """
    Simulate the current architecture's InfoNCE computation
    """
    print("=== Current Architecture Simulation ===")
    
    # Simulate batch_size=32, num_hist=3, feature_dim=7
    batch_size = 32
    num_hist = 3
    feature_dim = 7
    
    # Current approach: Extract from 64D mixed features -> 7D -> linear mapping -> InfoNCE
    
    # Simulate z_7d (first 7 dims of 64D features after averaging over patches)
    # These are unaligned features from the mixed latent space
    z_7d = torch.randn(batch_size, num_hist, feature_dim) * 0.5  # Mixed latent features
    
    # Simulate alignment matrix W (7x7)
    alignment_W = torch.randn(7, 7) * 0.01  # Small init as in code
    
    # Linear mapping: z_7d @ W -> z_aligned  
    z_aligned = torch.matmul(z_7d, alignment_W)  # (batch, time, 7)
    
    # Simulate ground truth states (normalized in dataset)
    z_target = torch.randn(batch_size, num_hist, feature_dim) * 0.3  # More realistic state range
    
    print("\nCurrent approach analysis:")
    return infonce_loss_debug(z_aligned, z_target)

def simulate_previous_architecture():
    """
    Simulate what the previous architecture might have looked like
    """
    print("\n=== Previous Architecture Simulation ===")
    
    batch_size = 32
    num_hist = 3
    feature_dim = 7
    
    # Previous approach: Likely extracted aligned features directly from a pre-aligned latent space
    # These features were probably better aligned already, requiring less transformation
    
    # Simulate already partially aligned features (smaller misalignment)
    z_aligned = torch.randn(batch_size, num_hist, feature_dim) * 0.2  # Better aligned baseline
    
    # Same ground truth
    z_target = torch.randn(batch_size, num_hist, feature_dim) * 0.3
    
    # Add some correlation to simulate better alignment
    z_aligned = z_aligned + 0.5 * z_target + 0.1 * torch.randn_like(z_aligned)
    
    print("\nPrevious approach analysis:")
    return infonce_loss_debug(z_aligned, z_target)

def main():
    torch.manual_seed(42)
    
    print("Analyzing InfoNCE consistency loss magnitude differences")
    print("="*60)
    
    current_loss = simulate_current_architecture()
    previous_loss = simulate_previous_architecture()
    
    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"Current architecture InfoNCE loss: {current_loss.item():.4f}")
    print(f"Previous architecture InfoNCE loss: {previous_loss.item():.4f}")
    print(f"Difference: {current_loss.item() - previous_loss.item():.4f} ({(current_loss.item() / previous_loss.item() - 1)*100:.1f}% increase)")
    
    print(f"\nPOSSIBLE CAUSES:")
    print(f"1. Mixed 64D latent features are less aligned with states initially")
    print(f"2. Linear mapping W initialized randomly, needs more training")
    print(f"3. Feature scales in mixed space may be different from previous aligned space")
    print(f"4. Batch size or feature distribution changes affecting InfoNCE computation")

if __name__ == "__main__":
    main()