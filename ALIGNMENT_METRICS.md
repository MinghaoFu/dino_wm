# State Alignment Metrics

This document describes the metrics used to measure how well the DINO encoder features align with the true state variables from the robomimic dataset.

## Alignment Architecture
- **DINO Features**: 128-dimensional projected features from DinoV2 encoder  
- **State Features**: First 64 dimensions used for state alignment
- **Projection Layer**: Learned linear mapping: 64D DINO features → state_dim (e.g., 23D for robomimic)
- **Alignment Loss**: MSE between projected predictions and true states

## Key Point: Dimension Handling
The alignment works by learning a projection from 64D visual features to the true state dimension. Both predicted_state and true_state have the same dimensionality after projection, so standard metrics like MAE work correctly.

## Main Alignment Metrics (logged every batch)

### 1. State Consistency Loss (`state_consistency_loss`)
- **Type**: MSE Loss
- **Description**: Primary training loss between predicted and true states
- **Range**: [0, ∞), lower is better
- **Usage**: Directly optimized during training

### 2. Mean Absolute Error (`state_mae`)
- **Type**: L1 Loss
- **Description**: Average absolute difference per state dimension
- **Range**: [0, ∞), lower is better
- **Interpretation**: Average error magnitude in original state units

### 3. Cosine Similarity (`state_cosine_similarity`) 
- **Type**: Cosine similarity
- **Description**: Measures directional alignment between predicted and true state vectors
- **Range**: [-1, 1], higher is better
- **Interpretation**: 1.0 = perfect alignment, 0.0 = orthogonal, -1.0 = opposite

### 4. Normalized RMSE (`state_normalized_rmse`)
- **Type**: RMSE normalized by state range
- **Description**: Root mean square error normalized by the range of each state dimension
- **Range**: [0, ∞), lower is better
- **Interpretation**: Scale-invariant error measure, accounts for different state magnitudes

### 5. R² Score (`state_r2_score`)
- **Type**: Coefficient of determination
- **Description**: Fraction of variance in true states explained by predictions
- **Range**: (-∞, 1], higher is better
- **Interpretation**: 1.0 = perfect prediction, 0.0 = no better than mean, negative = worse than mean

### 6. Raw Feature Alignment (`dino_state_norm_correlation`)
- **Type**: Correlation between feature norms (before projection)
- **Description**: How well raw 64D DINO feature magnitudes correlate with state magnitudes
- **Range**: [-1, 1], higher is better
- **Interpretation**: Measures inherent alignment before learned projection

## Detailed Per-Dimension Metrics (logged every 100 training batches & each validation epoch)

For each of the 23 state dimensions, we track:

### Per-Dimension MAE (`state_dim_{i}_mae`)
- Mean absolute error for dimension i
- Identifies which state variables are hardest to predict

### Per-Dimension Correlation (`state_dim_{i}_correlation`)
- Pearson correlation coefficient for dimension i
- Measures linear relationship strength between predicted and true values

### Per-Dimension RMSE (`state_dim_{i}_rmse`)
- Root mean square error for dimension i
- Combines bias and variance errors for each dimension

## Interpretation Guidelines

### Good Alignment Indicators:
- `state_consistency_loss` < 0.1 (depends on state normalization)
- `state_cosine_similarity` > 0.8
- `state_r2_score` > 0.7
- `state_normalized_rmse` < 0.2
- Most per-dimension correlations > 0.6

### What to Monitor:
1. **Training Progress**: `state_consistency_loss` should decrease over time
2. **Quality**: `state_cosine_similarity` and `state_r2_score` should increase
3. **Generalization**: Validation metrics should track training metrics closely
4. **Dimension Analysis**: Check which state dimensions are hardest to predict

### Troubleshooting:
- If `state_cosine_similarity` is low but `state_mae` is reasonable: Check for scale mismatches
- If certain dimensions have very low correlation: Those state variables may not be visually observable
- If validation metrics diverge from training: Potential overfitting to state consistency

## Expected State Variables (robomimic dataset)
The 23 state dimensions typically include:
- Robot joint positions (7 dims)
- Robot joint velocities (7 dims) 
- End-effector pose (6 dims)
- Gripper state (1-3 dims)

Different dimensions may have different levels of visual observability and thus different alignment quality.