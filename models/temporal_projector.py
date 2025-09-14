import torch
import torch.nn as nn
from einops import rearrange, repeat

try:
    from .vit import Transformer, FeedForward
except ImportError:
    # For standalone testing
    from vit import Transformer, FeedForward

class TemporalProjector(nn.Module):
    """
    Temporal Projector with ViT backbone

    输入:
    - z_{t-1}: (b, t, num_patches_z, dim_z) - 前一时刻的projected space
    - x_t: (b, t, num_patches_x, dim_x) - 当前时刻的visual embedding

    输出:
    - z_t: (b, t, num_patches_z, dim_z) - 当前时刻的projected space

    架构:
    1. 分别投影z_{t-1}和x_t到共同维度
    2. 使用ViT Transformer处理跨模态交互
    3. 输出投影回z_t的目标维度
    """

    def __init__(
        self,
        # Standard projector interface (for compatibility)
        in_features=None,   # Will be mapped to x_dim
        out_features=None,  # Will be mapped to output_dim

        # Input dimensions
        num_hist=3,        # 历史窗口大小 (使用前num_hist个z states)
        z_patches=49,      # z_{t-i} patch数量 (7x7)
        z_dim=64,         # z_{t-i} feature维度
        x_patches=196,    # x_t patch数量 (14x14)
        x_dim=None,       # x_t feature维度 (visual + proprio), will use in_features if provided

        # Output dimensions (same as z input)
        output_patches=49,
        output_dim=None,  # will use out_features if provided

        # ViT parameters
        hidden_dim=512,   # 共同的hidden维度
        depth=4,          # Transformer层数
        heads=8,          # Attention头数
        mlp_dim=1024,     # MLP隐藏维度
        dim_head=32,      # 每个head的维度
        dropout=0.1,
        emb_dropout=0.1
    ):
        super().__init__()

        # Parameter mapping for compatibility with standard projector interface
        if in_features is not None:
            x_dim = in_features
        if out_features is not None:
            output_dim = out_features

        # Ensure we have valid dimensions
        assert x_dim is not None, "Must provide either x_dim or in_features"
        assert output_dim is not None, "Must provide either output_dim or out_features"

        self.num_hist = num_hist
        self.z_patches = z_patches
        self.z_dim = z_dim
        self.x_patches = x_patches
        self.x_dim = x_dim
        self.output_patches = output_patches
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # 简化版本：只需要x的投影
        self.x_input_proj = nn.Linear(x_dim, hidden_dim)

        # 只需要x的位置编码
        self.x_pos_embedding = nn.Parameter(torch.randn(1, x_patches, hidden_dim))

        # Dropout
        self.dropout = nn.Dropout(emb_dropout)

        # ViT Transformer处理x tokens
        self.transformer = Transformer(
            dim=hidden_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout
        )

        # 输出投影：从hidden_dim投影回目标维度
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout)
        )

        # 可选的patch数量调整层
        if z_patches != output_patches:
            self.patch_adjust = PatchAdjustment(
                input_patches=z_patches,
                output_patches=output_patches,
                dim=output_dim
            )
        else:
            self.patch_adjust = nn.Identity()

    def forward(self, x_sequence):
        """
        简化版本：直接将x_t投影到z_t，不使用历史信息

        Args:
            x_sequence: (b, t, x_patches, x_dim) - 整个序列的visual+proprio embedding
                       对于concat_dim=1: (b, t, 196, 416) - 每个patch都有visual+proprio

        Returns:
            z_sequence: (b, t, output_patches, output_dim) - 整个序列的projected space
                       (b, t, 196, 64) - 直接从x_t得到z_t
        """
        b, t, x_patches, x_dim = x_sequence.shape

        # 简单投影：直接处理所有时间步
        # Reshape: (b, t, x_patches, x_dim) -> (b*t, x_patches, x_dim)
        x_flat = x_sequence.view(b * t, x_patches, x_dim)

        # 投影到hidden维度
        x_proj = self.x_input_proj(x_flat)  # (b*t, x_patches, hidden_dim)

        # 添加位置编码
        x_proj = x_proj + self.x_pos_embedding

        # 添加dropout
        x_proj = self.dropout(x_proj)

        # ViT Transformer处理
        x_proj = self.transformer(x_proj)  # (b*t, x_patches, hidden_dim)

        # 输出投影
        z_out = self.output_proj(x_proj)  # (b*t, x_patches, output_dim)

        # Patch数量调整（如果需要）
        z_out = self.patch_adjust(z_out)  # (b*t, output_patches, output_dim)

        # 重塑回时序维度
        z_sequence = z_out.view(b, t, self.output_patches, self.output_dim)

        return z_sequence


class CrossModalAttention(nn.Module):
    """跨模态注意力：z tokens作为query，x tokens作为key/value"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, z_tokens, x_tokens):
        """
        Args:
            z_tokens: (b, z_patches, dim) - query
            x_tokens: (b, x_patches, dim) - key, value
        """
        z_norm = self.norm_q(z_tokens)
        x_norm = self.norm_kv(x_tokens)

        # Generate Q from z tokens, K,V from x tokens
        q = self.to_q(z_norm)
        k = self.to_k(x_norm)
        v = self.to_v(x_norm)

        # Reshape for multi-head attention
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        # Attention
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        # Residual connection
        return z_tokens + self.to_out(out)


class PatchAdjustment(nn.Module):
    """调整patch数量的层"""

    def __init__(self, input_patches, output_patches, dim):
        super().__init__()
        self.input_patches = input_patches
        self.output_patches = output_patches

        if input_patches == output_patches:
            self.adjust = nn.Identity()
        elif input_patches < output_patches:
            # 上采样 - 使用学习的插值
            self.adjust = nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, dim),
                nn.LayerNorm(dim)
            )
            # 学习插值权重
            self.interpolation = nn.Parameter(torch.randn(output_patches, input_patches))
        else:
            # 下采样 - 使用注意力池化
            self.adjust = nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=8,
                batch_first=True
            )
            self.queries = nn.Parameter(torch.randn(1, output_patches, dim))

    def forward(self, x):
        """
        Args:
            x: (b, input_patches, dim)
        Returns:
            out: (b, output_patches, dim)
        """
        if self.input_patches == self.output_patches:
            return self.adjust(x)
        elif self.input_patches < self.output_patches:
            # 上采样
            b = x.shape[0]
            # 线性插值
            weights = torch.softmax(self.interpolation, dim=-1)  # (output_patches, input_patches)
            x_upsampled = torch.matmul(weights, x)  # (b, output_patches, dim)
            return self.adjust(x_upsampled)
        else:
            # 下采样使用注意力
            b = x.shape[0]
            queries = self.queries.repeat(b, 1, 1)  # (b, output_patches, dim)
            out, _ = self.adjust(queries, x, x)  # (b, output_patches, dim)
            return out


if __name__ == "__main__":
    # 测试代码
    print("🧪 Testing TemporalProjector...")

    # 参数设置
    batch_size = 2
    time_steps = 3
    z_patches = 49  # 7x7
    z_dim = 64
    x_patches = 196  # 14x14
    x_dim = 394  # visual + proprio

    # 创建模型
    projector = TemporalProjector(
        z_patches=z_patches,
        z_dim=z_dim,
        x_patches=x_patches,
        x_dim=x_dim,
        output_patches=z_patches,
        output_dim=z_dim,
        hidden_dim=512,
        depth=4,
        heads=8,
        mlp_dim=1024
    )

    print(f"📊 Model parameters: {sum(p.numel() for p in projector.parameters()):,}")

    # 创建测试数据
    x_sequence = torch.randn(batch_size, time_steps, x_patches, x_dim)

    print(f"📥 Input shapes:")
    print(f"   x_sequence: {x_sequence.shape}")

    # Forward pass
    try:
        z_sequence = projector(x_sequence)
        print(f"✅ Output shape: {z_sequence.shape}")
        assert z_sequence.shape == (batch_size, time_steps, z_patches, z_dim)
        print("✅ Shape test passed!")

        # 测试梯度
        loss = z_sequence.sum()
        loss.backward()
        print("✅ Gradient test passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise

    print("🎉 All tests passed!")