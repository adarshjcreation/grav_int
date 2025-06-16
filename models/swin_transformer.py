import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Swin-transformer (our implementation)
# Reference: "Yu, P., Zhou, L., Zhou, S., Jiao, J., Huang, G., & Lu, P. (2025). 
#             Physics-Constrained Three-Dimensional Swin Transformer for Gravity Data Inversion. Remote Sensing, 17(1), 113."

class ConvBlock(nn.Module):
    """Basic convolutional block as shown in the Feature Extraction Module."""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class FeatureExtractionStage(nn.Module):
    """Stage in the Feature Extraction Module."""
    def __init__(self, in_channels, out_channels):
        super(FeatureExtractionStage, self).__init__()
        self.conv_block1 = ConvBlock(in_channels, out_channels)
        self.conv_block2 = ConvBlock(out_channels, out_channels)
        self.addition = True  # Indicates addition operation is used
    
    def forward(self, x):
        identity = x
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        # Adding identity if dimensions match, otherwise just return output
        if self.addition and identity.shape == out.shape:
            return out + identity
        return out

class FeatureExtractionModule(nn.Module):
    """Complete Feature Extraction Module with 3 stages."""
    def __init__(self, input_channels=1):
        super(FeatureExtractionModule, self).__init__()
        
        # As shown in diagram (b), three stages with increasing channel dimensions
        self.stage1 = FeatureExtractionStage(input_channels, 32)  # 1->32
        self.stage2 = FeatureExtractionStage(32, 64)  # 32->64
        self.stage3 = FeatureExtractionStage(64, 128)  # 64->128
        
    def forward(self, x):
        x = self.stage1(x)  # Stage 1: 1->32
        x = self.stage2(x)  # Stage 2: 32->64
        x = self.stage3(x)  # Stage 3: 64->128
        return x

class LayerNorm(nn.Module):
    """Layer Normalization module."""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        return self.norm(x)

class WindowAttention(nn.Module):
    """Window Multi-head Self-Attention (W-MSA) module."""
    def __init__(self, dim, window_size, num_heads=8, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Define query, key, value projections
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, H, W, C = x.shape
        
        # Reshape and permute for window attention
        # First, reshape to windows
        x = x.view(B, H // self.window_size[0], self.window_size[0], 
                  W // self.window_size[1], self.window_size[1], C)
        
        # Permute to [B, num_windows_h, num_windows_w, window_size_h, window_size_w, C]
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        
        # Reshape to [B*num_windows, window_size_h*window_size_w, C]
        num_windows = (H // self.window_size[0]) * (W // self.window_size[1])
        x = x.view(B * num_windows, self.window_size[0] * self.window_size[1], C)
        
        # W-MSA calculation
        qkv = self.qkv(x).reshape(x.shape[0], x.shape[1], 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(x.shape[0], x.shape[1], C)
        x = self.proj(x)
        
        # Reshape back to original format [B, H, W, C]
        x = x.view(B, 
                  H // self.window_size[0], 
                  W // self.window_size[1],
                  self.window_size[0], 
                  self.window_size[1], 
                  C)
        
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H, W, C)
        
        return x

class ShiftWindowAttention(nn.Module):
    """Shifted Window Attention (SW-MSA) module."""
    def __init__(self, dim, window_size, shift_size, num_heads=8, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        
        # Initialize the window attention module
        self.w_msa = WindowAttention(dim, window_size, num_heads, qkv_bias)
        
    def forward(self, x):
        B, H, W, C = x.shape
        
        # Apply cyclic shift if shift_size != 0
        if any(s > 0 for s in self.shift_size):
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        else:
            shifted_x = x
            
        # Apply window attention
        attn_out = self.w_msa(shifted_x)
        
        # Reverse cyclic shift
        if any(s > 0 for s in self.shift_size):
            x = torch.roll(attn_out, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
        else:
            x = attn_out
            
        return x

class MLP(nn.Module):
    """Multi-Layer Perceptron module as shown in Swim Transformer Block."""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class SwimTransformerBlock(nn.Module):
    """Swim Transformer Block as shown in diagram (c)."""
    def __init__(self, dim, window_size=(4, 4), num_heads=8, shift=False):
        super().__init__()
        
        # Layer normalization before attention
        self.norm1 = LayerNorm(dim)
        
        # Determine if using regular W-MSA or shifted SW-MSA
        shift_size = (0, 0) if not shift else (window_size[0] // 2, window_size[1] // 2)
        
        # Choose the appropriate attention module (W-MSA or SW-MSA)
        self.attn = WindowAttention(dim, window_size, num_heads) if not shift else \
                    ShiftWindowAttention(dim, window_size, shift_size, num_heads)
        
        # Layer normalization before MLP
        self.norm2 = LayerNorm(dim)
        
        # MLP module
        self.mlp = MLP(in_features=dim, hidden_features=4 * dim)
        
    def forward(self, x):
        # Convert from [B, C, H, W] to [B, H, W, C] for attention
        x = x.permute(0, 2, 3, 1).contiguous()
        
        # First branch with attention
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + x
        
        # Second branch with MLP
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut + x
        
        # Convert back to [B, C, H, W]
        x = x.permute(0, 3, 1, 2).contiguous()
        
        return x

class PatchMerging(nn.Module):
    """2D Patch Merging Layer as shown in diagram (a)."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.reduction = nn.Linear(in_channels * 4, out_channels, bias=False)
        self.norm = nn.LayerNorm(in_channels * 4)
        
    def forward(self, x):
        # Input shape: [B, C, H, W]
        B, C, H, W = x.shape
        
        # Ensure dimensions are divisible by 2
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        
        # Reshape and permute for patch merging
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        
        # Get patches
        x0 = x[:, 0::2, 0::2, :]  # [B, H//2, W//2, C]
        x1 = x[:, 0::2, 1::2, :]
        x2 = x[:, 1::2, 0::2, :]
        x3 = x[:, 1::2, 1::2, :]
        
        # Concatenate along feature dimension
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H//2, W//2, 4*C]
        
        # Apply norm and reduction
        x = self.norm(x)
        x = self.reduction(x)  # [B, H//2, W//2, C_out]
        
        # Reformat back to [B, C, H, W]
        x = x.permute(0, 3, 1, 2).contiguous()
        
        return x

class PatchExpand(nn.Module):
    """2D Patch Expanding Layer (the reverse of Patch Merging)."""
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.expansion = nn.Linear(in_channels, out_channels * (scale_factor**2), bias=False)
        self.norm = nn.LayerNorm(in_channels)
        self.scale_factor = scale_factor
        
    def forward(self, x):
        # Input shape: [B, C, H, W]
        B, C, H, W = x.shape
        
        # Change to [B, H, W, C] for processing
        x = x.permute(0, 2, 3, 1).contiguous()
        
        # Apply norm and expansion
        x = self.norm(x)
        x = self.expansion(x)  # [B, H, W, C_out*scale_factor^2]
        
        # Reshape to prepare for pixel shuffle
        x = x.view(B, H, W, self.scale_factor, self.scale_factor, -1)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        
        # Reshape to get the expanded dimensions
        x = x.view(B, -1, H * self.scale_factor, W * self.scale_factor)
        
        return x

class SwimTransformer(nn.Module):
    """Complete Swim Transformer Network for gravity inverse problem."""
    def __init__(self, 
                 input_channels=1, 
                 output_channels=16,
                 embed_dim=128,
                 depths=[2, 2, 6, 2],
                 window_size=(4, 4),
                 num_heads=[4, 8, 16, 32],
                 mlp_ratio=4.,
                 input_size=(32, 32)):
        super(SwimTransformer, self).__init__()
        
        # Feature Extraction Module
        self.feature_extraction = FeatureExtractionModule(input_channels)
        
        # Initial dimension after feature extraction
        current_dim = 128  # As per the diagram showing output of feature extraction module
        
        # Transformer Blocks (total of 5 as shown in diagram)
        self.transformer_blocks = nn.ModuleList()
        
        # Input projection
        self.input_proj = nn.Conv2d(current_dim, embed_dim, kernel_size=1)
        current_dim = embed_dim
        
        # First Swim Transformer Block
        self.transformer_blocks.append(
            SwimTransformerBlock(current_dim, window_size, num_heads[0], shift=False)
        )
        
        # 2D Patch Merging Layer (downsampling)
        self.patch_merging = PatchMerging(current_dim, current_dim*2)
        current_dim *= 2
        
        # Second Swim Transformer Block with shifted window attention
        self.transformer_blocks.append(
            SwimTransformerBlock(current_dim, window_size, num_heads[1], shift=True)
        )
        
        # Third Swim Transformer Block
        self.transformer_blocks.append(
            SwimTransformerBlock(current_dim, window_size, num_heads[1], shift=False)
        )
        
        # Fourth Swim Transformer Block with shifted window attention
        self.transformer_blocks.append(
            SwimTransformerBlock(current_dim, window_size, num_heads[1], shift=True)
        )
        
        # Fifth Swim Transformer Block
        self.transformer_blocks.append(
            SwimTransformerBlock(current_dim, window_size, num_heads[1], shift=False)
        )
        
        # 2D Upsampling layer
        self.patch_expand = PatchExpand(current_dim, current_dim//2)
        current_dim = current_dim//2
        
        # Output projection
        self.output_proj = nn.Conv2d(current_dim, output_channels, kernel_size=1)
        
    def forward(self, x):
        # Feature extraction
        x = self.feature_extraction(x)
        
        # Input projection
        x = self.input_proj(x)
        
        # First transformer block
        x = self.transformer_blocks[0](x)
        
        # Patch merging (downsampling)
        x = self.patch_merging(x)
        
        # Process through remaining transformer blocks
        for i in range(1, len(self.transformer_blocks)):
            x = self.transformer_blocks[i](x)
        
        # Patch expanding (upsampling)
        x = self.patch_expand(x)
        
        # Output projection
        x = self.output_proj(x)
        
        return x
