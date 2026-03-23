"""
Core Bidirectional Attention Layer (Like PALM) and concurreny FFN.
This layer do not include resiaul sum!

TODO: Add QK normalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import alibi_utils
from typing import List
import math


class AttentionLayer(nn.Module):

    def __init__(
        self,
        layer: int,
        d_model: int,
        num_heads: int,
        dropout: float,
        ffn_factor: int = 4,
    ):
        super(AttentionLayer, self).__init__()
        self.layer = layer
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.d_head = d_model // num_heads
        assert (
            self.d_head * num_heads == d_model
        ), "d_model must be divisible by num_heads"
        self.d_ffn = d_model * ffn_factor

        # RMS pre-normlization
        self.rms_norm = nn.RMSNorm(d_model)

        # Attention modules
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # FFW modules
        self.ffn_linear_up_1 = nn.Linear(d_model, self.d_ffn)
        self.ffn_linear_up_2 = nn.Linear(d_model, self.d_ffn)
        self.ffn_linear_dw = nn.Linear(self.d_ffn, d_model)
        # self.swi_glu = SwiGLU(d_model)

        # Gating Projection (G1), like: https://arxiv.org/abs/2505.06708
        self.gate_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # AliBi Slopes
        self.alibi_head_slopes: List[float] = alibi_utils.get_slopes(num_heads)
        self.register_buffer(
            "alibi_slopes",
            torch.tensor(self.alibi_head_slopes, dtype=torch.float32).view(num_heads, 1, 1)
        )
    
    def _compute_attention(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project and reshape for multi-head attention
        # q, k, v size: [batch_size, self.num_heads, seq_len, self.d_head]
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        
        # Calculate standard scaled dot-product scores
        # scores size: [batch_size, self.num_heads, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        # Generate the bias using the utility function
        alibi_bias = alibi_utils.generate_bidirectional_alibi_bias(
            seq_len=seq_len, 
            slopes=self.alibi_slopes,  # pyright: ignore[reportArgumentType]
            dtype=scores.dtype, 
            device=scores.device
        )
        
        # Add ALiBi bias to the attention scores
        scores = scores + alibi_bias

        # Softmax and context calculation
        attn_weights = F.softmax(scores, dim=-1)
        # Apply attention droput to blind the model over attention scores
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        # Context size: [batch_size, self.num_heads, seq_len, self.d_head]
        context = torch.matmul(attn_weights, v)

        # Reshape and compute attention output
        # Context traspose size  [batch_size, seq_lens, self.num_heads, self.d_head]
        attn_output = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Apply gating like Gated Attention for Large Language Models. 
        # Ref: https://arxiv.org/abs/2505.06708
        gate = torch.sigmoid(self.gate_proj(x))
        gate_attn_output = gate * attn_output

        gated_out = self.out_proj(gate_attn_output)
        # Dropout after final projection (GPT-J style droput, before sum)
        gated_out = F.dropout(gated_out, p=self.dropout, training=self.training)
        
        return gated_out
    
    def _compute_ffn(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute FFW layer like SwiGLU used by LLAMA.
        Ref: https://arxiv.org/abs/2302.13971
        """
        # Projection to hier dim
        x_in_proj_1 = self.ffn_linear_up_1(x)
        x_in_proj_2 = self.ffn_linear_up_2(x)
        # SwiGLU activation like LLAMA
        swish = x_in_proj_1 * torch.sigmoid(x_in_proj_1)
        swiglu = swish * x_in_proj_2
        
        # Inner droput to prevent overfitting
        swiglu = F.dropout(swiglu, p=self.dropout, training=self.training)
        
        # Downscale and dropuout (GPT-J style droput, before sum)
        x_out_proj = self.ffn_linear_dw(swiglu)
        x_out_proj = F.dropout(x_out_proj, p=self.dropout, training=self.training)

        return x_out_proj

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.rms_norm(x)
        attn_output = self._compute_attention(x_norm)
        ffn_output = self._compute_ffn(x_norm)

        return attn_output + ffn_output