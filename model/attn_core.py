"""
Core Bidirectional Attention Layer (PaLM-style parallel architecture).

This module implements the transformation function f_l(h_l) used within the
Attention Residuals (AttnRes) framework. In this architecture, each layer
computes Attention and FFN in parallel on the same normalized input.

Crucially, this layer does NOT include its own additive residual connection
(i.e., it computes f_l(h_l) but not h_l + f_l(h_l)), as the depth-wise
selective aggregation mechanism handles the information flow across depth.

Ref: https://arxiv.org/abs/2603.15031
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import alibi_utils
from typing import List, Optional
import math


class AttentionLayer(nn.Module):
    """
    Parallel Attention and FFN transformation layer.

    This implementation follows the PaLM/GPT-J style where Attention and
    Feed-Forward networks are computed concurrently on the same normalized input.
    """

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

        # RMS pre-normalization
        self.rms_norm = nn.RMSNorm(d_model)

        # Attention modules (Fused QKV projection for efficiency)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)

        # FFN modules
        self.ffn_linear_up = nn.Linear(d_model, self.d_ffn * 2)
        self.ffn_linear_dw = nn.Linear(self.d_ffn, d_model)

        # Gating Projection (G1), following gated attention architectures
        # Ref: https://arxiv.org/abs/2505.06708
        self.gate_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # AliBi Slopes
        self.alibi_head_slopes: List[float] = alibi_utils.get_slopes(num_heads)
        self.register_buffer(
            "alibi_slopes",
            torch.tensor(self.alibi_head_slopes, dtype=torch.float32).view(
                num_heads, 1, 1
            ),
        )

    def _compute_attention(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """"
        Compute Gated Scaled Dot-Product Attention.
        Gating Paper Ref: 

        Note: I don't use torch.nn.functional.scaled_dot_product_attention, it not optimized on M4 mac (which I'm using).

        Args:
            x: Input tensor [batch, seq_len, d_model].
            attention_mask: Optional mask for attention scores.
        
        Returns:
            torch.Tensor: Attention output [batch, seq_len, d_model].

        """

        batch_size, seq_len, _ = x.shape

        # Fused QKV projection
        # [batch, seq_len, 3 * d_model]
        qkv = self.qkv_proj(x)

        # Reshape and split into Q, K, V
        # [batch, seq_len, 3, num_heads, d_head] -> [3, batch, num_heads, seq_len, d_head]
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.d_head).permute(
            2, 0, 3, 1, 4
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # Add bidirectional ALiBi bias
        alibi_bias = alibi_utils.generate_bidirectional_alibi_bias(
            seq_len=seq_len,
            slopes=self.alibi_slopes,  # pyright: ignore[reportArgumentType]
            dtype=scores.dtype,
            device=scores.device,
        )
        scores = scores + alibi_bias

        # Apply attention mask if provided
        if attention_mask is not None:
            # attention_mask: [batch, seq_len] -> [batch, 1, 1, seq_len]
            mask = attention_mask.view(batch_size, 1, 1, seq_len)
            # Mask out padding tokens by adding a large negative value
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        context = torch.matmul(attn_weights, v)
        attn_output = (
            context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        )

        # Apply Gated Attention mechanism
        gate = torch.sigmoid(self.gate_proj(x))
        gate_attn_output = gate * attn_output
        gated_out = self.out_proj(gate_attn_output)
        return gated_out

    def _compute_ffn(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute SwiGLU FFN (as used in Llama architectures).
        Ref: https://arxiv.org/abs/2302.13971
        """
        x_in_proj = self.ffn_linear_up(x)
        x_in_proj_1, x_in_proj_2 = x_in_proj.chunk(2, dim=-1)


        # SwiGLU: Swish(xW) * xV
        swish = x_in_proj_1 * torch.sigmoid(x_in_proj_1)
        swiglu = swish * x_in_proj_2

        # Downscale
        x_out_proj = self.ffn_linear_dw(swiglu)
        return x_out_proj

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for the transformation f_l(h_l).

        Returns: 
            torch.Tensor: Dropout(SelfAttn(x) + FFN(x)).

        Note: This method computes the parallel Attention + FFN block. It does
        NOT include a residual connection, as that is managed by the depth-wise
        attention in the AttnRes wrapper.
        """
        x_norm = self.rms_norm(x)
        attn_output = self._compute_attention(x_norm, attention_mask=attention_mask)
        ffn_output = self._compute_ffn(x_norm)

        # Dropout applied only after sum for more training stability
        return F.dropout(attn_output + ffn_output, p=self.dropout, training=self.training)

