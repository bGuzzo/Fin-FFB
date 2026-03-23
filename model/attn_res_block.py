"""
Full Attention Residuals (AttnRes) implementation for the Fin-FFB model.

This module implements the "Full Attention Residuals" mechanism as described in 
the Kimi Team's technical report (2026). Instead of standard additive residuals 
(h_l = h_{l-1} + f_{l-1}), it uses a depth-wise softmax attention mechanism to 
selectively aggregate information from all preceding layers.

Ref: https://arxiv.org/abs/2603.15031
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from .attn_core import AttentionLayer

class AttnResBlock(nn.Module):
    """
    Wraps an AttentionLayer to implement Full Attention Residuals.
    
    Each layer selectively aggregates previous layer outputs (and the initial 
    embedding) using learned, content-dependent attention weights. This 
    mitigates PreNorm dilution by allowing the model to bypass the standard 
    'additive recurrence' bottleneck.
    
    In this architecture, each AttentionLayer (which computes Attention and 
    FFN concurrently) is treated as a single block in the depth-wise attention.
    """
    
    def __init__(
        self,
        layer_idx: int,
        d_model: int,
        num_heads: int,
        dropout: float,
        ffn_factor: int = 4,
    ):
        super(AttnResBlock, self).__init__()
        self.layer_idx = layer_idx
        self.d_model = d_model
        
        # The core transformation layer f_l(h_l).
        # In this project, f_l computes Attention and FFN in parallel (PaLM style).
        self.layer = AttentionLayer(
            layer=layer_idx,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            ffn_factor=ffn_factor
        )
        
        # Learned pseudo-query vector (w_l) for depth-wise attention.
        # Represented as a Linear projection to compute scalar logits for each source.
        self.attn_res_proj = nn.Linear(d_model, 1, bias=False)
        
        # Initialize pseudo-query to zero.
        # Per paper Section 5: This ensures initial attention weights are uniform 
        # across sources, reducing AttnRes to an equal-weight average at start-up.
        nn.init.zeros_(self.attn_res_proj.weight)
        
        # RMSNorm for key representations in depth-wise attention.
        # Section 3.1: Prevents layers with large-magnitude outputs from 
        # dominating the attention weights.
        self.attn_res_norm = nn.RMSNorm(d_model)

    def forward(
        self, 
        history: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass for Full Attention Residuals.
        
        Args:
            history: List of tensors [v_0, v_1, ..., v_{l-1}] where v_0 is the 
                     token embedding and v_{1...l-1} are previous layer outputs.
                     Shape: [Batch, SeqLen, d_model]
            
        Returns:
            v_l: The output of the current layer f_l(h_l).
            history: Updated history list including v_l.
        """
        # 1. Prepare Value matrix (V) from all preceding outputs.
        # V shape: [num_sources, Batch, SeqLen, d_model]
        V = torch.stack(history)
        
        # 2. Normalize keys (K) to stabilize softmax attention.
        K = self.attn_res_norm(V)
        
        # 3. Compute depth-wise attention logits.
        # logits = w_l^T * K
        # 'd' is d_model, 'n' is number of sources, 'b' is batch, 't' is seq len.
        wl = self.attn_res_proj.weight.squeeze(0)
        logits = torch.einsum('d, n b t d -> n b t', wl, K)
        
        # 4. Compute Softmax Weights (alpha).
        # Each layer l independently decides how much to attend to each source i < l.
        alpha = F.softmax(logits, dim=0)
        
        # 5. Selective Aggregation (h_l).
        # h_l = sum(alpha_i * v_i)
        hl = torch.einsum('n b t, n b t d -> b t d', alpha, V)
        
        # 6. Apply concurrent Attention & FFN transformation (f_l).
        vl = self.layer(hl)
        
        # 7. Update history for the next layer in the stack.
        history.append(vl)
        
        return vl, history
