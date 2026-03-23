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
from typing import List, Tuple, Optional, Union
from .attn_core import AttentionLayer

class AttnResBlock(nn.Module):
    """
    Wraps an AttentionLayer to implement Full Attention Residuals (AttnRes).
    
    This module serves two roles depending on 'final_layer':
    1.  Intermediate Layer: Selectively aggregates all preceding representations 
        into h_l, applies the transformation f_l(h_l), and updates the history.
    2.  Final Output Layer: Aggregates the full history (all layer outputs + 
        embedding) to produce the final model representation, skipping the 
        transformation and history update.
    
    Ref: https://arxiv.org/abs/2603.15031
    """
    
    def __init__(
        self,
        layer_idx: int,
        d_model: int,
        num_heads: int,
        dropout: float,
        ffn_factor: int = 4,
        final_layer: bool = False
    ):
        super(AttnResBlock, self).__init__()
        self.layer_idx = layer_idx
        self.d_model = d_model
        self.final_layer = final_layer
        
        if not self.final_layer:
            # The core transformation layer f_l(h_l).
            # Computes Attention and FFN in parallel (PaLM style).
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
        # Per paper Section 5: This ensures initial attention weights are uniform,
        # reducing AttnRes to an equal-weight average at start-up.
        nn.init.zeros_(self.attn_res_proj.weight)
        
        # RMSNorm for key representations in depth-wise attention.
        # Section 3.1: Prevents layers with large-magnitude outputs from 
        # dominating the attention weights.
        self.attn_res_norm = nn.RMSNorm(d_model)

    def forward(
        self, 
        history: List[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None
    ) -> Union[Tuple[torch.Tensor, List[torch.Tensor]], torch.Tensor]:
        """
        Forward pass for Full Attention Residuals.
        
        Args:
            history: List of tensors [v_0, v_1, ..., v_{l-1}] where v_0 is the 
                     token embedding and v_{1...l-1} are previous layer outputs.
                     Shape: [Batch, SeqLen, d_model]
            attention_mask: Optional mask for attention scores in core transformation.
            
        Returns:
            If final_layer:
                h_out: The final aggregated hidden state [Batch, SeqLen, d_model].
            Else:
                v_l: The output of the current layer transformation f_l(h_l).
                history: Updated history list including v_l.
        """
        # 1. Prepare Value matrix (V) from all preceding outputs.
        # V shape: [num_sources, Batch, SeqLen, d_model]
        V = torch.stack(history)
        
        # 2. Normalize keys (K) to stabilize softmax attention.
        K = self.attn_res_norm(V)
        
        # 3. Compute depth-wise attention logits using learned query w_l.
        wl = self.attn_res_proj.weight.squeeze(0)
        logits = torch.einsum('d, n b t d -> n b t', wl, K)
        
        # 4. Compute Softmax Weights (alpha) across the depth dimension.
        alpha = F.softmax(logits, dim=0)
        
        # 5. Selective Aggregation (h_l).
        # h_l = sum(alpha_i * v_i)
        hl = torch.einsum('n b t, n b t d -> b t d', alpha, V)
        
        # If this is the final aggregation layer, we return the hidden state 
        # directly for the LM head.
        if self.final_layer:
            return hl
        
        # 6. Apply core transformation f_l (Attention & FFN).
        vl = self.layer(hl, attention_mask=attention_mask)
        
        # 7. Update history for subsequent layers.
        history.append(vl)
        
        return vl, history
