"""
This module's objective is to mimic the role of the Amygdala in the biological brain, 
serving as a primary filter for salient information. 

By acting as 'Layer 0' in the Fin-FFB architecture, it provides an 'emotional' 
high-pass filter that scans for crucial tokens before deeper processing. In the 
context of the Full Attention Residuals (AttnRes) framework, this layer offers 
several architectural advantages that explain the observed performance gains:

1. **Dual-Path Initialization**: It creates a parallel path for gradients. 
   Subsequent layers can choose between raw embeddings and these salient 
   highlights, significantly accelerating convergence.
2. **Representation De-correlation**: The low-temperature (sharp) attention 
   helps break the 'embedding cone' effect (representation collapse) by 
   introducing high-entropy, non-linear signals early in the history list.
3. **Latent Efficiency**: Operating in a reduced dimensional space 
   (d_latent << d_model) allows for complex salience filtering with minimal 
   computational overhead.

This is achieved using latent attention with a low temperature (0.1 to 0.3), 
effectively sharpening the focus to only a few tokens. The dimensional reduction 
reflects the biological reality: the Amygdala is fast but lower-resolution, 
quickly scanning for salient features (or 'threats') rather than processing 
fine details.

Dropout is not used here as this functions as a dense, deterministic indexing layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional

class AmygdalaIndexer(nn.Module):
    """
    A latent attention layer that acts as a high-pass filter for salient tokens.
    
    Reduces the sequence to its most 'emotional' (salient) components using 
    low-temperature attention in a reduced dimensional space.
    """
    
    def __init__(self, d_model: int, d_latent: int = 64, temp: float = 0.1):
        super(AmygdalaIndexer, self).__init__()
        self.d_model = d_model
        self.d_latent = d_latent
        self.temp = temp
        
        # Pre-normalization to stabilize the sharp latent attention
        self.rms_norm = nn.RMSNorm(d_model)
        
        # Latent projections
        self.down_proj = nn.Linear(d_model, d_latent * 3) # grouped q, k, v projection
        self.up_proj = nn.Linear(d_latent, d_model)
        
        # Explicit initialization for stable start-up
        self._init_weights()
        
        logging.info(f"Initialized AmygdalaIndexer: d_model={d_model}, d_latent={d_latent}, temp={temp}")        

    def _init_weights(self):
        """Initialize weights to ensure a healthy initial signal."""
        nn.init.xavier_uniform_(self.down_proj.weight)
        nn.init.xavier_uniform_(self.up_proj.weight)
        if self.down_proj.bias is not None:
            nn.init.zeros_(self.down_proj.bias)
        if self.up_proj.bias is not None:
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Filters the input sequence through a sharp latent attention mechanism.
        """
        batch_size, seq_len, _ = x.size()
        
        # 1. Normalize and project to latent space
        x_norm = self.rms_norm(x)
        qkv = self.down_proj(x_norm)
        q, k, v = qkv.chunk(3, dim=-1)

        # 2. Scaled Dot-Product Attention in latent space.
        # Scale by d_latent instead of d_model to account for reduced space.
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / (self.d_latent**0.5)
        
        # 3. Apply masking for padding tokens
        if attention_mask is not None:
            # attention_mask: [batch, seq_len] -> [batch, 1, seq_len]
            # We mask out the 'keys' (tokens being attended to).
            mask = attention_mask.unsqueeze(1)
            attn_logits = attn_logits.masked_fill(mask == 0, float("-inf"))

        # 4. Sharpen distribution using temperature and compute context
        attn_weights = F.softmax(attn_logits / self.temp, dim=-1)
        latent_context = torch.matmul(attn_weights, v)

        # 5. Project back to original d_model
        return self.up_proj(latent_context)
