"""
This module objective is to mimick the role of the Amygdala in the brain.

It server as primary filter, providing high impulese (emotional responses) to scan for important infromation.
It 'mask-out' information and force the input to focus on few crucial tokens, the emotional input will then be weighted by each layer using Attention Residuals.

I achieve this by using a latent attention (d 64 or 128) with a lower temperature (0.1 or 0.3), this will effectively preserve only few tokens.
The reson for a such a dimentional reduction is that the amygdala is not accurate, it just quicly scan for threads but with a lower resolution.
Dropout don't used intially as it is intended as dense indexer layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional

class AmygdalaIndexer(nn.Module):
    
    def __init__(self, d_model: int, d_latent: int = 64, temp: float = 0.1):
        super(AmygdalaIndexer, self).__init__()
        self.d_model = d_model
        self.d_latent = d_latent
        self.temp = temp
        
        self.down_proj = nn.Linear(d_model, d_latent * 3) # grouped q, k, v projection
        self.up_proj = nn.Linear(d_latent, d_model)
        
        logging.info(f"Initialized AmygdalaIndexer with d_model={d_model}, d_latent={d_latent}, temp={temp}")        


    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()        # Project to latent space
        qkv = self.down_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Scaled Dot-Product Attention in latent space
        # Use low temperature to sharpen the distribution (focus on few tokens)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / (self.d_latent**0.5)
        
        if attention_mask is not None:
            # attention_mask: [batch, seq_len] -> [batch, 1, seq_len]
            # Mask out padding tokens (keys) by adding a large negative value
            mask = attention_mask.unsqueeze(1)
            attn_logits = attn_logits.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_logits / self.temp, dim=-1)

        # Context vector in latent space
        latent_context = torch.matmul(attn_weights, v)

        # Project back to original d_model
        return self.up_proj(latent_context)
