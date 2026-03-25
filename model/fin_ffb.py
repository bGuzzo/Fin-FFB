"""
Final Fin-FFB (Financial Fast Fat BERT) Model implementation.

This model serves as a shallow, wide, and highly-specialized encoder for 
financial forecasting and analysis. It integrates PaLM-style parallel 
computations, Full Attention Residuals (AttnRes), and Gated Attention.
"""

VERSION = "1.0.0"

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from .attn_res_block import AttnResBlock
from huggingface_hub import PyTorchModelHubMixin
import torch.nn.functional as F

class FinFFB(nn.Module, PyTorchModelHubMixin):
    """
    Fin-FFB (Financial Fast Fat BERT) Encoder.
    
    A text encoder designed for high speed and performance on financial texts.
    It uses a bidirectional transformer-like architecture with parallel FFN 
    and Attention, combined with the novel Full Attention Residuals (AttnRes) mechanism.
    
    Args:
        vocab_size (int): Size of the token vocabulary. Default 30000.
        d_model (int): Hidden dimension size. Default 768.
        num_layers (int): Number of transformation layers. Default 3.
        num_heads (int): Number of attention heads. Default 12.
        dropout (float): Dropout probability. Default 0.1.
        ffn_factor (int): Multiplier for the hidden layer in FFN. Default 4.
        padding_idx (Optional[int]): Index for padding token in embedding.
    """

    def __init__(
        self,
        vocab_size: int = 30000,
        d_model: int = 768,
        num_layers: int = 3,
        num_heads: int = 12,
        dropout: float = 0.1,
        ffn_factor: int = 4,
        padding_idx: Optional[int] = None
    ):
        super(FinFFB, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.dropout = dropout
        self.ffn_factor = ffn_factor
        self.num_heads = num_heads
        self.padding_idx = padding_idx

        self.info_str : str = f"Fin-FBB Version {VERSION}. d_model={d_model}, num_layers={num_layers}, num_heads={num_heads}, dropout(train)={dropout}, ffn_factor={ffn_factor}"
        print(self.info_str)
        
        # Token Embeddings
        self.embeddings = nn.Embedding(
            vocab_size, 
            d_model,
            padding_idx=padding_idx
        )
        
        # Final normalization
        self.rms_norm = nn.RMSNorm(d_model)
        
        # Transformation Layers (Intermediate)
        # Each layer l computes v_l = f_l(h_l) where h_l = AttnRes(v_0...v_{l-1})
        self.layers = nn.ModuleList([
            AttnResBlock(
                layer_idx=i,
                d_model=self.d_model,
                num_heads=self.num_heads,
                dropout=self.dropout,
                ffn_factor=self.ffn_factor,
                final_layer=False
            )
            for i in range(num_layers)
        ])
        
        # Final Aggregation Layer
        # Computes final representation h_{L+1} = AttnRes(v_0...v_L)
        self.final_aggregator = AttnResBlock(
            layer_idx=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            ffn_factor=ffn_factor,
            final_layer=True
        )

    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_history: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass of the Fin-FFB model.
        
        Args:
            input_ids (torch.Tensor): Integer token IDs of shape [batch, seq_len].
            attention_mask (torch.Tensor, optional): Binary mask of shape [batch, seq_len].
            return_history (bool): If True, also returns the full list of layer outputs.
            
        Returns:
            torch.Tensor: The final aggregated hidden states of shape [batch, seq_len, d_model].
            List[torch.Tensor] (optional): All intermediate layer representations [v_0, v_1, ..., v_L].
        """
        # 1. Initial Embedding (v_0) and dropout
        v0 = self.embeddings(input_ids)
        v0 = F.dropout(v0, p=self.dropout, training=self.training)
        
        # Initialize history with token embeddings
        history: List[torch.Tensor] = [v0]
        
        # 2. Sequential transformation blocks
        for layer in self.layers:
            # Each block performs selective aggregation (h_l), transformation (v_l),
            # and returns (v_l, updated_history).
            # AttnResBlock.forward returns Union, so we ignore type for history update.
            _, history = layer(history, attention_mask=attention_mask) # type: ignore
            
        # 3. Final selective aggregation of the complete history
        # Produces the final output representation h_{L+1}
        h_out = self.final_aggregator(history, attention_mask=attention_mask) # type: ignore
        
        # 4. Final normlization and linear
        h_out = self.rms_norm(h_out)
        
        if return_history:
            return h_out, history
            
        return h_out
