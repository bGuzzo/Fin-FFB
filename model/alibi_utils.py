"""
ALiBi (Attention with Linear Biases) Utilities.

This module implements the ALiBi positioning method as described in the paper:
"Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"
(Press et al., 2021) - https://arxiv.org/abs/2108.12409

ALiBi eliminates traditional positional embeddings and instead biases the attention
scores by a value proportional to the distance between tokens. This allows models
to extrapolate to sequence lengths longer than those seen during training.
"""

import torch
from typing import List, Dict, Tuple
import logging

# Global cache for ALiBi bias matrices: (seq_len, num_heads, device, dtype) -> bias_tensor
_ALIBI_CACHE: Dict[Tuple[int, int, torch.device, torch.dtype], torch.Tensor] = {}

def get_slopes(n_heads: int) -> List[float]:
    """
    Apply the general formula provided by the paper: m_i = (2^(-8)/n))^i
    For the head #1 we get 0.7 while for head #16 we get 1/256 (0,004). It converge to 1 for larger n.

    Note: it is a geometric series starting at 2ˆ(-8/16) = 2ˆ(-1/2), each time it's multiplied by itself. 
    """
    start = (2 ** ((-8)/n_heads))
    return [start * (start ** i) for i in range(n_heads)]


def generate_bidirectional_alibi_bias(
    seq_len: int, 
    slopes: torch.Tensor, 
    dtype: torch.dtype, 
    device: torch.device
) -> torch.Tensor:
    """
    Constructs the ALiBi bias matrix for bidirectional (non-causal) attention.
    Utilizes a global cache to avoid redundant computations.

    Args:
        seq_len: The length of the input sequence.
        slopes: A tensor of shape (num_heads, 1, 1) containing the pre-calculated slopes.
        dtype: The desired torch.dtype for the output bias matrix.
        device: The target device (CPU/CUDA).

    Returns:
        A bias tensor of shape (num_heads, seq_len, seq_len).
    """
    # Create a unique key for the cache based on parameters that affect the result.
    # We use the number of heads (slopes.shape[0]) to allow sharing the bias matrix
    # across different layers with the same configuration.
    num_heads = slopes.shape[0]
    cache_key = (seq_len, num_heads, device, dtype)
    
    if cache_key in _ALIBI_CACHE:
        # Verify shape just in case
        cached_bias = _ALIBI_CACHE[cache_key]
        if cached_bias.shape[1] == seq_len:
            return cached_bias

    # Create a 1D tensor of positions: [0, 1, 2, ..., seq_len-1]
    positions = torch.arange(seq_len, device=device, dtype=torch.long)

    # Compute absolute bidirectional distances using broadcasting.
    # Resulting 'distances' shape: (seq_len, seq_len)
    distances = torch.abs(positions.unsqueeze(1) - positions.unsqueeze(0))

    # Apply the slopes and negate. 
    # Result shape: (num_heads, seq_len, seq_len)
    alibi_bias = -1.0 * slopes * distances.to(dtype)
    alibi_bias = alibi_bias.to(device) # Parse to proper dtype format
    
    # Update cache
    _ALIBI_CACHE[cache_key] = alibi_bias

    return alibi_bias

# Test only
if __name__ == "__main__":
    logging.info(get_slopes(32))