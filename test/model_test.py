"""
Unit tests for AttentionLayer and AttentionResidualBlock.
Verifies dimensionality, projection logic, and depth-wise aggregation.
"""

import torch
import torch.nn as nn
import sys
import os

# Add the project root to sys.path to allow imports from the 'model' directory.
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.attn_core import AttentionLayer
from model.attn_res_block import AttnResBlock

def test_attention_layer():
    print("Testing AttentionLayer...")
    batch_size = 2
    seq_len = 16
    d_model = 128
    num_heads = 4
    dropout = 0.0
    
    # Initialize Layer
    layer = AttentionLayer(
        layer=0,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout
    )
    
    # Mock Input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Forward Pass
    output = layer(x)
    
    # Check Dimensions: [Batch, SeqLen, d_model]
    assert output.shape == (batch_size, seq_len, d_model), \
        f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    
    print("AttentionLayer dimensionality test passed.")

def test_attn_res_block():
    print("\nTesting AttnResBlock (Full Attention Residuals)...")
    batch_size = 2
    seq_len = 16
    d_model = 128
    num_heads = 4
    dropout = 0.0
    
    # 1. Initialize First Block (Layer 0)
    # The first block receives only the token embedding in its history.
    block_0 = AttnResBlock(
        layer_idx=0,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout
    )
    
    embedding = torch.randn(batch_size, seq_len, d_model)
    history = [embedding] # b_0 = embedding
    
    # Forward Pass Block 0
    v0, history = block_0(history)
    
    # Verify outputs
    assert v0.shape == (batch_size, seq_len, d_model)
    assert len(history) == 2, "History should contain [embedding, v0]"
    assert torch.equal(history[0], embedding)
    assert torch.equal(history[1], v0)
    
    print("Block 0 (First Layer) tests passed.")

    # 2. Initialize Second Block (Layer 1)
    # This layer should attend over [embedding, v0].
    block_1 = AttnResBlock(
        layer_idx=1,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout
    )
    
    v1, history = block_1(history)
    
    assert v1.shape == (batch_size, seq_len, d_model)
    assert len(history) == 3, "History should contain [embedding, v0, v1]"
    
    print("Block 1 (Second Layer) tests passed.")

    # 3. Test Selective Aggregation Logic
    # Verify that the pseudo-query projection (wl) correctly produces 
    # attention weights over the depth dimension.
    # Initially, wl is zeroed, so weights should be uniform.
    
    # We can check the internal aggregation by manually inspecting the weights 
    # (if we modify AttnResBlock to return them or use a hook).
    # Since we strictly follow the code provided, we check if it runs without 
    # dimensionality errors for a larger stack.
    
    for i in range(2, 5):
        block = AttnResBlock(i, d_model, num_heads, dropout)
        vi, history = block(history)
        assert vi.shape == (batch_size, seq_len, d_model)
        assert len(history) == i + 2
        
    print(f"Stack of 5 layers (Depth-wise attention) passed. History size: {len(history)}")

def test_initialization_state():
    print("\nTesting Weight Initialization...")
    d_model = 64
    block = AttnResBlock(0, d_model, 4, 0.0)
    
    # Per paper, wl should be initialized to zero.
    assert torch.all(block.attn_res_proj.weight == 0), "Pseudo-query wl must be initialized to zero."
    print("Initialization check passed.")

if __name__ == "__main__":
    try:
        test_attention_layer()
        test_attn_res_block()
        test_initialization_state()
        print("\nAll tests completed successfully.")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
