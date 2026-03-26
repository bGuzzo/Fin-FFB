import torch
import logging
from transformers import AutoTokenizer
from model.fin_ffb import FinFFB
from utils.train_utils import load_config, get_device

# The logging configuration is already applied via test/__init__.py import
import test

def run_test(checkpoint_path: str = None, config_name: str = "nano"):
    device = get_device()
    config = load_config(config_name)
    
    logging.info(f"Initializing model with config: {config_name}")
    
    # Initialize Model
    model = FinFFB(
        vocab_size=config["model"]["vocab_size"],
        d_model=config["model"]["d_model"],
        num_layers=config["model"]["num_layers"],
        num_heads=config["model"]["num_heads"],
        dropout=config["model"]["dropout"],
        ffn_factor=config["model"]["ffn_factor"],
        padding_idx=0
    ).to(device)
    
    # Load Weights if path provided
    if checkpoint_path:
        logging.info(f"Loading weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Check if it's a full training checkpoint or just state_dict
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        # Handle potential MLM wrapper prefix if loaded from a training checkpoint
        if any(k.startswith("fin_ffb.") for k in state_dict.keys()):
            state_dict = {k.replace("fin_ffb.", ""): v for k, v in state_dict.items() if k.startswith("fin_ffb.")}
        model.load_state_dict(state_dict)
    else:
        logging.warning("No checkpoint provided, using randomly initialized weights.")
    
    model.eval()
    
    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
    sentence = "Financial markets are showing high volatility today."
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    
    # Inference
    with torch.no_grad():
        output, history = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_history=True
        )
    
    # logging.info Results
    logging.info(f"Input Sentence: {sentence}")
    logging.info(f"Output shape: {output.shape}")
    
    # Stack history to show depth-wise tensors [Layers, Batch, Seq, Dim]
    stacked_history = torch.stack(history)
    logging.info(f"Stacked History shape (v0 to vL): {stacked_history.shape}")
    
    logging.info("--- Final Model Output (h_out) [First 5 dims of first token] ---")
    logging.info(output[0, 0, :5])
    
    logging.info("--- Stacked History (v_0 to v_L) [First 5 dims of first token across layers] ---")
    for i, h in enumerate(history):
        layer_name = "Embedding (v0)" if i == 0 else f"Layer {i} (v{i})"
        logging.info(f"{layer_name}: {h[0, 0, :5]}")

if __name__ == "__main__":
    import sys
    # Example usage: python test/load_model_test.py [checkpoint_path] [config_name]
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    cfg = sys.argv[2] if len(sys.argv) > 2 else "nano"
    run_test(ckpt, cfg)
