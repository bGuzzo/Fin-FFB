"""
Training pipeline for Fin-FFB (Financial Fast Fat BERT).

This script orchestrates the pre-training of the Fin-FFB model using the
Masked Language Modeling (MLM) objective. It is designed to be robust and
efficient on consumer hardware, supporting:
- Multi-device execution (Nvidia CUDA, Apple MPS, CPU).
- Automatic Mixed Precision (AMP) with support for BF16 and FP16.
- Gradient Accumulation to handle large models on limited VRAM.
- Structured checkpointing and comprehensive training logging.
"""

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from data_loader.collector import get_dataloader
from model.fin_ffb import FinFFB
from model.fin_ffb_mlm import FinFFBForMaskedLM


def load_config(config_name: str) -> Dict[str, Any]:
    """
    Loads a YAML configuration file from the config/ directory.

    Args:
        config_name: The stem of the YAML file (e.g., 'nano', 'small').

    Returns:
        A dictionary containing model and training hyperparameters.
    """
    config_path = Path(f"config/{config_name}.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    """
    Detects the best available hardware accelerator.

    Returns:
        torch.device: 'cuda' if Nvidia GPU is available, 'mps' for Apple Silicon,
                      otherwise 'cpu'.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_autocast_context(device: torch.device, dtype: torch.dtype, enabled: bool):
    """
    Provides a unified interface for Automatic Mixed Precision (AMP) autocasting.

    Handles differences between CUDA, CPU, and MPS (Apple Silicon) autocast implementations
    across different PyTorch versions.

    Args:
        device: The target execution device.
        dtype: The target data type for autocasting (e.g., torch.bfloat16).
        enabled: Whether mixed precision is enabled.

    Returns:
        A context manager for mixed precision execution.
    """
    if not enabled or dtype == torch.float32:
        return nullcontext()

    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)
    elif device.type == "cpu":
        return torch.autocast(device_type="cpu", dtype=dtype, enabled=True)
    elif device.type == "mps":
        try:
            # Autocast for MPS is available in newer PyTorch versions (>= 2.4+)
            return torch.autocast(device_type="mps", dtype=dtype, enabled=True)
        except (RuntimeError, TypeError):
            # Gracefully fallback if the environment doesn't support MPS autocast
            return nullcontext()
    return nullcontext()


def main():
    """
    Main entry point for the training pipeline.
    Initializes components, manages the training loop, and saves results.
    """
    parser = argparse.ArgumentParser(description="Train Fin-FFB MLM Model")
    parser.add_argument(
        "--config",
        type=str,
        default="nano",
        help="Config name (nano, small, medium, large)",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument(
        "--save_steps", type=int, default=1000, help="Save checkpoint every X steps"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # --- 1. Workspace Preparation ---
    models_dir = Path("./dumps/models")
    training_dir = Path("./dumps/training")
    models_dir.mkdir(parents=True, exist_ok=True)
    training_dir.mkdir(parents=True, exist_ok=True)

    # Load hyperparameters and hardware configuration
    config = load_config(args.config)

    device = get_device()
    print(f"Using device: {device}")

    # --- 2. Model Initialization ---
    print(f"Initializing model using '{args.config}' config...")
    # Instantiate the core Fin-FFB encoder
    encoder = FinFFB(
        vocab_size=config["model"]["vocab_size"],
        d_model=config["model"]["d_model"],
        num_layers=config["model"]["num_layers"],
        num_heads=config["model"]["num_heads"],
        dropout=config["model"]["dropout"],
        ffn_factor=config["model"]["ffn_factor"],
    )
    # Wrap with MLM head for Masked Language Modeling pre-training
    model = FinFFBForMaskedLM(encoder)
    model.to(device)

    # --- 3. Dataloader Initialization ---
    print("Initializing dataloader...")
    # The dataloader handles JIT tokenization and MLM masking to save disk/RAM
    dataloader = get_dataloader(
        batch_size=config["training"]["batch_size"],
        max_length=config["dataset"]["max_seq_len"],
        mlm_probability=config["dataset"]["mask_probability"],
        tokenizer_name="albert-base-v2",
        num_workers=4,
    )

    # --- 4. Optimizer & Scheduler Setup ---
    # Separate parameters into decay and no-decay groups (standard practice)
    # Weight decay is typically applied to weights, not biases or normalization scales.
    no_decay = ["bias", "rms_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": config["training"]["weight_decay"],
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=float(config["training"]["learning_rate"]),
        betas=tuple(config["training"]["betas"]),
        eps=float(config["training"]["eps"]),
    )

    # Calculate total training steps for the scheduler
    grad_accum_steps = config["training"]["gradient_accumulation_steps"]
    # Account for partial accumulation steps at the end of epochs
    steps_per_epoch = math.ceil(len(dataloader) / grad_accum_steps)
    total_steps = steps_per_epoch * args.epochs

    # Cosine scheduler with linear warmup to stabilize early training
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["training"]["warmup_steps"],
        num_training_steps=total_steps,
    )

    # --- 5. Mixed Precision Setup ---
    # Mixed precision uses float16 or bfloat16 for faster computation and lower VRAM usage.
    mixed_precision = config["training"]["mixed_precision"]
    use_amp = mixed_precision in ["fp16", "bf16"]
    scaler = None
    dtype = torch.float32  # Default precision

    if use_amp:
        if device.type == "cuda":
            # CUDA supports BF16 (Ampere+) and FP16 (with GradScaler)
            if mixed_precision == "bf16" and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
                print("Using CUDA BF16 Mixed Precision")
            else:
                dtype = torch.float16
                # GradScaler prevents underflow of small gradients in float16
                scaler = (
                    torch.amp.GradScaler("cuda")
                    if hasattr(torch.amp, "GradScaler")
                    else torch.cuda.amp.GradScaler()
                )
                print("Using CUDA FP16 Mixed Precision with GradScaler")
        elif device.type == "cpu":
            # CPU autocast predominantly uses bfloat16
            dtype = torch.bfloat16
            print("Using CPU BF16 Mixed Precision")
        elif device.type == "mps":
            # Apple Silicon prefers float16 for mixed precision
            dtype = torch.float16
            print("Using MPS FP16 Mixed Precision")

    # --- 6. Training Loop ---
    print(
        f"Starting training for {args.epochs} epochs with {config['training']['batch_size']} batch size..."
    )
    print(f"Total optimization steps: {total_steps}, Gradient accumulation: {grad_accum_steps}")
    
    start_time = time.time()
    global_step = 0
    accumulated_loss = 0.0
    losses = []
    total_samples = 0

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        # Initialize tqdm progress bar for the current epoch
        epoch_iterator = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            dynamic_ncols=True,
            unit="batch",
        )

        for step, batch in enumerate(epoch_iterator):
            # Move batch data to the target device
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            
            total_samples += input_ids.size(0)

            # --- Forward Pass ---
            with get_autocast_context(device, dtype, use_amp):
                # Returns (loss, logits) - we only need the loss for training
                loss, _ = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                # Scale loss for gradient accumulation
                loss = loss / grad_accum_steps

            # Check for NaN or Inf loss to prevent gradient corruption.
            # This is particularly important with mixed precision (fp16).
            if not math.isfinite(loss.item()):
                tqdm.write(f"\nWarning: Non-finite loss ({loss.item()}) detected at step {step}. "
                           "Skipping batch and clearing accumulated gradients to prevent corruption.")
                optimizer.zero_grad(set_to_none=True)
                continue

            # --- Backward Pass ---
            if scaler is not None:
                # Scaled backward pass for float16
                scaler.scale(loss).backward()
            else:
                # Standard backward pass for bfloat16 or float32
                loss.backward()

            accumulated_loss += loss.item()

            # --- Optimizer Step (Gradient Accumulation Boundary) ---
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
                if scaler is not None:
                    # Unscale gradients, apply clipping, then step the optimizer
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config["training"]["clip_grad_norm"]
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard step logic
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config["training"]["clip_grad_norm"]
                    )
                    optimizer.step()

                # Update learning rate
                scheduler.step()
                
                # Prepare for next accumulation cycle
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                # Track metrics and update live UI
                losses.append(accumulated_loss)
                
                # Calculate throughput (samples per second)
                elapsed = time.time() - start_time
                throughput = total_samples / elapsed
                
                epoch_iterator.set_postfix({
                    "Loss": f"{accumulated_loss:.4f}",
                    "LR": f"{scheduler.get_last_lr()[0]:.2e}",
                    "S/s": f"{throughput:.1f}"
                })
                
                accumulated_loss = 0.0

                # --- Periodical Checkpointing ---
                if global_step % args.save_steps == 0:
                    checkpoint_path = training_dir / f"checkpoint-{global_step}.pt"
                    torch.save(
                        {
                            "step": global_step,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "config": config,
                        },
                        checkpoint_path,
                    )
                    
                    # Log progress to terminal occasionally to keep history
                    tqdm.write(
                        f"Step {global_step}: loss={losses[-1]:.4f}, throughput={throughput:.1f} samples/s"
                    )

    # --- 7. Finalization & Reporting ---
    end_time = time.time()
    training_duration = end_time - start_time

    print("\nTraining complete. Saving final model...")
    # Save raw state dictionary
    final_model_path = models_dir / "fin_ffb_final.pt"
    torch.save(model.state_dict(), final_model_path)
    
    # Save config for reproducibility and HF compatibility
    with open(models_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    # Save in HuggingFace-compatible format (model.safetensors + config.json)
    try:
        model.save_pretrained(models_dir)
    except Exception as e:
        print(f"HuggingFace save_pretrained skipped: {e}")

    # Generate a comprehensive JSON log of the training session
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "hardware": {
            "device": str(device),
            "mixed_precision": config["training"]["mixed_precision"],
            "dtype": str(dtype),
        },
        "training": {
            "epochs": args.epochs,
            "total_steps": global_step,
            "total_samples": total_samples,
            "duration_seconds": training_duration,
            "avg_throughput": total_samples / training_duration,
            "final_loss": losses[-1] if losses else None,
            "avg_loss_last_100_steps": (
                sum(losses[-100:]) / len(losses[-100:])
                if len(losses) >= 100
                else sum(losses) / max(1, len(losses))
            ),
        },
        "config": config,
    }

    log_path = training_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=4)

    print(f"Training log saved to {log_path}")


if __name__ == "__main__":
    main()
