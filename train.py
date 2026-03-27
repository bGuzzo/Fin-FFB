"""
Training pipeline for Fin-FFB (Financial Fast Fat BERT).

This script orchestrates the pre-training of the Fin-FFB model using the
Masked Language Modeling (MLM) objective. It is designed to be robust and
efficient, leveraging high-level utilities for hardware acceleration,
mixed precision, and lifecycle management.
"""

import argparse
import logging
import math
import time
from datetime import datetime
from typing import Any, Dict

import torch
from tqdm import tqdm

from data_loader.mlm_loader import get_dataloader
from data_loader.pd_adpt import PdDataset
from data_loader.mock_dtst import MockDataset

from model.fin_ffb import FinFFB
from model.fin_ffb_mlm import FinFFBForMaskedLM
from utils.train_utils import (
    clear_memory_cache,
    get_autocast_context,
    get_device,
    initialize_optimizer_and_scheduler,
    initialize_weights,
    load_checkpoint,
    load_config,
    log_training_results,
    plot_loss_curve,
    save_checkpoint,
    save_final_artifacts,
    setup_mixed_precision,
    setup_workspace,
)


def _parse_args() -> argparse.Namespace:
    """
    Handles command-line argument parsing for the training pipeline.

    Returns:
        Populated namespace of arguments.
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
    parser.add_argument(
        "--mock", action="store_true", help="Use mock dataset for testing"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint for resumption"
    )
    return parser.parse_args()


def main() -> None:
    """
    Main execution entry point. Orchestrates the high-level training lifecycle:
    1. Environment & Hardware Setup
    2. Model & Data Initialization
    3. Optimization Loop
    4. Artifact Persistence
    """
    args = _parse_args()

    # --- 1. Environment Initialization ---
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = load_config(args.config)
    device = get_device()
    models_dir, training_dir = setup_workspace()
    
    logging.info(f"Using device: {device}")
    logging.info(f"Initializing model using '{args.config}' config...")

    # --- 2. Component Initialization ---
    encoder = FinFFB(
        vocab_size=config["model"]["vocab_size"],
        d_model=config["model"]["d_model"],
        num_layers=config["model"]["num_layers"],
        num_heads=config["model"]["num_heads"],
        dropout=config["model"]["dropout"],
        ffn_factor=config["model"]["ffn_factor"],
        padding_idx=0 # Use 0 as ALBERT use 0
    )
    model = FinFFBForMaskedLM(encoder).to(device)

    # Initialize weights according to standards
    initialize_weights(model, config)

    dataset = MockDataset() if args.mock else PdDataset()
    dataloader = get_dataloader(
        dataset=dataset,
        batch_size=config["training"]["batch_size"],
        max_length=config["dataset"]["max_seq_len"],
        mlm_probability=config["dataset"]["mask_probability"],
        tokenizer_name="albert-base-v2"
    )

    optimizer, scheduler = initialize_optimizer_and_scheduler(
        model, config, len(dataloader), args.epochs
    )

    dtype, scaler, use_amp = setup_mixed_precision(config, device)

    # --- Training State Recovery ---
    start_epoch = 0
    global_step = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler)
        global_step = checkpoint.get("step", 0)
        # Approximate epoch from global_step
        grad_accum_steps = config["training"]["gradient_accumulation_steps"]
        steps_per_epoch = math.ceil(len(dataloader) / grad_accum_steps)
        start_epoch = global_step // steps_per_epoch
        logging.info(f"Resuming from epoch {start_epoch + 1}, global step {global_step}")

    # --- 3. Training Loop ---
    grad_accum_steps = config["training"]["gradient_accumulation_steps"]
    total_steps = math.ceil(len(dataloader) / grad_accum_steps) * args.epochs
    
    logging.info(f"Starting training: {args.epochs} epochs, {total_steps} opt steps.")
    
    start_time = time.time()
    accumulated_loss = 0.0
    losses = []
    total_samples = 0

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, args.epochs):
        epoch_iterator = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            dynamic_ncols=True,
            unit="batch",
        )

        for step, batch in enumerate(epoch_iterator):
            # Skip steps already processed if resuming within an epoch
            steps_per_epoch = math.ceil(len(dataloader) / grad_accum_steps)
            current_step_in_epoch = step // grad_accum_steps
            if epoch == start_epoch and current_step_in_epoch < (global_step % steps_per_epoch):
                if (step + 1) % grad_accum_steps == 0:
                    epoch_iterator.set_description(f"Skipping Step {current_step_in_epoch} (checkpoint)")
                    logging.info(f"Skipping Step {current_step_in_epoch}, it was already done before (from checkpoint).")
                continue

            # Data Movement
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            total_samples += input_ids.size(0)

            # Forward Pass
            with get_autocast_context(device, dtype, use_amp):
                loss, _ = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                # Normalize the loss so gradients scale correctly
                loss = loss / grad_accum_steps

            if not math.isfinite(loss.item()):
                tqdm.write(f"Warning: Non-finite loss at step {step}. Skipping...")
                optimizer.zero_grad(set_to_none=True)
                continue

            # Backward Pass
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            clear_memory_cache(device)
            accumulated_loss += loss.item()

            # Optimization Step
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config["training"]["clip_grad_norm"]
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config["training"]["clip_grad_norm"]
                    )
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                # Normalize loss by gradient accumulation steps
                accumulated_loss = accumulated_loss / grad_accum_steps
                losses.append(accumulated_loss)


                # Metrics Reporting
                throughput = total_samples / (time.time() - start_time)
                epoch_iterator.set_postfix({
                    "Loss": f"{accumulated_loss:.8f}", # Show last loss (not the sum accumulated)
                    "LR": f"{scheduler.get_last_lr()[0]:.2e}",
                    "S/s": f"{throughput:.1f}"
                })
                accumulated_loss = 0.0

                # Checkpointing
                if global_step % args.save_steps == 0:
                    save_checkpoint({
                        "step": global_step,
                        "model_state_dict": model.state_dict(),
                        "encoder_state_dict": model.fin_ffb.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "config": config,
                    }, training_dir, global_step, args.config)
            # End Optimization Step
        # End Dataset Loop
    # End Epoch loop

    # --- 4. Finalization ---
    training_duration = time.time() - start_time
    logging.info("Training complete. Saving artifacts...")
    
    save_final_artifacts(model, config, models_dir, args.config)

    log_training_results({
        "timestamp": datetime.now().isoformat(),
        "hardware": {"device": str(device), "dtype": str(dtype)},
        "metrics": {
            "total_steps": global_step,
            "duration": training_duration,
            "throughput": total_samples / training_duration,
            "final_loss": losses[-1] if losses else None,
        },
        "config": config,
    }, training_dir, args.config)

    # Generate loss visualization
    plot_loss_curve(losses, training_dir, args.config)

    logging.info(f"Session complete. Logs: {training_dir}")


if __name__ == "__main__":
    main()
