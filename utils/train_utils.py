"""
Training Utilities for Fin-FFB (Financial Fast Fat BERT).

This module provides high-level abstractions for training setup, hardware management,
mixed precision, and persistence, following SOLID principles to ensure the training
pipeline is maintainable and scalable.
"""

import gc
import json
import math
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler
from transformers import get_cosine_schedule_with_warmup

import logging

TIME_FORMAT = "%Y-%m-%d_%H-%M-%S"


def load_config(config_name: str) -> Dict[str, Any]:
    """
    Loads a YAML configuration file from the config/ directory.

    Args:
        config_name: The stem of the YAML file (e.g., 'nano', 'small').

    Returns:
        A dictionary containing model and training hyperparameters.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
    """
    config_path = Path(f"config/{config_name}.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    logging.info(f"Loading configuration from: {config_path}")
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
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logging.info(f"Using device: {device}")
    return device


def setup_workspace(base_dir: str = "./dumps") -> Tuple[Path, Path]:
    """
    Initializes the directory structure for models and training logs.

    Args:
        base_dir: Root directory for output artifacts.

    Returns:
        A tuple of (models_dir, training_dir) Path objects.
    """
    models_dir = Path(base_dir) / "models"
    training_dir = Path(base_dir) / "training"
    models_dir.mkdir(parents=True, exist_ok=True)
    training_dir.mkdir(parents=True, exist_ok=True)

    logging.info(
        f"Workspace setup: models_dir={models_dir}, training_dir={training_dir}"
    )
    return models_dir, training_dir


def clear_memory_cache(device: torch.device) -> None:
    """
    Triggers Python garbage collection and clears device-specific caches.

    Args:
        device: The current execution device.
    """
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def get_autocast_context(
    device: torch.device, dtype: torch.dtype, enabled: bool
) -> Any:
    """
    Provides a unified interface for Automatic Mixed Precision (AMP) autocasting.

    Args:
        device: The target execution device.
        dtype: The target data type for autocasting.
        enabled: Whether mixed precision is enabled.

    Returns:
        A context manager for mixed precision execution.
    """
    if not enabled or dtype == torch.float32:
        return nullcontext()

    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)
    if device.type == "cpu":
        return torch.autocast(device_type="cpu", dtype=dtype, enabled=True)
    if device.type == "mps":
        try:
            return torch.autocast(device_type="mps", dtype=dtype, enabled=True)
        except (RuntimeError, TypeError):
            logging.exception(
                f"Unable to get autocast context for {device}, falling back to nullcontext."
            )
            return nullcontext()

    logging.warning(f"Unknown device type: {device.type}, falling back to nullcontext.")
    return nullcontext()


def _init_weights(module: torch.nn.Module, initializer_range: float):
    if isinstance(module, torch.nn.Linear):
        # BERT standard: normal distribution for weights, zeros for bias
        torch.nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, torch.nn.Embedding):
        # BERT standard: normal distribution for weights
        # Embedding don't use biases (only a lookup matrix)
        torch.nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, (torch.nn.RMSNorm, torch.nn.LayerNorm)):
        # RMSNorm/LayerNorm standard: weights to 1, bias to 0
        torch.nn.init.ones_(module.weight)
        # RMSNorm don't use bias
        if hasattr(module, "bias") and module.bias is not None:
            torch.nn.init.zeros_(module.bias)


def initialize_weights(model: torch.nn.Module, config: Dict[str, Any]) -> None:
    """
    Initializes model parameters according to BERT, SwiGLU, and RMSNorm standards.

    Args:
        model: The model to initialize.
        config: Configuration dictionary containing 'model.initializer_range'.
    """
    initializer_range = config["model"].get("initializer_range", 0.02)

    # Apply standard initialization
    model.apply(lambda m: _init_weights(m, initializer_range))

    logging.info(
        f"Model weights initialized with initializer_range: {initializer_range}"
    )


def initialize_optimizer_and_scheduler(
    model: torch.nn.Module,
    config: Dict[str, Any],
    dataloader_len: int,
    epochs: int,
) -> Tuple[AdamW, LRScheduler]:
    """
    Configures the AdamW optimizer with weight decay groups and a cosine scheduler.

    Args:
        model: The model to optimize.
        config: Training configuration dictionary.
        dataloader_len: Number of batches per epoch.
        epochs: Total number of training epochs.

    Returns:
        A tuple of (optimizer, scheduler).
    """
    params = _get_optimizer_grouped_parameters(
        model, config["training"]["weight_decay"]
    )

    optimizer = AdamW(
        params,
        lr=float(config["training"]["learning_rate"]),
        betas=tuple(config["training"]["betas"]),
        eps=float(config["training"]["eps"]),
    )
    logging.info(
        f"Initialized optimizer with learning rate: {config['training']['learning_rate']}"
    )

    # Calculate total training steps
    grad_accum_steps = config["training"]["gradient_accumulation_steps"]
    steps_per_epoch = math.ceil(dataloader_len / grad_accum_steps)
    total_steps = steps_per_epoch * epochs

    # Calculate warmup steps as a percentage of total steps
    warmup_steps_perc = config["training"].get("warmup_steps_perc", 0.1)
    num_warmup_steps = int(total_steps * warmup_steps_perc)

    logging.info(f"Total training steps: {total_steps} ({steps_per_epoch} steps/epoch)")
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps,
    )

    logging.info(
        f"Initialized scheduler, total steps: {total_steps}, warmup steps: {num_warmup_steps} ({warmup_steps_perc*100:.1f}%)"
    )
    return optimizer, scheduler


def setup_mixed_precision(
    config: Dict[str, Any], device: torch.device
) -> Tuple[torch.dtype, Optional[torch.amp.GradScaler], bool]:
    """
    Configures AMP settings based on hardware capabilities.

    Args:
        config: Training configuration dictionary.
        device: Target execution device.

    Returns:
        A tuple of (dtype, scaler, use_amp).
    """
    mixed_precision = config["training"]["mixed_precision"]
    use_amp = mixed_precision in ["fp16", "bf16"]
    dtype = torch.bfloat16  # Use bfloat16 as defualt
    scaler = None

    if not use_amp:
        return dtype, scaler, use_amp

    if device.type == "cuda":
        if mixed_precision == "bf16" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
            scaler = (
                torch.amp.GradScaler("cuda")
                if hasattr(torch.amp, "GradScaler")
                else torch.cuda.amp.GradScaler()
            )
    elif device.type == "cpu":
        dtype = torch.bfloat16
    elif device.type == "mps":
        # Using M4, use bf16!
        dtype = torch.bfloat16

    logging.info(
        f"Configured mixed precision: dtype={str(dtype)}, use_amp={use_amp}, scaler={str(scaler)} on device {device}"
    )
    return dtype, scaler, use_amp


def save_checkpoint(
    state: Dict[str, Any], training_dir: Path, global_step: int, config_name: str
) -> None:
    """
    Persists a training checkpoint to disk.

    Args:
        state: Dictionary containing training state (model, optimizer, etc.).
        training_dir: Path to the training artifacts directory.
        global_step: Current global optimization step.
        config_name: Name of the configuration used.
    """
    timestamp = datetime.now().strftime(TIME_FORMAT)
    checkpoint_path = (
        training_dir
        / f"checkpoint_{config_name or ''}_{timestamp}_step_{global_step}.pt"
    )
    torch.save(state, checkpoint_path)
    logging.info(f"Saved checkpoint to {checkpoint_path} for step {global_step}")


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRScheduler] = None,
) -> Dict[str, Any]:
    """
    Loads a training checkpoint from disk and restores component states.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        model: The model instance to load weights into.
        optimizer: Optional optimizer to restore state.
        scheduler: Optional scheduler to restore state.

    Returns:
        The full checkpoint dictionary containing metadata (step, epoch, etc.).
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    logging.info(f"Loading checkpoint from: {path}")
    checkpoint = torch.load(path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logging.info("Restored optimizer state.")
        
    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logging.info("Restored scheduler state.")

    logging.info(f"Checkpoint loaded successfully. Resuming from step {checkpoint.get('step', 'unknown')}")
    return checkpoint


def save_final_artifacts(
    model: torch.nn.Module, config: Dict[str, Any], models_dir: Path, config_name: str
) -> None:
    """
    Saves the final model state and configuration files.

    Args:
        model: The trained model.
        config: Training configuration dictionary.
        models_dir: Path to the models directory.
        config_name: Name of the configuration used.
    """
    timestamp = datetime.now().strftime(TIME_FORMAT)

    # Save standard PyTorch state dict
    state_dict_path = models_dir / f"fin_ffb_mlm_{config_name or ''}_{timestamp}_final.pt"
    torch.save(model.state_dict(), state_dict_path)
    logging.info(f"Saved final model state (MLM) to {state_dict_path}")

    # Save PyTorch state dict for core model (pure Fin-FFB)
    embd_state_dict_path = models_dir / f"fin_ffb_endc_{config_name or ''}_{timestamp}_final.pt"
    torch.save(model.fin_ffb.state_dict(), state_dict_path)
    logging.info(f"Saved final model state (Fin-FFB) to {state_dict_path}")

    # Save config for reproducibility
    config_save_path = models_dir / f"config_{config_name or ''}_{timestamp}.json"
    with open(config_save_path, "w") as f:
        json.dump(config, f, indent=4)
        logging.info(f"Saved JSON config to {config_save_path}")

    # Attempt HuggingFace-style save
    try:
        if hasattr(model, "save_pretrained"):
            hf_save_path = models_dir / f"hf_{config_name or ''}_{timestamp}"
            model.save_pretrained(hf_save_path)
            logging.info(f"Saved HuggingFace-style model to {hf_save_path}")
    except Exception as e:
        logging.exception(f"Non-critical: HF save_pretrained skipped: {e}")


def log_training_results(
    results: Dict[str, Any], training_dir: Path, config_name: str
) -> None:
    """
    Saves a comprehensive JSON log of the training session.

    Args:
        results: Dictionary containing training metrics and hardware info.
        training_dir: Path to the training artifacts directory.
        config_name: Name of the configuration used.
    """
    timestamp = datetime.now().strftime(TIME_FORMAT)
    log_path = training_dir / f"training_log_{config_name or ''}_{timestamp}.json"
    with open(log_path, "w") as f:
        json.dump(results, f, indent=4)
    logging.info(f"Training results logged to {log_path}")


def _get_optimizer_grouped_parameters(
    model: torch.nn.Module, weight_decay: float
) -> List[Dict[str, Any]]:
    """
    Groups parameters into decay and no-decay sets.

    Args:
        model: Model to extract parameters from.
        weight_decay: Weight decay factor for applicable layers.

    Returns:
        List of parameter groups for the optimizer.
    """
    no_decay = ["bias", "rms_norm.weight"]

    decay_params = [
        p
        for n, p in model.named_parameters()
        if not any(nd in n for nd in no_decay) and p.requires_grad
    ]
    no_decay_params = [
        p
        for n, p in model.named_parameters()
        if any(nd in n for nd in no_decay) and p.requires_grad
    ]

    logging.info(
        f"Optimizer groups: {len(decay_params)} params with weight_decay={weight_decay}, "
        f"{len(no_decay_params)} params with weight_decay=0.0"
    )

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def plot_loss_curve(losses: List[float], training_dir: Path, config_name: str) -> None:
    """
    Generates and saves a line chart of the training loss.

    Args:
        losses: List of loss values recorded during training.
        training_dir: Path to the directory where the chart will be saved.
        config_name: Name of the configuration used.
    """
    if not losses:
        logging.warning("No losses recorded. Skipping loss curve generation.")
        return

    timestamp = datetime.now().strftime(TIME_FORMAT)
    plot_path = training_dir / f"loss_curve_{config_name or ''}_{timestamp}.png"

    plt.figure(figsize=(10, 6))
    plt.plot(losses, label="Training Loss")
    plt.title(f"Training Loss Curve - {config_name}")
    plt.xlabel("Optimization Step")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    logging.info(f"Loss curve saved to {plot_path}")
