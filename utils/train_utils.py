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
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler
from transformers import get_cosine_schedule_with_warmup


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
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
            return nullcontext()
    return nullcontext()


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

    # Calculate total training steps
    grad_accum_steps = config["training"]["gradient_accumulation_steps"]
    steps_per_epoch = math.ceil(dataloader_len / grad_accum_steps)
    total_steps = steps_per_epoch * epochs

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["training"]["warmup_steps"],
        num_training_steps=total_steps,
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
    dtype = torch.float32
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
        dtype = torch.float16

    return dtype, scaler, use_amp


def save_checkpoint(
    state: Dict[str, Any], training_dir: Path, global_step: int
) -> None:
    """
    Persists a training checkpoint to disk.

    Args:
        state: Dictionary containing training state (model, optimizer, etc.).
        training_dir: Path to the training artifacts directory.
        global_step: Current global optimization step.
    """
    checkpoint_path = training_dir / f"checkpoint-{global_step}.pt"
    torch.save(state, checkpoint_path)


def save_final_artifacts(
    model: torch.nn.Module,
    config: Dict[str, Any],
    models_dir: Path,
) -> None:
    """
    Saves the final model state and configuration files.

    Args:
        model: The trained model.
        config: Training configuration dictionary.
        models_dir: Path to the models directory.
    """
    # Save standard PyTorch state dict
    torch.save(model.state_dict(), models_dir / "fin_ffb_final.pt")

    # Save config for reproducibility
    with open(models_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    # Attempt HuggingFace-style save
    try:
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(models_dir)
    except Exception as e:
        print(f"Non-critical: HF save_pretrained skipped: {e}")


def log_training_results(
    results: Dict[str, Any], training_dir: Path
) -> None:
    """
    Saves a comprehensive JSON log of the training session.

    Args:
        results: Dictionary containing training metrics and hardware info.
        training_dir: Path to the training artifacts directory.
    """
    log_path = training_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(results, f, indent=4)


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
    return [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
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
