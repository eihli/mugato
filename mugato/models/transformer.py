"""
Transformer model implementation for Mugato.

This module contains the transformer-specific code that works with the Mugato
architecture.
It provides functions to initialize and configure a transformer model that can be used
as the sequence model within Mugato.
"""
import math
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

from mugato.mugato import Mugato, MugatoConfig, TransformerConfig
from mugato.nano_gpt import GPT, Block

# Default transformer configuration
block_size = 768
n_layer = 6
n_head = 4
n_embd = 512
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?


transformer_model_args = {
    "n_layer": n_layer,
    "n_head": n_head,
    "n_embd": n_embd,
    "block_size": block_size,
    "bias": bias,
    "vocab_size": 50257,  # tiktoken.get_encoding("r50k_base").n_vocab
    "dropout": dropout,
}

mugato_model_args = {
    "n_embd": n_embd,
    "block_size": block_size,
    "vocab_size": 51281,  # text vocab + discrete vocab
}


def _as_abspath(path: str | None) -> Path | None:
    """Convert a string path to an absolute Path object if it's not None."""
    if path is not None and os.path.isabs(path):
        return Path(path)
    return None


def create_transformer(config: TransformerConfig) -> nn.ModuleDict:
    """Create a transformer model using the given configuration."""
    return nn.ModuleDict({
        "wpe": nn.Embedding(config.block_size, config.n_embd),
        "drop": nn.Dropout(config.dropout),
        "h": nn.ModuleList(
            [Block(config) for _ in range(config.n_layer)]
        ),
    })


def init_from_scratch(
    tokenizer, config_overrides: dict[str, Any] | None = None
) -> tuple[Mugato, TransformerConfig, MugatoConfig]:
    """Initialize a new model from scratch with optional configuration overrides."""
    # Apply any overrides to the default configuration
    model_args = transformer_model_args.copy()
    mugato_args = mugato_model_args.copy()

    if config_overrides:
        for k, v in config_overrides.items():
            if k in model_args:
                model_args[k] = v
            if k in mugato_args and k in ("n_embd", "block_size", "vocab_size"):
                mugato_args[k] = v

    # Create configurations
    transformer_config = TransformerConfig(**model_args)
    mugato_config = MugatoConfig(**mugato_args)

    # Create transformer and model
    transformer = create_transformer(transformer_config)
    model = Mugato(tokenizer, transformer, mugato_config)

    return model, transformer_config, mugato_config


def init_from_resume(
    tokenizer, checkpoint_path: str, device: str = "cpu"
) -> tuple[Mugato, dict[str, Any], int, float]:
    """Resume training from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    checkpoint_model_args = checkpoint["model_args"]

    # Force these config attributes to be equal otherwise we can't resume training
    model_args = transformer_model_args.copy()
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = checkpoint_model_args[k]

    # Create the model
    transformer_config = TransformerConfig(**model_args)
    transformer = create_transformer(transformer_config)
    mugato_config = MugatoConfig(**mugato_model_args)
    model = Mugato(tokenizer, transformer, mugato_config)

    # Load the state dict
    state_dict = checkpoint["model"]
    # Fix the keys of the state dictionary if needed
    unwanted_prefix = "_orig_mod."
    for k, _ in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)

    # Return additional checkpoint info
    iter_num = checkpoint.get("iter_num", 0)
    best_val_loss = checkpoint.get("best_val_loss", float('inf'))

    return model, checkpoint, iter_num, best_val_loss


def init_from_gpt2(
    tokenizer, model_type: str, override_args: dict[str, Any] | None = None
) -> tuple[GPT, dict[str, Any]]:
    """Initialize from OpenAI GPT-2 weights."""
    override_args = override_args or {}
    model = GPT.from_pretrained(model_type, override_args)

    # Read off the created config params to store in checkpoint correctly
    updated_model_args = transformer_model_args.copy()
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        updated_model_args[k] = getattr(model.config, k)

    return model, updated_model_args


def init_model(
    tokenizer,
    init_from: str = "scratch",
    resume_path: str | None = None,
    gpt2_model_type: str | None = None,
    config_overrides: dict[str, Any] | None = None,
    device: str = "cpu"
) -> tuple[Mugato | GPT, dict[str, Any], int, float]:
    """
    Initialize a model based on the specified initialization strategy.

    Args:
        tokenizer: The tokenizer to use
        init_from: One of "scratch", "resume", or "gpt2"
        resume_path: Path to checkpoint when init_from is "resume"
        gpt2_model_type: GPT-2 model type when init_from is "gpt2"
        config_overrides: Optional configuration overrides
        device: Device to load the model on

    Returns:
        Tuple of (model, model_args, iter_num, best_val_loss)
    """
    iter_num = 0
    best_val_loss = float('inf')

    if init_from == "scratch":
        model, _, _ = init_from_scratch(tokenizer, config_overrides)
        return model, transformer_model_args, iter_num, best_val_loss

    elif init_from == "resume":
        if not resume_path:
            raise ValueError("resume_path must be provided when init_from is 'resume'")
        model, checkpoint, iter_num, best_val_loss = init_from_resume(
            tokenizer, resume_path, device
        )
        return model, transformer_model_args, iter_num, best_val_loss

    elif init_from.startswith("gpt2"):
        if not gpt2_model_type:
            # Use init_from as model type if not explicitly provided
            gpt2_model_type = init_from
        model, updated_model_args = init_from_gpt2(
            tokenizer, gpt2_model_type, config_overrides
        )
        return model, updated_model_args, iter_num, best_val_loss

    else:
        raise ValueError(
            f"Unknown init_from value: {init_from}. "
            "Must be 'scratch', 'resume', or start with 'gpt2'"
        )


def crop_block_size(
    model: Mugato | GPT, block_size: int, model_args: dict[str, Any]
) -> tuple[Mugato | GPT, dict[str, Any]]:
    """
    Crop the model's block size if needed.

    Args:
        model: The model to modify
        block_size: The new block size
        model_args: The model arguments dict to update

    Returns:
        Tuple of (modified model, updated model_args)
    """
    if isinstance(model, GPT):
        if block_size < model.config.block_size:
            model.crop_block_size(block_size)
            model_args["block_size"] = block_size
    elif isinstance(model, Mugato):
        if block_size < model.config.block_size:
            model.transformer.wpe.weight = nn.Parameter(
                model.transformer.wpe.weight[:block_size]
            )
            for block in model.transformer.h:
                if hasattr(block.attn, "bias"):
                    block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]
            model.config.block_size = block_size
            model_args["block_size"] = block_size

    return model, model_args


def get_learning_rate(
    iter_num: int,
    learning_rate: float,
    warmup_iters: int,
    lr_decay_iters: int,
    min_lr: float,
    decay_lr: bool = True
) -> float:
    """
    Calculate learning rate based on a schedule with warmup and cosine decay.

    Args:
        iter_num: Current iteration number
        learning_rate: Maximum learning rate
        warmup_iters: Number of warmup iterations
        lr_decay_iters: Number of iterations over which to decay learning rate
        min_lr: Minimum learning rate
        decay_lr: Whether to decay the learning rate

    Returns:
        The learning rate for the current iteration
    """
    if not decay_lr:
        return learning_rate

    # Linear warmup for warmup_iters steps
    if iter_num < warmup_iters:
        return learning_rate * iter_num / warmup_iters

    # If it > lr_decay_iters, return min learning rate
    if iter_num > lr_decay_iters:
        return min_lr

    # In between, use cosine decay down to min learning rate
    decay_ratio = (iter_num - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)
