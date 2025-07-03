"""
Transformer model implementation for Mugato.

There's, unfortunately, a tight coupling between the Mugato architecture and any
sequence model implementation, so in this "transformer"-named module you're
going to see Mugato-specific code that works with a transformer model.

This module contains the transformer-specific code that works with the Mugato
architecture.

It provides functions to initialize and configure a transformer model that can
be used as the sequence model within Mugato.
"""
import logging
import math
import os
from pathlib import Path
from typing import Any

import torch

from mugato.config import (
    ModelArgs,
    MugatoConfig,
    TransformerConfig,
)
from mugato.mugato import Mugato
from mugato.nano_gpt import GPT, GPTConfig
from mugato.tokenizer import Tokenizer

logger = logging.getLogger(__name__)

def _as_abspath(path: str | None) -> Path | None:
    """Convert a string path to an absolute Path object if it's not None."""
    if path is not None and os.path.isabs(path):
        return Path(path)
    return None


def create_transformer(config: TransformerConfig) -> GPT:
    """Create a transformer model using the given configuration."""
    # Convert TransformerConfig to GPTConfig for compatibility
    logger.info("Using GPT2 as sequence model with config: {config:!r}")
    gpt_config = GPTConfig(
        block_size=config.block_size,
        vocab_size=config.vocab_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        dropout=config.dropout,
        bias=config.bias
    )

    return GPT(gpt_config)


def init_from_scratch(
    tokenizer: Tokenizer,
    transformer_config: TransformerConfig,
    mugato_config: MugatoConfig,
) -> Mugato:
    """Initialize a new model from scratch with the given configurations."""
    # Create transformer and model
    transformer = create_transformer(transformer_config)
    model = Mugato(tokenizer, transformer, mugato_config)
    return model


def init_from_resume(
    tokenizer: Any, checkpoint_path: str, device: str = "cpu"
) -> tuple[Mugato, ModelArgs, int, float]:
    """Resume training from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_args = checkpoint["model_args"]

    # Extract configs from ModelArgs
    if isinstance(model_args, dict):
        # Handle dict format for compatibility during transition
        # This is temporary and can be removed once all checkpoints use ModelArgs
        sequence_model_args = TransformerConfig(**model_args)
        mugato_args = MugatoConfig(
            device=device,
            n_embd=sequence_model_args.n_embd,
            block_size=sequence_model_args.block_size,
            vocab_size=51281
        )
        model_args = ModelArgs(
            sequence_model_class_name="GPT",
            sequence_model_args=sequence_model_args,
            mugato_args=mugato_args
        )

    # Update device in mugato config
    model_args.mugato_args.device = device

    # Create the model based on sequence model class
    if model_args.sequence_model_class_name == "GPT":
        transformer = create_transformer(model_args.sequence_model_args)
        model = Mugato(tokenizer, transformer, model_args.mugato_args)
    else:
        raise ValueError(
            f"Unknown sequence model: {model_args.sequence_model_class_name}"
        )

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

    return model, model_args, iter_num, best_val_loss


def init_from_gpt2(
    tokenizer: Any, model_type: str, override_args: dict[str, Any] | None = None
) -> tuple[GPT, TransformerConfig]:
    """Initialize from OpenAI GPT-2 weights."""
    override_args = override_args or {}
    model = GPT.from_pretrained(model_type, override_args)

    # Create TransformerConfig from the loaded model
    transformer_config = TransformerConfig(
        block_size=model.config.block_size,
        vocab_size=model.config.vocab_size,
        n_layer=model.config.n_layer,
        n_head=model.config.n_head,
        n_embd=model.config.n_embd,
        dropout=model.config.dropout,
        bias=model.config.bias
    )

    return model, transformer_config


def init_model(
    tokenizer: Any,
    sequence_config: TransformerConfig,
    model_config: MugatoConfig,
    init_from: str = "scratch",
    resume_path: str | None = None,
    gpt2_model_type: str | None = None,
    device: str = "cpu"
) -> tuple[Mugato, ModelArgs, int, float]:
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
        Tuple of (model, config/model_args, iter_num, best_val_loss)
    """
    iter_num = 0
    best_val_loss = float('inf')

    if init_from == "scratch":
        mugato_model = init_from_scratch(
            tokenizer, sequence_config, model_config
        )
        # Create ModelArgs for serialization
        model_args = ModelArgs(
            sequence_model_class_name="GPT",
            sequence_model_args=sequence_config,
            mugato_args=model_config
        )
        return mugato_model, model_args, 0, float('inf')

    elif init_from == "resume":
        if not resume_path:
            raise ValueError("resume_path must be provided when init_from is 'resume'")
        mugato_model, model_args, iter_num, best_val_loss = init_from_resume(
            tokenizer, resume_path, device
        )
        return mugato_model, model_args, iter_num, best_val_loss

    elif init_from.startswith("gpt2"):
        if not gpt2_model_type:
            # Use init_from as model type if not explicitly provided
            gpt2_model_type = init_from
        gpt_model, transformer_config = init_from_gpt2(
            tokenizer, gpt2_model_type, None
        )
        # Wrap GPT2 in Mugato
        mugato_model = Mugato(tokenizer, gpt_model, model_config)
        model_args = ModelArgs(
            sequence_model_class_name="GPT",
            sequence_model_args=transformer_config,
            mugato_args=model_config
        )
        return mugato_model, model_args, iter_num, best_val_loss

    else:
        raise ValueError(
            f"Unknown init_from value: {init_from}. "
            "Must be 'scratch', 'resume', or start with 'gpt2'"
        )


def crop_block_size(
    model: Mugato | GPT, block_size: int, model_args: ModelArgs
) -> tuple[Mugato | GPT, ModelArgs]:
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
            model_args.sequence_model_args.block_size = block_size
    elif isinstance(model, Mugato):
        if block_size < model.config.block_size:
            # Mugato's sequence_model is a GPT, so delegate to its crop method
            model.sequence_model.crop_block_size(block_size)
            model.config.block_size = block_size
            model_args.sequence_model_args.block_size = block_size
            model_args.mugato_args.block_size = block_size

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
