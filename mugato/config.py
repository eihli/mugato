"""
Unified configuration system for Mugato.

This module provides typed dataclasses for all configuration needs,
eliminating duplicate definitions and ensuring consistency.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from mugato.utils import data_home, select_device


@dataclass
class TransformerConfig:
    """Configuration for the transformer model (GPT)."""
    block_size: int = 1024
    vocab_size: int = 50257  # tiktoken.get_encoding("r50k_base").n_vocab
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for compatibility with existing code."""
        return {
            "block_size": self.block_size,
            "vocab_size": self.vocab_size,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_embd": self.n_embd,
            "dropout": self.dropout,
            "bias": self.bias,
        }

@dataclass
class MugatoConfig:
    """Configuration for the Mugato model wrapper."""
    device: str = field(default_factory=select_device)
    n_embd: int = 512
    block_size: int = 1024
    vocab_size: int = 51281  # text vocab (50257) + discrete vocab (1024)
    out_dir: str = field(default_factory=lambda: os.path.join(data_home, "out"))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for compatibility with existing code."""
        return {
            "device": self.device,
            "n_embd": self.n_embd,
            "block_size": self.block_size,
            "vocab_size": self.vocab_size,
            "out_dir": self.out_dir,
        }


# This is the thing that gets serialized/deserialized
# when saving/loading a model.
@dataclass
class ModelArgs:
    sequence_model_class_name: str
    sequence_model_args: TransformerConfig
    mugato_args: MugatoConfig

@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    # Optimization
    learning_rate: float = 6e-4
    max_iters: int = 600000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # Learning rate schedule
    decay_lr: bool = True
    warmup_iters: int = 2000
    lr_decay_iters: int = 600000
    min_lr: float = 6e-5

    # Batch and sequence
    batch_size: int = 12
    block_size: int = 1024

    # Evaluation
    eval_interval: int = 2000
    eval_iters: int = 200

    # Checkpointing
    always_save_checkpoint: bool = True

    # System
    device: str = field(default_factory=select_device)
    dtype: str = "bfloat16"  # 'float32', 'bfloat16', or 'float16'
    compile: bool = True

    # Initialization
    init_from: str = "scratch"  # 'scratch', 'resume', or 'gpt2-*'


def get_default_transformer_config(size: str = "small") -> TransformerConfig:
    """Get default transformer configuration by size."""
    configs = {
        "small": TransformerConfig(
            n_layer=6,
            n_head=6,
            n_embd=384,
        ),
        "medium": TransformerConfig(
            n_layer=12,
            n_head=12,
            n_embd=768,
        ),
        "large": TransformerConfig(
            n_layer=24,
            n_head=16,
            n_embd=1024,
        ),
    }
    return configs.get(size, configs["small"])


def create_compatible_configs(
    transformer_overrides: dict[str, Any] | None = None,
    mugato_overrides: dict[str, Any] | None = None,
) -> tuple[TransformerConfig, MugatoConfig]:
    """
    Create compatible TransformerConfig and MugatoConfig instances.

    Ensures that shared parameters (n_embd, block_size, vocab_size) are synchronized.
    """
    # Start with defaults
    transformer_config = TransformerConfig()
    mugato_config = MugatoConfig()

    # Apply transformer overrides
    if transformer_overrides:
        for key, value in transformer_overrides.items():
            if hasattr(transformer_config, key):
                setattr(transformer_config, key, value)

    # Apply mugato overrides
    if mugato_overrides:
        for key, value in mugato_overrides.items():
            if hasattr(mugato_config, key):
                setattr(mugato_config, key, value)

    # Sync shared parameters from transformer to mugato
    # (transformer config takes precedence for shared params)
    shared_params = ["n_embd", "block_size"]
    for param in shared_params:
        if hasattr(transformer_config, param):
            setattr(mugato_config, param, getattr(transformer_config, param))

    # Mugato always uses the extended vocabulary
    mugato_config.vocab_size = 51281

    return transformer_config, mugato_config
