import os

import tiktoken
import torch

from mugato import train
from mugato.data.four_rooms import create_dataloader as create_four_rooms_dataloader
from mugato.data.shakespeare import create_dataloader as create_shakespeare_dataloader
from mugato.tokenizer import Tokenizer
from mugato.utils import data_home, select_device


def test_trainer_runs_and_outputs():
    # Use minimal config for rapid test
    config = train.MugatoConfig(
        device=select_device(),
        n_embd=128,
        block_size=1024,
        vocab_size=51281,  # Correct vocab size: text vocab (50257) + discrete vocab (1024)
        out_dir=os.path.join(data_home, "out", "test_trainer"),  # Use subdirectory for test
    )
    # Minimal transformer overrides
    transformer_overrides = {
        "n_layer": 2,
        "n_head": 8,
        "n_embd": 128,
        "block_size": 1024,
        "vocab_size": 51281,  # Must match the MugatoConfig vocab_size
        "dropout": 0.0,
        "bias": False,
    }
    # Construct a real Tokenizer using tiktoken
    text_tokenizer = tiktoken.get_encoding("r50k_base")
    tokenizer = Tokenizer(text_tokenizer)

    # Additional trainer args for rapid test
    # Let it use the default combined dataloader which now properly handles mask_keys
    trainer = train.Trainer(
        config,
        tokenizer,
        batch_size=4,
        max_iters=20,
        eval_iters=2,
        eval_interval=2,
        compile_model=False,
        config_overrides=transformer_overrides,
        gradient_accumulation_steps=1,  # Use 1 for faster testing
    )
    metrics = trainer.train()

    # Check that loss plot exists
    assert len(trainer.losses) == trainer.max_iters + 1, f"Expected {trainer.max_iters + 1} losses (including iter 0), got {len(trainer.losses)}"
    assert all(not torch.isnan(torch.tensor(loss)) for loss in trainer.losses), "Found NaN losses"

    loss_plot = os.path.join(config.out_dir, "loss.png")
    assert os.path.exists(loss_plot), f"Loss plot was not created at {loss_plot}"

    # Check that metrics are present and reasonable
    assert "num_parameters" in metrics
    assert "model_size_bytes" in metrics
    assert "time_per_iter" in metrics
    assert "final_loss" in metrics
    assert "avg_loss" in metrics
    assert metrics["num_parameters"] > 0
    assert metrics["model_size_bytes"] > 0
    assert metrics["time_per_iter"] > 0
    assert not torch.isnan(torch.tensor(metrics["final_loss"])), "Final loss is NaN"
    assert not torch.isnan(torch.tensor(metrics["avg_loss"])), "Average loss is NaN"

def test_shakespeare_training():
    """Test training on Shakespeare dataset"""
    config = train.MugatoConfig(
        device=select_device(),
        n_embd=128,
        block_size=1024,  # Increased to handle longer Shakespeare samples
        vocab_size=51281,
        out_dir=os.path.join(data_home, "out", "test_shakespeare"),
    )

    transformer_overrides = {
        "n_layer": 2,
        "n_head": 8,
        "n_embd": 128,
        "block_size": 1024,  # Must match config
        "vocab_size": 51281,
        "dropout": 0.0,
        "bias": False,
    }

    # Construct tokenizer
    text_tokenizer = tiktoken.get_encoding("r50k_base")
    tokenizer = Tokenizer(text_tokenizer)

    # Create Shakespeare dataloader
    batch_size = 4
    dataloader = create_shakespeare_dataloader(
        tokenizer,
        batch_size=batch_size,
        split="train",
        block_size=config.block_size
    )

    # Examine a batch
    print("\n=== Examining Shakespeare batch ===")
    batch = next(iter(dataloader))
    xs, ys, ms = batch

    print(f"\nBatch info:")
    print(f"  xs keys: {list(xs.keys())}")
    print(f"  ys keys: {list(ys.keys())}")
    print(f"  ms keys: {list(ms.keys())}")

    # Shakespeare has only 'text' key
    print(f"\nText data shapes:")
    print(f"  xs['text'] shape: {xs['text'].shape}")
    print(f"  ys['text'] shape: {ys['text'].shape}")
    print(f"  ms['text'] shape: {ms['text'].shape}")
    print(f"  mask sum: {ms['text'].sum()}")
    print(f"  mask mean: {ms['text'].float().mean():.3f}")

    # Show some actual text tokens
    print(f"\nSample text tokens (first 20):")
    print(f"  xs: {xs['text'][0, 0, :20]}")
    print(f"  ys: {ys['text'][0, 0, :20]}")

    # Run training
    trainer = train.Trainer(
        config,
        tokenizer,
        batch_size=batch_size,
        max_iters=5,
        compile_model=False,
        config_overrides=transformer_overrides,
        gradient_accumulation_steps=1,  # Use 1 for faster testing
    )

    metrics = trainer.train()

    print(f"\nFinal metrics: {metrics}")
    assert len(trainer.losses) > 0, "No losses recorded"
    assert metrics["avg_loss"] > 0, "Average loss should be positive"
