# Trainer class and minimal API for test-driven refactor
import os
import time

import matplotlib.pyplot as plt
import torch

from mugato.data.utils import create_combined_dataloader_from_module
from mugato.models.transformer import init_model
from mugato.mugato import MugatoConfig


class Trainer:
    def __init__(
        self,
        config: MugatoConfig,
        tokenizer,
        dataloader=None,  # Allow passing custom dataloader
        batch_size: int = 2,
        max_iters: int = 5,
        eval_iters: int = 2,
        eval_interval: int = 2,
        out_dir = None,
        compile: bool = False,
        config_overrides: dict = None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.device = torch.device(config.device)
        self.batch_size = batch_size
        self.max_iters = max_iters
        self.eval_iters = eval_iters
        self.eval_interval = eval_interval
        self.out_dir = out_dir or getattr(config, 'out_dir', './out')
        self.compile = compile
        self.config_overrides = config_overrides or {}

        # Pass the actual tokenizer to model and dataloader
        # init_model returns (model, model_args, iter_num, best_val_loss)
        self.model, self.model_args, _, _ = init_model(
            tokenizer=self.tokenizer,
            init_from="scratch",
            config_overrides=self.config_overrides,
            device=str(self.device),  # init_model expects string device
        )
        self.model.to(self.device)

        # Initialize weights if needed (helps with NaN issues)
        for m in self.model.modules():
            if isinstance(m, (torch.nn.Linear, torch.nn.Embedding)):
                if hasattr(m, 'weight') and m.weight is not None:
                    torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        # Use very low learning rate for stability during testing
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5, weight_decay=0.01)

        # Use provided dataloader or create default one
        if dataloader is None:
            self.dataloader = create_combined_dataloader_from_module(
                self.tokenizer, self.batch_size, split="train", block_size=self.config.block_size
            )
        else:
            self.dataloader = dataloader

        self.losses = []
        self.metrics = {}

    def train(self):
        start_time = time.time()

        # Handle both cycling and non-cycling dataloaders
        if hasattr(self.dataloader, '__next__'):
            # It's a cycling dataloader, get the next one
            dataloader = next(self.dataloader)
        else:
            # It's a regular dataloader
            dataloader = self.dataloader

        # Put model in training mode
        self.model.train()

        for iter_num, (xs, ys, ms) in enumerate(dataloader):
            if iter_num >= self.max_iters:
                break
            xs, ys, ms = xs.to(self.device), ys.to(self.device), ms.to(self.device)

            # Forward pass - Mugato expects (xs, ys, ms)
            try:
                logits, loss = self.model(xs, ys, ms)
            except Exception as e:
                print(f"Error during forward pass at iteration {iter_num}: {e}")
                print(f"xs keys: {xs.keys()}")
                for k, v in xs.items():
                    print(f"  xs['{k}'] shape: {v.shape}")
                raise

            # Check for NaN in logits or loss
            if torch.isnan(logits).any():
                print(f"Warning: NaN in logits at iteration {iter_num}")
                print(f"Logits stats: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}")

            if torch.isnan(loss):
                print(f"Warning: NaN loss at iteration {iter_num}")
                # Debug info
                print(f"xs shape: {list(xs[k].shape for k in xs.keys())}")
                print(f"ys shape: {list(ys[k].shape for k in ys.keys())}")
                print(f"ms shape: {list(ms[k].shape for k in ms.keys())}")
                print(f"logits shape: {logits.shape}")
                # Additional debug for embeddings
                for k, v in xs.items():
                    if v.size(-1) > 1:
                        print(f"  {k} has channel dim {v.size(-1)} > 1 (will use image embedding)")
                # Check target values
                print("Target (ys) stats:")
                for k, v in ys.items():
                    print(f"  {k}: min={v.min()}, max={v.max()}, unique values={v.unique().shape[0]}")
                # Check mask
                print("Mask (ms) stats:")
                for k, v in ms.items():
                    print(f"  {k}: sum={v.sum()}, shape={v.shape}")
                print(f"  Total mask sum: {sum(v.sum() for v in ms.values())}")

            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            self.losses.append(loss.item())

        elapsed = time.time() - start_time

        # Metrics
        num_parameters = sum(p.numel() for p in self.model.parameters())
        model_size_bytes = sum(p.element_size() * p.nelement() for p in self.model.parameters())
        time_per_iter = elapsed / max(1, len(self.losses))

        self.metrics = {
            "num_parameters": num_parameters,
            "model_size_bytes": model_size_bytes,
            "time_per_iter": time_per_iter,
            "final_loss": self.losses[-1] if self.losses else float('nan'),
            "avg_loss": sum(self.losses) / len(self.losses) if self.losses else float('nan'),
        }

        # Create output directory if it doesn't exist
        os.makedirs(self.out_dir, exist_ok=True)

        # Plot loss
        if self.losses:
            plt.figure(figsize=(10, 6))
            plt.plot(self.losses)
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.title("Training Loss")
            plt.grid(True)
            out_path = os.path.join(self.out_dir, "loss.png")
            plt.savefig(out_path)
            plt.close()

        return self.metrics
