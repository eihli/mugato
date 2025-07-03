# Trainer class and minimal API for test-driven refactor
import os
import time
from contextlib import nullcontext
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from mugato.config import MugatoConfig, TransformerConfig
from mugato.data.utils import create_combined_dataloader_from_module
from mugato.models.transformer import crop_block_size, get_learning_rate, init_model
from mugato.mugato import Mugato
from mugato.nano_gpt import GPT


class Trainer:
    """Trainer class for μGATO with full training capabilities"""

    def __init__(
        self,
        config: MugatoConfig,
        tokenizer: Any,
        # Training params
        batch_size: int = 6,
        max_iters: int = 6000,
        eval_interval: int = 100,
        eval_iters: int = 6,
        log_interval: int = 1,
        # Optimizer params
        learning_rate: float = 6e-4,
        weight_decay: float = 1e-1,
        beta1: float = 0.9,
        beta2: float = 0.95,
        grad_clip: float = 1.0,
        # LR decay params
        decay_lr: bool = True,
        warmup_iters: int = 100,
        lr_decay_iters: int = 6000,
        min_lr: float = 6e-5,
        # System params
        gradient_accumulation_steps: int = 40,
        dtype: str = "float16",
        compile_model: bool = False,
        # Checkpointing
        out_dir: str | None = None,
        always_save_checkpoint: bool = True,
        init_from: str = "scratch",
        resume_path: str | None = None,
        # Logging
        wandb_log: bool = False,
        wandb_project: str = "mugato",
        wandb_run_name: str | None = None,
        # Config overrides
        config_overrides: dict[str, Any] | None = None,
        # DDP
        ddp: bool = False,
        ddp_rank: int = 0,
        ddp_local_rank: int = 0,
        ddp_world_size: int = 1,
        device_type: str | None = None,
        master_process: bool = True,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.device = torch.device(config.device)

        # Model will be initialized later - can be various types due to compilation/DDP
        self.model: Mugato | GPT | torch.nn.Module | Any
        self.unoptimized_model: Mugato | GPT | torch.nn.Module | Any | None = None

        # Detect device type if not provided
        if device_type is None:
            device_type = (
                "cuda" if "cuda" in str(self.device)
                else "mps" if "mps" in str(self.device)
                else "cpu"
            )
        self.device_type = device_type

        # Training params
        self.batch_size = batch_size
        self.max_iters = max_iters
        self.eval_interval = eval_interval
        self.eval_iters = eval_iters
        self.log_interval = log_interval

        # Optimizer params
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.grad_clip = grad_clip

        # LR decay params
        self.decay_lr = decay_lr
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.min_lr = min_lr

        # System params
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.dtype = dtype
        self.compile_model = compile_model

        # Checkpointing
        self.out_dir = out_dir or getattr(config, 'out_dir', './out')
        self.always_save_checkpoint = always_save_checkpoint
        self.init_from = init_from
        self.resume_path = resume_path

        # Logging
        self.wandb_log = wandb_log
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name

        # DDP
        self.ddp = ddp
        self.ddp_rank = ddp_rank
        self.ddp_local_rank = ddp_local_rank
        self.ddp_world_size = ddp_world_size
        self.master_process = master_process

        # Config overrides
        self.config_overrides = config_overrides or {}

        # Initialize model
        self.iter_num = 0
        self.best_val_loss = 1e9
        self._init_model()

        # Initialize dataloaders
        self._init_dataloaders()

        # Setup training context
        self._setup_training_context()

        # Initialize optimizer
        self._init_optimizer()

        # Compile model if requested
        if self.compile_model:
            self._compile()

        # Wrap in DDP if needed
        if self.ddp:
            self._wrap_ddp()

        # Setup wandb if requested
        if self.wandb_log and self.master_process:
            self._init_wandb()

        # Tracking
        self.losses: list[float] = []
        self.metrics: dict[str, Any] = {}
        self.running_mfu = -1.0

    def _init_model(self) -> None:
        """Initialize the model based on init_from strategy"""
        # Create transformer config with overrides
        transformer_config = TransformerConfig()
        if self.config_overrides:
            for key, value in self.config_overrides.items():
                if hasattr(transformer_config, key):
                    setattr(transformer_config, key, value)

        # Ensure configs are compatible
        self.config.n_embd = transformer_config.n_embd
        self.config.block_size = transformer_config.block_size

        if self.init_from == "resume":
            default_out_dir = self.out_dir or "/tmp/mugato"
            checkpoint_path = self.resume_path or os.path.join(
                default_out_dir, f"{datetime.now().strftime('%Y-%m-%d')}-ckpt.pt"
            )
            self.model, self.model_args, self.iter_num, self.best_val_loss = init_model(
                self.tokenizer,
                transformer_config,
                self.config,
                init_from=self.init_from,
                resume_path=checkpoint_path,
                device=str(self.device)
            )
        else:
            self.model, self.model_args, self.iter_num, self.best_val_loss = init_model(
                self.tokenizer,
                transformer_config,
                self.config,
                init_from=self.init_from,
                device=str(self.device)
            )

        # Crop block size if needed
        if self.config.block_size < self.model.config.block_size:
            self.model, self.model_args = crop_block_size(
                self.model, self.config.block_size, self.model_args
            )

        self.model.to(self.device)

    def _init_dataloaders(self) -> None:
        """Initialize train/val/test dataloaders"""
        self.train_dataloader = iter(
            create_combined_dataloader_from_module(
                self.tokenizer,
                self.batch_size,
                split="train",
                block_size=self.config.block_size
            )
        )
        self.val_dataloader = iter(
            create_combined_dataloader_from_module(
                self.tokenizer,
                self.batch_size,
                split="val",
                block_size=self.config.block_size
            )
        )
        self.test_dataloader = iter(
            create_combined_dataloader_from_module(
                self.tokenizer,
                self.batch_size,
                split="test",
                block_size=self.config.block_size
            )
        )

    def _setup_training_context(self) -> None:
        """Setup autocast context and grad scaler"""
        ptdtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[self.dtype]

        self.ctx = (
            nullcontext()
            if self.device_type in ["cpu", "mps"]
            else torch.amp.autocast(device_type=self.device_type, dtype=ptdtype)
        )

        # Initialize GradScaler for float16
        self.scaler = torch.amp.GradScaler(enabled=(self.dtype == "float16"))

    def _init_optimizer(self) -> None:
        """Initialize the optimizer"""
        # Only some models have configure_optimizers method
        if hasattr(self.model, 'configure_optimizers'):
            self.optimizer = self.model.configure_optimizers(  # type: ignore
                self.weight_decay,
                self.learning_rate,
                (self.beta1, self.beta2),
                self.device_type
            )
        else:
            # Fallback optimizer configuration
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                betas=(self.beta1, self.beta2),
                weight_decay=self.weight_decay
            )

        if self.init_from == "resume" and self.resume_path:
            checkpoint = torch.load(self.resume_path, map_location=self.device)
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            del checkpoint  # free up memory

    def _compile(self) -> None:
        """Compile the model with PyTorch 2.0"""
        print("compiling the model... (takes a ~minute)")
        torch._dynamo.config.optimize_ddp = False
        self.unoptimized_model = self.model
        self.model = torch.compile(self.model)

    def _wrap_ddp(self) -> None:
        """Wrap model in DistributedDataParallel"""
        self.model = DDP(self.model, device_ids=[self.ddp_local_rank])

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases logging"""
        import wandb
        wandb.init(
            project=self.wandb_project,
            name=self.wandb_run_name,
            config=self.config_overrides
        )

    def get_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a batch of data from the specified split"""
        if split == "train":
            X, Y, M = next(next(self.train_dataloader))
        elif split == "val":
            X, Y, M = next(next(self.val_dataloader))
        elif split == "test":
            X, Y, M = next(next(self.test_dataloader))
        X, Y, M = X.to(self.device), Y.to(self.device), M.to(self.device)
        return X, Y, M

    @torch.no_grad()
    def estimate_loss(self) -> dict[str, float]:
        """Estimate loss over train and validation sets"""
        out = {}
        self.model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(self.eval_iters)
            for k in tqdm(range(self.eval_iters), desc=f"Evaluating {split}"):
                X, Y, M = self.get_batch(split)
                with self.ctx:
                    logits, loss = self.model(X, Y, M)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        self.model.train()
        return out

    def save_checkpoint(self, losses: dict[str, float]) -> None:
        """Save a checkpoint"""
        if self.ddp:
            raw_model = self.model.module  # type: ignore
        else:
            raw_model = self.model

        checkpoint = {
            "model": raw_model.state_dict(),  # type: ignore
            "optimizer": self.optimizer.state_dict(),
            "model_args": self.model_args,
            "iter_num": self.iter_num,
            "best_val_loss": self.best_val_loss,
            "config": self.config_overrides,
        }
        default_out_dir = self.out_dir or "/tmp/mugato"
        print(f"saving checkpoint to {default_out_dir}")
        os.makedirs(default_out_dir, exist_ok=True)
        torch.save(checkpoint, os.path.join(
            default_out_dir,
            f"{datetime.now().strftime('%Y-%m-%d')}-ckpt.pt"),
        )

    def train(self, eval_only: bool = False) -> dict[str, Any]:
        """Main training loop with all features"""
        # Get initial batch
        X, Y, M = self.get_batch("train")
        t0 = time.time()
        local_iter_num = 0

        if self.ddp:
            raw_model = self.model.module  # type: ignore
        else:
            raw_model = self.model

        while True:
            # Set learning rate for this iteration
            lr = get_learning_rate(
                self.iter_num, self.learning_rate, self.warmup_iters,
                self.lr_decay_iters, self.min_lr, self.decay_lr
            )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # Evaluate and save checkpoints
            if self.iter_num % self.eval_interval == 0 and self.master_process:
                losses = self.estimate_loss()
                print(
                    f"step {self.iter_num}: train loss {losses['train']:.4f}, "
                    f"val loss {losses['val']:.4f}"
                )

                if self.wandb_log:
                    import wandb
                    wandb.log({
                        "iter": self.iter_num,
                        "train/loss": losses["train"],
                        "val/loss": losses["val"],
                        "lr": lr,
                        "mfu": self.running_mfu * 100,
                    })

                if losses["val"] < self.best_val_loss or self.always_save_checkpoint:
                    self.best_val_loss = losses["val"]
                    if self.iter_num > 0:
                        self.save_checkpoint(losses)

            if self.iter_num == 0 and eval_only:
                break

            # Forward backward update with gradient accumulation
            for micro_step in range(self.gradient_accumulation_steps):
                if self.ddp:
                    # Sync gradients only at the last micro step
                    self.model.require_backward_grad_sync = (
                        micro_step == self.gradient_accumulation_steps - 1
                    )
                with self.ctx:
                    logits, loss = self.model(X, Y, M)
                    loss = loss / self.gradient_accumulation_steps

                # Get next batch while GPU is working
                X, Y, M = self.get_batch("train")

                # Backward pass
                self.scaler.scale(loss).backward()

            # Clip gradients
            if self.grad_clip != 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            # Step optimizer
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            # Timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            if self.iter_num % self.log_interval == 0 and self.master_process:
                lossf = loss.item() * self.gradient_accumulation_steps
                self.losses.append(lossf)  # Track losses for plotting
                if local_iter_num >= 5:  # Let training settle
                    # Only GPT models have estimate_mfu method
                    if hasattr(raw_model, 'estimate_mfu'):
                        mfu = raw_model.estimate_mfu(  # type: ignore
                            self.batch_size * self.gradient_accumulation_steps, dt
                        )
                    else:
                        mfu = -1.0  # Fallback for models without MFU estimation
                    self.running_mfu = (
                        mfu if self.running_mfu == -1.0
                        else 0.9 * self.running_mfu + 0.1 * mfu
                    )
                print(
                    f"iter {self.iter_num}: loss {lossf:.4f}, "
                    f"time {dt*1000:.2f}ms, mfu {self.running_mfu*100:.2f}%"
                )

            self.iter_num += 1
            local_iter_num += 1

            # Check termination
            if self.iter_num > self.max_iters:
                break

        # Plot loss if we have losses
        if self.losses and self.master_process:
            default_out_dir = self.out_dir or "/tmp/mugato"
            os.makedirs(default_out_dir, exist_ok=True)
            plt.figure(figsize=(10, 6))
            plt.plot(self.losses)
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.title("Training Loss")
            plt.grid(True)
            out_path = os.path.join(default_out_dir, "loss.png")
            plt.savefig(out_path)
            plt.close()

        # Calculate metrics
        num_parameters = sum(p.numel() for p in self.model.parameters())
        model_size_bytes = sum(
            p.element_size() * p.nelement() for p in self.model.parameters()
        )

        return {
            "final_loss": lossf if 'lossf' in locals() else None,
            "best_val_loss": self.best_val_loss,
            "final_iter": self.iter_num,
            "num_parameters": num_parameters,
            "model_size_bytes": model_size_bytes,
            "time_per_iter": (time.time() - t0) / max(1, local_iter_num),
            "avg_loss": (
                sum(self.losses) / len(self.losses) if self.losses else float('nan')
            ),
        }
