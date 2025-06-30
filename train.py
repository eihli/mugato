"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)

To run with MPU on MacOS:
$ python train.py --device=mps

Credit to Andrej Karpathy. This code is adapted from [nanoGPT](https://github.com/karpathy/nanoGPT).
"""

import os
from datetime import datetime

import tiktoken
import torch
from torch.distributed import destroy_process_group, init_process_group

from mugato.mugato import MugatoConfig
from mugato.tokenizer import Tokenizer
from mugato.train import Trainer
from mugato.utils import data_home, select_device

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = data_home / "out"
eval_interval = 100
log_interval = 1
eval_iters = 6
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
init_from = "scratch"  # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False  # disabled by default
wandb_project = "mugato"
wandb_run_name = f"alpha-{datetime.now().isoformat()[:-7]}"
# data
gradient_accumulation_steps = 5 * 8  # used to simulate larger batch sizes
batch_size = 6  # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 6
n_head = 4
n_embd = 512
dropout = 0.0
bias = False
# optimizer
learning_rate = 6e-4  # max learning rate
max_iters = 6000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 100  # how many steps to warm up for
lr_decay_iters = max_iters  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = "nccl"  # 'nccl', 'gloo', etc.
# system
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False  # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
device = str(select_device())  # See configurator.py as to why this is a string.
exec(open("configurator.py").read())  # overrides from command line or config file
device = torch.device(device)
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------

def main():
    """Main entry point that handles DDP setup and training"""
    global gradient_accumulation_steps, device

    # DDP setup
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"{device}:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
        # Scale gradient accumulation by world size
        gradient_accumulation_steps = gradient_accumulation_steps // ddp_world_size
    else:
        # Single GPU
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        ddp_rank = 0
        ddp_local_rank = 0

    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    if master_process:
        os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Setup device type
    device_type = (
        "cuda" if "cuda" in str(device)
        else "mps" if "mps" in str(device)
        else "cpu"
    )

    # Create tokenizer
    text_tokenizer = tiktoken.get_encoding("r50k_base")
    tokenizer = Tokenizer(text_tokenizer)

    # Create Mugato config
    mugato_config = MugatoConfig(
        device=str(device),
        n_embd=n_embd,
        block_size=block_size,
        vocab_size=50257 + 1024,  # text + discrete tokens
        out_dir=str(out_dir),
    )

    # Config overrides for transformer
    config_overrides = {
        "n_layer": n_layer,
        "n_head": n_head,
        "n_embd": n_embd,
        "block_size": block_size,
        "dropout": dropout,
        "bias": bias,
        "vocab_size": mugato_config.vocab_size,
    }

    # Create trainer
    trainer = Trainer(
        config=mugato_config,
        tokenizer=tokenizer,
        # Training params
        batch_size=batch_size,
        max_iters=max_iters,
        eval_interval=eval_interval,
        eval_iters=eval_iters,
        log_interval=log_interval,
        # Optimizer params
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        beta1=beta1,
        beta2=beta2,
        grad_clip=grad_clip,
        # LR decay params
        decay_lr=decay_lr,
        warmup_iters=warmup_iters,
        lr_decay_iters=lr_decay_iters,
        min_lr=min_lr,
        # System params
        gradient_accumulation_steps=gradient_accumulation_steps,
        dtype=dtype,
        compile_model=compile,
        # Checkpointing
        out_dir=str(out_dir),
        always_save_checkpoint=always_save_checkpoint,
        init_from=init_from,
        # Logging
        wandb_log=wandb_log,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        # Config overrides
        config_overrides=config_overrides,
        # DDP
        ddp=ddp,
        ddp_rank=ddp_rank,
        ddp_local_rank=ddp_local_rank,
        ddp_world_size=ddp_world_size,
        device_type=device_type,
        master_process=master_process,
    )

    # Run training
    results = trainer.train(eval_only=eval_only)
    print(f"Training complete. Final results: {results}")

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
