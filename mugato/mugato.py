import math
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from timm.models.resnetv2 import ResNetV2
import tiktoken

from mugato.nano_gpt import GPTConfig
from mugato.tokenizer import Tokenizer
from mugato.utils import select_device

from typing import Callable, Optional


@dataclass
class Embedder:
    lookup_embedding: Callable
    image_embedding: Callable

    def embed(self, data):
        """Determines modality of data and returns appropriate embedding.

        The size of the lookup embedding table is the combined size of
        the text embedding table and the discrete embedding table. The paper
        chooses an ~arbitrary number of discrete embeddings to support, 1024,
        and those get tokenized to be in the range [text_vocab_size, text_vocab_size+1024).
        """
        B, E, T, C = data.shape
        n_embd = self.lookup_embedding.weight.size(-1)
        if (
            data.size(-1) > 1
        ):  # Images are the only modality that have a channel dim > 1.
            #                                           (C,  P,  P)
            return self.image_embedding(data.view(B * E * T, 3, 16, 16)).view(
                B, E, T, n_embd
            )
        else:
            # Zero grad dummy pass for image params
            dummy = sum(p.sum() * 0 for p in self.image_embedding.parameters())
            return (
                self.lookup_embedding(data.view(B * E * T)).view(B, E, T, n_embd)
                + dummy
            )


def sequence(embedder, xs, ys=None, ms=None, sequence_length=1024, pad=True):
    embeddings = torch.concat([embedder.embed(v) for k, v in xs.items()], dim=2)
    B, E, T, C = embeddings.shape
    embeddings = embeddings.view(B, E * T, C)
    if ys is not None:
        targets = torch.concat([v for _, v in ys.items()], dim=2)
        masks = torch.concat([v for _, v in ms.items()], dim=2)
        targets = targets.view(B, E * T)
        masks = masks.view(B, E * T)
        if pad:
            return (
                F.pad(
                    embeddings,
                    (0, 0, 0, sequence_length - embeddings.size(1), 0, 0),
                    value=0,
                ),
                F.pad(
                    targets, (0, sequence_length - embeddings.size(1), 0, 0), value=0
                ).to(torch.long),
                F.pad(masks, (0, sequence_length - embeddings.size(1), 0, 0), value=0),
            )
        else:
            return embeddings, targets, masks
    else:
        if pad:
            return F.pad(
                embeddings,
                (0, 0, 0, sequence_length - embeddings.size(1), 0, 0),
                value=0,
            )
        else:
            return embeddings


@dataclass
class MugatoConfig:
    device: str = select_device()
    n_embd: int = 512
    block_size: int = 1024
    vocab_size: int = 51281  # text vocab + discrete vocab


def init_default_config(transformer_model_args: GPTConfig) -> MugatoConfig:
    text_tokenizer = tiktoken.get_encoding("r50k_base")
    tokenizer = Tokenizer(text_tokenizer)
    transformer_config = GPTConfig(**transformer_model_args)
    return MugatoConfig(
        tokenizer=tokenizer,
    )


@dataclass
class TransformerConfig:
    block_size: int = 1024
    vocab_size: int = 50257  # tiktoken.get_encoding("r50k_base").n_vocab
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = (
        True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    )


class Mugato(torch.nn.Module):
    def __init__(
        self,
        tokenizer: Tokenizer,
        sequence_model: nn.Module,
        config: MugatoConfig,
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.device = torch.device(self.config.device)

        # Initialize components
        self.lookup_embedding = torch.nn.Embedding(
            self.config.vocab_size, self.config.n_embd
        ).to(self.device)

        self.image_embedding = ResNetV2(
            layers=[3, 4, 6, 3], num_classes=self.config.n_embd
        ).to(self.device)

        self.embedder = Embedder(self.lookup_embedding, self.image_embedding)
        # TODO:
        # Since we're doing our own embedding, we need to handle our own
        # position embedding.
        self.transformer = sequence_model  # TODO: rename to sequence_model?
        self.lm_head = torch.nn.Linear(self.config.n_embd, self.config.vocab_size).to(
            self.device
        )

    def forward(self, xs, ys=None, ms=None, pad=True, sequence: Callable = sequence):
        if ys is not None:
            tok_emb, ys, ms = sequence(
                self.embedder,
                xs,
                ys,
                ms,
                pad=pad,
                sequence_length=self.config.block_size,
            )
            b, t, c = tok_emb.size()
            pos = torch.arange(0, t, dtype=torch.long, device=self.device)  # shape (t)
            pos_emb = self.transformer.wpe(
                pos
            )  # position embeddings of shape (t, n_embd)
            xs = self.transformer.drop(tok_emb + pos_emb)
            for block in self.transformer.h:
                xs = block(xs)
            logits = self.lm_head(xs)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), ys.view(-1), reduction="none"
            )
            loss = loss * ms.view(-1)
            loss = loss.sum() / ms.sum()
        else:
            tok_emb = sequence(self.embedder, xs, pad=pad)
            b, t, c = tok_emb.size()
            pos = torch.arange(0, t, dtype=torch.long, device=self.device)  # shape (t)
            pos_emb = self.transformer.wpe(
                pos
            )  # position embeddings of shape (t, n_embd)
            xs = self.transformer.drop(tok_emb + pos_emb)
            for block in self.transformer.h:
                xs = block(xs)
            logits = self.lm_head(xs)
            loss = None
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS

        Estimate written for transformer as sequence model.
        """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        # TODO: Fix hardcoding this.
        # L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        L, H, Q, T = 6, 4, 128, 768
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        # flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        flops_promised = 22e12  # RTX 4060 Ti peak flops is 22 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu
