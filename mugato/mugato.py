import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, overload

import torch
import torch.nn as nn
from timm.models.resnetv2 import ResNetV2
from torch.nn import functional as F

from mugato.config import MugatoConfig
from mugato.nano_gpt import GPT, GPTConfig
from mugato.tokenizer import Tokenizer


@dataclass
class Embedder:
    lookup_embedding: nn.Embedding
    image_embedding: nn.Module

    def embed(self, data: torch.Tensor) -> torch.Tensor:
        """Determines modality of data and returns appropriate embedding.

        The size of the lookup embedding table is the combined size of
        the text embedding table and the discrete embedding table. The paper
        chooses an ~arbitrary number of discrete embeddings to support, 1024,
        and those get tokenized to be in the range
        [text_vocab_size, text_vocab_size+1024).
        """
        B, E, T, C = data.shape
        n_embd = self.lookup_embedding.weight.size(-1)
        if (
            data.size(-1) > 1
        ):  # Images are the only modality that have a channel dim > 1.
            # Image patches come as flattened features: (B, E, T, 768)
            # where 768 = 3 * 16 * 16 (channels * patch_height * patch_width)
            # Reshape to (B*E*T, 3, 16, 16) for ResNet processing
            images = data.view(B * E * T, 3, 16, 16)
            embeddings: torch.Tensor = self.image_embedding(images)
            return embeddings.view(B, E, T, n_embd)
        else:
            # Zero grad dummy pass for image params
            # Resolves:
            # RuntimeError: Expected to have finished reduction in the prior
            # iteration before starting a new one. This error indicates that
            # your module has parameters that were not used in producing loss.
            dummy = sum(p.sum() * 0 for p in self.image_embedding.parameters())
            lookup_result: torch.Tensor = self.lookup_embedding(
                data.view(B * E * T)
            ).view(B, E, T, n_embd)
            return lookup_result + dummy

def sequence(
    embedder: Embedder,
    xs: dict[str, torch.Tensor],
    ys: dict[str, torch.Tensor] | None = None,
    ms: dict[str, torch.Tensor] | None = None,
    sequence_length: int = 1024,
    pad: bool = True
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    embeddings = torch.concat([embedder.embed(v) for k, v in xs.items()], dim=2)
    B, E, T, C = embeddings.shape
    embeddings = embeddings.view(B, E * T, C)
    if ys is not None:
        targets = torch.concat([v for _, v in ys.items()], dim=2)
        targets = targets.view(B, E * T)

        if ms is not None:
            masks = torch.concat([v for _, v in ms.items()], dim=2)
            masks = masks.view(B, E * T)
        else:
            masks = torch.ones_like(targets, dtype=torch.float)
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


def init_default_config(transformer_model_args: GPTConfig) -> MugatoConfig:
    """Create a default MugatoConfig."""
    return MugatoConfig()


class Mugato(torch.nn.Module):
    def __init__(
        self,
        tokenizer: Tokenizer,
        sequence_model: GPT,
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
        self.sequence_model: GPT = sequence_model
        self.lm_head = torch.nn.Linear(self.config.n_embd, self.config.vocab_size).to(
            self.device
        )

    @overload
    def to(
        self,
        device: str | torch.device | int | None = ...,
        dtype: torch.dtype | None = ...,
        non_blocking: bool = ...
    ) -> "Mugato": ...

    @overload
    def to(self, dtype: torch.dtype, non_blocking: bool = ...) -> "Mugato": ...

    @overload
    def to(self, tensor: torch.Tensor, non_blocking: bool = ...) -> "Mugato": ...

    def to(self, *args: Any, **kwargs: Any) -> "Mugato":
        # Extract device if provided as first positional arg
        if args and isinstance(args[0], str | int | torch.device):
            device = args[0]
            if isinstance(device, str | int):
                self.device = torch.device(device)
            else:
                self.device = device

        # Delegate all arguments to parent
        return super().to(*args, **kwargs)

    def forward(
        self,
        xs: dict[str, torch.Tensor],
        ys: dict[str, torch.Tensor] | None = None,
        ms: dict[str, torch.Tensor] | None = None,
        pad: bool = True,
        sequence: Callable = sequence
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if ys is not None:
            assert ms is not None
            tok_emb, ys, ms = sequence(
                self.embedder,
                xs,
                ys,
                ms,
                pad=pad,
                sequence_length=self.config.block_size,
            )
            # After sequence call, these are definitely tensors, not None or dicts
            assert isinstance(ys, torch.Tensor)
            assert isinstance(ms, torch.Tensor)

            b, t, c = tok_emb.size()
            pos = torch.arange(0, t, dtype=torch.long, device=self.device)  # shape (t)
            pos_emb = self.sequence_model.transformer.wpe(  # type: ignore
                pos
            )  # position embeddings of shape (t, n_embd)
            xs = self.sequence_model.transformer.drop(tok_emb + pos_emb)  # type: ignore
            for block in self.sequence_model.transformer.h:  # type: ignore
                xs = block(xs)
            # Apply final layer norm from GPT before our lm_head
            xs = self.sequence_model.transformer.ln_f(xs)  # type: ignore
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
            pos_emb = self.sequence_model.transformer.wpe(  # type: ignore
                pos
            )  # position embeddings of shape (t, n_embd)
            xs = self.sequence_model.transformer.drop(tok_emb + pos_emb)  # type: ignore
            for block in self.sequence_model.transformer.h:  # type: ignore
                xs = block(xs)
            # Apply final layer norm from GPT before our lm_head
            xs = self.sequence_model.transformer.ln_f(xs)  # type: ignore
            logits = self.lm_head(xs)
            loss = None
        return logits, loss

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str,
    ) -> torch.optim.AdamW:
        # start with all of the candidate parameters
        param_dict = dict(self.named_parameters())
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed,
        # otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and
        # layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, "
            f"with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, "
            f"with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = {"fused": True} if use_fused else {}
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.sequence_model.transformer.wpe.weight.numel()  # type: ignore
        return n_params

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS

        Estimate written for transformer as sequence model.
        """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        # TODO: Fix hardcoding this.
        # L, H, Q, T = self.config.n_layer, self.config.n_head,
        # self.config.n_embd // self.config.n_head, self.config.block_size
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
