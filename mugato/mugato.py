import math
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import ResNetV2
import tiktoken

from mugato.nano_gpt import LayerNorm, Block, GPTConfig
from mugato.data import Tokenizer
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
        if (
            data.size(-1) > 1
        ):  # Images are the only modality that have a channel dim > 1.
            #                                           (C,  P,  P)
            return self.image_embedding(data.view(B * E * T, 3, 16, 16)).view(
                B, E, T, -1
            )
        else:
            return self.lookup_embedding(data.view(B * E * T)).view(B, E, T, -1)


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
    tokenizer: Tokenizer
    device: str = select_device()
    n_embd: int = 512
    sequence_length: int = 1024
    vocab_size: int = 51281  # text vocab + discrete vocab


def init_default_config(transformer_model_args: GPTConfig) -> MugatoConfig:
    text_tokenizer = tiktoken.get_encoding("r50k_base")
    tokenizer = Tokenizer(text_tokenizer)
    transformer_config = GPTConfig(**transformer_model_args)
    return MugatoConfig(
        tokenizer=tokenizer,
        transformer_config=transformer_config,
        sequence_length=1024,
        vocab_size=tokenizer.vocab_size,
    )


transformer_model_args = dict(
    n_layer=4, n_head=4, n_embd=512, block_size=1024, bias=False, dropout=0.0
)  # start with model_args from command line
default_config = init_default_config(transformer_model_args)

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

class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.sequence_length, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList(
                    [
                        Block(config)
                        for _ in range(config.n_layer)
                    ]
                ),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
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
            self.config.vocab_size, 
            self.config.n_embd
        ).to(self.device)
        
        self.image_embedding = ResNetV2(
            layers=[3, 4, 6, 3], 
            num_classes=self.config.n_embd
        ).to(self.device)
        
        self.embedder = Embedder(self.lookup_embedding, self.image_embedding)
        # TODO:
        # Since we're doing our own embedding, we need to handle our own
        # position embedding.
        self.transformer = sequence_model
        self.lm_head = torch.nn.Linear(
            self.config.n_embd, 
            self.config.vocab_size
        ).to(self.device)

    def forward(self, xs, ys=None, ms=None, pad=True):
        if ys is not None:
            tok_emb, ys, ms = sequence(self.embedder, xs, ys, ms, pad=pad)
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
