"""Decoder-only, pre-normalized GPT with causal multi-head self-attention."""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = True

    def __post_init__(self):
        for name in ('vocab_size', 'block_size', 'n_layer', 'n_head', 'n_embd'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if self.n_embd % self.n_head:
            raise ValueError('n_embd must be divisible by n_head')
        if not 0 <= self.dropout < 1:
            raise ValueError('dropout must be in [0, 1)')


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(batch, length, self.n_head, width // self.n_head).transpose(1, 2)
                   for t in (q, k, v)]
        # is_causal=True forbids attention to future tokens. Evaluation must pass zero dropout.
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(batch, length, width)
        return self.residual_dropout(self.proj(y))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)
        for name, parameter in self.named_parameters():
            if name.endswith('proj.weight'):
                nn.init.normal_(parameter, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        if idx.ndim != 2 or idx.shape[0] == 0 or not 0 < idx.shape[1] <= self.config.block_size:
            raise ValueError('tokens must have shape (batch, time), with 1 <= time <= block_size')
        if targets is not None and targets.shape != idx.shape:
            raise ValueError('targets must have the same shape as input tokens')
        positions = torch.arange(idx.shape[1], device=idx.device)
        x = self.dropout(self.token_embedding(idx) + self.position_embedding(positions))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None if targets is None else F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size), targets.reshape(-1))
        return logits, loss

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        if max_new_tokens < 0 or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('max_new_tokens must be nonnegative and temperature must be positive and finite')
        if top_k is not None and (type(top_k) is not int or top_k <= 0):
            raise ValueError('top_k must be a positive integer or None')
        if idx.ndim != 2 or idx.shape[0] == 0 or idx.shape[1] == 0:
            raise ValueError('generation requires a nonempty batch of prompt tokens')
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                logits, _ = self(idx[:, -self.config.block_size:])
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    threshold = torch.topk(logits, min(top_k, logits.shape[-1])).values[:, [-1]]
                    logits = logits.masked_fill(logits < threshold, float('-inf'))
                token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                idx = torch.cat((idx, token), dim=1)
        finally:
            self.train(was_training)
        return idx
