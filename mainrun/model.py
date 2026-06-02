import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    d_model: int
    dropout: float
    eos_id: int


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by swapping halves with a sign flip (for RoPE)."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding to x of shape (B, n_head, T, head_dim)."""
    return x * cos + rotate_half(x) * sin


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with RoPE and an additive (causal + per-title) mask."""

    rope_cos: torch.Tensor   # registered buffers (annotated so type-checkers see Tensors)
    rope_sin: torch.Tensor

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.head_dim = cfg.d_model // cfg.n_head
        assert self.head_dim % 2 == 0, "RoPE needs an even head_dim"
        self.n_head   = cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)  # residual-write projection
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop= nn.Dropout(cfg.dropout)
        # Rotary positional embedding (RoPE): encode position by rotating q/k
        # inside attention, rather than adding a learned positional vector.
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        pos = torch.arange(cfg.block_size).float()
        freqs = torch.outer(pos, inv_freq)        # (block_size, head_dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)   # (block_size, head_dim)
        self.register_buffer("rope_cos", emb.cos())
        self.register_buffer("rope_sin", emb.sin())

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """Attend over x (B, T, C); attn_mask (B, 1, T, T) adds -inf to blocked pairs."""
        B, T, C = x.size()
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).transpose(1, 3)
        q, k, v = qkv[..., 0, :, :], qkv[..., 1, :, :], qkv[..., 2, :, :]
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att + attn_mask   # additive mask: 0 where allowed, -inf where blocked (causal + per-title)
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    """Position-wise feed-forward network (Linear -> GELU -> Linear)."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.fc   = nn.Linear(cfg.d_model, 4 * cfg.d_model)
        self.act  = nn.GELU()
        self.proj = nn.Linear(4 * cfg.d_model, cfg.d_model)  # residual-write projection
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward network to x of shape (B, T, C)."""
        return self.drop(self.proj(self.act(self.fc(x))))


class Block(nn.Module):
    """Pre-LN transformer block: x + attn(ln1(x)), then x + mlp(ln2(x))."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.mlp  = MLP(cfg)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """Run attention then MLP, each added back to the residual stream."""
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """Decoder-only transformer with RoPE, residual-init scaling, and per-title masking."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop      = nn.Dropout(cfg.dropout)
        self.blocks    = nn.ModuleList()
        residual_projs = []
        for _ in range(cfg.n_layer):
            block = Block(cfg)
            self.blocks.append(block)
            residual_projs += [block.attn.proj, block.mlp.proj]  # write into residual stream
        self.ln_f      = nn.LayerNorm(cfg.d_model)
        self.head      = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)
        # GPT-2 residual scaling: shrink the projections that write into the
        # residual stream by 1/sqrt(2 * n_layer) so its variance stays ~constant
        # with depth at init (2 = attn proj + mlp proj per block).
        residual_std = 0.02 / math.sqrt(2 * cfg.n_layer)
        for proj in residual_projs:
            nn.init.normal_(proj.weight, mean=0.0, std=residual_std)
        self.head.weight = self.token_emb.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Initialise Linear/Embedding weights ~N(0, 0.02) and biases to zero."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return (logits, loss); loss is None when targets is None."""
        B, T = idx.size()
        # Document attention mask: each token attends only within its own title
        # (segments split at <eos>) and only to the past (causal). Derived from the
        # input ids, so evaluate() and the data pipeline stay untouched.
        eos = idx == self.cfg.eos_id
        seg = eos.cumsum(dim=1) - eos.long()              # (B, T) title index per position
        same_seg = seg.unsqueeze(2) == seg.unsqueeze(1)   # (B, T, T)
        causal = torch.ones(T, T, dtype=torch.bool, device=idx.device).tril()
        allowed = causal.unsqueeze(0) & same_seg          # (B, T, T)
        attn_mask = torch.zeros(B, 1, T, T, device=idx.device).masked_fill(
            ~allowed.unsqueeze(1), float("-inf"))
        tok = self.token_emb(idx)
        x = self.drop(tok)
        for block in self.blocks:
            x = block(x, attn_mask)
        x = self.ln_f(x)
        logits = self.head(x)
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='mean')
        return logits, loss
