"""
dit
===

Diffusion-transformer backbone, adaLN-Zero style (Peebles & Xie), adapted to a
1D time axis.

Tokenisation
------------
One token per month; the regional dimension is the *channel* dimension of the
token (2 * y_dim channels for tas and pr jointly). With D ~ 50 regions and
windows of a few hundred months this is the right factorisation: a token per
(month, region) would give tens of thousands of tokens for no benefit, since
attention over the region axis is not needed — regions are a fixed, unordered
set of ~50, and their covariance is captured perfectly well by the channel-mixing
MLPs inside each block.

So: attention mixes time, MLPs mix space.

Conditioning
------------
Two paths, because the two conditioning signals have different structure:
  - the diffusion timestep is global per sample -> adaLN-Zero modulation
  - z / memory / seasonality vary per timestep -> added to the token embeddings
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Sinusoidal embedding of the diffusion timestep, then an MLP."""

    def __init__(self, hidden, freq_dim=256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, t):
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.freq_dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return self.mlp(emb)


class Attention(nn.Module):
    def __init__(self, hidden, heads):
        super().__init__()
        if hidden % heads != 0:
            raise ValueError("hidden must be divisible by heads")
        self.heads = heads
        self.head_dim = hidden // heads
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=True)
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # Full (non-causal) attention: the denoiser sees the whole window at
        # once, including the observed context on both sides of any gap. That is
        # what makes this outpainting rather than autoregression.
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class DiTBlock(nn.Module):
    def __init__(self, hidden, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden, heads)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden),
        )
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 6 * hidden))
        # Zero-init so each block starts as the identity and the network begins
        # training as a well-behaved shallow model.
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, c):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.adaLN(c).chunk(6, dim=-1)
        x = x + gate_a.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_a, scale_a))
        x = x + gate_m.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_m, scale_m))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden, out_channels):
        super().__init__()
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden, out_channels)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, c):
        shift, scale = self.adaLN(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm(x), shift, scale))


class DiT1D(nn.Module):
    """
    Time-axis DiT.

    forward(x, t_diff, cond, mask) -> eps_hat, all [B, T, C] except t_diff [B].

    `mask` is 1 on observed (context) positions and 0 on positions being
    generated. It is fed to the model as an extra input channel so the network
    knows which positions it may trust.
    """

    def __init__(self, in_channels, cond_dim, max_len, hidden=384, depth=8, heads=6,
                 mlp_ratio=4.0):
        super().__init__()
        self.in_channels = in_channels
        self.max_len = max_len
        # +1 channel for the observed/generated mask
        self.x_embed = nn.Linear(in_channels + 1, hidden)
        self.cond_embed = nn.Linear(cond_dim, hidden)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, hidden))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.t_embed = TimestepEmbedder(hidden)
        self.blocks = nn.ModuleList([DiTBlock(hidden, heads, mlp_ratio) for _ in range(depth)])
        self.final = FinalLayer(hidden, in_channels)

    def forward(self, x, t_diff, cond, mask):
        B, T, _ = x.shape
        if T > self.max_len:
            raise ValueError(f"sequence length {T} exceeds max_len {self.max_len}")
        h = self.x_embed(torch.cat([x, mask], dim=-1))
        h = h + self.pos_embed[:, :T] + self.cond_embed(cond)
        c = self.t_embed(t_diff)
        for blk in self.blocks:
            h = blk(h, c)
        return self.final(h, c)
