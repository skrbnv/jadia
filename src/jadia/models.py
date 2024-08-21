import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from torch import Tensor


class DirectionalConv2d(nn.Module):
    def __init__(
        self,
        shape: tuple,
        in_channels: int,
        out_channels: int,
        kernel: int | tuple,
        stride: int | tuple = 1,
        padding: int | tuple = 0,
        groups: int = 1,
        bias: bool = True,
        extras: list | None = None,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
        self.extras = nn.Sequential(*extras) if extras is not None else None
        self.alignment = DirectionalAlignment(out_channels, shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        if self.extras is not None:
            x = self.extras(x)
        y = x.clone()
        y = self.alignment(y)
        x = x + y
        x = self.bn2(x)
        return x


class DirectionalAlignment(nn.Module):
    def __init__(self, planes: int, shape: tuple) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.Tensor(planes, shape[0], shape[0]))
        nn.init.kaiming_normal_(self.weights)
        self.bn = nn.BatchNorm2d(planes)
        self.dim = torch.sqrt(torch.tensor(shape[-1]))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x @ x.transpose(-1, -2)
        x = x / self.dim
        x = x * self.weights
        x = torch.sum(x, dim=-1, keepdim=True)
        return x

    def extra_repr(self) -> str:
        return f"DirectionalAlignment: planes={self.weights.shape[0]}, shape={self.weights.shape[1:]}"


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor] = None,
    ):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = nn.LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = nn.LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            nn.Linear(n_state, n_mlp), nn.GELU(), nn.Linear(n_mlp, n_state)
        )
        self.mlp_ln = nn.LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn_ln:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


class Model(nn.Module):
    def __init__(
        self,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=n_ctx // 2,
                kernel_size=(3, 7),
                stride=(1, 3),
                padding=(1, 3),
            ),
            nn.ReLU(),
            nn.BatchNorm2d(n_ctx // 2),
        )
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = DirectionalConv2d(
            in_channels=n_ctx // 2,
            out_channels=n_ctx,
            kernel=3,
            stride=1,
            padding=1,
            shape=(40, 32),
            extras=[],
        )
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))
        self.blocks = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = nn.LayerNorm(n_state)
        self.collapsing_weights = nn.Parameter(
            torch.nn.init.xavier_uniform_(torch.rand((n_state, 1)))
        )
        self.classification_head = nn.Sequential(
            nn.Linear(n_ctx, n_state),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(n_state, output_dim),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = x.flatten(2)
        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)
        for block in self.blocks:
            x = block(x)
        x = self.ln_post(x)
        x = x @ self.collapsing_weights
        x = x.flatten(1)
        x = self.classification_head(x)
        return x


def model_lite():
    return Model(n_ctx=256, n_state=320, n_head=4, n_layer=0, output_dim=256)
