"""Attention modules for the MLT-2019 / ReCTS detector, injected into Ultralytics.

Two variants of multi-scale CBAM are defined on purpose:

``MultiScaleCBAM``
    The module as it exists in the runs behind Table 8 and the deployed ``best.pt``.
    Ultralytics' ``parse_model`` registers it in both ``base_modules`` and
    ``repeat_modules``, so a YAML entry ``[channels, r]`` is rewritten to
    ``(c1, scaled_c2, n, r)`` before construction. The second positional binds to
    ``r``, which makes the reduction ratio equal to the channel count and collapses
    the channel-attention bottleneck to a single channel. Kept unchanged so the
    as-deployed arm of the probe reproduces the trained network exactly.

``MSCBAMFixed``
    Identical computation with the reduction ratio honoured. ``install_modules.py``
    registers it in ``base_modules`` only, so its args arrive as ``(c1, r)``.

If the fixed variant wins the probe it becomes the production module and should be
renamed; until then the names document which arm is which.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SimpleChannelAttention(nn.Module):
    """Squeeze-excite channel gate with a ``c // r`` bottleneck."""

    def __init__(self, in_channels: int, r: int, *args: object, **kwargs: object) -> None:
        super().__init__()
        hidden = max(1, in_channels // r)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, 1, 0, bias=True),
            nn.ReLU(),
            nn.Conv2d(hidden, in_channels, 1, 1, 0, bias=True),
        )
        self.act = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        return x * self.act(self.fc(self.pool(x)))


class MultiScaleSpatialAttention(nn.Module):
    """Spatial gate from three parallel kernels over pooled channel statistics."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__()
        self.conv3 = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.conv5 = nn.Conv2d(2, 1, kernel_size=5, padding=2, bias=False)
        self.conv7 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        pooled = torch.cat([avg_out, max_out], dim=1)
        attn = self.conv3(pooled) + self.conv5(pooled) + self.conv7(pooled)
        return x * self.sigmoid(attn)


class SingleScaleSpatialAttention(nn.Module):
    """Spatial gate from one 7x7 kernel, the standard CBAM formulation."""

    def __init__(self, kernel_size: int = 7, *args: object, **kwargs: object) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        pooled = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(pooled))


class MultiScaleCBAM(nn.Module):
    """As-deployed variant. See module docstring for why ``r`` is not honoured."""

    def __init__(self, c1: int, r: int = 16, *args: object, **kwargs: object) -> None:
        super().__init__()
        self.ca = SimpleChannelAttention(c1, r)
        self.sa = MultiScaleSpatialAttention()

    def forward(self, x: Tensor) -> Tensor:
        return self.sa(self.ca(x))


class MSCBAMFixed(nn.Module):
    """Channel-preserving MS-CBAM whose reduction ratio reaches the bottleneck."""

    def __init__(self, c1: int, r: int = 16, *args: object, **kwargs: object) -> None:
        super().__init__()
        self.ca = SimpleChannelAttention(c1, r)
        self.sa = MultiScaleSpatialAttention()

    def forward(self, x: Tensor) -> Tensor:
        return self.sa(self.ca(x))


class StandardCBAM(nn.Module):
    """Single-kernel counterpart to ``MSCBAMFixed``.

    Shares the channel-attention design so that a comparison against ``MSCBAMFixed``
    isolates the spatial kernel, which is the multi-scale claim under test. Ultralytics'
    own ``CBAM`` is not usable here: it is absent from ``parse_model``'s module sets and
    its channel branch has no bottleneck, which would confound the comparison.
    """

    def __init__(self, c1: int, r: int = 16, *args: object, **kwargs: object) -> None:
        super().__init__()
        self.ca = SimpleChannelAttention(c1, r)
        self.sa = SingleScaleSpatialAttention(7)

    def forward(self, x: Tensor) -> Tensor:
        return self.sa(self.ca(x))


class CrossAttention2D(nn.Module):
    """Multi-head cross-attention between two flattened feature maps."""

    def __init__(self, in_channels: int, n_heads: int = 8) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=in_channels, num_heads=n_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, query_source: Tensor, key_value_source: Tensor) -> Tensor:
        b, c, h, w = query_source.shape
        query_seq = query_source.flatten(2).permute(0, 2, 1)
        key_value_seq = key_value_source.flatten(2).permute(0, 2, 1)
        attn_output, _ = self.attention(
            query=query_seq, key=key_value_seq, value=key_value_seq
        )
        attended = self.norm(query_seq + attn_output)
        return attended.permute(0, 2, 1).view(b, c, h, w)


class CrossAttentionBlock(nn.Module):
    """Projects a key/value source to the query width, then cross-attends.

    ``parse_model`` passes inputs in YAML ``from`` order, so a ``from`` of ``[5, 10]``
    delivers layer 5 as the key/value source and layer 10 as the query.
    """

    def __init__(self, c1: int, c2: int, n_heads: int = 8) -> None:
        super().__init__()
        self.kv_proj = (
            nn.Conv2d(c2, c1, kernel_size=1, bias=False) if c1 != c2 else nn.Identity()
        )
        self.attention = CrossAttention2D(in_channels=c1, n_heads=n_heads)

    def forward(self, x: list[Tensor]) -> Tensor:
        key_value_source, query_source = x
        return self.attention(query_source, self.kv_proj(key_value_source))


__all__ = [
    "SimpleChannelAttention",
    "MultiScaleSpatialAttention",
    "SingleScaleSpatialAttention",
    "MultiScaleCBAM",
    "MSCBAMFixed",
    "StandardCBAM",
    "CrossAttention2D",
    "CrossAttentionBlock",
]
