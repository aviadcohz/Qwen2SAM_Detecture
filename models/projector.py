"""
Smooth bottleneck MLP projector: Qwen hidden space (2048) → SAM3 query space (256).

3-layer architecture with LayerNorm to prevent information loss during
the severe dimensionality reduction.

Architecture:
    Linear(2048 → 1024) + LayerNorm(1024) + GELU
    Linear(1024 → 512)  + GELU
    Linear(512  → 256)
"""

import torch
import torch.nn as nn


class DescriptionProjector(nn.Module):
    """
    Projects per-texture hidden states from Qwen's LLM space (2048-dim)
    to SAM3's query embedding space (256-dim).

    Applied per-slot to the 6-vector query sequence:
    [DUSTBIN, TEX_1, ..., TEX_K, PAD...]
    """

    def __init__(self, llm_dim: int = 2048, sam_dim: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(llm_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, sam_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 6, llm_dim) — query sequence in LLM space
        Returns:
            (B, 6, sam_dim) — query sequence in SAM3 space
        """
        return self.proj(x)
