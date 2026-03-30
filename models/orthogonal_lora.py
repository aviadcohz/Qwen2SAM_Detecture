"""
Orthogonal LoRA for nn.MultiheadAttention and nn.Linear layers.

Applies low-rank adaptation ΔW = A·B while regularizing ΔW to be
orthogonal to the original frozen weights W₀. This prevents catastrophic
forgetting of SAM3's zero-shot capabilities.

The orthogonality penalty projects ΔW onto the dominant singular vectors
of W₀ and penalizes the projection magnitude:
    L_orth = || U_k^T · ΔW ||_F^2
where U_k are the top-k left singular vectors of W₀.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrthogonalLoRALinear(nn.Module):
    """
    LoRA adapter for a single nn.Linear layer with orthogonality constraint.

    Wraps a frozen linear layer and adds a trainable low-rank bypass:
        output = frozen_linear(x) + (x @ A^T) @ B^T   [scaled by alpha/r]

    The orthogonality penalty ensures ΔW = B^T @ A^T is orthogonal to W₀'s
    dominant singular subspace.
    """

    def __init__(
        self,
        frozen_linear: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        n_singular: int = 32,
    ):
        """
        Args:
            frozen_linear: The original frozen nn.Linear layer.
            r: LoRA rank.
            alpha: LoRA scaling factor.
            n_singular: Number of top singular vectors of W₀ to project against.
        """
        super().__init__()
        self.frozen_linear = frozen_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_features = frozen_linear.in_features
        out_features = frozen_linear.out_features

        # LoRA matrices: A (r, in) and B (out, r)
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # Precompute top-k left singular vectors of W₀ (frozen, not a parameter)
        with torch.no_grad():
            W0 = frozen_linear.weight.data.float()  # (out, in)
            k = min(n_singular, min(W0.shape))
            U, S, Vt = torch.linalg.svd(W0, full_matrices=False)
            # U_k: (out, k) — dominant left singular vectors
            self.register_buffer("U_k", U[:, :k].clone())

        # Freeze the original linear
        for p in self.frozen_linear.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Frozen path
        base_out = self.frozen_linear(x)
        # LoRA path: x @ A^T @ B^T * scaling
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out

    def orthogonal_penalty(self) -> torch.Tensor:
        """
        Compute orthogonality penalty: || U_k^T @ ΔW ||_F^2

        ΔW = B @ A (out_features, in_features) scaled.
        Project ΔW onto U_k's column space and penalize.
        """
        delta_W = (self.lora_B @ self.lora_A) * self.scaling  # (out, in)
        # Project onto dominant subspace: U_k^T @ ΔW → (k, in)
        proj = self.U_k.T @ delta_W
        return (proj ** 2).sum()


def apply_orthogonal_lora_to_mha(
    mha: nn.MultiheadAttention,
    r: int = 8,
    alpha: float = 16.0,
    n_singular: int = 32,
    target_projections: tuple = ("q", "v"),
) -> dict:
    """
    Apply Orthogonal LoRA to selected projections of an nn.MultiheadAttention.

    nn.MultiheadAttention stores Q/K/V as a single packed `in_proj_weight`
    of shape (3 * embed_dim, embed_dim). We split it into 3 frozen linears
    and wrap the targeted ones with LoRA.

    Args:
        mha: The nn.MultiheadAttention module to wrap.
        r: LoRA rank.
        alpha: LoRA scaling factor.
        n_singular: Number of singular vectors for orthogonal constraint.
        target_projections: Which projections to apply LoRA to ("q", "k", "v").

    Returns:
        Dict mapping projection name to OrthogonalLoRALinear (for penalty collection).
    """
    embed_dim = mha.embed_dim
    lora_modules = {}

    # Check if using packed in_proj or separate q/k/v projections
    if mha.in_proj_weight is not None:
        # Packed: split into Q, K, V linears
        W = mha.in_proj_weight.data  # (3*embed_dim, embed_dim)
        b = mha.in_proj_bias.data if mha.in_proj_bias is not None else None

        splits = {"q": 0, "k": 1, "v": 2}
        frozen_linears = {}
        for name, idx in splits.items():
            start = idx * embed_dim
            end = (idx + 1) * embed_dim
            linear = nn.Linear(embed_dim, embed_dim, bias=(b is not None))
            linear.weight.data.copy_(W[start:end])
            if b is not None:
                linear.bias.data.copy_(b[start:end])
            frozen_linears[name] = linear

        # Replace the packed projection with split linears
        # We store them and override forward behavior
        for name in ("q", "k", "v"):
            if name in target_projections:
                lora_mod = OrthogonalLoRALinear(
                    frozen_linears[name], r=r, alpha=alpha, n_singular=n_singular,
                )
                lora_modules[name] = lora_mod
            else:
                # Keep frozen linear (no LoRA)
                for p in frozen_linears[name].parameters():
                    p.requires_grad = False
                lora_modules[name] = frozen_linears[name]

        # Disable the original packed projection
        mha.in_proj_weight = None
        mha.in_proj_bias = None
        mha._qkv_same_embed_dim = False

        # Store split modules on the MHA so they're part of the model
        mha.q_proj_lora = lora_modules["q"]
        mha.k_proj_lora = lora_modules["k"]
        mha.v_proj_lora = lora_modules["v"]

        # Monkey-patch the forward to use our split projections
        original_out_proj = mha.out_proj

        def patched_forward(
            query, key, value,
            key_padding_mask=None, need_weights=False,
            attn_mask=None, **kwargs,
        ):
            q = mha.q_proj_lora(query)
            k = mha.k_proj_lora(key)
            v = mha.v_proj_lora(value)

            # Use PyTorch's multi_head_attention_forward internals
            attn_output, attn_weights = F.multi_head_attention_forward(
                query=q, key=k, value=v,
                embed_dim_to_check=embed_dim,
                num_heads=mha.num_heads,
                in_proj_weight=None,
                in_proj_bias=None,
                bias_k=mha.bias_k, bias_v=mha.bias_v,
                add_zero_attn=mha.add_zero_attn,
                dropout_p=mha.dropout if mha.training else 0.0,
                out_proj_weight=original_out_proj.weight,
                out_proj_bias=original_out_proj.bias,
                training=mha.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=None,  # not used when q is pre-projected
                k_proj_weight=None,
                v_proj_weight=None,
            )
            return attn_output, attn_weights

        mha.forward = patched_forward

    else:
        # Already using separate q_proj, k_proj, v_proj
        proj_map = {"q": "q_proj", "k": "k_proj", "v": "v_proj"}
        for name in target_projections:
            attr = proj_map[name]
            original = getattr(mha, attr)
            lora_mod = OrthogonalLoRALinear(
                original, r=r, alpha=alpha, n_singular=n_singular,
            )
            setattr(mha, attr, lora_mod)
            lora_modules[name] = lora_mod

    # Return only the LoRA-wrapped modules (for penalty collection)
    return {k: v for k, v in lora_modules.items()
            if isinstance(v, OrthogonalLoRALinear)}


def collect_orthogonal_penalties(lora_modules: list) -> torch.Tensor:
    """
    Sum orthogonal penalties from all OrthogonalLoRALinear modules.

    Args:
        lora_modules: List of OrthogonalLoRALinear instances.
    Returns:
        Scalar penalty tensor.
    """
    penalty = torch.tensor(0.0, device=lora_modules[0].lora_A.device)
    for mod in lora_modules:
        penalty = penalty + mod.orthogonal_penalty()
    return penalty
