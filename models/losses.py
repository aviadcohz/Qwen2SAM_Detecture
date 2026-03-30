"""
Loss functions for Qwen2SAM-DeTexture.

L_total = λ_mask × (CE + Dice) + λ_text × Contrastive + λ_count × MSE
        + λ_lm × LM + λ_orth × Orthogonal

CRITICAL: CrossEntropyLoss receives RAW logits (applies LogSoftmax internally).
          Dice loss receives F.softmax(logits, dim=1) probabilities.
          PAD channels are set to -inf before both losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.orthogonal_lora import OrthogonalLoRALinear


# ===================================================================== #
#  Mask Losses                                                            #
# ===================================================================== #

def mask_pad_logits(logits: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    """
    Set PAD channel logits to -inf so they contribute nothing to
    CrossEntropyLoss or Softmax.

    Args:
        logits: (B, 6, H, W) raw logits.
        pad_mask: (B, 6) True for PAD channels.

    Returns:
        masked_logits: (B, 6, H, W) with PAD channels = -inf.
    """
    masked = logits.clone()
    # Expand pad_mask to spatial dims: (B, 6) → (B, 6, 1, 1)
    inf_mask = pad_mask.unsqueeze(-1).unsqueeze(-1).expand_as(masked)
    masked[inf_mask] = float("-inf")
    return masked


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor,
                       pad_mask: torch.Tensor) -> torch.Tensor:
    """
    Pixel-wise cross-entropy on raw logits.

    Args:
        logits: (B, 6, H, W) RAW logits — do NOT apply softmax.
        targets: (B, H, W) permuted GT index mask (values 0..5).
        pad_mask: (B, 6) True for PAD channels.

    Returns:
        Scalar CE loss.
    """
    masked_logits = mask_pad_logits(logits, pad_mask)
    return F.cross_entropy(masked_logits, targets, reduction="mean")


def dice_loss(logits: torch.Tensor, targets: torch.Tensor,
              pad_mask: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """
    Multi-class Dice loss on softmax probabilities.

    Args:
        logits: (B, 6, H, W) RAW logits.
        targets: (B, H, W) permuted GT index mask (values 0..5).
        pad_mask: (B, 6) True for PAD channels.
        smooth: Smoothing factor for numerical stability.

    Returns:
        Scalar Dice loss (1 - mean Dice across active channels).
    """
    masked_logits = mask_pad_logits(logits, pad_mask)
    probs = F.softmax(masked_logits, dim=1)  # (B, 6, H, W)

    # One-hot encode targets: (B, H, W) → (B, 6, H, W)
    gt_onehot = F.one_hot(targets.long(), num_classes=6)  # (B, H, W, 6)
    gt_onehot = gt_onehot.permute(0, 3, 1, 2).float()    # (B, 6, H, W)

    B = logits.shape[0]
    active_mask = ~pad_mask  # (B, 6) True for active channels

    dice_sum = torch.tensor(0.0, device=logits.device)
    n_active = 0

    for b in range(B):
        for c in range(6):
            if not active_mask[b, c]:
                continue
            pred_c = probs[b, c].reshape(-1)        # (H*W,)
            gt_c = gt_onehot[b, c].reshape(-1)      # (H*W,)
            inter = (pred_c * gt_c).sum()
            union = pred_c.sum() + gt_c.sum()
            dice_score = (2.0 * inter + smooth) / (union + smooth)
            dice_sum = dice_sum + (1.0 - dice_score)
            n_active += 1

    return dice_sum / max(n_active, 1)


def mask_loss(logits: torch.Tensor, targets: torch.Tensor,
              pad_mask: torch.Tensor,
              ce_weight: float = 1.0, dice_weight: float = 1.0) -> dict:
    """
    Combined mask loss: CE + Dice.

    Returns dict with individual and combined losses for logging.
    """
    ce = cross_entropy_loss(logits, targets, pad_mask)
    dc = dice_loss(logits, targets, pad_mask)
    total = ce_weight * ce + dice_weight * dc
    return {"mask_total": total, "mask_ce": ce, "mask_dice": dc}


# ===================================================================== #
#  Text Contrastive Loss                                                  #
# ===================================================================== #

def text_contrastive_loss(
    pred_embeds: torch.Tensor,
    gt_embeds: torch.Tensor,
    permutations: torch.Tensor,
    pad_mask: torch.Tensor,
    hallucinated_mask: torch.Tensor,
    k_preds: torch.Tensor,
    k_gts: torch.Tensor,
) -> torch.Tensor:
    """
    Cosine embedding loss between predicted and GT text embeddings.

    Matched pairs → target = +1 (attract).
    Hallucinated predictions → target = -1 against nearest GT (repel).

    Args:
        pred_embeds: (B, 5, D) Qwen <TEX_i> hidden states projected to CLIP space.
        gt_embeds: (B, 5, D) CLIP embeddings of GT descriptions (zero-padded).
        permutations: (B, 6) Hungarian matching result.
        pad_mask: (B, 6) True for PAD channels.
        hallucinated_mask: (B, 6) True for hallucinated predictions.
        k_preds: (B,) number of predicted textures.
        k_gts: (B,) number of GT textures.

    Returns:
        Scalar contrastive loss.
    """
    loss_fn = nn.CosineEmbeddingLoss(margin=0.3, reduction="mean")
    device = pred_embeds.device

    pred_list = []
    gt_list = []
    target_list = []

    B = pred_embeds.shape[0]
    for b in range(B):
        kp = int(k_preds[b].item())
        kg = int(k_gts[b].item())

        for i in range(kp):
            channel = i + 1  # channels 1..K_pred
            gt_class = permutations[b, channel].item()

            if pad_mask[b, channel]:
                continue

            if hallucinated_mask[b, channel]:
                # Repel against the nearest GT (use mean of all GT embeddings)
                if kg > 0:
                    mean_gt = gt_embeds[b, :kg].mean(dim=0)
                    pred_list.append(pred_embeds[b, i])
                    gt_list.append(mean_gt)
                    target_list.append(-1.0)
            elif gt_class > 0:
                # Attract toward matched GT
                pred_list.append(pred_embeds[b, i])
                gt_list.append(gt_embeds[b, gt_class - 1])
                target_list.append(1.0)

    if not pred_list:
        return torch.tensor(0.0, device=device)

    pred_stack = torch.stack(pred_list)     # (N, D)
    gt_stack = torch.stack(gt_list)         # (N, D)
    targets = torch.tensor(target_list, device=device)  # (N,)

    return loss_fn(pred_stack, gt_stack, targets)


# ===================================================================== #
#  Count Penalty Loss                                                     #
# ===================================================================== #

def count_loss(k_pred: torch.Tensor, k_gt: torch.Tensor) -> torch.Tensor:
    """MSE penalty on predicted vs GT texture count."""
    return F.mse_loss(k_pred.float(), k_gt.float())


# ===================================================================== #
#  Orthogonal Regularization Loss                                         #
# ===================================================================== #

def orthogonal_regularization(model: nn.Module) -> torch.Tensor:
    """
    Collect orthogonal penalties from all OrthogonalLoRALinear modules
    in the model.
    """
    penalty = torch.tensor(0.0)
    for module in model.modules():
        if isinstance(module, OrthogonalLoRALinear):
            p = module.orthogonal_penalty()
            penalty = penalty + p.to(penalty.device) if penalty.device != p.device else penalty + p
            if penalty.device == torch.device("cpu") and p.device != torch.device("cpu"):
                penalty = penalty.to(p.device)
    # Handle device
    return penalty


# ===================================================================== #
#  Combined Loss                                                          #
# ===================================================================== #

def combined_loss(
    mask_logits: torch.Tensor,
    permuted_gt: torch.Tensor,
    pad_mask: torch.Tensor,
    pred_text_embeds: torch.Tensor,
    gt_text_embeds: torch.Tensor,
    permutations: torch.Tensor,
    hallucinated_mask: torch.Tensor,
    k_preds: torch.Tensor,
    k_gts: torch.Tensor,
    lm_loss: torch.Tensor,
    model: nn.Module,
    cfg: dict,
) -> dict:
    """
    Compute the total loss with all components.

    Args:
        mask_logits: (B, 6, H, W) raw mask logits from SAM3.
        permuted_gt: (B, H, W) GT mask remapped to match output channels.
        pad_mask: (B, 6) True for PAD channels.
        pred_text_embeds: (B, 5, D) Qwen text embeddings in CLIP space.
        gt_text_embeds: (B, 5, D) GT CLIP embeddings.
        permutations: (B, 6) Hungarian matching result.
        hallucinated_mask: (B, 6) True for hallucinated predictions.
        k_preds: (B,) predicted texture counts.
        k_gts: (B,) GT texture counts.
        lm_loss: Scalar Qwen LM loss.
        model: The full model (for orthogonal penalty collection).
        cfg: Loss config dict with weight keys.

    Returns:
        Dict with total loss and all component losses for logging.
    """
    w = cfg.get("loss", {})
    lam_mask = w.get("mask_weight", 1.0)
    lam_text = w.get("text_contrastive_weight", 0.5)
    lam_count = w.get("count_weight", 0.1)
    lam_lm = w.get("lm_weight", 0.5)
    lam_orth = w.get("orthogonal_weight", 0.01)
    ce_w = w.get("ce_weight", 1.0)
    dice_w = w.get("dice_weight", 1.0)

    # Mask losses
    m_losses = mask_loss(mask_logits, permuted_gt, pad_mask,
                         ce_weight=ce_w, dice_weight=dice_w)

    # Text contrastive
    l_text = text_contrastive_loss(
        pred_text_embeds, gt_text_embeds,
        permutations, pad_mask, hallucinated_mask, k_preds, k_gts,
    )

    # Count penalty
    l_count = count_loss(k_preds, k_gts)

    # Orthogonal regularization
    l_orth = orthogonal_regularization(model)

    # Total
    total = (lam_mask * m_losses["mask_total"]
             + lam_text * l_text
             + lam_count * l_count
             + lam_lm * lm_loss
             + lam_orth * l_orth)

    return {
        "total": total,
        "mask_total": m_losses["mask_total"],
        "mask_ce": m_losses["mask_ce"],
        "mask_dice": m_losses["mask_dice"],
        "text_contrastive": l_text,
        "count_mse": l_count,
        "lm_loss": lm_loss,
        "orthogonal_reg": l_orth,
    }
