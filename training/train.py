"""
Training loop for Qwen2SAM-DeTexture.

Usage:
    cd /home/aviad/Qwen2SAM_DeTexture
    python -m training.train --config configs/detexture.yaml
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from models.qwen2sam_detexture import Qwen2SAMDeTexture
from models.hungarian import batch_hungarian_match, permute_gt_mask
from models.losses import combined_loss
from data.dataset import DeTextureDataset, DeTextureCollator
from training.utils import (
    load_config, set_seed, AverageMeter, WarmupCosineScheduler,
    get_lr, save_checkpoint,
)


def train_one_epoch(
    model, dataloader, optimizer, scheduler, scaler, epoch, cfg, device,
):
    """Run one training epoch."""
    model.train()
    meters = {
        k: AverageMeter() for k in [
            "total", "mask_ce", "mask_dice", "text_contrastive",
            "count_mse", "lm_loss", "orthogonal_reg",
        ]
    }

    accum_steps = cfg["training"]["gradient_accumulation_steps"]
    max_grad_norm = cfg["training"]["max_grad_norm"]
    seg_warmup = cfg["training"].get("seg_grad_warmup_epochs", 10)
    seg_grad_to_lm = (epoch >= seg_warmup)

    match_cfg = cfg.get("matching", {})
    text_weight = match_cfg.get("text_weight", 0.5)
    mask_weight = match_cfg.get("mask_weight", 0.5)

    optimizer.zero_grad()
    t0 = time.time()

    for step, batch in enumerate(dataloader):
        # Move to device
        qwen_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                       for k, v in batch["qwen_inputs"].items()}
        sam_images = batch["sam_images"].to(device)
        index_masks = batch["index_masks"].to(device)
        k_gts = batch["k_gts"].to(device)
        gt_text_embeds = batch.get("gt_text_embeds")
        if gt_text_embeds is not None:
            gt_text_embeds = gt_text_embeds.to(device)

        # Forward pass with AMP
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(
                qwen_inputs=qwen_inputs,
                sam_images=sam_images,
                seg_grad_to_lm=seg_grad_to_lm,
            )

            mask_logits = out["mask_logits"]   # (B, 6, H, W)
            tex_embeds = out["tex_embeds"]     # (B, 5, llm_dim)
            text_align = out["text_align_embeds"]  # (B, 5, clip_dim)
            k_preds = out["k_preds"]           # (B,)
            pad_mask = out["pad_mask"]         # (B, 6)
            lm_loss = out["lm_loss"]

            # Hungarian matching (no_grad)
            with torch.no_grad():
                permutations, pad_masks, hallucinated_masks = batch_hungarian_match(
                    pred_logits=mask_logits,
                    gt_masks=index_masks,
                    k_preds=k_preds,
                    k_gts=k_gts,
                    pred_text_embeds=text_align,
                    gt_text_embeds=gt_text_embeds,
                    text_weight=text_weight,
                    mask_weight=mask_weight,
                )
                permutations = permutations.to(device)
                pad_masks = pad_masks.to(device)
                hallucinated_masks = hallucinated_masks.to(device)

                # Remap GT masks to match output channel order
                B = mask_logits.shape[0]
                permuted_gts = []
                for b in range(B):
                    pg = permute_gt_mask(
                        index_masks[b], permutations[b], int(k_gts[b].item())
                    )
                    permuted_gts.append(pg)
                permuted_gt = torch.stack(permuted_gts).to(device)

            # Combined loss
            losses = combined_loss(
                mask_logits=mask_logits,
                permuted_gt=permuted_gt,
                pad_mask=pad_masks,
                pred_text_embeds=text_align,
                gt_text_embeds=gt_text_embeds,
                permutations=permutations,
                hallucinated_mask=hallucinated_masks,
                k_preds=k_preds,
                k_gts=k_gts,
                lm_loss=lm_loss,
                model=model,
                cfg=cfg,
            )

            loss = losses["total"] / accum_steps

        # Backward
        scaler.scale(loss).backward()

        # Optimizer step every accum_steps
        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_grad_norm,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        # Update meters
        for k in meters:
            if k in losses:
                meters[k].update(losses[k].item())

        # Log
        if (step + 1) % 10 == 0 or step == 0:
            elapsed = time.time() - t0
            lr = get_lr(optimizer)
            print(
                f"  [{step+1}/{len(dataloader)}] "
                f"loss={meters['total'].avg:.4f} "
                f"ce={meters['mask_ce'].avg:.4f} "
                f"dice={meters['mask_dice'].avg:.4f} "
                f"text={meters['text_contrastive'].avg:.4f} "
                f"count={meters['count_mse'].avg:.4f} "
                f"lm={meters['lm_loss'].avg:.4f} "
                f"orth={meters['orthogonal_reg'].avg:.6f} "
                f"lr={lr:.2e} "
                f"({elapsed:.1f}s)",
                flush=True,
            )

    return {k: m.avg for k, m in meters.items()}


@torch.no_grad()
def validate(model, dataloader, cfg, device):
    """Run validation and compute mean IoU."""
    model.eval()
    ious = []

    for batch in dataloader:
        qwen_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                       for k, v in batch["qwen_inputs"].items()}
        sam_images = batch["sam_images"].to(device)
        index_masks = batch["index_masks"].to(device)
        k_gts = batch["k_gts"]

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(qwen_inputs=qwen_inputs, sam_images=sam_images)

        mask_logits = out["mask_logits"]  # (B, 6, H, W)
        pad_mask = out["pad_mask"]

        # WTA assignment (argmax over non-PAD channels)
        masked_logits = mask_logits.clone()
        inf_mask = pad_mask.unsqueeze(-1).unsqueeze(-1).expand_as(masked_logits)
        masked_logits[inf_mask] = float("-inf")
        preds = masked_logits.argmax(dim=1)  # (B, H, W)

        # Per-sample IoU
        B = mask_logits.shape[0]
        for b in range(B):
            k = int(k_gts[b].item())
            sample_ious = []
            for c in range(1, k + 1):
                pred_c = (preds[b] == c)
                gt_c = (index_masks[b] == c)
                inter = (pred_c & gt_c).sum().float()
                union = (pred_c | gt_c).sum().float()
                iou = (inter / union.clamp(min=1)).item()
                sample_ious.append(iou)
            if sample_ious:
                ious.append(sum(sample_ious) / len(sample_ious))

    mean_iou = sum(ious) / len(ious) if ious else 0.0
    return mean_iou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/detexture.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Model --------------------------------------------------------- #
    print("Building Qwen2SAM-DeTexture model...")
    model = Qwen2SAMDeTexture(cfg, device=str(device))
    params = model.num_trainable_params()
    print(f"Trainable parameters:")
    for k, v in params.items():
        print(f"  {k}: {v:,}")

    # ---- Data ---------------------------------------------------------- #
    data_cfg = cfg["data"]
    train_ds = DeTextureDataset(
        data_cfg["train_metadata"],
        image_size=data_cfg.get("image_size", 1008),
        augment=data_cfg.get("augment", True),
    )
    val_ds = DeTextureDataset(
        data_cfg["val_metadata"],
        image_size=data_cfg.get("image_size", 1008),
        augment=False,
    )

    # Optional: load CLIP model for text contrastive targets
    clip_model = None
    clip_name = cfg["model"].get("clip_model")
    if clip_name:
        from transformers import CLIPModel
        clip_model = CLIPModel.from_pretrained(clip_name).to(device).eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        print(f"Loaded CLIP model: {clip_name}")

    collator = DeTextureCollator(model.processor, clip_model=clip_model)
    train_cfg = cfg["training"]

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 2),
        collate_fn=collator,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )

    # ---- Optimizer + Scheduler ----------------------------------------- #
    param_groups = model.get_parameter_groups(train_cfg["learning_rate"])
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )

    steps_per_epoch = len(train_loader) // train_cfg["gradient_accumulation_steps"]
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=train_cfg.get("warmup_epochs", 5),
        total_epochs=train_cfg["num_epochs"],
        min_lr=train_cfg.get("min_lr", 1e-6),
        steps_per_epoch=max(steps_per_epoch, 1),
    )

    scaler = torch.amp.GradScaler("cuda")

    # ---- Training loop ------------------------------------------------- #
    best_iou = 0.0
    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))

    print(f"\nStarting training: {train_cfg['num_epochs']} epochs, "
          f"{len(train_ds)} train / {len(val_ds)} val samples")
    print(f"Effective batch size: {train_cfg['batch_size']} × "
          f"{train_cfg['gradient_accumulation_steps']} = "
          f"{train_cfg['batch_size'] * train_cfg['gradient_accumulation_steps']}")

    for epoch in range(train_cfg["num_epochs"]):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{train_cfg['num_epochs']}")
        print(f"{'='*60}")

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, epoch, cfg, device,
        )

        # Validate
        val_iou = validate(model, val_loader, cfg, device)
        print(f"\n  Val mIoU: {val_iou:.4f}")

        # Save checkpoint
        is_best = val_iou > best_iou
        if is_best:
            best_iou = val_iou
            save_checkpoint(
                model, optimizer, epoch,
                str(ckpt_dir / "best.pt"),
                extra={"val_iou": val_iou, "train_metrics": train_metrics},
            )
            print(f"  ** New best: {best_iou:.4f} **")

        if (epoch + 1) % train_cfg.get("save_every", 5) == 0:
            save_checkpoint(
                model, optimizer, epoch,
                str(ckpt_dir / f"epoch_{epoch+1}.pt"),
                extra={"val_iou": val_iou},
            )

    print(f"\nTraining complete. Best val mIoU: {best_iou:.4f}")


if __name__ == "__main__":
    main()
