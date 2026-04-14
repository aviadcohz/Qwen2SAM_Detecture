#!/usr/bin/env python3
"""
V5 Oracle Evaluation: GT descriptions → training forward() → RWTD mIoU.

Bypasses Qwen generation entirely. Feeds GT descriptions through the
teacher-forced training path (with block-diagonal attention mask),
extracts [SEG] hidden states, projects through the bottleneck, and
runs SAM to produce masks.

This isolates the V5 geometric architecture (bottleneck projector +
block-diagonal mask + SAM Orth LoRA) from the broken text generation.

If mIoU is high on RWTD → the Information Bottleneck fixes Directional
Drift and the architecture generalizes. The only remaining issue is
LoRA text generation stability (a training dynamics fix, not architectural).

Usage:
    cd /home/aviad/Qwen2SAM_DeTexture
    python scripts/eval_rwtd_oracle.py --checkpoint checkpoints/best.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from models.qwen2sam_detexture import Qwen2SAMDeTexture, MAX_TEXTURES
from data.dataset import DeTextureDataset, DeTextureCollator
from training.utils import load_config, load_checkpoint
from training.monitor import _compute_ari
from scipy.optimize import linear_sum_assignment


def compute_miou(pred, gt, kp, kg):
    if kp == 0 or kg == 0:
        return 0.0
    cost = np.zeros((kp, kg))
    for pi in range(kp):
        for gi in range(kg):
            inter = ((pred == pi + 1) & (gt == gi + 1)).sum()
            union = ((pred == pi + 1) | (gt == gi + 1)).sum()
            cost[pi, gi] = 1.0 - inter / max(union, 1)
    r, c = linear_sum_assignment(cost)
    ious = [1.0 - cost[ri, ci] for ri, ci in zip(r, c) if ri < kp and ci < kg]
    return np.mean(ious) if ious else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/detexture.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--test-metadata",
                        default="/home/aviad/datasets/RWTD/metadata.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda")

    print("Building V5 model...")
    model = Qwen2SAMDeTexture(cfg, device="cuda")
    print(f"Loading {args.checkpoint}...")
    load_checkpoint(model, None, args.checkpoint, device="cuda")
    model.eval()

    # Use the TRAINING collator (inference=False) so it includes
    # the GT assistant text with <|seg|> tokens in the sequence.
    # This is what forward() expects for teacher forcing.
    ds = DeTextureDataset(
        args.test_metadata, image_size=1008, augment=False,
        qwen_gt_embeds_path=cfg["data"].get("qwen_gt_embeds_path"),
    )
    collator = DeTextureCollator(model.processor, inference=False)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collator,
    )

    print(f"RWTD Oracle eval: {len(ds)} samples")
    print(f"Using training forward() with block-diagonal mask + GT descriptions")
    print(f"Qwen generation BYPASSED — pure geometric architecture test\n")

    ious = []
    aris = []
    k_preds_list = []

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            qwen_inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch["qwen_inputs"].items()
            }
            sam_images = batch["sam_images"].to(device)
            index_masks = batch["index_masks"]
            k_gt = int(batch["k_gts"][0].item())

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model.forward(
                    qwen_inputs=qwen_inputs,
                    sam_images=sam_images,
                    seg_grad_to_lm=False,
                )

            mask_logits = out["mask_logits"]
            pad_mask = out["pad_mask"]
            k_pred = int(out["k_preds"][0].item())
            k_preds_list.append(k_pred)

            # Upsample + argmax
            masked = mask_logits.clone()
            inf_mask = pad_mask.unsqueeze(-1).unsqueeze(-1).expand_as(masked)
            masked[inf_mask] = float("-inf")
            H, W = index_masks.shape[1], index_masks.shape[2]
            if masked.shape[2] != H:
                masked = F.interpolate(
                    masked.float(), size=(H, W),
                    mode="bilinear", align_corners=False,
                )
            pred = masked[0].argmax(dim=0).cpu().numpy()
            gt = index_masks[0].numpy()

            miou = compute_miou(pred, gt, k_pred, k_gt)
            ious.append(miou)

            # ARI
            try:
                ari = _compute_ari(pred, gt)
            except Exception:
                ari = 0.0
            aris.append(ari)

            if (idx + 1) % 50 == 0:
                print(f"  {idx+1}/{len(ds)}  avg mIoU={np.mean(ious):.4f}  "
                      f"avg mARI={np.mean(aris):.4f}", flush=True)

    mean_iou = np.mean(ious)
    mean_ari = np.mean(aris)

    print(f"\n{'='*60}")
    print(f"  V5 ORACLE EVALUATION — RWTD (GT descriptions, no generation)")
    print(f"{'='*60}")
    print(f"  Checkpoint:  {args.checkpoint}")
    print(f"  mIoU:        {mean_iou:.4f}")
    print(f"  mARI:        {mean_ari:.4f}")
    print(f"  k_pred dist: {dict(sorted(zip(*np.unique(k_preds_list, return_counts=True))))}")
    print(f"  >0.7 mIoU:   {sum(1 for x in ious if x > 0.7)}/{len(ious)}")
    print(f"  <0.3 mIoU:   {sum(1 for x in ious if x < 0.3)}/{len(ious)}")
    print()
    print(f"  COMPARISON:")
    print(f"    V5 Oracle (this):     mIoU={mean_iou:.4f}  mARI={mean_ari:.4f}")
    print(f"    V4-Slim ep5 (live):   mIoU=0.7316  mARI=0.5727")
    print(f"    V4-Slim ep7 (live):   mIoU=0.7308")
    print(f"    V4 (10.5M) ep5:       mIoU=0.6921")
    print(f"    ZS baseline:          mIoU=0.7063")
    print()
    if mean_iou > 0.73:
        print(f"  → V5 BOTTLENECK GENERALIZES. Architecture validated.")
        print(f"    Fix: reduce qwen_lr_scale, add LM regularization weight.")
    elif mean_iou > 0.65:
        print(f"  → V5 partially works. Bottleneck helps but not fully there.")
    else:
        print(f"  → V5 weights may have collapsed. Check training dynamics.")


if __name__ == "__main__":
    main()
