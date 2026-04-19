#!/usr/bin/env python3
"""RWTD ablation: structured prompt with SHORT length cap (3-6 words) on ep10.

Hypothesis: GT RWTD descriptions are ~4-6 words. Training conditioned [SEG]
on GT-style short contexts. Previous experiments show the '10-15 words'
cap yielded 0.6757 mIoU and ~15-word generations; removing the cap gave
0.6052 with ~18-word generations (longer is worse). Explicit 3-6 word
cap should push generations closer to GT length and lift mIoU.

Single minimal-surgery change vs the structured prompt: '(approximately
10-15 words)' -> '(3-6 words)'.
"""

import argparse
import json
import re
import sys
from collections import Counter
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


STRUCTURED_SYSTEM_PROMPT = (
    "You analyze surface textures in images. Always respond in the exact "
    "format requested, with no extra text."
)

CONDITION_A_USER_PROMPT = (
    "This image contains exactly TWO main visually distinct regions separated "
    "by a boundary (for example, a prominent foreground object and its "
    "background, or two contrasting materials).\n\n"
    "Write a brief descriptive phrase (3-6 words) for each of the two regions. "
    "Include the following precise information:\n"
    "1. Semantic Name: A natural, common name for the material or object.\n"
    "2. Distinct Visual Features: The core visual attributes like color, "
    "pattern, or texture that strongly contrast with the other region.\n"
    "3. Spatial Context: A brief note on its general position (e.g., "
    "'foreground', 'background', 'top-left').\n\n"
    "IMPORTANT: Describe the ENTIRE region as a collective group, NOT "
    "individual objects within it. Think of each region as a surface/area, "
    "not as a single object.\n\n"
    "Format your response exactly like this:\n"
    "TEXTURE_1: Texture of <description>\n"
    "TEXTURE_2: Texture of <description>"
)


class ShortLengthCollator(DeTextureCollator):
    def __init__(self, processor, system_prompt, user_prompt):
        super().__init__(processor, inference=True)
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

    def __call__(self, samples):
        texts, images = [], []
        for s in samples:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.user_prompt},
                ]},
            ]
            texts.append(self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            ))
            images.append(s["image"])
        qwen_inputs = self.processor(
            text=texts, images=images, return_tensors="pt", padding=True,
        )
        qwen_inputs.pop("token_type_ids", None)
        return {
            "qwen_inputs": qwen_inputs,
            "sam_images": torch.stack([s["sam_image"] for s in samples]),
            "index_masks": torch.stack([s["index_mask"] for s in samples]),
            "k_gts": torch.tensor([s["k_gt"] for s in samples], dtype=torch.long),
            "qwen_gt_embeds": torch.stack([s["qwen_gt_embeds"] for s in samples]),
        }


def compute_miou_hungarian(pred, gt, k_pred, k_gt):
    if k_pred == 0 or k_gt == 0:
        return 0.0
    cost = np.zeros((k_pred, k_gt))
    for pi in range(k_pred):
        for gi in range(k_gt):
            inter = ((pred == pi + 1) & (gt == gi + 1)).sum()
            union = ((pred == pi + 1) | (gt == gi + 1)).sum()
            cost[pi, gi] = 1.0 - inter / max(union, 1)
    r, c = linear_sum_assignment(cost)
    ious = [1.0 - cost[ri, ci] for ri, ci in zip(r, c) if ri < k_pred and ci < k_gt]
    return float(np.mean(ious)) if ious else 0.0


def evaluate(model, loader, device, label):
    model.eval()
    ious, aris, k_preds_list, gen_word_counts = [], [], [], []
    per_sample = []
    tex_re = re.compile(r"TEXTURE_\d+:\s*(?:Texture of\s*)?(.+?)\s*(?:<\|[^|]*\|>)?$")

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            qwen_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                           for k, v in batch["qwen_inputs"].items()}
            sam_images = batch["sam_images"].to(device)
            index_masks = batch["index_masks"]
            k_gt = int(batch["k_gts"][0].item())

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model.inference_forward(
                    qwen_inputs=qwen_inputs, sam_images=sam_images,
                )

            mask_logits = out["mask_logits"]
            pad_mask = out["pad_mask"]
            k_pred = int(out["k_preds"][0].item())
            gen_text = out.get("generated_text", [""])[0]
            k_preds_list.append(k_pred)

            for line in gen_text.splitlines():
                m = tex_re.match(line.strip())
                if m:
                    gen_word_counts.append(len(m.group(1).strip().split()))

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

            miou = compute_miou_hungarian(pred, gt, k_pred, k_gt)
            ari = _compute_ari(pred, gt)
            ious.append(miou)
            aris.append(ari)

            per_sample.append({
                "idx": idx, "k_gt": k_gt, "k_pred": k_pred,
                "miou": float(miou), "ari": float(ari),
                "generated_text": gen_text,
            })

            if (idx + 1) % 50 == 0:
                print(f"    {idx+1}/{len(loader)} avg mIoU={np.mean(ious):.4f} "
                      f"mARI={np.mean(aris):.4f} "
                      f"avg_words/desc={np.mean(gen_word_counts):.1f}", flush=True)

    return {
        "label": label,
        "mean_iou": float(np.mean(ious)),
        "mean_ari": float(np.mean(aris)),
        "avg_words_per_desc": float(np.mean(gen_word_counts)) if gen_word_counts else 0.0,
        "k_dist": {int(k): int(v) for k, v in Counter(k_preds_list).items()},
        "per_sample": per_sample,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/detexture.yaml")
    p.add_argument("--checkpoint", default="checkpoints/epoch_10.pt")
    p.add_argument("--test-metadata", default="/home/aviad/datasets/RWTD/metadata.json")
    p.add_argument("--output-dir", default="checkpoints/ablation_short_length_ep10")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    device = torch.device("cuda")

    (out / "prompts.txt").write_text(
        f"=== SYSTEM ===\n{STRUCTURED_SYSTEM_PROMPT}\n\n"
        f"=== CONDITION A (exactly 2, 3-6 words) ===\n{CONDITION_A_USER_PROMPT}\n"
    )

    print("Building model...")
    model = Qwen2SAMDeTexture(cfg, device="cuda")
    print(f"Loading {args.checkpoint}")
    load_checkpoint(model, None, args.checkpoint, device="cuda")
    model.eval()

    test_ds = DeTextureDataset(
        args.test_metadata,
        image_size=cfg["data"].get("image_size", 1008),
        augment=False,
        qwen_gt_embeds_path=cfg["data"].get("qwen_gt_embeds_path"),
    )
    print(f"Test dataset: {len(test_ds)} samples")

    print(f"\n{'='*70}\n  CONDITION A: structured prompt (3-6 words), 'exactly TWO'\n{'='*70}")
    collator = ShortLengthCollator(model.processor, STRUCTURED_SYSTEM_PROMPT, CONDITION_A_USER_PROMPT)
    loader = torch.utils.data.DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collator,
    )
    res = evaluate(model, loader, device, "A_short_exactly_2")
    res["checkpoint"] = str(args.checkpoint)
    (out / "condition_A_short_exactly_2.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False)
    )

    summary = {
        "checkpoint": str(args.checkpoint),
        "n_samples": len(res["per_sample"]),
        "change_from_structured": "Length cap: '(approximately 10-15 words)' -> '(3-6 words)'",
        "condition_A_short_exactly_2": {
            "mean_iou": res["mean_iou"], "mean_ari": res["mean_ari"],
            "avg_words_per_desc": res["avg_words_per_desc"], "k_dist": res["k_dist"],
        },
        "baselines_ep10_exactly_2": {
            "structured_10_15_words": {"mean_iou": 0.6757, "mean_ari": 0.5023, "avg_words": "~15"},
            "structured_no_length_cap": {"mean_iou": 0.6052, "mean_ari": 0.3754, "avg_words": "~18"},
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*70}\n  RESULTS: ep10 structured prompt, exactly_2, 3-6 words cap\n{'='*70}")
    print(f"  {'Variant':<30} {'mIoU':>8}  {'mARI':>8}  {'avg_words':>10}")
    print(f"  {'-'*70}")
    print(f"  {'10-15 words (baseline)':<30} {0.6757:>8.4f}  {0.5023:>8.4f}  {'~15':>10}")
    print(f"  {'no length cap':<30} {0.6052:>8.4f}  {0.3754:>8.4f}  {'~18':>10}")
    print(f"  {'3-6 words (NEW)':<30} {res['mean_iou']:>8.4f}  "
          f"{res['mean_ari']:>8.4f}  {res['avg_words_per_desc']:>10.1f}")
    print(f"\n  Delta vs 10-15 baseline: mIoU {res['mean_iou']-0.6757:+.4f} "
          f"mARI {res['mean_ari']-0.5023:+.4f}")
    print(f"  Output: {out}")


if __name__ == "__main__":
    main()
