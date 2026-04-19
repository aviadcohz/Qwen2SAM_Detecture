#!/usr/bin/env python3
"""
RWTD ablation: structured prompt WITHOUT the '(approximately 10-15 words)'
length constraint, on epoch_10.pt only.

Hypothesis: GT RWTD descriptions are ~4-6 words (e.g. 'Texture of smooth
shell surface'). Training taught the projector to condition [SEG] hidden
states on short GT-style contexts. The '10-15 words' directive in the
inference prompt pushes Qwen to produce longer phrases, shifting the
[SEG] linguistic context away from the training distribution. Removing
the word-count cap should let Qwen default to more natural (likely
shorter) phrasings closer to the trained distribution.

Single minimal-surgery change vs the structured prompt: delete the
'(approximately 10-15 words) ' parenthetical. Everything else identical.

Usage:
    python scripts/ablation_no_length_constraint_ep10.py \
        --checkpoint checkpoints/epoch_10.pt \
        --output-dir checkpoints/ablation_no_length_ep10
"""

import argparse
import json
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


# ===================================================================== #
#  Prompts — structured prompt minus the '(approximately 10-15 words)'    #
# ===================================================================== #

STRUCTURED_SYSTEM_PROMPT = (
    "You analyze surface textures in images. Always respond in the exact "
    "format requested, with no extra text."
)

CONDITION_A_USER_PROMPT = (
    "This image contains exactly TWO main visually distinct regions separated "
    "by a boundary (for example, a prominent foreground object and its "
    "background, or two contrasting materials).\n\n"
    "Write a single, highly descriptive phrase for each of the two regions. "
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

CONDITION_B_USER_PROMPT = (
    "This image contains between 1 and 6 main visually distinct regions "
    "separated by a boundary (for example, a prominent foreground object "
    "and its background, or two contrasting materials).\n\n"
    "Write a single, highly descriptive phrase for each region. Include "
    "the following precise information:\n"
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
    "TEXTURE_2: Texture of <description>\n"
    "...\n"
    "TEXTURE_N: Texture of <description>"
)


class StructuredNoLengthCollator(DeTextureCollator):
    """Custom system+user prompt; no word-count cap."""

    def __init__(self, processor, system_prompt: str, user_prompt: str):
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
                out = model.inference_forward(
                    qwen_inputs=qwen_inputs, sam_images=sam_images,
                )

            mask_logits = out["mask_logits"]
            pad_mask = out["pad_mask"]
            k_pred = int(out["k_preds"][0].item())
            gen_text = out.get("generated_text", [""])[0]
            k_preds_list.append(k_pred)

            # Count words per TEXTURE_N: line (excluding the "TEXTURE_N: Texture of " header)
            import re as _re
            for line in gen_text.splitlines():
                m = _re.match(r"TEXTURE_\d+:\s*(?:Texture of\s*)?(.+?)\s*(?:<\|[^|]*\|>)?$",
                              line.strip())
                if m:
                    desc = m.group(1).strip()
                    gen_word_counts.append(len(desc.split()))

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
                "idx": idx,
                "k_gt": k_gt,
                "k_pred": k_pred,
                "miou": float(miou),
                "ari": float(ari),
                "generated_text": gen_text,
            })

            if (idx + 1) % 50 == 0:
                print(f"    {idx+1}/{len(loader)} avg mIoU={np.mean(ious):.4f} "
                      f"mARI={np.mean(aris):.4f} "
                      f"avg_words/desc={np.mean(gen_word_counts):.1f}", flush=True)

    mean_iou = float(np.mean(ious))
    mean_ari = float(np.mean(aris))
    k_dist = Counter(k_preds_list)
    avg_words = float(np.mean(gen_word_counts)) if gen_word_counts else 0.0

    print(f"\n  [{label}] RESULTS:")
    print(f"    mIoU: {mean_iou:.4f}  mARI: {mean_ari:.4f}")
    print(f"    avg words per description: {avg_words:.2f}  (GT baseline: ~4-6)")
    print(f"    k_pred distribution: {dict(sorted(k_dist.items()))}")
    print(f"    Samples > 0.7: {sum(1 for x in ious if x > 0.7)}/{len(ious)}")
    print(f"    Samples < 0.3: {sum(1 for x in ious if x < 0.3)}/{len(ious)}")

    return {
        "label": label,
        "mean_iou": mean_iou,
        "mean_ari": mean_ari,
        "avg_words_per_desc": avg_words,
        "k_dist": {int(k): int(v) for k, v in k_dist.items()},
        "per_sample": per_sample,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/detexture.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/epoch_10.pt")
    parser.add_argument("--test-metadata",
                        default="/home/aviad/datasets/RWTD/metadata.json")
    parser.add_argument("--output-dir",
                        default="checkpoints/ablation_no_length_ep10")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    cfg = load_config(args.config)
    device = torch.device("cuda")

    (output_dir / "prompts.txt").write_text(
        f"=== SYSTEM ===\n{STRUCTURED_SYSTEM_PROMPT}\n\n"
        f"=== CONDITION A (exactly 2, no length cap) ===\n{CONDITION_A_USER_PROMPT}\n"
    )

    print("Building model...")
    model = Qwen2SAMDeTexture(cfg, device="cuda")
    print(f"Loading checkpoint: {args.checkpoint}")
    load_checkpoint(model, None, args.checkpoint, device="cuda")
    model.eval()

    test_ds = DeTextureDataset(
        args.test_metadata,
        image_size=cfg["data"].get("image_size", 1008),
        augment=False,
        qwen_gt_embeds_path=cfg["data"].get("qwen_gt_embeds_path"),
    )
    print(f"Test dataset: {len(test_ds)} samples")

    # Condition A ONLY (winner of the prior structured-prompt ablation)
    print(f"\n{'='*70}")
    print(f"  CONDITION A: structured prompt (no length cap), 'exactly TWO'")
    print(f"{'='*70}")
    collator_a = StructuredNoLengthCollator(
        model.processor,
        system_prompt=STRUCTURED_SYSTEM_PROMPT,
        user_prompt=CONDITION_A_USER_PROMPT,
    )
    loader_a = torch.utils.data.DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=collator_a,
    )
    res_a = evaluate(model, loader_a, device, "A_no_length_exactly_2")
    res_a["checkpoint"] = str(args.checkpoint)
    with open(output_dir / "condition_A_no_length_exactly_2.json", "w") as f:
        json.dump(res_a, f, indent=2, ensure_ascii=False)

    # Summary
    summary = {
        "checkpoint": str(args.checkpoint),
        "n_samples": len(res_a["per_sample"]),
        "change_from_structured": "Removed the '(approximately 10-15 words)' parenthetical",
        "condition_A_no_length_exactly_2": {
            "mean_iou": res_a["mean_iou"],
            "mean_ari": res_a["mean_ari"],
            "avg_words_per_desc": res_a["avg_words_per_desc"],
            "k_dist": res_a["k_dist"],
        },
        "baseline_structured_ep10_with_length_cap_exactly_2": {
            "mean_iou": 0.6757,
            "mean_ari": 0.5023,
        },
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  RESULTS: ep10 structured prompt, exactly_2, no length cap")
    print(f"{'='*70}")
    print(f"  {'Condition':<35} {'mIoU':>8}  {'mARI':>8}  {'avg_words':>10}")
    print(f"  {'-'*70}")
    print(f"  {'baseline (with length cap)':<35} "
          f"{0.6757:>8.4f}  {0.5023:>8.4f}  {'~15':>10}")
    print(f"  {'NEW: no length cap':<35} "
          f"{res_a['mean_iou']:>8.4f}  {res_a['mean_ari']:>8.4f}  "
          f"{res_a['avg_words_per_desc']:>10.1f}")
    delta_miou = res_a["mean_iou"] - 0.6757
    delta_mari = res_a["mean_ari"] - 0.5023
    print(f"\n  Delta: mIoU {delta_miou:+.4f}, mARI {delta_mari:+.4f}")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
