#!/usr/bin/env python3
"""
RWTD ablation: OBJECT-FRIENDLY user prompt vs current texture-only prompt.

Hypothesis: RWTD images include objects-with-textures (not pure textures like
ADE20K-DeTexture). Our training prompt explicitly instructs Qwen to treat
regions as "surfaces/areas, not single objects", which may be leaving RWTD
performance on the table because the GT regions ARE often objects.

This script evaluates best.pt (ep7) under a more permissive prompt that
allows either "continuous material surface OR prominent foreground element".
Two k-conditions per prompt to disentangle prompt framing from Hungarian
inflation effects.

Saves per-condition JSON including full Qwen-generated text for each
sample for qualitative comparison vs GT descriptions.

Usage:
    python scripts/ablation_object_friendly_prompt.py \
        --checkpoint checkpoints/best.pt \
        --output-dir checkpoints/ablation_object_friendly
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
from data.dataset import DeTextureDataset, DeTextureCollator, SYSTEM_PROMPT
from training.utils import load_config, load_checkpoint
from training.monitor import _compute_ari
from scipy.optimize import linear_sum_assignment


# ===================================================================== #
#  Object-friendly user prompt (differs only in the IMPORTANT paragraph) #
# ===================================================================== #

OBJECT_FRIENDLY_PROMPT_TEMPLATE = (
    "This image contains exactly {N} main visually distinct regions separated "
    "by boundaries (for example, contrasting materials, surfaces, or textures).\n\n"
    "Write a single, highly descriptive phrase (approximately 10-15 words) for "
    "each region. Include the following precise information:\n"
    "1. Semantic Name: A natural, common name for the material or surface.\n"
    "2. Distinct Visual Features: The core visual attributes like color, pattern, "
    "or texture that strongly contrast with the other regions.\n"
    "3. Spatial Context: A brief note on its general position (e.g., 'foreground', "
    "'background', 'top-left', 'bottom-right', 'center', 'top-right', 'bottom-left').\n\n"
    "IMPORTANT: Identify the {N} most prominent visually distinct regions. A region "
    "can be a continuous material surface OR a prominent, distinct foreground element "
    "that occupies a clear area. Describe the visual texture and pattern of each region.\n\n"
    "Format your response exactly like this:\n"
    "TEXTURE_1: Texture of <description>\n"
    "TEXTURE_2: Texture of <description>\n"
    "...\n"
    "TEXTURE_{N}: Texture of <description>"
)


class ObjectFriendlyCollator(DeTextureCollator):
    """Uses the object-friendly prompt instead of the training-time prompt."""

    def __init__(self, processor, n_value: str):
        super().__init__(processor, inference=True)
        self.n_value = n_value
        self.user_prompt = OBJECT_FRIENDLY_PROMPT_TEMPLATE.format(N=n_value)

    def __call__(self, samples):
        texts, images = [], []
        for s in samples:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
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
    ious, aris, k_preds_list = [], [], []
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
                      f"mARI={np.mean(aris):.4f}", flush=True)

    mean_iou = float(np.mean(ious))
    mean_ari = float(np.mean(aris))
    k_dist = Counter(k_preds_list)

    print(f"\n  [{label}] RESULTS:")
    print(f"    mIoU: {mean_iou:.4f}  mARI: {mean_ari:.4f}")
    print(f"    k_pred distribution: {dict(sorted(k_dist.items()))}")
    print(f"    Samples > 0.7: {sum(1 for x in ious if x > 0.7)}/{len(ious)}")
    print(f"    Samples < 0.3: {sum(1 for x in ious if x < 0.3)}/{len(ious)}")

    return {
        "label": label,
        "mean_iou": mean_iou,
        "mean_ari": mean_ari,
        "k_dist": {int(k): int(v) for k, v in k_dist.items()},
        "per_sample": per_sample,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/detexture.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--test-metadata",
                        default="/home/aviad/datasets/RWTD/metadata.json")
    parser.add_argument("--output-dir",
                        default="checkpoints/ablation_object_friendly")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    cfg = load_config(args.config)
    device = torch.device("cuda")

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

    # Save the prompt text itself for the record
    (output_dir / "prompt_template.txt").write_text(OBJECT_FRIENDLY_PROMPT_TEMPLATE)

    # ---- Condition A: object-friendly + "1 to 6" ----------------------- #
    print(f"\n{'='*70}")
    print(f"  CONDITION A: object-friendly prompt, N='1 to 6'")
    print(f"{'='*70}")
    collator_a = ObjectFriendlyCollator(model.processor, n_value="1 to 6")
    loader_a = torch.utils.data.DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=collator_a,
    )
    res_a = evaluate(model, loader_a, device, "object_friendly_1_to_6")
    res_a["checkpoint"] = str(args.checkpoint)
    res_a["prompt_template"] = OBJECT_FRIENDLY_PROMPT_TEMPLATE
    res_a["n_value"] = "1 to 6"
    with open(output_dir / "condition_A_object_friendly_1_to_6.json", "w") as f:
        json.dump(res_a, f, indent=2, ensure_ascii=False)

    # ---- Condition B: object-friendly + "exactly 2" -------------------- #
    print(f"\n{'='*70}")
    print(f"  CONDITION B: object-friendly prompt, N='2' (exactly 2)")
    print(f"{'='*70}")
    collator_b = ObjectFriendlyCollator(model.processor, n_value="2")
    loader_b = torch.utils.data.DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=collator_b,
    )
    res_b = evaluate(model, loader_b, device, "object_friendly_exactly_2")
    res_b["checkpoint"] = str(args.checkpoint)
    res_b["prompt_template"] = OBJECT_FRIENDLY_PROMPT_TEMPLATE
    res_b["n_value"] = "2"
    with open(output_dir / "condition_B_object_friendly_exactly_2.json", "w") as f:
        json.dump(res_b, f, indent=2, ensure_ascii=False)

    # ---- Summary -------------------------------------------------------- #
    summary = {
        "checkpoint": str(args.checkpoint),
        "n_samples": len(res_a["per_sample"]),
        "prompt_change": "Replaced training-time 'NOT individual objects' directive with 'can be material surface OR prominent foreground element'",
        "condition_A_object_friendly_1_to_6": {
            "mean_iou": res_a["mean_iou"],
            "mean_ari": res_a["mean_ari"],
            "k_dist": res_a["k_dist"],
        },
        "condition_B_object_friendly_exactly_2": {
            "mean_iou": res_b["mean_iou"],
            "mean_ari": res_b["mean_ari"],
            "k_dist": res_b["k_dist"],
        },
        "baseline_from_previous_ablation_texture_only_ep7": {
            "standard_1_to_6": {"mean_iou": 0.7394, "mean_ari": 0.5764},
            "exactly_2":       {"mean_iou": 0.6943, "mean_ari": 0.5212},
        },
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  ABLATION: Object-Friendly Prompt on RWTD (ep7 / best.pt)")
    print(f"{'='*70}")
    print(f"  {'Condition':<35} {'mIoU':>8}  {'mARI':>8}")
    print(f"  {'-'*70}")
    print(f"  {'Prev ablation: texture-only 1to6':<35} "
          f"{0.7394:>8.4f}  {0.5764:>8.4f}  (baseline)")
    print(f"  {'Prev ablation: texture-only ex2':<35} "
          f"{0.6943:>8.4f}  {0.5212:>8.4f}  (baseline)")
    print(f"  {'NEW: object-friendly 1to6':<35} "
          f"{res_a['mean_iou']:>8.4f}  {res_a['mean_ari']:>8.4f}")
    print(f"  {'NEW: object-friendly exactly_2':<35} "
          f"{res_b['mean_iou']:>8.4f}  {res_b['mean_ari']:>8.4f}")
    print(f"\n  Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
