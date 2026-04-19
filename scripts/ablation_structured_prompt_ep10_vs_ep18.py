#!/usr/bin/env python3
"""
Definitive RWTD ablation: structured object-aware prompt on V6-Final checkpoints.

Tests ep10 (end of Stage 1) and ep18 (final V6 best.pt) under two k-conditions
with a structured prompt that explicitly allows 'prominent foreground object'
framing while keeping the training-compatible TEXTURE_N: / <|seg|> format.

Custom SYSTEM + USER prompts (both overridden vs training defaults):
  SYSTEM:  "You analyze surface textures in images. Always respond in the
            exact format requested, with no extra text."
  USER:    Condition-specific (see CONDITION_A_USER / CONDITION_B_USER below).

Output format keeps TEXTURE_1 ... TEXTURE_N (digit-based) so the model's
regex fallback and [SEG] emission habit remain intact.

Usage:
    python scripts/ablation_structured_prompt_ep10_vs_ep18.py \
        --checkpoints checkpoints/epoch_10.pt checkpoints/best.pt \
        --output-dir checkpoints/ablation_structured_prompt
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
#  Prompts                                                                #
# ===================================================================== #

STRUCTURED_SYSTEM_PROMPT = (
    "You analyze surface textures in images. Always respond in the exact "
    "format requested, with no extra text."
)

CONDITION_A_USER_PROMPT = (
    "This image contains exactly TWO main visually distinct regions separated "
    "by a boundary (for example, a prominent foreground object and its "
    "background, or two contrasting materials).\n\n"
    "Write a single, highly descriptive phrase (approximately 10-15 words) "
    "for each of the two regions. Include the following precise information:\n"
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
    "Write a single, highly descriptive phrase (approximately 10-15 words) "
    "for each region. Include the following precise information:\n"
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


class StructuredPromptCollator(DeTextureCollator):
    """Override both SYSTEM and USER prompts; keep training-compatible format."""

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


# ===================================================================== #
#  Metrics                                                                #
# ===================================================================== #

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


# ===================================================================== #
#  Driver                                                                 #
# ===================================================================== #

def run_checkpoint(ckpt_path: Path, cfg: dict, device, test_ds, out_dir: Path,
                   tag: str) -> dict:
    print(f"\n{'#'*72}")
    print(f"#  CHECKPOINT: {ckpt_path}  (tag={tag})")
    print(f"{'#'*72}")

    print("Building model...")
    model = Qwen2SAMDeTexture(cfg, device="cuda")
    load_checkpoint(model, None, str(ckpt_path), device="cuda")
    model.eval()

    results = {"checkpoint": str(ckpt_path), "tag": tag, "n_samples": len(test_ds)}

    # Condition A ---------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  [{tag}] CONDITION A: structured prompt, 'exactly TWO'")
    print(f"{'='*70}")
    collator_a = StructuredPromptCollator(
        model.processor,
        system_prompt=STRUCTURED_SYSTEM_PROMPT,
        user_prompt=CONDITION_A_USER_PROMPT,
    )
    loader_a = torch.utils.data.DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=collator_a,
    )
    res_a = evaluate(model, loader_a, device, f"{tag}_A_exactly_2")
    results["condition_A_exactly_2"] = {
        "mean_iou": res_a["mean_iou"],
        "mean_ari": res_a["mean_ari"],
        "k_dist": res_a["k_dist"],
    }
    (out_dir / f"{tag}_condition_A_exactly_2.json").write_text(
        json.dumps({**res_a, "checkpoint": str(ckpt_path)}, indent=2, ensure_ascii=False)
    )

    # Condition B ---------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  [{tag}] CONDITION B: structured prompt, '1 to 6'")
    print(f"{'='*70}")
    collator_b = StructuredPromptCollator(
        model.processor,
        system_prompt=STRUCTURED_SYSTEM_PROMPT,
        user_prompt=CONDITION_B_USER_PROMPT,
    )
    loader_b = torch.utils.data.DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=collator_b,
    )
    res_b = evaluate(model, loader_b, device, f"{tag}_B_1_to_6")
    results["condition_B_1_to_6"] = {
        "mean_iou": res_b["mean_iou"],
        "mean_ari": res_b["mean_ari"],
        "k_dist": res_b["k_dist"],
    }
    (out_dir / f"{tag}_condition_B_1_to_6.json").write_text(
        json.dumps({**res_b, "checkpoint": str(ckpt_path)}, indent=2, ensure_ascii=False)
    )

    # Free GPU before next checkpoint
    del model
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/detexture.yaml")
    parser.add_argument("--checkpoints", nargs="+",
                        default=["checkpoints/epoch_10.pt", "checkpoints/best.pt"])
    parser.add_argument("--test-metadata",
                        default="/home/aviad/datasets/RWTD/metadata.json")
    parser.add_argument("--output-dir",
                        default="checkpoints/ablation_structured_prompt")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    cfg = load_config(args.config)
    device = torch.device("cuda")

    # Save prompts for the record
    (output_dir / "prompts.txt").write_text(
        f"=== SYSTEM ===\n{STRUCTURED_SYSTEM_PROMPT}\n\n"
        f"=== CONDITION A (exactly 2) ===\n{CONDITION_A_USER_PROMPT}\n\n"
        f"=== CONDITION B (1 to 6) ===\n{CONDITION_B_USER_PROMPT}\n"
    )

    # Build the test dataset once (reused across checkpoints)
    test_ds = DeTextureDataset(
        args.test_metadata,
        image_size=cfg["data"].get("image_size", 1008),
        augment=False,
        qwen_gt_embeds_path=cfg["data"].get("qwen_gt_embeds_path"),
    )
    print(f"Test dataset: {len(test_ds)} samples")

    # Run each checkpoint --------------------------------------------------
    all_results = []
    for ckpt in args.checkpoints:
        ckpt_path = Path(ckpt)
        tag = ckpt_path.stem  # e.g. "epoch_10" or "best"
        res = run_checkpoint(ckpt_path, cfg, device, test_ds, output_dir, tag)
        all_results.append(res)

    # Summary --------------------------------------------------------------
    summary = {
        "n_samples": len(test_ds),
        "system_prompt": STRUCTURED_SYSTEM_PROMPT,
        "condition_A_user_prompt_preview": CONDITION_A_USER_PROMPT[:120] + "...",
        "condition_B_user_prompt_preview": CONDITION_B_USER_PROMPT[:120] + "...",
        "checkpoints": all_results,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # 4-row comparison table ----------------------------------------------
    print(f"\n{'='*78}")
    print(f"  DEFINITIVE STRUCTURED-PROMPT ABLATION on RWTD")
    print(f"{'='*78}")
    print(f"  {'Checkpoint':<18} {'Condition':<20} {'mIoU':>8}  {'mARI':>8}  {'k_dist':<20}")
    print(f"  {'-'*78}")
    for r in all_results:
        tag = r["tag"]
        a = r["condition_A_exactly_2"]
        b = r["condition_B_1_to_6"]
        kd_a = " ".join(f"{k}:{v}" for k, v in sorted(a["k_dist"].items()))
        kd_b = " ".join(f"{k}:{v}" for k, v in sorted(b["k_dist"].items()))
        print(f"  {tag:<18} {'exactly 2':<20} {a['mean_iou']:>8.4f}  {a['mean_ari']:>8.4f}  {kd_a}")
        print(f"  {tag:<18} {'1 to 6':<20} {b['mean_iou']:>8.4f}  {b['mean_ari']:>8.4f}  {kd_b}")
    print(f"\n  Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
