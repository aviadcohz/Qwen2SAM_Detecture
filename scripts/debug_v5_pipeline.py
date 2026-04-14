#!/usr/bin/env python3
"""
V5 Pipeline Debugger: 3-step diagnostic to isolate the inference failure.

Step 1: Linguistic Sanity — does the LoRA generate coherent text with <|seg|>?
Step 2: GT Bypass — train forward (Run A) vs inference Pass 2 (Run B) on same GT text
Step 3: Vector Parity — are the 256-D queries identical between Run A and Run B?

Usage:
    cd /home/aviad/Qwen2SAM_DeTexture
    python scripts/debug_v5_pipeline.py --checkpoint checkpoints/best.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from models.qwen2sam_detexture import (
    Qwen2SAMDeTexture, MAX_TEXTURES, NUM_QUERY_SLOTS, SEG_TOKEN,
)
from data.dataset import (
    DeTextureDataset, DeTextureCollator, SYSTEM_PROMPT, TRAIN_USER_PROMPT,
    build_assistant_text, create_labels,
)
from training.utils import load_config, load_checkpoint
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


def masks_from_output(out, index_masks):
    """Convert model output to predicted mask and compute mIoU."""
    ml = out["mask_logits"]
    pm = out["pad_mask"]
    kp = int(out["k_preds"][0].item())

    m = ml.clone()
    inf_m = pm.unsqueeze(-1).unsqueeze(-1).expand_as(m)
    m[inf_m] = float("-inf")
    H, W = index_masks.shape[1], index_masks.shape[2]
    if m.shape[2] != H:
        m = F.interpolate(m.float(), size=(H, W), mode="bilinear",
                          align_corners=False)
    pred = m[0].argmax(dim=0).cpu().numpy()
    gt = index_masks[0].numpy()
    kg = int(index_masks.max().item())
    miou = compute_miou(pred, gt, kp, kg)
    return miou, kp, pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/detexture.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--metadata",
                        default="/home/aviad/datasets/ADE20k_DeTexture/metadata.json")
    parser.add_argument("--n-samples", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda")

    print("Building V5 model...")
    model = Qwen2SAMDeTexture(cfg, device="cuda")
    print(f"Loading {args.checkpoint}...")
    load_checkpoint(model, None, args.checkpoint, device="cuda")
    model.eval()

    # Load dataset
    with open(args.metadata) as f:
        all_meta = json.load(f)

    ds = DeTextureDataset(
        args.metadata, image_size=1008, augment=False,
        qwen_gt_embeds_path=cfg["data"].get("qwen_gt_embeds_path"),
    )
    # Inference collator (no assistant text — for generation)
    infer_collator = DeTextureCollator(model.processor, inference=True)
    # Training collator (with assistant text — for teacher forcing)
    train_collator = DeTextureCollator(model.processor, inference=False)

    indices = list(range(0, len(ds), max(1, len(ds) // args.n_samples)))[:args.n_samples]

    for sample_idx in indices:
        meta = all_meta[sample_idx]
        sample = ds[sample_idx]
        k_gt = sample["k_gt"]
        descriptions = sample["descriptions"]

        print(f"\n{'='*70}")
        print(f"  SAMPLE #{sample_idx}  k_gt={k_gt}")
        print(f"  Image: {meta['image_path']}")
        print(f"  GT descriptions:")
        for i, d in enumerate(descriptions):
            print(f"    TEX_{i+1}: {d[:80]}")
        print(f"{'='*70}")

        # ============================================================ #
        #  STEP 1: Linguistic Sanity Check (Pass 1 Generation)          #
        # ============================================================ #
        print(f"\n  --- STEP 1: Linguistic Sanity (Pass 1 Generation) ---")

        infer_batch = infer_collator([sample])
        qi = {k: v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in infer_batch["qwen_inputs"].items()}
        sam_images = infer_batch["sam_images"].to(device)
        index_masks = infer_batch["index_masks"]

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            # Just run Pass 1 generation
            gen_out = model.qwen.generate(
                **qi,
                max_new_tokens=300,
                output_hidden_states=False,
                return_dict_in_generate=True,
                do_sample=False,
            )
            generated_ids = gen_out.sequences
            prompt_len = qi["input_ids"].shape[1]
            gen_text = model.processor.tokenizer.decode(
                generated_ids[0, prompt_len:], skip_special_tokens=False,
            )

        has_seg = SEG_TOKEN in gen_text
        n_seg = gen_text.count(SEG_TOKEN)
        print(f"  Generated text ({len(gen_text)} chars):")
        # Print line by line
        for line in gen_text.strip().split("\n")[:8]:
            print(f"    {line[:100]}")
        print(f"  Contains <|seg|>: {has_seg} (count: {n_seg})")
        print(f"  Contains <|im_end|>: {'<|im_end|>' in gen_text}")

        # ============================================================ #
        #  STEP 2: GT Bypass — Run A (training forward) vs Run B        #
        #          (inference Pass 2)                                    #
        # ============================================================ #
        print(f"\n  --- STEP 2: GT Bypass (train forward vs inference Pass 2) ---")

        # Build training batch with GT text (teacher forcing)
        train_batch = train_collator([sample])
        tqi = {k: v.to(device) if isinstance(v, torch.Tensor) else v
               for k, v in train_batch["qwen_inputs"].items()}
        t_sam = train_batch["sam_images"].to(device)
        t_masks = train_batch["index_masks"]

        # --- Run A: Training forward() with block-diagonal mask ---
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            try:
                out_a = model.forward(
                    qwen_inputs=tqi,
                    sam_images=t_sam,
                    seg_grad_to_lm=False,
                )
                miou_a, kp_a, _ = masks_from_output(out_a, t_masks)
                q256_a = out_a["query_256"].float().cpu()  # (1, 7, 256)
                print(f"  Run A (train forward): mIoU={miou_a:.4f}  k_pred={kp_a}")
            except Exception as e:
                print(f"  Run A FAILED: {type(e).__name__}: {e}")
                miou_a, kp_a, q256_a = None, None, None

        # --- Run B: Inference Pass 2 logic on GT text ---
        # Reconstruct the full sequence (prompt + GT assistant text)
        # and run through inference_forward's Pass 2 extraction
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            try:
                # The training input_ids already contain the full sequence
                # (prompt + GT assistant with <|seg|>). We can use it directly
                # for Pass 2 extraction.
                full_ids = tqi["input_ids"]  # (1, L) — full teacher-forced seq

                # Build block-diagonal mask
                custom_mask = model.create_independent_texture_mask(full_ids)

                # Pre-compute position_ids from 2D mask
                attn_2d = tqi.get("attention_mask")
                qwen_vl_model = model.qwen.base_model.model.model
                position_ids, _ = qwen_vl_model.get_rope_index(
                    full_ids,
                    tqi.get("image_grid_thw"),
                    tqi.get("video_grid_thw"),
                    attn_2d,
                )

                # Run forward with custom mask + position_ids
                pass2_inputs = {k: v for k, v in tqi.items()
                                if k != "attention_mask" and k != "labels"}
                pass2_inputs["attention_mask"] = custom_mask
                pass2_inputs["position_ids"] = position_ids

                pass2_out = model.qwen(**pass2_inputs, output_hidden_states=True)
                hidden_b = pass2_out.hidden_states[-1]  # (1, L, 4096)

                # Extract [SEG] hidden states
                seg_embeds_b, k_preds_b = model.extract_seg_hidden_states(
                    hidden_b, full_ids,
                )

                # Build queries + project
                query_embeds_b, pad_mask_b = model.build_query_slots(
                    seg_embeds_b, k_preds_b,
                )
                q256_b = model.projector(query_embeds_b).float().cpu()

                # Run SAM
                with torch.no_grad():
                    backbone_out = model.sam3.backbone.forward_image(t_sam)
                    backbone_out["img_batch_all_stages"] = t_sam
                mask_logits_b = model.run_sam3_semantic(
                    backbone_out, q256_b.to(device).to(torch.bfloat16), pad_mask_b,
                )

                out_b = {
                    "mask_logits": mask_logits_b,
                    "pad_mask": pad_mask_b,
                    "k_preds": k_preds_b,
                    "query_256": q256_b,
                }
                miou_b, kp_b, _ = masks_from_output(out_b, t_masks)
                print(f"  Run B (Pass 2 logic):  mIoU={miou_b:.4f}  k_pred={kp_b}")
            except Exception as e:
                import traceback
                print(f"  Run B FAILED: {type(e).__name__}: {e}")
                traceback.print_exc()
                miou_b, kp_b, q256_b = None, None, None

        # Diagnosis
        if miou_a is not None and miou_b is not None:
            print(f"\n  DIAGNOSIS:")
            if miou_a > 0.5 and miou_b > 0.5:
                print(f"    Both healthy → weights good, inference logic good")
            elif miou_a > 0.5 and miou_b < 0.2:
                print(f"    Run A healthy, Run B garbage → BUG in Pass 2 logic")
                print(f"    (position_ids / 4D mask reconstruction is broken)")
            elif miou_a < 0.2 and miou_b < 0.2:
                print(f"    Both garbage → trained WEIGHTS have collapsed")
            elif miou_a < 0.2 and miou_b > 0.5:
                print(f"    Run A garbage, Run B healthy → BUG in train forward()")
            else:
                print(f"    Ambiguous — Run A={miou_a:.4f}, Run B={miou_b:.4f}")

        # ============================================================ #
        #  STEP 3: Vector Parity Check (256-D queries)                  #
        # ============================================================ #
        print(f"\n  --- STEP 3: Vector Parity (256-D query comparison) ---")

        if q256_a is not None and q256_b is not None:
            # Compare each query slot
            for slot in range(min(k_gt + 1, NUM_QUERY_SLOTS)):
                va = q256_a[0, slot]
                vb = q256_b[0, slot]
                cos = F.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0)).item()
                l2 = (va - vb).norm().item()
                slot_name = "DUSTBIN" if slot == 0 else f"SEG_{slot}"
                print(f"    {slot_name:>10}: cos={cos:.6f}  L2={l2:.4f}  "
                      f"norm_A={va.norm():.2f}  norm_B={vb.norm():.2f}")

            # Overall parity
            active_slots = min(k_gt + 1, NUM_QUERY_SLOTS)
            all_cos = []
            for s in range(active_slots):
                c = F.cosine_similarity(
                    q256_a[0, s].unsqueeze(0), q256_b[0, s].unsqueeze(0),
                ).item()
                all_cos.append(c)
            mean_cos = np.mean(all_cos)
            print(f"\n    Mean parity (active slots): cos={mean_cos:.6f}")
            if mean_cos > 0.999:
                print(f"    → IDENTICAL (math is correct, bug is elsewhere)")
            elif mean_cos > 0.95:
                print(f"    → CLOSE but not identical (numerical precision drift)")
            elif mean_cos > 0.5:
                print(f"    → DIFFERENT (Pass 2 logic produces different hidden states)")
            else:
                print(f"    → COMPLETELY DIFFERENT (fundamental bug in Pass 2)")
        else:
            print(f"    SKIPPED — one or both runs failed")

    print(f"\n{'='*70}")
    print(f"  DEBUG COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
