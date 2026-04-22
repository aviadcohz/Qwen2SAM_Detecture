#!/usr/bin/env python3
"""
Debug script — probe what Qwen actually generates for RWTD images after V7
training, before SAM ever sees it.

Addresses the question: "live inference returns 0 mIoU — is it because
Qwen isn't emitting <|seg|> tokens, because it's emitting garbage, or
because the pipeline downstream is broken?"

Runs live generation on the FIRST N RWTD images (default 5), with the
training-time prompt, and prints:
  - raw generated text (verbatim, no cleaning)
  - count of <|seg|> special tokens emitted
  - whether "TEXTURE_N:" pattern is present
  - token-by-token breakdown for the last sample

Also runs an aggregate sweep (default 20 samples) and reports:
  - % of samples that emit >= 1 <|seg|> token
  - distribution of k_seg_emitted across the batch
  - % of samples with a valid "TEXTURE_N:" prefix

Usage:
    python scripts/debug_qwen_generation.py \\
        --checkpoint checkpoints/best.pt \\
        --n-detailed 5 --n-aggregate 30
"""

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import torch
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from models.qwen2sam_detecture import Qwen2SAMDetecture, SEG_TOKEN
from data.dataset import (
    SYSTEM_PROMPT, TRAIN_USER_PROMPT, preprocess_image_for_sam3, SAM3_SIZE,
)
from training.utils import load_config, load_checkpoint

RWTD_IMAGES_DIR = Path("/home/aviad/datasets/RWTD/images")

TEXTURE_RE = re.compile(r"TEXTURE[_\s]*(\d+)\s*:", re.IGNORECASE)


def load_image(crop_name: str, image_size: int):
    img_path = RWTD_IMAGES_DIR / f"{crop_name}.jpg"
    bgr = cv2.imread(str(img_path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(rgb)


@torch.no_grad()
def run_generation(model, image_pil, device, max_new_tokens=300):
    """Run Qwen.generate() directly on a prompt+image; return raw string
    and token IDs."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": TRAIN_USER_PROMPT},
        ]},
    ]
    text = model.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    qwen_inputs = model.processor(
        text=[text], images=[image_pil], return_tensors="pt", padding=True,
    )
    qwen_inputs.pop("token_type_ids", None)
    qwen_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                   for k, v in qwen_inputs.items()}

    prompt_len = qwen_inputs["input_ids"].shape[1]

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        gen_out = model.qwen.generate(
            **qwen_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
        )

    gen_ids = gen_out.sequences[0, prompt_len:]
    gen_text = model.processor.tokenizer.decode(
        gen_ids, skip_special_tokens=False,
    )
    return gen_text, gen_ids, prompt_len


def analyze_generation(gen_text: str, gen_ids: torch.Tensor, seg_token_id: int):
    n_seg = int((gen_ids == seg_token_id).sum().item())
    n_texture_hits = len(TEXTURE_RE.findall(gen_text))
    has_im_end = "<|im_end|>" in gen_text
    return {
        "n_seg_tokens": n_seg,
        "n_texture_prefixes": n_texture_hits,
        "has_im_end": has_im_end,
        "generated_length": len(gen_ids),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/detecture.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--gt-json", default=str(_PROJECT_ROOT / "RWTD_GT.json"))
    p.add_argument("--n-detailed", type=int, default=5,
                   help="Samples to print verbose generation for.")
    p.add_argument("--n-aggregate", type=int, default=30,
                   help="Samples to sweep for aggregate stats.")
    p.add_argument("--max-new-tokens", type=int, default=300)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)
    image_size = cfg["data"].get("image_size", SAM3_SIZE)

    # ---------------- Load model ---------------- #
    print(f"Loading model from {args.checkpoint}...")
    model = Qwen2SAMDetecture(cfg, device=str(device))
    epoch_idx = load_checkpoint(model, None, args.checkpoint, device=str(device))
    print(f"  Checkpoint epoch_idx={epoch_idx} (displayed Epoch {epoch_idx + 1})")
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad = False

    seg_token_id = model.processor.tokenizer.convert_tokens_to_ids(SEG_TOKEN)
    print(f"  SEG_TOKEN='{SEG_TOKEN}' -> id {seg_token_id}")

    # Print the actual system+user prompt tokens being fed to Qwen
    print(f"\n{'='*70}")
    print(f"  SYSTEM PROMPT:")
    print(f"{'='*70}")
    print(SYSTEM_PROMPT)
    print(f"\n{'='*70}")
    print(f"  TRAIN_USER_PROMPT (used at inference — this is what Qwen sees):")
    print(f"{'='*70}")
    print(TRAIN_USER_PROMPT)

    # Load RWTD sample list
    with open(args.gt_json) as f:
        items = [it for it in json.load(f) if it.get("parse_ok", False)]

    # ================================================================= #
    # Detailed: print verbose output for first N samples
    # ================================================================= #
    print(f"\n\n{'#'*70}")
    print(f"#  DETAILED VIEW — first {args.n_detailed} samples")
    print(f"{'#'*70}")

    for i, it in enumerate(items[:args.n_detailed]):
        crop = it["crop_name"]
        print(f"\n{'='*70}")
        print(f"  SAMPLE {i+1}/{args.n_detailed}: crop_name={crop}")
        print(f"{'='*70}")

        image_pil = load_image(crop, image_size)
        gen_text, gen_ids, prompt_len = run_generation(
            model, image_pil, device, max_new_tokens=args.max_new_tokens,
        )
        stats = analyze_generation(gen_text, gen_ids, seg_token_id)

        print(f"  prompt_len={prompt_len}  generated_tokens={stats['generated_length']}")
        print(f"  n_seg_tokens_emitted: {stats['n_seg_tokens']}")
        print(f"  n_texture_prefixes:   {stats['n_texture_prefixes']}")
        print(f"  emitted <|im_end|>:   {stats['has_im_end']}")
        print(f"\n  GT descriptions (for reference):")
        print(f"    A: {it['parsed']['desc_a']}")
        print(f"    B: {it['parsed']['desc_b']}")
        print(f"\n  RAW GENERATED TEXT:")
        print(f"  {'-'*68}")
        # Indent the generated text for readability
        for line in gen_text.split("\n"):
            print(f"  |  {line}")
        print(f"  {'-'*68}")

    # ================================================================= #
    # Aggregate sweep
    # ================================================================= #
    print(f"\n\n{'#'*70}")
    print(f"#  AGGREGATE SWEEP — {args.n_aggregate} samples")
    print(f"{'#'*70}")

    seg_counts = []
    texture_counts = []
    n_samples = min(args.n_aggregate, len(items))

    for i, it in enumerate(items[:n_samples]):
        crop = it["crop_name"]
        image_pil = load_image(crop, image_size)
        gen_text, gen_ids, _ = run_generation(
            model, image_pil, device, max_new_tokens=args.max_new_tokens,
        )
        stats = analyze_generation(gen_text, gen_ids, seg_token_id)
        seg_counts.append(stats["n_seg_tokens"])
        texture_counts.append(stats["n_texture_prefixes"])
        if (i + 1) % 5 == 0:
            print(f"    {i+1}/{n_samples} avg_seg={sum(seg_counts)/(i+1):.2f} "
                  f"avg_tex={sum(texture_counts)/(i+1):.2f}")

    n_with_seg = sum(1 for c in seg_counts if c > 0)
    n_with_texture = sum(1 for c in texture_counts if c > 0)
    from collections import Counter
    seg_dist = Counter(seg_counts)
    tex_dist = Counter(texture_counts)

    print(f"\n{'='*70}")
    print(f"  AGGREGATE RESULTS ({n_samples} samples)")
    print(f"{'='*70}")
    print(f"  Samples emitting >=1 <|seg|> token:     "
          f"{n_with_seg}/{n_samples} ({100*n_with_seg/n_samples:.1f}%)")
    print(f"  Samples with >=1 'TEXTURE_N:' prefix:   "
          f"{n_with_texture}/{n_samples} ({100*n_with_texture/n_samples:.1f}%)")
    print(f"\n  Distribution of <|seg|> count per sample:")
    for k in sorted(seg_dist.keys()):
        print(f"    {k} SEG tokens: {seg_dist[k]} samples")
    print(f"\n  Distribution of 'TEXTURE_N:' count per sample:")
    for k in sorted(tex_dist.keys()):
        print(f"    {k} TEXTURE prefixes: {tex_dist[k]} samples")

    print(f"\n{'='*70}")
    print(f"  DIAGNOSIS GUIDE")
    print(f"{'='*70}")
    pct_seg = 100 * n_with_seg / n_samples
    pct_tex = 100 * n_with_texture / n_samples
    if pct_seg > 90:
        print(f"  SEG emission healthy ({pct_seg:.0f}%). Issue is downstream "
              f"(SAM path, regex fallback, or pair matching).")
    elif pct_tex > 90 and pct_seg < 30:
        print(f"  Qwen emits TEXTURE format ({pct_tex:.0f}%) but NOT <|seg|> "
              f"tokens ({pct_seg:.0f}%). Root cause: LM loss weight on SEG "
              f"position is 0 — Qwen has no gradient signal to learn SEG "
              f"emission. Fix by raising lm_weight at SEG positions.")
    elif pct_tex < 30:
        print(f"  Qwen not emitting TEXTURE format at all ({pct_tex:.0f}%). "
              f"Much deeper language issue — generations may be garbage, "
              f"OOD, or refusing to follow the instruction.")
    else:
        print(f"  Mixed signal: {pct_tex:.0f}% TEXTURE, {pct_seg:.0f}% SEG. "
              f"Partial learning — inspect DETAILED VIEW for patterns.")


if __name__ == "__main__":
    main()
