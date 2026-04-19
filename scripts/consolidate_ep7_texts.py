#!/usr/bin/env python3
"""Consolidate ep7 RWTD text outputs (GT + both prompt conditions) into one JSON."""

import json
import re
from pathlib import Path

ROOT = Path("/home/aviad/Qwen2SAM_DeTexture")
META = Path("/home/aviad/datasets/RWTD/metadata.json")
COND_A = ROOT / "checkpoints/ablation_ep5_vs_ep7/ep7/condition_A_1_to_6.json"
COND_B = ROOT / "checkpoints/ablation_ep5_vs_ep7/ep7/condition_B_exactly_2.json"
OUT = ROOT / "checkpoints/ablation_ep5_vs_ep7/ep7/consolidated_texts.json"

TEX_RE = re.compile(r"^\s*TEXTURE_(\d+)\s*:\s*(.+?)\s*$")


def parse_textures(gen_text: str) -> list[str]:
    """Extract TEXTURE_N: <desc> lines from a generation, ordered by N."""
    if not gen_text:
        return []
    # Drop special tokens that may trail the last line
    cleaned = re.sub(r"<\|[^|]*\|>", "", gen_text)
    pairs = []
    for line in cleaned.splitlines():
        m = TEX_RE.match(line)
        if m:
            pairs.append((int(m.group(1)), m.group(2).strip()))
    pairs.sort(key=lambda p: p[0])
    return [desc for _, desc in pairs]


def main():
    metadata = json.loads(META.read_text())
    cond_a = json.loads(COND_A.read_text())
    cond_b = json.loads(COND_B.read_text())

    a_by_idx = {s["idx"]: s for s in cond_a["per_sample"]}
    b_by_idx = {s["idx"]: s for s in cond_b["per_sample"]}

    assert len(metadata) == len(cond_a["per_sample"]) == len(cond_b["per_sample"]), (
        f"length mismatch: meta={len(metadata)} "
        f"A={len(cond_a['per_sample'])} B={len(cond_b['per_sample'])}"
    )

    out = {}
    for idx, meta in enumerate(metadata):
        img_id = meta["id"]
        gt = [t["description"] for t in meta["textures"]]
        sample_a = a_by_idx[idx]
        sample_b = b_by_idx[idx]
        out[img_id] = {
            "idx": idx,
            "image_path": meta["image_path"],
            "k_gt": len(gt),
            "gt_descriptions": gt,
            "pred_1_to_6": parse_textures(sample_a["generated_text"]),
            "pred_exactly_2": parse_textures(sample_b["generated_text"]),
            "mIoU_1_to_6": sample_a["miou"],
            "mIoU_exactly_2": sample_b["miou"],
            "k_pred_1_to_6": sample_a["k_pred"],
            "k_pred_exactly_2": sample_b["k_pred"],
        }

    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Wrote {len(out)} samples -> {OUT}")
    print(f"File size: {OUT.stat().st_size / 1024:.1f} KB")

    # Quick sanity sample
    first_id = next(iter(out))
    first = out[first_id]
    print(f"\nSample '{first_id}':")
    print(f"  GT ({len(first['gt_descriptions'])}):")
    for d in first["gt_descriptions"]:
        print(f"    - {d}")
    print(f"  pred_1_to_6 ({len(first['pred_1_to_6'])}):")
    for d in first["pred_1_to_6"]:
        print(f"    - {d}")
    print(f"  pred_exactly_2 ({len(first['pred_exactly_2'])}):")
    for d in first["pred_exactly_2"]:
        print(f"    - {d}")


if __name__ == "__main__":
    main()
