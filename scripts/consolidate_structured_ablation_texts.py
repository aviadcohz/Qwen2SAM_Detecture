#!/usr/bin/env python3
"""Consolidate structured-prompt ablation texts: GT + both conditions per ckpt.

For each checkpoint (ep10, ep18), produce a JSON keyed by RWTD image_id with:
  - gt_descriptions         : list of GT texture descriptions
  - pred_exactly_2          : list of Qwen-generated descriptions (Condition A)
  - pred_1_to_6             : list of Qwen-generated descriptions (Condition B)
  - per-sample mIoU / mARI / k_pred / k_gt from each condition
  - idx + image_path for cross-referencing

Runs after the background ablation (b22t0tc9c) finishes writing per-condition JSONs.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path("/home/aviad/Qwen2SAM_DeTexture")
ABLATION_DIR = ROOT / "checkpoints/ablation_structured_prompt"
META = Path("/home/aviad/datasets/RWTD/metadata.json")

TAGS = ["epoch_10", "best"]  # best = epoch 18 (V6 final)
TEX_RE = re.compile(r"^\s*TEXTURE_(\d+)\s*:\s*(.+?)\s*$")


def parse_textures(gen_text: str) -> list[str]:
    if not gen_text:
        return []
    cleaned = re.sub(r"<\|[^|]*\|>", "", gen_text)
    pairs = []
    for line in cleaned.splitlines():
        m = TEX_RE.match(line)
        if m:
            pairs.append((int(m.group(1)), m.group(2).strip()))
    pairs.sort(key=lambda p: p[0])
    return [desc for _, desc in pairs]


def consolidate_one_checkpoint(tag: str, metadata: list[dict]) -> Path:
    path_a = ABLATION_DIR / f"{tag}_condition_A_exactly_2.json"
    path_b = ABLATION_DIR / f"{tag}_condition_B_1_to_6.json"

    missing = [p for p in (path_a, path_b) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"[{tag}] Missing ablation outputs (is the run still in progress?):\n"
            + "\n".join(f"  - {p}" for p in missing)
        )

    cond_a = json.loads(path_a.read_text())
    cond_b = json.loads(path_b.read_text())
    a_by_idx = {s["idx"]: s for s in cond_a["per_sample"]}
    b_by_idx = {s["idx"]: s for s in cond_b["per_sample"]}

    n = len(metadata)
    assert len(cond_a["per_sample"]) == n == len(cond_b["per_sample"]), (
        f"[{tag}] length mismatch: meta={n} "
        f"A={len(cond_a['per_sample'])} B={len(cond_b['per_sample'])}"
    )

    out = {}
    for idx, meta in enumerate(metadata):
        img_id = meta["id"]
        sa, sb = a_by_idx[idx], b_by_idx[idx]
        out[img_id] = {
            "idx": idx,
            "image_path": meta["image_path"],
            "k_gt": len(meta["textures"]),
            "gt_descriptions": [t["description"] for t in meta["textures"]],
            "pred_exactly_2": parse_textures(sa["generated_text"]),
            "pred_1_to_6": parse_textures(sb["generated_text"]),
            "k_pred_exactly_2": sa["k_pred"],
            "k_pred_1_to_6": sb["k_pred"],
            "mIoU_exactly_2": sa["miou"],
            "mIoU_1_to_6": sb["miou"],
            "mARI_exactly_2": sa["ari"],
            "mARI_1_to_6": sb["ari"],
        }

    out_path = ABLATION_DIR / f"{tag}_consolidated_texts.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out_path


def main():
    metadata = json.loads(META.read_text())
    print(f"Loaded {len(metadata)} RWTD samples from metadata")

    for tag in TAGS:
        try:
            out_path = consolidate_one_checkpoint(tag, metadata)
            data = json.loads(out_path.read_text())
            kb = out_path.stat().st_size / 1024
            print(f"\n[{tag}] -> {out_path} ({kb:.1f} KB, {len(data)} samples)")

            # Show one sample
            first_id = next(iter(data))
            first = data[first_id]
            print(f"  sample id={first_id} (idx={first['idx']}, k_gt={first['k_gt']}):")
            print(f"    GT:")
            for d in first["gt_descriptions"]:
                print(f"      - {d}")
            print(f"    pred_exactly_2 (k_pred={first['k_pred_exactly_2']}, "
                  f"mIoU={first['mIoU_exactly_2']:.3f}):")
            for d in first["pred_exactly_2"]:
                print(f"      - {d}")
            print(f"    pred_1_to_6 (k_pred={first['k_pred_1_to_6']}, "
                  f"mIoU={first['mIoU_1_to_6']:.3f}):")
            for d in first["pred_1_to_6"]:
                print(f"      - {d}")
        except FileNotFoundError as e:
            print(f"\n[{tag}] SKIPPED — {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
