#!/usr/bin/env bash
set -euo pipefail
cd /home/aviad/Qwen2SAM_DeTexture

OUT_ROOT=/home/aviad/Qwen2SAM_DeTexture/checkpoints/ablation_ep5_vs_ep7
mkdir -p "$OUT_ROOT"

echo "=== Eval 1/2: epoch_5.pt ===" | tee "$OUT_ROOT/progress.log"
date | tee -a "$OUT_ROOT/progress.log"
python scripts/ablation_exact_k2_rwtd.py \
    --checkpoint checkpoints/epoch_5.pt \
    --output-dir "$OUT_ROOT/ep5" \
    2>&1 | tee "$OUT_ROOT/ep5.log"
echo "=== Eval 1/2 DONE ===" | tee -a "$OUT_ROOT/progress.log"
date | tee -a "$OUT_ROOT/progress.log"

echo "=== Eval 2/2: best.pt (ep7) ===" | tee -a "$OUT_ROOT/progress.log"
date | tee -a "$OUT_ROOT/progress.log"
python scripts/ablation_exact_k2_rwtd.py \
    --checkpoint checkpoints/best.pt \
    --output-dir "$OUT_ROOT/ep7" \
    2>&1 | tee "$OUT_ROOT/ep7.log"
echo "=== Eval 2/2 DONE ===" | tee -a "$OUT_ROOT/progress.log"
date | tee -a "$OUT_ROOT/progress.log"

echo "ALL_DONE" >> "$OUT_ROOT/progress.log"
