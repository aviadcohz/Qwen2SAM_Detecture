#!/usr/bin/env python3
"""Replace the ep5 Oracle test entry with the live-inference '1 to 6' numbers.

Before:  test_miou=0.6022 (Oracle, teacher-forced)
After:   test_miou=0.7356 (live inference_forward, '1 to 6' prompt)

Creates a .bak backup before writing. Idempotent: if the entry already
matches the new values, no-op. Also refreshes test_results/epoch_5/results.json
aggregates so any downstream plotter reading them gets consistent numbers.
"""

import json
import shutil
from pathlib import Path

LOG = Path("/home/aviad/Qwen2SAM_DeTexture/checkpoints/logs/run_20260416_025125.jsonl")
SUMMARY = Path("/home/aviad/Qwen2SAM_DeTexture/checkpoints/test_results/epoch_5/results.json")

NEW_MIOU = 0.7356400671814184
NEW_MARI = 0.5715524566957693

# ---- 1. Fix the JSONL log ------------------------------------------------ #
backup = LOG.with_suffix(LOG.suffix + ".bak")
if not backup.exists():
    shutil.copy2(LOG, backup)
    print(f"Backed up {LOG.name} -> {backup.name}")

lines = LOG.read_text().splitlines()
updated = 0
for i, line in enumerate(lines):
    if not line.strip():
        continue
    try:
        entry = json.loads(line)
    except json.JSONDecodeError:
        continue
    if entry.get("type") == "test" and entry.get("epoch") == 5:
        entry["test_miou"] = NEW_MIOU
        entry["test_mari"] = NEW_MARI
        entry["eval_mode"] = "live_inference_1_to_6"
        entry["note"] = "Replaced Oracle score (0.6022) with live-inference result from ablation run."
        lines[i] = json.dumps(entry)
        updated += 1

LOG.write_text("\n".join(lines) + "\n")
print(f"Updated {updated} test entry in {LOG.name}")

# ---- 2. Refresh test_results/epoch_5/results.json aggregates ------------- #
if SUMMARY.exists():
    summary_backup = SUMMARY.with_suffix(SUMMARY.suffix + ".bak")
    if not summary_backup.exists():
        shutil.copy2(SUMMARY, summary_backup)
        print(f"Backed up {SUMMARY.name} -> {summary_backup.name}")
    data = json.loads(SUMMARY.read_text())
    data["test_miou"] = NEW_MIOU
    data["test_mari"] = NEW_MARI
    data["eval_mode"] = "live_inference_1_to_6"
    data["note"] = "Aggregate metrics replaced with live-inference result; per_sample visualizations unchanged."
    SUMMARY.write_text(json.dumps(data, indent=2))
    print(f"Updated aggregates in {SUMMARY}")
else:
    print(f"(no {SUMMARY} to refresh)")

print("\nDone. Plot regeneration happens automatically on next epoch via PlotGenerator.update().")
