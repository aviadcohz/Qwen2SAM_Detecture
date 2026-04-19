# Qwen2SAM-DeTexture (V7)

**End-to-End VLM-Guided Multi-Texture Segmentation with [SEG] Token Grounding**

An E2E architecture that fuses a Vision-Language Model (**Qwen3-VL-8B**) with a Geometric Segmentation Engine (**SAM 3**) to segment images into 1-6 distinct texture regions. The model uses a dedicated **[SEG] token** to decouple visual grounding from language context, a **block-diagonal attention mask** to prevent cross-texture contamination, and the **V7 Bridge** — a trainable projection into SAM 3's *native* text-encoder space that then reuses SAM's own frozen 1024 → 256 resizer, so Qwen's semantic richness is shape-matched into SAM instead of being squashed through a learned bottleneck.

This README documents V7, the current architecture. See [ablation/](ablation/) for the full V1 → V7 evolution and the pathologies each version resolved.

---

## Table of Contents

- [Key Innovations](#key-innovations)
- [Architecture Overview](#architecture-overview)
- [Detailed Architecture](#detailed-architecture)
  - [Module A: Qwen3-VL with [SEG] Token](#module-a-qwen3-vl-with-seg-token)
  - [Module B: V7 Bridge + Frozen SAM Resizer](#module-b-v7-bridge--frozen-sam-resizer)
  - [Module C: SAM 3 with Batch Multiplexing](#module-c-sam-3-with-batch-multiplexing)
  - [Module D: Multi-Texture Mask Head](#module-d-multi-texture-mask-head)
- [The Dustbin Query](#the-dustbin-query)
- [Block-Diagonal Attention Mask](#block-diagonal-attention-mask)
- [Exponential LM Loss (V7)](#exponential-lm-loss-v7)
- [Training Process](#training-process)
  - [Two-Stage Curriculum](#two-stage-curriculum)
  - [Forward Pass Walkthrough](#forward-pass-walkthrough)
  - [Loss Functions](#loss-functions)
  - [Differential Learning Rates](#differential-learning-rates)
- [Inference (Two-Pass Strategy)](#inference-two-pass-strategy)
- [Ablation History (V1-V7)](#ablation-history-v1-v7)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Getting Started](#getting-started)

---

## Key Innovations

| Innovation | What it solves |
|---|---|
| **[SEG] Token Grounding** | Dedicated `<\|seg\|>` token after each texture description. Qwen's LoRA learns to pack visually-grounded spatial information into this token, decoupling it from noisy language context. |
| **V7 Bridge to SAM's Native Width** | `Linear(4096 → 1024) + LayerNorm + GELU + Dropout(0.4)` maps Qwen [SEG] into SAM 3's native 1024-D text-encoder space; SAM's *frozen* pretrained resizer (`Linear 1024 → 256`) handles the final projection. Replaces V6's 4096→512→256 bottleneck that was squashing Qwen's semantic richness. |
| **Block-Diagonal Attention Mask** | Prevents Context Leakage: TEXTURE_2's `<\|seg\|>` hidden state cannot attend to TEXTURE_1's tokens. Proven to reduce inter-texture cosine similarity from 0.74 to 0.16. |
| **Exponential LM Regularization (V7)** | Per-token weight `Weight(d) = 1 − e^(−2d)` where `d` = distance to nearest assistant-region `<\|seg\|>`. Weight at the SEG token is *strictly zero* (total geometric freedom); weight at d≥3 is ≈1.0 (full linguistic anchoring). Replaces V6's cosine schedule; eliminates the DICE ↔ LM tug-of-war. |
| **Batch Multiplexing** | Each query is fed to SAM independently in its own "slot 1" position. Eliminates SAM3's pretrained positional bias where only the first query slot received attention. |
| **Dustbin Query** | A learned embedding that absorbs non-texture pixels (objects, sky, etc.), preventing false texture assignments. |
| **Permanent SAM Freeze (V7)** | SAM stays 100% frozen. Zero-Shot SAM 3 with an object-aware prompt hit **0.928 mIoU** on RWTD — fine-tuning only destroys this OOD generality. V5/V6 confirmed this: every SAM LoRA unfreezing attempt produced an OOD regression. |
| **Two-Stage Curriculum (V7)** | (1) Bridge warmup (Qwen + SAM frozen), (2) Qwen LoRA thaw at 1e-6 with Bridge LR decayed 10×. No Stage 3. Removes the regression locus from V5/V6. |
| **Universal Object-Aware Prompt** | Same prompt for training and inference: "a prominent foreground object and its background, or contrasting materials" — matches the object-aware framing that unlocked the 0.928 ZS SAM result on RWTD. |
| **Two-Pass Inference** | Pass 1: standard causal generation (no repetition). Pass 2: block-diagonal masked forward (decoupled [SEG] extraction). Train-test parity guaranteed. |

---

## Architecture Overview

```
                        +-----------------------+
                        |      Input Image      |
                        +-----------+-----------+
                                    |
                  +-----------------+-----------------+
                  |                                   |
                  v                                   v
    +----------------------------+     +----------------------------+
    |  MODULE A: Qwen3-VL-8B     |     |  SAM3 Backbone (frozen)    |
    |  (LoRA r=8 on q,v)         |     |  Image Encoder             |
    |                            |     |  1008x1008 -> FPN features |
    |  Input: image + prompt     |     +-------------+--------------+
    |  Output: text descriptions |                   |
    |  with <|seg|> markers      |                   |
    +-------------+--------------+                   |
                  |                                  |
                  v                                  |
    +----------------------------+                   |
    |  Block-Diagonal Mask       |                   |
    |  Extract <|seg|> hidden    |                   |
    |  states (decoupled)        |                   |
    |                            |                   |
    |  K vectors of dim 4096     |                   |
    +-------------+--------------+                   |
                  |                                  |
                  v                                  |
    +----------------------------+                   |
    |  Build 7 Query Slots       |                   |
    |                            |                   |
    |  [DUSTBIN, SEG_1, ...,     |                   |
    |   SEG_K, PAD, ..., PAD]    |                   |
    |                            |                   |
    |  7 vectors of dim 4096     |                   |
    +-------------+--------------+                   |
                  |                                  |
                  v                                  |
    +----------------------------+                   |
    |  MODULE B: V7 Bridge       |                   |
    |  (4.2M trainable params)   |                   |
    |                            |                   |
    |  Linear(4096 -> 1024) + LN |                   |
    |   + GELU + Dropout(0.4)    |                   |
    +-------------+--------------+                   |
                  |                                  |
                  v                                  |
    +----------------------------+                   |
    |  SAM.resizer (FROZEN)      |                   |
    |  Linear(1024 -> 256)       |                   |
    |  reused from SAM3's own    |                   |
    |  language_backbone         |                   |
    +-------------+--------------+                   |
                  |                                  |
                  +----------------------------------+
                  |                                  |
                  v                                  v
    +-------------------------------------------------------+
    |  MODULE C: SAM3 — Batch Multiplexed (FROZEN)          |
    |  (B, 7, 256) -> flatten -> (B*7, 1, 256)              |
    |  Each query gets its own "slot 1" position            |
    |                                                       |
    |  Fusion Encoder  |  Cross-Attn  |  Pixel Decoder      |
    |  (all frozen — no LoRA in V7)                         |
    +----------------------------+--------------------------+
                                 |
                                 v
    +-------------------------------------------------------+
    |  MODULE D: Multi-Texture Mask Head (trainable)        |
    |                                                       |
    |  einsum(queries, pixel_embed) -> (B, 7, H, W)         |
    |                                                       |
    |  Channel 0: DUSTBIN     Channels 1-K: Textures        |
    |  Channels K+1 to 6: PAD (masked to -inf)              |
    +-------------------------------------------------------+
```

---

## Detailed Architecture

### Module A: Qwen3-VL with [SEG] Token

**Model**: `Qwen/Qwen3-VL-8B-Instruct`
**Status**: Frozen base weights + LoRA (r=8) on `q_proj` and `v_proj`.
**Training**: LoRA receives gradients through the spatial path (mask loss → SAM → Bridge → [SEG]) AND linguistic supervision via the V7 exponential LM cliff.

**Output format** (teacher-forced during training):
```
TEXTURE_1: Texture of rough mossy stone with granular surface in the foreground <|seg|>
TEXTURE_2: Texture of smooth flowing water with reflective sheen in the center <|seg|>
TEXTURE_3: Texture of dry sandy ground with ripple patterns on the right <|seg|>
```

The `<|seg|>` token is a dedicated grounding anchor. Unlike natural language end-of-line tokens (which absorb context from ALL prior tokens via causal attention), `<|seg|>` hidden states are computed under a **block-diagonal attention mask** that isolates each texture block from previous blocks. This produces clean, decoupled visual representations.

**V7 prompt change**: both SYSTEM and USER prompts include an object-aware example ("a prominent foreground object and its background, or contrasting materials"), and the format template explicitly shows `<|seg|>` per TEXTURE line. Because the prompt contains literal `<|seg|>` text which tokenises as the SEG special token, all seg-position lookups filter by the assistant-region boundary.

---

### Module B: V7 Bridge + Frozen SAM Resizer

**V7 replaces V6's bottleneck projector with a wider trainable Bridge followed by SAM's own frozen resizer.**

```
Bridge (trainable, ~4.2M params):
    Linear(4096 -> 1024) + LayerNorm(1024) + GELU + Dropout(0.4)

Then:
    SAM.backbone.language_backbone.resizer (FROZEN, Linear 1024 -> 256)
```

**Why**: V1–V6 all projected through a 256-dim square bottleneck, compressing Qwen's rich 4096-D `[SEG]` representation ~16×. A parallel Zero-Shot SAM 3 experiment using a highly descriptive, object-aware prompt (no VLM, no projector, raw text path) achieved **0.928 mIoU on RWTD**. SAM natively understands complex semantic descriptions; the limiter was our projector squashing Qwen's vocabulary before SAM could see it.

V7's fix: widen the trainable projection into SAM's **native** text-encoder width (1024-D), then hand the final 1024→256 step back to SAM's own pretrained `resizer` layer, kept frozen. The Bridge's job is shape-matching, not semantic translation — SAM's pretrained resizer already knows that map.

**Dropout(0.4)** is load-bearing. At 4.2M params (2× V6) without heavy dropout, the Directional Drift pathology from V4 (10.5M projector memorising ADE20K directions) would reappear.

---

### Module C: SAM 3 with Batch Multiplexing

Each of the 7 query vectors is fed to SAM **independently** by flattening the query dimension into the batch dimension:

```
(B, 7, 256) -> reshape -> (B*7, 1, 256) -> SAM -> (B*7, 1, H, W) -> reshape -> (B, 7, H, W)
```

This eliminates SAM3's pretrained positional bias (V2 ablation: Slot 1 received 90.5% of pixels, Slot 2 received 0.0%). Image features are indexed via `image_ids` (no memory copy).

| Component | Status in V7 | Role |
|---|---|---|
| Image Encoder (ViT) | **Frozen** | Multi-scale visual features (FPN) |
| Fusion Encoder | **Frozen** (no LoRA) | Cross-attends image features with queries |
| `cross_attend_prompt` | **Frozen** (no LoRA) | Enriches encoder hidden states |
| Pixel Decoder | **Frozen** | Dense pixel embeddings (B, 256, H, W) |
| `language_backbone.resizer` | **Frozen** (reused by Bridge) | Linear(1024, 256) — semantic channel Qwen→SAM |

**V7 has no SAM LoRA.** V5 and V6 both tried to fine-tune SAM (orthogonal LoRA in cross-attention); both produced OOD regression on RWTD. With the ZS ceiling at 0.928, training SAM at all is net-negative.

---

### Module D: Multi-Texture Mask Head

**Status**: Fully trainable.

```python
query_proj  = MLP(queries)          # (B, 7, 256) -> (B, 7, 256)
pixel_proj  = Conv1x1(pixel_embed)  # (B, 256, H, W) -> (B, 256, H, W)
mask_logits = einsum("bqc, bchw -> bqhw", query_proj, pixel_proj)  # (B, 7, H, W)
```

---

## The Dustbin Query

Channel 0 is a learned 4096-dim embedding that absorbs all non-texture pixels.

```
Index:  [  0     ,  1   ,  2   , ...,  K  , K+1 , ...,  6  ]
Role:   [DUSTBIN , SEG_1, SEG_2, ..., SEG_K, PAD , ..., PAD ]
```

PAD slots have logits set to `-inf` before softmax/CE.

---

## Block-Diagonal Attention Mask

During both training and inference extraction, a custom 4D attention mask prevents Context Leakage between texture blocks:

```
         prefix  TEX_1  TEX_2  TEX_3
prefix  [causal  ────   ────   ─── ]
TEX_1   [  ok   causal ────   ──── ]
TEX_2   [  ok    BLOCK causal ──── ]   <- TEX_2 CANNOT see TEX_1
TEX_3   [  ok    BLOCK  BLOCK causal]   <- TEX_3 CANNOT see TEX_1 or TEX_2
```

All blocks can attend to the shared prefix (system + image + user prompt). Within each block, standard causal attention applies.

**Why this matters**: Without the mask, Qwen's causal attention causes `<|seg|>_2` to absorb semantic noise from TEXTURE_1's description. Ablation proved this inflates inter-texture cosine similarity from 0.16 (with mask) to 0.74 (without mask), causing Directional Drift in the projector.

---

## Exponential LM Loss (V7)

V6's cosine-decayed LM schedule kept loss pressure ~0.5 across most of each block, producing a constant tug-of-war between DICE (wants geometric freedom on `[SEG]`) and LM (wants linguistic fidelity on every text token). V6 DICE plateaued at 0.43.

V7 replaces the cosine with a sharp exponential **cliff**:

```
Weight(d) = 1 − exp(−α · d),   α = 2.0
```

where `d` is the token-wise distance to the nearest assistant-region `<|seg|>`.

| Distance | Weight |
|---|---|
| d = 0 (the SEG token) | **0.000** (strictly zero — total geometric freedom) |
| d = 1 | 0.865 |
| d = 2 | 0.982 |
| d = 3 | 0.998 |
| d ≥ 4 | ≈ 1.0 (full LM supervision) |

The grounding token is geometrically free; every other text token is under near-full linguistic anchoring. Prompt-side `<|seg|>` tokens (from the V7 format template) are filtered out of the distance calculation — only real assistant-region SEGs count.

---

## Orthogonal LoRA (retired in V7)

V5 and V6 applied Orthogonal LoRA to SAM3's cross-attention to enable constrained fine-tuning. V7 removes SAM LoRA entirely. The ZS SAM 3 breakthrough and V6's Stage-3 RWTD regression both show that any SAM training hurts OOD generalisation relative to the pretrained weights. The mechanism is retained in the codebase (`models/orthogonal_lora.py`) for reference but is never instantiated.

---

## Training Process

### Two-Stage Curriculum

| Stage | Epochs | Trainable | Frozen | Key event |
|---|---|---|---|---|
| **1. Bridge Warmup** | 1-12 | Bridge (1e-4) + mask head + dustbin | Qwen LoRA, SAM (full) | Bridge learns to route Qwen's 4096-D into SAM's 1024-D native space |
| **2. Qwen Sync + Bridge Decay** | 13-20 | + Qwen LoRA (1e-6). **Bridge LR decayed 10× → 1e-5** | SAM (full) | Qwen LoRA builds on a slowed-down Bridge; exponential LM keeps language stable |

SAM is permanently frozen across all stages (no Stage 3).

### Forward Pass Walkthrough

```
Step 1: Qwen Forward (teacher forcing with <|seg|> tokens)
  |  Image + prompt + "TEXTURE_1: desc <|seg|>\nTEXTURE_2: desc <|seg|>"
  |  Block-diagonal attention mask applied
  |  -> hidden_states + qwen_logits (for exponential-LM weighted CE)
  |
Step 2: Extract <|seg|> Hidden States
  |  Clean token lookup filtered to assistant region (V7 prompt contains literal <|seg|>)
  |  K vectors of dim 4096, each decoupled from other texture blocks
  |
Step 3: Build 7 Query Slots
  |  [DUSTBIN, SEG_1, ..., SEG_K, PAD, ..., PAD]
  |
Step 4: Bridge + frozen SAM resizer
  |  Bridge(4096 -> 1024) + SAM.resizer(1024 -> 256)
  |  -> (B, 7, 256)
  |
Step 5: SAM3 Batch Multiplexed (frozen)
  |  (B, 7, 256) -> (B*7, 1, 256) -> Fusion Encoder -> Pixel Decoder
  |  -> (B, 7, H, W) mask logits
  |
Step 6: Loss
  |  Mask: CE + heavy Dice (weight 3.0) on pixel predictions
  |  LM: Exponential-cliff weighted CE (w=0 at [SEG], w≈1 elsewhere)
```

### Loss Functions

```
L_total = mask_weight * (CE + 3.0 * Dice) + lm_weight * LM_exponential
```

| Loss | Weight | Notes |
|---|---|---|
| **Cross-Entropy** | 1.0 | Pixel-wise, PAD channels = -inf |
| **Dice** | 3.0 | Per-channel, PAD excluded |
| **LM (exponential cliff)** | 0.1 | `Weight(d) = 1 − e^(−2d)`; SEG token itself at 0 |
| ~~Orthogonal Reg~~ | 0.0 | Retired in V7 (no SAM LoRA) |

### Differential Learning Rates (stage-dependent)

| Component | Stage 1 (1-12) | Stage 2 (13-20) |
|---|---|---|
| Bridge (4.2M) | 1e-4 | **1e-5** (decayed 10×) |
| Mask Head + Dustbin | 1e-4 | 1e-4 |
| Qwen LoRA (3.8M) | frozen | 1e-6 (0.01×) |
| SAM (full) | frozen | frozen |

---

## Inference (Two-Pass Strategy)

```
Pass 1 — Generation (standard causal mask):
  Qwen generates the full texture description sequence naturally.
  Standard causal attention ensures no repetition.

Pass 2 — Extraction (block-diagonal mask):
  The FULL generated sequence is re-run through Qwen with the
  block-diagonal custom mask. Each <|seg|> hidden state is computed
  in isolation from other texture blocks — matching training conditions.
  Only assistant-region <|seg|> tokens count (prompt-side ones filtered).

  If <|seg|> tokens are not yet emitted (early training), a regex
  fallback extracts from TEXTURE_N: line-end positions in Pass 1,
  also filtered to the generated region (>= prompt_len).
```

This two-pass approach guarantees train-test parity while preserving generation quality.

---

## Ablation History (V1-V7)

Each architectural component was motivated by a specific pathology discovered through rigorous ablation:

| Version | Key change | RWTD mIoU (best honest) | Pathology resolved / failure |
|---|---|---|---|
| V1 | Initial VLM + SAM wiring | 0.678 | Qwen LoRA overfitting |
| V2 | Query-slot design | 0.695 | **Slot-1 Addiction** discovered |
| V3 | LLM co-training | 0.703 | Count Collapse + projector drift + SAM regression |
| V4 | 10.5M projector | 0.692 | **Directional Drift** (memorises ADE20K directions) |
| V4-Slim | Bottleneck 2.1M | 0.732 | **First to beat ZS baseline** |
| V5-Oracle | Block-diagonal mask | 0.810 (Oracle) | Geometry-only ceiling; language collapsed in live |
| V5-Live | | 0.136 | **Language Collapse** — LoRA forgot language |
| V6 | Proximity-decayed LM | 0.694 (live exactly_2) | Language preserved; **Latent Co-Adaptation Churn** in Stage 3 |
| ZS SAM 3 (object-aware prompt) | No training | **0.928** | Reframed the ceiling: SAM natively understands object-aware descriptions |
| **V7** | **Bridge + frozen resizer + exponential LM + SAM frozen forever** | **target ≥ 0.80** | Closes the projector bottleneck; eliminates Stage-3 regression |

Full ablation data: `ablation/v1/` through `ablation/v7/`.

---

## Project Structure

```
Qwen2SAM_DeTexture/
|
+-- configs/
|   +-- detexture.yaml              # Main V7 training config
|
+-- models/
|   +-- qwen2sam_detexture.py       # Main model (SEG token, block mask, two-pass inference, Bridge + frozen resizer)
|   +-- bridge.py                   # V7 Bridge (Linear 4096 -> 1024 + LN + GELU + Dropout 0.4)
|   +-- projector.py                # V5/V6 bottleneck projector (retained for reference)
|   +-- orthogonal_lora.py          # Retired in V7 (retained for reference; never instantiated)
|   +-- losses.py                   # Losses (mask + exponential-weighted LM)
|
+-- data/
|   +-- dataset.py                  # DeTextureDataset + collator (V7 prompts, exponential lm_weights)
|
+-- training/
|   +-- train.py                    # V7 two-stage training loop (decay_bridge_lr at Stage 2)
|   +-- monitor.py                  # Sanity checker, logger, plotter, live test evaluator
|   +-- utils.py                    # Config, seed, scheduler, checkpointing
|
+-- scripts/
|   +-- ablation_exact_k2_rwtd.py          # RWTD prompt-constraint ablation
|   +-- ablation_object_friendly_prompt.py # Object-friendly prompt variant
|   +-- ablation_structured_prompt_ep10_vs_ep18.py  # ep10 vs ep18 structured-prompt test
|   +-- ablation_short_length_ep10.py      # Length-cap sensitivity test
|   +-- consolidate_structured_ablation_texts.py    # Merge GT + generations per checkpoint
|   +-- test_on_dataset.py                 # Generic live-inference driver
|   +-- eval_rwtd_oracle.py                # Oracle-mode evaluator
|   +-- analyze_vector_collapse.py         # Pre/post projector cosine analysis
|   +-- analyze_visual_bias.py             # Orthogonal subspace projection
|
+-- ablation/
|   +-- v1/ v2/ v3/ v4/ v5/ v6/ v7/  # Per-version ablation studies + analyses
|   +-- vector_collapse_analysis.json
|   +-- visual_bias_analysis.json
|   +-- live_ade20k_control.json
|
+-- checkpoints/
|   +-- archive_v6/                 # All prior V6 .pt files (pre-V7 baseline)
|   +-- logs/  plots/  test_results/  (populated during training)
```

---

## Configuration

```yaml
model:
  qwen_model: "Qwen/Qwen3-VL-8B-Instruct"
  lora_r: 8                       # Qwen LoRA rank
  lora_alpha: 16
  qwen_lr_scale: 0.01             # Qwen LR = base * 0.01 = 1e-6
  projector_hidden_dim: 1024      # V7 Bridge output width (SAM3 native text width)
  projector_dropout: 0.4          # V7 heavy dropout on Bridge

curriculum:
  projector_warmup_epochs: 12     # Stage 1: Bridge warmup (Qwen + SAM frozen)
  projector_lr_decay_at_stage2: 0.1  # Bridge LR x0.1 at Stage 2 entry

loss:
  mask_weight: 1.0
  ce_weight: 1.0
  dice_weight: 3.0
  lm_weight: 0.1                  # Exponential cliff, alpha=2.0 (hardcoded in dataset.create_labels)
  orthogonal_weight: 0.0          # Retired (no SAM LoRA)

training:
  batch_size: 2
  gradient_accumulation_steps: 4  # Effective batch = 8
  num_epochs: 20
  learning_rate: 1.0e-4
```

---

## Getting Started

### Prerequisites

- PyTorch >= 2.1
- transformers >= 4.45
- peft >= 0.7
- sam3 (Meta AI)
- scipy, opencv-python, Pillow

### Training

```bash
cd /home/aviad/Qwen2SAM_DeTexture
python -m training.train --config configs/detexture.yaml
```

Resume from checkpoint:
```bash
python -m training.train --config configs/detexture.yaml --resume auto
```

### Data Format

```json
[
  {
    "image_path": "/path/to/image.jpg",
    "textures": [
      {"description": "Texture of rough stone...", "mask_path": "/path/to/mask1.png"},
      {"description": "Texture of smooth water...", "mask_path": "/path/to/mask2.png"}
    ]
  }
]
```

---

## Trainable Parameters

```
+-----------------------------------------------------------+
|                  V7 TRAINABLE COMPONENTS                  |
+-----------------------------------------------------------+
| Qwen3-VL LoRA (r=8, q_proj + v_proj)    3,833,856         |
| V7 Bridge (Linear 4096->1024 + LN + Dr)  4,197,376        |
| Multi-Texture Mask Head                    197,376        |
| DUSTBIN embedding (4096-dim)                 4,096        |
+-----------------------------------------------------------+
| TOTAL TRAINABLE                          8,232,704 (8.23M)|
+-----------------------------------------------------------+

+-----------------------------------------------------------+
|                   V7 FROZEN COMPONENTS                    |
+-----------------------------------------------------------+
| Qwen3-VL base weights                   ~8B params        |
| SAM3 Image Encoder + Fusion + Decoder   ~300M params      |
| SAM3 language_backbone.resizer (reused)  1024x256 + 256   |
+-----------------------------------------------------------+
```
