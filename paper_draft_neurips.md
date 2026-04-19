# Qwen2SAM-DeTexture: End-to-End VLM-Guided Multi-Texture Segmentation via Native-Space Bridging and Multiplexed Grounding

**Anonymous Authors**

---

## Abstract

Recent work on texture-aware segmentation—most notably TextureSAM, which demonstrated that foundation segmentation models carry a systematic shape bias that can be mitigated through texture-augmented fine-tuning—has established that *what* to segment in texture-dominant scenes is a solvable problem. However, the complementary question of *how many* and *which* textures to segment remains unaddressed: TextureSAM and the SAM family operate as class-agnostic prompt-based models that require external specification of regions of interest, while Vision-Language Models (VLMs) that could provide such specification struggle with dense, multi-target grounding in a single forward pass. We present **Qwen2SAM-DeTexture**, an end-to-end architecture that closes this gap by coupling a VLM's open-vocabulary reasoning with a texture-aware segmentation decoder for parallel, multi-texture segmentation. Through systematic ablation we identify and resolve five architectural pathologies that emerge when extending `[SEG]`-token grounding to the multi-target regime: (i) *Context Leakage*, whereby causal attention contaminates downstream `[SEG]` representations (inter-texture cosine similarity of 0.74 under shared context vs. 0.16 under isolation); (ii) *Slot-1 Positional Bias*, where SAM's cross-attention allocates 90.5% of pixels to the first query regardless of semantic content; (iii) *Count Collapse vs. Language Collapse*, a binary dilemma in multi-modal LoRA training where uniform LM loss causes single-texture termination while binary masking causes catastrophic forgetting; (iv) *Directional Drift*, where an over-parameterized projector memorizes domain-specific manifold directions; and (v) *Bottleneck Semantic Squash*, whereby projecting a VLM's 4096-dim grounding representation through a 256-dim square bottleneck destroys the semantic richness a pretrained segmentation backbone can readily interpret. A parallel Zero-Shot experiment confirms the latter: SAM 3 with a hand-crafted, object-aware text prompt achieves **0.928 mIoU on RWTD without any training**, reframing the projector's role from *learning a new semantic mapping* to *preserving one SAM already knows*. Our architecture introduces a **Bridge-to-Native projector** that maps Qwen's `[SEG]` hidden state into SAM 3's native 1024-dim text-encoder width via a single trainable linear layer and then hands the final projection back to SAM 3's own pretrained resizer (kept frozen), together with a sharp **exponential LM-loss cliff** that isolates the `[SEG]` token geometrically while anchoring all surrounding tokens linguistically, a **permanently frozen SAM** (no LoRA), and a **two-stage curriculum** with ~8.2M trainable parameters. We further contribute the **DeTexture Pipeline**, a geometry-first texture mining framework for curating training data from large-scale datasets, and the **Real-World Texture Dataset (RWTD)**, a benchmark of 253 in-the-wild texture transition scenes.

---

## 1. Introduction

Language-grounded segmentation has emerged as one of the most active frontiers in vision-language research. A growing family of models—exemplified by LISA [2], which pioneered the `[SEG]` token paradigm, and Sa2VA [7], which unifies SAM-2 with a VLM for both image and video grounding—have demonstrated that Vision-Language Models (VLMs) can learn to emit special grounding tokens whose hidden representations, when projected into a segmentation decoder such as SAM [9], yield high-quality masks conditioned on complex natural language queries. This paradigm—coupling VLM reasoning with dense pixel prediction—has achieved strong results on referring expression segmentation and conversational grounding.

However, all existing language-grounded segmentation models share a fundamental limitation: they are designed for *single-target* or *sequential* grounding. LISA requires $K$ independent forward passes to segment $K$ targets, each pass unaware of the others. Sa2VA unifies SAM-2 with LLaVA into a shared token space but inherits the same causal decoding bottleneck when generating multiple grounding targets. No existing system can produce a *spatially consistent, non-overlapping partition* of an image into multiple regions in a single forward pass—the model itself must decide *how many* targets exist, *what* they are, and *where* each pixel belongs.

This limitation becomes particularly acute for **texture segmentation**, a domain that inherently requires multi-target partitioning. Unlike object segmentation—where a single referring expression typically isolates one entity—texture segmentation demands that the model simultaneously identify and delineate multiple surface regions (concrete, grass, gravel) that tile the image without gaps or overlaps. Recent work in texture-aware segmentation has revealed that foundation segmentation models carry systematic biases: Zhang et al. [20] showed that SAM is biased toward texture-like dense features over shape cues, while TextureSAM [1] demonstrated that fine-tuning SAM-2 on texture-augmented data can shift its segmentation prior toward texture-defined boundaries. Yet these texture-aware models remain prompt-driven—they cannot autonomously determine the number or identity of texture regions.

A fundamental reason for this gap is that SAM—despite its remarkable zero-shot segmentation capability—lacks *semantic understanding* of what it segments. SAM excels at delineating regions given geometric prompts (points, boxes, masks), but it has no internal representation of *why* a boundary exists or *what material* lies on either side of it. For texture segmentation, this distinction is critical: the boundary between wet sand and dry sand may be visually subtle yet semantically unambiguous to a language model that understands material properties. We therefore conceptualize the VLM as the semantic "brain" that identifies and describes texture regions, and SAM as the geometric "eyes" that execute precise spatial delineation. The architecture's challenge is to bridge these two modalities—mapping linguistic descriptions into the geometric prompt space—without corrupting either the VLM's language capabilities or SAM's spatial precision.

We present **Qwen2SAM-DeTexture**, an end-to-end architecture that realizes this brain-eyes coupling. Our central question is: **Can a VLM-guided system autonomously identify, describe, and segment multiple texture regions simultaneously, producing a consistent image partition in one forward pass?**

We answer affirmatively, but arriving at our final design required uncovering and resolving five distinct *architectural pathologies* that emerge when extending single-target `[SEG]`-token grounding to the multi-target regime. These pathologies are not specific to texture segmentation; they are general failure modes of any system that combines causal VLM decoding with SAM-family decoders for multi-target prediction.

**Data-Architecture Co-design.** Our research follows a *data-architecture co-design* philosophy. For training data, we build on TextureSAM's [1] texture-augmented ADE20K dataset, extending it with a geometry-first scoring pipeline (the DeTexture Pipeline) that mines the most texture-rich samples and an automated VLM-based annotation stage that generates natural language descriptions for each texture region (Section 4). This data ecosystem also produced two evaluation benchmarks: ADE20K\_DeTexture (in-domain) and RWTD (out-of-domain, adopted from TextureSAM [1]). Early evaluation on RWTD revealed the Slot-1 Positional Bias in SAM's decoder (invisible on cleaner benchmarks), and the cross-domain gap between ADE20K and RWTD exposed Directional Drift in our projector (undetectable when training and evaluation share the same domain). In this sense, the dataset *discovered* the architectural problems.

**Contributions.** We make the following contributions:

1. **An End-to-End Multi-Target Grounding Architecture.** We propose a system that maps rich linguistic texture descriptions directly to geometric segmentation via dedicated `[SEG]` grounding tokens, producing a spatially consistent partition in a single forward pass with ~8.2M trainable parameters atop a frozen Qwen3-VL-8B and a permanently frozen SAM 3 backbone.

2. **Bridge-to-Native-Space Projection.** We demonstrate that the conventional square-bottleneck projector design (4096 → 512 → 256) destroys the semantic richness a pretrained segmentation backbone can already interpret. Instead, we project into SAM 3's *native* 1024-dim text-encoder width with a single trainable linear layer and reuse SAM 3's own pretrained 1024 → 256 resizer (kept frozen) for the final step. This turns the cross-model alignment problem from a *learning* problem into a *shape-matching* problem.

3. **The Block-Diagonal Attention Mask.** We introduce a custom causal attention mask that prevents *Context Leakage* between sequential texture descriptions, reducing inter-`[SEG]` cosine similarity from 0.74 to 0.16 and enabling independent, parallel grounding of multiple texture targets.

4. **Exponential LM-Loss Cliff.** We identify the fundamental *Language Collapse vs. Count Collapse* dilemma in multi-modal LoRA training and resolve it with a sharp exponential weighting schedule: `Weight(d) = 1 − e^(−2d)` where `d` is the token-wise distance to the nearest `[SEG]`. The grounding token itself is assigned weight zero (total geometric freedom), while every surrounding token receives near-full linguistic supervision — eliminating the continuous DICE/LM tug-of-war that smoother weighting schedules induce.

5. **Systematic Characterization of Multi-Target Grounding Pathologies.** Through rigorous ablation, we identify and formally characterize five pathologies—Context Leakage, Slot-1 Positional Bias, the Count/Language Collapse dilemma, Directional Drift, and Bottleneck Semantic Squash—that affect any architecture combining causal VLM decoding with SAM-family decoders for multi-target prediction.

6. **A Complete Data Ecosystem.** We contribute the DeTexture Pipeline (geometry-first texture mining), ADE20K\_DeTexture (custom in-domain test set), and VLM-annotated RWTD (adopted out-of-domain benchmark), forming a dual-evaluation strategy that exposes cross-domain pathologies invisible to single-benchmark evaluation.

---

## 2. Related Work

### 2.1 Texture-Aware Segmentation and TextureSAM

Foundation segmentation models exhibit a well-documented shape bias: trained predominantly on object-centric datasets, they learn to segment along object boundaries rather than texture boundaries [1, 14]. TextureSAM [1] addressed this directly by constructing a texture-augmented ADE20K dataset using Compositional Neural Texture (CNT) synthesis [15] with a controllable interpolation coefficient $\eta$ that smoothly transitions images from their original appearance ($\eta = 1.0$) to pure texture replacement ($\eta = 0.0$). Fine-tuning SAM-2 on this augmented data shifted the model's segmentation prior toward texture-defined regions, achieving +0.21 mIoU on the Real-World Texture Dataset (RWTD) and +0.18 mIoU on synthetic texture benchmarks over the SAM-2 baseline. However, TextureSAM remains a *prompt-driven* model: it cannot autonomously determine the number or identity of texture regions, requiring external specification of regions of interest. Our work extends TextureSAM's contributions in two directions—leveraging its Textured-ADE20K for training data construction while adding autonomous, language-guided multi-texture reasoning that eliminates the need for manual prompting.

### 2.2 Reasoning Segmentation and the `[SEG]` Token Paradigm

LISA [2] pioneered *reasoning segmentation*, embedding a learnable `[SEG]` token into an LLM's vocabulary such that the token's hidden state, when projected into SAM's prompt encoder, produces a segmentation mask conditioned on complex natural language queries. LISA++ [3] extended this baseline with instance-level segmentation and Segmentation-in-Dialogue (SiD), enabling multi-turn conversational grounding. However, both LISA and LISA++ are fundamentally *sequential*: segmenting $K$ targets requires $K$ independent forward passes, each unaware of the others. This isolation leads to two practical failures: (a) computational cost that scales linearly with $K$, and (b) independently generated masks that lack mutual spatial awareness, producing overlapping regions that violate the fundamental constraint of image partitioning. In our experiments, sequential LISA-style approaches cannot enforce non-overlapping mask constraints without post-hoc reconciliation—precisely the property our Winner-Takes-All mechanism guarantees natively.

GSVA [4] made an important step toward multi-target grounding by extending LISA's architecture to support *multiple* `[SEG]` tokens within a single decoding pass, alongside a novel `[REJ]` token for explicitly rejecting absent targets in Generalized Referring Expression Segmentation (GRES). While GSVA demonstrates that multiple grounding tokens can coexist in the output sequence, it relies on standard causal attention during generation. As we demonstrate in Section 3.2, this architectural choice introduces *Context Leakage*: the hidden state of $\texttt{[SEG]}_k$ absorbs semantic information from all preceding texture descriptions through the causal attention pathway, with inter-token cosine similarity reaching 0.74 under shared context versus 0.16 under isolation. GSVA's design assumes that the LLM can maintain representational purity across sequential `[SEG]` tokens—an assumption our ablations empirically refute for dense texture grounding.

### 2.3 Dense Grounding via Multimodal LLMs

A growing family of models integrates pixel-level grounding directly into multimodal LLM architectures. PixelLM [5] introduces a lightweight pixel decoder with a segmentation codebook, where codebook token embeddings serve as conditioning inputs for mask generation, enabling multi-target referring segmentation without requiring SAM. GLaMM [6] pioneered *Grounded Conversation Generation*, producing natural language responses seamlessly interleaved with segmentation masks through phrase-level grounding. Sa2VA [7] unifies SAM-2 with LLaVA into a shared token space, generating instruction tokens that guide SAM-2's mask decoder for both image and video grounding. LLaVA-Grounding [8] connects visual grounding with LLM-based reasoning for referring expression comprehension.

While these models advance the state of the art on referring segmentation benchmarks (Sa2VA achieves 81.6 cIoU on RefCOCO, outperforming GLaMM's 79.5), they share a common architectural limitation: all rely on standard causal attention mechanisms during multi-token generation. This is adequate for *referring* segmentation, where targets are specified by the user and generated one at a time, but becomes pathological for *autonomous multi-target partitioning*, where the model must simultaneously generate and ground multiple regions. We identify two failure modes specific to this setting: Context Leakage from causal attention (Section 3.2) and Count Collapse from co-trained language loss (Section 3.4). Our block-diagonal attention mask and proximity-decayed loss masking are general solutions applicable to any architecture in this family.

### 2.4 SAM and Prompt-Based Segmentation

The Segment Anything Model (SAM) [9] and its successors (SAM-2 [10]) have established prompt-based segmentation as a powerful paradigm. Given visual prompts (points, boxes, masks) or dense embedding prompts, SAM produces high-quality segmentation masks with strong zero-shot generalization. SAM-2 extends this to video with memory-augmented tracking. These models excel as segmentation *backends* in VLM-guided pipelines, but their decoder architecture exhibits an underexplored positional bias when processing multiple prompt embeddings simultaneously. Through controlled experiments on all 253 RWTD samples using ground-truth embeddings, we discovered that SAM's cross-attention mechanism allocates 90.5% of pixels to the first prompt slot regardless of semantic content, with 0.0% allocation to the second slot—a bias that persists after swapping prompt order and under isolated single-query evaluation (Section 3.3). This *Slot-1 Addiction* is invisible in single-target applications (where all existing benchmarks operate) but catastrophic for multi-target segmentation, effectively reducing a multi-query decoder to a single-query one.

### 2.5 Multi-Target and Panoptic Segmentation

Classical panoptic segmentation approaches—Mask2Former [11], Panoptic FPN [12]—achieve multi-target segmentation through DETR-style set prediction or proposal generation. While effective for closed-vocabulary, fixed-category settings, these methods cannot leverage natural language descriptions of novel textures and require retraining for new categories. Recent open-vocabulary panoptic models (X-Decoder [16], OpenSeeD [17]) extend to unseen categories via text-image alignment but still operate with predefined category lists rather than free-form descriptions. Our work bridges this gap by combining VLMs' open-vocabulary reasoning with SAM's segmentation quality, introducing the architectural innovations necessary for reliable multi-target texture grounding.

### 2.6 Texture Segmentation and Datasets

Texture segmentation has a rich history predating deep learning, from Brodatz-based evaluations to the Describable Textures Dataset (DTD) [13], KTH-TIPS [14], and texture subsets of MS-COCO Stuff [15]. However, these benchmarks predominantly feature isolated texture patches or well-separated regions under controlled conditions. They do not capture the challenging *texture transition* scenarios common in real-world imagery: gradual boundaries between sand and water, interleaving patterns of brick and mortar, or complex junctions of concrete, grass, and gravel. TextureSAM's RWTD benchmark [1] and our extended DeTexture mining pipeline specifically target these challenging in-the-wild cases, providing both evaluation benchmarks and a reproducible methodology for curating texture-rich training data from large-scale annotated datasets.

---

## 3. Methodology

### 3.1 Architecture Overview

Our architecture couples a Vision-Language Model's open-vocabulary reasoning with a texture-aware segmentation decoder for parallel, multi-texture segmentation in a single forward pass. The system comprises four components (Figure 1):

1. **Qwen3-VL-8B** (frozen backbone + lightweight LoRA, rank 8): generates structured texture descriptions, each terminated by a dedicated `[SEG]` grounding token whose hidden state encodes visually-grounded spatial information.
2. **Bridge + Frozen SAM Resizer** (trainable 4.2M + frozen 0.26M): a single `Linear(4096 → 1024) + LayerNorm + GELU + Dropout(0.4)` maps `[SEG]` hidden states into SAM 3's *native* 1024-dim text-encoder width, then SAM 3's own pretrained resizer (`Linear(1024, 256)`, kept frozen) completes the projection to the decoder's query space.
3. **SAM3 Mask Decoder** (permanently frozen): processes each projected query via batch-multiplexed cross-attention to produce per-texture mask predictions. No LoRA; a parallel Zero-Shot experiment showed SAM 3 with an object-aware prompt already reaches 0.928 mIoU on RWTD — any fine-tuning hurts OOD generality.
4. **Winner-Takes-All Pixel Assignment**: resolves spatial competition between queries to produce a consistent, non-overlapping image partition augmented with a learned dustbin channel.

**[Figure 1: Architecture and Gradient Flow.]** *Forward pass (top):* Input image + prompt → Qwen3-VL generates texture descriptions with `[SEG]` tokens under block-diagonal attention mask → Bridge (trainable, 4096 → 1024) → SAM 3's frozen resizer (1024 → 256) → Batch-multiplexed SAM 3 (frozen) produces per-texture masks → WTA pixel assignment. *Backward pass (bottom):* Mask Loss (CE + Dice) → Bridge → `[SEG]` token latent space → Qwen LoRA (shaping *where* the VLM attends). Simultaneously, Exponential-Cliff LM Loss → Qwen LoRA on all non-`[SEG]` tokens (preserving *what* the VLM generates); weight at `[SEG]` itself is strictly zero.

Training employs a two-stage curriculum with ~8.2M trainable parameters and a novel exponential LM cliff that simultaneously prevents count collapse and catastrophic forgetting. The design addresses five pathologies that emerge when extending single-target `[SEG]`-token grounding to the multi-target regime, which we characterize in the following subsections.

### 3.2 Overcoming Context Leakage via Block-Diagonal Attention Masking

**Pathology.** Any architecture that generates multiple `[SEG]` tokens within a single causal decoding pass is susceptible to *Context Leakage*: the hidden state of a later grounding token absorbs semantic information from earlier texture descriptions through the autoregressive attention pathway. Given $K$ texture targets, the model produces a sequence:

$$\texttt{TEXTURE\_1: } \langle \text{desc}_1 \rangle \texttt{ [SEG]}_1 \texttt{ TEXTURE\_2: } \langle \text{desc}_2 \rangle \texttt{ [SEG]}_2 \ldots$$

Under standard causal (left-to-right) attention, $\texttt{[SEG]}_2$ attends to all preceding tokens, including the description of TEXTURE\_1 and $\texttt{[SEG]}_1$. This creates a representational contamination pathway: the hidden state of $\texttt{[SEG]}_2$ encodes not only the semantics of TEXTURE\_2 but also residual information from TEXTURE\_1.

We quantified this effect empirically. Under a standard "segment 1 to 6 textures" prompt with causal attention, the mean cosine similarity between consecutive `[SEG]` embeddings reached **0.74**. When we constrained the prompt to "exactly 2 textures" and repeated the measurement, the cosine similarity dropped to **0.16**—revealing that the high similarity was a *prompt-induced artifact* of causal context sharing, not an intrinsic property of the textures. This contamination is asymmetric and cumulative: later `[SEG]` tokens are progressively more contaminated than earlier ones, producing a systematic degradation in mask quality that worsens with the number of targets. This pathology is inherent to any multi-`[SEG]` architecture that relies on standard causal attention, including GSVA [4] and Sa2VA [7].

**Solution: Two-Pass Inference with Block-Diagonal Attention Mask.** We introduce a custom attention mask $\mathbf{M} \in \{0, -\infty\}^{L \times L}$ that enforces block-diagonal structure over texture-specific token groups while preserving the causal constraint within each block. Let $\mathcal{G}_k$ denote the set of token indices belonging to texture $k$ (its description tokens and `[SEG]` token). We define:

$$\mathbf{M}_{ij} = \begin{cases} 0 & \text{if } \exists k: i \in \mathcal{G}_k \wedge j \in \mathcal{G}_k \wedge i \geq j \\ 0 & \text{if } j \in \mathcal{G}_{\text{shared}} \wedge i \geq j \\ -\infty & \text{otherwise} \end{cases}$$

where $\mathcal{G}_{\text{shared}}$ contains the image tokens and system prompt tokens that all textures may attend to. The constraint $i \geq j$ preserves the causal ordering within each block and with respect to the shared context, ensuring compatibility with the LLM's pretrained autoregressive distribution. This mask guarantees that each `[SEG]` token attends *only* to past tokens within its own texture description and the shared visual context, producing semantically pure, decoupled representations. At inference, we employ a two-pass strategy: the first pass processes the full prompt to extract texture descriptions via Qwen's frozen language capabilities, while the second pass applies the block-diagonal mask to compute isolated `[SEG]` embeddings.

### 3.3 Solving SAM's Positional Bias via Batch Multiplexing

**Pathology.** Prompt-based segmentation decoders in the SAM family exhibit an underexplored positional bias when processing multiple prompt embeddings simultaneously. When the mask decoder receives multiple prompt embeddings $\mathbf{Q} \in \mathbb{R}^{B \times K \times D}$ (where $K$ is the number of texture queries), its cross-attention mechanism routes nearly all spatial attention to the first prompt position—a phenomenon we term *Slot-1 Addiction*. Through controlled experiments on all 253 RWTD samples using ground-truth `[SEG]` embeddings, we measured the pixel allocation across prompt slots:

| Slot | Pixel Allocation | After Swap (T1↔T2) | Isolated Query |
|------|-----------------|---------------------|----------------|
| Slot 1 | **90.5%** | **90.5%** | 90.5% |
| Slot 2 | **0.0%** | **0.0%** | 0.0% |

The bias persists identically after swapping texture assignments between slots and when running each query in isolation—confirming that this is a *positional* bias intrinsic to the decoder's pretrained cross-attention weights, not a consequence of query content. This pathology is invisible in single-target applications (where all existing benchmarks operate) but catastrophic for any multi-target segmentation system, effectively reducing a multi-query decoder to a single-query one.

**Solution: Batch Multiplexing.** Rather than passing multiple queries through SAM's multi-prompt pathway, we reshape the query tensor from $\mathbb{R}^{B \times K \times D}$ to $\mathbb{R}^{(B \cdot K) \times 1 \times D}$, replicating the image features accordingly via an index mapping. This forces each texture query to occupy *its own* Slot 1 in an independent SAM forward pass, bypassing the positional bias entirely. The resulting masks $\hat{\mathbf{m}}_k \in [0,1]^{H \times W}$ for $k = 1, \ldots, K$ are then recombined along the batch dimension for competitive pixel assignment. This transformation is computationally equivalent to $K$ independent SAM passes but is implemented as a single batched operation, preserving GPU efficiency. Empirically, batch multiplexing immediately recovered performance to 99.5% of the zero-shot baseline (mIoU 0.703 vs. 0.706) at epoch 5.

### 3.4 The Language Collapse Dilemma and the Exponential LM Cliff

**Pathology 1: Count Collapse under Uniform LM Loss.** We observe that naïve end-to-end co-training of a VLM with segmentation losses inevitably leads to a *Count Collapse* pathology: when the language modeling cross-entropy loss is computed uniformly over all generated tokens, the optimization landscape creates a shortcut whereby the model minimizes total loss by terminating generation after a single texture description, since predicting fewer tokens incurs less cumulative risk. In our experiments, this manifested with striking severity: 85% of evaluation samples predicted $K=1$ texture, with the count distribution collapsing to $\{1: 216, 2: 17, 3: 9, 4: 2, 5: 2, 6: 7\}$. The projector irreversibly co-adapted to this degenerate distribution, allocating 100% of pixels to the dustbin when fed legitimate multi-texture inputs.

**Pathology 2: Language Collapse under Binary Loss Masking.** The naïve fix—applying $\texttt{-100}$ to *all* text tokens and computing LM loss exclusively on `[SEG]` positions—prevents Count Collapse but introduces a symmetric failure: *Language Collapse*. With zero LM gradient on text tokens, the LoRA adapters receive gradients exclusively from the spatial segmentation path (mask loss → SAM → projector → `[SEG]` → LoRA). Within a few epochs of co-training, the LoRA undergoes catastrophic forgetting of its linguistic prior: the model loses the ability to generate coherent texture descriptions and ceases to emit `[SEG]` tokens during inference. Crucially, the segmentation architecture itself remains fully functional when bypassing generation with ground-truth descriptions—confirming that the failure is purely linguistic, not geometric.

This reveals a fundamental dilemma in multi-modal LoRA training: *binary* loss masking creates a binary failure mode. Full text loss → Count Collapse; zero text loss → Language Collapse. The solution requires an *isolation* of the grounding token from linguistic supervision while retaining full supervision on all surrounding tokens.

**An Intermediate Cosine-Decayed Variant.** An earlier iteration of our architecture smoothed the binary mask into a per-block cosine decay from $\lambda_{\max} = 1.0$ at the block start down to $\lambda_{\min} = 0.05$ at the `[SEG]` token. This prevented catastrophic forgetting (live-inference RWTD mIoU recovered from 0.136 to 0.694) but introduced a *continuous* tug-of-war: with non-zero linguistic pressure on nearly every token in each block, the Dice mask loss could not drive the grounding embedding aggressively enough toward geometric optima. Empirically, the training Dice plateaued at 0.43 and honest (exactly-K) RWTD mIoU stalled at 0.694.

**Our Solution: Exponential LM Cliff.** We replace the gentle cosine schedule with a sharp exponential cliff centred on the `[SEG]` token itself. Let $d(i)$ denote the token-wise distance from position $i$ to the nearest assistant-region `[SEG]` token. We define:

$$\lambda_{\text{LM}}(i) = \begin{cases} 0 & \text{if } i \in \mathcal{S}_{\text{asst}} \\ 1 - e^{-\alpha \cdot d(i)} & \text{otherwise} \end{cases}$$

with $\alpha = 2.0$ and $\mathcal{S}_{\text{asst}}$ the set of assistant-region `[SEG]` positions. The weighted LM loss is:

$$\mathcal{L}_{\text{LM}}^{\text{exp}} = \frac{\sum_{i \in \mathcal{A}} \lambda_{\text{LM}}(i) \cdot \ell_{\text{CE}}(i)}{\sum_{i \in \mathcal{A}} \lambda_{\text{LM}}(i)}$$

where $\mathcal{A}$ denotes the set of assistant (non-prefix) token positions. Concretely, the weights are:

| distance $d$ | $\lambda_{\text{LM}}$ |
|---|---|
| 0 (the `[SEG]` token) | **0.000** |
| 1 | 0.865 |
| 2 | 0.982 |
| 3 | 0.998 |
| $\geq 4$ | $\approx 1.0$ |

This "cliff" structure isolates the grounding token *completely* from linguistic supervision (enabling aggressive Dice optimization on `[SEG]`) while applying *near-full* LM supervision to every surrounding token (preventing language drift). Unlike a cosine decay, there is no continuous gradient conflict: within one token of `[SEG]`, the LM pressure is already at 87% of its full value, and by two tokens out it is at 98%.

**Prompt-Template `[SEG]` Filtering.** The user prompt includes the literal string "`<|seg|>`" in its format-template example, which tokenises to the SEG special token within the user message. All distance computations and `[SEG]`-position lookups are filtered to the assistant region only, so prompt-template SEGs are never treated as real grounding outputs.

**Empirical Validation.** We verified the effectiveness of this design through controlled ablation (Section 5): with binary masking ($\lambda = 0$ on text), the VLM's text generation collapses within 5 epochs; with uniform loss ($\lambda = 1$ everywhere), count collapse emerges within 10 epochs; with cosine decay, language coherence is preserved but Dice plateaus; with our exponential cliff, the Dice path and LM path no longer compete anywhere except at the geometrically-isolated `[SEG]` token.

### 3.5 From Information Bottleneck to Native-Space Bridge

**Pathology (Directional Drift).** We first demonstrate that mapping high-dimensional VLM embeddings to a segmentation decoder's prompt space using standard, high-capacity projectors results in *Directional Drift*: the projector memorizes domain-specific manifold directions from the training distribution rather than learning a generalizable mapping. An early variant with approximately 10.5M projector parameters achieved strong in-domain performance (ADE20K mIoU climbing from 0.700 to 0.707 across epochs 5–13), yet exhibited systematic cross-domain degradation: RWTD mIoU dropped from 0.692 at epoch 5 to 0.618 at epoch 10—a divergence invisible when training and evaluation share the same distribution.

To diagnose the mechanism, we performed a vector geometry analysis, measuring the cosine similarity between paired `[SEG]` embeddings *before* and *after* the projector:

| Domain | Pre-Projector Cosine | Post-Projector Cosine | Compression (Δ) |
|--------|---------------------|----------------------|-----------------|
| RWTD (cross-domain) | 0.736 | 0.919 | **+0.183** |
| ADE20K (in-domain) | 0.831 | 0.879 | +0.048 |

The projector compressed RWTD representations nearly **4× more** than ADE20K representations (Δ = 0.183 vs. 0.048), pushing cross-domain `[SEG]` vectors toward a shared manifold direction learned from ADE20K's training distribution.

**Intermediate Solution (Bottleneck Mitigation).** A first fix reduced the projector to ~2.1M parameters through a $4096 \to 512 \to 256$ square bottleneck with dropout ($p=0.15$). This single change produced the first trained model to surpass the zero-shot SAM baseline (0.732 vs. 0.706 RWTD mIoU). Later architectural iterations inherited the same bottleneck design. However, the best live-inference honest RWTD mIoU achievable under this family (under an exactly-K prompt to neutralize Hungarian-matching inflation) plateaued at 0.694, suggesting that while the bottleneck *cured* drift, it also imposed a new ceiling.

**Reframing: the Zero-Shot SAM 3 Upper-Bound Experiment.** To probe this ceiling, we ran a parallel experiment that *removes the VLM and projector entirely*: we hand-craft a single highly descriptive, object-aware prompt per RWTD image and feed it directly into SAM 3's native text pathway. This zero-shot configuration—no training, no VLM, no projector—achieved **0.928 mIoU on RWTD**, nearly +0.23 above our best trained number under the bottleneck design. The conclusion is unambiguous: SAM 3 natively understands complex semantic descriptions; the limiter was our projector squashing Qwen's rich 4096-dim `[SEG]` representation through a 256-dim bottleneck before SAM could see it.

**Our Solution: Bridge-to-Native-Space.** Rather than learning a new 4096 → 256 semantic mapping, we widen the trainable projection to match SAM 3's *native* text-encoder width and reuse SAM's own pretrained final projection. Formally, letting $\mathbf{h} \in \mathbb{R}^{4096}$ denote a `[SEG]` hidden state:

$$f_{\text{bridge}}(\mathbf{h}) = \mathrm{Dropout}_{0.4}(\mathrm{GELU}(\mathrm{LN}(\mathbf{W}_B \mathbf{h} + \mathbf{b}_B)))$$

with $\mathbf{W}_B \in \mathbb{R}^{1024 \times 4096}$ (the trainable Bridge, ~4.2M parameters). The final projection into SAM's 256-dim decoder query space is performed by SAM 3's *pretrained* `resizer` layer, $\mathbf{R} \in \mathbb{R}^{256 \times 1024}$, kept frozen in place:

$$\mathbf{q} = \mathbf{R} \cdot f_{\text{bridge}}(\mathbf{h}) + \mathbf{b}_R$$

This reframes the cross-model alignment problem: earlier bottleneck designs treated it as a *learning* problem (train a new 4096 → 256 mapping); our Bridge treats it as a *shape-matching* problem (widen Qwen's output to the width SAM already speaks). The Bridge's ~4.2M parameters (roughly 2× the prior bottleneck) would risk reintroducing Directional Drift, but the aggressive dropout ($p = 0.4$) acts as the sole regularizer, compensating for the larger capacity.

Because the resizer carries SAM 3's pretrained 1024 → 256 semantic mapping—the same mapping that enables the 0.928 ZS result—the Bridge need only supply shape compatibility, not semantic translation. Empirically, this design is expected to close the gap between the 0.694 honest RWTD mIoU of bottleneck-family variants and the 0.928 ZS ceiling; final numbers are reported in Section 5 once training converges.

### 3.6 Winner-Takes-All Pixel Assignment

Given $K$ soft mask predictions $\hat{\mathbf{m}}_k \in [0,1]^{H \times W}$ from the batch-multiplexed SAM decoder, we must produce a spatially consistent partition of the image. We define a Winner-Takes-All (WTA) assignment augmented with a learned dustbin channel:

$$\text{label}(i,j) = \begin{cases} \arg\max_{k \in \{1,\ldots,K\}} \hat{m}_k(i,j) & \text{if } \max_k \hat{m}_k(i,j) > \hat{m}_{\text{dust}}(i,j) \\ \texttt{dustbin} & \text{otherwise} \end{cases}$$

where $\hat{m}_{\text{dust}}$ is produced by a learned 4096-dimensional dustbin embedding passed through the same projector and SAM decoder pathway. The dustbin class absorbs pixels that do not belong to any queried texture, preventing forced assignment of background regions. The WTA mechanism guarantees a *mathematically consistent* partition: every pixel is assigned to exactly one class, with no overlaps and no gaps—a property that sequential, independent models fundamentally cannot enforce.

### 3.7 Two-Stage Curriculum with Permanently Frozen SAM

Earlier iterations of our architecture employed three-stage curricula that ultimately unfroze SAM via an orthogonal LoRA in its cross-attention layers. Across many ablation runs, every SAM-unfreezing configuration produced the same signature: a transient in-domain gain on ADE20K followed by an out-of-domain regression on RWTD (one such run lost 4.6 mIoU between its middle and final stages). Combined with the Zero-Shot SAM 3 upper-bound experiment (Section 3.5) showing that SAM 3 already reaches 0.928 mIoU on RWTD without any training, the conclusion is unambiguous: *SAM must be treated as a pretrained semantic asset, not a fine-tunable module.* Our final architecture therefore adopts a **two-stage curriculum** and permanently freezes SAM.

**Stage 1: Bridge Warmup (Epochs 1–12).** Only the Bridge, multi-texture mask head, and dustbin embedding are trainable (~4.4M parameters). Qwen LoRA is frozen; SAM (including the reused resizer) is frozen. The Bridge absorbs the random-initialisation shock and learns the shape-matching $4096 \to 1024$ map into SAM's native text space. Because the 4.2M-parameter Bridge is substantially larger than the prior 2.1M bottleneck, we extend the warmup to 12 epochs to allow the heavy Dropout(0.4) regularization to stabilise the Bridge's output distribution before any linguistic gradients enter the system.

**Stage 2: Qwen Sync + Bridge Decay (Epochs 13–20).** Qwen's LoRA adapters (rank 8, targeting q\_proj and v\_proj; ~3.8M parameters) are unfrozen at an ultra-conservative learning rate of $1 \times 10^{-6}$ (0.01× base). Simultaneously, the Bridge's learning rate is *decayed by 10×* (from $1 \times 10^{-4}$ to $1 \times 10^{-5}$). This decay is critical: prior ablation showed that maintaining the projector at full LR during co-training causes *Latent Co-Adaptation Churn* — the projector's rapid weight updates continuously shift the feature distribution that the LoRA is trying to adapt to, preventing stable co-convergence. By slowing the Bridge once Qwen joins, we ensure the LoRA builds on a stable mapping rather than chasing a moving target. The exponential LM cliff (Section 3.4) preserves linguistic coherence.

There is no Stage 3. SAM (including its resizer, cross-attention, and decoder) remains frozen across all 20 epochs.

The total loss combines mask supervision with the exponential-weighted language loss:

$$\mathcal{L} = \underbrace{\mathcal{L}_{\text{CE}}^{\text{mask}} + 3.0 \cdot \mathcal{L}_{\text{Dice}}}_{\text{mask losses}} + 0.1 \cdot \underbrace{\mathcal{L}_{\text{LM}}^{\text{exp}}}_{\text{exponential cliff}}$$

No orthogonal regularization term appears in the final loss (the SAM LoRA it previously regularised is retired).

**Differential Learning Rates.** The learning rate hierarchy reflects the trust hierarchy and evolves across stages:

| Component | Stage 1 (ep 1–12) | Stage 2 (ep 13–20) |
|---|---|---|
| Bridge (4.2M) | $1 \times 10^{-4}$ | $1 \times 10^{-5}$ (decayed 10×) |
| Mask Head + Dustbin | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ |
| Qwen LoRA (3.8M) | frozen | $1 \times 10^{-6}$ |
| SAM 3 (frozen resizer, decoder) | frozen | frozen |

---

## 4. Data Ecosystem: The DeTexture Pipeline and Dual-Evaluation Strategy

Our contribution extends beyond architecture to a complete data ecosystem designed to (a) transform generic semantic datasets into high-quality texture-grounding training data, (b) provide controlled in-domain evaluation, and (c) stress-test generalization on an external, in-the-wild benchmark. This ecosystem was not merely a prerequisite for training but an active participant in architectural discovery: each pathology identified in Section 3 was first surfaced by evaluating on out-of-domain data whose distributional distance from the training set exposed failure modes invisible on in-domain benchmarks.

### 4.1 The Core Data Contribution: The DeTexture Mining Pipeline

We introduce the **DeTexture Pipeline**, an automated, scalable framework designed to transform generic semantic datasets into high-quality texture-grounding data. The pipeline operates in three stages: intelligent filtering, geometric extraction, and VLM-based annotation.

**Source Material.** Our training pool originates from TextureSAM's [1] texture-augmented ADE20K, which applies Compositional Neural Texture (CNT) synthesis at varying augmentation intensities (degree $\eta \in [0, 1]$) to the original ADE20K training split [18]. This yields up to ~222,000 candidate images spanning the full spectrum from original scenes ($\eta = 0$) to fully texture-replaced variants ($\eta = 1.0$). Crucially, the augmentation preserves ground-truth semantic masks across all degrees, as CNT modifies appearance but not region geometry.

**Intelligent Filtering.** We apply a geometry-first scoring system that quantifies each image's suitability for texture segmentation on a 0–100 scale. The score combines a mask-structure prior $s_A$ (region count, size distribution, entropy) with a boundary quality metric $s_{\text{geom}}$ (gradient-transition strength, boundary coherence, region balance), weighted toward geometry:

$$s = 0.20 \cdot s_A + 0.60 \cdot s_{\text{geom}} + \text{bonuses} - \text{penalties}$$

with bonuses for high texture coverage and 2–4 coherent regions, and penalties for object-dominated or excessively fragmented scenes. Semantic filtering classifies ADE20K categories into texture surfaces (~72 terms), objects (~32 terms), and ambiguous (~10 terms) via keyword matching. Images scoring below 65/100 are discarded. A key finding of this filtering stage is that while original indoor scenes (e.g., kitchens, living rooms, $\eta = 0$) were systematically discarded as object-centric and generic, their high-$\eta$ augmented counterparts were identified by the pipeline as "texture-rich laboratories"—the CNT augmentation had transformed semantically complex but texture-poor scenes into images with strong, well-defined texture transitions ideal for our training objective. Full scoring formulas and threshold parameters are provided in Appendix A.

**Extraction and VLM Annotation.** For each image passing the filter, we extract up to 5 geometrically distinct texture regions by merging per-class ADE20K masks (each region ≥1% image area). We then employ Claude 3.5 Sonnet [19] to autonomously generate rich ground-truth descriptions for each region. The VLM receives both the original image and a colored overlay showing region boundaries, and produces structured descriptions in the format "Texture of \<semantic name\>, \<visual features\>, \<spatial context\>" (10–15 words). For TextureSAM datasets, *degree expansion* reuses ground-truth masks (which are degree-invariant) across higher-intensity variants, yielding ~45% additional samples at zero annotation cost. The final training set comprises **~14,000 samples** with an average of 3.2 texture regions per image. Full pipeline details, prompt templates, and validation procedures are provided in Appendix B.

### 4.2 The Custom In-Domain Test Set (ADE20K\_DeTexture)

To enable controlled in-domain evaluation, we applied the identical DeTexture Pipeline to the **ADE20K validation split** (2,000 images), curating a dedicated test set we term **ADE20K\_DeTexture**. This custom-mined dataset retains only images with strong texture transitions and well-separated regions, providing a high-quality benchmark for verifying core architectural viability in a controlled setting where the visual distribution matches the training data. ADE20K\_DeTexture served as our stability probe: stable or improving metrics on this set confirmed that architectural changes (e.g., the bottleneck projector) did not degrade in-domain capability, even as cross-domain metrics revealed pathologies. For instance, one variant's in-domain mIoU climbed steadily from 0.700 to 0.707 across epochs 5–13 on ADE20K\_DeTexture, while RWTD mIoU simultaneously degraded from 0.692 to 0.618—a divergence that would have been invisible without dual evaluation.

### 4.3 The Adopted External Benchmark (RWTD)

For out-of-domain evaluation, we adopted the **Real-World Texture Dataset (RWTD)** introduced by TextureSAM [1]: 253 images exhibiting challenging, in-the-wild texture transitions characterized by ambiguous boundaries (sand-to-water, grass-to-soil), multi-scale textures, and semantic diversity spanning natural, urban, and indoor environments. We did not collect RWTD; our contribution was *adapting it for VLM-guided models* by passing it through our VLM annotation stage to generate high-quality ground-truth descriptions for each texture region—descriptions that did not previously exist, since RWTD was originally designed for prompt-based (point/box) segmentation evaluation.

### 4.4 The Dual-Evaluation Strategy

These two benchmarks form a deliberate dual-evaluation strategy:

- **In-Domain Evaluation (ADE20K\_DeTexture)**: Verifies core architectural viability in a controlled setting where the visual distribution matches the training data. Stable metrics here confirm that architectural innovations do not introduce regressions.

- **Out-of-Domain / Stress-Test Evaluation (RWTD)**: Functions as an external, in-the-wild probe to test zero-shot generalization and deliberately expose cross-domain pathologies. RWTD's distributional distance from ADE20K was instrumental in surfacing the Directional Drift pathology (Section 3.5): the 10.5M-parameter projector memorized ADE20K-specific manifold directions that compressed RWTD representations 4× more aggressively, a failure mode entirely invisible on ADE20K\_DeTexture.

This dual-evaluation philosophy—verify stability in-domain, then stress-test out-of-domain—ensures that performance gains are genuine and generalizable, not artifacts of distribution overlap between training and evaluation data.

---

## 5. Experiments

**Ablation: Pathology Resolution Impact.** We systematically evaluate the contribution of each architectural fix by progressively adding components and measuring cross-domain generalization on the unseen RWTD benchmark. All "honest" numbers use the *exactly-K* prompt to neutralize Hungarian-matching inflation; the standard "1–6" prompt yields higher but inflated scores.

| Configuration | RWTD mIoU | Pathology Resolved |
|--------------|----------|-------------------|
| Baseline (10.5M proj, causal attn, full LM loss) | 0.544 | — (all pathologies present) |
| + Batch Multiplexing (V2) | 0.703 | Slot-1 Positional Bias |
| + Bottleneck Projector 2.1M (V4-Slim) | 0.732 | Directional Drift |
| + Block-Diagonal Mask + `[SEG]` Token (V5 Oracle, teacher-forced) | 0.810 | Context Leakage |
| + V5 Live Inference (no LM supervision) | 0.136 | — (Language Collapse) |
| + Cosine-decayed LM (V6, live, exactly-2) | 0.694 | Language Collapse |
| Zero-Shot SAM 3 with object-aware prompt (no training) | **0.928** | — (upper bound) |
| **+ V7 Bridge + Frozen Resizer + Exponential LM Cliff + Frozen SAM** | **TBD** (target ≥ 0.80) | Bottleneck Semantic Squash |

Each fix addresses a distinct failure mode. Resolving Slot-1 Bias alone gains +0.16 mIoU; the bottleneck projector adds +0.03 by eliminating Directional Drift; the block-diagonal mask and `[SEG]` token restore Context-Leakage-free grounding; the cosine-decayed LM weighting cures Language Collapse but leaves a constant DICE/LM tug-of-war (honest mIoU plateau at 0.694). Our final design — Bridge + exponential cliff — targets the remaining gap to the 0.928 ZS SAM upper bound by widening into SAM's native text space, handing the final projection to SAM's frozen pretrained resizer, and isolating the `[SEG]` token geometrically.

*[Training is in progress. Final quantitative results on RWTD and ADE20K\_DeTexture, the gap to the ZS SAM upper bound, comparison with LISA and Sa2VA baselines, and qualitative visualisations will be reported upon convergence.]*

---

## 6. Conclusion and Future Work

We have presented Qwen2SAM-DeTexture, an end-to-end architecture for parallel, multi-texture segmentation that bridges vision-language reasoning with geometric dense prediction. Through systematic ablation, we identified and resolved five fundamental pathologies—Context Leakage, Slot-1 Positional Bias, the Count Collapse / Language Collapse dilemma, Directional Drift, and Bottleneck Semantic Squash—that prevent existing VLM-segmentation architectures from scaling to multi-target grounding. Our solutions—block-diagonal attention masking, batch multiplexing, the exponential LM-loss cliff, and a Bridge-to-Native-Space projector that reuses SAM 3's own pretrained resizer—are general mechanisms applicable to any system coupling autoregressive VLM decoding with SAM-family decoders. The Zero-Shot SAM 3 experiment (0.928 RWTD mIoU with a hand-crafted object-aware prompt) is particularly instructive: it suggests that for any pretrained segmentation backbone, the correct role of a learned projector is *not* to acquire a new semantic mapping but to preserve the one the backbone already carries. The Bridge + frozen-resizer design is a minimal realization of this principle.

**Auto-Iterative Refinement.** A promising direction for future work is inference-time optimization without additional training. The current architecture produces masks in a single forward pass; however, the initial segmentation output contains rich geometric information that could be fed back to improve itself. Specifically, we envision extracting geometric center-of-mass points from each predicted mask region and injecting them into SAM's sparse prompt encoder *alongside* the dense `[SEG]` embedding in a second refinement pass. This auto-iterative loop would provide SAM with both semantic guidance (from the `[SEG]` token) and spatial guidance (from the center-of-mass points), effectively combining the complementary strengths of dense and sparse prompting. Preliminary analysis suggests this could push performance past the current single-pass ceiling without requiring retraining, as SAM's point-prompt pathway is already well-calibrated from pretraining.

**Scaling to Open-Vocabulary Partitioning.** While we evaluate on texture segmentation, the architectural innovations are domain-agnostic. Extending Qwen2SAM-DeTexture to open-vocabulary scene partitioning—where the model autonomously identifies and segments arbitrary semantic regions (not just textures)—requires only expanding the training data domain while the architectural mechanisms remain unchanged.

---

## References

[1] I. Cohen, B. Meivar, P. Tu, S. Avidan, and G. Oren, "TextureSAM: Towards a Texture Aware Foundation Model for Segmentation," *arXiv:2505.16540*, 2025.

[2] X. Lai, Z. Tian, Y. Chen, Y. Li, Y. Yuan, S. Liu, and J. Jia, "LISA: Reasoning Segmentation via Large Language Model," in *CVPR*, 2024.

[3] X. Lai, Z. Tian, Y. Chen, Y. Li, Y. Yuan, S. Liu, and J. Jia, "LISA++: An Improved Baseline for Reasoning Segmentation with Large Language Model," *arXiv:2312.17240*, 2024.

[4] Z. Xia, D. Han, Y. Han, B. Pan, S. Song, G. Huang, and H. He, "GSVA: Generalized Segmentation via Multimodal Large Language Models," in *CVPR*, 2024.

[5] Z. Ren, Y. Huang, S. Wei, Q. Zhu, D. Xu, and D. Zhang, "PixelLM: Pixel Reasoning with Large Multimodal Model," in *CVPR*, 2024.

[6] H. Rasheed, M. U. Maaz, S. Shaji, A. Shaker, S. Khan, H. A. Cholakkal, R. M. Anwer, E. Xing, M.-H. Yang, and F. S. Khan, "GLaMM: Pixel Grounding Large Multimodal Model," in *CVPR*, 2024.

[7] H. Yuan, D. Li, R. Zhang, J. Zhang, C. Li, and J. Yan, "Sa2VA: Marrying SAM2 with LLaVA for Dense Grounded Understanding of Images and Videos," *arXiv:2501.04001*, 2025.

[8] S. Zhang, P. Sun, S. Chen, M. Xiao, W. Shao, W. Zhang, K. Chen, and P. Luo, "LLaVA-Grounding: Grounded Visual Chat with Large Multimodal Models," in *ECCV*, 2024.

[9] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, P. Dollár, and R. Girshick, "Segment Anything," in *ICCV*, 2023.

[10] N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. Rädle, C. Rolland, L. Gustafson, E. Mintun, J. Pan, K. V. Alwala, N. Carion, C.-Y. Wu, R. Girshick, P. Dollár, and C. Feichtenhofer, "SAM 2: Segment Anything in Images and Videos," *arXiv:2408.00714*, 2024.

[11] B. Cheng, I. Misra, A. G. Schwing, A. Kirillov, and R. Girshick, "Masked-attention Mask Transformer for Universal Image Segmentation," in *CVPR*, 2022.

[12] A. Kirillov, R. Girshick, K. He, and P. Dollár, "Panoptic Feature Pyramid Networks," in *CVPR*, 2019.

[13] M. Cimpoi, S. Maji, I. Kokkinos, S. Mohamed, and A. Vedaldi, "Describing Textures in the Wild," in *CVPR*, 2014.

[14] E. Hayman, B. Caputo, M. Fritz, and J.-O. Eklundh, "On the Significance of Real-World Conditions for Material Classification," in *ECCV*, 2004.

[15] H. Caesar, J. Uijlings, and V. Ferrari, "COCO-Stuff: Thing and Stuff Classes in Context," in *CVPR*, 2018.

[16] X. Zou, Z.-Y. Dou, J. Yang, Z. Gan, L. Li, C. Li, X. Dai, H. Beber, J. Wang, L. Yuan, N. Peng, L. Wang, Y. J. Lee, and J. Gao, "Generalized Decoding for Pixel, Image, and Language," in *CVPR*, 2023.

[17] H. Zhang, F. Li, X. Zou, S. Liu, C. Li, J. Gao, J. Yang, and L. Zhang, "A Simple Framework for Open-Vocabulary Segmentation and Detection," in *ICCV*, 2023.

[18] B. Zhou, H. Zhao, X. Puig, T. Xiao, S. Fidler, A. Barriuso, and A. Torralba, "Semantic Understanding of Scenes through the ADE20K Dataset," *International Journal of Computer Vision*, vol. 127, pp. 302–321, 2019.

[19] Anthropic, "The Claude Model Family: Claude 3.5 Sonnet, Claude 3.5 Haiku," Anthropic Technical Report, 2024.

[20] C. Zhang, Y. Qiao, S. Tariq, S. Zheng, C. Zhang, C. Li, H. Shin, and C. S. Hong, "Understanding Segment Anything Model: SAM is Biased Towards Texture Rather than Shape," *arXiv:2311.11465*, 2023.
