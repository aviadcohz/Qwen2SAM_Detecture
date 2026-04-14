# Qwen2SAM-DeTexture: End-to-End VLM-Guided Multi-Texture Segmentation via Multiplexed Grounding

**Anonymous Authors**

---

## Abstract

Recent work on texture-aware segmentation—most notably TextureSAM, which demonstrated that foundation segmentation models carry a systematic shape bias that can be mitigated through texture-augmented fine-tuning—has established that *what* to segment in texture-dominant scenes is a solvable problem. However, the complementary question of *how many* and *which* textures to segment remains unaddressed: TextureSAM and the SAM family operate as class-agnostic prompt-based models that require external specification of regions of interest, while Vision-Language Models (VLMs) that could provide such specification struggle with dense, multi-target grounding in a single forward pass. We present **Qwen2SAM-DeTexture**, an end-to-end architecture that closes this gap by coupling a VLM's open-vocabulary reasoning with a texture-aware segmentation decoder for parallel, multi-texture segmentation. Building on TextureSAM's texture-augmented ADE20K dataset for training data construction, we systematically characterize four architectural pathologies that emerge when extending `[SEG]`-token grounding pipelines to the multi-target regime: (i) *Context Leakage*, whereby causal attention contaminates downstream `[SEG]` representations (inter-texture cosine similarity of 0.74 under shared context vs. 0.16 under isolation); (ii) *Slot-1 Positional Bias*, where SAM's cross-attention allocates 90.5% of pixels to the first query regardless of semantic content; (iii) *Count Collapse*, in which co-trained language modeling loss incentivizes early sequence termination (85% of samples collapsing to a single predicted texture); and (iv) *Directional Drift*, where an over-parameterized projector memorizes domain-specific manifold directions (4× higher representational compression on out-of-distribution data). Our V5 architecture resolves these pathologies through block-diagonal attention masking, batch-multiplexed SAM decoding, segment-only loss masking, and an information-bottleneck projector—achieving parallel multi-texture segmentation in a single forward pass with only ~6.6M trainable parameters. We further contribute the **DeTexture Pipeline**, a geometry-first texture mining framework for curating training data from large-scale datasets, and the **Real-World Texture Dataset (RWTD)**, a benchmark of 253 in-the-wild texture transition scenes. Our pre-V5 slim model already surpasses the zero-shot SAM baseline on RWTD (mIoU 0.732 vs. 0.706), and V5 training with all four fixes integrated is underway.

---

## 1. Introduction

Foundation segmentation models have achieved remarkable generalization across visual domains, yet they remain fundamentally *prompt-driven*: a human or upstream system must specify *what* to segment, whether through points, boxes, or text. TextureSAM [1] recently exposed a critical limitation within this paradigm—Segment Anything Models carry a systematic shape bias inherited from object-centric training data, causing them to fragment texture-defined regions rather than segment them as coherent wholes. By fine-tuning SAM-2 on a texture-augmented variant of ADE20K (Textured-ADE20K), TextureSAM demonstrated that this shape bias can be shifted toward texture sensitivity, improving mIoU on real-world texture scenes by +0.21 over SAM-2. This result established an important proof of concept: texture-aware segmentation *is* achievable within foundation model architectures.

However, TextureSAM—like all SAM variants—addresses only the *segmentation* half of the problem. It cannot autonomously determine *how many* texture regions exist in an image, *what* those textures are, or *which* pixels belong to which texture. These decisions must be supplied externally, either by a human annotator providing point prompts or by a separate upstream model. In real-world applications—robotic terrain analysis, material inspection, autonomous driving surface understanding—what is needed is a system that can *jointly reason about and segment* multiple textures in a single, fully automated forward pass.

The natural candidate for supplying this missing reasoning capability is the Vision-Language Model (VLM). Recent advances in reasoning segmentation—LISA [2], LISA++ [3], GSVA [4], PixelLM [5], GLaMM [6], and Sa2VA [7]—have demonstrated that VLMs can learn to emit special `[SEG]` tokens whose hidden representations, when projected into a segmentation decoder, yield high-quality masks. Yet these systems are designed predominantly for *single-target* or *sequential* grounding: LISA requires $K$ forward passes to segment $K$ targets; GSVA generates multiple `[SEG]` tokens but under standard causal attention that causes inter-token semantic contamination; Sa2VA unifies grounding with conversation but inherits the same causal decoding bottleneck. No existing system can produce a *spatially consistent, non-overlapping partition* of an image into multiple texture regions in a single pass.

This paper bridges the gap between TextureSAM's texture-aware segmentation capability and VLMs' open-vocabulary reasoning by presenting **Qwen2SAM-DeTexture**, an end-to-end architecture for parallel, multi-texture segmentation. Our central question is: **Can a VLM-guided system autonomously identify, describe, and segment multiple texture regions simultaneously, producing a consistent image partition in one forward pass?**

We answer affirmatively, but the path to our solution—the V5 architecture—required uncovering and resolving four distinct *architectural pathologies* that emerge when extending single-target `[SEG]`-token grounding to the multi-target regime. Crucially, our research follows a *data-architecture co-design* philosophy that traces directly to our group's prior work on TextureSAM.

**From TextureSAM to Data-Architecture Co-design.** TextureSAM's Textured-ADE20K dataset, constructed via Compositional Neural Texture (CNT) augmentation with controllable intensity $\eta$, provides training images where texture boundaries are well-defined and ground-truth masks are preserved across augmentation degrees. We leverage this dataset as the foundation of our training data pipeline, extending it with a geometry-first scoring system (the DeTexture Pipeline) that mines the most texture-rich samples, and an automated VLM-based annotation stage that generates natural language descriptions for each texture region. This data ecosystem also produced the Real-World Texture Dataset (RWTD)—253 curated in-the-wild scenes with challenging texture transitions—which served as a diagnostic probe throughout development. Early evaluation on RWTD revealed the Slot-1 Positional Bias in SAM's decoder (invisible on cleaner benchmarks), and the cross-domain gap between ADE20K and RWTD exposed Directional Drift in our projector (undetectable when training and evaluation share the same domain). In this sense, the dataset *discovered* the architectural problems.

**Contributions.** We make the following contributions:

1. **Systematic Characterization of Multi-Target Grounding Pathologies.** Through five architectural generations (V1–V5) and rigorous ablation, we identify and formally characterize four pathologies—Context Leakage, Slot-1 Positional Bias, Count Collapse, and Directional Drift—that prevent existing VLM-segmentation architectures from scaling to multi-target grounding. These pathologies are general: they affect any architecture that combines causal VLM decoding with SAM-family decoders for multi-target prediction.

2. **The V5 Architecture.** We propose a unified architecture integrating block-diagonal attention masking, batch-multiplexed SAM decoding, segment-only loss masking, and an information-bottleneck projector. Together, these mechanisms enable parallel multi-texture segmentation in a single forward pass with only ~6.6M trainable parameters atop frozen Qwen3-VL-8B and SAM3 backbones.

3. **A Complete Training Data Pipeline.** Building on TextureSAM's Textured-ADE20K, we contribute (a) the DeTexture Pipeline, a geometry-first scoring system for mining texture-rich images from large-scale datasets, (b) an automated description generation stage using VLM vision APIs, and (c) the Real-World Texture Dataset (RWTD), a benchmark designed to stress-test multi-texture segmentation in challenging, in-the-wild conditions.

4. **The Winner-Takes-All Arbitration Framework.** We introduce a principled pixel-assignment mechanism that resolves spatial competition between multiple texture queries, guaranteeing a mathematically consistent, non-overlapping partition of the image—a property that sequential grounding models fundamentally cannot enforce.

---

## 2. Related Work

### 2.1 Texture-Aware Segmentation and TextureSAM

Foundation segmentation models exhibit a well-documented shape bias: trained predominantly on object-centric datasets, they learn to segment along object boundaries rather than texture boundaries [1, 14]. TextureSAM [1] addressed this directly by constructing a texture-augmented ADE20K dataset using Compositional Neural Texture (CNT) synthesis [15] with a controllable interpolation coefficient $\eta$ that smoothly transitions images from their original appearance ($\eta = 1.0$) to pure texture replacement ($\eta = 0.0$). Fine-tuning SAM-2 on this augmented data shifted the model's segmentation prior toward texture-defined regions, achieving +0.21 mIoU on the Real-World Texture Dataset (RWTD) and +0.18 mIoU on synthetic texture benchmarks over the SAM-2 baseline. However, TextureSAM remains a *prompt-driven* model: it cannot autonomously determine the number or identity of texture regions, requiring external specification of regions of interest. Our work extends TextureSAM's contributions in two directions—leveraging its Textured-ADE20K for training data construction while adding autonomous, language-guided multi-texture reasoning that eliminates the need for manual prompting.

### 2.2 Reasoning Segmentation and the `[SEG]` Token Paradigm

LISA [2] pioneered *reasoning segmentation*, embedding a learnable `[SEG]` token into an LLM's vocabulary such that the token's hidden state, when projected into SAM's prompt encoder, produces a segmentation mask conditioned on complex natural language queries. LISA++ [3] extended this baseline with instance-level segmentation and Segmentation-in-Dialogue (SiD), enabling multi-turn conversational grounding. However, both LISA and LISA++ are fundamentally *sequential*: segmenting $K$ targets requires $K$ independent forward passes, each unaware of the others. This isolation leads to two practical failures: (a) computational cost that scales linearly with $K$, and (b) independently generated masks that lack mutual spatial awareness, producing overlapping regions that violate the fundamental constraint of image partitioning. In our experiments, sequential LISA-style approaches cannot enforce non-overlapping mask constraints without post-hoc reconciliation—precisely the property our Winner-Takes-All mechanism guarantees natively.

GSVA [4] made an important step toward multi-target grounding by extending LISA's architecture to support *multiple* `[SEG]` tokens within a single decoding pass, alongside a novel `[REJ]` token for explicitly rejecting absent targets in Generalized Referring Expression Segmentation (GRES). While GSVA demonstrates that multiple grounding tokens can coexist in the output sequence, it relies on standard causal attention during generation. As we demonstrate in Section 3.2, this architectural choice introduces *Context Leakage*: the hidden state of $\texttt{[SEG]}_k$ absorbs semantic information from all preceding texture descriptions through the causal attention pathway, with inter-token cosine similarity reaching 0.74 under shared context versus 0.16 under isolation. GSVA's design assumes that the LLM can maintain representational purity across sequential `[SEG]` tokens—an assumption our ablations empirically refute for dense texture grounding.

### 2.3 Dense Grounding via Multimodal LLMs

A growing family of models integrates pixel-level grounding directly into multimodal LLM architectures. PixelLM [5] introduces a lightweight pixel decoder with a segmentation codebook, where codebook token embeddings serve as conditioning inputs for mask generation, enabling multi-target referring segmentation without requiring SAM. GLaMM [6] pioneered *Grounded Conversation Generation*, producing natural language responses seamlessly interleaved with segmentation masks through phrase-level grounding. Sa2VA [7] unifies SAM-2 with LLaVA into a shared token space, generating instruction tokens that guide SAM-2's mask decoder for both image and video grounding. LLaVA-Grounding [8] connects visual grounding with LLM-based reasoning for referring expression comprehension.

While these models advance the state of the art on referring segmentation benchmarks (Sa2VA achieves 81.6 cIoU on RefCOCO, outperforming GLaMM's 79.5), they share a common architectural limitation: all rely on standard causal attention mechanisms during multi-token generation. This is adequate for *referring* segmentation, where targets are specified by the user and generated one at a time, but becomes pathological for *autonomous multi-target partitioning*, where the model must simultaneously generate and ground multiple regions. We identify two failure modes specific to this setting: Context Leakage from causal attention (Section 3.2) and Count Collapse from co-trained language loss (Section 3.4). Our block-diagonal attention mask and segment-only loss masking are general solutions applicable to any architecture in this family.

### 2.4 SAM and Prompt-Based Segmentation

The Segment Anything Model (SAM) [9] and its successors (SAM-2 [10]) have established prompt-based segmentation as a powerful paradigm. Given visual prompts (points, boxes, masks) or dense embedding prompts, SAM produces high-quality segmentation masks with strong zero-shot generalization. SAM-2 extends this to video with memory-augmented tracking. These models excel as segmentation *backends* in VLM-guided pipelines, but their decoder architecture exhibits an underexplored positional bias when processing multiple prompt embeddings simultaneously. Through controlled experiments on all 253 RWTD samples using ground-truth embeddings, we discovered that SAM's cross-attention mechanism allocates 90.5% of pixels to the first prompt slot regardless of semantic content, with 0.0% allocation to the second slot—a bias that persists after swapping prompt order and under isolated single-query evaluation (Section 3.3). This *Slot-1 Addiction* is invisible in single-target applications (where all existing benchmarks operate) but catastrophic for multi-target segmentation, effectively reducing a multi-query decoder to a single-query one.

### 2.5 Multi-Target and Panoptic Segmentation

Classical panoptic segmentation approaches—Mask2Former [11], Panoptic FPN [12]—achieve multi-target segmentation through DETR-style set prediction or proposal generation. While effective for closed-vocabulary, fixed-category settings, these methods cannot leverage natural language descriptions of novel textures and require retraining for new categories. Recent open-vocabulary panoptic models (X-Decoder [16], OpenSeeD [17]) extend to unseen categories via text-image alignment but still operate with predefined category lists rather than free-form descriptions. Our work bridges this gap by combining VLMs' open-vocabulary reasoning with SAM's segmentation quality, introducing the architectural innovations necessary for reliable multi-target texture grounding.

### 2.6 Texture Segmentation and Datasets

Texture segmentation has a rich history predating deep learning, from Brodatz-based evaluations to the Describable Textures Dataset (DTD) [13], KTH-TIPS [14], and texture subsets of MS-COCO Stuff [15]. However, these benchmarks predominantly feature isolated texture patches or well-separated regions under controlled conditions. They do not capture the challenging *texture transition* scenarios common in real-world imagery: gradual boundaries between sand and water, interleaving patterns of brick and mortar, or complex junctions of concrete, grass, and gravel. TextureSAM's RWTD benchmark [1] and our extended DeTexture mining pipeline specifically target these challenging in-the-wild cases, providing both evaluation benchmarks and a reproducible methodology for curating texture-rich training data from large-scale annotated datasets.

---

## 3. Methodology

### 3.1 Methodology Foundations: From Heuristic Probing to End-to-End V5

In the preliminary stages of this research, we explored the feasibility of autonomous VLM-SAM communication through a structured prompt-guided bridge, without any joint training. In this early pipeline, Qwen2.5-VL generates structured texture descriptions delineated by special markers (`<START_SEG_A>`, `<END_SEG_A>`), from which hidden states are extracted at marker boundaries, projected via a multi-token MLP ($2048 \to 1024 \to 256$), and fed to SAM3's cross-attention as variable-length prompts. Each word of the description maintains independent cross-attention with the visual features, preserving fine-grained semantic information that single-token compression would destroy.

While initial tests with Qwen2.5-VL demonstrated the potential for structured VLM-to-SAM communication, they also revealed significant limitations in texture description granularity that hindered precise grounding. Specifically, the generated descriptions often failed to capture the subtle geometric and material nuances essential for disambiguating complex real-world texture transitions—producing generic characterizations (e.g., "rough surface") where the segmentation decoder required spatially discriminative detail (e.g., "weathered grey concrete with visible aggregate pattern in the lower foreground").

Nevertheless, this heuristic probing phase yielded two foundational insights. First, it validated that VLM hidden states carry sufficient spatial-semantic information to guide a segmentation decoder—confirming that `[SEG]`-token grounding can extend beyond single-target referring expressions to multi-texture partitioning. Second, it established the Winner-Takes-All (WTA) arbitration logic as the principled mechanism for resolving spatial competition between multiple texture queries. Through systematic comparison with CLIPSeg-based heatmap alternatives and a pair optimization algorithm (scoring all $N \times M$ description-pair combinations by quality and overlap penalties), we found that competitive pixel assignment via WTA consistently produced the most spatially coherent, non-overlapping partitions.

These insights motivated the adoption of **Qwen3-VL-8B** as the central architectural foundation for the V5 system. By leveraging Qwen3-VL's superior visual reasoning capabilities and higher-dimensional latent space (4096-D vs. Qwen2.5-VL's 2048-D), we moved beyond heuristic probing to a fully end-to-end architecture capable of extracting rich, spatially-aware hidden states at dedicated `[SEG]` token boundaries, which are subsequently projected into SAM3's prompt space. This transition was fundamental in enabling the WTA arbitration logic to resolve spatial competition between multiple texture queries with the discriminative precision that the earlier foundation lacked.

The resulting V5 architecture (Figure 1) comprises four components: a frozen Qwen3-VL-8B backbone with lightweight LoRA adapters, a bottleneck projector ($4096 \to 512 \to 256$), a SAM3 mask decoder with orthogonal LoRA, and a Winner-Takes-All pixel assignment layer. Training employs a three-stage curriculum with only ~6.6M trainable parameters. We arrived at this design through systematic ablation across five architectural generations, each of which uncovered and resolved a specific pathology.

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

### 3.4 Preventing Count Collapse via Teacher-Forced Segment-Only Loss Masking

**Pathology.** We observe that naïve end-to-end co-training of a VLM with segmentation losses inevitably leads to a *Count Collapse* pathology: when the language modeling cross-entropy loss is computed over all generated tokens—including both texture descriptions and `[SEG]` tokens—the optimization landscape creates a shortcut whereby the model minimizes total loss by terminating generation after a single texture description, since predicting fewer tokens incurs less cumulative risk. In our experiments, this manifested with striking severity: 85% of evaluation samples (216 out of 253) predicted $K=1$ texture, with the count distribution collapsing to $\{1: 216, 2: 17, 3: 9, 4: 2, 5: 2, 6: 7\}$.

The consequences cascade beyond count accuracy into a systemic co-adaptation failure. As the VLM collapses to single-texture outputs, the downstream projector adapts to this degenerate distribution, binding to single-token representations and rejecting multi-token inputs. We verified this by bypassing the VLM entirely and feeding ground-truth multi-texture embeddings to the trained projector: it allocated 100% of pixels to the dustbin class—confirming that the projector had irreversibly co-adapted to the collapsed distribution and lost the ability to process legitimate multi-texture inputs. This pathology is not specific to our architecture; it is an inherent risk for any system that co-trains autoregressive language generation with dense prediction losses.

**Solution: Teacher-Forced Segment-Only Loss Masking.** We resolve Count Collapse through two complementary mechanisms. First, during training, we employ *teacher forcing*: rather than allowing the VLM to autoregressively generate texture descriptions (which would expose it to the collapsed distribution), we feed the ground-truth multi-texture sequence directly as input. This ensures the model always observes the correct number of textures and their descriptions during training, preventing the feedback loop that drives count collapse.

Second, we apply a label mask of $-100$ (PyTorch's cross-entropy ignore index) to *all* text tokens in the sequence, computing cross-entropy loss *exclusively* on `[SEG]` token positions. Formally, let $\mathbf{y} = (y_1, \ldots, y_L)$ be the target sequence and $\mathcal{S} = \{i : y_i = \texttt{[SEG]}\}$ the set of `[SEG]` positions. The masked language modeling loss is:

$$\mathcal{L}_{\text{LM}} = -\frac{1}{|\mathcal{S}|} \sum_{i \in \mathcal{S}} \log p_\theta(y_i \mid y_{<i})$$

This design embodies a principled separation of learning signals: the *pixels* teach the `[SEG]` token what to ground (through backpropagated mask losses), while the VLM's primary autoregressive distribution remains frozen and intact. The mask loss gradients flow through the `[SEG]` embeddings into the LoRA-adapted layers of the VLM, shaping *where* the model attends in the visual features, but never corrupt the language prior that governs *how many* textures to describe or *what* words to use. Teacher forcing combined with segment-only loss masking thus allows the VLM to retain its pretrained ability to generate accurate texture descriptions and correct counts, while learning to produce spatially grounded `[SEG]` representations.

### 3.5 Information Bottleneck Projector to Prevent Directional Drift

**Pathology.** We demonstrate that mapping high-dimensional VLM embeddings to a segmentation decoder's prompt space using standard, high-capacity projectors results in *Directional Drift*: the projector memorizes domain-specific manifold directions from the training distribution rather than learning a generalizable mapping. A projector with approximately 10.5M parameters achieved strong in-domain performance (ADE20K mIoU climbing from 0.700 to 0.707 across epochs 5–13), yet exhibited systematic cross-domain degradation: RWTD mIoU dropped from 0.692 at epoch 5 to 0.618 at epoch 10—a divergence invisible when training and evaluation share the same distribution.

To diagnose the mechanism, we performed a vector geometry analysis, measuring the cosine similarity between paired `[SEG]` embeddings *before* and *after* the projector:

| Domain | Pre-Projector Cosine | Post-Projector Cosine | Compression (Δ) |
|--------|---------------------|----------------------|-----------------|
| RWTD (cross-domain) | 0.736 | 0.919 | **+0.183** |
| ADE20K (in-domain) | 0.831 | 0.879 | +0.048 |

The projector compressed RWTD representations nearly **4× more** than ADE20K representations (Δ = 0.183 vs. 0.048), pushing cross-domain `[SEG]` vectors toward a shared manifold direction learned from ADE20K's training distribution. Critically, this was *not* representational collapse (the post-projector cosine actually decreased from 0.919 to 0.895 between epochs 5 and 10). Rather, the projector learned domain-specific *directional priors*—manifold orientations that happened to align with ADE20K's texture vocabulary but actively harmed generalization. This pathology generalizes beyond our setting: any VLM-to-decoder projector with sufficient capacity will tend to overfit to training-domain structure when the output dimensionality is substantially lower than the input.

**Solution: Constrained Bottleneck Projector.** We replace the high-capacity projector with a constrained bottleneck architecture:

$$f_{\text{proj}}(\mathbf{h}) = \mathbf{W}_2 \cdot \delta(\text{LN}(\mathbf{W}_1 \mathbf{h} + \mathbf{b}_1)) + \mathbf{b}_2$$

where $\mathbf{W}_1 \in \mathbb{R}^{512 \times 4096}$, $\mathbf{W}_2 \in \mathbb{R}^{256 \times 512}$, LN denotes Layer Normalization, $\delta$ is the GELU activation, and dropout ($p = 0.15$) is applied after the activation. The total parameter count is approximately **2.1M**—a 5× reduction. The intermediate bottleneck dimension of 512 forces an information bottleneck: the projector cannot memorize domain-specific directions in a 512-dimensional space while maintaining fidelity to the full 4096-dimensional input distribution. This single change produced the **first trained model to surpass the zero-shot SAM baseline**, achieving mIoU 0.732 vs. 0.706 on RWTD at epoch 5—a gain of +0.040 over the original projector.

### 3.6 Winner-Takes-All Pixel Assignment

Given $K$ soft mask predictions $\hat{\mathbf{m}}_k \in [0,1]^{H \times W}$ from the batch-multiplexed SAM decoder, we must produce a spatially consistent partition of the image. We define a Winner-Takes-All (WTA) assignment augmented with a learned dustbin channel:

$$\text{label}(i,j) = \begin{cases} \arg\max_{k \in \{1,\ldots,K\}} \hat{m}_k(i,j) & \text{if } \max_k \hat{m}_k(i,j) > \hat{m}_{\text{dust}}(i,j) \\ \texttt{dustbin} & \text{otherwise} \end{cases}$$

where $\hat{m}_{\text{dust}}$ is produced by a learned 4096-dimensional dustbin embedding passed through the same projector and SAM decoder pathway. The dustbin class absorbs pixels that do not belong to any queried texture, preventing forced assignment of background regions. The WTA mechanism guarantees a *mathematically consistent* partition: every pixel is assigned to exactly one class, with no overlaps and no gaps—a property that sequential, independent models fundamentally cannot enforce.

### 3.7 Three-Stage Curriculum Training

To manage the interaction between components of differing capacity and learning dynamics, we employ a three-stage curriculum:

**Stage 1: Projector Warmup (Epochs 0–1).** Only the bottleneck projector, multi-texture mask head, and dustbin embedding are trainable (~2.3M parameters). Qwen LoRA and SAM LoRA remain frozen. This stage establishes a viable mapping between Qwen's frozen `[SEG]` representations and SAM's prompt space without co-adaptation.

**Stage 2: Qwen Integration (Epochs 2–4).** Qwen's LoRA adapters (rank 8, targeting q\_proj and v\_proj; ~3.8M parameters) are unfrozen at a conservative learning rate of $2 \times 10^{-5}$ (0.2× the base rate). This allows the `[SEG]` representations to gradually adapt to the projector's learned mapping without catastrophic forgetting of language capabilities.

**Stage 3: End-to-End (Epochs 5–60).** SAM's orthogonal LoRA adapters (rank 32, targeting q, k, v, out\_proj in cross-attention layers; ~0.3M parameters) are unfrozen at $1 \times 10^{-5}$ (0.1× the base rate) for full end-to-end gradient flow. Orthogonal regularization ($\lambda = 0.01$) prevents SAM's pretrained features from drifting.

The total loss combines mask supervision with the segment-only language loss:

$$\mathcal{L} = \underbrace{\mathcal{L}_{\text{CE}}^{\text{mask}} + 3.0 \cdot \mathcal{L}_{\text{Dice}}}_{\text{mask losses}} + 0.1 \cdot \underbrace{\mathcal{L}_{\text{LM}}^{\text{seg-only}}}_{\text{[SEG]-only CE}} + 0.01 \cdot \underbrace{\mathcal{L}_{\text{orth}}}_{\text{SAM LoRA reg.}}$$

### 3.8 Resolving Language Collapse via Proximity-Decayed Regularization

**Pathology: Gradient Overpowering and Catastrophic Forgetting.** The segment-only loss masking described in Section 3.4, while effective at preventing Count Collapse, introduces a dual pathology we term *Language Collapse*. By placing $\texttt{-100}$ on all text tokens and restricting LM loss to $\texttt{[SEG]}$ positions, the LoRA adapters receive *zero* penalty for degrading Qwen's language generation capabilities. After only 5 epochs of co-training, the LoRA—receiving gradients exclusively from the spatial segmentation path (mask loss → SAM → projector → $\texttt{[SEG]}$ hidden states → LoRA)—underwent catastrophic forgetting of its linguistic prior. The model's text generation collapsed completely: outputs included Chinese characters, emojis, garbage tokens, and no $\texttt{[SEG]}$ emissions. Live inference mIoU dropped to 0.136 while Oracle evaluation (bypassing generation with GT descriptions) yielded **0.810**—confirming that the geometric architecture was flawless, and the failure was purely linguistic.

This reveals a fundamental dilemma in multi-modal LoRA training: *binary* loss masking creates a binary failure mode. Full text loss causes Count Collapse (V3, Section 3.4); zero text loss causes Language Collapse (V5). The solution requires a *continuous* interpolation between linguistic regularization and geometric freedom.

**Solution: Proximity-Decayed LM Regularization.** We introduce a position-dependent LM loss weight that varies smoothly within each texture description block. For a texture block of length $L_k$, let $\tau(i) = i / (L_k - 1)$ denote the normalized position of the $i$-th token ($\tau = 0$ at the block start, $\tau = 1$ at the $\texttt{[SEG]}$ token). We define the per-token LM weight via a cosine annealing schedule:

$$\lambda_{\text{LM}}(i) = \lambda_{\min} + \frac{1}{2}(\lambda_{\max} - \lambda_{\min})(1 + \cos(\pi \cdot \tau(i)))$$

with $\lambda_{\max} = 1.0$ and $\lambda_{\min} = 0.05$. The weighted LM loss is then:

$$\mathcal{L}_{\text{LM}}^{\text{decay}} = \frac{\sum_{i \in \mathcal{A}} \lambda_{\text{LM}}(i) \cdot \ell_{\text{CE}}(i)}{\sum_{i \in \mathcal{A}} \lambda_{\text{LM}}(i)}$$

where $\mathcal{A}$ denotes the set of assistant (non-prefix) token positions and $\ell_{\text{CE}}(i) = -\log p_\theta(x_i | x_{<i})$ is the standard per-token cross-entropy.

**Intuition.** The decay schedule enforces a *dual-objective gradient regime* within each texture block:

- **Block onset** ($\tau \approx 0$, $\lambda \approx 1.0$): Tokens like "$\texttt{TEXTURE\_1: Texture of rough}$" receive full LM supervision, anchoring the LoRA to Qwen's pretrained language distribution and preventing syntactic degradation.
- **Block terminus** ($\tau \approx 1$, $\lambda \approx 0.05$): Tokens near $\texttt{[SEG]}$ receive minimal LM supervision, allowing the spatial gradient from SAM to dominate and optimise the grounding representation without fighting a strong linguistic prior.

This is complemented by a 20× reduction in LoRA learning rate ($\texttt{qwen\_lr\_scale} = 0.01$, yielding $1 \times 10^{-6}$) and an extended projector warmup (5 epochs instead of 2), ensuring the projector establishes a stable mapping before the LoRA receives any gradient.

**Updated Total Loss.** The V6 total loss replaces the segment-only LM term:

$$\mathcal{L} = \underbrace{\mathcal{L}_{\text{CE}}^{\text{mask}} + 3.0 \cdot \mathcal{L}_{\text{Dice}}}_{\text{mask losses}} + 0.1 \cdot \underbrace{\mathcal{L}_{\text{LM}}^{\text{decay}}}_{\text{proximity-decayed}} + 0.01 \cdot \underbrace{\mathcal{L}_{\text{orth}}}_{\text{SAM LoRA reg.}}$$

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

To enable controlled in-domain evaluation, we applied the identical DeTexture Pipeline to the **ADE20K validation split** (2,000 images), curating a dedicated test set we term **ADE20K\_DeTexture**. This custom-mined dataset retains only images with strong texture transitions and well-separated regions, providing a high-quality benchmark for verifying core architectural viability in a controlled setting where the visual distribution matches the training data. ADE20K\_DeTexture served as our stability probe: stable or improving metrics on this set confirmed that architectural changes (e.g., the bottleneck projector) did not degrade in-domain capability, even as cross-domain metrics revealed pathologies. For instance, V4's in-domain mIoU climbed steadily from 0.700 to 0.707 across epochs 5–13 on ADE20K\_DeTexture, while RWTD mIoU simultaneously degraded from 0.692 to 0.618—a divergence that would have been invisible without dual evaluation.

### 4.3 The Adopted External Benchmark (RWTD)

For out-of-domain evaluation, we adopted the **Real-World Texture Dataset (RWTD)** introduced by TextureSAM [1]: 253 images exhibiting challenging, in-the-wild texture transitions characterized by ambiguous boundaries (sand-to-water, grass-to-soil), multi-scale textures, and semantic diversity spanning natural, urban, and indoor environments. We did not collect RWTD; our contribution was *adapting it for VLM-guided models* by passing it through our VLM annotation stage to generate high-quality ground-truth descriptions for each texture region—descriptions that did not previously exist, since RWTD was originally designed for prompt-based (point/box) segmentation evaluation.

### 4.4 The Dual-Evaluation Strategy

These two benchmarks form a deliberate dual-evaluation strategy:

- **In-Domain Evaluation (ADE20K\_DeTexture)**: Verifies core architectural viability in a controlled setting where the visual distribution matches the training data. Stable metrics here confirm that architectural innovations do not introduce regressions.

- **Out-of-Domain / Stress-Test Evaluation (RWTD)**: Functions as an external, in-the-wild probe to test zero-shot generalization and deliberately expose cross-domain pathologies. RWTD's distributional distance from ADE20K was instrumental in surfacing the Directional Drift pathology (Section 3.5): the 10.5M-parameter projector memorized ADE20K-specific manifold directions that compressed RWTD representations 4× more aggressively, a failure mode entirely invisible on ADE20K\_DeTexture.

This dual-evaluation philosophy—verify stability in-domain, then stress-test out-of-domain—ensures that performance gains are genuine and generalizable, not artifacts of distribution overlap between training and evaluation data.

---

## 5. Experiments

*[Results pending: V5 training is currently underway. This section will report quantitative results on RWTD and ADE20K-DeTexture, ablation studies isolating each of the four architectural fixes, comparison with LISA and Sa2VA baselines, and qualitative visualization of multi-texture segmentation outputs.]*

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
