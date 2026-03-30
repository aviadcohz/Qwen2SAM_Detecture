"""
Qwen2SAM-DeTexture: End-to-End VLM-Guided Multi-Texture Segmentation.

Architecture:
  Qwen2.5-VL (frozen + LoRA) → generates texture descriptions with <TEX_i> tokens
    → Extract single hidden state at each <TEX_i> position
    → Build 6 query slots: [DUSTBIN, TEX_1, ..., TEX_K, PAD...]
    → MLP Projector (2048 → 1024 → 512 → 256)
    → Single batched pass through SAM3:
        Fusion Encoder (frozen + Orthogonal LoRA) with (B, 6, 256) as prompt
      → SegHead cross_attend_prompt (frozen + Orthogonal LoRA)
      → Pixel Decoder (frozen) → pixel_embed (B, 256, H, W)
      → Multi-texture mask head: einsum(queries, pixel_embed) → (B, 6, H, W)

Channel layout: [0=DUSTBIN, 1..K=textures, K+1..5=PAD]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from models.projector import DescriptionProjector
from models.orthogonal_lora import apply_orthogonal_lora_to_mha, OrthogonalLoRALinear


# ===================================================================== #
#  Constants                                                              #
# ===================================================================== #

MAX_TEXTURES = 5
NUM_QUERY_SLOTS = MAX_TEXTURES + 1  # 5 textures + 1 dustbin = 6

TEX_TOKENS = [f"<TEX_{i}>" for i in range(1, MAX_TEXTURES + 1)]


# ===================================================================== #
#  Multi-Texture Mask Head                                                #
# ===================================================================== #

class MultiTextureMaskHead(nn.Module):
    """
    Dot-product mask head: projects queries and pixel features into a shared
    space, then computes masks via einsum.

    Mimics SAM3's MaskPredictor but with our 6 query vectors.
    """

    def __init__(self, embed_dim: int = 256, mask_dim: int = 256):
        super().__init__()
        # Project query vectors
        self.query_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, mask_dim),
        )
        # Project pixel embeddings
        self.pixel_head = nn.Conv2d(embed_dim, mask_dim, kernel_size=1)

    def forward(self, queries: torch.Tensor,
                pixel_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries: (B, 6, embed_dim) — the 6 query vectors.
            pixel_embed: (B, embed_dim, H, W) — pixel features from PixelDecoder.

        Returns:
            mask_logits: (B, 6, H, W) — raw logits.
        """
        q = self.query_mlp(queries)          # (B, 6, mask_dim)
        p = self.pixel_head(pixel_embed)     # (B, mask_dim, H, W)
        return torch.einsum("bqc, bchw -> bqhw", q, p)


# ===================================================================== #
#  Qwen utilities (standalone — no cross-project imports)                 #
# ===================================================================== #

def load_qwen_processor(model_name: str):
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(model_name)


def load_qwen_model(model_name: str, dtype=torch.bfloat16):
    from transformers import Qwen3VLForConditionalGeneration
    return Qwen3VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=dtype,
    )


def apply_qwen_lora(model, cfg: dict):
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.0),
        target_modules=cfg.get("lora_target_modules", ["q_proj", "v_proj"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)


def add_tex_tokens(processor, model):
    """Add <TEX_1> through <TEX_5> special tokens."""
    tokenizer = processor.tokenizer
    num_added = tokenizer.add_tokens(TEX_TOKENS, special_tokens=True)
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
    ids = {t: tokenizer.convert_tokens_to_ids(t) for t in TEX_TOKENS}
    return ids


# ===================================================================== #
#  SAM3 loading                                                           #
# ===================================================================== #

def load_sam3(model_cfg: dict, device):
    """Load SAM3 image model."""
    import sam3 as sam3_module
    from sam3.model_builder import build_sam3_image_model

    bpe_path = str(
        Path(sam3_module.__path__[0]) / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    )
    checkpoint_path = model_cfg.get("sam3_checkpoint", None)
    return build_sam3_image_model(
        bpe_path=bpe_path,
        eval_mode=False,
        checkpoint_path=checkpoint_path,
        load_from_HF=(checkpoint_path is None),
        enable_segmentation=True,
        device=device,
    )


# ===================================================================== #
#  Main Model                                                             #
# ===================================================================== #

class Qwen2SAMDeTexture(nn.Module):
    """
    End-to-End multi-texture segmentation model.

    Combines Qwen2.5-VL (text generation) with SAM3 (segmentation) to
    produce K+1 masks (K textures + dustbin) from a single image.
    """

    def __init__(self, cfg: dict, device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device)
        self.cfg = cfg
        model_cfg = cfg["model"]

        # ---- Qwen2.5-VL ------------------------------------------------ #
        qwen_dtype = getattr(torch, model_cfg.get("qwen_dtype", "bfloat16"))
        self.processor = load_qwen_processor(model_cfg["qwen_model"])
        self.qwen = load_qwen_model(model_cfg["qwen_model"], dtype=qwen_dtype)

        # Add TEX tokens
        self.tex_token_ids = add_tex_tokens(self.processor, self.qwen)
        self.tex_id_list = [self.tex_token_ids[t] for t in TEX_TOKENS]

        # Apply LoRA to Qwen
        self.qwen = apply_qwen_lora(self.qwen, model_cfg)
        if model_cfg.get("gradient_checkpointing", True):
            self.qwen.enable_input_require_grads()
            self.qwen.gradient_checkpointing_enable()
        self.qwen.to(self.device)

        # ---- SAM3 ------------------------------------------------------ #
        self.sam3 = load_sam3(model_cfg, self.device)
        self._freeze_sam3()
        self.sam3_lora_modules = self._apply_sam3_orthogonal_lora(model_cfg)
        self.sam3.to(device=self.device)

        # ---- LLM dim --------------------------------------------------- #
        qwen_cfg = getattr(self.qwen.config, "text_config", self.qwen.config)
        self.llm_dim = qwen_cfg.hidden_size  # 2048 for Qwen2.5-VL-3B

        # ---- MLP Projector ---------------------------------------------- #
        self.projector = DescriptionProjector(
            llm_dim=self.llm_dim, sam_dim=256,
        ).to(self.device)

        # ---- Learnable DUSTBIN embedding -------------------------------- #
        self.dustbin_embed = nn.Parameter(
            torch.randn(1, 1, self.llm_dim, device=self.device) * 0.02
        )

        # ---- Multi-texture mask head ------------------------------------ #
        self.mask_head = MultiTextureMaskHead(
            embed_dim=256, mask_dim=256,
        ).to(self.device)

        # ---- Text alignment head (for contrastive loss) ----------------- #
        clip_dim = model_cfg.get("clip_dim", 512)
        if clip_dim > 0:
            self.text_align_head = nn.Linear(self.llm_dim, clip_dim).to(self.device)
        else:
            self.text_align_head = None

    # ------------------------------------------------------------------ #
    #  SAM3 freezing + Orthogonal LoRA                                     #
    # ------------------------------------------------------------------ #

    def _freeze_sam3(self):
        """Freeze ALL SAM3 parameters."""
        for param in self.sam3.parameters():
            param.requires_grad = False

    def _apply_sam3_orthogonal_lora(self, model_cfg: dict) -> list:
        """
        Apply Orthogonal LoRA to the Fusion Encoder's cross-attention layers
        and the SegHead's cross_attend_prompt.

        Returns list of all OrthogonalLoRALinear modules for penalty collection.
        """
        r = model_cfg.get("sam3_lora_r", 8)
        alpha = model_cfg.get("sam3_lora_alpha", 16.0)
        n_sing = model_cfg.get("sam3_lora_n_singular", 32)
        targets = tuple(model_cfg.get("sam3_lora_targets", ["q", "v"]))

        all_lora_modules = []

        # 1. Fusion Encoder layers — cross_attn_image (image ↔ text prompt)
        for layer in self.sam3.transformer.encoder.layers:
            if hasattr(layer, "cross_attn_image"):
                mha = layer.cross_attn_image
                lora_dict = apply_orthogonal_lora_to_mha(
                    mha, r=r, alpha=alpha, n_singular=n_sing,
                    target_projections=targets,
                )
                all_lora_modules.extend(lora_dict.values())

        # 2. SegHead cross_attend_prompt
        seg_head = self.sam3.segmentation_head
        if seg_head is not None and hasattr(seg_head, "cross_attend_prompt"):
            if seg_head.cross_attend_prompt is not None:
                lora_dict = apply_orthogonal_lora_to_mha(
                    seg_head.cross_attend_prompt, r=r, alpha=alpha,
                    n_singular=n_sing, target_projections=targets,
                )
                all_lora_modules.extend(lora_dict.values())

        n_lora = len(all_lora_modules)
        n_params = sum(
            p.numel() for m in all_lora_modules
            for p in [m.lora_A, m.lora_B]
        )
        print(f"  Orthogonal LoRA: {n_lora} modules, {n_params:,} params")
        return all_lora_modules

    # ------------------------------------------------------------------ #
    #  Token extraction                                                    #
    # ------------------------------------------------------------------ #

    def extract_tex_hidden_states(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> tuple:
        """
        Extract the single hidden state at each <TEX_i> token position.

        Causal LMs concentrate context in the final token of each description,
        which is the <TEX_i> marker appended at the end.

        Args:
            hidden_states: (B, seq_len, llm_dim) from Qwen last layer.
            input_ids: (B, seq_len) token IDs.

        Returns:
            tex_embeds: (B, 5, llm_dim) — hidden states at TEX tokens (zero for missing).
            k_preds: (B,) — number of TEX tokens found per sample.
        """
        B = hidden_states.shape[0]
        tex_embeds = torch.zeros(B, MAX_TEXTURES, self.llm_dim,
                                 device=hidden_states.device,
                                 dtype=hidden_states.dtype)
        k_preds = torch.zeros(B, dtype=torch.long, device=hidden_states.device)

        for b in range(B):
            ids = input_ids[b]
            count = 0
            for i, tex_id in enumerate(self.tex_id_list):
                positions = (ids == tex_id).nonzero(as_tuple=True)[0]
                if len(positions) > 0:
                    # Take the first (and should be only) occurrence
                    pos = positions[0].item()
                    tex_embeds[b, i] = hidden_states[b, pos]
                    count = i + 1  # TEX tokens are sequential
            k_preds[b] = count

        return tex_embeds, k_preds

    # ------------------------------------------------------------------ #
    #  Query slot assembly                                                 #
    # ------------------------------------------------------------------ #

    def build_query_slots(
        self,
        tex_embeds: torch.Tensor,
        k_preds: torch.Tensor,
    ) -> tuple:
        """
        Build the 6-slot query sequence: [DUSTBIN, TEX_1, ..., TEX_K, PAD...]

        Args:
            tex_embeds: (B, 5, llm_dim) — hidden states at TEX tokens.
            k_preds: (B,) — number of predicted textures per sample.

        Returns:
            query_embeds: (B, 6, llm_dim) — full query sequence.
            pad_mask: (B, 6) — True for PAD slots.
        """
        B = tex_embeds.shape[0]
        device = tex_embeds.device

        # Slot 0: DUSTBIN (broadcast learned embedding)
        dustbin = self.dustbin_embed.expand(B, 1, self.llm_dim)

        # Slots 1..5: TEX embeddings (already zero-padded for missing ones)
        query_embeds = torch.cat([dustbin, tex_embeds], dim=1)  # (B, 6, llm_dim)

        # PAD mask: channels beyond k_pred+1 (dustbin + k textures)
        pad_mask = torch.zeros(B, NUM_QUERY_SLOTS, dtype=torch.bool, device=device)
        for b in range(B):
            kp = int(k_preds[b].item())
            # Channels kp+1 .. 5 are PAD (channel 0 is dustbin, always active)
            pad_mask[b, kp + 1:] = True

        return query_embeds, pad_mask

    # ------------------------------------------------------------------ #
    #  SAM3 pipeline                                                       #
    # ------------------------------------------------------------------ #

    def _get_img_feats(self, backbone_out, img_ids):
        """Extract image features from SAM3 backbone for the fusion encoder."""
        n_levels = self.sam3.num_feature_levels
        vis_feats = backbone_out["backbone_fpn"][-n_levels:]
        vis_pos_enc = backbone_out["vision_pos_enc"][-n_levels:]
        vis_feat_sizes = [x.shape[-2:] for x in vis_pos_enc]

        img_feats = [x[img_ids].flatten(2).permute(2, 0, 1) for x in vis_feats]
        img_pos_embeds = [x[img_ids].flatten(2).permute(2, 0, 1) for x in vis_pos_enc]
        return img_feats, img_pos_embeds, vis_feat_sizes

    def run_sam3_semantic(
        self,
        backbone_out: dict,
        query_256: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single batched pass through SAM3's semantic path:
        Fusion Encoder → cross_attend_prompt → Pixel Decoder → mask head.

        Args:
            backbone_out: dict from sam3.backbone.forward_image()
            query_256: (B, 6, 256) projected query vectors.
            pad_mask: (B, 6) True for PAD slots.

        Returns:
            mask_logits: (B, 6, H, W) raw logits.
        """
        B = query_256.shape[0]
        device = query_256.device
        img_ids = torch.arange(B, device=device)

        img_feats, img_pos_embeds, vis_feat_sizes = self._get_img_feats(
            backbone_out, img_ids
        )

        # Prompt in seq-first format: (6, B, 256)
        prompt = query_256.transpose(0, 1)
        prompt_pos = torch.zeros_like(prompt)

        # Fusion Encoder: cross-attends image features with all 6 queries
        memory_dict = self.sam3.transformer.encoder(
            src=[f.clone() for f in img_feats],
            src_key_padding_mask=None,
            src_pos=[p.clone() for p in img_pos_embeds],
            prompt=prompt,
            prompt_pos=prompt_pos,
            prompt_key_padding_mask=pad_mask,
            feat_sizes=vis_feat_sizes,
        )
        encoder_hidden_states = memory_dict["memory"]  # (HW, B, 256)

        # SegHead cross_attend_prompt: enc_hs attends to our 6 queries
        seg_head = self.sam3.segmentation_head
        enc_hs = encoder_hidden_states
        if seg_head.cross_attend_prompt is not None:
            tgt2 = seg_head.cross_attn_norm(enc_hs)
            tgt2 = seg_head.cross_attend_prompt(
                query=tgt2, key=prompt, value=prompt,
                key_padding_mask=pad_mask,
            )[0]
            enc_hs = tgt2 + enc_hs

        # Pixel Decoder: enc_hs + backbone FPN → pixel_embed (B, 256, H, W)
        pixel_embed = seg_head._embed_pixels(
            backbone_feats=backbone_out["backbone_fpn"],
            image_ids=img_ids,
            encoder_hidden_states=enc_hs,
        )

        # Multi-texture mask head: dot product → (B, 6, H, W)
        mask_logits = self.mask_head(query_256, pixel_embed)

        return mask_logits

    # ------------------------------------------------------------------ #
    #  Full forward pass                                                   #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        qwen_inputs: dict,
        sam_images: torch.Tensor,
        seg_grad_to_lm: bool = True,
    ) -> dict:
        """
        Full E2E forward pass.

        Args:
            qwen_inputs: dict from Qwen processor (input_ids, attention_mask, etc.)
            sam_images: (B, 3, 1008, 1008) SAM3-preprocessed images.
            seg_grad_to_lm: Whether segmentation gradients flow back to Qwen LoRA.

        Returns:
            dict with:
                mask_logits: (B, 6, H, W) raw logits.
                tex_embeds: (B, 5, llm_dim) hidden states at TEX tokens.
                text_align_embeds: (B, 5, clip_dim) projected for contrastive loss.
                k_preds: (B,) predicted texture counts.
                pad_mask: (B, 6) True for PAD channels.
                lm_loss: scalar Qwen LM loss.
        """
        # Step 1: Qwen forward (teacher forcing)
        qwen_out = self.qwen(**qwen_inputs, output_hidden_states=True)
        hidden_states = qwen_out.hidden_states[-1]  # (B, seq, llm_dim)
        lm_loss = qwen_out.loss if qwen_out.loss is not None else torch.tensor(0.0, device=self.device)

        # Step 2: Extract <TEX_i> hidden states
        input_ids = qwen_inputs["input_ids"]
        tex_embeds, k_preds = self.extract_tex_hidden_states(hidden_states, input_ids)

        # Step 3: Text alignment embeddings (for contrastive loss)
        if self.text_align_head is not None:
            text_align_embeds = self.text_align_head(tex_embeds.float())
        else:
            text_align_embeds = tex_embeds

        # Step 4: Optionally detach before segmentation path
        if not seg_grad_to_lm:
            tex_embeds = tex_embeds.detach()

        # Step 5: Build 6 query slots [DUSTBIN, TEX_1..5]
        query_embeds, pad_mask = self.build_query_slots(tex_embeds, k_preds)

        # Step 6: MLP Projection → (B, 6, 256)
        query_256 = self.projector(query_embeds)

        # Step 7: SAM3 backbone (frozen)
        with torch.no_grad():
            backbone_out = self.sam3.backbone.forward_image(sam_images)
            backbone_out["img_batch_all_stages"] = sam_images

        # Step 8: SAM3 semantic path → (B, 6, H, W)
        mask_logits = self.run_sam3_semantic(backbone_out, query_256, pad_mask)

        return {
            "mask_logits": mask_logits,
            "tex_embeds": tex_embeds,
            "text_align_embeds": text_align_embeds,
            "k_preds": k_preds,
            "pad_mask": pad_mask,
            "lm_loss": lm_loss,
        }

    # ------------------------------------------------------------------ #
    #  Parameter groups                                                    #
    # ------------------------------------------------------------------ #

    def get_parameter_groups(self, base_lr: float) -> list:
        sam3_lr_scale = self.cfg["model"].get("sam3_lr_scale", 0.1)

        qwen_params = [p for p in self.qwen.parameters() if p.requires_grad]
        proj_params = list(self.projector.parameters())
        mask_head_params = list(self.mask_head.parameters())
        dustbin_params = [self.dustbin_embed]

        # Orthogonal LoRA params (inside SAM3)
        sam3_lora_params = []
        for m in self.sam3_lora_modules:
            sam3_lora_params.extend([m.lora_A, m.lora_B])

        groups = [
            {"params": qwen_params, "lr": base_lr, "name": "qwen_lora"},
            {"params": proj_params, "lr": base_lr, "name": "projector"},
            {"params": mask_head_params, "lr": base_lr, "name": "mask_head"},
            {"params": dustbin_params, "lr": base_lr, "name": "dustbin"},
        ]
        if sam3_lora_params:
            groups.append({
                "params": sam3_lora_params,
                "lr": base_lr * sam3_lr_scale,
                "name": "sam3_orth_lora",
            })
        if self.text_align_head is not None:
            groups.append({
                "params": list(self.text_align_head.parameters()),
                "lr": base_lr,
                "name": "text_align",
            })
        return groups

    def num_trainable_params(self) -> dict:
        qwen_n = sum(p.numel() for p in self.qwen.parameters() if p.requires_grad)
        proj_n = sum(p.numel() for p in self.projector.parameters())
        mask_head_n = sum(p.numel() for p in self.mask_head.parameters())
        dustbin_n = self.dustbin_embed.numel()
        sam3_lora_n = sum(
            p.numel() for m in self.sam3_lora_modules
            for p in [m.lora_A, m.lora_B]
        )
        align_n = sum(p.numel() for p in self.text_align_head.parameters()) if self.text_align_head else 0

        total = qwen_n + proj_n + mask_head_n + dustbin_n + sam3_lora_n + align_n
        return {
            "qwen_lora": qwen_n,
            "projector": proj_n,
            "mask_head": mask_head_n,
            "dustbin": dustbin_n,
            "sam3_orth_lora": sam3_lora_n,
            "text_align": align_n,
            "total": total,
        }
