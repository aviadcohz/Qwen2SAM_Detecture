"""
Multi-texture dataset and collator for Qwen2SAM-DeTexture.

Each sample has K (1-5) textures with:
  - Image (PIL)
  - K texture descriptions (strings)
  - Index mask: (H, W) with values {0=dustbin, 1..K=textures}

The collator builds Qwen chat messages using the proven prompt template,
with N="1 to 5" during training to prevent train-test mismatch.
"""

import json
import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


# ===================================================================== #
#  SAM3 preprocessing (standalone)                                        #
# ===================================================================== #

SAM3_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
SAM3_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)
SAM3_SIZE = 1008


def preprocess_image_for_sam3(image: np.ndarray, size: int = SAM3_SIZE) -> torch.Tensor:
    """Resize and normalize RGB uint8 image → (3, size, size) float32 tensor."""
    if image.shape[0] != size or image.shape[1] != size:
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0
    image = (image - SAM3_MEAN) / SAM3_STD
    return torch.from_numpy(np.transpose(image, (2, 0, 1)))


# ===================================================================== #
#  Prompt templates (exact proven text — DO NOT MODIFY)                   #
# ===================================================================== #

SYSTEM_PROMPT = (
    "You are an expert at analyzing surface textures in images. Always respond "
    "in the exact format requested, with no extra text."
)

USER_PROMPT_TEMPLATE = (
    "This image contains exactly {N} main visually distinct regions separated "
    "by boundaries (for example, contrasting materials, surfaces, or textures).\n\n"
    "Write a single, highly descriptive phrase (approximately 10-15 words) for "
    "each region. Include the following precise information:\n"
    "1. Semantic Name: A natural, common name for the material or surface.\n"
    "2. Distinct Visual Features: The core visual attributes like color, pattern, "
    "or texture that strongly contrast with the other regions.\n"
    "3. Spatial Context: A brief note on its general position (e.g., 'foreground', "
    "'background', 'top-left', 'bottom-right', 'center', 'top-right', 'bottom-left').\n\n"
    "IMPORTANT: Describe the ENTIRE region as a collective group, NOT individual "
    "objects within it. Think of each region as a surface/area, not as a single "
    "object.\n\n"
    "Format your response exactly like this:\n"
    "TEXTURE_1: Texture of <description>\n"
    "TEXTURE_2: Texture of <description>\n"
    "...\n"
    "TEXTURE_{N}: Texture of <description>"
)

# Training: use "1 to 5" to force model to dynamically decide count
TRAIN_USER_PROMPT = USER_PROMPT_TEMPLATE.format(N="1 to 5")


def build_assistant_text(descriptions: list[str]) -> str:
    """
    Build assistant response with <TEX_i> tokens at the END of each description.

    Example output:
        "TEXTURE_1: Texture of rough mossy stone in the foreground <TEX_1>
         TEXTURE_2: Texture of smooth water in the center <TEX_2>"
    """
    lines = []
    for i, desc in enumerate(descriptions):
        lines.append(f"TEXTURE_{i+1}: {desc} <TEX_{i+1}>")
    return "\n".join(lines)


# ===================================================================== #
#  Label creation (standalone — masks system/user tokens)                 #
# ===================================================================== #

def create_labels(input_ids: torch.Tensor, attention_mask: torch.Tensor,
                  tokenizer) -> torch.Tensor:
    """
    Create LM labels: -100 for system/user/padding tokens, actual IDs for
    assistant tokens only.

    Finds the assistant turn by locating the assistant header token sequence
    in the tokenized chat.
    """
    labels = input_ids.clone()
    B, L = labels.shape

    # Find the last occurrence of the assistant turn marker
    # In Qwen's chat template, the assistant section starts after
    # the "assistant\n" tokens.
    assistant_marker = tokenizer.encode("assistant\n", add_special_tokens=False)
    marker_len = len(assistant_marker)

    for b in range(B):
        ids = input_ids[b].tolist()
        # Find last occurrence of assistant marker
        assistant_start = -1
        for i in range(len(ids) - marker_len, -1, -1):
            if ids[i:i + marker_len] == assistant_marker:
                assistant_start = i + marker_len
                break

        if assistant_start >= 0:
            # Mask everything before assistant response
            labels[b, :assistant_start] = -100
        else:
            # If not found, mask everything (safety)
            labels[b, :] = -100

        # Mask padding
        labels[b, attention_mask[b] == 0] = -100

    return labels


# ===================================================================== #
#  Dataset                                                                #
# ===================================================================== #

class DeTextureDataset(Dataset):
    """
    Multi-texture dataset.

    Expected metadata JSON format (list of dicts):
    [
        {
            "image_path": "/path/to/image.jpg",
            "textures": [
                {"description": "Texture of rough stone...", "mask_path": "/path/to/mask1.png"},
                {"description": "Texture of smooth water...", "mask_path": "/path/to/mask2.png"},
            ]
        },
        ...
    ]

    The index mask is built from per-texture binary masks:
    - Pixel belongs to texture i → value i+1 (1-indexed)
    - Pixel belongs to no texture → value 0 (dustbin)
    """

    def __init__(
        self,
        metadata_path: str,
        image_size: int = SAM3_SIZE,
        augment: bool = False,
    ):
        with open(metadata_path) as f:
            self.samples = json.load(f)
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        meta = self.samples[idx]

        # Load image
        image_np = cv2.imread(meta["image_path"])
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_np = cv2.resize(image_np, (self.image_size, self.image_size),
                              interpolation=cv2.INTER_LINEAR)

        # Load per-texture masks and build index mask
        textures = meta["textures"]
        k_gt = len(textures)
        descriptions = []
        index_mask = np.zeros((self.image_size, self.image_size), dtype=np.int64)

        for i, tex in enumerate(textures):
            descriptions.append(tex["description"])
            mask = cv2.imread(tex["mask_path"], cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask = cv2.resize(mask, (self.image_size, self.image_size),
                                  interpolation=cv2.INTER_NEAREST)
                # Assign texture index (1-based, 0 is dustbin)
                index_mask[mask > 127] = i + 1

        # Build assistant text with <TEX_i> tokens
        assistant_text = build_assistant_text(descriptions)

        # Preprocess for SAM3
        sam_image = preprocess_image_for_sam3(image_np, self.image_size)

        # PIL image for Qwen
        image_pil = Image.fromarray(image_np)

        return {
            "image": image_pil,
            "assistant_text": assistant_text,
            "sam_image": sam_image,
            "index_mask": torch.from_numpy(index_mask),
            "k_gt": k_gt,
            "descriptions": descriptions,
        }


# ===================================================================== #
#  Collator                                                               #
# ===================================================================== #

class DeTextureCollator:
    """
    Collate function for Qwen2SAM-DeTexture DataLoader.

    Builds Qwen chat messages, tokenizes, creates LM labels,
    and stacks SAM3 inputs + GT masks.
    """

    def __init__(
        self,
        processor,
        clip_model=None,
        inference: bool = False,
    ):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.clip_model = clip_model
        self.inference = inference

    def __call__(self, samples: list) -> dict:
        texts = []
        images = []

        for s in samples:
            if self.inference:
                # During inference, let model generate freely
                assistant_text = ""
            else:
                assistant_text = s["assistant_text"]

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": TRAIN_USER_PROMPT},
                    ],
                },
            ]
            if assistant_text:
                messages.append({"role": "assistant", "content": assistant_text})

            texts.append(
                self.processor.apply_chat_template(
                    messages, tokenize=False,
                    add_generation_prompt=(not assistant_text),
                )
            )
            images.append(s["image"])

        # Tokenize
        qwen_inputs = self.processor(
            text=texts, images=images, return_tensors="pt", padding=True,
        )
        qwen_inputs.pop("token_type_ids", None)

        # Labels for LM loss
        if not self.inference:
            labels = create_labels(
                qwen_inputs["input_ids"],
                qwen_inputs["attention_mask"],
                self.tokenizer,
            )
            qwen_inputs["labels"] = labels

        # Stack SAM3 inputs
        sam_images = torch.stack([s["sam_image"] for s in samples])
        index_masks = torch.stack([s["index_mask"] for s in samples])
        k_gts = torch.tensor([s["k_gt"] for s in samples], dtype=torch.long)

        # Pre-compute CLIP GT text embeddings if clip model provided
        gt_text_embeds = None
        if self.clip_model is not None and not self.inference:
            gt_text_embeds = self._encode_gt_texts(samples)

        return {
            "qwen_inputs": qwen_inputs,
            "sam_images": sam_images,
            "index_masks": index_masks,
            "k_gts": k_gts,
            "gt_text_embeds": gt_text_embeds,
        }

    @torch.no_grad()
    def _encode_gt_texts(self, samples: list) -> torch.Tensor:
        """Encode GT descriptions via frozen CLIP text encoder."""
        B = len(samples)
        embeds = torch.zeros(B, 5, self.clip_model.config.projection_dim)

        for b, s in enumerate(samples):
            descs = s["descriptions"]
            if not descs:
                continue
            import transformers
            tokenizer = transformers.CLIPTokenizer.from_pretrained(
                self.clip_model.config.name_or_path
            )
            inputs = tokenizer(descs, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.clip_model.device) for k, v in inputs.items()}
            text_features = self.clip_model.get_text_features(**inputs)
            embeds[b, :len(descs)] = text_features.cpu()

        return embeds
