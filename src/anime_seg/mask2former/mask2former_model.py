from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


class Mask2FormerAnimeSegModel(nn.Module):
    def __init__(self, base_model: str, num_classes: int = 13) -> None:
        super().__init__()
        try:
            from transformers import Mask2FormerForUniversalSegmentation
        except ImportError as exc:
            raise ImportError("transformers が必要です: pip install transformers") from exc

        self.num_classes = num_classes
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(base_model)

        if getattr(self.model.config, "num_labels", None) != num_classes:
            hidden_dim = int(getattr(self.model.config, "hidden_dim", 256))
            self.model.config.num_labels = num_classes
            self.model.class_predictor = nn.Linear(hidden_dim, num_classes + 1)

    def load_checkpoint(self, checkpoint_path: str) -> Tuple[list, list]:
        state_dict = load_file(checkpoint_path)
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

        if any(k.startswith("model.") for k in state_dict):
            state_dict = {k[len("model."):] if k.startswith("model.") else k: v for k, v in state_dict.items()}

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        return missing, unexpected

    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        h, w = pixel_values.shape[-2:]
        outputs = self.model(pixel_values=pixel_values)

        cls_logits = outputs.class_queries_logits
        mask_logits = outputs.masks_queries_logits

        cls_probs = F.softmax(cls_logits, dim=-1)[..., : self.num_classes]
        up_mask_logits = F.interpolate(mask_logits, size=(h, w), mode="bilinear", align_corners=False)
        up_mask_probs = up_mask_logits.sigmoid()

        sem_prob = torch.einsum("bqc,bqhw->bchw", cls_probs, up_mask_probs)
        sem_prob = sem_prob / sem_prob.sum(dim=1, keepdim=True).clamp(min=1e-6)
        sem_logits = torch.log(sem_prob.clamp(min=1e-6))

        return {
            "semantic_logits": sem_logits,
            "query_mask_logits": up_mask_logits,
            "query_part_logits": cls_logits,
        }
