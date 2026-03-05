from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download, list_repo_files
from PIL import Image
from safetensors.torch import load_file

from .mask2former_model import Mask2FormerAnimeSegModel


DEFAULT_NUM_CLASSES = 12

# Mask2Former training scripts use 12 classes (id=11 is accessory).
ID_TO_COLOR_12 = {
    0: (0, 0, 0),
    1: (255, 220, 180),
    2: (100, 150, 255),
    3: (255, 0, 0),
    4: (0, 255, 255),
    5: (255, 255, 0),
    6: (150, 255, 0),
    7: (0, 255, 100),
    8: (255, 140, 0),
    9: (255, 0, 150),
    10: (180, 0, 255),
    11: (128, 128, 0),
}


def _build_id_to_color(num_classes: int) -> Dict[int, Tuple[int, int, int]]:
    if num_classes <= 12:
        return {k: v for k, v in ID_TO_COLOR_12.items() if k < num_classes}

    id_to_color = dict(ID_TO_COLOR_12)
    # Additional classes are rendered as dark gray by default.
    for class_id in range(12, num_classes):
        id_to_color[class_id] = (64, 64, 64)
    return id_to_color


class Mask2FormerAnimeSegPipeline(PyTorchModelHubMixin):
    def _infer_merged_full_from_checkpoint(self, checkpoint_path: str) -> bool:
        lower = checkpoint_path.lower()
        try:
            if lower.endswith(".safetensors"):
                keys = list(load_file(checkpoint_path).keys())
            elif lower.endswith(".pt") or lower.endswith(".pth"):
                raw = torch.load(checkpoint_path, map_location="cpu")
                if isinstance(raw, dict):
                    candidate_keys = ["state_dict", "model_state_dict", "model", "module"]
                    resolved = None
                    for key in candidate_keys:
                        val = raw.get(key)
                        if isinstance(val, dict):
                            resolved = val
                            break
                    state_dict = resolved if resolved is not None else raw
                    keys = list(state_dict.keys())
                else:
                    return False
            else:
                return False
        except Exception:
            return False

        normalized = [k.replace("_orig_mod.", "") for k in keys]
        return any(
            k.startswith("model.pixel_level_module.encoder.embeddings.patch_embeddings.projection")
            or k.startswith("pixel_level_module.encoder.embeddings.patch_embeddings.projection")
            for k in normalized
        )

    def _resolve_local_checkpoint_path(self, path_str: str) -> Optional[str]:
        if not path_str:
            return None

        candidate = Path(path_str)
        if candidate.is_file():
            return str(candidate)

        root_candidate = Path.cwd() / path_str
        if root_candidate.is_file():
            return str(root_candidate)

        return None

    def __init__(
        self,
        repo_id: str = "suzukimain/AnimeSeg",
        filename: str = "",
        token: Optional[str] = None,
        device: Optional[str] = None,
        base_model: str = "facebook/mask2former-swin-base-ade-semantic",
        config_name: str = "models/model_config.json",
        remove_bg: bool = True,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.remove_bg = remove_bg
        if self.remove_bg:
            from anime_seg.remove_bg.bg_remover_pipeline import BgRemover
            self.bg_remover = BgRemover.from_single_file(device=self.device)

        model_meta = self._resolve_model_meta(
            repo_id=repo_id,
            token=token,
            filename=filename,
            config_name=config_name,
        )

        config_obj = model_meta.get("Config", {}) if isinstance(model_meta, dict) else {}
        if not isinstance(config_obj, dict):
            config_obj = {}
        self.num_classes = int(config_obj.get("num_classes", DEFAULT_NUM_CLASSES))
        merged_full = bool(config_obj.get("merged_full", False))
        self.id_to_color = _build_id_to_color(self.num_classes)

        selected_filename = filename or model_meta.get("FilePath", "")
        selected_base_model = model_meta.get("BaseModel", base_model)
        self.train_image_size = int(model_meta.get("TrainImageSize", 768))

        local_checkpoint_path = self._resolve_local_checkpoint_path(selected_filename)
        if local_checkpoint_path is not None:
            checkpoint_path = local_checkpoint_path
        else:
            if not selected_filename:
                selected_filename = self._auto_detect_latest_model(repo_id, token)
            checkpoint_path = hf_hub_download(repo_id=repo_id, filename=selected_filename, token=token)

        inferred_merged_full = self._infer_merged_full_from_checkpoint(checkpoint_path)
        effective_merged_full = merged_full or inferred_merged_full

        fallback_base_models = [
            selected_base_model,
            "facebook/mask2former-swin-base-ade-semantic",
            "facebook/mask2former-swin-large-ade-semantic",
        ]
        unique_base_models: List[str] = []
        for candidate in fallback_base_models:
            if candidate and candidate not in unique_base_models:
                unique_base_models.append(candidate)

        last_error: Optional[Exception] = None
        model_impl: Optional[Mask2FormerAnimeSegModel] = None
        for candidate_base_model in unique_base_models:
            try:
                model = Mask2FormerAnimeSegModel(
                    base_model=candidate_base_model,
                    num_classes=self.num_classes,
                    load_base_pretrained=not effective_merged_full,
                )
                model.load_checkpoint(checkpoint_path)
                model_impl = model
                break
            except RuntimeError as exc:
                last_error = exc

        if model_impl is None:
            raise RuntimeError(
                "Failed to load mask2former checkpoint with all candidate base models. "
                f"Tried: {unique_base_models}"
            ) from last_error

        self.model = model_impl
        self.model.to(self.device)
        self.model.eval()

    def _auto_detect_latest_model(self, repo_id: str, token: Optional[str]) -> str:
        files = list_repo_files(repo_id=repo_id, token=token)
        pattern = re.compile(r"models/anime_seg_mask2former_v(\d+)\.([A-Za-z0-9]+)$")
        candidates: List[Tuple[int, str]] = []
        for file_path in files:
            match = pattern.search(file_path)
            if match:
                candidates.append((int(match.group(1)), file_path))
        if not candidates:
            raise RuntimeError(
                "File management system appears to be broken. "
                "Failed to resolve model from model_config.json and fallback file pattern. "
                "Please try loading with explicit repo_id and filename."
            )
        candidates.sort(reverse=True)
        return candidates[0][1]

    def _resolve_model_meta(
        self,
        repo_id: str,
        token: Optional[str],
        filename: str,
        config_name: str,
    ) -> Dict:
        try:
            if os.path.isfile(config_name):
                config_path = config_name
            else:
                config_path = hf_hub_download(repo_id=repo_id, filename=config_name, token=token)
            with open(config_path, "r", encoding="utf-8") as file:
                data = json.load(file)
        except Exception:
            return {}

        entries: List[Dict]
        if isinstance(data, dict) and isinstance(data.get("models"), list):
            entries = [x for x in data["models"] if isinstance(x, dict)]
        elif isinstance(data, list):
            entries = [x for x in data if isinstance(x, dict)]
        else:
            return {}

        if filename:
            for item in entries:
                if item.get("FilePath") == filename:
                    return item

        mask_items = [x for x in entries if str(x.get("Architecture", "")).lower() == "mask2former"]
        if not mask_items:
            return {}
        mask_items.sort(key=lambda x: int(x.get("Version", 0)), reverse=True)
        return mask_items[0]

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        img_resized = image.resize((self.train_image_size, self.train_image_size), 2)
        img_np = np.array(img_resized, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_np = (img_np - mean) / std
        tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float()
        return tensor.to(self.device)

    def __call__(
        self,
        image: Union[str, Image.Image],
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> Image.Image:
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.convert("RGB")

        if getattr(self, "remove_bg", False) and hasattr(self, "bg_remover"):
            img = self.bg_remover(img)

        original_size = img.size
        target_size = (
            int(width) if width is not None else original_size[0],
            int(height) if height is not None else original_size[1],
        )

        if target_size[0] <= 0 or target_size[1] <= 0:
            raise ValueError("output size must be positive")
        input_tensor = self._preprocess(img)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            preds = torch.argmax(outputs["semantic_logits"], dim=1).cpu().numpy()[0]

        h, w = preds.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        for class_id, color in self.id_to_color.items():
            colored[preds == class_id] = color

        return Image.fromarray(colored).resize(target_size, 0)

    def to(self, *args, **kwargs):
        target = kwargs.get("device")
        if target is None and len(args) > 0:
            target = args[0]
        if target is None:
            return self
        self.device = str(target)
        self.model.to(target)
        if getattr(self, "remove_bg", False) and hasattr(self, "bg_remover"):
            self.bg_remover.to(target)
        return self
