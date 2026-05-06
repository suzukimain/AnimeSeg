from __future__ import annotations

import json
import getpass
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

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


# Explicit 31-class color palette (RGB tuples, indexed 0-30)
# Provides deterministic, visually distinct colors with LR pairs for contrast
_EXPLICIT_31CLASS_COLORS = {
    0: (0, 0, 0),               # background
    1: (40, 20, 60),            # back_hair
    2: (0, 102, 204),           # bottomwear
    3: (100, 180, 220),         # ears_left
    4: (220, 120, 80),          # ears_right
    5: (70, 120, 200),          # earwear_left
    6: (220, 100, 40),          # earwear_right
    7: (50, 100, 200),          # eyebrow_left
    8: (200, 80, 30),           # eyebrow_right
    9: (40, 80, 180),           # eyelash_left
    10: (180, 60, 20),          # eyelash_right
    11: (100, 160, 240),        # eyewear_left
    12: (240, 140, 60),         # eyewear_right
    13: (200, 240, 255),        # eyewhite_left
    14: (255, 240, 200),        # eyewhite_right
    15: (100, 150, 255),        # face
    16: (32, 64, 96),           # footwear
    17: (50, 30, 80),           # front_hair
    18: (192, 192, 192),        # handwear
    19: (200, 100, 50),         # headwear
    20: (80, 140, 220),         # irides_left (cool blue)
    21: (220, 180, 80),         # irides_right (warm gold)
    22: (204, 51, 102),         # legwear
    23: (255, 0, 150),          # mouth
    24: (210, 170, 140),        # neck
    25: (100, 100, 100),        # neckwear
    26: (255, 140, 0),          # nose
    27: (128, 128, 128),        # objects
    28: (200, 50, 50),          # tail
    29: (0, 128, 0),            # topwear
    30: (255, 255, 0),          # wings
}

def _build_id_to_color(num_classes: int) -> Dict[int, Tuple[int, int, int]]:
    if num_classes == 31:
        return _EXPLICIT_31CLASS_COLORS
    if num_classes <= 12:
        return {k: v for k, v in ID_TO_COLOR_12.items() if k < num_classes}

    id_to_color = dict(ID_TO_COLOR_12)
    for class_id in range(12, num_classes):
        hue = (class_id * 0.6180339887498949) % 1.0
        saturation = 0.55
        value = 0.92
        r, g, b = _hsv_to_rgb(hue, saturation, value)
        id_to_color[class_id] = (r, g, b)
    return id_to_color


def _hsv_to_rgb(hue: float, saturation: float, value: float) -> Tuple[int, int, int]:
    import colorsys

    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
    return (
        int(round(red * 255.0)),
        int(round(green * 255.0)),
        int(round(blue * 255.0)),
    )


def _normalize_color_triplet(color: Sequence[int]) -> Tuple[int, int, int]:
    if len(color) != 3:
        raise ValueError("class_colors entries must be RGB triplets")
    red, green, blue = (int(c) for c in color)
    for channel in (red, green, blue):
        if channel < 0 or channel > 255:
            raise ValueError("class_colors values must be in 0..255")
    return red, green, blue


class Mask2FormerAnimeSegPipeline(PyTorchModelHubMixin):
    def _download_hf_file(self, repo_id: str, filename: str, token: Optional[str]) -> str:
        try:
            return hf_hub_download(repo_id=repo_id, filename=filename, token=token)
        except Exception as exc:
            if token is not None:
                raise
            if not self._looks_like_auth_error(exc):
                raise
            prompted_token = self._prompt_for_token(repo_id)
            return hf_hub_download(repo_id=repo_id, filename=filename, token=prompted_token)

    @staticmethod
    def _looks_like_auth_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return any(
            marker in message
            for marker in (
                "401",
                "403",
                "unauthorized",
                "authentication",
                "gated",
                "token",
                "private",
            )
        )

    @staticmethod
    def _prompt_for_token(repo_id: str) -> str:
        if not sys.stdin.isatty():
            raise RuntimeError(
                f"Access to {repo_id} requires a Hugging Face token. Set HF_TOKEN or pass token=... explicitly."
            )
        token = getpass.getpass(f"Hugging Face token required for {repo_id}. Enter token: ").strip()
        if not token:
            raise RuntimeError(
                f"Access to {repo_id} requires a Hugging Face token. Set HF_TOKEN or pass token=... explicitly."
            )
        return token

    @staticmethod
    def _resolve_num_classes(config_obj: Dict) -> int:
        raw_value = (
            config_obj.get("num_classes")
            or config_obj.get("NumClasses")
            or config_obj.get("class_count")
            or config_obj.get("ClassCount")
        )
        if raw_value is None:
            class_names = config_obj.get("class_names") or config_obj.get("ClassNames")
            if isinstance(class_names, list) and class_names:
                return len(class_names)
            return DEFAULT_NUM_CLASSES
        try:
            return int(raw_value)
        except (TypeError, ValueError):
            return DEFAULT_NUM_CLASSES

    @staticmethod
    def _resolve_class_names(config_obj: Dict, num_classes: int) -> List[str]:
        raw_names = config_obj.get("class_names") or config_obj.get("ClassNames")
        if not isinstance(raw_names, list):
            return [f"class_{idx}" for idx in range(num_classes)]

        class_names = [str(name).strip() for name in raw_names if str(name).strip()]
        if len(class_names) < num_classes:
            class_names.extend(f"class_{idx}" for idx in range(len(class_names), num_classes))
        return class_names[:num_classes]

    @staticmethod
    def _resolve_class_colors(config_obj: Dict, num_classes: int) -> Dict[int, Tuple[int, int, int]]:
        raw_colors = config_obj.get("class_colors") or config_obj.get("ClassColors")
        if isinstance(raw_colors, list) and raw_colors:
            colors: Dict[int, Tuple[int, int, int]] = {}
            for class_id, color in enumerate(raw_colors[:num_classes]):
                if isinstance(color, (list, tuple)):
                    colors[class_id] = _normalize_color_triplet(color)
            if len(colors) == num_classes:
                return colors
        return _build_id_to_color(num_classes)

    @staticmethod
    def _infer_num_classes_from_checkpoint(checkpoint_path: str) -> Optional[int]:
        checkpoint_lower = checkpoint_path.lower()
        try:
            if checkpoint_lower.endswith(".safetensors"):
                state_dict = load_file(checkpoint_path)
            elif checkpoint_lower.endswith(".pt") or checkpoint_lower.endswith(".pth"):
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
                else:
                    return None
            else:
                return None
        except Exception:
            return None

        normalized = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        for candidate in ("class_predictor.weight", "model.class_predictor.weight"):
            weight = normalized.get(candidate)
            if weight is not None and hasattr(weight, "shape") and len(weight.shape) >= 1:
                return int(weight.shape[0]) - 1
        return None

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
        base_model: str = "facebook/mask2former-swin-large-ade-semantic",
        config_name: str = "config.json",
        remove_bg: bool = False,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.remove_bg = remove_bg
        self.use_amp = str(self.device).startswith("cuda")
        if self.use_amp:
            torch.backends.cudnn.benchmark = True
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

        selected_filename = filename or str(model_meta.get("FilePath", "") if model_meta else "")
        if not selected_filename:
            raise RuntimeError(
                "No FilePath found in Hugging Face config.json for requested model. "
                "Please update config.json on HF first."
            )

        selected_base_model = str(model_meta.get("BaseModel", base_model)) if model_meta else base_model
        self.train_image_size = int(model_meta.get("TrainImageSize", 768)) if model_meta else 768

        local_checkpoint_path = self._resolve_local_checkpoint_path(selected_filename)
        if local_checkpoint_path is not None:
            checkpoint_path = local_checkpoint_path
        else:
            checkpoint_path = self._download_hf_file(repo_id=repo_id, filename=selected_filename, token=token)

        inferred_num_classes = self._infer_num_classes_from_checkpoint(checkpoint_path)
        config_num_classes = self._resolve_num_classes(config_obj)
        if inferred_num_classes is not None:
            self.num_classes = inferred_num_classes
        else:
            self.num_classes = config_num_classes
        self.class_names = self._resolve_class_names(config_obj, self.num_classes)
        self.id_to_color = self._resolve_class_colors(config_obj, self.num_classes)
        merged_full = bool(config_obj.get("merged_full", False)) if config_obj else False

        inferred_merged_full = self._infer_merged_full_from_checkpoint(checkpoint_path)
        effective_merged_full = merged_full or inferred_merged_full

        fallback_base_models = [
            selected_base_model,
            "facebook/mask2former-swin-large-ade-semantic",
            "facebook/mask2former-swin-base-ade-semantic",
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
                "Failed to resolve model from config.json and fallback file pattern. "
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
            config_path = self._download_hf_file(repo_id=repo_id, filename=config_name, token=token)
        except Exception:
            return {}

        try:
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
        output_overlay: bool = False,
    ) -> Image.Image:
        if isinstance(image, str):
            source_img = Image.open(image).convert("RGB")
        else:
            source_img = image.convert("RGB")

        img = source_img
        if getattr(self, "remove_bg", False) and hasattr(self, "bg_remover"):
            # Get mask from bg_remover and cut background via OpenCV/Numpy logic
            mask = self.bg_remover(img, return_mask=True, return_type="numpy")
            img_np = np.array(img)
            bg_color = np.array([255, 255, 255], dtype=np.uint8)
            img_np = (mask * img_np + (1 - mask) * bg_color).astype(np.uint8)
            img = Image.fromarray(img_np)

        original_size = img.size
        target_size = (
            int(width) if width is not None else original_size[0],
            int(height) if height is not None else original_size[1],
        )

        if target_size[0] <= 0 or target_size[1] <= 0:
            raise ValueError("output size must be positive")
        input_tensor = self._preprocess(img)

        with torch.inference_mode():
            if self.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = self.model(input_tensor)
            else:
                outputs = self.model(input_tensor)
            preds = torch.argmax(outputs["semantic_logits"], dim=1).cpu().numpy()[0]

        h, w = preds.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        for class_id, color in self.id_to_color.items():
            colored[preds == class_id] = color

        mask_img = Image.fromarray(colored).resize(target_size, 0)
        
        if output_overlay:
            # Create overlay: 60% mask, 40% source
            source_resized = source_img.resize(target_size, Image.BILINEAR)
            mask_overlay = mask_img.convert("RGB")
            return Image.blend(source_resized, mask_overlay, 0.6)
            
        return mask_img

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
