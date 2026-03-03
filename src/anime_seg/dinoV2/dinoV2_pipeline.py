import torch
import numpy as np
import json
from PIL import Image
from huggingface_hub import list_repo_files, hf_hub_download, PyTorchModelHubMixin
import re
import os
from pathlib import Path
from typing import Union, Optional, Dict, List, Tuple
from safetensors.torch import load_file
from .dinoV2_model import create_model

import warnings

try:
    from transformers import AutoModel
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    warnings.warn("Transformers not installed. Install with: pip install transformers")

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


# RGB color definitions for each segmentation class
# hair_thin is rendered with darkened red, unknown is dark gray
COLORS = {
    'background': (0, 0, 0),
    'hair_main': (255, 0, 0),        # Main hair - bright red
    'hair_thin': (128, 0, 0),        # Thin hair - dark red
    'skin': (255, 220, 180),
    'face': (100, 150, 255),
    'clothes': (180, 0, 255),
    'right_eyebrow': (0, 255, 100),
    'left_eyebrow': (150, 255, 0),
    'nose': (255, 140, 0),
    'mouth': (255, 0, 150),
    'right_eye': (255, 255, 0),
    'left_eye': (0, 255, 255),
    'unknown': (64, 64, 64),         # Unknown/ignore - dark gray
}

# Explicit class ID mapping
# Order is fixed and must not change during training
CLASS_TO_ID = {
    'background': 0,
    'skin': 1,
    'face': 2,
    'hair_main': 3,       # Primary hair (thick)
    'left_eye': 4,
    'right_eye': 5,
    'left_eyebrow': 6,
    'right_eyebrow': 7,
    'nose': 8,
    'mouth': 9,
    'clothes': 10,
    'hair_thin': 11,      # Secondary hair (thin lines, ahoges)
    'unknown': 12,        # Background alternative (for clothes uncertainty)
}
ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}

# Number of classes including background
NUM_CLASSES = len(CLASS_TO_ID)
ID_TO_COLOR = {cls_id: COLORS[cls_name] for cls_name, cls_id in CLASS_TO_ID.items()}

class DinoV2AnimeSegPipeline(PyTorchModelHubMixin):
    """
    MVP Pipeline for Anime Character Segmentation using DINOv2 + a lightweight decoder.
    
    Minimal Usage:
        pipe = AnimeSegPipeline()  # Auto-loads latest version from HF
        mask = pipe(image_path)
        
    Args:
        repo_id (str): Hugging Face repository ID. Default: "suzukimain/AnimeSeg"
        filename (str): Specific model filename. If empty, auto-detects latest version.
        token (str): Hugging Face token for private repos.
        device (str): Device ('cuda' or 'cpu'). Auto-detects if None.
    """
    def __init__(
        self, 
        repo_id: str = "suzukimain/AnimeSeg",
        filename: str = "",
        token: Optional[str] = None,
        device: Optional[str] = None,
        config_name: str = "models/model_config.json",
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = NUM_CLASSES

        model_meta = self._resolve_model_meta(
            repo_id=repo_id,
            token=token,
            filename=filename,
            config_name=config_name,
        )
        selected_filename = filename or str(model_meta.get("FilePath", ""))
        self.train_image_size = int(model_meta.get("TrainImageSize", 512))

        local_checkpoint_path = self._resolve_local_checkpoint_path(selected_filename)
        if local_checkpoint_path is not None:
            checkpoint_path = local_checkpoint_path
            print(f"Loading from local file: {checkpoint_path}")
        else:
            if not selected_filename:
                selected_filename = self._auto_detect_latest_model(repo_id, token)
            print(f"Downloading model: {repo_id}/{selected_filename}")
            checkpoint_path = hf_hub_download(repo_id=repo_id, filename=selected_filename, token=token)

        model_size = self._parse_model_size(selected_filename, model_meta)
        
        # Create model with LoRA enabled, backbone frozen (MVP settings)
        print(f"Initializing {model_size} model...")
        self.model = create_model(
            num_classes=self.num_classes,
            model_size=model_size,
            use_lora=True,
            lora_r=8,
            lora_alpha=16,
            freeze_backbone=True
        )
        
        # Load weights (strict=False for LoRA weights)
        print(f"Loading weights from {checkpoint_path}...")
        checkpoint_lower = checkpoint_path.lower()
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
                raise RuntimeError("Unsupported checkpoint format in .pt/.pth file")
        else:
            raise RuntimeError(f"Unsupported checkpoint extension: {checkpoint_path}")
        
        # Clean state dict keys
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        
        if len(missing) > 0:
            print(f"Missing keys (likely backbone): {len(missing)}")
        if len(unexpected) > 0:
            print(f"Unexpected keys: {len(unexpected)}")
        
        self.model.to(self.device)
        self.model.eval()
        print("Model ready")

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

        dino_items = [x for x in entries if str(x.get("Architecture", "")).lower() == "dinov2"]
        if not dino_items:
            return {}
        dino_items.sort(key=lambda x: int(x.get("Version", 0)), reverse=True)
        return dino_items[0]
    
    def _auto_detect_latest_model(self, repo_id: str, token: Optional[str]) -> str:
        """Fallback when model_config.json is unavailable."""
        files = list_repo_files(repo_id=repo_id, token=token)

        pattern = re.compile(r"models/anime_seg_dinov2_v(\d+)\.([A-Za-z0-9]+)$")
        candidates: List[Tuple[int, str]] = []
        for f in files:
            match = pattern.search(f)
            if match:
                version = int(match.group(1))
                candidates.append((version, f))

        if not candidates:
            raise RuntimeError(
                "File management system appears to be broken. "
                "Failed to resolve model from model_config.json and fallback file pattern. "
                "Please try loading with explicit repo_id and filename."
            )

        candidates.sort(reverse=True)
        return candidates[0][1]

    def _parse_model_size(self, filename: str, model_meta: Dict) -> str:
        base_model = str(model_meta.get("BaseModel", "")).lower()
        if "giant" in base_model:
            return "giant"
        if "large" in base_model:
            return "large"
        if "small" in base_model:
            return "small"
        if "base" in base_model:
            return "base"
        match = re.search(r"dinov2_(\w+)_v\d+", filename)
        if match:
            return match.group(1)
        return "large"
    
    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image to 512x512 with ImageNet normalization."""
        img_resized = image.resize((self.train_image_size, self.train_image_size), Image.BILINEAR)
        img_np = np.array(img_resized, dtype=np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_np = (img_np - mean) / std
        
        # Convert to tensor
        tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float()
        return tensor.to(self.device)
    
    def __call__(
        self,
        image: Union[str, Image.Image],
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> Image.Image:
        """
        Run segmentation inference.
        
        Args:
            image: Input image (PIL Image or file path)
            
        Returns:
            Colored segmentation mask (PIL Image)
        """
        # Load image
        if isinstance(image, str):
            img = Image.open(image).convert('RGB')
        else:
            img = image.convert('RGB')
        
        original_size = img.size
        target_size = (
            int(width) if width is not None else original_size[0],
            int(height) if height is not None else original_size[1],
        )

        if target_size[0] <= 0 or target_size[1] <= 0:
            raise ValueError("output size must be positive")
        
        # Preprocess
        input_tensor = self._preprocess(img)
        
        # Inference
        with torch.no_grad():
            logits = self.model(input_tensor)  # (1, num_classes, 512, 512)
            preds = torch.argmax(logits, dim=1).cpu().numpy()[0]  # (512, 512)
        
        # Colorize prediction
        h, w = preds.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        for class_id, color in ID_TO_COLOR.items():
            colored[preds == class_id] = color
        
        # Resize to target size (default: original input size)
        mask_img = Image.fromarray(colored).resize(target_size, Image.NEAREST)
        
        return mask_img

    def to(self, *args, **kwargs):
        target = kwargs.get("device")
        if target is None and len(args) > 0:
            target = args[0]
        if target is None:
            return self
        self.device = str(target)
        self.model.to(target)
        return self


AnimeSegPipeline = DinoV2AnimeSegPipeline
