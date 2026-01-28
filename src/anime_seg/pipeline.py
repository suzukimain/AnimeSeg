import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from huggingface_hub import list_repo_files, hf_hub_download, PyTorchModelHubMixin
import re
import os
from typing import Union, Optional
from safetensors.torch import load_file

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


class SimpleDecoder(nn.Module):
    """Minimal decoder: reshape hidden states + conv + resize."""

    def __init__(self, embed_dim: int, num_classes: int, decoder_channels: int = 256):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(embed_dim, decoder_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels, num_classes, kernel_size=1),
        )

    def forward(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        B, N, C = last_hidden_state.shape
        x = last_hidden_state[:, 1:, :]
        spatial_dim = int((N - 1) ** 0.5)
        x = x.transpose(1, 2).reshape(B, C, spatial_dim, spatial_dim)
        x = self.project(x)
        return F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)


class DINOv2ForSegmentation(nn.Module):
    """Simplified DINOv2 segmentation model."""

    def __init__(
        self,
        num_classes: int,
        model_name: str = "facebook/dinov2-base",
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        freeze_backbone: bool = True,
        decoder_channels: int = 256,
    ):
        super().__init__()

        if not HF_AVAILABLE:
            raise ImportError("transformers not installed")

        self.use_lora = use_lora
        self.backbone = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        embed_dim = self.backbone.config.hidden_size

        if use_lora and PEFT_AVAILABLE:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=['q_proj', 'v_proj'],
                lora_dropout=lora_dropout,
                bias='none',
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            self.backbone = get_peft_model(self.backbone, lora_config)

        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if 'lora' not in name.lower():
                    param.requires_grad = False

        self.decoder = SimpleDecoder(embed_dim, num_classes, decoder_channels)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=pixel_values, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        return self.decoder(last_hidden_state)


def create_model(
    num_classes: int,
    model_size: str = "base",
    use_lora: bool = True,
    lora_r: int = 8,
    lora_alpha: int = 16,
    freeze_backbone: bool = True,
    decoder_channels: int = 256,
) -> DINOv2ForSegmentation:
    model_names = {
        "small": "facebook/dinov2-small",
        "base": "facebook/dinov2-base",
        "large": "facebook/dinov2-large",
        "giant": "facebook/dinov2-giant",
    }
    return DINOv2ForSegmentation(
        num_classes=num_classes,
        model_name=model_names.get(model_size, model_names['base']),
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        freeze_backbone=freeze_backbone,
        decoder_channels=decoder_channels,
    )

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

class AnimeSegPipeline(PyTorchModelHubMixin):
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
        device: Optional[str] = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = NUM_CLASSES
        
        # Auto-detect filename if not provided
        if not filename:
            filename = self._auto_detect_latest_model(repo_id, token)
            
        print(f"Downloading model: {repo_id}/{filename}")
        checkpoint_path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
        
        # Extract model size from filename (e.g., "anime_seg_dinov2_large_v1.safetensors" -> "large")
        model_size = self._parse_model_size(filename)
        
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
        print("Loading weights...")
        state_dict = load_file(checkpoint_path)
        
        # Clean state dict keys
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        print("✓ Model ready")
    
    def _auto_detect_latest_model(self, repo_id: str, token: Optional[str]) -> str:
        """Auto-detect latest model file from repo."""
        print(f"Auto-detecting latest model in {repo_id}...")
        files = list_repo_files(repo_id=repo_id, token=token)
        
        # Pattern: models/anime_seg_dinov2_{size}_v{version}.safetensors
        pattern = re.compile(r"models/anime_seg_dinov2_(\w+)_v(\d+)\.safetensors")
        
        candidates = []
        for f in files:
            match = pattern.search(f)
            if match:
                size = match.group(1)
                version = int(match.group(2))
                candidates.append((version, f))
        
        if not candidates:
            raise ValueError(
                f"No model files found in {repo_id}. "
                f"Expected pattern: models/anime_seg_dinov2_<size>_v<version>.safetensors"
            )
        
        # Return latest version
        candidates.sort(reverse=True)
        latest_file = candidates[0][1]
        print(f"Found latest: {latest_file}")
        return latest_file
    
    def _parse_model_size(self, filename: str) -> str:
        """Extract model size from filename."""
        # e.g., "anime_seg_dinov2_large_v1.safetensors" -> "large"
        match = re.search(r"dinov2_(\w+)_v\d+", filename)
        if match:
            return match.group(1)
        # Fallback
        return "large"
    
    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image to 512x512 with ImageNet normalization."""
        img_resized = image.resize((512, 512), Image.BILINEAR)
        img_np = np.array(img_resized, dtype=np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_np = (img_np - mean) / std
        
        # Convert to tensor
        tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float()
        return tensor.to(self.device)
    
    def __call__(self, image: Union[str, Image.Image]) -> Image.Image:
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
        
        # Resize to original size
        mask_img = Image.fromarray(colored).resize(original_size, Image.NEAREST)
        
        return mask_img

