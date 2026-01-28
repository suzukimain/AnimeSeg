import torch
import numpy as np
from PIL import Image
from huggingface_hub import list_repo_files, hf_hub_download, PyTorchModelHubMixin
import re
import os
from typing import Union, List, Optional
import torch.nn.functional as F

from .modeling import create_model
from .config import NUM_CLASSES, ID_TO_COLOR, CLASS_TO_ID
from safetensors.torch import load_file

class AnimeSegPipeline(PyTorchModelHubMixin):
    """
    Pipeline for Anime Character Segmentation using DINOv2 + U-Net++.
    
    Usage:
        pipe = AnimeSegPipeline() # Loads latest version automatically
        mask = pipe(image)
        
    Args:
        repo_id (str): Hugging Face repository ID.
        model_size (str): Model size ('base', 'large', 'small', 'giant').
        version (str): Specific version to load (e.g. 'v1'). If None, loads latest.
        token (str): Hugging Face token.
        device (str): Device to use ('cuda' or 'cpu').
    """
    def __init__(
        self, 
        repo_id: str = "suzukimain/AnimeSeg", 
        model_size: str = "large", 
        version: Optional[str] = None, 
        token: Optional[str] = None, 
        device: Optional[str] = None,
        filename: Optional[str] = None
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.num_classes = NUM_CLASSES
        self.model_size = model_size
        self.patch_size = 14
        
        # Check if repo_id is actually a local file path
        if os.path.isfile(repo_id):
            checkpoint_path = repo_id
            print(f"Loading weights from local file: {checkpoint_path}")
        else:
            # Resolve filename if not provided
            if not filename:
                 filename = self._resolve_model_file(repo_id, model_size, version, token)
            
            print(f"Loading AnimeSeg model from {repo_id}/{filename}...")
            # Download weights
            checkpoint_path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
        
        # Initialize Model
        # MVP: LoRA is enabled, backbone frozen
        self.model = create_model(
            num_classes=self.num_classes, 
            model_size=model_size,
            use_lora=True,
            lora_r=8,
            lora_alpha=16,
            freeze_backbone=True,
            use_attention=True
        )
        
        # Load Weights
        # Use safe_load_file or torch.load depending on extension
        print(f"Loading weights from {checkpoint_path}...")
        if checkpoint_path.endswith('.safetensors'):
            state_dict = load_file(checkpoint_path)
        else:
            # Fallback for .pt files
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        
        # Handle state dict keys if needed
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("_orig_mod.", "")
            new_state_dict[k] = v
            
        # strict=False is important for LoRA/Partial weights
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully.")

    def _resolve_model_file(self, repo_id, model_size, version, token):
        """Find the latest model file matching the pattern."""
        print(f"Searching for {model_size} models in {repo_id}...")
        files = list_repo_files(repo_id=repo_id, token=token)
        
        # Pattern: models/anime_seg_{arch}_{size}_v{ver}.safetensors
        # Assuming arch is dinov2
        pattern_str = f"models/anime_seg_dinov2_{model_size}_v(\\d+)\\.safetensors"
        pattern = re.compile(pattern_str)
        
        candidates = []
        for f in files:
            match = pattern.search(f)
            if match:
                ver = int(match.group(1))
                candidates.append((ver, f))
        
        if not candidates:
            # Try looser pattern if stricter one fails
            # Maybe architecture is not in name?
            # "models/anime_seg_{model_size}_v{version}.safetensors"
            pattern_str_loose = f"models/anime_seg_.*{model_size}_v(\\d+)\\.safetensors"
            pattern_loose = re.compile(pattern_str_loose)
            for f in files:
                match = pattern_loose.search(f)
                if match:
                    ver = int(match.group(1))
                    candidates.append((ver, f))
        
        if not candidates:
            raise ValueError(f"No model files found in {repo_id} for size '{model_size}'. Expected pattern like 'models/anime_seg_dinov2_{model_size}_v1.safetensors'.")
            
        # Sort by version descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        latest_ver, latest_file = candidates[0]
        
        if version:
            # Find specific version
            found = False
            for v, f in candidates:
                if str(v) == str(version) or f"v{version}" in f:
                    return f
            if not found:
                 print(f"Version {version} not found. Loading latest v{latest_ver} instead.")
        
        return latest_file

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Process image for DINOv2 (Resize to multiple of 14, Normalize)."""
        w, h = image.size
        
        # Target size: next multiple of patch_size (14)
        # Ideally keeping aspect ratio, but for simplicity we assume square resize 
        # as the model requires square features for simple reshaping.
        # User's v5 pipeline uses 518x518 to fix artifacts.
        # But MVP script specifies 512.
        target_size = 512 
        
        img_resized = image.resize((target_size, target_size), Image.BILINEAR)
        img_np = np.array(img_resized)
        
        # Normalize
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        return img_tensor.unsqueeze(0).to(self.device), (w, h)

    def __call__(
        self, 
        images: Union[Image.Image, str, List[Union[Image.Image, str]]], 
        return_color: bool = False,
        alpha: float = 0.6
    ) -> Union[Image.Image, List[Image.Image]]:
        """
        Run inference.
        
        Args:
            images: PIL Image or path, or list of them.
            return_color: If True, returns visualization (blended image). If False, returns class ID mask.
            alpha: blending factor for color mode.
            
        Returns:
            PIL Image(s) (Mask or Blended)
        """
        is_batch = isinstance(images, list)
        if not is_batch:
            images = [images]
            
        results = []
        
        for img_input in images:
            if isinstance(img_input, str):
                img = Image.open(img_input).convert('RGB')
            else:
                img = img_input.convert('RGB')
                
            input_tensor, original_size = self.preprocess(img)
            
            with torch.no_grad():
                logits = self.model(input_tensor) # (1, num_classes, 512, 512)
                preds = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy() # (512, 512)
            
            orig_w, orig_h = original_size
            
            if return_color:
                # Colorize 512x512 first
                h, w = preds.shape
                mask_overlay = np.zeros((h, w, 3), dtype=np.uint8)
                for class_id, color in ID_TO_COLOR.items():
                    if class_id == 0: continue
                    mask_overlay[preds == class_id] = color
                
                # Resize colored mask to original size
                mask_pil = Image.fromarray(mask_overlay).resize((orig_w, orig_h), Image.NEAREST)
                
                # Blend with original image
                img_np = np.array(img.resize((orig_w, orig_h))) # Ensure img is same size if it wasn't
                mask_np = np.array(mask_pil)
                
                blended = (img_np * (1 - alpha) + mask_np * alpha).astype(np.uint8)
                results.append(Image.fromarray(blended))
            else:
                # Return mask (P mode)
                mask_img = Image.fromarray(preds.astype(np.uint8))
                mask_img.putpalette([c for color in ID_TO_COLOR.values() for c in color])
                # Resize to original
                mask_img = mask_img.resize((orig_w, orig_h), Image.NEAREST)
                results.append(mask_img)
                
        if not is_batch:
            return results[0]
        return results
