import os
import torch
import numpy as np
import cv2
from PIL import Image
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import amp

from .isnet import ISNetDIS


def _with_prefix_stripped(state_dict, prefix):
    return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}


def _candidate_state_dicts(state_dict):
    candidates = [state_dict]
    for prefix in ("net.", "module.", "model."):
        stripped = _with_prefix_stripped(state_dict, prefix)
        if stripped:
            candidates.append(stripped)

    module_then_net = _with_prefix_stripped(state_dict, "module.net.")
    if module_then_net:
        candidates.append(module_then_net)

    return candidates

class BgRemover:
    def __init__(self, model: ISNetDIS, device="cuda"):
        self.device = device
        self.model = model.to(self.device)
        self.model.eval()
        
    @classmethod
    def from_single_file(cls, ckpt_path=None, device="cuda"):
        if ckpt_path is None:
            ckpt_path = hf_hub_download(repo_id="suzukimain/AnimeSeg", filename="models/remove_bg/BgRemover.safetensors")
        
        model = ISNetDIS()
        state_dict = load_file(ckpt_path)

        last_error = None
        for candidate in _candidate_state_dicts(state_dict):
            try:
                model.load_state_dict(candidate, strict=True)
                return cls(model, device)
            except RuntimeError as exc:
                last_error = exc

        if last_error is not None:
            raise RuntimeError(
                "Failed to load BgRemover checkpoint. Supported formats are raw ISNetDIS state_dict "
                "and prefixed keys such as 'net.*', 'module.*', or 'model.*'."
            ) from last_error

        return cls(model, device)

    def to(self, device):
        self.device = str(device)
        self.model.to(self.device)
        return self

    def __call__(self, image: Image.Image, use_amp=True, s=1024, return_type="pil", bg_color=(255, 255, 255)):
        if isinstance(image, str):
            image = Image.open(image)
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Input must be a PIL Image or a file path.:{type(image)}")
        input_img = np.array(image.convert("RGB"))
        input_img_norm = (input_img / 255.0).astype(np.float32)
        h0, w0 = input_img_norm.shape[:2]
        
        h, w = (s, int(s * w0 / h0)) if h0 > w0 else (int(s * h0 / w0), s)
        ph, pw = s - h, s - w
        
        img_input = np.zeros([s, s, 3], dtype=np.float32)
        img_input[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w] = cv2.resize(input_img_norm, (w, h))
        img_input = np.transpose(img_input, (2, 0, 1))
        img_input = img_input[np.newaxis, :]
        
        tmpImg = torch.from_numpy(img_input).type(torch.FloatTensor).to(self.device)
        
        with torch.no_grad():
            if use_amp and self.device == "cuda":
                with amp.autocast("cuda"):
                    pred = self.model(tmpImg)[0][0].sigmoid()
                pred = pred.to(dtype=torch.float32)
            else:
                pred = self.model(tmpImg)[0][0].sigmoid()
            
            pred = pred.cpu().numpy()[0]
            pred = np.transpose(pred, (1, 2, 0))
            pred = pred[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w]
            mask = cv2.resize(pred, (w0, h0))[:, :, np.newaxis]
            
        fgImg = (mask * input_img + (1 - mask) * np.array(bg_color)).astype(np.uint8)
        
        if return_type == "pil":
            return Image.fromarray(fgImg)
        elif return_type == "numpy":
            return fgImg
        return fgImg
