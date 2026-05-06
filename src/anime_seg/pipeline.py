from __future__ import annotations

import os
import warnings
from typing import Optional

from huggingface_hub import hf_hub_download

from .dinoV2.dinoV2_pipeline import DinoV2AnimeSegPipeline
from .mask2former.mask2former_pipeline import Mask2FormerAnimeSegPipeline
from .remove_bg.bg_remover_pipeline import BgRemover

class AnimeSegPipeline:
    def __init__(
        self,
        architecture: str = "dinov2",
        repo_id: str = "suzukimain/AnimeSeg",
        filename: str = "",
        token: Optional[str] = None,
        device: Optional[str] = None,
        base_model: str = "",
        config_name: str = "config.json",
        remove_bg: bool = False,
    ) -> None:
        if (
            architecture == "dinov2"
            and repo_id == "suzukimain/AnimeSeg"
            and filename == ""
            and token is None
            and device is None
            and base_model == ""
            and config_name == "config.json"
        ):
            warnings.warn(
                "AnimeSegPipeline() default constructor is deprecated and will be removed in a future release. "
                "Please use AnimeSegPipeline.from_dinoV2(), AnimeSegPipeline.from_mask2former(), or AnimeSegPipeline.from_bg_remover().",
                DeprecationWarning,
                stacklevel=2,
            )

        arch = architecture.strip().lower()
        if arch == "mask2former":
            self._impl = self.from_mask2former(
                repo_id=repo_id,
                filename=filename,
                token=token,
                device=device,
                base_model=base_model or "facebook/mask2former-swin-large-ade-semantic",
                config_name=config_name,
                remove_bg=remove_bg,
            )
        else:
            self._impl = self.from_dinoV2(
                repo_id=repo_id,
                filename=filename,
                token=token,
                device=device,
                config_name=config_name,
                remove_bg=remove_bg,
            )

    @classmethod
    def from_dinoV2(
        cls,
        repo_id: str = "suzukimain/AnimeSeg",
        filename: str = "",
        token: Optional[str] = None,
        device: Optional[str] = None,
        config_name: str = "config.json",
        remove_bg: bool = False,
    ) -> DinoV2AnimeSegPipeline:
        pipe = DinoV2AnimeSegPipeline(
            repo_id=repo_id,
            filename=filename,
            token=token,
            device=None,
            config_name=config_name,
            remove_bg=remove_bg,
        )
        if device is not None:
            pipe.to(device)
        return pipe

    @classmethod
    def from_mask2former(
        cls,
        repo_id: str = "suzukimain/AnimeSeg",
        filename: str = "",
        token: Optional[str] = None,
        device: Optional[str] = None,
        base_model: str = "facebook/mask2former-swin-large-ade-semantic",
        config_name: str = "config.json",
        remove_bg: bool = False,
    ) -> Mask2FormerAnimeSegPipeline:
        pipe = Mask2FormerAnimeSegPipeline(
            repo_id=repo_id,
            filename=filename,
            token=token,
            device=None,
            base_model=base_model,
            config_name=config_name,
            remove_bg=remove_bg,
        )
        if device is not None:
            pipe.to(device)
        return pipe

    @classmethod
    def from_bg_remover(
        cls,
        repo_id: str = "suzukimain/AnimeSeg",
        filename: str = "models/remove_bg/BgRemover.safetensors",
        token: Optional[str] = None,
        device: Optional[str] = None,
    ):
        if filename and os.path.isfile(filename):
            ckpt_path = filename
        else:
            ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
            
        pipe = BgRemover.from_single_file(ckpt_path=ckpt_path, device="cpu")
        if device is not None:
            pipe.to(device)
        return pipe

    def __call__(self, image, *args, **kwargs):
        return self._impl(image, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._impl, name)


__all__ = ["AnimeSegPipeline", "DinoV2AnimeSegPipeline", "Mask2FormerAnimeSegPipeline", "BgRemover"]
