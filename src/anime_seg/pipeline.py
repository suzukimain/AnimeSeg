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
        config_name: str = "models/model_config.json",
    ) -> None:
        if (
            architecture == "dinov2"
            and repo_id == "suzukimain/AnimeSeg"
            and filename == ""
            and token is None
            and device is None
            and base_model == ""
            and config_name == "models/model_config.json"
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
                base_model=base_model or "facebook/mask2former-swin-base-ade-semantic",
                config_name=config_name,
            )
        else:
            self._impl = self.from_dinoV2(
                repo_id=repo_id,
                filename=filename,
                token=token,
                device=device,
                config_name=config_name,
            )

    @classmethod
    def from_dinoV2(
        cls,
        repo_id: str = "suzukimain/AnimeSeg",
        filename: str = "",
        token: Optional[str] = None,
        device: Optional[str] = None,
        config_name: str = "models/model_config.json",
    ) -> DinoV2AnimeSegPipeline:
        if device is not None:
            warnings.warn(
                "`device` passed to AnimeSegPipeline.from_dinoV2(...) is ignored. "
                "Please call `.to(\"cuda\")` or `.to(device=\"cuda\")` after pipeline creation.",
                UserWarning,
            )
        return DinoV2AnimeSegPipeline(
            repo_id=repo_id,
            filename=filename,
            token=token,
            device=None,
            config_name=config_name,
        )

    @classmethod
    def from_mask2former(
        cls,
        repo_id: str = "suzukimain/AnimeSeg",
        filename: str = "",
        token: Optional[str] = None,
        device: Optional[str] = None,
        base_model: str = "facebook/mask2former-swin-base-ade-semantic",
        config_name: str = "models/model_config.json",
    ) -> Mask2FormerAnimeSegPipeline:
        if device is not None:
            warnings.warn(
                "`device` passed to AnimeSegPipeline.from_mask2former(...) is ignored. "
                "Please call `.to(\"cuda\")` or `.to(device=\"cuda\")` after pipeline creation.",
                UserWarning,
            )
        return Mask2FormerAnimeSegPipeline(
            repo_id=repo_id,
            filename=filename,
            token=token,
            device=None,
            base_model=base_model,
            config_name=config_name,
        )

    @classmethod
    def from_bg_remover(
        cls,
        repo_id: str = "suzukimain/AnimeSeg",
        filename: str = "models/remove_bg/BgRemover.safetensors",
        token: Optional[str] = None,
        device: Optional[str] = None,
    ):
        if device is not None:
            warnings.warn(
                "`device` passed to AnimeSegPipeline.from_bg_remover(...) is ignored. "
                "Please call `.to(\"cuda\")` or `.to(device=\"cuda\")` after pipeline creation.",
                UserWarning,
            )

        if filename and os.path.isfile(filename):
            ckpt_path = filename
        else:
            ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
            
        return BgRemover.from_single_file(ckpt_path=ckpt_path)

    def __call__(self, image, *args, **kwargs):
        return self._impl(image, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._impl, name)


__all__ = ["AnimeSegPipeline", "DinoV2AnimeSegPipeline", "Mask2FormerAnimeSegPipeline"]
