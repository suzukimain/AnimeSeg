from __future__ import annotations

import warnings
from typing import Optional

from .dinoV2.dinoV2_pipeline import DinoV2AnimeSegPipeline
from .mask2former.mask2former_pipeline import Mask2FormerAnimeSegPipeline


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
                "Please use AnimeSegPipeline.from_dinoV2() or AnimeSegPipeline.from_mask2former().",
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

    def __call__(self, image, *args, **kwargs):
        return self._impl(image, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._impl, name)


__all__ = ["AnimeSegPipeline", "DinoV2AnimeSegPipeline", "Mask2FormerAnimeSegPipeline"]
