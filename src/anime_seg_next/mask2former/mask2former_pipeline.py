from __future__ import annotations

from typing import Optional

from anime_seg.mask2former.mask2former_pipeline import Mask2FormerAnimeSegPipeline


DEFAULT_REPO_ID = "suzukimain/AnimeSeg-Next"


class AnimeSegNextPipeline(Mask2FormerAnimeSegPipeline):
    """AnimeSeg-Next Mask2Former pipeline with AnimeSeg-style loading semantics."""

    def __init__(
        self,
        repo_id: str = DEFAULT_REPO_ID,
        filename: str = "",
        token: Optional[str] = None,
        device: Optional[str] = None,
        base_model: str = "facebook/mask2former-swin-large-ade-semantic",
        config_name: str = "config.json",
        remove_bg: bool = False,
    ) -> None:
        super().__init__(
            repo_id=repo_id,
            filename=filename,
            token=token,
            device=device,
            base_model=base_model,
            config_name=config_name,
            remove_bg=remove_bg,
        )

    @classmethod
    def from_mask2former(
        cls,
        repo_id: str = DEFAULT_REPO_ID,
        filename: str = "",
        token: Optional[str] = None,
        hf_token: Optional[str] = None,
        device: Optional[str] = None,
        base_model: str = "facebook/mask2former-swin-large-ade-semantic",
        config_name: str = "config.json",
        remove_bg: bool = False,
    ) -> "AnimeSegNextPipeline":
        # Support both `token` and `hf_token` parameter names
        final_token = hf_token if hf_token is not None else token
        
        pipe = cls(
            repo_id=repo_id,
            filename=filename,
            token=final_token,
            device=None,
            base_model=base_model,
            config_name=config_name,
            remove_bg=remove_bg,
        )
        if device is not None:
            pipe.to(device)
        return pipe


AnimeSegPipeline = AnimeSegNextPipeline
