"""AnimeSeg - Anime Character Segmentation using DINOv2 + U-Net++"""

from .pipeline import AnimeSegPipeline

__version__ = "0.3.8"

__all__ = ["AnimeSegPipeline", "AnimeSegNextPipeline"]


def __getattr__(name: str):
    """Lazy loading to avoid circular imports"""
    if name == "AnimeSegNextPipeline":
        from anime_seg_next.mask2former.mask2former_pipeline import AnimeSegNextPipeline
        return AnimeSegNextPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

