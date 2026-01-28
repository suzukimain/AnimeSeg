"""
Quick test script for AnimeSegPipeline (without actual model download)
"""
import torch
from src.anime_seg.pipeline import AnimeSegPipeline

print("Testing AnimeSegPipeline import...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

print("\nTo use the pipeline:")
print("1. Upload your model to Hugging Face Hub: suzukimain/AnimeSeg")
print("2. Name it: models/anime_seg_dinov2_large_v1.safetensors")
print("3. Run: pipe = AnimeSegPipeline()")
print("4. Inference: mask = pipe('image.jpg')")

print("\n✓ Pipeline module loaded successfully!")
