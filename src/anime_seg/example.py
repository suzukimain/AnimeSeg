"""
Minimal example of using AnimeSegPipeline
"""
from src.anime_seg import AnimeSegPipeline
from PIL import Image

# Initialize pipeline (auto-downloads latest model from HF)
pipe = AnimeSegPipeline()

# Run inference
mask = pipe("path/to/your/image.jpg")

# Save result
mask.save("output_mask.png")

print("Segmentation complete!")
