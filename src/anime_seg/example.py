"""
Minimal example of using AnimeSegPipeline
"""
from src.anime_seg import AnimeSegPipeline

# 1) Mask2Former backend (recommended)
pipe = AnimeSegPipeline.from_mask2former().to("cuda")

# 2) DINOv2 backend
# pipe = AnimeSegPipeline.from_dinoV2().to(device="cuda")

# Run inference
mask = pipe("path/to/your/image.jpg")

# Save result
mask.save("output_mask.png")

print("Segmentation complete!")
