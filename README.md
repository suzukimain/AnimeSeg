# AnimeSeg

Anime Character Segmentation using DINOv2 + U-Net++ with LoRA fine-tuning.

## Installation

```bash
pip install -e .
```

## Usage

```python
from anime_seg import AnimeSegPipeline

# Initialize pipeline (auto-downloads latest model from Hugging Face)
pipe = AnimeSegPipeline()

# Run segmentation
mask = pipe("path/to/image.jpg")

# Save result
mask.save("output.png")
```

## Advanced Usage

```python
# Specify custom repo or filename
pipe = AnimeSegPipeline(
    repo_id="suzukimain/AnimeSeg",
    filename="models/anime_seg_dinov2_large_v1.safetensors",
    device="cuda"  # or "cpu"
)

# Use PIL Image
from PIL import Image
img = Image.open("image.jpg")
mask = pipe(img)
```

## Model Files

Models should follow the naming convention:
```
models/anime_seg_{architecture}_{size}_v{version}.safetensors
```

Example:
- `models/anime_seg_dinov2_large_v1.safetensors`
- `models/anime_seg_dinov2_base_v2.safetensors`

## Segmentation Classes

- Background
- Skin
- Face
- Hair (main)
- Hair (thin)
- Eyes (left/right)
- Eyebrows (left/right)
- Nose
- Mouth
- Clothes
- Unknown
