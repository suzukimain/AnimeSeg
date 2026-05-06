# AnimeSeg

<p>
    <a href="https://pepy.tech/project/anime_seg"><img alt="GitHub release" src="https://static.pepy.tech/badge/anime_seg"></a>
    <a href="https://github.com/suzukimain/AnimeSeg/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/suzukimain/AnimeSeg.svg"></a>
    <img src="https://visitor-badge.laobi.icu/badge?page_id=suzukimain.AnimeSeg" alt="Visitor Badge">
</p>


Anime Character Segmentation using Mask2Former and DINOv2 + U-Net++ with LoRA fine-tuning.
Also integrates Background Removal via [anime-segmentation](https://github.com/SkyTNT/anime-segmentation).

## sample image

<p align="center">
    <img src="https://raw.githubusercontent.com/suzukimain/AnimeSeg/refs/heads/main/images/sample2.png" alt="sample image" width="100%">
</p>


## Installation

```bash
pip install anime_seg
```

## Usage

```python
from anime_seg import AnimeSegPipeline
pipe = AnimeSegPipeline.from_mask2former().to("cuda")
mask = pipe("path/to/image.jpg")
mask.save("output.png")

# Background Removal (powered by anime-segmentation)
bg_pipe = AnimeSegPipeline.from_bg_remover().to("cuda")
no_bg_img = bg_pipe("path/to/image.jpg")
no_bg_img.save("no_bg_output.png")
```

### AnimeSeg-Next

```python
from anime_seg import AnimeSegNextPipeline

pipe = AnimeSegNextPipeline.from_mask2former().to("cuda")
mask = pipe("path/to/image.jpg")
mask.save("output.png")
```

If the Hugging Face repo is private or gated, pass `token="hf_..."` or set `HF_TOKEN`. When running interactively, the loader will prompt for a token after an auth failure.

`AnimeSegPipeline()` default constructor is deprecated. Use `from_mask2former()`, `from_dinoV2()`, or `from_bg_remover()`.

## Optional: output size

```python
# Same as input size (default)
mask_same = pipe("path/to/image.jpg")

# Fixed output size
mask_fixed = pipe("path/to/image.jpg", width=1024, height=1024)

# Width/height can be specified independently
mask_w = pipe("path/to/image.jpg", width=1024)
mask_h = pipe("path/to/image.jpg", height=1024)
```

## Advanced Usage

```python
# Load specific file from HF repo
pipe = AnimeSegPipeline.from_mask2former(
    repo_id="suzukimain/AnimeSeg",
    filename="models/anime_seg_mask2former_v3.safetensors"
).to(device="cuda")

# DINOv2 backend
pipe_dino = AnimeSegPipeline.from_dinoV2(
    filename="models/anime_seg_dinov2_v2.safetensors"
).to("cuda")

# Use PIL Image
from PIL import Image
img = Image.open("image.jpg")
mask = pipe(img)

# Background Removal (powered by anime-segmentation)
bg_pipe = AnimeSegPipeline.from_bg_remover().to("cuda")
no_bg_img = bg_pipe("path/to/image.jpg")
no_bg_img.save("no_bg_output.png")
```

## Model Files

Models should follow the naming convention:
```
models/anime_seg_{architecture}_v{version}.safetensors
```

Example:
- `models/anime_seg_dinov2_v2.safetensors`
- `models/anime_seg_mask2former_v3.safetensors`

Resolution order:
1. `config.json`
2. fallback scan by `models/anime_seg_{architecture}_v{max_version}.{ext}`

For AnimeSeg-Next, store Mask2Former metadata in `config.json` under `Config` with `num_classes`, `class_names`, and optionally `class_colors`:

```json
{
    "models": [
        {
            "FilePath": "models/anime_seg_next_mask2former_v1.safetensors",
            "TrainImageSize": 768,
            "Version": 1,
            "Architecture": "mask2former",
            "BaseModel": "facebook/mask2former-swin-large-ade-semantic",
            "Config": {
                "merged_full": true,
                "num_classes": 31,
                "class_names": [
                    "background",
                    "back_hair",
                    "bottomwear",
                    "ears_left",
                    "ears_right",
                    "earwear_left",
                    "earwear_right",
                    "eyebrow_left",
                    "eyebrow_right",
                    "eyelash_left",
                    "eyelash_right",
                    "eyewear_left",
                    "eyewear_right",
                    "eyewhite_left",
                    "eyewhite_right",
                    "face",
                    "footwear",
                    "front_hair",
                    "handwear",
                    "headwear",
                    "irides",
                    "legwear",
                    "mouth",
                    "neck",
                    "neckwear",
                    "nose",
                    "objects",
                    "tail",
                    "topwear",
                    "wings"
                ],
                "class_colors": [
                    [0, 0, 0],
                    [233, 92, 92],
                    [92, 233, 126],
                    [92, 165, 233],
                    [233, 193, 92],
                    [197, 92, 233],
                    [92, 233, 213],
                    [233, 140, 92],
                    [140, 92, 233],
                    [233, 92, 172],
                    [92, 233, 171],
                    [171, 92, 233],
                    [233, 220, 92],
                    [92, 204, 233],
                    [233, 92, 107],
                    [118, 233, 92],
                    [92, 118, 233],
                    [233, 155, 92],
                    [155, 92, 233],
                    [233, 92, 211],
                    [92, 233, 145],
                    [145, 92, 233],
                    [233, 185, 92],
                    [92, 185, 233],
                    [233, 92, 133],
                    [133, 233, 92],
                    [92, 133, 233],
                    [233, 110, 92],
                    [110, 92, 233],
                    [233, 92, 239],
                    [92, 233, 198]
                ]
            }
        }
    ],
    "status": {
        "last_operation": "created",
        "last_target": "models/anime_seg_next_mask2former_v1.safetensors",
        "count": 1
    }
}
```

If `num_classes` is omitted, the updater can infer it from the checkpoint's class head. If `class_colors` is omitted, it is generated deterministically to avoid hand-editing mistakes.

## Segmentation Classes and Mask Colors

Default `from_mask2former()` returns **12 classes**:

| ID | Class Key | RGB | Color |
|---:|---|---|---|
| 0 | background | (0, 0, 0) | Black |
| 1 | skin | (255, 220, 180) | Pale Orange |
| 2 | face | (100, 150, 255) | Blue |
| 3 | hair_main | (255, 0, 0) | Red |
| 4 | left_eye | (0, 255, 255) | Cyan |
| 5 | right_eye | (255, 255, 0) | Yellow |
| 6 | left_eyebrow | (150, 255, 0) | Yellow Green |
| 7 | right_eyebrow | (0, 255, 100) | Emerald Green |
| 8 | nose | (255, 140, 0) | Dark Orange |
| 9 | mouth | (255, 0, 150) | Magenta Pink |
| 10 | clothes | (180, 0, 255) | Purple |
| 11 | accessory | (128, 128, 0) | Olive |

`from_dinoV2()` returns **13 classes** (includes `unknown` as ID 12).

## DINOv2 Compatibility Note

Earlier versions primarily used DINOv2. Current recommendation is `from_mask2former()`, while `from_dinoV2()` remains for compatibility.
