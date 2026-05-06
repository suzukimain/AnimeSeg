from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from safetensors.torch import load_file


def _validate_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    file_path = str(entry.get("FilePath", "")).strip()
    if not file_path:
        raise ValueError("FilePath is required")

    train_image_size = int(entry.get("TrainImageSize", 0))
    version = int(entry.get("Version", 0))
    architecture = str(entry.get("Architecture", "")).strip()
    base_model = str(entry.get("BaseModel", "")).strip()
    config = entry.get("Config", {})

    if not isinstance(config, dict):
        raise ValueError("Config must be dict")

    return {
        "FilePath": file_path,
        "TrainImageSize": train_image_size,
        "Version": version,
        "Architecture": architecture,
        "BaseModel": base_model,
        "Config": config,
    }


def _infer_num_classes_from_checkpoint(checkpoint_path: Path) -> Optional[int]:
    if not checkpoint_path.exists():
        return None

    lower = checkpoint_path.name.lower()
    try:
        if lower.endswith(".safetensors"):
            state_dict = load_file(str(checkpoint_path))
        elif lower.endswith(".pt") or lower.endswith(".pth"):
            raw = torch.load(str(checkpoint_path), map_location="cpu")
            if isinstance(raw, dict):
                candidate_keys = ["state_dict", "model_state_dict", "model", "module"]
                resolved = None
                for key in candidate_keys:
                    val = raw.get(key)
                    if isinstance(val, dict):
                        resolved = val
                        break
                state_dict = resolved if resolved is not None else raw
            else:
                return None
        else:
            return None
    except Exception:
        return None

    normalized = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    for candidate in ("class_predictor.weight", "model.class_predictor.weight"):
        weight = normalized.get(candidate)
        if weight is not None and hasattr(weight, "shape") and len(weight.shape) >= 1:
            return int(weight.shape[0]) - 1
    return None


# Explicit 31-class color palette (RGB tuples, indexed 0-30)
# Provides deterministic, visually distinct colors with LR pairs for contrast
_EXPLICIT_31CLASS_COLORS = [
    [0, 0, 0],              # 0: background
    [40, 20, 60],           # 1: back_hair
    [0, 102, 204],          # 2: bottomwear
    [100, 180, 220],        # 3: ears_left
    [220, 120, 80],         # 4: ears_right
    [70, 120, 200],         # 5: earwear_left
    [220, 100, 40],         # 6: earwear_right
    [50, 100, 200],         # 7: eyebrow_left
    [200, 80, 30],          # 8: eyebrow_right
    [40, 80, 180],          # 9: eyelash_left
    [180, 60, 20],          # 10: eyelash_right
    [100, 160, 240],        # 11: eyewear_left
    [240, 140, 60],         # 12: eyewear_right
    [200, 240, 255],        # 13: eyewhite_left
    [255, 240, 200],        # 14: eyewhite_right
    [100, 150, 255],        # 15: face
    [32, 64, 96],           # 16: footwear
    [50, 30, 80],           # 17: front_hair
    [192, 192, 192],        # 18: handwear
    [200, 100, 50],         # 19: headwear
    [80, 140, 220],         # 20: irides_left (cool blue)
    [220, 180, 80],         # 21: irides_right (warm gold)
    [204, 51, 102],         # 22: legwear
    [255, 0, 150],          # 23: mouth
    [210, 170, 140],        # 24: neck
    [100, 100, 100],        # 25: neckwear
    [255, 140, 0],          # 26: nose
    [128, 128, 128],        # 27: objects
    [200, 50, 50],          # 28: tail
    [0, 128, 0],            # 29: topwear
    [255, 255, 0],          # 30: wings
]


def _generate_color(class_index: int) -> List[int]:
    hue = (class_index * 0.6180339887498949) % 1.0
    saturation = 0.55
    value = 0.92
    import colorsys

    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
    return [int(round(red * 255.0)), int(round(green * 255.0)), int(round(blue * 255.0))]


def _normalize_class_names(class_names: Sequence[Any], num_classes: int) -> List[str]:
    cleaned = [str(name).strip() for name in class_names if str(name).strip()]
    if len(cleaned) < num_classes:
        cleaned.extend(f"class_{idx}" for idx in range(len(cleaned), num_classes))
    return cleaned[:num_classes]


def _build_auto_colors(num_classes: int) -> List[List[int]]:
    # Use explicit 31-class palette if num_classes is exactly 31
    if num_classes == 31:
        return _EXPLICIT_31CLASS_COLORS
    # Otherwise generate deterministic HSV-based colors
    return [_generate_color(class_index) for class_index in range(num_classes)]


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {"models": [], "status": {}}

    with path.open("r", encoding="utf-8") as file:
        try:
            loaded = json.load(file)
        except json.JSONDecodeError:
            return {"models": [], "status": {}}

    if isinstance(loaded, list):
        return {"models": loaded, "status": {}}
    if isinstance(loaded, dict):
        if "models" not in loaded or not isinstance(loaded.get("models"), list):
            loaded["models"] = []
        if "status" not in loaded or not isinstance(loaded.get("status"), dict):
            loaded["status"] = {}
        return loaded
    return {"models": [], "status": {}}


def _upsert(models: List[Dict[str, Any]], entry: Dict[str, Any]) -> str:
    for idx, item in enumerate(models):
        if str(item.get("FilePath", "")).strip() == entry["FilePath"]:
            models[idx] = entry
            return "updated"
    models.append(entry)
    return "created"


def main() -> None:
    parser = argparse.ArgumentParser(description="Update config.json")
    parser.add_argument("--config-path", required=True, help="Path to config.json")
    parser.add_argument("--file-path", required=True)
    parser.add_argument("--train-image-size", required=True, type=int)
    parser.add_argument("--version", required=True, type=int)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--config-json", default="{}")
    args = parser.parse_args()

    path = Path(args.config_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    config_obj = json.loads(args.config_json) if args.config_json.strip() else {}
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path.strip() else None
    inferred_num_classes = _infer_num_classes_from_checkpoint(checkpoint_path) if checkpoint_path else None
    existing_class_names = config_obj.get("class_names") or config_obj.get("ClassNames")
    if inferred_num_classes is not None:
        config_obj["num_classes"] = inferred_num_classes
        if isinstance(existing_class_names, list):
            config_obj["class_names"] = _normalize_class_names(existing_class_names, inferred_num_classes)
        else:
            config_obj["class_names"] = [f"class_{idx}" for idx in range(inferred_num_classes)]
        if not isinstance(config_obj.get("class_colors"), list):
            config_obj["class_colors"] = _build_auto_colors(inferred_num_classes)

    if "class_names" in config_obj and isinstance(config_obj["class_names"], list):
        num_classes = int(config_obj.get("num_classes", len(config_obj["class_names"])))
        config_obj["class_names"] = _normalize_class_names(config_obj["class_names"], num_classes)
        if not isinstance(config_obj.get("class_colors"), list) or len(config_obj["class_colors"]) != num_classes:
            config_obj["class_colors"] = _build_auto_colors(num_classes)

    entry = _validate_entry(
        {
            "FilePath": args.file_path,
            "TrainImageSize": args.train_image_size,
            "Version": args.version,
            "Architecture": args.architecture,
            "BaseModel": args.base_model,
            "Config": config_obj,
        }
    )

    data = _load_json(path)
    status = _upsert(data["models"], entry)

    data["models"].sort(
        key=lambda x: (
            str(x.get("Architecture", "")),
            int(x.get("Version", 0)),
            str(x.get("FilePath", "")),
        )
    )

    data["status"] = {
        "last_operation": status,
        "last_target": entry["FilePath"],
        "count": len(data["models"]),
    }

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    print(f"status={status}")
    print(f"count={len(data['models'])}")
    print(f"target={entry['FilePath']}")


if __name__ == "__main__":
    main()
