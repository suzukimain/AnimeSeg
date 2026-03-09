from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anime_seg import AnimeSegPipeline  # noqa: E402


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _save_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _write_temp_config(data: dict) -> Path:
    tmp_dir = ROOT / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix="merge_cfg_", suffix=".json", dir=str(tmp_dir))
    Path(tmp_name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(tmp_name).chmod(0o666)
    Path(tmp_name).touch()
    return Path(tmp_name)


def _mask_similarity(pred: Image.Image, ref: Image.Image) -> Tuple[float, float]:
    pred_np = np.array(pred.convert("RGB"), dtype=np.uint8)
    ref_np = np.array(ref.convert("RGB"), dtype=np.uint8)

    if pred_np.shape != ref_np.shape:
        pred_np = np.array(
            Image.fromarray(pred_np).resize((ref_np.shape[1], ref_np.shape[0]), Image.Resampling.NEAREST),
            dtype=np.uint8,
        )

    flat_p = pred_np.reshape(-1, 3)
    flat_r = ref_np.reshape(-1, 3)

    pixel_acc = float(np.all(flat_p == flat_r, axis=1).mean())

    colors = np.unique(np.vstack([flat_p, flat_r]), axis=0)
    ious: List[float] = []
    for color in colors:
        pm = np.all(flat_p == color, axis=1)
        rm = np.all(flat_r == color, axis=1)
        union = np.logical_or(pm, rm).sum()
        if union == 0:
            continue
        inter = np.logical_and(pm, rm).sum()
        ious.append(float(inter / union))

    miou = float(np.mean(ious)) if ious else 0.0
    return pixel_acc, miou


def _sanitize_for_source(config_data: dict, idx: int) -> dict:
    data = copy.deepcopy(config_data)
    item = data["models"][idx]
    cfg = item.get("Config", {})
    if not isinstance(cfg, dict):
        cfg = {}
    cfg["merged_full"] = False
    item["Config"] = cfg

    arch = str(item.get("Architecture", "")).lower()
    if arch == "mask2former":
        base = str(item.get("BaseModel", ""))
        if base.startswith("models/"):
            item["BaseModel"] = "facebook/mask2former-swin-base-ade-semantic"
    return data


def _set_merged_flags(config_data: dict, idx: int, mask2former_config_relpath: str | None) -> dict:
    data = copy.deepcopy(config_data)
    item = data["models"][idx]
    cfg = item.get("Config", {})
    if not isinstance(cfg, dict):
        cfg = {}
    cfg["merged_full"] = True
    item["Config"] = cfg

    if mask2former_config_relpath is not None:
        item["BaseModel"] = mask2former_config_relpath

    return data


def _create_source_pipe(arch: str, filename: str, config_path: Path, device: str):
    if arch == "mask2former":
        return AnimeSegPipeline.from_mask2former(filename=filename, config_name=str(config_path)).to(device)
    if arch == "dinov2":
        return AnimeSegPipeline.from_dinoV2(filename=filename, config_name=str(config_path)).to(device)
    raise RuntimeError(f"Unsupported architecture: {arch}")


def _merge_one(entry: Dict, source_pipe, model_path: Path) -> str | None:
    arch = str(entry.get("Architecture", "")).lower()

    if arch == "mask2former":
        state_dict = {k: v.detach().cpu().contiguous() for k, v in source_pipe.model.model.state_dict().items()}
        save_file(state_dict, str(model_path))

        config_dir = model_path.parent / "mask2former_merged_config"
        config_dir.mkdir(parents=True, exist_ok=True)
        source_pipe.model.model.config.save_pretrained(str(config_dir))
        return str(config_dir.relative_to(ROOT).as_posix())

    if arch == "dinov2":
        if hasattr(source_pipe.model.backbone, "merge_and_unload"):
            source_pipe.model.backbone = source_pipe.model.backbone.merge_and_unload()
            source_pipe.model.use_lora = False

        state_dict = {k: v.detach().cpu().contiguous() for k, v in source_pipe.model.state_dict().items()}
        save_file(state_dict, str(model_path))
        return None

    raise RuntimeError(f"Unsupported architecture: {arch}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge AnimeSeg checkpoints into full single-file safetensors and validate similarity.")
    parser.add_argument("--config", default="config.json", help="Path to model config JSON")
    parser.add_argument("--image", default="tmp/img/source.png", help="Validation image path")
    parser.add_argument("--min-miou", type=float, default=0.995, help="Minimum mIoU between source and merged outputs")
    parser.add_argument("--min-acc", type=float, default=0.995, help="Minimum pixel accuracy between source and merged outputs")
    args = parser.parse_args()

    config_path = ROOT / args.config
    val_image_path = ROOT / args.image
    if not val_image_path.exists():
        raise FileNotFoundError(f"Validation image not found: {val_image_path}")

    config_data = _load_json(config_path)
    models = config_data.get("models", [])
    if not isinstance(models, list):
        raise RuntimeError("Invalid model config format: 'models' must be a list")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_image = Image.open(val_image_path).convert("RGB")

    print(f"device={device}")
    print(f"validation_image={val_image_path}")

    report: List[Tuple[str, float, float]] = []

    for index, item in enumerate(models):
        if not isinstance(item, dict):
            continue

        arch = str(item.get("Architecture", "")).lower()
        file_path = str(item.get("FilePath", "")).strip()
        if arch not in {"mask2former", "dinov2"} or not file_path:
            continue

        model_path = ROOT / file_path
        model_path.parent.mkdir(parents=True, exist_ok=True)

        if not Path(file_path).is_absolute() and model_path.exists():
            model_path.unlink()
            print(f"[info] removed local stale file: {model_path}")

        source_cfg = _sanitize_for_source(config_data, index)
        source_cfg_path = _write_temp_config(source_cfg)
        source_pipe = _create_source_pipe(arch, file_path, source_cfg_path, device)
        source_mask = source_pipe(val_image)

        merged_base = _merge_one(item, source_pipe, model_path)

        merged_cfg = _set_merged_flags(config_data, index, merged_base if arch == "mask2former" else None)
        merged_cfg_path = _write_temp_config(merged_cfg)
        merged_pipe = _create_source_pipe(arch, file_path, merged_cfg_path, device)
        merged_mask = merged_pipe(val_image)

        pixel_acc, miou = _mask_similarity(merged_mask, source_mask)
        report.append((arch, pixel_acc, miou))

        out_mask = ROOT / "tmp" / "img" / f"merged_check_{arch}_mask.png"
        out_mask.parent.mkdir(parents=True, exist_ok=True)
        merged_mask.save(out_mask)

        print(f"[ok] {arch} merged -> {model_path}")
        print(f"[ok] {arch} similarity pixel_acc={pixel_acc:.6f} mIoU={miou:.6f}")

        if pixel_acc < args.min_acc or miou < args.min_miou:
            raise RuntimeError(
                f"Merged model quality check failed for {arch}: "
                f"pixel_acc={pixel_acc:.6f} (min={args.min_acc}), "
                f"mIoU={miou:.6f} (min={args.min_miou})"
            )

        config_data = merged_cfg

    _save_json(config_path, config_data)
    print(f"[ok] updated config: {config_path}")

    for arch, pixel_acc, miou in report:
        print(f"[report] {arch}: pixel_acc={pixel_acc:.6f}, mIoU={miou:.6f}")


if __name__ == "__main__":
    main()
