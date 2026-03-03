from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


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
    parser = argparse.ArgumentParser(description="Update model_config.json")
    parser.add_argument("--config-path", required=True, help="Path to model_config.json")
    parser.add_argument("--file-path", required=True)
    parser.add_argument("--train-image-size", required=True, type=int)
    parser.add_argument("--version", required=True, type=int)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--config-json", default="{}")
    args = parser.parse_args()

    path = Path(args.config_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    config_obj = json.loads(args.config_json) if args.config_json.strip() else {}

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
