# AnimeSeg Quick Start Guide

<p>
    <a href="https://github.com/suzukimain/AnimeSeg/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/suzukimain/AnimeSeg.svg"></a>
    <img src="https://visitor-badge.laobi.icu/badge?page_id=suzukimain.AnimeSeg" alt="Visitor Badge">
</p>


## サンプル画像

<p align="center">
    <img src="https://raw.githubusercontent.com/suzukimain/AnimeSeg/refs/heads/main/images/sample2.png" alt="サンプル画像" width="100%">
</p>


### 1. インストール

```python
pip install anime_seg
```

### 2. 基本的な使用

```python
from anime_seg import AnimeSegPipeline
pipe = AnimeSegPipeline.from_mask2former().to("cuda")
mask = pipe("path/to/image.jpg")
mask.save("output.png")
```

`AnimeSegPipeline()` のデフォルト呼び出しは非推奨です。`from_mask2former()` または `from_dinoV2()` を使ってください。

### 3. オプション: 出力サイズ指定

```python
# 出力サイズを指定しない場合は入力画像サイズで返す
mask_same = pipe("path/to/image.jpg")

# 縦横を指定した場合はそのサイズで返す
mask_fixed = pipe("path/to/image.jpg", width=1024, height=1024)

# 片側のみ指定した場合、未指定側は入力サイズを維持
mask_w = pipe("path/to/image.jpg", width=1024)
mask_h = pipe("path/to/image.jpg", height=1024)
```

## 詳細設定

### カスタムリポジトリ・ファイル名の指定

```python
# HFリポジトリ上の特定ファイルを指定
pipe = AnimeSegPipeline.from_mask2former(
    repo_id="suzukimain/AnimeSeg",
    filename="models/anime_seg_mask2former_v3.safetensors"
).to(device="cuda")

# DINOv2 を使う場合
pipe_dino = AnimeSegPipeline.from_dinoV2(
    filename="models/anime_seg_dinov2_v2.safetensors"
).to("cuda")
```

### PIL Imageを直接使用

```python
from PIL import Image

img = Image.open("image.jpg")
mask = pipe(img)
```

## モデルファイルの命名規則

```
models/anime_seg_{アーキテクチャ}_v{バージョン}.safetensors
```

例:
- `models/anime_seg_dinov2_v2.safetensors`
- `models/anime_seg_mask2former_v3.safetensors`

解決順序:
1. `models/model_config.json`
2. フォールバックで `models/anime_seg_{architecture}_v{最大バージョン}.{拡張子}`

## セグメンテーションクラス

`from_mask2former()` のデフォルトは **12クラス** です。

| ID | クラスキー | 名前 | RGB | 色名 |
|---:|---|---|---|---|
| 0 | background | 背景 | (0, 0, 0) | 黒 |
| 1 | skin | 肌 | (255, 220, 180) | ペールオレンジ |
| 2 | face | 顔 | (100, 150, 255) | 青 |
| 3 | hair_main | 髪(太い部分) | (255, 0, 0) | 赤 |
| 4 | left_eye | 左目 | (0, 255, 255) | シアン |
| 5 | right_eye | 右目 | (255, 255, 0) | 黄 |
| 6 | left_eyebrow | 左眉 | (150, 255, 0) | 黄緑 |
| 7 | right_eyebrow | 右眉 | (0, 255, 100) | エメラルドグリーン |
| 8 | nose | 鼻 | (255, 140, 0) | ダークオレンジ |
| 9 | mouth | 口 | (255, 0, 150) | マゼンタピンク |
| 10 | clothes | 服 | (180, 0, 255) | パープル |
| 11 | accessory | アクセサリー | (128, 128, 0) | オリーブ |

`from_dinoV2()` は **13クラス**（ID 12 に `unknown` を含む）です。

## トラブルシューティング

### エラー: "No model files found"

モデルファイルが指定の命名規則に従っているか確認してください。

```python
# 手動でファイル名を指定
pipe = AnimeSegPipeline.from_dinoV2(
    filename="models/anime_seg_dinov2_v2.safetensors"
).to("cuda")
```

### DINOv2 互換利用について

以前のバージョンでは DINOv2 ベースの利用が中心でしたが、現在は `from_mask2former()` を推奨しています。`from_dinoV2()` は互換用途として残しています。

## 技術仕様

- **バックボーン**: DINOv2 (facebook/dinov2-large)
- **微調整手法**: LoRA (r=8, alpha=16)
- **デコーダ**: U-Net++ with CBAM attention
- **推論入力サイズ**: 学習時サイズ (`TrainImageSize`) に自動リサイズ
- **出力サイズ**: `width` / `height` 未指定時は入力画像サイズ、指定時は指定サイズ
