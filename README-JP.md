# AnimeSeg Quick Start Guide

## 最小限の使い方 (MVP)

### 1. インストール

```bash
pip install -e .
```

### 2. 基本的な使用

```python
from anime_seg import AnimeSegPipeline

# パイプラインの初期化 (HFから最新モデルを自動ダウンロード)
pipe = AnimeSegPipeline()

# 推論の実行
mask = pipe("path/to/image.jpg")

# 結果の保存
mask.save("output.png")
```

## 詳細設定

### カスタムリポジトリ・ファイル名の指定

```python
pipe = AnimeSegPipeline(
    repo_id="suzukimain/AnimeSeg",
    filename="models/anime_seg_dinov2_large_v1.safetensors",
    device="cuda"  # または "cpu"
)
```

### PIL Imageを直接使用

```python
from PIL import Image

img = Image.open("image.jpg")
mask = pipe(img)
```

### プライベートリポジトリの使用

```python
pipe = AnimeSegPipeline(
    repo_id="your-username/YourPrivateRepo",
    token="hf_..."  # Hugging Face token
)
```

## モデルファイルの命名規則

```
models/anime_seg_{アーキテクチャ}_{サイズ}_v{バージョン}.safetensors
```

例:
- `models/anime_seg_dinov2_large_v1.safetensors`
- `models/anime_seg_dinov2_base_v2.safetensors`
- `models/anime_seg_dinov2_small_v1.safetensors`

## セグメンテーションクラス (13クラス)

1. Background - 背景
2. Skin - 肌
3. Face - 顔
4. Hair (main) - 髪の毛 (太い部分)
5. Hair (thin) - 髪の毛 (細い部分)
6. Left Eye - 左目
7. Right Eye - 右目
8. Left Eyebrow - 左眉
9. Right Eyebrow - 右眉
10. Nose - 鼻
11. Mouth - 口
12. Clothes - 服
13. Unknown - 不明

## トラブルシューティング

### エラー: "No model files found"

モデルファイルが指定の命名規則に従っているか確認してください。

```python
# 手動でファイル名を指定
pipe = AnimeSegPipeline(
    filename="models/anime_seg_dinov2_large_v1.safetensors"
)
```

### メモリ不足エラー

より小さいモデルサイズを使用してください (small または base)。

## 技術仕様

- **バックボーン**: DINOv2 (facebook/dinov2-large)
- **微調整手法**: LoRA (r=8, alpha=16)
- **デコーダ**: U-Net++ with CBAM attention
- **入力サイズ**: 512x512 (自動リサイズ)
- **出力**: カラーセグメンテーションマスク (元のサイズに戻す)
