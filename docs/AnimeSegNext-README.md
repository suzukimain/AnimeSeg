# AnimeSegNext - 31-Class Anime Character Segmentation

> **⚠️ 認証 (Authentication) が必要です**
>
> このモデルはHugging Face上のプライベートリポジトリで配信されています。
> 使用するには Hugging Face のアカウントとアクセストークンが必要です。
>
> This model is distributed in a private repository on Hugging Face.
> You need a Hugging Face account and access token to use it.

## 概要 (Overview)

AnimeSegNext は Mask2Former ベースの **31クラス** アニメキャラクター セグメンテーションモデルです。
12クラス版 (AnimeSeg) を拡張し、目の左右 (`irides_left` / `irides_right`) を分離して、より詳細なセグメンテーションを実現します。

**特徴：**
- 31個のセマンティッククラス（背景含む）
- 左右対比設計：左側は冷色系、右側は暖色系で視認性向上
- Mask2Former (Swin-Large) ベース
- 推論時に自動的にクラスカラーを適用

---

## インストール (Installation)

### 必要なパッケージ

```bash
pip install torch torchvision
pip install transformers pillow safetensors huggingface-hub
```

### セットアップ

```bash
# Hugging Face トークンを設定
huggingface-cli login
# または環境変数で設定
export HF_TOKEN=your_hf_token_here
```

---

## 使用方法 (Usage)

### 基本的な推論（推奨）

```python
from anime_seg import AnimeSegNextPipeline
from PIL import Image

# パイプラインの初期化
# 初回実行時はモデルをダウンロード（認証トークンが必要）
pipe = AnimeSegNextPipeline.from_mask2former(
    hf_token="hf_xxxxxxxxxxxxx",  # HF トークン
).to('cuda')

# 画像をロード
image = Image.open('character.png').convert('RGB')

# セグメンテーション実行
mask_image = pipe(image)

# 保存
mask_image.save('mask.png')
```

### 従来の方法（互換性維持）

```python
import sys
sys.path.insert(0, r'path/to/AnimeSeg/src')

from anime_seg_next import AnimeSegNextPipeline
from PIL import Image

# パイプラインの初期化
pipe = AnimeSegNextPipeline.from_mask2former(
    token="hf_xxxxxxxxxxxxx"  # HF トークン
).to('cuda')

# 画像をロード
image = Image.open('character.png').convert('RGB')

# セグメンテーション実行
mask_image = pipe(image)

# 保存
mask_image.save('mask.png')
```

### ローカルチェックポイントの使用

```python
from anime_seg import AnimeSegNextPipeline

# ローカルに保存されたチェックポイントを使用
pipe = AnimeSegNextPipeline.from_mask2former(
    filename='path/to/anime_seg-next_mask2forme_v1.safetensors',
    hf_token="hf_xxxxxxxxxxxxx"  # HF トークン（オプション）
).to('cpu')

mask_image = pipe(image)
```

### GPU メモリ最適化

```python
from anime_seg import AnimeSegNextPipeline

# CPU推論（遅いが メモリ節約）
pipe = AnimeSegNextPipeline.from_mask2former(
    hf_token="hf_xxxxxxxxxxxxx"
).to('cpu')

# 混合精度推論（RTX 30/40シリーズ向け）
pipe = AnimeSegNextPipeline.from_mask2former(
    hf_token="hf_xxxxxxxxxxxxx"
).to('cuda')
# 自動的に AMP (Automatic Mixed Precision) が適用されます
```

### カラーマスク + オーバーレイ合成

```python
from anime_seg import AnimeSegNextPipeline
from PIL import Image

# パイプライン初期化
pipe = AnimeSegNextPipeline.from_mask2former(
    hf_token="hf_xxxxxxxxxxxxx"
).to('cuda')

# セグメンテーション実行
source_img = Image.open('character.png').convert('RGB')
mask_img = pipe(source_img)  # クラス色が自動適用

# オーバーレイ（元画像 + マスク 55% ブレンド）
overlay = Image.blend(
    source_img.resize(mask_img.size),
    mask_img,
    0.55  # マスク透明度 55%
)
overlay.save('overlay.png')
```

---

## クラス定義と色パレット (Class Definition & Color Palette)

31クラスのセグメンテーション結果。各クラスに固有の RGB カラーが自動適用されます。

⭐ マーク = 左右対比設計（左=冷色系、右=暖色系）

| ID  | RGB (R, G, B) | Color Name | CLASS_NAME | HEX |
|-----|---------------|-----------|---------------|--------|
| 0   | (0, 0, 0) | Black | background | #000000 |
| 1   | (40, 20, 60) | Dark Purple | back_hair | #28143c |
| 2   | (0, 102, 204) | Electric Blue | bottomwear | #0066cc |
| 3   | (100, 180, 220) | Sky Blue | ears_left | #64b4dc |
| 4   | (220, 120, 80) | Peach | ears_right | #dc7850 |
| 5   | (70, 120, 200) | Cornflower Blue | earwear_left | #4678c8 |
| 6   | (220, 100, 40) | Dark Orange | earwear_right | #dc6428 |
| 7   | (50, 100, 200) | Dodger Blue | eyebrow_left | #3264c8 |
| 8   | (200, 80, 30) | Burnt Orange | eyebrow_right | #c8501e |
| 9   | (40, 80, 180) | Deep Blue | eyelash_left | #2850b4 |
| 10  | (180, 60, 20) | Dark Orange-Red | eyelash_right | #b43c14 |
| 11  | (100, 160, 240) | Light Cyan | eyewear_left | #64a0f0 |
| 12  | (240, 140, 60) | Yellow-Orange | eyewear_right | #f08c3c |
| 13  | (200, 240, 255) | Alice Blue | eyewhite_left | #c8f0ff |
| 14  | (255, 240, 200) | Cream | eyewhite_right | #fff0c8 |
| 15  | (100, 150, 255) | Periwinkle | face | #6496ff |
| 16  | (32, 64, 96) | Dark Slate | footwear | #204060 |
| 17  | (50, 30, 80) | Deep Purple | front_hair | #321e50 |
| 18  | (192, 192, 192) | Silver | handwear | #c0c0c0 |
| 19  | (200, 100, 50) | Light Brown | headwear | #c86432 |
| 20  | (80, 140, 220) | **Cool Blue** | irides_left ⭐ | #508cdc |
| 21  | (220, 180, 80) | **Warm Gold** | irides_right ⭐ | #dcb450 |
| 22  | (204, 51, 102) | Crimson Rose | legwear | #cc3366 |
| 23  | (255, 0, 150) | Hot Magenta | mouth | #ff0096 |
| 24  | (210, 170, 140) | Tan | neck | #d2aa8c |
| 25  | (100, 100, 100) | Gray | neckwear | #646464 |
| 26  | (255, 140, 0) | Orange | nose | #ff8c00 |
| 27  | (128, 128, 128) | Medium Gray | objects | #808080 |
| 28  | (200, 50, 50) | Crimson | tail | #c83232 |
| 29  | (0, 128, 0) | Green | topwear | #008000 |
| 30  | (255, 255, 0) | Yellow | wings | #ffff00 |

---

## 設計の詳細 (Design Details)

### 左右対比カラーリング (LR Contrast Coloring)

左右のペアクラス（耳、眉毛、睫毛、アイウェア、眼白、瞳孔）は対照的な色で設定されています：

**左側（冷色系 - Cool Colors）:**
- 青系 (#3264c8 - #64b4dc)

**右側（暖色系 - Warm Colors）:**
- オレンジ/黄色系 (#c8501e - #dc7850)

**瞳孔（Irides）**
- `irides_left`: Cool Blue (#508cdc) - 落ち着いた、紫がかった青
- `irides_right`: Warm Gold (#dcb450) - 温かみのある、黄金色

この設計により、セグメンテーション結果を視覚的に検証する際に、左右の混同を防ぎ、モデルの精度を直感的に判断できます。

---

## トラブルシューティング (Troubleshooting)

### ❌ `ModuleNotFoundError: No module named 'anime_seg_next'`

**解決策 1（推奨）：**
```python
from anime_seg import AnimeSegNextPipeline
```

**解決策 2（従来の方法）：**
```python
import sys
sys.path.insert(0, r'path/to/AnimeSeg/src')
from anime_seg_next import AnimeSegNextPipeline
```

### ❌ `Unauthorized` / `401` / `403` エラー

```
huggingface_hub.utils._errors.GatedRepoError: 
Access to this repository requires authentication
```

**解決策：**
```bash
# 1. Hugging Face トークンを取得
#    https://huggingface.co/settings/tokens

# 2. ログイン
huggingface-cli login

# または

# 3. 環境変数を設定
export HF_TOKEN=hf_xxxxxxxxxxxxx

# 4. 再実行
python your_script.py
```
