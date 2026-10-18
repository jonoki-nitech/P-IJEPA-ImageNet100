# 実行手順

## 事前準備

学習済みモデル `jepa-latest.pth.tar` を以下からダウンロードし、`results/pretrained/` に置きます。
https://huggingface.co/jonoki-nitech/probabilistic_I-JEPA ("Files and Versions" → `jepa-latest.pth.tar`)

## 実行

```bash
conda activate <環境名>
DEVICE=cuda:0 bash revision/run.sh 2>&1 | tee run.log
```

## 成果物の保存先

| 成果物                                       | 保存先                                             |
| -------------------------------------------- | -------------------------------------------------- |
| σのcalibration(図表+metrics CSV+後分析用pt)  | `images/calibration/`                              |
| 7戦略の選択系列JSON(b=1〜4 × 7変種)          | `selected_patch_index/block_{1..4}_tmax_32_*.json` |
| MAE再構成誤差 vs t(グラフPNG / CSV / 生値pt) | `images/rec_images_tmax32/b{1..4}/`                |
| 再構成画像(最初の10サンプル、最終t)          | `images/rec_images_tmax32/b{1..4}/gallery/`        |
