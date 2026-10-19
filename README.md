# 実行手順

## 事前準備

- 学習済みモデル `jepa-latest.pth.tar` を以下からダウンロードし、`results/pretrained/` に置いていただきたいです。
  - https://huggingface.co/jonoki-nitech/probabilistic_I-JEPA ("Files and Versions" → `jepa-latest.pth.tar`)

- 仮想環境：p-ijepa

## 実行

```bash
conda activate p-ijepa
BATCH=64 DEVICE=cuda:0 bash revision/run.sh 2>&1 | tee run.log
```

(BATCHでバッチサイズを変更可能です．デフォルトは64ですが，メモリが足りない場合はお手数ですが減らして実行をお願いいたします．)

## 成果物の保存先

| 成果物                                        | 保存先                                                   |
| --------------------------------------------- | -------------------------------------------------------- |
| σのcalibration(図表+metrics CSV+後分析用pt)   | `images/calibration/`                                    |
| 選択系列JSON(9戦略、b=1〜4)                   | `selected_patch_index/block_{1..4}_tmax_{196/b²}_*.json` |
| MAE再構成誤差 vs t(グラフPNG / CSV / 生値pt)  | `images/rec_images_full/b{1..4}/`                        |
| 再構成画像(最初の10サンプル、最終t)           | `images/rec_images_full/b{1..4}/gallery/`                |
| 上記成果物をまとめたzip(実行終了時に自動生成) | `result.zip`(リポジトリ直下)                             |
