# 実行手順

## 事前準備

- 学習済みモデル `jepa-latest.pth.tar` を以下からダウンロードし、`results/pretrained/` に置いていただきたいです。
  - https://huggingface.co/jonoki-nitech/probabilistic_I-JEPA ("Files and Versions" → `jepa-latest.pth.tar`)

- 仮想環境：p-ijepa

## 実行

```bash
conda activate p-ijepa
DEVICE=cuda:0 bash revision/run.sh 2>&1 | tee run.log
```

## 成果物の保存先

`t_max` はブロックサイズごとの最大観測可能ステップ数 `196/b²`(14×14=196パッチ上の重複なし b×b ブロック数)です。各選択戦略は自身の飽和点まで走り、評価は全戦略の最小budgetに整列して描画します。

| 成果物                                        | 保存先                                             |
| --------------------------------------------- | -------------------------------------------------- |
| σのcalibration(図表+metrics CSV+後分析用pt)   | `images/calibration/`                              |
| 選択系列JSON(8戦略、b=1〜4)                   | `selected_patch_index/block_{1..4}_tmax_{196/b²}_*.json` |
| MAE再構成誤差 vs t(グラフPNG / CSV / 生値pt)  | `images/rec_images_full/b{1..4}/`                  |
| 再構成画像(最初の10サンプル、最終t)           | `images/rec_images_full/b{1..4}/gallery/`          |
| 上記成果物をまとめたzip(実行終了時に自動生成) | `result.zip`(リポジトリ直下)                       |
