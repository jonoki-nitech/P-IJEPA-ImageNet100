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

(seed=1〜4の結果を順に実行します．1seedあたり7〜14時間かかる見込みなので，全体で4倍の時間がかかります．)

(`BATCH`でバッチサイズを変更可能です．デフォルトは64ですが，メモリが足りない場合はお手数ですが減らして実行をお願いいたします．)

(seedを1つだけ指定して単発で実行したい場合は `SEED=<s> bash revision/run_seed.sh` を使ってください．デフォルトはSEED=0で，実行済みのseed 0の成果物と同じ場所・名前で出力します．)

## seedごとの成果物

結果は以下のように，seedごとにresult*seed*{SEED}.zipにまとめられるため，こちらを共有いただきたいです．

| seedごとの成果物                              | 保存先                                                               |
| --------------------------------------------- | -------------------------------------------------------------------- |
| 選択系列JSON + statsファイル                  | `selected_patch_index/seed_{SEED}/block_{1..4}_tmax_{196/b²}_*.json` |
| MAE再構成誤差 vs t(グラフPNG / CSV / 生値pt)  | `images/rec_images_full/seed_{SEED}/b{1..4}/`                        |
| 上記成果物をまとめたzip(実行終了時に自動生成) | `result_seed_{SEED}.zip`(リポジトリ直下)                             |

## 成果物の保存先

| 成果物                                        | 保存先                                                   |
| --------------------------------------------- | -------------------------------------------------------- |
| σのcalibration(図表+metrics CSV+後分析用pt)   | `images/calibration/`                                    |
| 選択系列JSON(8戦略 b=1 / 9戦略 b=2〜4)        | `selected_patch_index/block_{1..4}_tmax_{196/b²}_*.json` |
| σ総和の推移(提案手法)+lookahead選択順位       | `selected_patch_index/stats_block_*.json`                |
| MAE再構成誤差 vs t(グラフPNG / CSV / 生値pt)  | `images/rec_images_full/b{1..4}/`                        |
| 再構成画像(最初の10サンプル、最終t)           | `images/rec_images_full/b{1..4}/gallery/`                |
| 上記成果物をまとめたzip(実行終了時に自動生成) | `result.zip`(リポジトリ直下)                             |
