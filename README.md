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

(`BATCH`でバッチサイズを変更可能です．デフォルトは64ですが，メモリが足りない場合はお手数ですが減らして実行をお願いいたします．)

## seedごとの成果物

結果は以下のように，seedごとにresult*seed*{SEED}.zipにまとめられるため，こちらを共有いただきたいです．

| seedごとの成果物                              | 保存先                                                               |
| --------------------------------------------- | -------------------------------------------------------------------- |
| 選択系列JSON + statsファイル                  | `selected_patch_index/seed_{SEED}/block_{1..4}_tmax_{196/b²}_*.json` |
| MAE再構成誤差 vs t(グラフPNG / CSV / 生値pt)  | `images/rec_images_full/seed_{SEED}/b{1..4}/`                        |
| 上記成果物をまとめたzip(実行終了時に自動生成) | `result_seed_{SEED}.zip`(リポジトリ直下)                             |
