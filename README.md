# 実行手順

## 事前準備

- 学習済みモデル `jepa-latest.pth.tar` を以下からダウンロードし、`results/pretrained/` に置いていただきたいです。
  - https://huggingface.co/jonoki-nitech/probabilistic_I-JEPA ("Files and Versions" → `jepa-latest.pth.tar`)

- 仮想環境：p-ijepa

## 実行

```bash
conda activate p-ijepa
bash revision/run.sh 2>&1 | tee run.log
```

メモリに乗らない場合は，以下のコマンドで実行をお願いします．

```bash
bash revision/run.sh bs32 2>&1 | tee run.log
```

## 成果物

以下はすべて `results/vitb16.224-bs.2048-ep.300-det/`(DET訓練用logging folder)に保存されます．

| 成果物                                                                                       | 保存先                            |
| -------------------------------------------------------------------------------------------- | --------------------------------- |
| checkpoint(latest; 再開・MC Dropout評価用)                                                   | `jepa-det-latest.pth.tar`         |
| エポック別checkpoint(50epoch毎に自動保存)                                                    | `jepa-det-ep{50,100,...}.pth.tar` |
| 訓練lossログ(epoch/itr/loss/masks/時間)                                                      | `jepa-det_r0.csv`                 |
| 実行時設定のコピー                                                                           | `params-ijepa.yaml`               |
| STL10線形分類精度のモニタ(10epoch毎)                                                         | `stl10_acc_monitoring.csv`        |
| 上記の最新checkpoint+ログをまとめたzip(訓練終了後に自動生成; エポック別checkpointは含まない) | `result.zip`(リポジトリ直下)      |
