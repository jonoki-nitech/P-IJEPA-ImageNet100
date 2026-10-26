# 実行手順

## 事前準備

- 前回の実験で生成された学習済みモデル `results/vitb16.224-bs.2048-ep.300-det/jepa-det-latest.pth.tar` を `results/det-pretrained/` に置いていただきたいです。
- 仮想環境：p-ijepa

## 実行

```bash
conda activate p-ijepa
bash revision/run.sh 2>&1 | tee run_det_exp.log
```

## 成果物

実行が終わると，以下の成果物がリポジトリ直下の各フォルダに保存されます(seed 0はリポジトリ直下，seed 1..4は各 `seed_<s>/` サブフォルダ)．

| 成果物                                                                       | 保存先                                  |
| ---------------------------------------------------------------------------- | --------------------------------------- |
| 未観測patchの不確実性(MC-Dropout分散)のcalibration → coverage/diagram/points | `images/calibration_det/`               |
| MC-Dropout patch選択のJSON(+ 分散トラジェクトリ等のstats JSON)               | `selected_patch_index_mcdropout/...`    |
| MAE reconstruction-error-vs-t評価(gallery/error-vs-t図/CSV/pt)               | `images/rec_images_mcdropout/.../b<b>/` |
| per-seedの成果物zip(zip化は自動； 上記 + calibrationを含む)                  | `result_det_seed_<s>.zip`               |
| 全seed統合zip(同上; seed 0を含む全seed分)                                    | `result.zip`                            |
