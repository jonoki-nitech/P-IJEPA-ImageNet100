# 実行手順

## 事前準備

- σモデルの学習済みチェックポイント `results/pretrained/jepa-latest.pth.tar`(デフォルトの`--model_path`)と ImageNet-100 val(`configs/configs.yaml` の `data.root_path`）を配置していただきたいです。
- 仮想環境：p-ijepa

## 実行

```bash
conda activate p-ijepa
bash revision/run.sh 2>&1 | tee run_sample_lookahead.log
```

## 成果物

実行が終わると，以下の成果物が保存されます（seed 0はリポジトリ直下，seed 1..4は各 `seed_<s>/` サブフォルダ）．

| 成果物                                                                                         | 保存先                                                |
| ---------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| sampled-rollout lookahead 選択のJSON (+ σ総和トラジェクトリ等のstats JSON; K付き名前で保存)    | `selected_patch_index_lookahead_sample/...`           |
| MAE reconstruction-error-vs-t評価(CSVは `lookahead_sample` / `lookahead_sample4` の両行を含む) | `images/rec_image_lookahead_sample/<seed-path>/b<b>/` |
| 全成果物(seed 0..4； 選択 + 評価； ディレクトリ構造をそのまま)を1本に圧縮したzip               | `result.zip`                                          |
