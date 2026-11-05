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
| sampled-rollout lookahead 選択のJSON (+ σ総和トラジェクトリ等のstats JSON; K/H付き名前で保存)   | `selected_patch_index_lookahead_sample/...`           |
| MAE reconstruction-error-vs-t評価(この実験の全variantの行を含む)                                  | `images/rec_image_lookahead_sample/<seed-path>/b<b>/` |
| 全成果物(seed 0..4； 選択 + 評価； ディレクトリ構造をそのまま)を1本に圧縮したzip               | `result.zip`                                          |

- 選択JSONの名前はK(サンプリング数)/H(horizon)で区別されます:
  - K=1, H=1 → `block_<b>_tmax_<T>_test_lookahead_sample.json`
  - K>1 → 末尾に `<K>`， H>1 → 末尾に `h<H>`（例: K=4, H=2 → `..._lookahead_sample4h2.json`）
- `run.sh` のデフォルトは K=1, H=2（`SAMPLES`/`HORIZON` envで変更可）
