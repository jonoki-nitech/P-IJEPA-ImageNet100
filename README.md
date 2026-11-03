# 実行手順

## 事前準備

- σモデルの学習済みチェックポイント `results/pretrained/jepa-latest.pth.tar`(デフォルトの`--model_path`)と ImageNet-100 val(`configs/configs.yaml` の `data.root_path`）を配置していただきたいです。
- 仮想環境：p-ijepa

## 実行

```bash
conda activate p-ijepa
bash revision/run.sh 2>&1 | tee run_sample_lookahead.log
```

- 内容： σモデルでの **sampled-rollout 1-step lookahead** 選択実験（`lookahead_sample`）。
  既存のσ-lookaheadと同一プロトコルだが，候補ブロックの特徴を μ平均ではなく
  **予測分布からのサンプリング**（`feat_c = μ + σ·ε, ε~N(0,I)`）でシミュレーションする。
  サンプリングは `--samples K` で平均化できる（デフォルトは K="1 4" の両方を実行）。
- 初期ブロックは既存のランダム初期ブロック戦略と同一（同一画像・同一seedで一致；
  サンプリング乱数は専用 `torch.Generator` を使用し，初期ブロックのRNGを汚さない）。
- b = 2..4 のみ（tmax = 196/b² = 49/21/12）。center-start は行わない。
- スモークテスト: `MAX_T=2 bash revision/run.sh`（アーティファクトは `*_tmax_2_*` 名で
  本番と衝突しない）
- 過去の実験スクリプトは `revision/archive/` に移設
  - DET/MC-Dropout 不確実性実験: `revision/archive/run_mcdropout_uncertainty.sh`
  - DET 学習: `revision/archive/run_mcdropout_train.sh`

## 成果物

実行が終わると，以下の成果物が保存されます（seed 0はリポジトリ直下，seed 1..4は各 `seed_<s>/` サブフォルダ）．

| 成果物                                                                                          | 保存先                                     |
| ----------------------------------------------------------------------------------------------- | ------------------------------------------ |
| sampled-rollout lookahead 選択のJSON (+ σ総和トラジェクトリ等のstats JSON; K付き名前で保存)         | `selected_patch_index/...`                 |
| MAE reconstruction-error-vs-t評価(CSVは `lookahead_sample` / `lookahead_sample4` の両行を含む)      | `images/rec_images_full/<seed-path>/b<b>/` |

- 選択JSONの名前はK毎に区別されます:
  - K=1 → `block_<b>_tmax_<T>_test_lookahead_sample.json`
  - K=4 → `block_<b>_tmax_<T>_test_lookahead_sample4.json`
- 既存のσ戦略（greedy / lookahead / random / fixed* / AME）との比較プロットは
  事後に選択JSONを用いて行います（この実験の評価CSVは新戦略のみを含みます）。
- zip化はこのスクリプトでは行いません（成果物は上記フォルダのまま保持）。
