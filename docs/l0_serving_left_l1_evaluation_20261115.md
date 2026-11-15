# 左L1・右配膳L0：学習完了確認と性能評価（2026-11-15）

12体すべてが400更新・各28,800,000 stepsを完了した。
最終モデルを各L0と1,000エピソードずつ評価した結果、
**全3種類のL0で成功率95%以上を達成した個体は0/12体**だった。

## 結果

表の数値は成功率（%）。各セルの分母は1,000エピソード。
L0は右側で指定レシピの食材を調理し、完成皿を配膳する凍結ポリシー。
L1は左側で食材の調理と受け渡しを担当する。

| 環境 | seed | レタス＋玉ねぎL0 | トマト＋にんじんL0 | じゃがいも＋ブロッコリーL0 | 全L0で95%以上 |
|---|---:|---:|---:|---:|---|
| 1 全体観測・調理制限なし | 101 | 100.0 | 99.9 | 0.0 | 不合格 |
| 1 | 102 | 100.0 | 99.7 | 0.0 | 不合格 |
| 1 | 103 | 100.0 | 99.8 | 0.0 | 不合格 |
| 1 | 104 | 100.0 | 100.0 | 0.0 | 不合格 |
| 2 自分側＋共有カウンター観測・1食材制限 | 101 | 100.0 | 0.0 | 0.0 | 不合格 |
| 2 | 102 | 100.0 | 0.0 | 0.0 | 不合格 |
| 2 | 103 | 100.0 | 0.0 | 0.0 | 不合格 |
| 2 | 104 | 100.0 | 0.0 | 0.0 | 不合格 |
| 3 全体観測・1食材制限 | 101 | 100.0 | 0.0 | 0.0 | 不合格 |
| 3 | 102 | 100.0 | 0.0 | 0.0 | 不合格 |
| 3 | 103 | 100.0 | 0.0 | 0.0 | 不合格 |
| 3 | 104 | 99.9 | 0.0 | 0.0 | 不合格 |

環境1は4体とも2種類のL0に対して高い成功率だが、ブロッコリー側L0との
協調には成功していない。食材制限のある環境2・3では、今回の最終モデルは
玉ねぎ側L0との組み合わせでのみ成功した。全L0を均等に平均した成功率は、
環境1が66.57–66.67%、環境2が33.33%、環境3が33.30–33.33%。
この評価だけから失敗行動の原因や学習時間を延ばした場合の改善は判断しない。

## 完了確認と保存

全5台の既存Cloud VMで、チャット状態とは独立に以下を読み取り確認した。
全400行のmetrics（連続update1–400）、resume/latestのupdate400とsnapshot SHA、
complete.json・final.pt・final.json・latest.jsonの一致、left ego_side=0と右L0三種、
strict CPU load、worker/writer不在、container exit0・OOM/restarts0を確認済み。
有限managerはすべて終了済み。

保存先は`kei-student/re-NestRL`の既存Draft Release **443746098**、
tag `l1-left-l0-serving-3env-400u-s101-20261113`。
環境2のseed103/104は学習自体は完了していたが、最終multipart保存がHTTP404で
途中終了していた。元VMに残る同一最終bundleを一意の名前で保存し直し、
再取得と全payload照合を完了した。学習の再開や既存partial assetの変更はしていない。
以前拒否された補助managerコードは保存対象に含めていない。

| task | seeds | 最終transfer manifest asset ID | archive SHA-256 | 検証payload数 |
|---|---|---:|---|---:|
| e1-p01 | 101–104 | 743898187 | `2cfb1075dde237d4b0da1d6782ac4f6a1ef40267c6b1ec0c40df0c0c2ffc7ad3` | 120 |
| e2-p01 | 101–102 | 743020694 | `01d690eb9e07d08795076043c46a87f52f264f25e60698751bd68888a1ed0793` | 102 |
| e2-p02 | 103–104 | 751117626 | `7b9ca047c587a2f4c89210074f1a76d9dced5149d152910c9f6c5a5d1ee5b5a5` | 149 |
| e3-p01 | 101–102 | 743041039 | `652a819c022627804e1877bca1b83bff26b208dee45f4a7abf1b7bb872557507` | 144 |
| e3-p02 | 103–104 | 743072972 | `33af2de84b4d25dd80185a3d4974a5b7f96f28c6eca17ee4c32f4d2144ef0edb` | 178 |

コーディネーター側でも全5bundleをGitHubから再取得し、全693payloadのSHA、
12最終モデルと各resume snapshotのSHA、完了予算、strict CPU loadを検証した。
復旧snapshotを含む学習成果物は上記の最終bundleに保持されている。

## 評価手順と再現性

- 各環境4体、construction seed101–104の最終400更新checkpointを全て使用。
  選別、再学習、重み更新は行っていない。
- 相手は`l0-1-LettuceOnionSalad`、`l0-1-TomatoCarrotSalad`、
  `l0-1-PotatoBroccoliSalad`の3種類。学習時と同じ凍結L0プール。
- 各相手200 round ×5物理episode =1,000episode、1体3,000episode、
  合計36,000episode。random start、40step horizon、round内のGRU記憶は保持。
- 評価seed6001。episode seedは`6001 + 100000*partner + 1000*round + episode`。
  全条件・全モデルで同じ日程を使用。行動は確率的サンプリング。
- CPU推論、各worker Torch/BLAS1thread、総同時worker数4。
  環境1→3→2の順に実行し、条件をまたぐworker重複はない。
- 各L0で950/1,000成功以上という基準を個別に適用。平均だけで合格にはしない。
  実測成功率を扱い、未知L0への汎化や真の成功確率の保証は主張しない。
- 実行前にcommand/config/source/runtimeを保存。評価前後のcheckpoint SHAは全体で一致。
  全12体のepisode成功数を再集計し、行動traceのstep数・終了成功・artifact SHA、
  設定・ソース不変、分母・役割・seedを照合した。

学習ソースは`2d82b2e888b7970806ad9b9329bcf73303396d2a`、評価ソースは
`548406ec6b450192bee4b441afdc8e2a8f65f2bb`。
ブランチは`codex/l0-serving-left-l1-20261113`。
環境・model・L0実装を変更していないため、各元configとのstrict CPU推論互換性を維持。
異なる条件のcheckpoint混用や学習backendの変更はしていない。
Python3.12.14、NumPy2.5.4、PyTorch2.15.0。学習はLinux/x86_64 CPU、
今回の独立した評価はmacOS/arm64 CPUで実施し、そのruntimeを保存した。

| 環境 | 学習config canonical SHA-256 | 評価plan canonical SHA-256 |
|---|---|---|
| 1 | `6b50e2f1e0c22ef2f2a4482fd8e1bd409dfdf4fd9477588509c66ee706d34baf` | `4123ba8f93cc069906afb940bfd17d1f24f10a77413e7a1760f9120ea72e1175` |
| 2 | `f0bbdda67e4afe66a7bbc3db059e04198eae74dae963f73e8d030ac645a7e775` | `e052bec4700adc016506510d46deacd759c9c6f970a7166b44dc45bc7dbc0a5b` |
| 3 | `b74a3d2b47b25633d1a9dcc315ed2385fc8981c448ab1380cd26f63b3cff4001` | `c735db9d7f9f55d9b4fb23e94ad0ccf2fa4adebb367beed9688f83e0ce7af33b` |

評価planは`configs/evaluations/l1-left-l0-serving-*-20261115.json`。
個体別JSON/CSV、全episode/trace、実行記録、ダウンロード検証記録は
`outputs/l1-left-l0-serving-evaluation-20261115`に保存。
最終モデル本体を新たに作らず、上記の学習bundleを参照する。

再現例（新しいoutput-rootを指定すること）：

```sh
source .venv/bin/activate
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python scripts/record_run.py --run-id left-l1-e1-evaluation-20261115 \
  --config configs/evaluations/l1-left-l0-serving-full-20261115.json \
  --output outputs/l1-left-l0-serving-evaluation-20261115/execution-e1 \
  -- .venv/bin/python scripts/evaluate_l1_l0.py \
  --plan configs/evaluations/l1-left-l0-serving-full-20261115.json \
  --models-root outputs/l1-left-l0-serving-evaluation-20261115/models/e1 \
  --output-root outputs/l1-left-l0-serving-evaluation-20261115/evaluation/e1
```

検証：workspace check（dirty/clean）、PACEの固定6ソース照合、
unittest105件（成功103、Docker opt-in skip2）、差分checkを通過。

## 評価成果物の保存検証

既存Draft443746098へ`l1-left-l0-serving-evaluation-20261115.zip`を保存済み。
10,363,611 bytes、5 parts、全1,031payloadを新たにダウンロードしてSHA照合済み。
archive SHA-256は`d7bf984358d73cc4d6ae95570158ac071ee59bbc7c4cc69439eef89e191b412e`。
transfer manifest asset IDは**751137135**、manifest SHA-256は
`0cbafb1e545e196d3d5f46117915ff1e7fd27112965da57285d40e3caa816220`。
保存検証receipt asset IDは**751137466**。
bundleは集計・全episode/trace・評価ソース/config/runtime/実行記録・学習成果物参照を含み、
モデル本体は既存の検証済み学習bundleを参照する。
bundle内のreport.mdは保存前の報告書本文で、この保存receipt節はGit側に追記している。
