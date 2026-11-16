# L1最終モデル5体の性能評価（2026-11-16）

直近に起動した対称厨房4体と、旧レイアウト環境3の単体学習seed151を評価した。
各モデルを各右L0と1,000エピソードずつ組ませ、合計15,000エピソードを実行。
**全3種類のL0で成功率95%以上を満たしたのは1/5体、旧環境3のseed151のみだった。**
対称厨房では0/4体。異なる環境の成績を同一条件として平均・比較しない。

## 成功率

各セルは成功率%、分母は1,000。全L0で950/1,000以上を個別に要求する。

| 学習環境 | seed | レタス＋玉ねぎL0 | トマト＋にんじんL0 | じゃがいも＋ブロッコリーL0 | 全L0で95%以上 |
|---|---:|---:|---:|---:|---|
| 対称厨房・両者配膳可 | 101 | 99.9 | 0.0 | 0.0 | 不合格 |
| 対称厨房・両者配膳可 | 102 | 87.7 | 0.0 | 0.0 | 不合格 |
| 対称厨房・両者配膳可 | 103 | 100.0 | 0.0 | 0.0 | 不合格 |
| 対称厨房・両者配膳可 | 104 | 99.8 | 99.8 | 0.0 | 不合格 |
| 旧環境3・右L0のみ配膳 | 151 | 100.0 | 100.0 | 100.0 | **合格** |

対称厨房のseed101–103は玉ねぎ側L0との組み合わせに成功が集中し、
seed104は玉ねぎ・にんじん側で高い成功率だった。4体ともブロッコリー側では0/1,000。
これは相手ごとの成績の偏りであり、失敗行動の原因は本評価だけでは断定しない。

旧環境3のseed151は3,000/3,000成功、最小相手成功率も100%。
以前の旧環境3のseed101–104では全L0合格が0/4体だった
（[以前の評価](l0_serving_left_l1_evaluation_20261115.md)）。
seedとVMが異なる1試行なので、単体学習にしたことが性能改善の原因とは判断できない。
対称厨房とは配置・配膳可能側・観測次元も異なる。

## 固定した評価手順

- 各モデルの400更新・28,800,000stepsの最終checkpointを全て使用。選抜・重み更新・再学習なし。
- 右L0は固定レシピの3専門家。random_action_probability=0、agent_id=1。
- 各相手200round×5物理episode=1,000episode、1体3,000episode。
- 評価seed6001、episode seed=6001+100000×partner_index+1000×round+episode_index。
- 確率的行動サンプリング、random start、40step horizon。
  GRUはround開始のみresetし、物理episode間はepisode flag/previous actionをreset。
- 対称厨房は108入力・両側配膳、旧環境3は105入力・右配膳。どちらも全体観測・各agent各episode1食材制限。
- 対称厨房4体の評価はmacOS/arm64 CPU・4worker、旧環境3は元Linux/x86_64 CPU・1worker。
  全workerはTorch/BLAS1thread。runtimeはPython3.12.14、NumPy2.5.4、PyTorch2.15.0（Cloudは+cpu）。
- 相手ごとに950/1,000成功以上。平均成功率だけでは合格にしない。
  固定学習L0に対する実測であり、未知相手への汎化や真の成功率の保証ではない。

対称厨房の事前plan:
[configs/evaluations/l1-mirrored-full-single-20261116.json](../configs/evaluations/l1-mirrored-full-single-20261116.json)。
plan canonical SHA256:
`512b277d93afbe126593823b5490f6981d0423c49888f442356f60e1e65d6b57`。
旧環境3のplan canonical:
`d55d57cc7a7934835c9ab8bbc0474504a06662ab8a17a4d13938994b8516df2d`。

## ソースと互換性

| 条件 | 学習source | 評価source | 学習config canonical SHA256 |
|---|---|---|---|
| 対称厨房 | `4ad6d8a6d82c2eb04a3062b91bbe7db0690029bf` | `b02e9a3471bc8e9fcb080343462749416f406c9b` | `f08c233af8b7a7102bb61aea2ec512bc23f2b72b978dacbc5dc633933e4b949c` |
| 旧環境3 | `16216917ce67bf2e21515dc328b72255d20ea103` | 同左 | `72defc6b1fd0e9caca4a25f1832a38d4af018d0c57da07486fab9c64155df266` |

対称厨房の評価commitはplan/test/docsの追加のみで、src/scriptsと元学習configは変更していない。
各モデルは自身のconfig・environment/implementation source hashと一致する実装でstrict CPU loadした。
105と108のモデルを混用せず、旧モデルのmanifestやcheckpointを書き換えていない。
推論互換性・既存runの同条件CPU recovery互換性に変更なし。

| 条件・seed | 最終checkpoint SHA256 |
|---|---|
| 対称101 | `60e51a285643fd7531fe38ca769f3c11d1301988575e692b53521004a5f60b64` |
| 対称102 | `6f3edec7752787b94e7c08348f425ce9083d9b0361308a7a2eb00ee1477d514d` |
| 対称103 | `257348da3802c8290aa5abc0dd3ccd6993b410306d2d6f065af08eac00180529` |
| 対称104 | `c3e54aa5e2846c11bdd6fb4b5c3f041ee615299528ece328e615aa91b685e0f1` |
| 旧151 | `360d7141bfaf3e679a01f28aa56775d6618e821fba6f44102d627b72ff2b9681` |

## 独立検証と保存

全5体の学習完走・最終保存を元VMで確認し、調整側でも最終bundleを再取得して
archive/全payload/source inventory/400連続metrics/final/resume400/strict CPU loadを検証した。
対称p01は88payloadと入れ子source64原本、p02は140payloadとsource64原本を照合。
seed151の最終モデル・評価は専用Draft446571560を維持。

全15,000episode・464,507 trace stepsについて、独立に成功数/分母/seed/round位置/
連続step/終端成功/reward/役割/action範囲を再集計し、集計JSON/CSVと一致した。
評価前後のcheckpoint SHA、protocol/artifact SHA、source/config不変を確認。
旧151は元taskでも全episodeを環境と凍結右L0から独立再生している。

対称厨房のplan契約testを追加し、workspace check・固定PACE6ソース照合・
全120unittest（118成功、Docker opt-in skip2）・差分checkを通過。

評価成果物のローカルnamespace:
`outputs/l1-five-model-evaluation-20261116-coordinator`。
集計JSON/CSV、全episode/trace、評価source/config/runtime/command、
独立検証と学習model参照を保存。モデル本体は元学習bundleを参照する。

旧151の評価保存先はDraft446571560、archive
`evaluation-s151-1794822348795589846.zip`、132payloadを調整側でも再取得して照合済み。
manifest asset754287690、manifest SHA
`162168d3bcd50c2acaa6b93831030bd0b3d9f547db85eba1d8c2d4110367427d`、
archive SHA
`23e6f833835285f9ce81b2905fff063eda49739a8d26b7998543b674472d1765`。
対称厨房4体と比較集計の新bundleはDraft446495466へ保存し、receiptを別途記録する。

## 評価bundleの保存確認

Draft446495466へ対称厨房の全評価データと5体の比較集計を保存した。
archiveは`l1-mirrored-full-single-evaluation-20261116.zip`、3,709,578bytes、2parts。
再取得後のarchiveと全389payload SHAを照合した。元モデル本体は再収録せず、
学習bundleのmanifestとcheckpoint SHAで参照している。

- transfer manifest asset: **754310736**
- manifest SHA256: `ddbf926fce1d98177e206e51036c6026186d0ec24195382cfdd11d56aadc3575`
- archive SHA256: `42cfc6ff69134a168a95e0533399756f906d541f3e5ab11cbbd9cd23b863a09c`
- receipt: `outputs/l1-five-model-evaluation-20261116-coordinator/evaluation-storage-verified.json`

bundle内のreport.mdは保存前の本文で、このreceipt節はGit側に追記した。
旧151の全評価データは上記の専用Draft446571560（132payload）に保持し、
対称厨房側bundleには比較集計・独立検証と参照を含めた。
