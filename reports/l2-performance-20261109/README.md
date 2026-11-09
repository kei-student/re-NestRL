# 完成済みL2全8体の性能評価 — 2026-11-09

クラウド環境Cで、追加学習・重み更新なしに全8体を評価しました。指示、結果確認およびこの報告の保存はローカルのre-NestRLチャットが担当しました。Aは使用していません。

## 結果

| L2 seed | 固定L0（成功/150） | 学習済みL1（成功/400） |
| --- | ---: | ---: |
| 401 | 1/150 (0.67%) | 375/400 (93.75%) |
| 402 | 1/150 (0.67%) | 369/400 (92.25%) |
| 403 | 3/150 (2.00%) | 313/400 (78.25%) |
| 404 | 2/150 (1.33%) | 167/400 (41.75%) |
| 405 | 32/150 (21.33%) | 166/400 (41.50%) |
| 406 | 7/150 (4.67%) | 375/400 (93.75%) |
| 407 | 8/150 (5.33%) | 171/400 (42.75%) |
| 408 | 24/150 (16.00%) | 167/400 (41.75%) |
| 全体 | 78/1200 (6.50%) | 2103/3200 (65.72%) |

L1 seed104–111はL2の学習で使ったconstruction方策です。上のL1評価は学習済み相手に対する性能であり、未知相手への汎化性能ではありません。

学習済みL1への成功率はモデル間で41.50–93.75%と大きく異なりました。固定L0では全体6.50%で、十分な協調性能は確認できませんでした。全条件を通じてTomatoCarrotSaladの完成は0件です。固定L0の低さの原因（相互適応への依存、recipeの偏り等）は、この評価だけでは特定していません。

## 評価条件

- L2 seed401–408の全8体。L2モデルRelease ID438779737。
- 固定L0：本文3レシピのscripted peer、各相手10round×5episode、1モデル150episode。
- 学習済みL1：seed104–111の全8体（Release ID433508587）、各相手10round×5episode、1モデル400episode。
- 合計4400episode / 880round、40step horizon、確率的行動、評価seed1501。
- round開始seedは1501+100000×partner_index+1000×round_index。各conditionでpartner_indexを再開し、全L2で同じseed導出を使用。episode環境seedはround seed+episode_index。
- 同一roundの5episodeで相手を固定。両方策の履歴はepisode間で保持し、round境界でreset。
- 追加学習なし。全16checkpointの前後SHA一致をCで確認。
- 成功は有効料理の配膳。途中の報酬を成功として数えない。
- 初回の入力manifest形式への対応でpreflightを修正し、採点前に再実行。最終評価script/hash/protocolと失敗preflightログは結果ZIPに保持。

## roundとepisode位置

- 全5episode成功round：L0 1/240（0.42%）、L1 334/640（52.19%）。
- episode位置1→5の成功率：L0 9.58/5.42/5.00/5.42/7.08%。
- episode位置1→5の成功率：L1 57.34/66.25/67.19/68.75/69.06%。
- 位置別の変化の統計的有意差や、相互適応の因果効果は検定していません。
- 完成recipe件数 L0：LettuceOnion67、TomatoCarrot0、PotatoBroccoli11。
- 完成recipe件数 L1：LettuceOnion909、TomatoCarrot0、PotatoBroccoli1194。

## 検証と保存先

Cは全episodeの分母・重複・round完全性と、154845stepのaction/event traceを配膳成功と照合しました。評価処理は約210秒です。

- [結果のDraft Release](https://github.com/kei-student/re-NestRL/releases/tag/untagged-5edae42b2712b1421f89)（ID438811508）。未公開。
- 結果ZIP：l2-performance-evaluation-results.zip、1283278bytes、asset ID726504924。
- ZIP SHA-256：ac3e23f30967d5488ed932a21f1d42f5b5cd59654c29a0701a284ecf1497eb0c。
- ZIPには相手別CSV、episode記録、圧縮済み全trace、評価script、protocol、source/config/checkpoint参照とchecksumを含みます。モデル重みは重複同梱していません。
- Cはアップロードした全assetを再取得してSHA一致を確認。ローカルチャットからもGitHub APIでReleaseとassetの存在・size・digestを確認しました。
- このローカルチャットから評価用branchを作り、requested-protocol.jsonをcommitし、内容を再読込して一致確認しました。C側のGitHub権限の確認とは独立した読み書きの実動作確認です。

## 限界

未知L1、実人間、別layoutでの評価や、Generalist等のbaselineとの同条件比較は今回行っていません。現行の再構成環境（93有効特徴+12予約ゼロ等）での結果であり、著者の観測仕様との同一性や論文の性能再現を示すものではありません。

このbranchは評価記録の保存先です。branch上の初期コードを評価に使用したという意味ではありません。実際の評価に使ったsource snapshot、モデルのmanifest/config、入力archiveのhashと実行protocolは結果ZIPを参照してください。
