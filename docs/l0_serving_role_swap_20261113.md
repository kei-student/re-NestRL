# L0配膳役・左側L1の役割交換実験（2026-11-13）

この分岐は既存の右側L1実験とは独立して管理する。L1を左側agent0、凍結L0を右側agent1とし、
右側L0が配膳まで行う。既存のCloud chat、学習run、manager、保存先には書き込まない。

## 右側L0の方策設計

実装は既存の `ScriptedPeer(recipe, agent_id=1, random_action_probability=0)` を用いる。
レシピはLettuceOnionSalad、TomatoCarrotSalad、PotatoBroccoliSaladの3種類で、それぞれ固定の専門家。
L0は自身のレシピを変更せず、学習もしない。エピソード中の制御は次のとおり。

1. 自分側の対応食材（Onion / Carrot / Broccoli）を取り、右のまな板へ運ぶ。
2. 刻み、回収して中央の共有皿へ載せる。
3. 皿が完成するまで待つ。完成前に皿を引き上げない。
4. 皿の食材集合が自分の指定レシピの2種類と完全一致したら、皿を取る。
5. 右側の配膳場所へ運び、提出する。余分・不一致・未完成の料理は提出しない。

移動は既存の位置・向きを状態とする最短経路探索を使う。L1の行動を強制せず、
左側L1が対応食材を準備して共有皿へ渡す必要がある。L0への正解レシピ指定は学習L1には渡さない。
方策の右側配膳処理は既に実装済みなので、同じロジックを複製・改変せず、役割と検証を固定する。

## 学習条件

3条件は全体観測、各エージェント各エピソード1種類の調理制限、両方の組合せ。
各条件のL1はseed101–104の4体でfresh initializationから学習する。
ランダム開始、400更新（28,800,000steps）、CPU、torch/BLAS各1thread、40ステップ、
5エピソードのラウンド履歴、報酬/PPO/105次元入力は元条件を維持する。
設定では `l1_ego_side=0`、`ego_side=1` とし、L1の学習相手metadataは `l0-1-<recipe>` となる。
後者のego_sideはL2/Generalist用の反対側役割で、今回L1の配膳担当を指定するものではない。

環境1は1Cloudタスクで4体並列、環境2・3はそれぞれ2Cloudタスクで2体並列。
まず環境1の4体の保存更新、資源実測、startup保存・再download/hashを確認してから展開する。
新しいconfig/registry/output namespaceとDraft Releaseを使い、元チャットの12体とは混ぜない。

## 互換性

物理配置や環境ダイナミクス、既存peer実装は変更しない。役割交換は新しい科学的条件であり、
同じ105次元・同じseedでも旧モデルを継続・移植しない。config/registryの厳密な照合を維持する。
元チャットでの過去・実行中のモデルや記録は変更しない。

## 検証結果と実行設定

- 3条件×3レシピ×12左開始位置×12右開始位置の1296ケースで、右L0による時間内提出を確認。
  左側は皿を持たず、最終行動は右側のPlateを持ったinteract、共有return16.5。
- 部分観測・1種類制限では、共有部分に見える右食材から初めてレシピを推定し、
  それまで左側は待つ観測可能な追従baselineでも全432配置で時間内提出できる。
- 未完成皿の待機、誤った食材や余分な食材を含む皿を提出しないことを確認。
- 小規模pipelineテストでL1のmanifestがego_side0、学習・評価相手が全てl0-1であることを確認。
  左L1のCPU再開も、連続実行とmodel/optimizer/エピソード履歴が完全一致する。

| 条件 | config | canonical SHA-256 |
|---|---|---|
| 全体観測 | `configs/l1-left-l0-serving-full-400u-s101-20261113.json` | `6b50e2f1e0c22ef2f2a4482fd8e1bd409dfdf4fd9477588509c66ee706d34baf` |
| 1種類制限 | `configs/l1-left-l0-serving-single-ingredient-400u-s101-20261113.json` | `f0bbdda67e4afe66a7bbc3db059e04198eae74dae963f73e8d030ac645a7e775` |
| 両方 | `configs/l1-left-l0-serving-full-single-ingredient-400u-s101-20261113.json` | `b74a3d2b47b25633d1a9dcc315ed2385fc8981c448ab1380cd26f63b3cff4001` |

L0の固定仕様は`configs/l0/right-serving-recipe-pool-v1.json`。
5タスクの割当・正確なlaunch/record commandは
`docs/experiments/l1-left-l0-serving-3env-400u-s101-20261113-plan.json`。
保存先は新しいDraft Release443746098、tag `l1-left-l0-serving-3env-400u-s101-20261113`。
元チャットのRelease443670277や担当chatには書き込まない。

全5task・12体の保存更新、役割metadata、startup保存・再download/hash・strict CPU loadが揃った。
[起動記録・Cloud thread・保存manifest](l0_serving_left_l1_training_20261113.md)を参照。
400更新完走と最終保存は別途担当managerが記録する。
