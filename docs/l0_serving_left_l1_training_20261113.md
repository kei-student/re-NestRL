# 左L1・右配膳L0：学習起動記録（2026-11-13）

この分岐専用の新規実験。L1は左agent0、レシピ固定L0専門家は右agent1で配膳を行う。
元チャットの右L1の12体・5タスク・manager・保存先は操作していない。
新しい[設計と検証](l0_serving_role_swap_20261113.md)および
[5タスクの計画](experiments/l1-left-l0-serving-3env-400u-s101-20261113-plan.json)に従う。

固定学習source: `2d82b2e888b7970806ad9b9329bcf73303396d2a`。
branch: `codex/l0-serving-left-l1-20261113`。
各条件seed101–104、400更新/28,800,000steps、CPU・torch/BLAS1thread、毎更新snapshot保存。
全体観測を1タスク4体並列、partialの1種類制限・fullの1種類制限を各2タスク2体並列とする。

| task | seed | Cloud thread（全hostId=durable） | 起動時の確認 |
|---|---|---|---|
| e1-p01 | 101,102,103,104 | 6af67760-7a80-8191-be6b-c88c046dd0ae | 保存更新・startup検証済み |
| e2-p01 | 101,102 | 6af679f5-fe78-8191-ad32-b535f493c0b0 | 保存更新・startup検証済み |
| e2-p02 | 103,104 | 6af67a05-3a50-8191-a1aa-75450474b0bc | 保存更新・startup検証済み |
| e3-p01 | 101,102 | 6af67a11-ac44-8191-9d0a-0caa493a20a9 | 保存更新・startup検証済み |
| e3-p02 | 103,104 | 6af67a28-51e8-8191-866d-769364a5c087 | 保存更新・startup検証済み |

## 環境1の先行gate

4 workerと独立セッションの単一manager PID1349/PPID1が稼働。VM上限4CPU・16GiB、
使用約4.6GiB、OOM/restart0。Docker個別制限は非適用なのでVM全体と実使用で判断した。
固定世代は101:update1/72000steps、102:update2/144000、103:update1/72000、104:update1/72000。
104 unit tests成功（Docker opt-in2件skip）、pip check・clean preflight・PACE6ソース検証成功。

新しいDraft Release443746098へ保存し、Cloud側で再download/hash検証後に残り4taskを作成した。
調整側でも再取得し、104 payload SHA、全snapshot/pointer、seed・config・環境sourceを照合。
全4snapshotのmetadata.ego_side=0とtraining_partners=l0-1の3種を確認し、
collector内の各360 scriptがagent_id1・random_action_probability0であることも確認。
model stateはstrict CPU load成功。観測を左L1に渡すcollector/pipelineとbit-exact resumeは専用unit test済み。

- startup transfer manifest asset: `741994046`
- manifest SHA-256: `e0128850e3695cffd5ca7266dd702fa769e52bbaebb09cbb18e55564a0ba40c0`
- archive SHA-256: `5ef76942394b7ec46eed426679b6820960a2addbc9a057091effec2927285cb1`

これらは起動時点の保存更新であり、400更新完走・最終モデル保存完了の記録ではない。
各managerが担当runのみを最大48時間の有限処理で扱い、全予算後のfinal/manifest/latest/complete/recoveryを
保存し、再download/hash/strict CPU loadを確認する。新しい定期ジョブは作っていない。

## 後続4taskの確認結果

全taskで104 tests（102成功・2skip）、clean preflight・pip checkが成功。
workerとwriter、単一の有限manager、snapshot/pointerと役割metadataを確認した。
各VM上限は4CPU/16GiB、各workerは約1CPU・torch/BLAS1thread。OOM・worker restartはいずれも0。

| task | startup固定世代 | その後の読み取り確認時update | VMメモリ | manager PID / PPID |
|---|---|---|---|---|
| e2-p01 | 101:1、102:1 | 101:3、102:3 | 約3.38GiB | 1080 /1 |
| e2-p02 | 103:1、104:1 | 103:6、104:6 | 約3.4GiB | 1520 /1 |
| e3-p01 | 101:1、102:1 | 101:2、102:3 | 約3.25GiB | 1425 /1 |
| e3-p02 | 103:2、104:1 | 103:3、104:3 | 約3.4GiB | 1813 /1 |

各updateのstepsは72,000倍。上表は各確認時点の記録であり、全taskの同時刻の値ではない。
全12snapshotで左L1/右L0を独立確認し、360×12=4320の保存済みL0 scriptも右agent1であることを照合した。

| task | manifest asset | manifest SHA-256 | archive SHA-256 | payload数 |
|---|---:|---|---|---:|
| e2-p01 | 742022014 | `114afbfcc34558f105ac9710bf071c60357d88d3c11e827cea116803b661a50c` | `c7f747a091dd07607fe5b5beae290e75bfb89ad35f7b0a6c56c91e28e568ed54` | 92 |
| e2-p02 | 742017683 | `cc713e06f2cec66f8a73e76c72710d0e6fa0c51c5ff19e3543b38911fd0df592` | `5eb896511d887b61eb135aee7c52fa3a1d27b84f92ffb542bb44ef5717953300` | 134 |
| e3-p01 | 742022053 | `fe11dc916f7563787c2cb0d1eef55ddda95cf18b82475880ea5714fd53cfdd04` | `dd925ecc954f74e817820f8bdaed59ac32ce06fd753eb1ff1ba373a1a7fcf56c` | 136 |
| e3-p02 | 742024124 | `0d9c4dac0720e6d4a3b5ee47c59fb1528402fdcc1b25ba73ea3af85927ace592` | `0021ef0ec1c13a837ce59e7809f15e492a7ad34d076cc515d74abbd53128bd8e` | 164 |

調整側で全5archiveの再downloadと630 payload SHA、全12正規runのpointer/snapshot、
seed/config/environment hash、strict CPU model loadを検証済み。
e3-p02には保存失敗時の`ops/startup-attempt1-preserved`も含むため、
回復対象は正規run直下のpointerで識別し、失敗stagingのpointerを現行世代と混同しない。

e3側ではDocker `top` のcgroup参照や固定image内の`ps`不在を、ホスト`/proc`の資源収集に修正した。
e2-p02は初回manager終了を検出し、workerを止めず独立セッションの単一managerへ起動方式を修正した。
いずれも学習source/config/runtime/予算には変更なし。e3-p02のmanagerは元の48時間期限を維持。

e2-p02の補足managerコード単独転送は、具体的payloadへの外部共有承認未確認として自動承認審査に
拒否されたため実施せず、状態監査JSONの保存・再取得で記録した。拒否対象コードはローカル保全し、
今後のfinal bundleからも除外することを担当taskが確認した。科学的sourceとstartupモデル・回復状態の保存は成功済み。

## 実験識別と保存先

| condition | canonical config SHA-256 |
|---|---|
| full | `6b50e2f1e0c22ef2f2a4482fd8e1bd409dfdf4fd9477588509c66ee706d34baf` |
| partial + single ingredient | `f0bbdda67e4afe66a7bbc3db059e04198eae74dae963f73e8d030ac645a7e775` |
| full + single ingredient | `b74a3d2b47b25633d1a9dcc315ed2385fc8981c448ab1380cd26f63b3cff4001` |

保存先: `kei-student/re-NestRL` Draft443746098、tag `l1-left-l0-serving-3env-400u-s101-20261113`。
調整記録・prompt・独立download検証は `outputs/l1-left-l0-serving-3env-400u-s101-20261113-coordinator`。
物理環境・既存peer・学習器sourceは変更せず、configの役割を交換した新しい実験である。
旧モデルの継続・移植、元チャットのrun/registry/managerへの操作は行っていない。

