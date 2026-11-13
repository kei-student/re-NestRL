# 3環境・各4体のL1学習開始（2026-11-13）

ユーザー指定により、全条件で同じconstruction seed101,102,103,104を使う。
環境1は1Cloudタスクで4体並列、環境2・3は各2Cloudタスクで2体並列、計5タスク・12体。
各モデル400更新・28,800,000steps、CPU、torch/BLAS1thread、毎更新回復snapshot保存。
ラウンド履歴、ランダム開始、報酬、観測/調理制限は各条件の固定configどおり。
最終予算完走・最終モデル保存完了を報告する文書ではない。

固定source commit: `21d9d1d40948b6756b57875a181c69cc33cffcc8`。
branch: `codex/l1-environment-variants-20261113`。
実行計画は[l1-environment-variants-4agents-s101-20261113-plan.json](experiments/l1-environment-variants-4agents-s101-20261113-plan.json)。
以前の別seed計画は保持する。先行chatで旧seed11001–11004のprocess/container/writer/manager不在を確認し、
学習未起動のまま共通seed版へ切り替えた。旧モデルや途中checkpointからの移植は行わない。
同じseedのrun IDが環境間で重なるため、必ず条件別output root/config/registryとセットで識別する。

| task | 担当seed | Cloud thread（hostIdは全てdurable） | 確認状態 |
|---|---|---|---|
| e1-p01 | 101,102,103,104 | 6af65a75-70e0-8191-8924-e54b2630f08f | startup検証済み・学習継続 |
| e2-p01 | 101,102 | 6af65d2d-3ac4-8191-b253-898fb92a90a1 | startup検証済み・学習継続 |
| e2-p02 | 103,104 | 6af65d37-5e38-8191-abcd-79be474da0ae | startup検証済み・学習継続 |
| e3-p01 | 101,102 | 6af65d41-c31c-8191-b284-c01c4da760b9 | startup検証済み・学習継続 |
| e3-p02 | 103,104 | 6af65d4b-2368-8191-ae56-0d422d52c082 | startup検証済み・学習継続 |

## 環境1の先行確認

- 4体ともworker稼働・保存更新を確認し、単一の有限managerが継続。OOM/restart0。
- VMの実効CPU quota4コア相当、memory.max16GiB。観測使用量約3.73GiB、各worker約1コア。
  Docker個別3GiB/1CPU指定はrootless環境で非適用だが、VM ceiling内での実使用量を確認した。
  platform limitの変更/迂回はしていない。
- manager初回起動でPython symlinkを解決し専用.venvから外れた問題を修正。
  学習ソース/workerには影響せず、managerは単一の正しい.venv実行に直した。
  後続タスクにもsymlink resolveを避ける手順を引き継いだ。
- startupに固定した保存世代はseed101:update2/144000steps、102:update3/216000、
  103:update4/288000、104:update4/288000。各pointerとsnapshot SHAを検証。
- Draft Release443670277へ保存。transfer manifest asset741675724、manifest SHA-256:
  `02efd9379953b709d1864ad06c53c26b39c575fcee63c504701bdf0a59319b72`。
  archive SHA-256:
  `02ed8a3a56005a73be6bf828d3126a9ee68065ae63fbb2ddec039691981b40c2`。
- 調整側Macでも独立に新パスへdownloadし、100 payload member（checksum一覧込み101 zip member）の
  SHA-256、4 snapshot/pointer、seed/config/environment sourceを検証し、model stateをstrict CPU load。
- このgate通過後に後続4タスクを作成した。実行中の初期世代を上記bundleと混同しない。

## 後続4タスクの起動確認

全タスクで同じ固定source/config、97tests（2skip）、pip check・preflightを確認した。
以下は各startup確認時点の保存更新であり、最終400更新の結果ではない。

| task | seedごとのupdate / steps | VM使用メモリ /16GiB | 有限manager PID / PPID | OOM / restart |
|---|---|---:|---|---|
| e2-p01 | 101:2/144000、102:2/144000 | 約3.31GiB | 1218 /1 | 0 /0 |
| e2-p02 | 103:1/72000、104:1/72000 | 約3.30GiB | 1115 /1 | 0 /0 |
| e3-p01 | 101:1/72000、102:1/72000 | 約2.23GiB | 1201 /1 | 0 /0 |
| e3-p02 | 103:1/72000、104:1/72000 | 約3.21GiB | 1140 /1 | 0 /0 |

全workerはCPU約1コア・thread1、各VMはCPU quota4コア相当・memory.max16GiB。
各managerは担当2体のみ、単一プロセス・独立セッションで稼働し、48時間上限。
別seed、別予算、別backendの追加や周期ジョブはない。

| task | startup transfer manifest asset | manifest SHA-256 | archive SHA-256 | payload member数 |
|---|---:|---|---|---:|
| e2-p01 | 741695992 | `a688880f0f0173635849db837add6bd29b43e7610d7b8cdd34ad3ce93d79167c` | `c582cec000c3e9bffc49b2ba142918651ada11e63c54e0eda03928136d83ac43` | 87 |
| e2-p02 | 741692913 | `7df81cac0b3f5825f06beecb266be7573655a86cb1c6850f2cbf1dda2196127a` | `76fd539267885cbb050f850d45a30f848578f1586b67dfec8c16bcf5f89b8378` | 84 |
| e3-p01 | 741696304 | `23a2b49810f4cf2943e8868decb251cacc833a59da001d0b88550fe190ab95e2` | `6be1606ec97303a82f310521ce109d0f71da8362501dcac786ad69838e67d4e4` | 86 |
| e3-p02 | 741697520 | `fd55e1ac8479cd8f78449fda4efff34fa011f4e162fa3fe1df095f14a0356805` | `4f709ea7b0007698effacc4bacec00f698b5f662f8bb109e8d033a43a505e5f7` | 91 |

各Cloud側で新しいパスへの再download、全体・member SHA照合を完了。
調整側でも独立に全startupをdownloadし、全payload SHAとsnapshot/pointer、
seed/config/environment-source一致、全12 model stateのstrict CPU loadを確認済み。

## 条件別configのcanonical SHA-256

| 条件 | config | SHA-256 |
|---|---|---|
| 1 全体観測 | `configs/l1-random-start-400u-full-4agents-s101-20261113.json` | `2457507712257f8a1b1623d6cd5081e2eb8a1b1c98ce4ea12bc59f20995f4f20` |
| 2 1種類制限 | `configs/l1-random-start-400u-single-ingredient-4agents-s101-20261113.json` | `6d7ca74a4e41410bf286c96e78d7f1ff3a5803ff2574e80cdcac719f2621f967` |
| 3 両方 | `configs/l1-random-start-400u-full-single-ingredient-4agents-s101-20261113.json` | `746a747c9f22c135b2f002ef65b0ddace31ad8ed72aff1461cb100de7e15890c` |

保存先: `kei-student/re-NestRL` Draft443670277、tag `l1-environment-variants-400u-4agents-20261113`。
startup/final/recoveryは異なる一意なassetに保存し、hashの異なる既存assetを上書きしない。
各担当タスクの有限managerがこの担当run群だけを扱い、最終保存の再download/hash/strict CPU loadまで確認する。
新しい定期監視・自動タスクは作成していない。97 unittest成功（Docker opt-in2件skip）、workspace dirty/clean検査成功。
source/config/model guardは維持し、新条件はfreshモデルのみ。Cloud学習checkoutは固定commitのまま変更しない。

調整側のdispatch・prompt・gate・独立download検証記録:
`outputs/l1-environment-variants-4agents-20261113-coordinator`。

