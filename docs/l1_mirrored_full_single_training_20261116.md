# 左右対称・両側配膳：全体観測＋1食材制限 L1学習

ユーザー指定：4体、Cloud2タスク×各2体並列。
全体観測＋各agent/各physical episodeにつき1食材種類、ランダム開始、400更新を学習する。
L1は左agent0、相手は右agent1の凍結L0三専門家。両者とも自分側の配膳口を使用できる。

学習ソース：`4ad6d8a6d82c2eb04a3062b91bbe7db0690029bf`。
branch：`codex/mirrored-dual-delivery-20261115`。
config：`configs/l1-mirrored-dual-delivery-full-single-ingredient-400u-s101-20261115.json`。
canonical SHA-256：`f08c233af8b7a7102bb61aea2ec512bc23f2b72b978dacbc5dc633933e4b949c`。
file SHA-256：`7a6ff4d8d6c0c6c577aa1028fd61468836617f23be9880a121a83b7f423c533b`。

| task | construction seed | Cloud thread（hostId=durable） | 専用output root |
|---|---|---|---|
| p01 | 101,102 | `6af9c633-3eac-8191-aa25-02a5dee1a093` | `outputs/l1-mirrored-full-single-400u-s101-20261116-p01` |
| p02 | 103,104 | `6afa5968-b220-8191-a1ff-db8c03aa70b0` | `outputs/l1-mirrored-full-single-400u-s101-20261116-p02` |

各体400 PPO更新/28,800,000steps、4体合計115,200,000steps。
batch72,000、CPU・Torch/BLAS1thread、毎更新snapshot。観測108、action6、40step、
5episode recurrent round。construction101–104、validation111、test121–125のpartitionは固定。
以前の105次元モデルを移植・再開せず、全て新規初期化する。追加評価やモデル選別はこの起動に含めない。

## 先行タスクの起動確認

条件確認中に用意したp01の同じVMで、条件確定後にseed101/102を開始した。
科学ソース/config/runtimeは固定、clean。実効上限4CPU/16GiB、startup使用約2.2GiB。
各worker約100%CPU、RSS約529/488MiB、各1thread。worker/writerが稼働しOOM/restart0。
単一の有限managerはPID/SID934、PPID1。最大48hで担当runの400更新と最終保存検証まで処理する。
定期ジョブや追加Cloudタスクは作らず、同じrunの検証付きCPU recoveryだけを限定的に許可する。

保存固定世代は両seedともupdate3/216,000steps。
その後の実測報告では両seedともupdate12/864,000stepsまで進行。
コーディネーター側で両snapshotのSHA、config/registry/source hash、左L1/right L0三種、
collectorの両側観測(360,2,108)とL1入力(360,108)、360凍結right L0 scriptsを検証し、
108入力モデルへのstrict CPU loadを確認した。

## 保存経路の補正と後続起動

保存先は新しいDraft Release **446495466**、repo `kei-student/re-NestRL`、
tag `l1-mirrored-full-single-400u-s101-20261116`。

p01の初回bundleには`.py`の一律除外により固定Python25ソースが欠けていた。
これは自動承認審査の拒否ではなく、担当タスクのbundle選別処理が原因だった。
不足分だけを一意な補完bundleへ保存し、全64の記録source原本が元の実行記録・固定ソースと
SHA一致することをコーディネーター側でも確認した。既存assetは上書きしていない。
科学ソース/worker/managerは変更せず、final用の全64原本archiveを実行記録に追加した。
既存managerが最終保存時にそのarchiveを取り込む。

| p01保存証拠 | transfer manifest asset | manifest SHA-256 | archive SHA-256 |
|---|---:|---|---|
| startup、74payload | 752945446 | `dedbfb9e30ed30d4e8d7f604fa201fbb19c6a171f79fbf48f6335d02e43d2a76` | `10b90d2fe30bb5bc3656aa24c6f0ec27ce1575e098835c4607641ba01edcfd24` |
| 固定Pythonソース補完、26payload | 752958608 | `8c72e0524cbe22180dff277aa2137fb5e18a45b407c819683b9802924b20e20a` | `e7eb859cba1214638fd20bd7dbbd99ff17ad39b52739be3a3c68b058aa7e8e59` |
| worker/manager追加実測、2payload | 752949379 | `90e10c2a2c09925eedc781114c1cf3f5407900790fff0d9492cafe1ae211cb31` | `f5a62d4ebb3b71da04e500c53c3fe91ab6cffd32a1bbd40dcbc5924e0a2ff530` |

コーディネーターが新パスへ再downloadし、全payload・64ソース・両モデルを照合した後に
p02を作成した。p02へは同じ手順と、固定Pythonソースを一律除外しない修正を引き継いだ。

## 後続タスクの確認結果

p02も2体の保存更新・稼働・startup保存検証を通過した。
固定保存世代はseed103/104ともupdate1/72,000steps、その後のCloud実測は両方update3/216,000steps。
実効上限は4CPU/16GiB、使用量は約3.4–3.5GiB。workerとwriterは稼働、OOM/restart0。
単一managerはPID/SID1236、PPID1の独立sessionで継続する。
rootless Dockerの個別制限やPID列挙・stats表示と、実効VM上限/実使用は区別して判断した。

p02 startupは`p02-startup-1794792124.tar.gz`。
transfer manifest asset **752981004**、manifest SHA-256
`6afecc06c6c591cceebe3aec156197ab1d40e48bf841d5bf14ea29baf2c48602`、
archive SHA-256 `4170eefcc46fad6f69292d1f571190a3e65fa9c525d61f209a2caa7007440e2a`。
コーディネーターが新たに取得して全126payload、全64ソース原本、両snapshotとregistry、
左右役割/360 frozen right L0/観測108/CPU strict model loadを検証した。

全4体のstartup検証は完了。2タスクの保存物で合計228payload hash、各64ソース、
4snapshotと1,440保存L0 scriptの役割をコーディネーター側で照合した。
4体とも400更新を目指して学習中で、完走/最終モデル保存はまだ未確認。
最新の同時刻測定ではなく、各taskの上記確認時点の値を報告している。

## 最終成果物と互換性

各有限managerは担当runの全予算後、final.pt/final.json/latest/complete、全metrics/episodes/log、
config/registry/source/runtime/実行記録とrecoveryを保存し、再download/SHA/strict CPU loadまで検証する。
起動時のsaved updateとstartup保存は、400更新完走・最終保存完了の主張ではない。
途中停止時はwriter不在とsource/config/registry/runtime/backend/threads/snapshot SHAの一致が必要。
旧checkpoint/fresh初期化で停止runを置き換えず、互換性検査を迂回しない。

ローカル検証：workspace check、PACE6ソース、119unittest（117成功/2 Docker skip）。
ソースと設定を変更しない起動・保存・状態記録のみ。
計画は`docs/experiments/l1-mirrored-full-single-400u-s101-20261116-plan.json`、
生のdispatch/prompt・独立download検証は`outputs/l1-mirrored-full-single-400u-s101-20261116-coordinator`。
