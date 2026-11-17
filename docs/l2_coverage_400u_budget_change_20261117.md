# 被覆L2 6体：総400更新への予算変更

2026年11月17日、起動後のユーザー指示「400更新で構いません」を適用する。
対象はre-NestRL環境3の被覆L2、seed181–186の3タスク×2体だけ。
旧L1のみL2の18体・他の実験は対象外。

総400更新＝28,800,000stepsであり、追加400更新ではない。
L0 3体＋各L2で選択済みL1 3体、相手集合・環境・報酬・PPO・モデル・seedは維持する。

## 親の保全

旧1000更新用managerを引継ぎ待ちにし、各workerを既存SIGTERM処理で更新境界停止。
snapshot・metrics・config/registry/source/runtime・writer解放を確認してcopyを保全した。

| seed | 保全更新 | 400までの残り |
|---|---:|---:|
| 181 | 113 | 287 |
| 182 | 114 | 286 |
| 183 | 91 | 309 |
| 184 | 96 | 304 |
| 185 | 92 | 308 |
| 186 | 93 | 307 |

親config・run・元snapshot・元assetは変更しない。
1000という値は親の計画予算であって、1000更新済みの重みを巻き戻す操作ではない。

## 検証付きの子run作成

`scripts/prepare_l2_budget_reduction.py` は書き込み不在のlockを取得し、
親config/registry/source/runtimeとsnapshot SHAを厳密検証する。
許す変更はconfigのtotal_stepsだけで、子の新規run/config/registry識別を作成する。
モデル・optimizer・collector・GRU・RNG・保存update・累積時間は完全保持する。
元のL1三体とL0三体の順序・出自も通常loaderで照合する。
親selectionを原本として残し、子selectionにはbudget_overrideと新config bindingを記録する。
profile文字列は予算以外の差を避けるため元のまま。実際の予算はtotal_stepsとlineageを正とする。

```sh
python PATH_TO_VERIFIED_DRIVER/prepare_l2_budget_reduction.py \
  --parent-root PARENT_ROOT --parent-run-id l2-SEED \
  --child-root NEW_CHILD_ROOT --child-run-id l2-SEED-u400-from1000-plan \
  --target-updates 400 --expected-config-sha256 PARENT_CONFIG_SHA \
  --expected-snapshot-sha256 VERIFIED_PARENT_SNAPSHOT_SHA
```

通常のload_resumeで子snapshotを再読し、metadataのconfig/registry/run識別以外の
全状態が一致することを確認する。再開ガードを無効化しない。
親と同じ専用.venv・CPU・threads1・deterministic設定で実行する。
学習coreはd49d6e9のまま、driverはoutputsへhash確認後にmaterializeできる。
src/environment/model/PACEは変更しない。

子はnative CLIの明示run IDで復旧する。

```sh
python -m nestrl_research train --stage l2 --split construction \
  --config NEW_CHILD_ROOT/config.json --output-root NEW_CHILD_ROOT \
  --seed SEED --run-id l2-SEED-u400-from1000-plan \
  --device cpu --threads 1 --checkpoint-every 1 --resume
```

各taskで2体同時・単一の有限managerとし、旧Docker/managerを再開しない。
同一VM・同一CPU runtimeのnative recoveryである。VM上限の変更はしない。
再開主体を単一にし、processと排他writer不在を確認した同条件回復のみ許す。
400到達後はfinal/latest/complete、全400行のmetrics・履歴・lineage・元参照を、
既存Draft448263501の新しい400u asset名へ保存・再取得・SHA/CPUロード検証する。

## 検証と互換性

小規模fixtureで、元の大きい予算の途中snapshotから小さい総予算へ引き継ぎ、
最初から小予算で走らせた実行とmodel/optimizer/collector/RNGがbit-exact一致することを検証。
active writer・異なるhash・既存child・target超過を拒否し、既にtargetのsnapshotなら
追加更新なしでfinalを作れることも確認する。
旧runは元設定のまま保存し、新しい条件は別root/run/configとする。
source/runtimeや環境・相手の互換性検査は既存どおり維持する。
