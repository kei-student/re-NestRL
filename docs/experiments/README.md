# re-NestRL 実験台帳

正本は[registry.json](registry.json)。既存L1 104〜111、L2 401〜408、11月9日のL2評価、
別チャット管理下のL1 112〜127を登録した。これは過去の条件の記録であり、将来の実験の既定値ではない。
新Cでの短いpipeline・保存復元・CPU resume検証も `RE-CLOUD-ACCEPTANCE-20261111` として別枠で登録した。
[受け入れ記録](../cloud_acceptance_20261111.md)に実施範囲と保存manifest/hashを記載している。

2026-11-12のランダム開始L1・400更新×20体は、過去の台帳を変更せず
[新しい実行計画](l1-random-start-400u-20261112-plan.json)で管理する。
construction seed1001〜1020を10個の新規Cloudタスクへ2体ずつ割り当てる。
この計画は実行指示であり、学習完了や成果物保存完了の記録とは区別する。

ランダム開始seed1021〜1040の400更新モデルは、2026-11-13に全3種L0へ各1000エピソード
ずつ評価した。全相手95%以上は16/20体（80%）。[全20体の評価結果と保存証跡](../l1_random_start_l0_evaluation_20261113.md)
を参照。元の学習config・分割registry・重みは変更していない。

合格16体の全48,000エピソードを[行動分析](../l1_passing16_behavior_20261113.md)した。
複数食材の準備が中心で、最寄り優先には個体差がある。稀な1036の待機2例については
独立した相手開始遅延probeで手掛かりへの反応を検証した。

次の実験用に、ランダム開始・400更新の[3環境条件](../l1_environment_variants_20261113.md)
（全体観測／各エージェント各エピソード1種類の調理／両方）を追加した。
環境・設定の作成と検証のみで、これらの条件の学習は起動していない。

その後のユーザー指示による各条件4体の学習は、別の
[12体・5タスクの実行計画](l1-environment-variants-4agents-20261113-plan.json)
で扱う。環境1の4体並列を先に検証してから環境2・3の各2タスクへ展開する。
起動計画と実測進捗・完了/保存確認は区別して記録する。

起動準備中のユーザー変更により、現在の計画は
[全条件seed101–104を揃えた版](l1-environment-variants-4agents-s101-20261113-plan.json)。
上記の別seed割当は履歴として保持し、新条件の出力とは混ぜない。

全5タスク・12体の保存更新・startup保存検証が揃った。
[起動確認結果、Cloud thread、保存manifest](../l1_environment_variants_training_20261113.md)
を参照。最終400更新の完走・最終モデル保存とは区別する。

分岐チャットの新実験は[L0配膳役・左L1の役割交換](../l0_serving_role_swap_20261113.md)。
同じ3環境・各seed101–104を、L0右/L1左の独立設定・runで学習する。
元チャットの上記12体や管理タスクには干渉しない。

役割交換した新12体も全5taskで起動確認済み。
[左L1・右配膳L0の起動証跡](../l0_serving_left_l1_training_20261113.md)に
snapshot、役割、資源、manager、startup保存の検証を記録した。400更新完走・最終保存は未完了。

- 元A/Cのソースは[provenance記録](../provenance/a_snapshot_20261111.json)と元source commitで識別する。
- 112〜127の実効設定は[履歴設定](../../configs/history/paper-l1-112-127.json)として元bytesを保存した。
- Release target commitと、dirtyな実行時source、再構成用commitを区別する。元manifestを更新しない。
- GitHubのasset情報は取得時点の一覧。モデルの状態や学習の進捗は元の管理チャットで確認する。
- 学習済み相手での評価をheld-out汎化と呼ばない。

研究全体の案内は[NestRLの総合案内](https://github.com/kei-student/NestRL/blob/codex/cloud-workflow-20261111/docs/INDEX.md)。
基盤・保存・実験ごとの指定方法は[Cloud基盤](../cloud_base.md)。
