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

- 元A/Cのソースは[provenance記録](../provenance/a_snapshot_20261111.json)と元source commitで識別する。
- 112〜127の実効設定は[履歴設定](../../configs/history/paper-l1-112-127.json)として元bytesを保存した。
- Release target commitと、dirtyな実行時source、再構成用commitを区別する。元manifestを更新しない。
- GitHubのasset情報は取得時点の一覧。モデルの状態や学習の進捗は元の管理チャットで確認する。
- 学習済み相手での評価をheld-out汎化と呼ばない。

研究全体の案内は[NestRLの総合案内](https://github.com/kei-student/NestRL/blob/codex/cloud-workflow-20261111/docs/INDEX.md)。
基盤・保存・実験ごとの指定方法は[Cloud基盤](../cloud_base.md)。
