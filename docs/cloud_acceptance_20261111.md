# 再現用Cloud基盤の受け入れ記録

2026-11-11。基盤名は `re-NestRL-repro-base`（公開済み、自分のみ）。
実行コードは `53d82cda076d6aec59a503111db69be2a6210c3c`。
運用文書・台帳の後続commitと、実行時コードのSHAを区別する。

- 空の専用.venvからPython3.12.14・NumPy2.5.4・PyTorch2.15.0+cpuを構築できた。
- PACE `f008c19aa743ec209550375c9ee9c25379ac8b24` の6ソースと歴史src14ファイルが一致。
- start・pip check・clean preflight成功。Mac/Linuxとも67 tests中65成功・2 skip。
- 新C `6af47527-2218-8191-9453-e1469d5290ac` のprivate output-rootで検証した。
- smoke pipelineは7モデルが各2更新・160stepsで完了し、L2/generalist評価まで成功。
- bundle再取得後の全hash、7モデルのmanifest/latestとstrictロード、frozen L1 pool参照を確認。
- 旧L2 seed401 ZIP（asset726376755）は元hashと一致し、推論用strictロード成功。
- 正規L1 ID `l1-construction-11` で80stepsのpaused bundleを保存・再取得し、復元コピーを160stepsまで再開。
  別rootの連続実行とmodel・optimizer・乱数・collector state・数値metrics・episodesが一致した。
  時間・resumeイベントは比較から除外。最終ZIPの191ファイルと両final checkpointも検証済み。

任意のL1内部IDによる初回指示は0更新で拒否された。記録を保持し、一意なoutput-rootと
正規内部IDで再検証した。既存run・C1〜C4・監視、元asset、学習source/configは変更していない。
未解決事項はない。検証は短い新規CPU runの範囲であり、旧L2や異なるbackendでの厳密resumeは未検証。
この条件や短い成功率を本実験・将来の予算の既定値には使わない。

[診断Draft](https://github.com/kei-student/re-NestRL/releases/tag/untagged-49b993c26f4f17403789): `441776236`。

| 保存物 | manifest asset ID | manifest SHA-256 | archive SHA-256 |
| --- | --- | --- | --- |
| pipeline | 735740420 | 695b6752a5d48d3bbe2ca5584bed04ad727be4fe9817a36042c456537e998c0b | 21f7411e2082e5f3f816677abb46cc26455cd5d9a4572ad27db117767b0a75c3 |
| paused | 735752265 | f3419ecc82f319b4f0878bbfdcfdf9c3ed888e4bcac937d9a0780f9a5b5904e2 | 5d0962fa0005ab01c5ba26548b17bb152524fd7787ffcf1ca220248c10528888 |
| resume検証最終 | 735753319 | d5004cf4296da3249ab1b62c50863f144126aedc4de4fe09ed59d28968de2b96 | 2dc0b300f86f569cfd95994d4779f09118620a8072f465f2f03a16d70f80289b |

学習器config SHA-256: `451e0b0fc1e5485188ab36dc2050089e735cabd084c42a473226ba2cbae16f28`。
resume検証のwrapper実効条件SHA-256: `2433ca7a7667633b14985a81595670e2ec86f664a917650263da61b71df20934`。
詳細とasset一覧は[実験台帳](experiments/registry.json)、両環境の共通記録は
[NestRL側の受け入れ記録](https://github.com/kei-student/NestRL/blob/codex/cloud-workflow-20261111/docs/cloud_acceptance_20261111.md)を参照する。
