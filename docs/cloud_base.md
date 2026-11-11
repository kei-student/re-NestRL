# 再利用する基盤と実験ごとの指示

更新: 2026-11-11。

## 固定する共通基盤

- ソース: `kei-student/re-NestRL`の指定commit。研究用NestRLとは別の環境・`.venv`。
- Python3.12、NumPy2.5.4、PyTorch2.15.0（CPU wheelの`+cpu`表記可）。
- 固定PACE: `f008c19aa743ec209550375c9ee9c25379ac8b24`。上流ソースを変更しない。
- 定義: [cloud_base.json](../configs/cloud_base.json)、[依存constraints](../constraints/cloud-cpu-py312.txt)。
- Install: `sh scripts/setup_cloud.sh`。Start: `sh scripts/start_cloud.sh`。学習・resumeは開始しない。
- GitHubの対象repoへの既存認可と、Release保存用の環境Secret `RELEASE_GITHUB_TOKEN`。
- bundle・manifest・checksumによる保存/復元と、実際のコマンド・source/config/runtime・ログの記録。

## 実験ごとに指定する条件

手法、stage、相手pool、seed、総更新数・step予算、停止規則、評価方法、並列数を、
各タスクの指示からconfig/CLI引数へ落とし込んで保存する。A/Cは仮の名称で、恒久的な実験フローを定めない。
`paper.json`、`paper-eight-l1.json`や過去の140更新は既存条件であり、将来の実験の既定値ではない。
条件が変わる実験は新しいconfig/run IDで始め、既存のモデル・manifest・実行記録を書き換えない。
CLIで条件を上書きする場合はbase configと上書き値を含む実効条件JSONを作り、記録用の
`--config`へ渡す。学習器の内部config hashと実効条件hashを区別し、元configも保持する。

実行記録は、明示的なコマンドを渡して作成できる。

```sh
. .venv/bin/activate
python scripts/record_run.py \
  --run-id <今回のID> --config <今回の設定JSON> \
  --output outputs/<今回のID>/execution \
  -- .venv/bin/python -m nestrl_research <今回の引数>
```

記録には設定の原本、code SHA、source bytes/hash、依存、stdout/stderr、終了状態を含める。
学習器自身が保存する解決後config、registry、checkpointのmanifestも一緒に扱う。
source/configの途中変更は`inputs_changed`とし、既存の記録ディレクトリは上書きしない。

## GitHub保存・再取得

`scripts/release_artifacts.py`は実験方法を選ばず、指定bundleを指定Draft Releaseへ保存する。
事前にモデル、manifest、latest、config、source/runtime、metrics/log、必要なrecoveryをbundleへ含める。
認証値は環境から読み、引数や出力へ表示しない。

```sh
python scripts/release_artifacts.py --repository kei-student/re-NestRL \
  create --tag <今回の一意タグ> --title <実験名>
python scripts/release_artifacts.py --repository kei-student/re-NestRL \
  upload --release-id <ID> --file <bundle.zip>
python scripts/release_artifacts.py --repository kei-student/re-NestRL \
  download --manifest-asset-id <保存結果のID> \
  --manifest-sha256 <保存結果のSHA256> --output <新しい復元先.zip>
```

大きなbundleは既定2,500,000bytesのpartsに分け、順番・各part hash・全体hashをmanifestへ保存する。
保存後に全体を再取得・復元して照合する。同名assetは同じbytesなら再利用し、異なる場合は失敗する。
CDNへのredirectでは新しい認証なしのリクエストを使う。復元先に異なるファイルがあれば上書きしない。
実行完了と保存確認は別に報告する。

## 元ソースの保全と互換性

Aで検証した実行ソースはcommit `e8433ffb2332e2b964e0d026d3d139b4ccbdf4ad`へ先に保存した。
対応は[provenance記録](provenance/a_snapshot_20261111.json)で確認する。
過去のRelease target commitやモデルmanifestを新commitで置き換えない。
モデルの推論用ロードと、環境/RNG/optimizer/履歴を含む厳密resumeは別の契約である。
現行のrecovery実装の検査を通し、同じ条件/backendでだけresumeする。
この基盤整備では`src/nestrl_research`、科学的config、既存checkpoint形式を変更しない。
稼働中のC1〜C4とその監視・割当は変更しない。

## 新しいタスクへの引き継ぎ

基盤を選んだ後、source branch/commit、一意な実験ID、config/CLI条件、手法・相手・split・seed、
予算と停止条件、backend/並列数、評価条件、保存先Draftを今回の指示として指定する。
prepared filesystemのGit状態を実行元と決めつけず、指定sourceをclean checkoutへ取得してstart検査する。
不足する科学的条件を基盤の過去設定から補って本学習を開始しない。
完了時は実行記録とmodel/manifest/latest/recoveryを保存し、再取得hashとモデル読み込みを確認する。
L1の内部run IDはregistryで `l1-<split>-<seed>` に固定されるため、任意の`--run-id`を指定しない。
実験全体の一意性は`--output-root`と記録用wrapperのIDで確保し、内部IDとnamespaceを両方保存する。
詳細な[依頼テンプレート](https://github.com/kei-student/NestRL/blob/codex/cloud-workflow-20261111/docs/cloud_workflow.md#新しい実験を依頼するとき)は研究用repoの共通運用文書を参照する。

## 2026-11-11の構築確認

`53d82cda076d6aec59a503111db69be2a6210c3c` をLinuxで空の専用.venvから構築できた。
Python3.12.14、NumPy2.5.4、PyTorch2.15.0+cpu、start・pip check・clean preflight成功。
67 unit tests中65件成功・2件skip。PACE6ソースと歴史src14ファイルのhash一致を確認した。
新環境名は `re-NestRL-repro-base`。GitHubへの実pushとremote SHA照合、Draftへの保存・再取得・
hash照合・同じassetの再利用まで成功し、基盤を公開した。
新しいC `6af47527-2218-8191-9453-e1469d5290ac` でパイプライン・保存復元・旧モデル推論ロードと、
CPUでの80→160stepsの復元resume確認も完了した。[受け入れ記録](cloud_acceptance_20261111.md)を参照。
既存のre-NestRL環境・C1〜C4は変更していない。
