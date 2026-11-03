# クラウドCPU学習と復旧

クラウド内のDockerデーモンが学習コンテナを管理します。起動コマンドや
チャットの実行セッションから独立し、Macのスリープやクライアントの切断に
依存しません。ただし、クラウドのマシン自体が停止すれば学習も停止します。
この仕組みはサービスのアイドル停止・実行時間制限を無効にしません。
環境の削除・再作成に備え、必要な成果物はrun全体を別途バックアップしてください。
Save and Publishは実行中プロセスの保存機能ではありません。

## 準備

既存の `/workspace/re-NestRL` とPython 3.12の `.venv` を使います。
Git worktreeを追加する必要はありません。Dockerが動いているこのクラウドの
Linux環境で検証しています。Dockerイメージを一度取得します。

```sh
cd /workspace/re-NestRL
source .venv/bin/activate
env -u DOCKER_HOST -u DOCKER_CONTEXT -u DOCKER_TLS -u DOCKER_TLS_VERIFY -u DOCKER_CERT_PATH \
  docker --host=unix:///var/run/docker.sock pull \
  python@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159
python scripts/bootstrap_upstream.py
python scripts/check_workspace.py --device cpu --allow-dirty
```

ジョブは既存 `.venv` とそのベースPythonを読み取り専用でマウントして使います。
新しいPython/PyTorchへの置き換えはありません。ネットワークを無効にし、
指定した実験の出力先だけを書き込み可能にしています。
この方式は同じクラウド内のランタイムパスを前提とし、任意の外部サーバーへ
コンテナ単体を移して動かすためのイメージではありません。

## 3体を起動する

未使用の出力先を選びます。次は実行例であり、環境の起動時に自動実行しません。

```sh
python scripts/cloud_train.py start \
  --config configs/paper.json \
  --output-root outputs/l1-paper-cpu-recovery-001 \
  --seeds 101 102 103
```

各ジョブをCPU 1コア相当、PyTorch 1スレッド、メモリ上限8 GiBで実行します。
3体合計でCPU 3コア相当・メモリ上限24 GiBです。`paper.json` の予算は各
10,080,000ステップ、72,000ステップ/PPO更新、140更新です。
ハイパーパラメータや分割seedは変更しません。

通常のコマンド実行セッションの `&` や `nohup` に頼らず、`docker run --detach`
でデーモンに起動を依頼します。起動後、コマンドは終了して構いません。
Dockerは非ゼロ終了時に最大3回、チェックポイントからの再起動を試みます。
手動停止やクラウド自体の停止後は、自動再開を前提にせず状態を確認してください。

## 進捗・停止・再開

上と同じ設定・出力先・seedを指定します。

```sh
python scripts/cloud_train.py status \
  --output-root outputs/l1-paper-cpu-recovery-001 --seeds 101 102 103

python scripts/cloud_train.py stop \
  --output-root outputs/l1-paper-cpu-recovery-001 --seeds 101 102 103

python scripts/cloud_train.py resume \
  --output-root outputs/l1-paper-cpu-recovery-001 --seeds 101 102 103
```

`status` はコンテナの状態・終了コード・再起動回数・最終メトリクス・
チェックポイントの有無を表示します。ステップ数は最新のログ値で、
強制停止時の再開点はハッシュ検証済み `resume/latest.json` が指す更新です。
`resume` は既存の停止コンテナを再起動するか、失われたコンテナを作り直します。
完了済みrunは再学習しません。

`stop` はSIGTERMを送ります。処理中のPPO更新を終えて保存・停止するまで
最大120秒待ちます。強制終了時も、最後に保存済みの更新から再開できます。
1更新ぶんの未保存処理は再計算します。

各体のログは `seed-101.log`、runのメトリクスは
`l1-construction-101/metrics.jsonl`、完了の根拠は `complete.json` です。
このスクリプトにチャットへの定期通知機能はありません。

Dockerを使わず、既存runを手動で再開することもできます。

```sh
python -m nestrl_research train --config configs/paper.json \
  --output-root outputs/l1-paper-cpu-recovery-001 \
  --stage l1 --split construction --seed 101 --device cpu --threads 1 --resume
```

## 保存内容と制限

- 開始時と各PPO更新後に `resume/update-NNNNNN.pt` を保存し、直近2世代を残します。
  ハッシュと最新更新を `resume/latest.json` に記録します。
- 重み、Adamの状態、GRUの状態、各環境の現在エピソードの行動列、相手方策の履歴、
  エピソード記録、Python/NumPy/PyTorchの乱数を保存します。
  PACEの環境はseedと行動列から再構築し、観測の一致を検証します。
- 保存は一時ファイルとrename・fsyncを使います。新世代の公開前に強制終了しても、
  公開済みの世代を保持します。破損した最新世代を自動的に無視することはありません。
- 二重書き込みはOSのファイルロックで拒否します。プロセスが死ねばロックは解放され、
  残ったPIDファイルを削除してロックを回避する必要はありません。
- 再開には同じ設定・registry・実装ソース・学習相手と、同じPython/NumPy/PyTorch・
  CPUバックエンド・スレッド数を使います。異なる設定・コード・依存バージョン・
  デバイス・スレッド数は拒否します。異なるCPU機種をまたぐビット単位の一致は未検証です。
- CPU再開を実装・検証しています。CUDA/MPSでは `--checkpoint-every 0` を使い、
  厳密な学習再開は利用しません。既存の最終チェックポイントの評価機能は維持しています。
- `--checkpoint-every N` で更新間隔を指定できます。既定はCPUで1です。
  `--max-updates N` は予算を変更せず、その呼び出しをN更新で途中保存・停止する検証用です。
- この機能導入前に停止したrunは状態を保存していないため、復旧できません。

## 検証

```sh
python -m unittest discover -s tests -q
NESTRL_DOCKER_TESTS=1 python -m unittest discover -s tests -q
```

通常のテストに加え、後者はローカルDockerが必要な結合テストも実行します。
起動コマンド終了後の進捗、Dockerの強制停止、別コマンドからの再開、
3,200ステップでの完了を検証します。テスト専用のコンテナと出力は後片付けします。
CPUの連続実行と途中再開で、重み・最適化器・環境/GRU/乱数の状態が一致すること、
保存中断・破損・二重書き込みの拒否もテストします。
