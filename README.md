# re-NestRL

[NestRL: A Nested Training Regime for Mutual Adaptation in Human-AI Teaming](https://arxiv.org/html/2602.17737v2)
の **新規・独立した再構成実装**です。ユーザー提供の `nestRL.pdf` と arXiv v2
を根拠にしています。以前のローカル NestRL 実装・設定・チェックポイントは使用していません。
このフォルダ自体が独立した Git リポジトリです。

現時点では **学習・評価パイプラインの動作確認まで完了**しています。論文の性能値の
再現は未達です。長期学習、人実験、PACE / LIAM / LIAMX / LILI の比較実装は未実施です。

## 実装内容

- 図1の7×8調理環境。公式 PACE の固定版を未変更で使用し、配置・報酬・観測をローカルで構成。
- 本文の3レシピ: Lettuce + Onion、Tomato + Carrot、Potato + Broccoli。
  Appendix B.2 の別の組み合わせは、ユーザー指示に従い採用しません。
- 両側の L0 方策。最短経路で食材取得・切断・盛付・右側から配膳。
- L1: 固定 L0 を相手に学習する、MLP[128,128] + GRU128 の適応方策。
- L2: **凍結した construction L1 だけ**を相手に学習。
- Generalist: 同じモデル・予算で L0 を相手に学習する比較条件。
- PPO / GAE / 20ステップ BPTT、5エピソード単位の履歴、分割された seed registry。
- 相手別・エピソード位置別・全5エピソード成功率、全行動の記録、SHA-256付き成果物。

論文の匿名公開コードは取得時に HTTP 410 でした。観測105次元の全 index、視野、
初期配置、総学習量などは本文に十分な指定がなく、著者実装との一致は確認できません。
とくに **図1の配置では PACE の観測式が93次元になるため、残り12枠を予約ゼロにしています**。
これは明示的な再構成上の差分です。

根拠と差分は [再現仕様](docs/reproduction_spec.md)、[資料監査](docs/source_audit.md)、
[環境・観測仕様](docs/environment_schema.md) に記録しています。
実行した48テストと短い学習の結果は [検証記録](docs/validation_report.md) を参照してください。

## 準備

Python 3.12 を使い、このフォルダ内で実行します。

macOS:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python scripts/bootstrap_upstream.py
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python scripts/bootstrap_upstream.py
```

上流は `external/PACE` にダウンロードされ、固定commitとアーカイブSHA-256、
実行する6ソースのSHA-256を検証します。既存フォルダは上書きしません。
GPU学習では、仮想環境の PyTorch がそのCUDA環境を利用できることを先に確認してください。

## 検証

```sh
python scripts/check_workspace.py --device auto --allow-dirty
python -m unittest discover -s tests -q
```

クリーンなブランチを別マシンへ渡す前は `--allow-dirty` を外してください。

## 小さい動作確認

```sh
python -m nestrl_research pipeline --config configs/smoke.json --output-root outputs/smoke-new --device cpu
```

5個の L1（construction 2 / validation 1 / test 2）、L2、Generalist を順に学習し、
validation / test L1 と固定 L0 で評価します。各方策160環境ステップ、各相手2ラウンド×5エピソードです。
**性能評価用の学習量ではありません。** 完了済み run ID は上書き・再開できません。
CPUの未完了runは、途中保存があれば `train --resume` で再開できます。

## 長期学習用の設定

```sh
python -m nestrl_research pipeline --config configs/paper.json --output-root outputs/paper-cuda-001 --device cuda
```

Table 3 の batch=72,000、4 epochs、18 minibatches と他のPPO条件を使用します。
L1 は construction 3 / validation 1 / test 5 の9個です。
L1・L2・Generalist の **各方策10,080,000ステップ、合計110,880,000ステップ**を要します。
この予算・construction数・seed値は論文未記載のため、この実装で固定した条件です。
CPUでこの長期学習は起動していません。

段階ごとの実行も可能です。

```sh
python -m nestrl_research train --config configs/paper.json --output-root outputs/paper-cuda-001 --stage l1 --split construction --seed 101 --device cuda
python -m nestrl_research train --config configs/paper.json --output-root outputs/paper-cuda-001 --stage l1 --split construction --seed 102 --device cuda
python -m nestrl_research train --config configs/paper.json --output-root outputs/paper-cuda-001 --stage l1 --split construction --seed 103 --device cuda
python -m nestrl_research train --config configs/paper.json --output-root outputs/paper-cuda-001 --stage l2 --device cuda
```

held-out 評価の前には、その設定で定義された validation / test L1 を各 seed で学習します。
L1 の学習はすべて固定 L0 相手であり、L2 はそのうち construction 方策にしか触れません。
最終予算のチェックポイントを固定して評価し、この実装はテスト成績による選択を行いません。

## 成果物と実験の扱い

`outputs/<experiment>/registry.json` は seed / role / split を凍結します。
各 run には設定、実行環境、学習相手のhash、学習曲線、エピソード記録を保存します。
`checkpoints/` の `final.pt`・`final.json`・`latest.json` をセットで扱い、読み込み時にhashを検証します。
評価の `protocol.json` / `summary.json` / `episodes.json` / `traces.jsonl` は、評価条件と全行動を残します。

異なるバックエンドで同じrunを継続せず、新しい出力先とrun IDを使用してください。
CPU学習では既定で各PPO更新後に環境/RNG/履歴を含む再開用状態を保存します。
クラウドでチャットから独立して起動する方法と途中再開の条件は
[クラウドCPU学習と復旧](docs/cloud_training.md) を参照してください。
最終 `checkpoints/final.pt` 単体は推論・評価用です。学習再開には `resume/` を含むrun全体が必要です。
旧 NestRL 実装や著者のチェックポイントとの互換性はありません。

今回学習した8体のL1を凍結して相手に使う準備と、L2の短い動作確認は
[L2試作の準備](docs/l2_prototype.md) にまとめています。

専用の共有先は [kei-student/re-NestRL](https://github.com/kei-student/re-NestRL) です。
旧 NestRL の共有リポジトリは使用しません。
