# 今回の8体のL1を使うL2試作

## 準備済みの構成

- 出力先：`outputs/l2-prototype-20261109-001`
- 相手：construction L1 seed 104〜111の8体すべて。評価成績で選別しない。
- L2は左側（agent 0）、相手L1は学習時と同じ右側（agent 1）。
- 各環境で8体から一様に相手を抽選し、5エピソードのラウンドを維持した後に再抽選。
  相手IDはL2の観測に追加しない。
- L1はevalモード・勾配無効で凍結。L2だけPPOで更新する。
- 構造はMLP 128×128＋GRU128、約13.1万パラメータ。
- CPU版PyTorch、1ジョブ1コア・1スレッド・メモリ上限3 GiB。
- 各PPO更新後に再開用スナップショットを保存。Dockerが独立ジョブを管理する。

元モデルはGitHub Release ID `433508587`、実験 `l1-eight-cpu-20261103T050914Z`、
学習ソースコミット `be94a05a2f2397d6f571493dbc1c2ba5e6d587a8` に対応する。
ダウンロードし復元・ハッシュ検証済みの元ZIPから、最終チェックポイントと
設定・完了記録を別の出力先にコピーした。元ZIPと元モデルは変更していない。
対応関係は出力先の `l2-preparation.json` に保存している。

## 再配置する手順

```sh
cd /workspace/re-NestRL
source .venv/bin/activate
python scripts/prepare_l2.py \
  --config configs/paper-eight-l1.json \
  --model-index /workspace/scratch/nestrl-release-verify/assets/model-index.json \
  --archives-dir /workspace/scratch/nestrl-release-verify/download-verification/reassembled \
  --output-root outputs/l2-prototype-20261109-001
```

モデル台帳、ZIP、チェックポイントのハッシュ・設定・分割を検証する。
既存の異なるファイルは上書きしない。再実行しても同じモデルを配置できる。
8体を読み込んで凍結状態でのバッチ推論を確認する。

## 実施済みの動作確認

確認用seed **499** のL2を1更新だけ動かした。
`l2-499/paused.json` が72,000ステップでの途中停止を記録している。
`l2-preflight-verification.json` に以下の確認結果を保存している。

- 実環境・実L1モデル8体を使って72,000ステップを収集。
- 全8体が訓練相手として出現。
- PPOでL2の重みと最適化器が更新された。
- L1のチェックポイント8体分は変更されていない。
- 再開用スナップショットのSHA-256が一致。
- Dockerジョブは終了コード0で停止。

初回更新は約40秒、成功率は0%だった。実行確認であり、性能評価ではない。
確認用の499と、以下の試作候補401は別runとして扱う。

## 次に試作学習を開始する場合

以下は準備した手順であり、まだ実行していない。

```sh
python scripts/cloud_train.py start --stage l2 \
  --config configs/paper-eight-l1.json \
  --output-root outputs/l2-prototype-20261109-001 \
  --seeds 401 --memory-gib 3 --max-updates 10
```

最初の試作は10更新＝720,000ステップで途中保存して停止する。
`--max-updates` は今回の呼び出しの上限であり、実験設定の総予算
10,080,000ステップや設定ハッシュを変更しない。

```sh
python scripts/cloud_train.py status --stage l2 \
  --config configs/paper-eight-l1.json \
  --output-root outputs/l2-prototype-20261109-001 --seeds 401

python scripts/cloud_train.py stop --stage l2 \
  --config configs/paper-eight-l1.json \
  --output-root outputs/l2-prototype-20261109-001 --seeds 401

python scripts/cloud_train.py resume --stage l2 \
  --config configs/paper-eight-l1.json \
  --output-root outputs/l2-prototype-20261109-001 --seeds 401 --memory-gib 3
```

保持したコンテナの再開では、元の起動条件（10更新での停止）も再利用する。
学習を拡張する場合も勝手に出力やモデルを上書きせず、計画を決めてから行う。
クラウド自体の停止や削除を越える常時実行・ファイル保持は保証しない。

## 設定と評価の注意

現在のローダーは、L1チェックポイントとL2に同じ設定ハッシュを要求する。
そのため既存の `paper-eight-l1.json` を共有し、学習率0.001、batch 72,000、
epochs 4、minibatches 18、BPTT 20等を変更していない。
L2だけ学習率やネットワーク等を変更する実験には、L1の学習設定とL2の設定を
明示的に分離する追加対応が必要。ハッシュを無視したりモデルのmanifestを
書き換えて通すことはしない。

今回の8体はすべてconstructionで、validation用seed 201とtest用301〜305は
まだこの出力先に配置していない。construction相手の成績を未知相手への性能と
呼ばない。今後は別に用意したvalidation/test L1で、相手別・エピソード位置別・
5エピソードすべて成功する割合を評価する。

L1自体にはレシピごとの弱点がある。L2がうまく協調できない場合も、L1の能力や
役割に起因する失敗とL2の適応の失敗を分けて分析する。
