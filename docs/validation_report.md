# 検証記録 — 2026-11-02

対象は新規の `re-NestRL` のみ。論文の性能値の再現は未達です。

## 実行した確認

Python 3.12.14 / PyTorch 2.15.0 / NumPy 2.5.4、このフォルダ専用の `.venv`、CPUで実行しました。
この実行環境では MPS と CUDA は利用不可でした。Windows / CUDA の実機検証は未実施です。

```sh
source .venv/bin/activate
python scripts/check_workspace.py --device auto --allow-dirty
python -m unittest discover -s tests -q
python -m nestrl_research pipeline --config configs/smoke.json --output-root outputs/smoke-20261102-verified --device cpu
```

- workspace check: 成功。独立したGit root、専用venv、上流6ソースのhash、設定、モデル実行を確認。
- unit / integration tests: **48件成功**。
- 本物の調理環境で、両側の固定L0がすべての料理を完成・配膳。
- PPOでencoder・GRU・actorのパラメータが更新され、凍結した相手には勾配が流れない。
- rolloutの旧log probability再現、finite horizon GAE、20-step sequence、独立contextを確認。
- registryの不変性、split/roleの整合、checkpoint / manifest / latestの改竄検出。
- 評価時の無勾配、round内の履歴保持、traceの完全性とhash、集計分母を確認。

| L0ペアの完成料理 | 完成までのステップ | 共有return |
| --- | ---: | ---: |
| LettuceOnionSalad | 16 | 16.5 |
| TomatoCarrotSalad | 19 | 16.5 |
| PotatoBroccoliSalad | 17 | 16.5 |

## 短い学習・評価

L1 construction2 / validation1 / test2、L2、Generalistの7方策を各160ステップ学習しました。
1方策につきPPO rollout2回（1回80環境ステップ）です。
checkpoint7セットと評価6セットを生成しました。

| 方法 | validation L1 | test L1 | 固定L0 |
| --- | ---: | ---: | ---: |
| L2 | 0 / 10 episodes | 0 / 20 episodes | 0 / 30 episodes |
| Generalist | 0 / 10 episodes | 0 / 20 episodes | 0 / 30 episodes |

これは短すぎる学習の動作確認であり、方策がタスクを学習した証拠ではありません。
論文の81.3%等との比較には使用しません。0成功なのでrecipe entropyとCVはnullです。

同じseedによる先行smoke `outputs/smoke-20261102` と最終smokeの **全7 checkpointの
SHA-256が一致**しました。環境・実装hashを後から追加したため、JSON manifest自体は異なります。
数値的な更新は同一CPU条件で再現しましたが、異なるバックエンドでの同一性は主張しません。

最終成果物の例:

- `outputs/smoke-20261102-verified/l2-41/checkpoints/final.pt`
- `outputs/smoke-20261102-verified/l2-41/metrics.jsonl`
- `outputs/smoke-20261102-verified/eval-l2-41-l1-test/summary.json`
- `outputs/smoke-20261102-verified/eval-l2-41-l1-test/traces.jsonl`
- `outputs/smoke-20261102-verified/eval-l2-41-l1-test/artifacts.json`

成果物はローカルに保持し、Git管理から除外しています。

## 固定設定とソース

| 項目 | SHA-256 |
| --- | --- |
| paper設定（canonical JSON） | `adb7e5ce6b32308185a2f1595e9c84a0557a4aba0331d3cdd99fe50087ca91a2` |
| smoke設定（canonical JSON） | `451e0b0fc1e5485188ab36dc2050089e735cabd084c42a473226ba2cbae16f28` |
| 環境・peer・上流ソースmanifest | `315e2fd71de5ffbc39ed90a73d8a5c589c7245aae35fe77a5d0774c193de2628` |
| 実装ソース一式 | `18ce5843ee2c780094bc753809c3d8be2a2b284636fcefab2a90154d227dc627` |

初回検証のブランチは `codex/from-paper`。旧NestRLのremoteは引き継いでいません。
その後、ユーザー指定の専用共有先 `https://github.com/kei-student/re-NestRL.git` を設定しました。
環境ソースまたは設定のhashが異なるcheckpointは読み込みを拒否します。
旧実装・著者モデルとのcheckpoint互換性はありません。

## 未完了の再現範囲

- paper設定による長期学習と成功率・相互適応の検証。
- 著者の105観測indexと視野の確定（現行は93特徴＋12予約ゼロ）。
- 著者の正確なpeer pool・初期化・role配置・総学習量との照合。
- PACE / LIAM / LIAMX / LILIの比較実装。
- 論文の人実験、統計検定、行動分析の再現。

詳細は [再現仕様](reproduction_spec.md) と [資料監査](source_audit.md) を参照してください。
