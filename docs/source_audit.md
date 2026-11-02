# 一次資料の監査

監査日: 2026-11-02 (Asia/Tokyo)。既存の再現実装は参照していない。

## 資料と同一性

| 資料 | 確認内容 |
| --- | --- |
| [arXiv HTML v2](https://arxiv.org/html/2602.17737v2) | 本文、Appendix B、C、D、F を確認。v2 は 2026-06-01 付。 |
| [`nestRL.pdf`](../nestRL.pdf) | 27 ページ。PDF metadata の arXivID は `2602.17737v2`。本文と Appendix B/F の主要条件が HTML と整合。 |
| PDF SHA-256 | `c81b9eec95d79929b2516d18ecf0c2ec95cb9ee3deff1837a8be51f187819a7b` |
| [匿名の著者コード](https://anonymous.4open.science/r/anon-5BD5/README.md) | 論文 Appendix B の脚注に掲載。Web ツールで取得できず、別途 HTTP 410 を確認。著者コードを検証できたという扱いにはしない。 |
| [公式 PACE](https://github.com/bigai-ai/PACE) | README / GitHub の説明が ICML 2024 PACE の公式実装と明記。環境の一次コードとして参照可能。 |
| PACE 固定 revision | `f008c19aa743ec209550375c9ee9c25379ac8b24`。`external/PACE` は変更しない。 |

PDF はテキスト抽出だけでなく p.2 の図 1 と p.20 の Appendix F / Table 3 を画像で
目視した。以下のページ番号は PDF の表示ページ番号である。

## 論文で確認できる仕様

| 仕様 | 根拠 | 備考 |
| --- | --- | --- |
| L1 は固定 L0 集合に対して学習 | §3.2, pp.4-5 | L0 は異なる convention を表す。 |
| L2 は学習済み L1 集合に対して学習 | §3.2, pp.4-5 | 同時に双方の重みを更新する IPPO とは異なる。 |
| L1 の未見 seed に対する評価 | §4, p.6 | seed の具体値と学習 pool サイズは未記載。 |
| 観測 105 次元 | Appendix B.1, p.14 | position、orientation、held/front objects、visibility flags のカテゴリを列挙。 |
| 6 行動 | Appendix B.1, p.14 | 上下左右、interact、no-op。整数 ID の対応は未記載。 |
| 共有の shaped reward | Appendix B.1, p.15 | interaction 0.5、progress 1.0、delivery 10。 |
| history encoder と policy の PPO 学習 | §3.2, p.6 / Appendix B.3, pp.15-16 | explicit recursive Bayes update を実行する実装ではない。 |
| GRU128、単層、BPTT20 | Appendix F, p.20 | actor と critic が GRU と前段を共有。 |
| PPO hyperparameters | Appendix F / Table 3, p.20 | Level-n Training の行を参照。 |
| 10 rounds x 5 episodes / partner | §5.1, p.7 | 表の数値の分母とは不整合がある。 |
| 5 個の held-out adaptive partner | Table 1, p.7 | P1-P5。 |
| waiting / initiative は創発的行動 | §5.3 / Appendix C | 固定の待機ルールを L1/L2 に加えない。 |

## レシピの矛盾と採用方針

本文 §4 (p.6)、図 1 (p.2)、人実験の説明 (Appendix D, p.23) は、
`LettuceOnionSalad`、`TomatoCarrotSalad`、`PotatoBroccoliSalad` を指定する。
Appendix B.2 (p.15) は最初の二つを Tomato + Onion / Carrot + Lettuce と記載する。
PDF と HTML の両方でこの不一致があり、HTML の変換だけが原因とは考えられない。

ユーザーが Appendix の組み合わせを無視してよいと明示したため、本文の LO / TC / PB
を採用する。これは黙って誤記を修正したものでも、著者コードを確認して決めたものでも
なく、この再現で明示的に確定した条件である。

## 詳細が未記載の条件

以下を論文記載の数値と断定しない。採用値は再現側の設定に固定する。

| 未記載項目 | 影響と処理 |
| --- | --- |
| 105 次元の全 index、正規化、不可視時の値 | PACE の環境実装を参照し、採用した index と shape を検証する。 |
| 正確な視野形状・遮蔽物処理 | visibility flags があることだけでは視野角を確定できない。 |
| chopping 回数、interaction の詳細、料理/皿の状態遷移 | 報酬の大きさだけでは dynamics を確定できない。 |
| simulation の horizon | Appendix D は人実験を 40-step と記載する一方、UI は 40 秒、backend は 10 FPS と記載する。シミュレーションの値をそこから無条件に推定しない。 |
| learner / partner の左右配置の全組合せ | B.2 の L0 は右側。L1/L2 の全訓練・評価配置は明確な一覧がない。 |
| history の入力と reset 境界 | `(o,a)` の RNN と round 内適応は説明するが、完全な API 定義はない。 |
| latent と現在観測の結合場所 | 数式の条件付けはあるが、完全な network graph はない。 |
| 総学習 steps、環境並列数、seed 値、本数 | 論文だけで元実験の training budget を確定できない。 |
| L0/L1 pool サイズ、L1 の各訓練部分集合 | §3.2 は集合・部分集合を説明するだけ。 |
| optimizer epsilon、value coefficient、advantage normalization | Table 3 の項目外。PPO の通常値を論文値と呼ばない。 |
| early stopping と checkpoint 選択規則 | validation 用条件を事前に設定する必要がある。 |
| construction / validation / test の seed registry | 名称と実装は再現側で追加する実験管理。 |
| PACE exploration bonus の全 schedule | Appendix F は補助損失係数等の一部のみ。 |

### 105 次元と配置の整合性

公開 PACE の `get_obs_size` は、環境中の物体数と食材数から次元を計算する。
その既定の部分観測環境は、論文図 1 と異なる 7 x 13 の配置と食材数を持つ。
図 1 の 7 x 8・6 食材・2 まな板で同じ式を使うと有効特徴は 93 次元となり、
論文の 105 次元にはならない。論文は全 index を公開しておらず、匿名の著者コードも
利用できないため、差の 12 次元の内容を確定できない。

この再現では図 1 の配置を優先し、有効特徴 93 + 常にゼロの予約領域 12 にする。
これは独自の adapter 定義であり、著者の observation encoding を完全復元した
ことを意味しない。添付の 105-slot 仕様と hash により、この差分を追跡する。
視野は同じ側と中央カウンター、step cost は 0、horizon は 40 として固定する。
これらも論文記載そのものと区別する。

## PPO と encoder 損失の読み分け

Appendix B.3 は NestRL の encoder と policy の end-to-end PPO を述べている。
Appendix F では PACE に auxiliary loss 1.0、LIAM に observation/action prediction
各 1.0、LILI に reward/next-observation prediction 各 1.0 を指定する。
これらを NestRL 自身への必須補助損失と読む根拠はない。

Generalist、PACE、LIAM、LILI は L0 のみで学習するという §4 の条件と、
NestRL が凍結 L1 で学習する条件を分離する。補助損失のない L0 学習器を
「PACE 再現済み」と表示しない。

## 数値結果の内部不整合

これらは元資料の問題であり、再現の結果を都合のよい分母に合わせる理由にはしない。

1. **評価 episode 数**: §5.1 / Table 1 caption は 10 x 5 episodes とする。一方、
   同節の統計説明は 5 partners 合計 75 episodes / arm とする。Table 1 の
   0.67, 0.80, 0.93, 0.87, 0.80 と Wilson 区間は各 partner 15 episodes、計 75
   と整合する。明示した 10 x 5 を優先し、実際の分子・分母を出す。
2. **L1 waiting の N**: Appendix C.1 (p.16) は N=1,100 としながら、後段で
   waiting n=2,474 と non-waiting n=81 を報告する。同じ母集団とは解釈できない。
3. **L2 leading rate**: 本文 §5.3 / Appendix C summary は 88%、Appendix C.2 の
   結果段落は 68% を報告する。本文内の別の比較段落も 68% を用いる。
4. **self-play 個数**: §5.2 は five seeds とする一方、図 3 は 9 policies の
   cross-play matrix を表示する。別実験が混在している可能性を排除できない。
5. **人実験**: N=25 とするが統計の自由度が t(21) であり、除外規則が十分に
   説明されない。Appendix E の「7-point」表で effort=7.77 も記載される。
6. **人実験の質問値**: Appendix E の表と後段の本文で PACE の temporal demand、
   proactiveness、adaptiveness の平均値に不一致がある。

自動試験は deterministic dynamics、情報漏洩防止、履歴境界、PPO の更新、
checkpoint の hash、独立な held-out 評価を検証する。
81.3% 等の公表成功率を、動作確認の合否条件として流用しない。

## 再現の到達度

新規実装と smoke 実行で検証できるのは、採用仕様の実装と短い学習経路である。
凍結した実験設定で複数 seed の十分な training と held-out 評価を実施するまでは、
論文の定量結果を再現したとは扱わない。
人実験の再現には別途参加者、倫理手続き、評価設計、原データが必要であり、
本リポジトリでのシミュレーション検証からは独立である。
