# NestRL 再現仕様

この再現は、論文 `arXiv:2602.17737v2` とリポジトリ直下の
[`nestRL.pdf`](../nestRL.pdf) を出発点に新規実装する。過去の NestRL 再現のコード、
チェックポイント、結果は参照しない。論文の数値結果を得たことと、実装が動くことは
別々に検証する。

根拠と未記載事項の一覧は [source_audit.md](source_audit.md) を参照する。

## 1. 再現する学習構造

1. **L0**: 特定の一つのレシピに従う固定方策。パートナーの内部状態を推定しない。
2. **L1**: 複数の固定 L0 方策に対して学習する、履歴を持つ PPO 方策。
   学習後の重みを凍結して L2 用パートナーにする。
3. **L2 / NestRL**: 複数の凍結済み L1 方策を相手に学習する、履歴を持つ PPO 方策。
   L2 の学習に伴って L1 の重みを更新しない。
4. **Generalist 比較条件**: 同じ学習器を L0 のみに対して学習する。
   L2 との差は adaptive L1 に学習中に接するかどうかである。

L1 は重みが凍結されていても、相互作用履歴から recurrent state が変化するため
行動を適応させられる。重みの更新と、実行時の適応を区別する。
L1 の待機、L2 の先導、レシピの切り替えは手続きとして追加しない。論文の
Appendix C は、これらを学習の結果として調べている。

## 2. 環境

| 項目 | 再現条件 | 根拠 |
| --- | --- | --- |
| 人数 | 2 人、完全協力、共有報酬 | Appendix B.1 |
| 料理 | Lettuce + Onion / Tomato + Carrot / Potato + Broccoli | 本文 §4、図 1、Appendix D |
| 食材へのアクセス | 各側で一方の食材のみ扱え、両者の貢献が必要 | 本文図 1、Appendix B.1 |
| 手順 | 入手、まな板へ置く、切る、回収、共有皿へ載せる、完成皿の配達 | Appendix B.1 |
| 配達 | 右側のエージェントが担当 | Appendix B.1 |
| 観測 | 1 人あたり 105 slots、部分観測 (下記の再現差分あり) | Appendix B.1 |
| 行動 | 上、下、左、右、interact、no-op の 6 種 | Appendix B.1 |
| interaction reward | 同じ物体への繰り返しを除き 0.5 | Appendix B.1 |
| progress reward | レシピの進行に対し 1.0 | Appendix B.1 |
| completion reward | 有効な料理の配達に対し 10.0 | Appendix B.1 |

Appendix B.2 にある最初の二つのレシピの組み合わせは、ユーザーの明示指示により
採用しない。本文の 3 レシピで統一する。

105 次元という長さだけで著者の観測と一致したとは扱わない。公開 PACE の既定の
環境と、図 1 の 6 食材・2 まな板の配置は異なる。図 1 の配置に PACE の特徴式を
適用すると部分観測は有効 93 次元となるため、残り 12 slots は予約領域として常に
ゼロにする。予約領域を追加の privileged information や partner ID に使用しない。
この 105-slot adapter は再現側で定義したもので、著者の全 index との一致は未検証。

環境は固定した公式 PACE の World / Recipe を再利用し、NestRL 用の配置、観測、
報酬 adapter は `src/nestrl_research` 側に置く。PACE 自体は変更しない。
視野は同じ側と中央カウンターまでとする。反対側の不可視情報は mask し、
非可視の partner state を直接 learner に渡さない。
step cost は 0、horizon は 40 steps とする。視野、step cost、シミュレーション
horizon は再現側の選択であり、論文から完全に確定できた設定ではない。

### 図 1 から読み取れる配置

図 1 (PDF p.2) は 7 列 x 8 行に見える。0 始まりの `(x, y)` で、
左食材は tomato `(0,1)`、potato `(0,2)`、lettuce `(0,3)`。
左まな板は `(0,4)`、共有皿は `(3,3)`。
右配達口は `(6,3)`、右食材は onion `(6,4)`、carrot `(6,5)`、broccoli `(6,6)`。
右まな板は `(5,7)`。中央の `x=3` はカウンターで、床は左 `x=1,2`、
右 `x=4,5`、`y=1..6` に分離される。

これは図の目視による読み取りであり、本文に数値として記載された座標ではない。
図との違いが残る場合は、その違いを環境の設定と監査記録に残す。

## 3. 履歴と recurrent policy

本文 §3.2 / Appendix B.3 は、履歴 `(observation, action)` を RNN encoder に通し、
潜在状態と現在の観測から行動を決める構造を指定している。latent は partner ID や
相手の内部 belief を直接入力するものではない。

Appendix F で確認できる共通ネットワーク条件は以下の通り。

- 前段の hidden dimensions は `[128, 128]`、activation は ReLU。
- GRU は単層、hidden size は 128。
- actor と critic は GRU とその前段を共有する。
- BPTT の区切りは 20 steps。
- encoder と policy は PPO により end-to-end で更新する。

観測だけを recurrent core に通すか、前行動も明示入力するか、latent を後段で
どのように現在観測と合わせるかまでは論文に完全な定義がない。実装した入力と
テンソル形状はコード・設定・テストを正とし、著者実装を確認したという扱いにしない。

実装では前行動の 6-way one-hot と episode 開始 flag を現在観測に付加する。
これらの入力から `[128,128]` の共有 ReLU MLP と単層 GRU128 を通し、actor と
critic を出力する。episode 開始時の前行動は 6 要素すべてゼロとする。

同じパートナーとの round 内適応を調べるには、物理環境の episode reset と
履歴の round reset を区別する必要がある。採用する再現条件は、同一 round の
5 episodes でパートナーを固定し、両方の学習済み方策の履歴を保持し、round 境界で
履歴を消すこととする。PPO の GAE 境界は episode 終端とし、履歴を保持しても
次の episode の報酬を前の episode に無条件で加算しない。
この reset 方針は論文の round 内適応という目的から選ぶ再現上の条件である。

## 4. PPO 条件

Appendix F / Table 3 の **Level-n Training** の行を使用する。

| パラメータ | 値 |
| --- | ---: |
| learning rate | 0.001 |
| PPO clip epsilon | 0.2 |
| entropy coefficient | 0.03 |
| discount gamma | 0.99 |
| GAE lambda | 0.95 |
| rollout batch size | 72,000 |
| PPO epochs / rollout | 4 |
| minibatches / epoch | 18 |
| global gradient norm clip | 15.0 |
| activation | ReLU |
| hidden dimensions | [128, 128] |
| GRU hidden size / layers | 128 / 1 |
| truncated BPTT length | 20 |

NestRL 自体への partner-ID 分類損失や探索 bonus は本文 §3.2 / Appendix B.3 / F に
指定されていないため、デフォルトで追加しない。PACE の補助損失係数 1.0 を
NestRL に流用しない。PACE、LIAM、LILI の補助損失は、それぞれの比較実装として
明示的に追加・検証する必要がある。

現在の比較実装は Generalist のみである。PACE、LIAM、LIAMX、LILI の学習器・
補助損失・探索 schedule は未実装であり、それらの比較結果を再現したとは扱わない。

optimizer の細部、value loss 係数、総 training steps、seed 値と本数、L1 pool
サイズ、early stopping、checkpoint selection は論文では確定できない。
実際の値は実験開始前に設定で固定し、論文記載値と再現者選択を混同しない。
smoke 用の小さい batch と短い予算は動作確認用であり、Table 3 の学習条件による
性能実験とは別の run にする。

### この再現で固定するプロファイル

設定の実体は `configs` と `src/nestrl_research/config.py` に置き、実行時には設定の
SHA-256 を保存する。`paper` という名称は、記載された PPO 値を使うプロファイル名で
あり、論文の未記載事項まで同じことや、結果の再現を意味しない。

| 条件 | paper | smoke |
| --- | --- | --- |
| 並列環境数 | 360 | 2 |
| rollout steps / environment | 200 | 40 |
| batch | 72,000 | 80 |
| 総 training steps / learner | 10,080,000 (140 updates) | 160 (2 updates) |
| epochs / minibatches | 4 / 18 | 1 / 2 |
| construction L1 seeds | 101, 102, 103 | 11, 12 |
| validation L1 seeds | 201 | 21 |
| test L1 seeds | 301, 302, 303, 304, 305 | 31, 32 |
| evaluation rounds / partner | 10 | 2 |
| episodes / round | 5 | 5 |

総予算 10,080,000、並列数、seed 分割は再現者が固定した条件である。
optimizer は Adam、epsilon は `1e-5`、value loss coefficient は `0.5`、
value clipping は有効とする。これらは Table 3 の明記項目ではない再現選択である。
L1 は右側 (side 1) とし L0 左側 (side 0) に対して学習する。
L2 と Generalist は左側に置き、それぞれ L1 右側、L0 右側に対して学習する。
L0 を両方の役割で使うことと、この階層ごとの役割交代は、論文に完全な配置表が
ないために加えた再現条件である。

## 5. パートナーとデータ分割

論文 §4 は、L2 に未知の L1 を別の random seed で生成することを指定している。
同じ checkpoint に別の名前を付けたものや、construction checkpoint の別時点を
独立の held-out partner として扱わない。

再現では `construction`、`validation`、`test` を事前固定の registry 分割とする。
この分割名称と具体的な seed は再現側で追加する実験管理条件である。

- training の相手抽出は construction の凍結済み方策に限定する。
- validation はモデルの選択・診断に使用できるが、学習相手に混ぜない。
- test は最後の評価用に固定し、評価値を見て seed、予算、停止条件を選ばない。
- L1 の parameter hash、生成 seed、元設定 hash と split を保存する。
- round 中の相手を固定し、agent ごとの重み・履歴を混線させない。

## 6. 評価

本文 §5.1 の明示条件である **10 rounds x 5 episodes / partner** を採用する。
Table 1 の adaptive partners は P1-P5 の 5 個である。したがってこの条件では
1 partner あたり 50 episodes、5 partners 全体で 250 episodes となる。
paper profile は 5 個の test L1 と 10 rounds x 5 episodes を固定し、
Table 1 に明示された partner 数と round 数を用いる。
別の数に減らした smoke 評価は、結果にその旨を明示する。

共有の shaped return と成功率は別に保存する。成功は有効な料理を制限時間内に
delivery square に届けたこととし、途中の interaction / progress reward だけでは
成功にしない。

保存する主な集計は以下の通り。

- partner 別の episode 成功率、分子・分母、Wilson 95% 区間。
- partner 間の macro mean と CV (standard deviation / mean)。
- round 内 episode 位置 1-5 ごとの成功率。
- 完成レシピ別の成功件数 (3 レシピすべて、ゼロ件を含む) と、成功 episode の
  完成レシピ分布に対する entropy (nats)、`log(3)` で正規化した entropy。
  成功ゼロ件では両 entropy を `null` とし、未定義値を 0 と混同しない。
- seed、run ID、環境条件を含む設定と registry / checkpoint の hash。

本文、表、Appendix C には評価分母や leading rate の不一致がある。
公表値を合格閾値としてコードに埋め込まず、実際の分子・分母を出力する。
人を対象とした user study とその統計値は自動評価では再現できない。

## 7. 実行と互換性

開発・smoke・診断には Mac、長い学習には Windows + CUDA を基本とする。
バックエンドを変更して科学実験を継続する場合は、新しい run ID を付ける。
1 つの run directory を 2 台から同時に書き込まない。

checkpoint は重み、manifest、設定 (環境条件を含む)・registry の hash を組として扱う。
旧再現の checkpoint は互換性が検証できないため、この新規実装へ読み込まない。
新実装内でも観測、action、報酬、layout、履歴 reset の変更は互換性に関わる変更として
記録する。短い smoke 実行の成功だけで、論文結果の再現完了とは報告しない。
