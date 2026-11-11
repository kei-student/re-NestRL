# Cloud再構成・記録・保存の基盤

この手順はPython環境、GitHub認証、記録形式、保存/復元を扱う。実験の手法・相手・
seed・更新数・学習予算・評価方法は決定しない。A/Cは仮環境である。現時点の
A=環境整備、C=実験実行、Mac=研究/分析、Windows=指示管理という分担は運用例であり、
永久的な制約ではない。本実験のマシンや科学的設定は別途選べる。

## GitHubから新しいcheckoutを作る

対象は `https://github.com/kei-student/re-NestRL.git` のみ。検証する完全なcommit SHAを
環境設定へ保存し、可変branch名だけを実験のコード識別子にしない。既存のdirty checkoutや
実行中runをreset/checkoutして環境更新せず、別のcheckoutを使う。

```sh
git clone https://github.com/kei-student/re-NestRL.git re-NestRL-env
cd re-NestRL-env
git checkout --detach "$PINNED_COMMIT"
bash scripts/setup_cpu.sh
source .venv/bin/activate
python -m unittest discover -s tests -q
bash scripts/start_checks.sh
```

`PINNED_COMMIT`はその環境へ引き継ぐ完全SHA。setupはPython3.12の専用`.venv`を作り、
元pyprojectの依存宣言と `requirements/constraints-cpu-py312.txt` に沿ってCPU PyTorchを
導入する。NestRLを同一venvへinstallしない。今回検証したPython patchは3.12.14、
PyTorch2.15.0+cpu、NumPy2.5.4。Python3.12の別patchをsetupが受け付けても、旧runの
厳密resumeを保証しない。GPU/MPS用依存環境はこのCPU constraintsの検証範囲外。

PACEは `f008c19aa743ec209550375c9ee9c25379ac8b24` に固定し、bootstrapがarchive SHAと
取り込む6ソースを検証する。既存の異なるPACEを上書きしない。setup/startには
train/pipeline/resumeを含めていない。startはclean checkoutを要求する。

## 歴史的snapshotと実行時コードの区別

元AのHEADは `50bd1b2a82345a469cdd336b8f8b3d409e431036`、dirty13パスを含む
11/9と11/11の33ファイルmanifestは
`7822542cbcaf68eafbac1bf72fbcc16ad2ad0b9dfcb6235f27568865b59c5c2e`。
そのソース一式をbytesを変えずに復元したcommitは
`e8433ffb2332e2b964e0d026d3d139b4ccbdf4ad`。これは実行時HEADを書き換える値ではない。
`provenance/original-A-20261111.json`に元HEAD、dirty paths、全source hash、archive hash、
復元commitを別欄で保持する。その後の基盤commitは依存宣言や運用ファイルを変えるが、
既存src実装・実験config・起動スクリプトのbytesを維持する。

歴史的実行部分だけを比較する場合:

```sh
python scripts/source_manifest.py --check provenance/original-A-20261111.json
```

この検査は過去実装の選択的検査であり、将来の研究実装へ適用する固定ルールではない。
`source_manifest.py`だけなら現在のGit管理分+未追跡sourceの一覧とcanonical hashを出す。
canonical形式は `json.dumps(mapping, sort_keys=True, separators=(',', ':'), ensure_ascii=True)`
をUTF-8化したbytes（末尾改行なし）。ファイルhashは改行正規化なしのbytes SHA-256。

L1 Release433508587、L2 Release438779737、評価Release438811508との対応は
`provenance/legacy-artifact-map.json`に記録。14個のcore sourceを、checksummed source
archiveとcheckpoint manifest/indexで照合した。これらに実行時Git HEADが記録されていない
場合はnull/不明のままにする。source indexの旧commit、dirty source hash、archive、
復元commitを混同しない。評価harnessは既存評価archive内の`evaluate_all.py`とそのSHAで
保存されており、新しい運用scriptへ置き換えない。

## 実験ごとの記録と明示的な起動

実験担当がconfigとargvを指定する。`scripts/run_recorded.py`は新しいinvocation directoryへ
コードHEAD・dirty source hash・依存runtime・config bytes/実効defaults・完全argvを保存し、
その後に指定したcommandだけを実行する。科学的条件の既定値は追加しない。
現在coreの`-m nestrl_research`にはconfig defaults展開を記録する。他utilityの場合は
宣言JSONとargvを記録する。子プロセスで実際に適用されたruntime/configの正本は
runが作る`config.json` / `run.json` / recovery metadataであり、これらも一緒に保存する。

下の変数は、その実験の指示から設定する。設定を決めていない状態で実行しない。
既存coreで未対応の手法/観測等を変える場合は別途実装・検証したSHAを使う。
本基盤は科学的条件の変更ごとに追加の人間承認フローを設けない。

```sh
python scripts/run_recorded.py \
  --config "$EXPERIMENT_CONFIG" --record-dir "$INVOCATION_DIR" \
  --metadata "$EXPERIMENT_MANIFEST" -- \
  .venv/bin/python -m nestrl_research train \
    --config "$EXPERIMENT_CONFIG" --output-root "$OUTPUT_ROOT" \
    --stage "$STAGE" --split "$SPLIT" --seed "$SEED" \
    --device "$DEVICE" --threads "$THREADS"
```

`--record-only`は意図の記録だけでcommandを実行しない。`--max-updates`等は必要な実験の
argvに追加する。再開は別のinvocation directoryを作り、条件を変えずに明示的に`--resume`
を付ける。新しい条件には新runを作る。Docker launcherも明示的commandとして記録できるが、
launcherの終了は学習jobの完了を意味しない。workerのrun記録・logs・完了markerで確認する。
credentialsはargv/metadataへ入れず、既存のSecret bindingを使う。

## 標準ライブラリだけの保存utility

`scripts/release_artifacts.py`はre-NestRLやtorchをimportしない独立utilityであり、
repo、Release ID、対象ファイルを引数で受け取る。Mac側の別repoでも同じファイルを使える。
install/startとは別に、実験担当が保存を指示する。学習開始機能はない。

停止したrunを選び、モデル`final.pt/final.json/latest.json`、config、source/runtime記録、
metrics/学習log、resume stateを対象として明示する。registryと相手モデルのhash参照、
invocation記録も含める。完全な相手weightsが必要なら、それも別途指定/参照する。
`.writer.lock`があるrunのbundleは拒否する。停止を示すmarkerだけに依存せず、job状態も確認する。

```sh
python scripts/release_artifacts.py bundle \
  --root "$EXPERIMENT_ROOT" --include "$RUN_DIRECTORY" \
  --include registry.json --include "$WORKER_LOG" --include "$INVOCATION_DIRECTORY" \
  --output "$BUNDLE_FILE"
```

出力JSONのbundle SHA-256を実験台帳に保存する。ZIP内`bundle-files.json`は全対象ファイルを
列挙し、そのbytesのhashを固定する。Secret/認証ファイル、`.env`、venv、Git内部を含めない。
Secretが混入していないことを対象選定時に確認する。古いZIPはこの新bundle schemaとは異なる。

既存Draftを使う場合はIDを指定する。必要な場合にだけ新しいDraftを作る:

```sh
python scripts/release_artifacts.py create-draft --repo "$GITHUB_REPOSITORY" \
  --tag "$EXPERIMENT_TAG" --target "$PINNED_COMMIT" --title "$RELEASE_TITLE"
python scripts/release_artifacts.py upload --repo "$GITHUB_REPOSITORY" \
  --release-id "$RELEASE_ID" --manifest-name "$UNIQUE_PARTS_MANIFEST" --file "$BUNDLE_FILE"
```

既定は2,500,000bytesごとに分割。各partと元ファイルのsize/hash、順序をparts manifestに保存する。
同名assetはhash一致なら再利用、不一致は上書きせず失敗する。保存/再利用した全assetを
再downloadして照合し、結合後の全体hashも検証する。parts manifest自体のSHAは出力JSONへ
返すので、Release ID・manifest名・SHAを別の台帳へ保存する。検証を省略するオプションはない。
途中失敗しても既存assetを削除せず、同じ入力で再実行できる。Draft以外へのuploadは拒否し、
Publishや既存assetの変更・削除は行わない。

```sh
python scripts/release_artifacts.py download --repo "$GITHUB_REPOSITORY" \
  --release-id "$RELEASE_ID" --manifest-name "$UNIQUE_PARTS_MANIFEST" \
  --manifest-sha256 "$PARTS_MANIFEST_SHA256" --output-dir "$DOWNLOAD_DIRECTORY"
python scripts/release_artifacts.py restore-bundle --file "$DOWNLOADED_BUNDLE" \
  --sha256 "$BUNDLE_SHA256" --output-dir "$NEW_RESTORE_DIRECTORY"
```

Downloadは指定repo/Releaseのmanifestだけを扱い、CDNへAuthorizationを転送しない。
異なる同名ローカルファイルや既存runへ上書きしない。復元後も学習は自動開始しない。
復元したrunにresume可能かはcoreのsource/config/runtime検証で判断する。

## checkpoint互換性とresume

既存core14ファイルを変更しないため、従来の105次元観測/6行動/MLP+GRUモデルの推論形式は
維持される。最終checkpointはmanifest/latest/hash、config/environment hash、strict state dictで
読み込む。旧NestRLや著者モデルとの互換性はない。

CPUの厳密resumeにはrun全体の`resume/latest.json`、指すstate（モデル・optimizer・RNG・環境・
履歴）とmetrics/config/run/registryおよび凍結相手が必要。config/registry/core source/environment/
stage/split/seed/side/run ID/相手hashとPython/Torch/NumPy/device/threads/determinismの照合を通す。
変更をmanifest書換えで回避しない。CPU中断前後のbit一致は同じ検証環境でunit testする。
異なるCPU機種・OS・バックエンドでのbit一致は保証しない。GPU/MPSの厳密resumeは未対応。
L2の既存最終transfer ZIPにはresume状態が含まれないため、最終重みだけで厳密resumeできない。

## Cloud保存設定として残すもの

- 完全なsource commitと配置先、Python3.12専用venv、setup=`bash scripts/setup_cpu.sh`、start=`bash scripts/start_checks.sh`。
- 現在のproxy/CAを保持。制限する場合はgithub.com/api.github.com/uploads.github.com/codeload.github.com/
  release-assets.githubusercontent.com、PyPI/PyTorch、Docker使用時の実際のregistry/CDNが必要。
- Gitは既存のplatform認証を使う。Release用Secretは`RELEASE_GITHUB_TOKEN`、対象repo限定、
  Contents Read and write、allowed domainsはapi.github.comとuploads.github.comだけ。
  Secretはproxy placeholderのまま通常利用し、値を抽出/表示/保存しない。
- 実験条件・割当・学習開始・resumeは保存環境のinstall/startへ書かない。各実験のmanifest/argvへ記録する。

CloudのPublish/Republishと別環境の作成・引き継ぎ検証は、このソース整備タスクでは実施しない。
