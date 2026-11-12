# Implementation log

## 2026-11-02 — independent reconstruction

- Scope: only `re-NestRL`; no earlier local reproduction or checkpoints were read/imported.
- Sources: arXiv 2602.17737v2 HTML and the user-provided 27-page PDF.
- Reference PDF SHA-256: `c81b9eec95d79929b2516d18ecf0c2ec95cb9ee3deff1837a8be51f187819a7b`.
- Adopted main-text recipes with the user's explicit direction to ignore Appendix B.2 combinations.
- Official PACE cooking world/recipe source pinned to
  `f008c19aa743ec209550375c9ee9c25379ac8b24`, without upstream modifications.
- Dynamics reconstructed from Figure 1: 7x8, separated rooms, two chopping boards,
  shared plate, right-only delivery. Initial positions inferred from the figure.
- Rewards: interaction0.5 / progress1 / delivery10; undocumented step cost set to0.
  Upstream interaction novelty and recipe progress semantics are retained.
- Observation: 93 effective PACE dense-partial features plus12 reserved zeros.
  This is a documented reconstruction and is not verified against the authors'105 indices.
- Visibility reconstructed as own room plus shared counter; its exact author counterpart is unknown.
- Training: shared ReLU MLP128,128 / GRU128 actor-critic, PPO+GAE,
  BPTT20 and the Table3 hyperparameters. Adam eps1e-5, orthogonal initialization,
  value clipping and value coefficient0.5 are implementation choices.
- Histories include own previous action and an episode-start indicator. Reset
  at each5-episode round; GAE terminates at every physical episode's finite40-step horizon.
- Hierarchy: L0-left -> L1-right -> L2-left. Generalist-left vs L0-right.
  The stage-by-stage role assignment is not fully specified in the paper.
- Immutable construction/validation/test seed registry; independent recurrent
  state per environment, frozen lower-level weights, no test partner in L2 updates.
- Final-budget checkpoint only; no validation/test-based early stopping or tuning.
- Run/checkpoint/evaluation manifests include environment and implementation
  source hashes. Loading refuses a different environment source hash.
- New checkpoints are incompatible with the earlier reproduction and author checkpoints.
  Exact training resume is unsupported; cross-backend inference starts a new evaluation.
- Current scope: NestRL and Generalist; PACE/LIAM/LIAMX/LILI and human study remain unimplemented.

The smoke profile exercises the complete execution path and does not establish
successful task learning or the paper's performance claims. See `validation_report.md`
for executed checks and local artifacts.

## 2026-11-11 — reproducible base and per-experiment configuration

Preserved the original A/C dirty source bytes in commit e8433ffb2332e2b964e0d026d3d139b4ccbdf4ad.
Its 33-file manifest matches 7822542cbcaf68eafbac1bf72fbcc16ad2ad0b9dfcb6235f27568865b59c5c2e.
Added Python3.12 runtime constraints, install/start validation, per-command source/config/runtime recording,
and immutable multipart GitHub Draft Release upload/download with read-back verification.
Pinned declared NumPy2.5.4 and PyTorch2.15.0 to the already verified runtime; experiments supply their own
method/partners/seeds/budget/evaluation. Install/start does not launch or resume training.
No learning/environment source, historical scientific config, checkpoint format, or compatibility changed.
Existing C1-C4 runs and controller were not modified. Historical source/model manifests remain unchanged.

Archived the exact existing 112–127 configuration under configs/history; its file and canonical hashes match the original campaign. Existing running jobs, registry and monitoring were untouched.

Cold-install verification found that the CPU wheel index does not contain every
fixed transitive dependency. Install the pinned non-torch packages from PyPI first,
then the exact `+cpu` wheel. Artifact reconstruction atomically refuses a destination
created during download. Learning and checkpoint compatibility are unchanged.
Upload requests accept the JSON asset response while sending binary data;
octet-stream Accept is reserved for downloads.

Published re-NestRL-repro-base and verified a fresh Cloud task at 53d82cd.
The smoke pipeline completed seven models (two updates/160 steps each), their
evaluations, Draft upload/download, all model/manifest/latest checks and frozen
pool restoration. Historical L2 seed401 also passed inference-only strict load.
Using the canonical L1 registry ID in distinct output-root namespaces, a paused
80-step state was downloaded and resumed to 160 steps. Model, optimizer, RNG,
collector state, numeric metrics and episodes matched uninterrupted execution
(excluding time and resume events). Re-downloaded the final bundle and verified
all 191 files plus strict model loading. An earlier invalid-ID instruction was
rejected before updates; its failure record was preserved. No learning source,
scientific config, checkpoint format or compatibility was changed.

## 2026-11-12 — random episode starting positions

- User-requested scientific condition: independently sample each chef uniformly
  from the 12 floors in its own room at every episode reset. Chef 0 stays left,
  chef 1 stays right; initial facing, layout, recipes, rewards, observations,
  horizon, seeds, budgets and round-history rules are unchanged. Repeats are
  allowed (144 joint starts); no upstream PACE source was modified.
- Added explicit `environment.random_start=true` profiles
  `paper-random-start.json` and `smoke-random-start.json`, with new profile/output
  namespaces. Programmatic defaults use random starts. Historical JSON with no
  flag loads as fixed-start; false is omitted from canonical serialization to
  preserve historical config hashes. Existing configs/records remain unchanged.
- Explicit `reset(seed)` restarts the local RNG; `reset()` advances it. Existing
  training/evaluation episode seeds reproduce starts and checkpoint episode replay.
- Tests cover legal floor starts, all candidate cells, role separation, observation
  positions, RNG streams/reseeding, recipe completion, real training-collector
  resets, legacy config hashes and validation. Existing CPU recovery tests now
  exercise the random condition, including partial-episode replay.
- Checkpoint compatibility: tensor shapes and observation/action encoding are
  unchanged, but this is a different initial-state distribution. Historical
  checkpoints are rejected by strict source/environment hashes; they cannot be
  resumed or mixed into the new condition. Use new runs/registries and newly
  trained frozen construction partners. Historical runs use their recorded source.
- No scientific training or historical run/controller was started or modified.
- Validation from the activated repository `.venv` (Python 3.12.14, NumPy 2.5.4,
  PyTorch 2.15.0): `python scripts/check_workspace.py --device auto --allow-dirty`
  passed on CPU with all six pinned PACE sources verified;
  `python -m unittest discover -s tests -q` passed (75 tests, two opt-in Docker
  integration tests skipped); `git diff --check` passed.
- Canonical config SHA-256 (the run/checkpoint contract):
  - `paper-random-start.json`:
    `01b192fd04dadce581d6bc719427412ae14da731c01b9af7827796b97e8f1c90`
  - `smoke-random-start.json`:
    `27154afcf49c953a7948efac8e774ff3118c031848006181c03ea69885158d2c`
