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
