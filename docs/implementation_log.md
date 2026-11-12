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

## 2026-11-12 — L1 random-start, 400 updates, twenty fresh seeds

- User authorized ten new Codex Cloud tasks, two L1 models per task, each trained
  for 400 PPO updates with random episode starts. Added a separate config and
  task/command/seed/storage plan; no historical scientific config or run changed.
- Config `configs/l1-random-start-400u-20261112.json`: 360 environments × 200
  rollout steps = 72,000 steps/update, 28,800,000 steps/model, 576,000,000 total
  steps for twenty models. Keep paper PPO/reward/recipe/history conditions;
  learner right, frozen scripted L0 left. Final-budget checkpoint only, no early
  stopping, performance-based selection, L2 training or post-training evaluation.
- New construction seeds 1001–1020, paired consecutively across tasks t01–t10.
  Audited existing configs, historical registry, local run/config records,
  documented seed499 prototype and Draft Release names/assets; no overlap found.
  Reserved future validation2001, test3001–3005, learner4001 and evaluation5001
  without starting those stages. Existing immutable partitions remain unchanged.
- Each new `re-NestRL-repro-base` VM runs two CPU models with one thread/CPU
  and a 6GiB memory limit per model. Record actual cgroup quota, affinity and
  memory limits. Use unique task output roots and canonical L1 internal run IDs.
  Workers use pinned clean source, record commands/config/source/runtime before
  launch and never write the coordinator's source branch or another run directory.
- Storage: a new Draft Release tagged `l1-random-start-400u-20261112` in
  `kei-student/re-NestRL`, unique task/seed assets, including model manifests,
  latest pointers, config/source/runtime, logs and supported CPU recovery state.
  Upload/download hashes, final budget and strict model loads must be verified
  separately from computation completion.
- Compatibility: fresh random-start runs only; no historical checkpoint imports
  or continuation under a changed budget/backend. No environment/model source
  change was needed for this campaign. Added a contract test for the exact
  budget, all twenty seeds, two-model assignments and provenance command paths.
- Canonical config SHA-256:
  `303dadcb6c6096748b68928019fa44f33733079d71f11c63e4688413245834ca`.
- Config file SHA-256:
  `0cc01070431cb31bb893562a9e1f7aa9cccdc0097e16cceceeda6c4859402f60`.
- Local validation: activated `.venv` workspace check with `--device auto
  --allow-dirty` passed on CPU; all six pinned PACE files verified; unittest
  discovery passed (76 tests, two opt-in Docker tests skipped); diff check passed.

## 2026-11-12 — verify one Cloud task before launching the rest

- Adopted the user's requested launch sequence in AGENTS.md and cloud_base.md:
  start one requested task, verify setup/resource suitability, each assigned
  model's saved training update and startup artifact upload/download hashes,
  resolve problems, then launch the remaining tasks with the working procedure.
- The first task counts toward the total (ten tasks means one plus nine), with
  unchanged scientific conditions and full budgets. Verification does not require
  completing the first model's full budget or another routine user approval.
- Distinguish effective VM limits from unenforced Docker flags; judge actual
  resource suitability without changing or bypassing platform limits.
- Documentation/workflow only. No source, config, seeds, model format or
  checkpoint compatibility change. This does not authorize resuming the twenty
  stopped models; the user's stop remains in effect. No Cloud task was started.

## 2026-11-12 — launch one fresh random-start L1 pilot task

- User explicitly requested one new Cloud task and a status report, following
  the single-task launch gate. Start only seeds1021/1022, two models in one new
  re-NestRL-repro-base VM, each for the full400 updates/28,800,000 CPU steps.
  The previous twenty runs remain stopped; no checkpoint transfer or resume.
- Config `l1-random-start-400u-pilot-20261112.json` preserves all environment,
  PPO, roles, history and budget choices. Construction1021–1040 are reserved in
  a separate immutable registry so expansion can retain a common config hash;
  seeds1023–1040 are not authorized to start in this request. Existing recorded
  construction seeds through1020 do not overlap the new range.
- Use one Torch/BLAS thread per model. Judge effective VM cgroup limits and
  observed memory/CPU suitability; rootless Docker ignoring individual resource
  flags alone is not a blocker. Do not modify or bypass platform limits.
- Gate: preflight, both models progressing through saved updates, pointer/hash
  consistency and verified startup artifact upload/download. Report actual
  progress, throughput, resources and issues before any additional task launch.
- New run namespace and Draft Release tag:
  `l1-random-start-400u-pilot-20261112`; asset names are task/seed specific.
  Historical records and checkpoints are immutable; scientific checkpoint
  compatibility is unchanged, but this pilot uses fresh initialization only.
- Canonical config SHA-256:
  `fd5c44797dc80917d390eedb5007997e1a8f2d5495affd07616cb08275818454`.
- Config file SHA-256:
  `163e4e10b78cfa2e1e82b7366482665e2a800ee5b3f1b7c5381447457975d815`.

## 2026-11-13 — evaluate twenty random-start L1 policies against every L0

- User requested performance evaluation of seeds1021–1040 and the count whose
  success rate is at least95% against every L0. Fix the final400-update models;
  no weight updates, tuning, checkpoint selection or additional training.
- Evaluate all three frozen left-side recipe experts against the right-side L1:
  LettuceOnionSalad, TomatoCarrotSalad, PotatoBroccoliSalad. Each partner gets
  200 rounds ×5 episodes =1000 episodes/model, 3000/model, 60,000 in total.
  Stochastic policy actions; evaluation seed6001; random starts and the original
  five-episode recurrent-context semantics,40-step horizon and rewards.
- Pass criterion is all three per-partner counts >=950/1000, rather than an
  average across partners. Report empirical rates and all twenty seeds; this is
  fixed training-partner performance, not unseen-partner generalization or a
  statistical guarantee that the true success probability exceeds95%.
- Added independent evaluation round/seed arguments to preserve the original
  training config hash during strict checkpoint loading. Protocols record actual
  evaluation values; tests verify episode seed schedules, all L0 identities,
  unchanged training config, no updates and invalid-argument rejection.
- Download final bundles from Draft442708737 with transfer-manifest and archive
  SHA checks, retain final.pt/final.json/latest.json/config/run/complete records,
  verify complete400 updates/28,800,000steps and strict CPU load before inference.
  Record commands/config/source/runtime and checkpoint hashes before execution.
- Compatibility: no environment/model/training-config change; historical random-
  start checkpoints retain strict inference compatibility. Exact training resume
  across this changed evaluation implementation is not claimed or performed.
- Training config SHA-256 remains
  `fd5c44797dc80917d390eedb5007997e1a8f2d5495affd07616cb08275818454`.
- Executed all60,000 episodes on CPU, four independent workers with one thread
  each, using evaluation commit `a5795fc00f33cfaa8d23297b9908a280e993243e`.
  All20 execution records confirm unchanged source/config and successful exit.
  Recounted every episode/partner/seed schedule and verified all output hashes;
  all20 original checkpoint SHA-256 values remain unchanged.
- Result:16/20 policies (80%) pass every L0. Passing seeds:
  1021,1023,1024,1027–1033,1035–1040. Failed seeds1022,1025,1026,1034.
  Lowest per-L0 success among passing policies is99.5%. Failures:1022 has2.9%
  onTC;1025 has0% onTC;1026 has0% onLO;1034 has71.7% onTC.
- Pre-commit workspace check (dirty) and clean pre-execution workspace check
  passed;80 unit tests completed successfully (two Docker opt-in tests skipped).
- Evaluation plan canonical SHA-256:
  `14fe7c50bc7f75845c65f7feaa8d158118b8b83d41c27ad256d3df791d3b8395`.
- Evaluation plan file SHA-256:
  `ecc6de4e0540cb976aac510d1f4ff196e72ff53a7578beb728303a31e4818ef4`.
- Stored evaluation bundle in the original Draft442708737, seven parts,
  transfer manifest asset741434439, manifest SHA-256
  `2cf963a4b60d819327e2c361bf0162d5cb95d7a6fba57b082ed2bedca13f1ebe`.
  Archive SHA-256:
  `3fce77050e82502c7634dd6a408b22a09a5b9285272fe2f624e43ce86ea7c5d5`.
  Upload read-back, fresh re-download and all990 member hashes verified.
  Full twenty-model table and provenance: l1_random_start_l0_evaluation_20261113.md.
