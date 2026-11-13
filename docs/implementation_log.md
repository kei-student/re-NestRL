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

## 2026-11-13: behavior of all sixteen passing random-start L1 policies

- Analyze all48,000 saved evaluation episodes (including failures), without new
  training, selection or changes to the original evaluation. Replayed all
  1,328,971 joint actions through the pinned world adapter and matched rewards,
  events, success, boundaries and episode totals. Verified original artifact
  hashes, unchanged environment-source hashes and all sixteen checkpoint hashes.
- Added descriptive behavioral extraction, shortest pickup-distance comparison,
  visible-cue timing and round-position/partner summaries. Distinguish hidden
  L0 pickup, first L1-visible cue and first L1 raw pickup. Productive chopping
  and turning are not stationary inactivity. Tests cover an explicitly waiting
  peer, proactive peer, chopping, replay corruption and shortest paths from all
  initial right-room poses/orientations.
- Core result:42,139/48,000 episodes prepare >=2 right-side ingredients;
  6618 prepare all3. There are strong policy differences in nearest-first versus
  preferred-ingredient starts. Correct own-side plate contents occur in47,978
  episodes, with47,976 successful deliveries. Nearest-first is not universal.
- Only3 episodes acquire a first ingredient after a visible cue. Two1036 cases
  meet the conservative inactivity screen and share seed267001. The third is a
  failed1032 plate-handling trajectory. One1040 failure never acquires food;
  one1033 failure never receives a visible cue. Do not silently exclude them.
- Independently configured CPU evaluation-only probe delays the two1036 cases'
  L0 starts by0/4/8/12 steps. Both delay0 baselines exactly reproduce original
  actions/rewards/events. Cue indices11/15/19/23 give pickup12/16/20/24 for both
  partners, supporting rare cue-contingent waiting. PB still first picksCarrot,
  so this does not establish a wait-then-correct-recipe strategy. The probe is
  selected-case mechanism evidence, not a population success estimate.
- Rechecked paper B.1/C.1 and clarified source audit: partial observability is
  published, but the exact visibility mask is not. Own room plus divider is an
  explicit reconstruction choice, not a confirmed original-paper condition.
- New immutable analysis/probe configs underconfigs/analyses; file hashes:
  `c3499c8f3a8d7d8f5915aab3747b54b6fd0a5e94092dc3e9fd9059c618b3228f`
  and`934548863142469cf46bb9ab88141a60e7874632f745c7a5aa156285da07369d`.
  Per-command source/config/runtime snapshots and verification are recorded in
  outputs/l1-passing16-behavior-20261113. See l1_passing16_behavior_20261113.md.
- Checkpoint compatibility: environment, observations/actions/rewards, training
  config and model are unchanged; strict CPU inference remains compatible.
  No checkpoint writes, training resume or backend migration was performed.
- Validation: workspace check with --allow-dirty passed;85 unittest tests
  completed successfully (2 Docker opt-in skips); diff/compile checks passed.
- Local archive:185 members,1,492,859 bytes, all member hashes verified,
  SHA-256 `840bb344b2990c3ab600d728378c00d01ebe7556035017222d9f7481ca6e2095`.
  Automatic approval review rejected external upload to existing Draft442708737
  for lack of explicit approval of this analysis/source/verification payload.
  No upload was executed; local computation and validation are complete.
  External storage and push of the branch containing these results await approval.
- Follow-up explicit user approval on2026-11-13 authorized the exact1.49MB
  bundle and branch push. Stored unchanged bytes in Draft442708737; transfer
  manifest asset741605889, manifest SHA-256
  `74d712d89568aab656851ec93d62b733cdb185884218f03763fcdaa8259eaf97`.
  Fresh re-download and all185 member hashes verified; archive SHA unchanged.

## 2026-11-13: three explicit random-start /400-update environment variants

- User requested full observation, one-ingredient cooking, and their combination
  on a new branch. Confirmed limit unit: one ingredient type per agent per
  physical episode. Created task configs for all3, preserving random starts,
  400 complete PPO batches (28,800,000 steps), rewards, recipes, horizon,
  architecture and recurrent round semantics. No training task was launched.
- `visibility=full` reveals both rooms and all positions/directions/food states;
  physical access stays separated. Partial/unrestricted defaults and historical
  config canonical hashes remain unchanged. New fields are canonically omitted
  when they select historical default behavior.
- `max_ingredients_per_agent=1` commits on first successful chop, independently
  per agent; raw pickup does not commit. Another type's chop becomes no-op,
  consumes time, adds no new penalty and emits an explicit blocked event.
  Already chopped objects and plate handling retain normal semantics.
  Successful chops, not attempted interactions, establish the commitment.
- Limited schemas use the old12 reserved observation slots for self/peer
  six-type commitment one-hots, masking invisible peers. All4 schema names
  distinguish partial/full × unrestricted/limited; input105/actions6 retained.
  Each reset clears commitments; seed/action replay reconstructs them during
  recovery. Upstream PACE sources are unchanged.
- New task configs use disjoint construction blocks11001–11020,21001–21020,
  31001–31020 and separate held-out/learner/evaluation seeds. Canonical hashes:
  full `e4cadb24affe169c54bbc82d52b2f0f89bff066afdd6e73c2f20f0d3362666a9`;
  single `56e484a864edda6e3b862261301e0d68cd597f2801bc7128e8d5aa226bd7a6da`;
  both `ca761e86dde4a204d9079bd0f069638ab5b43d75cacb630a4df3d96e8fa54ab5`.
- Compatibility: changed scientific conditions require fresh runs/models and
  output registries. Shape105 is not permission to migrate old weights/runs.
  New environment source hash also causes strict rejection of old manifests;
  preserve old source for historical inference/resume, without bypassing guards.
- Validation covers all3 recipes from12 random seeds under all3 conditions,
  full visibility and unchanged physics/rewards, real sequential cooking that
  blocks the second ingredient, independent simultaneous choices, raw pickup,
  reset/unlock, partial masking and exact action-history state reconstruction.
  See docs/l1_environment_variants_20261113.md and environment_schema.md.
- Finite validation provenance captured before execution under
  outputs/l1-environment-variants-20261113:10 dedicated tests,108 expert-pair
  episodes; each new schema accepted by a fresh105-input neural model. Replayed
  15 old1023 evaluation episodes/383 joint steps with exact reward/event/
  success/boundary matching in the unchanged default condition. No old policy
  inference, training, checkpoint writes or Cloud launches. Source/config stayed
  unchanged throughout validation. Workspace dirty check and95 tests passed
  (2 Docker opt-in skips).

## 2026-11-13: launch four L1 agents per new environment condition

- User requested12 models: full-view condition1 in oneCloud task with4 parallel
  learners; condition2 and condition3 in twoCloud tasks each,2 learners/task.
  Prepare dedicated four-construction-model configs without changing the
  earlier20-model preparation configs or any existing experiments.
- Fresh construction seeds:11001–11004,21001–21004,31001–31004. Maintain all
  environment semantics, CPU backend, oneTorch/BLAS thread,400 full PPO batches
  and28,800,000 steps per model. No max-updates limit, old checkpoints or
  additional evaluation/model selection. Other historical runs stay untouched.
- Fixed task allocation and exact record/launch commands are in
  docs/experiments/l1-environment-variants-4agents-20261113-plan.json.
  First verify condition1's4 simultaneous jobs, saved updates, VM limits/use
  and startup artifact upload/fresh-download hashes, then dispatch remaining4
  tasks. Per-job requested memory3GiB for4-model pilot,6GiB for2-model tasks;
  distinguish unenforced rootless Docker flags from effective VM limits.
- Dedicated Draft Release443670277, tag
  l1-environment-variants-400u-4agents-20261113, stores startup/final/recovery
  bundles with unique names and hash-verified read-back/re-download.
- New config canonical hashes: full
  `0de8db77a9416c7b5b5ac3e77b93da985e4528be9f3b50c5f6ca8eee922ba411`;
  single `04111735d95efbf8c4e76691ae48c479f9a97003b1b42707f426e5523c76ca25`;
  both `9e37376eb1c96c4b5ed46ae7b78df4ab38a3f52fc0a6a7dceca191fd814fdebd`.
  Environment/model implementation unchanged from50e7e07; new config/registry
  requires fresh model runs. Exact source commit is fixed in each dispatch.

### Superseding seed instruction in the same launch session

- User explicitly chose matching100-series seeds across the three different
  conditions. New plan/configs with `s101` in their names use construction
  101,102,103,104 in all3 conditions, with independent output roots/registries.
  validation111,test121–125,learner131,evaluation141 are also aligned.
  Preserve the original distinct-seed plan/configs and any partial run records;
  do not rename old runs or import their checkpoints into the new experiment.
- Only the condition1 pilot chat had been created when the user changed seeds.
  Read-only status and conditional pre-launch hold were requested before any
  switch. The initial safe-stop message was rejected by automatic approval
  review as lacking explicit approval for stopping an already-active run;
  no stop action was transmitted. Confirm actual process state first.
- Pilot confirmed the old-seed processes/containers/writers/managers were all
  absent; no training had started, so no stop was necessary. Switched the same
  Cloud chat to fixed matched-seed source21d9d1d40948b6756b57875a181c69cc33cffcc8.
  Pilot4 workers reached saved updates2/3/4/4 within4vCPU/16GiB, used~3.73GiB,
  OOM/restarts0. Fixed manager's symlink-resolved interpreter path to use the
  existing.venv; no scientific source/runtime/worker changes. Startup upload
  and fresh-download verified, then created remaining4 chats.
- All5 chats/12 models passed startup: e2-p01 seeds101/102 update2 each;
  e2-p02 seeds103/104 update1; e3-p01 seeds101/102 update1; e3-p02 seeds103/104
  update1. All pointer/snapshot hashes match, workers/writers and single finite
  managers run, OOM/restarts0. Dedicated Draft443670277 startup transfer
  manifests741675724/741695992/741692913/741696304/741697520. See the detailed
  task/resource/hash report in docs/l1_environment_variants_training_20261113.md.
  These are startup observations, not completed400-update or final-storage claims.
- Coordinator independently downloaded all5 startup archives and verified448
  payload member hashes, all12 resume pointers/snapshots, exact seed/config/
  environment hashes and strict CPU model-state loading. 97 tests passed
  (2 Docker opt-in skips), workspace dirty/clean checks passed. Shared branch
  remains a record branch; all learning checkouts stay pinned to21d9d1d.

## 2026-11-13: forked experiment with left L1 and right serving L0

- User forked the chat and assigned previous models to the original chat.
  Use a separate managed worktree and codex/l0-serving-left-l1-20261113 branch;
  leave original checkout/Cloud chats/runs/managers and Draft443670277 untouched.
- Freeze the existing right-side ScriptedPeer recipe specialists as the new
  serving L0 pool. They prepare/plate their own recipe ingredient, wait for the
  exact two-ingredient recipe, then pick up the plate and deliver on the right.
  They neither learn nor change recipe, and reject wrong/extra plate contents.
  Formal contract: configs/l0/right-serving-recipe-pool-v1.json. Reuse existing
  side-aware peer implementation; no duplicated or changed environment policy.
- New configs set l1_ego_side0 and ego_side1. Actual L1 learner/observation is
  agent0; FrozenPartners automatically constructs agent1 scripts, and recorded
  training/evaluation partner identities are l0-1-<recipe>. Preserve full-view,
  partial-view/one-ingredient, and combined variants, random starts,105 input,
  40-step horizon,5-episode recurrent rounds,400 updates/28,800,000steps and PPO.
- All conditions use fresh models at seeds101–104. One4-model task for full
  view; two2-model tasks per remaining condition. First verify the pilot4
  workers, saved updates/resources and startup upload/download before expansion.
  Separate namespace l1-left-l0-serving-3env-400u-s101-20261113 and dedicated
  Draft443746098; no previous checkpoint transfer or old-run resumes.
- Validation:1296 prescribed-recipe joint-start cases complete with right L0
  delivering and return16.5. A visible-cue-only wait/follow left baseline also
  completes all432 starts in partial/one-ingredient mode. Unit checks cover
  incomplete/wrong plates, role-specific pipeline metadata/evaluation and
  bit-exact CPU resume of left L1. No scientific core src change was needed.
- Compatibility: original conditions and peer/environment behavior are
  unchanged. New role assignment/config/registry requires fresh runs and rejects
  cross-config checkpoint loading; existing strict guards remain unchanged.
- Canonical config hashes: full
  `6b50e2f1e0c22ef2f2a4482fd8e1bd409dfdf4fd9477588509c66ee706d34baf`;
  single `f0bbdda67e4afe66a7bbc3db059e04198eae74dae963f73e8d030ac645a7e775`;
  both `b74a3d2b47b25633d1a9dcc315ed2385fc8981c448ab1380cd26f63b3cff4001`.
- Full validation passed104 tests (102 success,2 Docker opt-in skips), clean
  and dirty workspace checks. Learning source fixed at2d82b2e888b7970806ad9b9329bcf73303396d2a.
  Pilot4 models reached snapshots1/2/1/1, ego_side0 and360 right-agent1 L0
  scripts per snapshot. Verified4CPU/16GiB,~4.6GiB use,OOM/restarts0 and a
  single independent manager, then dispatched remaining4 chats.
- All12 left L1 models reached saved update>=1 and passed startup transfer,
  role metadata and snapshot hash checks. Coordinator independently downloaded
  all5 archives, verified630 payload hashes and12 primary snapshots,4320 saved
  right-side frozen scripts and strict CPU model-state loading. Startup
  manifests:741994046,742022014,742017683,742022053,742024124 in Draft443746098.
  All worker OOM/restarts0; source/config/budgets unchanged, models continue
  toward400 updates under single finite managers. Final completion/storage is
  not yet claimed. See docs/l0_serving_left_l1_training_20261113.md.
- Corrected only operational resource monitoring/manager independence under
  outputs, retaining failed staging as evidence. e2-p02 supplemental manager
  code upload was rejected by automatic review; used saved/read-back status
  audit JSON instead, preserved the code locally, and excluded that rejected
  payload from future final bundles. Scientific source/recovery/model storage
  remains verified, and workers were not stopped for these corrections.
