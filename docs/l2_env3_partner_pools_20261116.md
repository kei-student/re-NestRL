# Environment 3: L2 partner-count comparison

This is preparation for three independent L2 conditions, each with 400 PPO
updates (360 environments × 200 steps × 400 = 28,800,000 environment steps).
Preparation and setup do not start or resume training.

The environment remains the same full-observation, random-start reconstruction
with at most one ingredient type cooked per agent per episode. L2 occupies the
left room (side 0); frozen L1 occupies the right room (side 1). Rewards, recipes,
horizon 40, MLP[128,128]/GRU128, PPO, five-episode round history and CPU/thread-1
execution remain fixed. Partners are sampled uniformly per environment at each
round boundary, with replacement across rounds, using the existing collector.

## Candidate cohort and independent draws

The candidate pool contains exactly these fifteen construction L1s:

`101,102,103,151,153,157,160,161,162,164,165,166,167,168,170`.

All passed at least 950/1000 episodes against each of the three fixed L0
conventions. Selection is based on the already completed qualification, without
L2 results or further tuning. Original models/configs/registries/manifests and
their hashes remain unchanged. Historical validation/test partitions are not
relabelled or used for L2 training.

The user's latest instruction is an **independent draw for every L2 agent**.
Each 3/5-partner L2 draws uniformly without replacement from these fifteen.
Fresh OS entropy supplies a selection seed, independently of the L2 learner
seed and condition. NumPy PCG64 and the selected identities are recorded before
training. Distinct L2 agents may overlap or coincidentally draw the same subset;
no condition-wide subset, nesting, forced disjointness or uniqueness is imposed.
The 15-partner condition necessarily uses all fifteen every time.

| Condition | Pool selection for each L2 | L2 updates |
|---|---|---:|
| k03 | a fresh independent draw of 3 | 400 |
| k05 | a fresh independent draw of 5 | 400 |
| k15 | all fifteen candidates | 400 |

Current study manifest: `configs/l2-env3-per-agent-20261116.json`.
Use `scripts/create_l2_config.py` for each L2; it writes a new `config.json`
and `selection.json` in that agent's output root and never starts learning.
Supply the learner seed explicitly. Do not derive or share a selection seed
across L2s. The optional `--selection-seed` is for reproducing a recorded draw.
Membership remains fixed throughout one L2 run. To resume, reuse that run's
saved config and selection; the generator refuses an existing output root.

The earlier `configs/l2-env3-experiment-20261116.json`, fixed-subset configs,
and prepared outputs remain unchanged historical examples/templates. Their
fixed L1 IDs are replaced when a per-agent config is generated. They are not
the launch plan for the user's per-agent sampling instruction. No replicate
count or Cloud launch is scheduled by preparation.

| Historical template | Canonical config SHA-256 |
|---|---|
| k03 | `05f5e9a413a8df02f7bf2e4981753a05692724995e125b3313e2b60919661417` |
| k05 | `07a126a01d39059c141fd00063c8cd63fb5632563049c0070a9a0a6b47172dd1` |
| k15 | `ba4f9ad47ea468aaa8a51f69e7970a1bcec853245cdc654fe5cd7676d87af4f4` |

The catalog `configs/partner_pools/env3-passing15-20261116.json` authenticates
all original artifact bytes and Draft Release transfer manifests. Its file
SHA-256 is `78e47375a85c5fe6d02851781f3fb04b9e50a0eb1f0c3b891da390c36d1c834f`.
Sources are Drafts 443670277 (seeds 101–103) and 446096945 (the twelve newer L1s).
Weights and raw outputs stay outside Git. No evaluation files are uploaded by
these preparation tools.

Inspect a historical selection, without changing files or running an experiment:

```sh
.venv/bin/python scripts/select_l1_partners.py \
  --catalog configs/partner_pools/env3-passing15-20261116.json \
  --count 3 --selection-seed 2026111603
```

## Download and prepare; no learning

Use this repository's Python3.12 `.venv`. The Cloud environment supplies
`RELEASE_GITHUB_TOKEN`; never put a token in arguments, logs, or bundles.

```sh
.venv/bin/python scripts/fetch_l1_catalog.py \
  --catalog configs/partner_pools/env3-passing15-20261116.json \
  --output-dir outputs/l2-env3-partner-pools-20261116/archives

.venv/bin/python scripts/create_l2_config.py \
  --condition k03 --learner-seed 181 \
  --output-root outputs/l2-env3-per-agent-20261116/k03-s181-a

.venv/bin/python scripts/prepare_l2.py \
  --config outputs/l2-env3-per-agent-20261116/k03-s181-a/config.json \
  --catalog configs/partner_pools/env3-passing15-20261116.json \
  --archives-dir outputs/l2-env3-partner-pools-20261116/archives \
  --output-root outputs/l2-env3-per-agent-20261116/k03-s181-a
```

Create another unique output root for every additional L2, including within
the same condition. Then prepare its own config. Preparation authenticates archive/manifest/member
bytes, preserves each model's **original** config and registry, checks roles,
history, environment and model-source compatibility, performs strict CPU load,
freezes all L1 weights and checks batched inference. Existing different files
are never overwritten. All fifteen source models are staged for the later
common comparison, but training loads only the condition's selected 3/5/15.

The loader does not bypass the original-config checkpoint check: it loads each
L1 with its own authenticated config, then validates the policy contract against
L2. A mixed catalog is permitted only for explicit L2 training, never as an
implicit request to generate new L1s or execute the full pipeline.

## Future explicit training command

The following is a command template, not something setup executes:

```sh
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/record_run.py \
  --run-id l2-env3-k03-s181-a-launch \
  --config outputs/l2-env3-per-agent-20261116/k03-s181-a/config.json \
  --output outputs/l2-env3-per-agent-20261116/k03-s181-a/execution/launch \
  -- .venv/bin/python -m nestrl_research train \
  --config outputs/l2-env3-per-agent-20261116/k03-s181-a/config.json \
  --output-root outputs/l2-env3-per-agent-20261116/k03-s181-a \
  --stage l2 --seed 181 --device cpu --threads 1 --checkpoint-every 1
```

The resulting run ID is `l2-181` within this agent's unique root and config
SHA. A later Cloud launch must pin its source/runtime, select its Draft storage
destination, and use the one-task startup gate before expansion. CPU resume
continues only the same authenticated config, registry and ordered frozen
partner provenance. Partner replacement or source changes reject resume.

## Common comparison after learning

An explicit common evaluation pool includes all fifteen candidates:
`configs/partner_pools/env3-common15-evaluation-20261116.json`.
Evaluate each trained condition against this same pool and episode seeds:

```sh
.venv/bin/python -m nestrl_research evaluate \
  --config outputs/l2-env3-per-agent-20261116/k03-s181-a/config.json \
  --checkpoint outputs/l2-env3-per-agent-20261116/k03-s181-a/l2-181/checkpoints/final.pt \
  --output-root outputs/l2-env3-per-agent-20261116/k03-s181-a \
  --run-id eval-l2-181-common15-s6001 \
  --partner-level l1 \
  --partner-pool configs/partner_pools/env3-common15-evaluation-20261116.json \
  --rounds 200 --evaluation-seed 6001 --device cpu --threads 1
```

This is 1,000 episodes per L1, 15,000 per L2. The protocol records
`construction_diagnostic` and each partner's `seen_in_l2_training` flag.
Report the common fifteen plus seen/untrained-within-cohort results separately.
These partners remain construction policies: this is not validation/test
generalization. The ordinary held-out evaluator refuses to silently relabel
this cohort as test data.

## Compatibility and validation

Old config hashes and ordinary L1 inference remain compatible. Original L1
weights, manifests and registry bytes are preserved. New L2 configs have new
hashes and explicit pool provenance. Historical exact training resumes still
require their historical implementation; this source change does not authorize
resuming old L1/L2 trajectories under new code.

Tests cover heterogeneous source configs, semantic mismatch/corruption/split
rejection, immutable staging, selected-partner-only L2 training, CPU recovery,
frozen weights, independent draws and construction-diagnostic evaluation. Real
model preparation and rollout preflight reports are stored in the ignored
`outputs/l2-env3-partner-pools-20261116/` namespace. Those preflights perform zero
PPO updates and create no production L2 training run.
