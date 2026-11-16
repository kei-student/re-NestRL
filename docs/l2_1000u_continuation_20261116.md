# L2 environment 3: 400 to 1,000 update continuation

The user requested all 18 evaluated L2 policies (six each of k03/k05/k15,
learner seeds181–186) continue to **1,000 total updates**, not 1,000 additional
updates. Each policy retains its own original frozen partner subset and CPU
backend. The additional budget is600 updates,43,200,000 steps; the cumulative
budget is72,000,000 steps per model. L1 policies remain frozen.

The immutable plan is
`configs/continuations/l2-env3-1000u-20261116.json` (file SHA-256
`7fbaaaab6ded241b13e5b6524921a125c04ed0bb908d84926b99ef8ec9b0e46c`).
It authenticates every parent configuration, final model and400-update snapshot,
and declares every child root, run ID and new configuration hash.

## Explicit continuation, immutable parent

`scripts/prepare_budget_continuation.py` prepares a new child without starting
training. Only `total_steps` changes in the scientific configuration. It checks
the completed parent's final/latest/manifest, registry,400 ordered metrics,
snapshot model/optimizer, CPU runtime, core implementation/environment hashes,
and frozen partner provenance. It takes the parent's existing nonblocking
training lock and refuses an active writer, existing child, runtime mismatch,
corruption, unsupported stage or non-increasing budget.

The child snapshot preserves model, optimizer, collector observations, partially
completed environments, both policies' GRU histories, episode/round positions,
episode history and all RNG states exactly. Its metadata explicitly identifies
the new config/registry/run; `continuation.json` records the authenticated parent
and every changed identity. The script reads back and compares all other state
and verifies parent bytes remain unchanged. A `prepared.json` is published last.
No completed-parent marker is removed and no old artifact is overwritten.

Run preparation on the original Cloud VM with its retained CPU runtime and
threads1. Resume each prepared child through the unchanged native trainer:

```text
python -m nestrl_research train --config CHILD_ROOT/config.json \
  --output-root CHILD_ROOT --stage l2 --split construction --seed SEED \
  --run-id CHILD_RUN_ID --device cpu --threads 1 --checkpoint-every 1 --resume
```

Do not use `--max-updates` as a substitute total budget. The unchanged trainer
starts at401 and finishes at1000; child metrics retain1–400 from the parent.
Every invocation records its actual command, runtime/source and lineage first.
Resume only one writer per child; inspect Docker/process/manager state rather
than inferring state from chat activity. Preserve platform resource limits.

## Launch and monitoring

Use the same nine existing Cloud tasks, two policies per task. Verify the pilot
pair has saved updates above400 and that its finite manager/storage process is
healthy before dispatching the other eight pairs. Use uniquely named new
artifacts under Draft Release446574423, retaining the400-update assets. Verify
uploads by re-download, byte/hash checks and strict CPU load.

The user requested one recurring six-hour check attached to the coordinating
thread. It may query these existing tasks and resume stopped, incomplete child
runs on the same VM only after snapshot/config/source/runtime/partner checks and
writer exclusion. Respect later user stops. Never duplicate a running manager,
fall back to fresh initialization, migrate backend, extend beyond1000, or resume
unrelated old L1/L2 runs. Report inaccessible/missing/corrupt evidence explicitly.
After all18 complete and artifact verification finishes, disable the monitor.

## Compatibility and validation

No `src/nestrl_research` or PACE bytes change. Old checkpoint inference and old
same-config CPU recovery remain compatible. A400 snapshot does not natively
authenticate a1000 configuration: the explicit child preparation creates and
documents that new identity instead of relaxing the normal resume validator.

Tests compare uninterrupted and extended CPU training bit-for-bit, including
optimizer, collector, both recurrent histories and RNG, validate frozen catalog
copying with a real small L2 run, and reject wrong hashes/runtime, active writers,
budget reductions and destination collisions.
