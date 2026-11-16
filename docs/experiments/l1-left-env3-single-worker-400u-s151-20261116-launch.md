# Environment 3 single-worker launch record

2026-11-16: fresh managed VM, dedicated checkout at requested base 2d82b2e888b7970806ad9b9329bcf73303396d2a. Existing prepared checkout is preserved. No existing learners, writers, managers or Docker containers observed. Remote experiment branch absent before creation.

Verified baseline raw/canonical hashes. Only profile/construction_seeds differ; src/scripts have identical bytes to base. Python 3.12.14, NumPy 2.5.4, PyTorch 2.15.0+cpu in dedicated .venv. PACE pinned with six sources verified. Dirty preflight succeeds; 106 unit tests OK (2 skipped); pip check succeeds.

Observed VM cpu.max=400000 100000, affinity=0-4 (quota remains 4 CPU), memory.max=17179869184, initial memory.current=104652800, OOM counters zero. Docker limits and worker resources will be recorded after launch. No platform limits are modified.

Dedicated Draft Release ID 446571560; tag l1-left-env3-single-worker-400u-s151-20261116. This ID is the sole storage destination. Transfer uses scripts/release_artifacts.py and environment credential without exposing its value.

Exact launch command is in the plan JSON. record_run.py records command/config bytes/canonical hash/final commit/source inventory/dependencies before executing it once. The final source commit is resolved after this record is committed; outputs/execution/run.json is authoritative.

The finite manager owns no training restart. The unchanged cloud_train.py Docker policy owns at most three failure retries, with exact recovery validation by the existing pipeline. Manager records its PID/SID/PPID, command/hash and resources; startup and final saves each copy a published immutable snapshot into staging, inventory every payload SHA, upload to the fixed Draft, download to a new path, check every payload and strict CPU model load. Manager implementation itself is not transferred as a standalone artifact. Full recorded scientific source inventory is included.

Startup and final training/storage statuses, UTC/Unix times, metrics elapsed seconds and save durations are kept separately in this task's outputs/operations. Startup success does not imply 400 updates have completed. No additional task, learner, evaluation or recurring job is created.
