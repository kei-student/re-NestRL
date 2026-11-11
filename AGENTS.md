# re-NestRL research base

- Shared source: `https://github.com/kei-student/re-NestRL.git`. Use clean
  checkouts, `git pull --ff-only`, and short-lived `codex/` branches. Never
  force-push or rewrite shared history. One writer per branch/run directory.
- This reconstruction is independent from NestRL. Use Python3.12 and this
  repository's `.venv`; do not install both projects in one environment.
- Keep PACE at `f008c19aa743ec209550375c9ee9c25379ac8b24` and unmodified.
  Verify the six imported sources with `scripts/bootstrap_upstream.py`.
- Fix only runtime reconstruction, GitHub access, provenance, and artifact
  storage/recovery in the common base. Install/start must not choose an
  experiment, train, resume, or create a recurring training job.
- Accept method, partners, seeds, updates, budget, stopping/evaluation rules,
  and concurrency from each task. A/C names and execution-machine roles are
  temporary. Record the actual command/config/code SHA before execution.
- Keep historical runs and their source/config/manifest hashes immutable.
  Different scientific conditions/backends require a distinct config/run ID.
  Preserve construction/validation/test splits and frozen partner policies.
- Do not change a running checkout or experiment. Idle chats may still have
  active training processes. Existing C1-C4 runs and their controller are
  outside the cloud-base migration and must not be modified.
- Store models/run bundles in a specified GitHub Draft Release, with model,
  manifest, `latest.json`, effective config, source/runtime, logs, and supported
  recovery state. Verify SHA-256 after download. Hash-matching existing assets
  may be reused; never replace conflicting assets or forward API authorization
  to a download CDN. Use environment credentials, never token command arguments.
- Never commit `.venv`, external PACE, outputs, checkpoints, caches, credentials,
  or machine-specific paths. Use repository/run-relative `pathlib` paths and LF.
- Scientific changes must update config, meaningful tests, and
  `docs/implementation_log.md`, including checkpoint compatibility.
- Before commit: `python scripts/check_workspace.py --device auto --allow-dirty`
  and `python -m unittest discover -s tests -q` from the activated `.venv`.
  Before handoff, run the workspace check again without `--allow-dirty`.
  Push intentional files and report branch, SHA, checks, config SHA-256, and
  checkpoint compatibility.
