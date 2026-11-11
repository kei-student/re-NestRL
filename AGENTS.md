# Environment and artifact work

This repository is `kei-student/re-NestRL`, independent of NestRL. Use Python
3.12 and this checkout's `.venv`; never install the two projects into one venv.

The stable foundation is dependency setup, GitHub authentication, provenance
recording, and byte-verified artifact storage/restoration. Method, opponents,
seeds, update limits, budgets, and evaluation protocols belong to each experiment's
instructions/config/argv. Do not bake them into install/start scripts. New
conditions use new run identities/configs; never rewrite an existing run,
registry, model manifest, or resume state to make it accept changed conditions.
Routine new experiments do not require an extra human approval workflow here.

`scripts/setup_cpu.sh` installs and checks; `scripts/start_checks.sh` checks a
clean checkout. Neither starts/resumes training. Optional historical verification
uses `scripts/source_manifest.py --check provenance/original-A-20261111.json`.
That reference describes a past implementation, not policy for future research.

Keep scientific core changes separate from environment maintenance. Preserve
the selected code SHA and runtime hashes for each invocation and archive the
run-created effective config, runtime, logs and source records. Preserve final
checkpoint/latest/manifest together; exact CPU resume additionally needs the
complete recovery run and matching runtime/source. See docs/cloud_reconstruction.md.

Machine letters and responsibilities are provisional. At this handoff A does
environment maintenance, C executes separately specified experiments, Mac holds
research/analysis work, and Windows hosts instruction management. These are not
permanent restrictions or experiment defaults. Do not touch another task's active
environments/runs (including C1–C4) during this source/environment work.

GitHub credentials use existing managed authentication. Never print Secret
values, credential files, Authorization headers, or environment-variable dumps.
Release storage uses RELEASE_GITHUB_TOKEN only for api.github.com and
uploads.github.com; never forward Authorization to the asset CDN. Keep inherited
proxy and CA trust. No force push, main updates, artifact clobber, or implicit
Release publication. Follow the user's task scope for authorized actions.

Verification for environment changes:

    source .venv/bin/activate
    python scripts/check_workspace.py --device auto --allow-dirty
    python -m unittest discover -s tests -q

After committing, run workspace checks without --allow-dirty. Docker integration
tests are opt-in; never use an existing experiment as a test fixture. A setup
check is not evidence of paper-performance reproduction or cross-hardware exact
trajectory equivalence.
