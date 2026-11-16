# Mirrored L1 evaluation, 2026-11-16

Evaluate every final400-update L1 from the mirrored full-observation/single-ingredient campaign:
seeds101–104, source4ad6d8a6d82c2eb04a3062b91bbe7db0690029bf,
config canonical SHA256 f08c233af8b7a7102bb61aea2ec512bc23f2b72b978dacbc5dc633933e4b949c.

The frozen plan is configs/evaluations/l1-mirrored-full-single-20261116.json.
Each right-side frozen L0 recipe expert receives200 rounds of5 random-start physical episodes
at horizon40, seed6001, sampled actions, and recurrent state held within each round.
Each of four models receives3000 episodes; all three partners need950/1000 successes.
Final checkpoints are used without selection, optimization or training resumption.
CPU inference uses one Torch/BLAS thread per worker, four workers.

Draft446495466 holds the original models/recovery/source bundles. Verify their downloaded
archive/member SHA, all400 metrics, final and resume400, config/registry/provenance and strict108-input
CPU load before evaluation. Evaluation checks checkpoint SHA before/after and independently
recounts every episode and trace. A new evaluation namespace records command/source/config/runtime.

Environment3 seed151 is evaluated separately on its original105-input source16216917ce67bf2e21515dc328b72255d20ea103
and Draft446571560 using the same1000-episode schedule. The105 and108 models are not mixed.
Evaluation plans do not change inference or exact CPU recovery compatibility; historical
configs, manifests, models and running checkouts remain immutable.

Results and final storage verification will be appended after the evaluations finish.
