# Cross-play of coverage L2 with the author's L0

Evaluate the six frozen coverage L2 agents (seeds181–186, total400updates) on
unchanged re-NestRL environment3, left ego and right L0. This is cross-play on
the reconstruction, not evaluation in the author's native environment.

The author's public source is upasana27/adaptive-RL commit
918d33fc831c520545460e7c06d0b6d797ab8be4. The full policy.py and cooking_zoo.py
bytes are SHA-pinned in configs/evaluations/l2-coverage-author-l0-20261118.json.
The driver executes only the original event/RuleBasedPolicy/generate_policy_pool
AST plus their five original constant assignments. The imported object class
names are bound to re-NestRL's already pinned PACE world classes. The author's
Ingred2ID is literal-read from its pinned original source. No policy methods,
event weights, action map, reset behaviour or navigation are rewritten.
Unrelated pretrained/CUDA/storage/gym imports are not executed.

The author L0 reads the world state. It does not require its native73-dimensional
observation; the L2 still receives its original105-dimensional environment3 input.
The reconstruction's fixed ingredient/plate placement, full visibility, per-agent
per-episode one-ingredient restriction, random agent starts, reward and horizon40
remain unchanged. Native author environment randomizes other objects and differs
in observation/reward; those differences are not introduced into this experiment.

Use the author's three-recipe full factory, pool_seed1, conventionNone,
plate_take_requires_deliverFalse and native env_name fc_small_divider_test.
The factory creates fixed event weights once and preserves global NumPy RNG state.
Canonical reporting order is LO/TC/PB; construction order remains the author's
shuffled order so weights match its factory. The original reset() is a no-op,
including its current event; do not silently replace it with refresh_event.

Primary p=0.25 follows the author's FC/PC evaluation scripts (nav_p=tar_p=rand_p).
Secondary p=0 removes these three noise knobs. Event selection and random fixed
weights remain stochastic at p=0: it is not a deterministic controller.

For each model and condition evaluate each L0 for200 rounds x5=1000episodes.
CPU, Torch/BLAS1thread, stochastic ego actions, evaluation_seed6001.
Reset ego recurrent context at each five-episode round; keep it across the five.
Environment seeds follow the prior native evaluator's canonical recipe index rule.
Total six models x two conditions x three recipes x1000=36000episodes.

The driver records exact author/driver/core/config/checkpoint identities, factory
event probabilities, commands/runtime, complete traces/episodes/progress and an
artifact hash index. It verifies all training-run files, author sources, config,
registry and driver bytes before/after, and verifies ego in-memory tensors.
A separate streaming pass checks every trace/episode step, action range, return,
final success and independent summary counts. Original runs/weights stay frozen.

Start evaluation in the existing p01 VM first; validate its completed p0.25
generation and immutable inputs/resources, then instruct the existing p02/p03.
Each task runs two CPU evaluation workers, sequential conditions per learner.
No new Cloud VM, training, recurring schedule, or evaluation artifact upload.

The tests exercise wrong-source rejection, exact factory parameters and RNG
preservation, original reset lifecycle, all three recipe pair trajectories,
reproducible frozen evaluation with both noise conditions and corrupted-trace
rejection. Set RE_NESTRL_AUTHOR_SOURCE_ROOT to a checkout of the pinned author
revision when running the optional real-source integration tests; no author code
is vendored into this repository.

Only scripts/config/tests/docs change. src/model/environment/peers/PACE are
unchanged; all existing checkpoints remain inference-compatible. New evaluation
IDs distinguish these partners from the reconstructed L0 results.

