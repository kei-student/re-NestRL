# Frozen evaluation of the eighteen environment-3 L2 agents

The user's clarified comparison is fixed L0 performance and performance with
L1 policies not used by each L2 during training. No recipe-restricted environment
is introduced. All policies remain frozen; environment dynamics, observations,
rewards, roles, training seeds and 400-update budgets stay unchanged.

Protocol: CPU, torch/BLAS one thread per worker, stochastic actions, evaluation
seed6001, horizon40, random starts and five episodes of recurrent history per
round. Reset both policies at each round; retain hidden state between its five
episodes while resetting previous action and episode-boundary inputs.

- Each fixed L0:200 rounds x5=1,000 episodes/model/L0. A model passes all recipes
  exactly when each of the three fixed L0 counts is at least950/1000.
- Common15 L1 diagnostic:200 rounds per L1,1,000 episodes per L1. For k03 report
  the complement of that model's own3 training partners (12 L1s); for k05 report
  its complement of5 (10 L1s). The running common15 evaluation supplies these
  counts without changing or re-running scientific inputs. Seen-cohort results
  are supplementary.
- k15 has no unused L1 in this cohort. Sample uniformly with replacement from
  the fifteen at each of3,000 rounds (15,000 episodes/model). Independent
  recorded PCG64 selection seeds2026111781..2026111786 correspond to L2
  seeds181..186. Selection uses a dedicated generator, separate from action
  and environment RNG. For global round r and position e, the episode seed is
  6001+1000*r+e. Primary random-partner performance is pooled episode success;
  per-partner macro/minimum and observed sample counts are also retained.

This is an untrained-partner comparison within a fixed construction cohort,
not an immutable validation/test-split generalization claim. k15 random results
use known training partners and must be labeled separately. Do not infer
recipe-specific success rates by dividing only among successfully cooked dishes.

Plan: configs/evaluations/l2-env3-final-comparison-20261116.json.
Native CLI evaluates L0 and common15. scripts/evaluate_l2_random_partners.py
uses the same PolicyContext, CookingEnv and summarizer for random round partners,
records the full precomputed schedule and verifies checkpoint files and all
in-memory policy tensors before/after. It never updates weights.
Tests compare a single-partner case bit-for-bit with native evaluation, including
all joint actions/events, stochastic inference and round history.

Clouds retain their pinned training checkout24e53e4. They may fetch this branch
without checkout and materialize only the hash-verified evaluation script into a
new outputs directory. Record the driver commit and file SHA separately from
the unchanged core/model commit. No checkpoint is converted or resumed.

Results and audit files stay on the originating VMs and local coordinator;
GitHub evaluation upload requires a separate user request.

