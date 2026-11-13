# Cooking environment reconstruction

The immutable PACE implementation at commit
`f008c19aa743ec209550375c9ee9c25379ac8b24` supplies `CookingWorld`, object
classes, collision/interact/chop/merge dynamics and recipe graphs. A private
module loader executes those unchanged sources without importing their legacy
gym/PettingZoo package initializer or adding aliases to `sys.modules`.
NestRL-specific layout, observations and orchestration live under
`src/nestrl_research`; no upstream file is modified.

## Layout and termination

The fixed layout is reconstructed from Figure 1 of the supplied paper. Coordinates
are zero-based `(x, y)`, with x increasing to the right and y increasing downward.
It is 7 columns by 8 rows. Floors occupy x in `{1,2,4,5}`, y in `{1,...,6}`;
all other cells are counters, with the following replacements and objects:

| Object | Position |
|---|---|
| Left chopping board | (0,4) |
| Right chopping board | (5,7) |
| Delivery | (6,3) |
| Shared plate | (3,3) |
| Lettuce / Tomato / Potato | (0,3) / (0,1) / (0,2) |
| Onion / Carrot / Broccoli | (6,4) / (6,5) / (6,6) |
| Chef 0 start (random mode) | Uniform floor in x={1,2}, y={1,...,6} |
| Chef 1 start (random mode) | Uniform floor in x={4,5}, y={1,...,6} |

With `environment.random_start=true`, each reset independently samples each
chef's location uniformly from the 12 floor cells in its own room (144 possible
joint starts). Rooms and chef roles stay fixed, so starts cannot overlap or land
on counters/objects. Independent samples may repeat a previous location.
`reset(seed=N)` restarts a local NumPy generator and reproduces the initial
observation; subsequent `reset()` calls advance that generator rather than
reseeding it. Training and evaluation already supply an explicit episode seed,
so their recorded seeds also reproduce the sampled starts and resume replay.

Standalone environments and programmatic configs default to random starts.
Use `configs/paper-random-start.json` or `configs/smoke-random-start.json` for
new experiments. Historical JSON configs that omit `random_start` retain fixed
starts `(2,5)` / `(5,5)` and their original canonical config hashes. Explicit
`random_start=false` also selects that legacy condition and is serialized in
the same canonical form with the field omitted. Historical config files and
experimental records are unchanged. Use a new output root/registry and run
namespace for the random condition; fixed-start checkpoints cannot be reused
as construction partners or resumed under the new condition. Strict checkpoint
loading also rejects historical environment/source hashes in this revision.

The PACE agent constructor initializes both chefs facing left. The divider at
x=3 is impassable. Only the paper's main-text recipes are enabled:
`LettuceOnionSalad`, `TomatoCarrotSalad`, and `PotatoBroccoliSalad`. Other
ingredient pairings from Appendix B.2 are intentionally excluded, as requested.
The environment terminates when one permitted recipe is delivered and truncates
at 40 simultaneous steps otherwise. A completion at step 40 is a termination.

Actions retain PACE semantics: `0=wait`, `1=left`, `2=right`, `3=down`, `4=up`,
`5=interact`. A directional action changes orientation even when the destination
is a counter. Interacting with raw food on a board chops it once; interacting
again picks it up. Food must be chopped before merging into a plate. The two
actions are submitted together and resolved by the upstream world.

## Shared reward

The PACE reward procedure is retained with all ingredient desires set to one:
0.5 for a previously unrewarded `(chef, held object, held state)` interaction,
1.0 for each newly completed recipe-graph node, and 10.0 for delivery completion.
The largest per-recipe reward is shared by the two chefs. Chopping, plating and
plate contents use the upstream recipe graphs, including their nested Plate
nodes; changing an earlier achieved state can produce negative progress reward.
Picking up an unchanged object repeatedly by the same chef cannot repeat its
0.5 bonus. A different chef or a newly chopped/filled state is a distinct
interaction, matching PACE. The paper specifies no step penalty, so `step_cost`
defaults to **0.0**; the separate upstream experiment YAML's 0.2 is not inherited.

## Observation schema

Baseline schema identifier: `pace_dense_partial_93_plus_reserved12_v1`.
`reset(seed)` returns a float32 `(2,105)` array. No recipe identifier is appended.
`step((a0,a1))` returns `(obs, shared_reward, terminated, truncated, info)`;
`info` includes `success`, completed `recipe` (otherwise null), `events`, and `t`.

Appendix B.1 states a 105-dimensional partial observation but does not publish
indices, visibility geometry, masking, or the initialization details. PACE's
checked-in 105-dimensional partial configuration describes a **different**
7-by-13 layout with nine food objects and one chopping board. Applying PACE's
same dense partial feature formula to Figure 1's six foods and two chopping
boards yields **93** dimensions. Consequently this reconstruction keeps those
93 meaningful features and reserves the final 12 values as zero. It does not
invent undocumented features to fill the discrepancy. Exact observation-level
equivalence to the paper is **unverified**.

Visibility is reconstructed as the chef's own side of the vertical divider plus
the divider itself: x<=3 for chef 0, x>=3 for chef 1. This differs from the
upstream large-map YAML's horizontal y-based visibility split and is an explicit
Figure 1 adaptation. Hidden positions, directions and food states are zeroed,
and their visibility flags are zero. Local held/front/adjacency features retain
the PACE definitions.

The following half-open ranges are zero-based:

| Indices | Feature |
|---|---|
| [0,36) | 12 position triples `(x/7,y/8,visible)` |
| [36,46) | Own then peer orientation: four one-hot values + visibility |
| [46,58) | Six food pairs `(chopped,visible)` |
| [58,62) | Adjacent non-floor static objects: east,south,west,north |
| [62,69) | Held types: Tomato,Lettuce,Onion,Plate,Carrot,Potato,Broccoli |
| [69,75) | Held food done flags: Tomato,Lettuce,Onion,Carrot,Potato,Broccoli |
| [75,87) | Front types: Counter,CutBoard,DeliverSquare,Tomato,Lettuce,Onion,Plate,Agent,Blender,Carrot,Potato,Broccoli |
| [87,93) | Front food done flags in the same food order |
| [93,105) | Reserved zeros |

Position triples are ordered: own chef, other chef, left board, right board,
delivery, plate, Lettuce, Tomato, Potato, Onion, Carrot, Broccoli. Own position is
absolute; all others are relative to the observing chef. Food-state pairs follow
Lettuce,Tomato,Potato,Onion,Carrot,Broccoli. Orientation order is left,right,down,up.
Plate contents additionally activate their ingredient held-type/done features.

## Recipe peers and validation

`ScriptedPeer(recipe, agent_id, seed, random_action_probability)` is a new level-0
recipe-expert reconstruction. It uses breadth-first search over floor positions
and orientations. Each chef prepares its own side's ingredient and puts it in
the shared plate; the right chef delivers only after both correct ingredients
are present. It has no training or access to previous NestRL implementations.
An optional seeded random action probability supplies an explicit diversity
parameter; the deterministic expert uses zero. Against one another, the experts
complete LO/TC/PB from the legacy fixed starts in 16/19/17 steps, with shared
returns of 16.5 respectively. With random starts the completion length varies.
The paper does not publish these exact peer controllers, so peer-policy
equivalence is also unverified.

Tests check all three recipes with both side controllers, partial visibility,
reserved slots, shared chopping rewards, repeated pickup bonuses, invalid recipe
delivery, unprepared food, no-op truncation, seed/episode-stream reproducibility,
valid random starts on all 12 floor cells per room, actual collector resets,
bit-exact CPU resume with random starts, legacy config hashes, and isolation
from legacy package imports.

## Explicit visibility and cooking-limit variants (2026-11-13)

The original partial/unrestricted condition remains the default. New task
configs opt into three independent conditions; see
[the three-variant plan](l1_environment_variants_20261113.md).
Historical files and their canonical config hashes are unchanged.

| visibility | max_ingredients_per_agent | Observation schema |
|---|---|---|
| own_side | null | pace_dense_partial_93_plus_reserved12_v1 |
| full | null | pace_dense_full_93_plus_reserved12_v1 |
| own_side | 1 | pace_dense_partial_93_plus_ingredient_locks12_v1 |
| full | 1 | pace_dense_full_93_plus_ingredient_locks12_v1 |

`full` removes the observation masks for both rooms, all object/food positions
and states, and the other agent's position/orientation. It does not change
physical access, the divider, start distribution, recipes or rewards. This is
an explicitly requested experimental condition, not a claim about the paper's
unpublished visibility geometry.

Limit1 means one ingredient **type per agent per physical episode**. The first
successful chop selects that agent's type. Raw pickup, transport and replacing
food do not select it. A subsequent attempted chop of another type is executed
as no-op, consumes one step, emits `ingredient_limit_blocked`, and introduces
no additional penalty. Normal world/reward processing still occurs. The other
agent acts normally on the same step. `ingredient_selected` records first
selection. Already chopped food can still be picked up and plated. This does
not restrict an agent to handling only one object or preselect a recipe for it.
The upstream PACE sources remain unmodified.

In limit1 schemas, the 12 old reserved slots encode the added dynamic state:
`[93,99)` is the observer's ingredient selection and `[99,105)` the peer's
selection, each one-hot in Lettuce,Tomato,Potato,Onion,Carrot,Broccoli order.
All-zero means not yet selected, except that an invisible peer's slots are
always masked to zero. Thus the partial variant does not leak hidden choices.
Each physical reset clears both selections, including resets inside a
five-episode round. The recurrent round context is still maintained normally.
The existing seed/action-history recovery reconstructs these selections by
replaying successful chops; they are not reset on a mid-episode recovery.

`info.observation_schema` records the actual configured schema. Limited
conditions also report `chosen_ingredients` and `executed_actions`; denied
attempts retain the original requested action in ordinary evaluation traces.
Observation shape stays105 and action count stays6. Schema/visibility/limit
mismatches and limits other than null or integer1 are rejected.

Changing these conditions requires a new config, output root and model run.
Existing400-update models must not be resumed under a new condition. This
revision changes the environment source hash, so strict loaders also reject
historical model manifests even when using the legacy condition. Use each old
run's recorded source for its exact inference/resume; no guard is bypassed.
