# First-touchdown pose and six-support deadline

**Status: approved GPU trial and evaluation completed; no policy accepted.**
A preserved late branch reduces nominal heading drift below 1° median, but its landing
loads and failures regress. See the separate post-hoc diagnostic below.
The preceding [pad/heading trial](PAD_HEADING.md#completed-gpu-results) was rejected.
The next warm start remains the earlier guarded +10 candidate, not the rejected
third-trial finalist. The original checkpoint remains a separate comparison control.

## Measured gap being corrected

The third trial's frozen finalist lost four nominal qualifications across two fresh
32-start holdouts, against 64/64 for +10. Every loss touched down at 343.8–344.8°,
inside the existing 340–380° rotation guard, but with upright cosine 0.9616–0.9651,
below evaluation's 0.966. The landing reward required the robot to be upright *now*,
not at its first touchdown. Later recovery could therefore regain landing credit for
an initially invalid touchdown.

The third trial also left approximately 22–23° heading drift, failing the ±10° goal.
Its saved learning rate stayed 1e-5 at every milestone; do not blame a learning-rate
increase that did not occur.

## Current implementation

The command now latches first-touchdown upright cosine and control-step index alongside
forward rotation. It also records the first subsequent six-support step. Every latch
is idempotent within a control step, and a partial reset clears only reset environments.

A training-only `landing_pose` termination rejects either:

- first-touchdown upright cosine below 0.966; or
- failure to reach six supports within 100 ms after that first touchdown.

Landing credit requires that original pose and a timely first six-support stance.
A later recovery cannot repair a tilted first touchdown or a missed support deadline.
Exactly 100 ms is allowed, matching the evaluator's inclusive window; 105 ms is late.
The existing 340–380° rotation guard remains.

Replay omits both training landing terminations, keeping the full failed attempt
visible for force evaluation. A clean-looking shortened episode cannot count as a
soft landing. Robot mechanics, reference, observation layout and collision limits are
unchanged; actor/critic/optimizer restoration remains exact.

The separate yaw-rate cost is increased from -0.5 to -10. The measured original charge
was only 0.203 per nominal episode, versus 16.14 for heading. Gyro and gravity make
world-up yaw rate observable to the actor; absolute heading is not an actor input.
This is a proposed learning correction, not evidence that heading has been fixed.
The absolute worst/mean pad cost and heading cost retain their third-trial values.

## CPU controls

Windows AMD CPU, native backend, sensor noise on, seed 1234:

| Control | Measured result |
|---|---|
| Synthetic first-touchdown cases | Valid and exactly-100-ms landings get credit 1 |
| Tilted touchdown, later upright | Credit remains 0 |
| Six supports first reached at 105 ms | Credit remains 0; guard rejects |
| Under-rotated touchdown, later 360° | Credit remains 0 |
| Startup and repeated readers | No false touchdown or duplicate state advance |
| +10 nominal, guards enabled | 8/8 old-definition flips; zero landing-guard terminations |
| +10 randomized, guards enabled | 15/32 survive; 17 training-guard rejections; zero collision terminations |
| Real partial resets | Surviving touchdown pose/step/support latches unchanged |
| Actual training smoke | 32 CPU updates, 8 environments, exact actor/critic/optimizer/iteration/counter/LR restoration |

Nominal continuation-vs-immediate-termination margin stays positive: +49.82 median /
+48.44 minimum. Among surviving randomized landings it is +44.46 median / +33.33
minimum. The randomized guard counts are **training constraint rejections**, not
physical crash counts from an uncensored comparison. Neither batch passes the heading
goal yet; these checks validate the rule, not an improved trained policy.

Full repository suite after this correction: **1221 passed, 28 skipped, zero failures**,
105.88 s. Lint and whitespace checks pass. No task-specific tests were added under
`tests/`; the one-off synthetic probe lives outside the repository, with evidence
under `out/frontflip-landing-pose-guard/`.

## Approved bounded GPU validation

Private dataset: `claytoncrumbly/jumper-frontflip-pose-guard-inputs-20261111`.
Private kernel: `claytoncrumbly/jumper-frontflip-pose-guard-20261111`.

Two seeds, 1234 and 2718, each train 150 updates from the same +10 checkpoint at 256
`warp:cuda:0` environments. Save 0/10/25/50/100/150 snapshots. Verify actual CUDA
physics, all source hashes and exact milestone-zero actor/critic/optimizer restoration.

Paired eight-start nominal/randomized screens retain the previous selection rule:
preserve old-definition flip counts against +10 and admit no physical failures;
freeze at most one candidate per training seed by highest straight-flip count, then
lowest summed worst-limb median, then non-pad median. Compare against both original
and +10 controls. Freeze the shortlist before any new holdouts.

Fresh holdout seeds are **17011 and 28027**, each with 32 nominal and 32 randomized
starts, observation noise on, both training landing guards disabled. Do not reuse the
third trial's adapted-to holdouts as fresh confirmation. Preserve the ±10° first-touchdown
and final heading requirement, ≥90% qualification, no worse old flip count, no physical
failures/incomplete windows, no limb median/p90 regression, lower worst-limb median,
and lower non-pad median without higher p90, against both controls.

## Completed GPU trial and frozen holdouts

Both seeds completed 150 updates on a verified Tesla T4, `warp:cuda:0`, 256 environments.
A real CUDA physics step and all 540 source hashes were verified. Both milestone-zero
actor/critic/optimizer states match +10 exactly. The learning rate remains 1e-5 at all
saved milestones. Mean training reward changed -42.94 → -30.81 (seed 1234) and
-46.59 → -30.73 (seed 2718); those improvements are not policy acceptance.

Ten checkpoints were screened. Only `seed-1234-after-10` entered the frozen shortlist.
Its fresh holdouts, noise on and both training landing guards disabled, measured:

| Seed / physics | Old flip count, +10 → new | New straight flips | Physical failures, +10 → new | Heading median, +10 → new | Worst-limb median, +10 → new |
|---|---:|---:|---:|---:|---:|
| 17011 / nominal | 32 → 31 | 0 | 0 → 0 | 24.5 → 20.7° | 44.8 → 46.1 N |
| 17011 / randomized | 11 → 12 | 0 | 2 → 4 | 23.6 → 19.7° | 36.8 → 41.3 N |
| 28027 / nominal | 32 → 32 | 0 | 0 → 0 | 24.6 → 20.3° | 44.2 → 45.2 N |
| 28027 / randomized | 12 → 9 | 0 | 2 → 2 | 24.4 → 20.8° | 39.6 → 39.9 N |

The frozen finalist fails every 90% straight-flip gate and multiple load/reliability
gates. It does not replace either original or +10.

## Preserved late heading branch — diagnostic, not acceptance

Checkpoint: `out/frontflip-landing-pose-guard/gpu-output/seed-2718/model_after_150.pt`,
SHA-256 `06d22d3ff414b3ad24a8491f284ba918438405248b5f30dfb39361836faab304`.
It was trained for 150 updates from +10 on the verified T4 at 256 environments.

This late checkpoint passed 7/8 nominal straight flips in the screen, with 1.00° median
heading drift, but failed the randomized screen: 0/8 straight flips and three physical
failures. It was therefore **not** eligible for the frozen shortlist. Do not quietly
promote it after viewing those results.

A separate **post-hoc nominal-only diagnostic**, seed 80817, 32 paired starts, noise on,
was run to characterize that heading progress. It is not a fresh acceptance holdout:

| Measurement | Stable +10 | Heading branch |
|---|---:|---:|
| Median absolute final heading drift | 24.24° | 0.89° |
| Proper rotation/upright/support landings | 32/32 | 24/32 |
| Physical failures | 0/32 | 6/32 |
| Non-pad peak median / p90 | 63.75 / 68.44 N | 96.23 / 100.54 N |
| Worst-limb median, complete nonfailed windows only | 45.93 N | 49.22 N |

The heading improvement is real under this nominal condition, but the policy is not
robust or lower-load. Preserve this as a **heading-study branch**, not a new accepted
policy or an automatic replacement of the stable +10 warm start.

Stock measured replays were completed for both the frozen finalist and this heading
branch: `native:cpu`, one environment, observation noise on, 400 Hz physics / 200 Hz
control, 700 control steps. Each CSV contains 699 pre-reset samples, t=0–3.49 s, peak
absolute actuator torque 1.7464 N·m. `measure/` directories are beside their checkpoints.
Ground-reaction evidence comes from the independent probe, not the joint CSV.

The heading branch's fixed seed-1234 nominal illustration finishes -0.38° from its
starting heading, on six supports, but reaches 98.04 N non-pad contact. The video is
therefore explicitly experimental:

- [Quarter speed, 14 s](../../../out/frontflip-landing-pose-guard/video/experimental-heading-after150-nominal-quarter-speed.mp4)
- [Normal speed](../../../out/frontflip-landing-pose-guard/video/experimental-heading-after150-nominal-side.mp4)

## Independent no-training symmetry diagnostic

A bounded eight-start CPU probe applied `(f(x) + M f(M x)) / 2` to the stable +10 actor,
using the repository's tested observation/action mirror transform. This changes the
inference policy; it is **not** performance of the checkpoint alone, and no simulator
forces, state or robot mechanics were changed.

Nominal: 8/8 straight flips, zero failures, median heading 0.52°; however worst-limb
median rises from 43.65 to 53.96 N. Randomized: 0/8 straight flips, one physical failure.
Removing learned left/right bias can therefore remove nominal yaw without retraining,
but it does not solve pad loads or randomized landing quality. Raw evidence is under
`out/frontflip-mirror-check/`. No symmetry wrapper was added to the production task,
exported or deployed.

Further work must preserve heading while lowering landing loads, rather than accept a
visually straighter but harder landing. Training-time exact symmetry and a separated
landing-only fine-tune are possible next experiments, not validated solutions. The
approved GPU budget is finished; no additional GPU run is active.

No checkpoint has been accepted, exported or deployed. Rust/controller build gates
remain unverified under the training-only waiver; no hardware claim is made.
