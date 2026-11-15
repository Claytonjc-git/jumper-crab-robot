# Softer reference under randomization — 2026-11-12

**Status: candidate reference saved; not adopted.** `ref/frontflip_soft.npz` passes all 14
qualification gates in nominal physics. `ref/frontflip.npz` is unchanged and remains the
task default. Every existing checkpoint tracks the old reference, so adopting the new one
needs a new training run. No training, export or hardware test was done.

## Why the reference

Four reward fine-tunes from the +10 checkpoint failed acceptance. The reference itself was
fragile: in a zero-residual replay under the training randomization, it touches down at a
median 80 N non-pad force, and its 272.9° flight turn falls below the 270° gate in most draws.
A policy that tracks it inherits that margin.

## Method

CPU, native backend, Windows AMD machine. Randomization equals the training ranges: pad
friction 0.8–1.5, base COM ±25/±25/±30 mm, encoder bias ±0.015 rad, subtracted from every
target as the action term does.

Each candidate runs once in nominal physics with the staged contact-triggered controller;
the recorded `q_cmd` is then replayed open loop under every draw. That replay is what a
zero-residual policy executes. A replay of the nominal draw reproduces the closed-loop run
exactly (joint error 0, base error 0.02 mm), and the training environment reproduces the
nominal 34.9 N of the shipped reference exactly.

Cross-entropy search over 61 parameters: flight tuck, landing pose and timing per limb
group, the 24 staged launch parameters, and the middle-leg drive. Every draw set used for
search was different from the draws used to compare below. The first three stages scored
the closed-loop controller, which adapts to each draw and overstated the gain; the final
stage scored the fixed replay.

## Results

Fixed replay, 96 fresh draws (seed 60000), "any contact" is the larger of the non-pad peak
and the worst support-pad peak:

| | Shipped | Soft |
|---|---:|---:|
| Qualified (all 14 gates) | 3/96 | 23/96 |
| Any-contact peak median / p90 | 80.1 / 224.2 N | 58.6 / 80.5 N |
| Non-pad peak median / p90 | 79.3 / 224.2 N | 54.2 / 79.9 N |
| Worst pad median / p90 | 43.8 / 59.5 N | 43.8 / 61.3 N |
| Peak under 50 N | 11/96 | 19/96 |
| Flight turn median / p10 | 251.5 / 231.3° | 271.4 / 247.2° |
| Self-contact ≥ 1 N / shell ≥ 10 N | 21 / 6 | 2 / 0 |

Training environment (`env_cfg(play=True)`), zero residual, startup randomization, all
terminations except time-out removed so a hard landing is measured instead of cut short,
64 starts: non-pad median **87.7 → 48.5 N**, worst start **290 → 110 N**. With terminations
on, the same check reads lower for the shipped reference (42.9 N median) because episodes
end at the shell contact or the 100 N limb strike; do not compare censored peaks.

Nominal physics:

| | Shipped | Soft |
|---|---:|---:|
| Flight turn | 272.9° | 289.2° |
| Non-pad peak | 34.9 N | 36.8 N |
| Claw / middle pad at landing | 72.0 / 29.8 N | 35.4 / 25.1 N |
| Rear-drive forward impulse | 0.25 N·s | 0.52 N·s |
| Claw spin | 0.145 N·m·s | 0.143 N·m·s |
| Peak torque | 1.7464 N·m | 1.7464 N·m |

## Lean-back preparation — `ref/frontflip_lean.npz`

In the soft reference the claws carry 8–10 N at rest, and the straight joint-space fold
from HOME to the tuck pose passes 46 mm below the floor. The claw tips are pressed into the
floor (up to 22 N) and slide 58 mm inward. Raising the front alone does not clear them: the
whole-robot COM is 17 mm ahead of the middle feet, so lifting the claws tips the robot
onto its nose.

`momentum.py` now accepts an optional `prep_lift` in the variant (references without it
are unchanged):

1. Both middle legs step forward (hip yaw), while the claws and the rear feet carry the robot.
2. The rear legs lower and the middle legs raise the front: the body leans back about 19°.
3. The claws lift straight up once the feet carry the weight, then tuck in the air.
4. Middle and rear legs return to the load pose, and the launch proceeds as before.

The lean poses were found with a nominal search on the preparation alone. The launch,
flight and lean parameters were then re-tuned together with the fixed-replay search, with
claw slide added to the cost. Same 96 draws (seed 60000) as above:

| | Soft | Lean |
|---|---:|---:|
| Qualified (all 14 gates) | 23/96 | 28/96 |
| Claw slide nominal / median | 57.9 / 63.1 mm | 2.8 / 2.7 mm |
| Any-contact peak median / p90 | 58.6 / 80.5 N | 56.1 / 75.3 N |
| Non-pad peak median / p90 | 54.2 / 79.9 N | 48.6 / 74.9 N |
| Peak under 50 N | 19/96 | 29/96 |
| Flight turn median / p10 | 271.4 / 247.2° | 267.1 / 246.3° |
| Self-contact ≥ 1 N / shell ≥ 10 N | 2 / 0 | 3 / 0 |

Training environment, zero residual, terminations off, 64 randomized starts, reset seed 0:
non-pad median 50.2 → 50.9 N, worst start 98.3 → 129.1 N. The soft reference read 48.5 N
in the earlier run of the same check, so a 2 N difference is within run-to-run noise. Nominal: flight
turn 285.5°, non-pad peak 30.8 N, peak torque 1.7464 N·m, qualified.

## Flight-turn margin — `ref/frontflip_lean_turn.npz`

GPU trial on `frontflip_lean.npz` (Tesla T4, `warp:cuda:0`, 256 envs): a from-scratch policy
qualified 64/64 nominal but only 10/64 randomized, and every run lost rotation once
randomization reached full strength (first-touchdown turn median 329°, gate 340°). The
lean reference was re-searched from `frontflip_lean` with a stronger flight-turn cost
(2 per 10° below 295° per draw; 50 generations, 48 candidates, 24 draws). Same 96 draws:

| | Lean | Lean turn |
|---|---:|---:|
| Qualified (all 14 gates) | 28/96 | 22/96 |
| Flight turn median / p10 | 267.1 / 246.3° | 282.9 / 254.6° |
| Any-contact peak median / p90 | 56.1 / 75.3 N | 61.4 / 77.1 N |
| Self-contact ≥ 1 / ≥ 5 / ≥ 15 N | 3 / 2 / 1 | 25 / 3 / 1 |
| Claw slide median | 2.7 mm | 2.8 mm |

Nominal: flight turn 297.2°, non-pad peak 30.3 N, qualified. Most new self-contacts are
below the 5 N training penalty.

## Using it

```sh
MJRL_FRONTFLIP_REFERENCE=tasks/jumper/frontflip/ref/frontflip_soft.npz python scripts/train.py --task jumper.frontflip ...
```

`LIMB_PENALTY_FROM_N` (35 N) names the shipped reference's own landing peak; the soft
reference peaks at 36.8 N nominal. Revisit it before adopting.

Tools, logs and per-stage results: `out/frontflip-reference-search/`.
