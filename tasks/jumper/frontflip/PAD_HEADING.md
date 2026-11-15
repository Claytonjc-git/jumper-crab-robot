# Absolute pad load and heading preservation

**Status: GPU trial and fresh holdouts completed; no candidate accepted.**
The subsequent [landing-pose correction](LANDING_POSE_GUARD.md) is CPU-validated only;
it was not part of this completed GPU trial.

This is a separately named follow-up to [ROTATION_GUARD.md](ROTATION_GUARD.md), not a
replacement of the previous experiment's results. The +10 candidate is a useful warm
start even though it failed promotion criteria. The original checkpoint stays a fixed
comparison control. Neither checkpoint is hardware-validated.

## Fixed controls and warm start

- Original: `out/frontflip-rotation-guard/kaggle-inputs/model_1500.pt`, SHA-256
  `cea9a567a1bc577f031a94ba526e77882a3169ed3f5bd08b20d699072b32e1de`.
- Warm start: `out/frontflip-rotation-guard/gpu-output/seed-1234/model_after_10.pt`, SHA-256
  `b7f474991f2b0c442a6b5239e7f216349beb85af0ae5c18227142ed5c29a50ef`.
- Warm-start provenance: Tesla T4, `warp:cuda:0`, 256 environments, 10 updates from the
  original; stored iteration 1509, environment counter 36264, learning rate 1e-5.

The archive for the preceding experiment and the task sources before this change are
preserved. New artifacts live only under `out/frontflip-pad-heading/`.

## Why these changes

The prior finalist softened non-pad strikes while raising some support-pad median/p90
peaks. A scale-invariant sharing penalty cannot distinguish equally shared high forces
from equally shared gentle forces. Add an absolute overload cost, using simultaneous
400 Hz force vectors and combining both front-claw pads before magnitudes:

`mean_substeps(max_limbs(overload²) + 0.5 * mean_limbs(overload²))`

Here `overload = max(limb_force - 30 N, 0)`, in a fixed 1.0–2.0 s landing/recovery
window. Weight is -1.5. It is not masked by termination or by successful forward
rotation, so a bad landing does not erase its own load cost. The previous non-pad cost,
bounded load-sharing cost and physical collision limits remain unchanged.

The exact video finishes 23.59° left, versus 28.59° for the original; turning observation
noise off still gives 24.24° for the candidate. Add a reference-relative world-up twist
cost, weight -40 per rad², and a separate world-z angular-velocity error cost, weight
-0.5. Both act from 0.5–3.5 s. Quaternion error against the phase-matched reference is
used instead of singular forward-axis heading during inversion. Antipodal quaternions
must produce identical cost; an undefined 180° swing is not treated as zero error.

The existing actor observations and model architecture are unchanged. There is no added
simulator-only heading input. The actor has gyro and gravity information but not an
absolute heading measurement; these rewards target the learned launch asymmetry and
yaw-rate trajectory, not a claimed general compass-based heading controller.

The first-touchdown 340–380° guard and valid-rotation landing credit remain. **There is
no new immediate heading termination:** killing the 23.6° warm start before it can learn
would discard all its successful nominal landings. Heading is a smooth training cost
and a separate evaluation requirement.

The fine-tune explicitly restarts a reset-time randomization ramp: hold nominal for 20
updates, ramp to the original full ranges over 80 updates, then train at full ranges.
Its start is anchored to the restored environment counter, not implemented by changing
that counter. Normal from-scratch curriculum defaults and full evaluation distributions
are unchanged. This is a combined intervention, not a causal ablation of each term.

## CPU evidence before GPU submission

Windows AMD CPU, `native:cpu`, observation noise on, seed 1234:

| Control | Result |
|---|---|
| Synthetic quaternion control | 16 pitch/yaw combinations; inverted and antipodal cases pass |
| Equal loads 20 / 40 / 60 N | Raw overload cost 0 / 150 / 1350 N²; equal sharing cannot hide force |
| Terminal and claw-vector controls | Cost remains charged; opposing vectors cancel before magnitude |
| Randomization ramp | Levels 0 / 0 / 0.5 / 1 / 1; full ranges restored, counter not rewritten |
| Actual training smoke | 32 updates, 8 environments; entire first episode and reset exercised |

At weight -0.15, the nominal warm start paid only 0.427 median pad cost per episode,
against 16.14 heading cost. That measured imbalance motivated -1.5 before submission.
With final weights, eight nominal starts retain 8/8 old rotation/support qualification,
zero physical failures, but 0/8 new heading-qualified flips. Their discounted
continuation-vs-immediate-termination margin is +51.80 median / +50.55 minimum.

Thirty-two full-randomization starts retain 16/32 old qualification and zero physical
failures, but 0/32 new heading-qualified flips. Continuation margin among properly
rotated, fully qualified old-definition landings stays positive: minimum +35.55.
Two under-rotated attempts, 326.2° and 315.9°, have negative margins (-0.62 / -4.19);
they remain explicit failed flips, not favorable load results. The training rotation
guard rejects those first touchdowns. Uncensored replay retains their complete evidence.

Both two-update and 32-update CPU smokes verify exact actor, critic, optimizer,
iteration, learning-rate and environment-counter restoration. Input/output tensor
shapes remain unchanged. The CPU smoke is not equivalent to GPU learning performance.

## Predeclared GPU trial and comparisons

Private dataset: `claytoncrumbly/jumper-frontflip-pad-heading-inputs-20261111`.
Private kernel: `claytoncrumbly/jumper-frontflip-pad-heading-20261111`.

Two training seeds, 1234 and 2718, each start from +10 and train 150 updates at 256
`warp:cuda:0` environments. Save 0/10/25/50/100/150 milestones. Verify actual CUDA
physics, source hashes and bit-exact milestone-zero actor/critic/optimizer state.
No deployment, GitHub push or original-checkpoint substitution is part of this trial.

Screen every candidate against both controls using paired eight-start nominal and
randomized replays. Preserve old-definition successful-flip counts and admit no physical
failures for shortlist eligibility. Freeze at most one candidate per training seed,
ranking by highest new straight-flip count, then lowest summed worst-limb median, then
lowest summed non-pad median. Freeze the shortlist before holdout measurements.

Fresh holdouts are **seeds 31415 and 92653**, each with 32 nominal and 32 randomized
starts, observation noise on, training rotation termination disabled. Do not reuse the
previous experiment's adapted-to holdouts as fresh confirmation.

New qualification adds **at most ±10° heading change at first upright touchdown and at
final upright six-support stance**, measured against the attempt's initial heading.
Old forward-turn/support qualification is also reported separately. Acceptance requires
at least 90% new qualification, no worse old flip count, no physical failures or
incomplete windows, no limb median/p90 regression, lower worst-limb median, and lower
non-pad median without higher p90, against **both** original and warm-start controls.
These stricter criteria do not retroactively relabel the completed prior trial.

All attempts and failures remain visible; only complete, nonfailed landing windows
contribute force summaries. Cached measurements must match checkpoint, task-recipe and
probe hashes. A stock measured replay must accompany any selected checkpoint report.

## Completed GPU results

Downloaded verification confirms a Tesla T4, CUDA 13.0 and an actual CUDA physics
step. Both seeds completed 150 updates at 256 `warp:cuda:0` environments; both
milestone-zero actor/critic/optimizer states match +10 exactly. The learning rate
remained 1e-5 at every saved milestone: an adaptive learning-rate increase does **not**
explain this trial's regression. Reported training mean reward changed -32.14 → -24.84
(seed 1234) and -32.67 → -32.88 (seed 2718). The first run's higher reward did not
translate into acceptable flips.

All ten checkpoints were screened. The frozen shortlist contained only
`seed-2718-after-10`, which preserved the old-definition eight-start screen counts.
Its fresh 32-start holdouts, sensor noise on and training guard disabled, measured:

| Seed / physics | Old flip count, +10 → new | New straight flips | Physical failures, +10 → new | Heading median, +10 → new | Worst-limb median, +10 → new |
|---|---:|---:|---:|---:|---:|
| 31415 / nominal | 32 → 31 | 0 | 0 → 0 | 24.6 → 22.8° | 45.7 → 45.8 N |
| 31415 / randomized | 9 → 8 | 0 | 2 → 2 | 24.6 → 22.0° | 35.5 → 35.2 N |
| 92653 / nominal | 32 → 29 | 0 | 0 → 0 | 24.3 → 22.8° | 45.3 → 46.4 N |
| 92653 / randomized | 6 → 6 | 0 | 4 → 3 | 24.9 → 22.2° | 47.4 → 49.0 N |

The slight yaw reduction and one fewer randomized failure do not offset nominal
regression, 0/128 straight flips, and limb median/p90 regressions. No policy was
accepted or substituted. **Retain +10 as the next warm start**, not this finalist.

Experimental finalist: `out/frontflip-pad-heading/gpu-output/seed-2718/model_after_10.pt`,
trained for 10 further updates from +10 on the verified T4. A stock CPU measured
replay completed 700 control steps with noise on and stop-on-done. It recorded 699
pre-reset joint samples (CSV t=0–3.49 s), peak absolute torque 1.7464 N·m; files are
beside the checkpoint in `measure/`. Joint torque is not a ground-reaction measurement.

All four lost nominal qualifications were first-touchdown upright failures: accumulated
turn 343.8–344.8° satisfied the old turn guard, but upright cosine 0.9616–0.9651 failed
the evaluator's 0.966 condition. Recovering upright later could still earn landing
reward. That measured gap motivates the separate pose/deadline correction, not a
reinterpretation of these rejected results.

The exact measured runtime sources are preserved in `out/frontflip-pad-heading/post-trial-source/`
and the 540-file source archive. Current task sources include the later local correction;
old cached comparisons must not be treated as measurements of those new sources.

## Separate Windows validation repair

The unrelated validation todo was completed before these task changes: full suite
**1221 passed, 28 skipped, zero failures**, no test deselected, 110.84 s. Fixes cover
CRLF lockfile detection and translation digests, byte-preserving corruption fixtures,
unprivileged fixture copies, Linux-only ABI test scope, Windows shared TCP listener
checks, and an isolated signal-delivery test. All ten translation stamps already match
their LF source content: no translation was blindly restamped.

Rust/controller build gates remain explicitly unverified under the operator's existing
training-only waiver. A Python suite pass is not a controller build or hardware test.
